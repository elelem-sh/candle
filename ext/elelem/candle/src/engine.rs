use crate::gguf;
use crate::qwen3::{ModelWeights, Profile};
use crate::template::Template;
use crate::tokenizer;
use anyhow::{anyhow, bail, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::cell::{RefCell, RefMut};
use std::time::Instant;
use tokenizers::Tokenizer;

pub struct Options {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub seed: u64,
    pub chunk: usize,
    pub stop: Vec<String>,
    pub profile: bool,
}

pub struct Stats {
    pub stop_reason: &'static str,
    pub stop_token: Option<String>,
    pub n_prompt: usize,
    pub n_reused: usize,
    pub n_decoded: usize,
    pub ms_prefill: f64,
    pub ms_decode: f64,
    pub profile: Option<[[f64; 6]; 2]>,
}

pub enum Failure<E> {
    Model(anyhow::Error),
    Callback(E),
}

impl<E> From<anyhow::Error> for Failure<E> {
    fn from(e: anyhow::Error) -> Self {
        Failure::Model(e)
    }
}

impl<E> From<candle_core::Error> for Failure<E> {
    fn from(e: candle_core::Error) -> Self {
        Failure::Model(e.into())
    }
}

struct Session {
    weights: ModelWeights,
    cached: Vec<u32>,
}

impl Session {
    fn truncate(&mut self, len: usize) {
        self.weights.truncate(len);
        self.cached.truncate(len);
    }
}

pub struct Engine {
    session: RefCell<Session>,
    tokenizer: Tokenizer,
    template: Template,
    stop: Vec<u32>,
    device: Device,
    pub context: usize,
}

impl Engine {
    pub fn load(path: &str, context: usize) -> Result<Self> {
        let (ct, mut reader) = gguf::open(path)?;
        let architecture = tokenizer::text(&ct, "general.architecture")?;
        if architecture != "qwen3" {
            bail!("{path}: unsupported architecture {architecture}, only qwen3");
        }
        let trained = ct
            .metadata
            .get("qwen3.context_length")
            .map(|v| v.to_u32())
            .transpose()?
            .unwrap_or(4096) as usize;
        let tokenizer = tokenizer::from_gguf(&ct)?;
        let template = Template::new(tokenizer::text(&ct, "tokenizer.chat_template")?)?;
        let stop = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();
        let device = Device::Cpu;
        let weights = ModelWeights::from_gguf(ct, &mut reader, &device)?;
        Ok(Self {
            session: RefCell::new(Session {
                weights,
                cached: Vec::new(),
            }),
            tokenizer,
            template,
            stop,
            device,
            context: if context == 0 {
                trained
            } else {
                context.min(trained)
            },
        })
    }

    fn session(&self) -> Result<RefMut<'_, Session>> {
        self.session
            .try_borrow_mut()
            .map_err(|_| anyhow!("generate is already running on this model"))
    }

    pub fn reset(&self) -> Result<()> {
        self.session()?.truncate(0);
        Ok(())
    }

    pub fn render(
        &self,
        messages: &serde_json::Value,
        tools: &serde_json::Value,
        thinking: bool,
    ) -> Result<String> {
        self.template.render(messages, tools, thinking)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec())
    }

    pub fn generate<E>(
        &self,
        prompt: &str,
        opts: &Options,
        mut on_piece: impl FnMut(&str) -> Result<(), E>,
        mut check: impl FnMut() -> Result<(), E>,
    ) -> Result<Stats, Failure<E>> {
        let mut session = self.session()?;
        let tokens = self.encode(prompt)?;
        if tokens.len() >= self.context {
            return Err(anyhow!(
                "prompt is {} tokens, context is {}",
                tokens.len(),
                self.context
            )
            .into());
        }
        let budget = self.context - tokens.len();
        let max_tokens = if opts.max_tokens == 0 {
            budget
        } else {
            opts.max_tokens.min(budget)
        };
        let sampling = if opts.temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::TopKThenTopP {
                k: opts.top_k,
                p: opts.top_p,
                temperature: opts.temperature,
            }
        };
        let mut stop = self.stop.clone();
        for text in &opts.stop {
            match self.encode(text)?.as_slice() {
                [id] => stop.push(*id),
                _ => return Err(anyhow!("stop {text:?} is not a single token").into()),
            }
        }
        let mut sampler = LogitsProcessor::from_sampling(opts.seed, sampling);
        if session.weights.len() != session.cached.len() {
            session.truncate(0);
        }
        let n_reused = session
            .cached
            .iter()
            .zip(&tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(tokens.len().saturating_sub(1));
        session.truncate(n_reused);

        let mut prefill = Profile::new(opts.profile);
        let mut decode = Profile::new(opts.profile);
        let started = Instant::now();
        let mut logits = None;
        for chunk in tokens[n_reused..].chunks(opts.chunk.max(1)) {
            let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
            let mut interrupted = None;
            let out = session
                .weights
                .forward(&input, &mut prefill, &mut || match check() {
                    Ok(()) => false,
                    Err(e) => {
                        interrupted = Some(e);
                        true
                    }
                })?;
            match (out, interrupted) {
                (Some(out), _) => logits = Some(out),
                (None, Some(e)) => return Err(Failure::Callback(e)),
                (None, None) => return Err(anyhow!("prefill stopped without a reason").into()),
            }
            session.cached.extend_from_slice(chunk);
        }
        let ms_prefill = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let mut stream = TextStream::default();
        let mut logits = logits.ok_or_else(|| anyhow!("empty prompt"))?;
        let mut stop_reason = "length";
        let mut stop_token = None;
        let mut n_decoded = 0;
        while n_decoded < max_tokens {
            check().map_err(Failure::Callback)?;
            let next = sampler.sample(&logits.squeeze(0)?)?;
            n_decoded += 1;
            if stop.contains(&next) {
                stop_reason = "stop";
                stop_token = self.tokenizer.id_to_token(next);
                break;
            }
            if let Some(piece) = stream.push(&self.tokenizer, next)? {
                on_piece(&piece).map_err(Failure::Callback)?;
            }
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = session
                .weights
                .forward(&input, &mut decode, &mut || false)?
                .ok_or_else(|| anyhow!("decode stopped without a reason"))?;
            session.cached.push(next);
        }
        if let Some(piece) = stream.flush(&self.tokenizer)? {
            on_piece(&piece).map_err(Failure::Callback)?;
        }
        Ok(Stats {
            stop_reason,
            stop_token,
            n_prompt: tokens.len(),
            n_reused,
            n_decoded,
            ms_prefill,
            ms_decode: started.elapsed().as_secs_f64() * 1000.0,
            profile: opts.profile.then_some([prefill.ms, decode.ms]),
        })
    }
}

#[derive(Default)]
struct TextStream {
    tokens: Vec<u32>,
    emitted: usize,
}

impl TextStream {
    fn push(&mut self, tokenizer: &Tokenizer, token: u32) -> Result<Option<String>> {
        self.tokens.push(token);
        let text = self.decode(tokenizer)?;
        if text.ends_with('\u{FFFD}') {
            return Ok(None);
        }
        Ok(self.take(&text))
    }

    fn flush(&mut self, tokenizer: &Tokenizer) -> Result<Option<String>> {
        let text = self.decode(tokenizer)?;
        Ok(self.take(&text))
    }

    fn decode(&self, tokenizer: &Tokenizer) -> Result<String> {
        tokenizer
            .decode(&self.tokens, false)
            .map_err(|e| anyhow!(e))
    }

    fn take(&mut self, text: &str) -> Option<String> {
        let piece = text
            .get(self.emitted..)
            .filter(|p| !p.is_empty())
            .map(str::to_string);
        self.emitted = self.emitted.max(text.len());
        piece
    }
}
