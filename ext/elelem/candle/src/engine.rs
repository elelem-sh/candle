use crate::gguf;
use crate::template::Template;
use crate::tokenizer;
use anyhow::{anyhow, bail, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights;
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
}

pub struct Stats {
    pub stop_reason: &'static str,
    pub stop_token: Option<String>,
    pub n_prompt: usize,
    pub n_decoded: usize,
    pub ms_prefill: f64,
    pub ms_decode: f64,
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

pub struct Engine {
    weights: ModelWeights,
    tokenizer: Tokenizer,
    template: Template,
    stop: Vec<u32>,
    device: Device,
    pub context: usize,
    pub architecture: String,
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
        let context = if context == 0 {
            trained
        } else {
            context.min(trained)
        };
        Ok(Self {
            weights,
            tokenizer,
            template,
            stop,
            device,
            context,
            architecture,
        })
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
        &mut self,
        prompt: &str,
        opts: &Options,
        mut on_piece: impl FnMut(&str) -> Result<(), E>,
        mut check: impl FnMut() -> Result<(), E>,
    ) -> Result<Stats, Failure<E>> {
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
        self.weights.clear_kv_cache();

        let started = Instant::now();
        let mut logits = None;
        for (i, chunk) in tokens.chunks(opts.chunk.max(1)).enumerate() {
            check().map_err(Failure::Callback)?;
            let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
            logits = Some(self.weights.forward(&input, i * opts.chunk.max(1))?);
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
            logits = self.weights.forward(&input, tokens.len() + n_decoded - 1)?;
        }
        if let Some(piece) = stream.flush(&self.tokenizer)? {
            on_piece(&piece).map_err(Failure::Callback)?;
        }
        Ok(Stats {
            stop_reason,
            stop_token,
            n_prompt: tokens.len(),
            n_decoded,
            ms_prefill,
            ms_decode: started.elapsed().as_secs_f64() * 1000.0,
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
        let text = tokenizer
            .decode(&self.tokens, false)
            .map_err(|e| anyhow!(e))?;
        if text.ends_with('\u{FFFD}') {
            return Ok(None);
        }
        let piece = text
            .get(self.emitted..)
            .filter(|p| !p.is_empty())
            .map(str::to_string);
        self.emitted = self.emitted.max(text.len());
        Ok(piece)
    }

    fn flush(&mut self, tokenizer: &Tokenizer) -> Result<Option<String>> {
        let text = tokenizer
            .decode(&self.tokens, false)
            .map_err(|e| anyhow!(e))?;
        Ok(text
            .get(self.emitted..)
            .filter(|p| !p.is_empty())
            .map(str::to_string))
    }
}
