mod engine;
mod gguf;
mod qwen3;
mod template;
mod tokenizer;

use engine::{Engine, Failure, Options};
use magnus::{block::Proc, function, method, prelude::*, Error, RHash, Ruby};
use std::cell::RefCell;

#[magnus::wrap(class = "Elelem::Candle::Model", free_immediately, size)]
struct Model(RefCell<Engine>);

fn runtime_error(ruby: &Ruby, e: impl std::fmt::Display) -> Error {
    Error::new(ruby.exception_runtime_error(), format!("candle: {e:#}"))
}

fn parse_json(ruby: &Ruby, s: &str) -> Result<serde_json::Value, Error> {
    serde_json::from_str(s).map_err(|e| runtime_error(ruby, e))
}

fn render_template(
    ruby: &Ruby,
    source: String,
    messages: String,
    tools: String,
    thinking: bool,
) -> Result<String, Error> {
    let (messages, tools) = (parse_json(ruby, &messages)?, parse_json(ruby, &tools)?);
    template::Template::new(source)
        .and_then(|t| t.render(&messages, &tools, thinking))
        .map_err(|e| runtime_error(ruby, e))
}

impl Model {
    fn load(ruby: &Ruby, path: String, context: usize) -> Result<Self, Error> {
        Engine::load(&path, context)
            .map(|e| Self(RefCell::new(e)))
            .map_err(|e| runtime_error(ruby, e))
    }

    fn context(&self) -> usize {
        self.0.borrow().context
    }

    fn count(ruby: &Ruby, rb_self: &Self, text: String) -> Result<usize, Error> {
        rb_self
            .0
            .borrow()
            .encode(&text)
            .map(|t| t.len())
            .map_err(|e| runtime_error(ruby, e))
    }

    fn reset(ruby: &Ruby, rb_self: &Self) -> Result<(), Error> {
        rb_self
            .0
            .try_borrow_mut()
            .map(|mut e| e.reset())
            .map_err(|_| runtime_error(ruby, "generate is already running on this model"))
    }

    fn architecture(&self) -> String {
        self.0.borrow().architecture.clone()
    }

    fn render(
        ruby: &Ruby,
        rb_self: &Self,
        messages: String,
        tools: String,
        thinking: bool,
    ) -> Result<String, Error> {
        rb_self
            .0
            .borrow()
            .render(
                &parse_json(ruby, &messages)?,
                &parse_json(ruby, &tools)?,
                thinking,
            )
            .map_err(|e| runtime_error(ruby, e))
    }

    fn generate(
        ruby: &Ruby,
        rb_self: &Self,
        prompt: String,
        opts: RHash,
        on_piece: Proc,
    ) -> Result<RHash, Error> {
        let opts = Options {
            max_tokens: opts.fetch(ruby.to_symbol("max_tokens"))?,
            temperature: opts.fetch(ruby.to_symbol("temperature"))?,
            top_k: opts.fetch(ruby.to_symbol("top_k"))?,
            top_p: opts.fetch(ruby.to_symbol("top_p"))?,
            seed: opts.fetch(ruby.to_symbol("seed"))?,
            chunk: opts.fetch(ruby.to_symbol("chunk"))?,
            stop: opts
                .lookup::<_, Option<Vec<String>>>(ruby.to_symbol("stop"))?
                .unwrap_or_default(),
            profile: opts
                .lookup::<_, Option<bool>>(ruby.to_symbol("profile"))?
                .unwrap_or(false),
        };
        let mut engine = rb_self
            .0
            .try_borrow_mut()
            .map_err(|_| runtime_error(ruby, "generate is already running on this model"))?;
        let stats = engine
            .generate(
                &prompt,
                &opts,
                |piece| on_piece.call::<_, magnus::Value>((piece,)).map(|_| ()),
                || ruby.thread_check_ints(),
            )
            .map_err(|f| match f {
                Failure::Model(e) => runtime_error(ruby, e),
                Failure::Callback(e) => e,
            })?;
        let hash = ruby.hash_new();
        hash.aset(ruby.to_symbol("stop_reason"), stats.stop_reason)?;
        hash.aset(ruby.to_symbol("stop_token"), stats.stop_token)?;
        hash.aset(ruby.to_symbol("n_prompt"), stats.n_prompt)?;
        hash.aset(ruby.to_symbol("n_reused"), stats.n_reused)?;
        hash.aset(ruby.to_symbol("n_decoded"), stats.n_decoded)?;
        hash.aset(ruby.to_symbol("ms_prefill"), stats.ms_prefill)?;
        hash.aset(ruby.to_symbol("ms_decode"), stats.ms_decode)?;
        if let Some([prefill, decode]) = stats.profile {
            for (name, ms) in [("prefill", prefill), ("decode", decode)] {
                let sections = ruby.hash_new();
                for (section, value) in qwen3::SECTIONS.iter().zip(ms) {
                    sections.aset(ruby.to_symbol(section), value)?;
                }
                hash.aset(ruby.to_symbol(format!("profile_{name}")), sections)?;
            }
        }
        Ok(hash)
    }
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let candle = ruby.define_module("Elelem")?.define_module("Candle")?;
    candle.define_singleton_method("render_template", function!(render_template, 4))?;
    let model = candle.define_class("Model", ruby.class_object())?;
    model.define_singleton_method("load", function!(Model::load, 2))?;
    model.define_method("context", method!(Model::context, 0))?;
    model.define_method("architecture", method!(Model::architecture, 0))?;
    model.define_method("count", method!(Model::count, 1))?;
    model.define_method("reset", method!(Model::reset, 0))?;
    model.define_method("render", method!(Model::render, 3))?;
    model.define_method("generate", method!(Model::generate, 3))?;
    Ok(())
}
