use magnus::{
    block::Proc, function, method, prelude::*, Error, Ruby,
};

/// Wraps a loaded candle model + tokenizer. Right now this is a skeleton:
/// `Model::new` just records where the weights *would* load from, and
/// `generate` proves out the streaming-token-across-FFI plumbing without a
/// real forward pass yet.
///
/// Next steps to make this real (see candle-transformers examples for the
/// reference implementations):
///   1. Load a tokenizer (tokenizers::Tokenizer::from_file / from hf-hub).
///   2. Load safetensors weights into a candle_core::Device (Cpu, or Cuda/
///      Metal once those candle-core features are enabled).
///   3. Build the model (e.g. candle_transformers::models::mistral::Model).
///   4. In `generate`, run the actual token-by-token forward pass + sampling
///      loop, calling `token_cb.call` per generated piece instead of the
///      canned stub output below.
#[magnus::wrap(class = "Elelem::Candle::Model", free_immediately, size)]
struct Model {
    path: String,
    device: String,
}

impl Model {
    fn new(path: String, device: String) -> Self {
        Self { path, device }
    }

    /// Streams generated text back to Ruby by invoking `token_cb` once per
    /// piece. This is the interesting FFI boundary: Rust owns the generation
    /// loop, Ruby supplies a block, and each token crosses the boundary as
    /// it's produced rather than buffering the whole response.
    ///
    /// `&Ruby` must come before `self` here -- that's the signature
    /// `RubyMethod2` (what `method!(_, 2)` resolves to for a 2-Ruby-arg
    /// method that also wants the `&Ruby` handle) actually requires:
    /// `Fn(&Ruby, RbSelf, T1, T2) -> Res`. A `&self` receiver puts `self`
    /// first instead, which is why the original shape failed to satisfy
    /// the trait bound at compile time.
    fn generate(ruby: &Ruby, rb_self: &Model, prompt: String, token_cb: Proc) -> Result<(), Error> {
        if rb_self.path.is_empty() {
            return Err(Error::new(
                ruby.exception_runtime_error(),
                "Elelem::Candle::Model: no model path configured",
            ));
        }

        // TODO(real inference): replace this with the actual candle
        // generation loop. Left as a stub so the Ruby <-> Rust streaming
        // contract (Provider#fetch -> Model#generate -> block) can be
        // exercised end to end before a real model is wired in.
        let stub = format!(
            "[stub:{} on {}] {}",
            rb_self.path, rb_self.device, prompt
        );
        for word in stub.split_whitespace() {
            token_cb.call::<_, magnus::Value>((format!("{word} "),))?;
        }

        Ok(())
    }
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let elelem = ruby.define_module("Elelem")?;
    let candle = elelem.define_module("Candle")?;
    let model = candle.define_class("Model", ruby.class_object())?;

    model.define_singleton_method("new", function!(Model::new, 2))?;
    model.define_method("generate", method!(Model::generate, 2))?;

    Ok(())
}
