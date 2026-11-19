# elelem-candle

An [elelem](https://github.com/elelem-sh/elelem) provider plugin backed by
[huggingface/candle](https://github.com/huggingface/candle), a pure-Rust
tensor/ML framework, via a real Ruby native extension (not a spawned
subprocess).

## Status

Experimental. Runs Qwen3 GGUF models on the CPU, including
`Qwen3-4B-UD-Q4_K_XL.gguf` from the elelem installer. Tensors stored as
IQ4_XS, which candle cannot read, are converted to Q8_0 at load time.
The tokenizer and chat template come from the GGUF file.

Limits:

- Qwen3 only, CPU only.
- No tool calls yet. The reply is plain text.
- Every request reads the whole conversation again; nothing is reused
  between turns.
- Prompt reading is slow and gets slower as the prompt grows.

## Why a native extension instead of a subprocess

`elelem-ratatui` shells out to a standalone Rust binary and talks JSON over
stdio. This gem instead uses [`rb_sys`](https://github.com/oxidize-rb/rb_sys) +
[`magnus`](https://github.com/matsadler/magnus) to compile candle straight
into a loadable `.so`, called in-process via real Ruby<->Rust FFI (typed
arguments, Ruby exceptions on Rust errors, streaming callbacks) -- the same
mechanism most modern Ruby+Rust gems use, and the standard `bundle gem
--ext=rust` scaffold.

## Requirements

- Rust toolchain (`cargo`, `rustc`): see https://rustup.rs
- A working C toolchain + `clang`/`libclang` (needed by `rb_sys`'s bindgen
  step to generate Ruby FFI bindings). On Fedora: `sudo dnf install clang`.
- Ruby 4.0

## Development

```
bundle install
bundle exec rake compile   # builds the Rust extension via rb_sys
bundle exec rake build     # -> bundle exec rake compile
```

## Usage

```
elelem-candle ask --model ~/.agents/models/default.gguf "hi"
ELELEM_CANDLE_MODEL=~/.agents/models/default.gguf elelem -p candle
```

| Variable | Default | Meaning |
|---|---|---|
| `ELELEM_CANDLE_MODEL` | | Path to a Qwen3 `.gguf` file, used when `--model` is not given |
| `ELELEM_CANDLE_N_CTX` | `8192` | Context in tokens, capped at the model's |
| `ELELEM_CANDLE_MAX_TOKENS` | `0` | Output tokens per reply. 0 means the rest of the context |
| `ELELEM_CANDLE_TEMP` | `0.6` | Temperature. 0 is greedy |
| `ELELEM_CANDLE_TOP_K` | `20` | Top-k |
| `ELELEM_CANDLE_TOP_P` | `0.95` | Top-p |
| `ELELEM_CANDLE_SEED` | `-1` | Sampling seed. Negative means random |
| `ELELEM_CANDLE_THINKING` | `0` | `1` lets the model think before it replies |
| `ELELEM_CANDLE_TARGET_CPU` | `native` | Build time. The `-C target-cpu` passed to rustc |

## Testing

```
bin/test
ELELEM_CANDLE_TEST_MODEL=~/.agents/models/default.gguf bin/test
```

The second form also runs the specs that load a real model.
