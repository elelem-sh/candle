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

It calls tools: the tool list goes into the chat template, and each
`<tool_call>` the model writes becomes a `doing` event. A call cut off by
the output limit is reported and never run. Each request reuses the cached
prompt prefix from the last one, so a tool step reads only the new tokens.

Use `Qwen3-4B`. `Qwen3-0.6B` is fast but experimental with tools. On
`script/eval`:

| Model | Calls that parse | Right tool | Right arguments | Multi-step tasks |
|---|---|---|---|---|
| Qwen3-4B | 40/40 | 40/40 | 40/40 | 7/8 |
| Qwen3-0.6B | 39/40 | 22/40 | 22/40 | 3/8 |

On a 4-core x86_64 with AVX2 (`script/bench`):

| Model | Prompt reading | Replying | A tool step, to first token |
|---|---|---|---|
| Qwen3-0.6B | about 95 tokens/s | 28 tokens/s, 24 at 1.9k context | |
| Qwen3-4B | about 15 tokens/s | 5 tokens/s, also at 1.9k context | about 25 s |

The KV cache is stored as f16, half the memory of f32, so replying stays
fast as the conversation grows.

Limits:

- Qwen3 only, CPU only.
- Small models reply slower past about 2k tokens of context: 0.6B drops
  to about 22 tokens/s at 4.1k and 14 at 7.5k.
- The reply appears when it is finished; only thinking streams.
- One tool call per reply. When the model writes several, only the first
  runs, so a later call can use its result.

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
| `ELELEM_CANDLE_MAX_TOKENS` | `1024`, or `2048` when thinking | Output tokens per reply. 0 means the rest of the context |
| `ELELEM_CANDLE_TEMP` | `0.7`, or `0.6` when thinking | Temperature. 0 is greedy |
| `ELELEM_CANDLE_TOP_K` | `20` | Top-k |
| `ELELEM_CANDLE_TOP_P` | `0.8`, or `0.95` when thinking | Top-p |
| `ELELEM_CANDLE_SEED` | `-1` | Sampling seed. Negative means random |
| `ELELEM_CANDLE_THINKING` | `0` | `1` lets the model think before it replies |
| `ELELEM_CANDLE_TARGET_CPU` | `native` | Build time. The `-C target-cpu` passed to rustc |

## Testing

```
bin/test
ELELEM_CANDLE_TEST_MODEL=~/.agents/models/default.gguf bin/test
```

The second form also runs the specs that load a real model.

```
script/bench MODEL [--tokens N] [--decode N] [--chunk N] [--step] [--profile]
script/eval MODEL [--no-gate] [--only N] [--no-runs]
```

`script/bench` prints prompt reading and reply speed. `--step` simulates
a tool step, and `--profile` shows where the time goes.

`script/eval` sends 40 one-step tasks with elelem's tools and checks the
tool, its arguments and that the call parses, then runs 8 multi-step tasks
in a temporary directory. It exits 1 when parsing or tool choice is under
85%. It takes about 9 minutes on 4B and is not run in CI.
