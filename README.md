# elelem-candle

An [elelem](https://github.com/elelem-sh/elelem) provider plugin backed by
[huggingface/candle](https://github.com/huggingface/candle), a pure-Rust
tensor/ML framework, via a real Ruby native extension (not a spawned
subprocess).

## Status

Experimental / hobby project. This is a skeleton: the Ruby <-> Rust boundary
(`Elelem::Candle::Provider` -> `Elelem::Candle::Model#generate` -> streaming
block) works end to end, but `Model#generate` in
`ext/elelem/candle/src/lib.rs` is still a stub that echoes the prompt back
token-by-token instead of running real inference. See the TODOs in that file
for what's next (tokenizer + safetensors loading + an actual candle-transformers
model forward pass).

## Why a native extension instead of a subprocess

`elelem-ratatui` shells out to a standalone Rust binary and talks JSON over
stdio. This gem instead uses [`rb_sys`](https://github.com/oxidize-rb/rb_sys) +
[`magnus`](https://github.com/matsadler/magnus) to compile candle straight
into a loadable `.so`, called in-process via real Ruby<->Rust FFI (typed
arguments, Ruby exceptions on Rust errors, streaming callbacks) -- the same
mechanism most modern Ruby+Rust gems use, and the standard `bundle gem
--ext=rust` scaffold.

## Requirements

- Rust toolchain (`cargo`, `rustc`) -- see https://rustup.rs
- A working C toolchain + `clang`/`libclang` (needed by `rb_sys`'s bindgen
  step to generate Ruby FFI bindings). On Fedora: `sudo dnf install clang`.

## Development

```
bundle install
bundle exec rake compile   # builds the Rust extension via rb_sys
bundle exec rake build     # -> bundle exec rake compile
```

## Usage

```
export ELELEM_CANDLE_MODEL=/path/to/model
bundle exec exe/elelem-candle ask "hi"
```
