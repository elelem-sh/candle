As a `user of elelem-candle on a CPU`, I `want long prompts read quickly`,
so that `a system prompt and tool list do not take minutes per request`.

# SYNOPSIS

Find and fix why prompt reading is close to decode speed and slows as the
prompt grows.

# DESCRIPTION

On a 4-core x86_64 with AVX2, Qwen3-0.6B reads a 645-token prompt at about
87 tokens/s and a 1559-token prompt at about 39 tokens/s, using about 2.5
cores. Qwen3-4B reads about 15 tokens/s. The slowdown with length points at
the CPU attention path in candle-transformers' quantized_qwen3 rather than
the quantized matmuls.

Baseline from `script/bench`, greedy, 32 output tokens, chunk 128:

| Model | Run | Prompt tokens | Prefill tok/s | Decode tok/s | First token | Cores |
|---|---|---|---|---|---|---|
| 0.6B | `--tokens 256` | 295 | 74.6 | 31.0 | 3.95 s | 3.36 |
| 0.6B | `--tokens 1800` | 2073 | 21.5 | 15.9 | 96.22 s | 3.37 |
| 4B | `--step` turn 1 | 292 | 12.6 | 5.3 | 23.09 s | 3.42 |
| 4B | `--step` turn 2 | 662 | 11.1 | 4.3 | 59.47 s | 3.50 |

Decode also slows as the context grows.

After cache reuse (005) and gemm attention, same commands:

| Model | Run | Prompt tokens | Reused | Prefill tok/s | Decode tok/s | First token |
|---|---|---|---|---|---|---|
| 0.6B | `--tokens 256` | 255 | 0 | 96.3 | 29.5 | 2.65 s |
| 0.6B | `--tokens 1800` | 1843 | 0 | 93.1 | 14.7 | 19.80 s |
| 4B | `--step` turn 1 | 292 | 0 | 14.5 | 5.5 | 20.09 s |
| 4B | `--step` turn 2 | 662 | 288 | 15.0 | 4.5 | 24.88 s |

`script/bench --profile` showed attention at 75% of prefill on 0.6B at
1843 tokens. Reading it with two gemm calls per KV head straight from the
cache brought it to 17%; quantized matmuls are now most of prefill.
candle's own decode kernel stayed faster than a hand-written one, so
decode keeps it. Decode attention is bound by memory bandwidth on the f32
KV cache, so decode still slows as the context grows.

Ctrl-C is checked between prompt chunks of 128 tokens, so on the 4B model it
can take several seconds to stop.

# SEE ALSO

* [ ] ext/elelem/candle/src/engine.rs - chunked prefill
* [ ] candle-transformers 0.11 src/models/quantized_qwen3.rs - attention
* [ ] lib/elelem/candle/provider.rb - chunk size

# Tasks

* [x] `script/bench` with `--profile` section timers
* [x] gemm attention for prefill, reading the KV cache in place
* [x] Check for interrupts before each layer, so Ctrl-C stops within a second
* [x] Chunk size sweep: 32, 128 and 512 are within 6%, so 128 stays

# Acceptance Criteria

* [x] Prompt reading speed is measured before and after
* [ ] Reading speed does not drop with prompt length up to the context size (holds to 1.8k: 96 vs 93 tok/s; not measured beyond)
* [ ] Prompt reading uses all cores (3.3 to 3.6 of 4)
