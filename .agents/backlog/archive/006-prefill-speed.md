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

After the f16 KV cache (007), `--tokens N --context 9000 --profile`:

| Prompt tokens | Prefill tok/s | Decode tok/s | First token | Cores | Attention share of prefill |
|---|---|---|---|---|---|
| 268 | 93.8 | 28.1 | 2.86 s | 3.46 | 5% |
| 1978 | 84.5 | 28.1 | 23.41 s | 3.28 | 15% |
| 4112 | 77.9 | 22.3 | 52.79 s | 3.26 | 24% |
| 7560 | 66.1 | 13.5 | 114.32 s | 3.27 | 35% |

The remaining drop is attention's quadratic work. At 7560 tokens that is
about 6.5 TFLOP over 40 s, close to what gemm reaches on 4 cores, so only
fewer flops (sliding window, sparse attention) would flatten it further.

Ctrl-C is checked between prompt chunks of 128 tokens, so on the 4B model it
can take several seconds to stop.

# SEE ALSO

* [x] ext/elelem/candle/src/engine.rs - chunked prefill
* [x] candle-transformers 0.11 src/models/quantized_qwen3.rs - attention
* [x] lib/elelem/candle/provider.rb - chunk size

# Tasks

* [x] `script/bench` with `--profile` section timers
* [x] gemm attention for prefill, reading the KV cache in place
* [x] Check for interrupts before each layer, so Ctrl-C stops within a second
* [x] Chunk size sweep: 32, 128 and 512 are within 6%, so 128 stays

# Acceptance Criteria

* [x] Prompt reading speed is measured before and after
* [x] Reading speed does not drop with prompt length up to the context size, beyond attention's quadratic cost (94 tok/s at 268 tokens, 66 at 7.5k)
* [x] Prompt reading uses all cores (3.3 to 3.5 of 4)
