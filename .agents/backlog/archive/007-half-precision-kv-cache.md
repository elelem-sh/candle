As a `user in a long elelem-candle session`, I `want replies to keep their speed as the conversation grows`, so that `turn 20 is not half as fast as turn 1`.

# SYNOPSIS

Store the KV cache in f16 to halve the memory decode reads per token.

# DESCRIPTION

Decode attention reads the whole KV cache for every token. The cache is
f32: on 0.6B at 1843 tokens that is about 420 MB per token, and decode
attention is 59% of decode time (`script/bench --tokens 1800 --profile`).
Decode drops from 29.5 tok/s at 255 tokens to 14.7 tok/s at 1843.

Store K and V as f16 in `KvCache`, convert on write, and read them in the
gemm prefill kernel and a decode kernel that accepts f16.

# SEE ALSO

* [x] ext/elelem/candle/src/qwen3.rs - KvCache
* [x] ext/elelem/candle/src/attention.rs - causal
* [x] candle-nn 0.11 src/attention/cpu_flash/causal.rs - causal_decode_f32_interleaved

# Tasks

* [x] Move `KvCache` to `attention.rs`, one f16 K and V buffer per KV head
* [x] Decode kernel that reads each KV head once for its whole query group
* [x] AVX2, FMA and F16C path: 4 positions per block, vector exp, V in register tiles
* [x] Scalar path for other CPUs
* [x] Run decode on candle's thread pool, not rayon, so the two pools do not fight
* [x] Prefill converts each KV head to f32 and keeps the gemm kernel

# Results

`script/bench --decode 64`, the baseline and f16 runs back to back:

| Model | Context | f32 decode | f16 decode |
|---|---|---|---|
| 0.6B | 268 | 21.5 tok/s | 28.1 tok/s |
| 0.6B | 1951 | 10.9 tok/s | 23.9 tok/s |
| 4B | 268 | 4.6 tok/s | 4.7 tok/s |
| 4B | 1951 | 2.7 tok/s | 4.7 tok/s |

Prefill is unchanged. The decode kernel alone reaches about 24 GB/s at 4k
context, the memory bandwidth of the test machine.

# Acceptance Criteria

* [x] Decode at 1.8k context is at least 85% of decode at 256
* [x] `script/eval` scores do not drop
* [x] Cache memory per token halves

`script/eval` on 4B with f16: parse 40/40, tool choice 40/40, arguments
40/40, runs 7/8, the same as f32. The failing run is the same name.txt task.
