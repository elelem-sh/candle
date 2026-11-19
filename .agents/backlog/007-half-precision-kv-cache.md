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

* [ ] ext/elelem/candle/src/qwen3.rs - KvCache
* [ ] ext/elelem/candle/src/attention.rs - causal
* [ ] candle-nn 0.11 src/attention/cpu_flash/causal.rs - causal_decode_f32_interleaved

# Tasks

* [ ] TBD (filled in design mode)

# Acceptance Criteria

* [ ] Decode at 1.8k context is at least 85% of decode at 256
* [ ] `script/eval` scores do not drop
* [ ] Cache memory per token halves
