As a `user in a long elelem-candle session`, I `want each turn to read
only the new messages`, so that `replies do not slow down as the
conversation grows`.

# SYNOPSIS

Keep the KV cache between requests and drop only what changed.

# DESCRIPTION

elelem sends the whole conversation on every request. `Engine::generate`
clears the KV cache and reads every token again
(ext/elelem/candle/src/engine.rs). candle-transformers'
`quantized_qwen3::ModelWeights` can only clear its cache, not cut it to a
length, so reuse means vendoring that model and adding a truncate.

Keep the token ids of the last request, find the shared prefix with the
next one, cut the cache to it and feed only the rest.

# SEE ALSO

* [ ] ext/elelem/candle/src/engine.rs - generate
* [ ] candle-transformers 0.11 src/models/quantized_qwen3.rs - KV caches
* [ ] candle-nn 0.11 src/kv_cache.rs - RawInterleavedKvCache

# Tasks

* [x] Vendor quantized_qwen3 as `qwen3.rs` with one KV cache shared by prefill and decode
* [x] Keep the last request's token ids, cut the cache to the shared prefix and read the rest
* [x] Roll the cache back when a forward pass fails or is interrupted

# Acceptance Criteria

* [x] The second turn reads only its new tokens
* [x] A conversation that changes earlier messages still gives correct output
* [x] Stats report how many prompt tokens were reused
