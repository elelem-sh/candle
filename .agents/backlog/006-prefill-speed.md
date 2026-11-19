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

Ctrl-C is checked between prompt chunks of 128 tokens, so on the 4B model it
can take several seconds to stop.

# SEE ALSO

* [ ] ext/elelem/candle/src/engine.rs - chunked prefill
* [ ] candle-transformers 0.11 src/models/quantized_qwen3.rs - attention
* [ ] lib/elelem/candle/provider.rb - chunk size

# Tasks

* [ ] TBD (filled in design mode)

# Acceptance Criteria

* [ ] Prompt reading speed is measured before and after
* [ ] Reading speed does not drop with prompt length up to the context size
* [ ] Prompt reading uses all cores
