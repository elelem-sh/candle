As a `user trying elelem-candle`, I `want --model to choose the
model`, so that `the provider works like the other providers`.

# SYNOPSIS

Accept `model:` in the provider block, and fall back to the environment.

# DESCRIPTION

Core passes `--model NAME` to the provider block as `model:`. The
candle provider block takes no arguments (candle.rb:11), so `--model`
does nothing and the model comes only from `ELELEM_CANDLE_MODEL`.

Accept `model:` the way elelem-llama and elelem-ollama do
(`do |model: nil, **|`), and fall back to `ELELEM_CANDLE_MODEL`.
Inference is still a stub (ext/elelem/candle/src/lib.rs:5), so this
only changes which value reaches it.

# SEE ALSO

* [ ] lib/elelem/candle.rb:10-18 - provider block
* [ ] elelem lib/elelem/cli.rb:42 - --model
* [ ] Formerly elelem-sh/elelem.sh#29

# Tasks

* [ ] TBD (filled in design mode)

# Acceptance Criteria

* [ ] `elelem chat -p candle --model X` passes X to the provider
* [ ] Without `--model`, `ELELEM_CANDLE_MODEL` is used as today
* [ ] With neither, the error names both ways to set the model
* [ ] A spec covers each case
