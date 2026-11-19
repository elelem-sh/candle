As a `contributor building elelem-candle`, I `want the extension to
depend only on crates it uses`, so that `the first build is fast and
the gem install is light while inference is a stub`.

# SYNOPSIS

Remove candle-core, candle-nn, candle-transformers, tokenizers and
hf-hub until inference lands.

# DESCRIPTION

ext/elelem/candle/Cargo.toml depends on candle-core, candle-nn,
candle-transformers, tokenizers and hf-hub. src/lib.rs uses none of
them: it imports only magnus, and the model is a skeleton whose
generate returns a placeholder (lib.rs:5-15, :49). Every build, and
every `gem install`, compiles these crates and their dependency trees
for nothing.

Remove them from Cargo.toml and update Cargo.lock. Add them back in the
story that implements inference, with the comment about CUDA and Metal
features.

# SEE ALSO

* [ ] ext/elelem/candle/Cargo.toml - unused dependencies
* [ ] ext/elelem/candle/src/lib.rs:1-15 - imports and skeleton notes
* [ ] Cargo.lock
* [ ] Formerly elelem-sh/elelem.sh#40

# Tasks

* [ ] TBD (filled in design mode)

# Acceptance Criteria

* [ ] Cargo.toml lists only crates that src/lib.rs uses
* [ ] Cargo.lock no longer contains candle-core, tokenizers or hf-hub
* [ ] bin/compile and bin/test pass
