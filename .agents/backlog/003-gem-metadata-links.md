As a `person who finds elelem-candle on RubyGems`, I `want the gem page to
link to its docs, backlog and changelog`, so that `I can learn to use it and
see known problems`.

# SYNOPSIS

Every gemspec link resolves.

# DESCRIPTION

The gemspec links are set: homepage and documentation at
https://elelem.sh/plugins/candle/, source and changelog at this repository,
and the bug tracker at `.agents/backlog`. They 404 until this repository is
public and elelem.sh is deployed with the candle page.

# Acceptance Criteria

* [x] `gem spec` on the built gem shows every link
* [ ] Every link resolves at release time
