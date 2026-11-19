As a `user running elelem-candle`, I `want the model to call tools`, so
that `it can read files and run commands like the other providers`.

# SYNOPSIS

Send the toolbox to the chat template and turn `<tool_call>` blocks into
`doing` events.

# DESCRIPTION

`Provider#fetch` renders the template with no tools and returns `[]`
(lib/elelem/candle/provider.rb). Qwen3's template lists tools in the
system prompt and the model answers with
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`. The tokenizer
keeps `<think>`, `<tool_call>` and `</tool_call>` as text, so a Ruby
parser over the streamed pieces can split thinking, tool calls and the
reply.

# SEE ALSO

* [ ] lib/elelem/candle/provider.rb - fetch
* [ ] ext/elelem/candle/src/template.rs - render
* [ ] elelem lib/elelem/agent.rb - doing events and tool messages

# Tasks

* [ ] TBD (filled in design mode)

# Acceptance Criteria

* [ ] Tools reach the template
* [ ] Each `<tool_call>` becomes a `doing` event with a Hash of arguments
* [ ] A tag split across streamed pieces still parses
* [ ] Bad JSON in a tool call is reported, not raised
* [ ] `elelem -p candle` runs `read` on a file and answers from it
