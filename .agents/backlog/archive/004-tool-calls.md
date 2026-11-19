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

* [x] Render tools with a `tojson` that matches Python byte for byte
* [x] Parse thinking, strict `<tool_call>` blocks and a strict bare JSON fallback (`Reply`)
* [x] Stream only the thinking text, holding back split tags (`ThinkStream`)
* [x] Stop on `<|im_start|>` and `<tool_response>` when tools are offered
* [x] Replay each call's raw arguments and reasoning so the next prompt extends the last
* [x] Count prompt tokens and cut the newest tool result to fit the context
* [x] `script/eval`: 4B scores 40/40 on parse, tool choice and arguments

# Acceptance Criteria

* [x] Tools reach the template
* [x] Each `<tool_call>` becomes a `doing` event with a Hash of arguments
* [x] A tag split across streamed pieces still parses
* [x] Bad JSON in a tool call is reported, not raised
* [x] `elelem -p candle` runs `read` on a file and answers from it
