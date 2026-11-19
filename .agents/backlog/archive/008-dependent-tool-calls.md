As a `user of elelem-candle with the 4B model`, I `want multi-step tasks to finish correctly`, so that `copying a file copies its contents`.

# SYNOPSIS

Stop the model from writing a placeholder for a result it has not seen yet.

# DESCRIPTION

`script/eval` runs 8 multi-step tasks; 4B passes 5. In the failures the
model sends two calls in one reply, for example `read(a.txt)` and
`write(b.txt, "<content>")`, so the write runs before the read result
exists and the file gets the placeholder. The model then reports success.

Options:

1. Run only the first call of a reply and drop the rest, so every later
   step sees the earlier result. Costs a request per call and changes the
   replayed history, so cache reuse stops at that reply.
2. Keep parallel calls, and add a line to the system prompt telling small
   models to make one call at a time when a call needs another's result.
3. Leave it to the model and document it.

# SEE ALSO

* [ ] lib/elelem/candle/provider.rb - remember
* [ ] script/eval_tasks.yml - run tasks 43, 45 and 46

# Tasks

* [x] Option 1: `Provider#remember` keeps the first call of a reply; `Reply` still parses them all

# Acceptance Criteria

* [x] `script/eval` passes at least 7 of 8 multi-step tasks on 4B (5/8 before, 7/8 after)
* [x] Single-step scores do not drop (40/40 on parse, tool and arguments)
