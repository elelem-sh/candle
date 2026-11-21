# elelem-candle

An [elelem](https://elelem.sh) provider that runs Qwen3 GGUF models on the
CPU with [candle](https://github.com/huggingface/candle), compiled into Ruby
as a native extension.

## Install

Needs Ruby 4.0, a Rust toolchain and `clang`.

```sh
gem install elelem-candle
ELELEM_CANDLE_MODEL=~/.agents/models/default.gguf elelem -p candle
```

## What it does

- Reads the tokenizer and chat template from the GGUF file, and converts
  IQ4_XS tensors to Q8_0 at load time.
- Calls tools. Each `<tool_call>` becomes a step, one per reply.
- Reuses the cached prompt from the last request, so a tool step reads only
  the new tokens.
- Keeps the KV cache in f16.

Use Qwen3-4B. Qwen3-0.6B is fast but often picks the wrong tool.

| Model | Right tool (`script/eval`) | Multi-step | Prompt reading | Replying |
|---|---|---|---|---|
| Qwen3-4B | 40/40 | 7/8 | 15 tokens/s | 5 tokens/s |
| Qwen3-0.6B | 22/40 | 3/8 | 95 tokens/s | 28 tokens/s |

Measured on a 4-core x86_64 with AVX2 (`script/bench`).

## Limits

- Qwen3 only, CPU only.
- Replies slow down as context grows: 0.6B drops to 22 tokens/s at 4.1k
  and 14 at 7.5k.
- Only thinking streams; the reply appears when it is done.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `ELELEM_CANDLE_MODEL` | | Path to a Qwen3 `.gguf`, unless `--model` is given |
| `ELELEM_CANDLE_N_CTX` | `8192` | Context in tokens, capped at the model's |
| `ELELEM_CANDLE_MAX_TOKENS` | `1024`, `2048` thinking | Output tokens per reply; 0 is the rest of the context |
| `ELELEM_CANDLE_TEMP` | `0.7`, `0.6` thinking | Temperature; 0 is greedy |
| `ELELEM_CANDLE_TOP_K` | `20` | Top-k |
| `ELELEM_CANDLE_TOP_P` | `0.8`, `0.95` thinking | Top-p |
| `ELELEM_CANDLE_SEED` | `-1` | Sampling seed; negative is random |
| `ELELEM_CANDLE_THINKING` | `0` | `1` lets the model think first |
| `ELELEM_CANDLE_TARGET_CPU` | `native` | Build time `-C target-cpu` |

## Develop

```sh
bin/setup
bin/test
ELELEM_CANDLE_TEST_MODEL=~/.agents/models/default.gguf bin/test
script/bench MODEL [--step] [--profile]
script/eval MODEL
```

The second `bin/test` also runs the specs that load a real model.
`script/eval` runs 40 one-step and 8 multi-step tool tasks, takes about 9
minutes on 4B, and exits 1 under 85%.

## License

MIT. See LICENSE.txt and LICENSE-THIRD-PARTY.txt.
