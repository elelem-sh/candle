# Changelog

## [0.2.0](https://github.com/elelem-sh/candle/compare/v0.1.0...v0.2.0) (2026-11-25)


### Features

* add a benchmark script for prefill, decode and a simulated tool step ([41436eb](https://github.com/elelem-sh/candle/commit/41436eb3b0b1048467fa6600faa336d5e3d6b5d7))
* call tools and replay each call exactly as the model wrote it ([4e8e3e3](https://github.com/elelem-sh/candle/commit/4e8e3e32bd6eaf9484f4bd31b4064f1e8c309130))
* count prompt tokens and cut the newest tool result to fit the context ([aea1106](https://github.com/elelem-sh/candle/commit/aea1106e4a171159889d397393dd4beee93425b4))
* link the gem page to its docs, backlog and changelog ([da150d6](https://github.com/elelem-sh/candle/commit/da150d64bb7a5b7b5ef6ba8ea51e54fc1656e98e))
* parse thinking, strict tool calls and bare json calls from a reply ([84b0761](https://github.com/elelem-sh/candle/commit/84b0761f0e9ad9ba51f31148b835935e14ec2fef))
* reuse the cached prompt prefix across requests ([77e9aba](https://github.com/elelem-sh/candle/commit/77e9abace2af45e4168abcc873390e3e17d0ce6d))
* run qwen3 gguf models on the cpu with candle ([85bc1ae](https://github.com/elelem-sh/candle/commit/85bc1aed9b3ae09f387a7fcd13dd5161d95d1eca))
* sample with qwen3 settings for each thinking mode and cap output tokens ([8c44b43](https://github.com/elelem-sh/candle/commit/8c44b43b4331e6f47313d4565fcfc5a319ef1646))
* stop on caller supplied tokens and report which one ([b526a74](https://github.com/elelem-sh/candle/commit/b526a745bfb6c1e2c54f9f73e5023b7b43d8df63))
* stream only the thinking text while tags arrive split across pieces ([0e6cb5b](https://github.com/elelem-sh/candle/commit/0e6cb5b7830b7f0dc87e4195810df3973f1d4905))
* vendor the qwen3 model with one kv cache that prefill and decode share ([69f37c8](https://github.com/elelem-sh/candle/commit/69f37c88bcb53039245e96f6cf6d94b7a94bf69a))


### Bug Fixes

* build rotary tables in f32 so long prompts keep their positions ([89a9a0b](https://github.com/elelem-sh/candle/commit/89a9a0b831bf5ddc038cfeeb312a9cf31b589fac))
* check for interrupts before each layer while reading a prompt ([f5914ae](https://github.com/elelem-sh/candle/commit/f5914aeba815ff567ec0f6c21ea3a226109e0f2f))
* check for interrupts on every decoded token ([063d3b1](https://github.com/elelem-sh/candle/commit/063d3b17751dd0cc010b19f0a0bdd8c9943c00cf))
* count only newly read tokens in the bench prefill rate ([c5ce372](https://github.com/elelem-sh/candle/commit/c5ce37213da0f3b971844bb05bd482f2480276f5))
* detect a prebuilt gem by its platform ([d3c8bc6](https://github.com/elelem-sh/candle/commit/d3c8bc6dffb9029172c67938e4bb1d0fc364dead))
* keep the attention tile inside heads narrower than 64 and smoke test generation on every platform ([00f3166](https://github.com/elelem-sh/candle/commit/00f3166799caad4ad3fb844d4af660a9a711335e))
* let read-only model calls run while generate yields ([febf9ed](https://github.com/elelem-sh/candle/commit/febf9ed1ba38976b168847fe2c8ecaa3745bcc73))
* quote the eval prompt that contains json ([cb7ffd7](https://github.com/elelem-sh/candle/commit/cb7ffd7df7c1fcd78cb4bdac81773e7f87bf36c5))
* render tojson like python so prompts match what qwen3 was trained on ([6e9aa30](https://github.com/elelem-sh/candle/commit/6e9aa30c1b5c2bfab8600142c00dc21b26b6d95c))
* run only the first tool call of a reply so later calls see its result ([8902bd2](https://github.com/elelem-sh/candle/commit/8902bd24b0934c828d81c333470556a0265ee898))


### Performance Improvements

* read prompts with gemm attention straight from the kv cache ([237b121](https://github.com/elelem-sh/candle/commit/237b1211620cff90d2fc8817309eb1579845bf59))
* store the kv cache as f16 and read it once per kv head in decode ([e829f49](https://github.com/elelem-sh/candle/commit/e829f493818d75f77a759ada494c97dc33c2c250))
* time each section of the forward pass when profiling ([ca0b72e](https://github.com/elelem-sh/candle/commit/ca0b72ee0ec74ef8a67659f442e4c10d2d612715))

## 0.1.0 (2026-11-17)


### Bug Fixes

* build the rust extension (magnus RubyMethod2 argument order, block-&gt;Proc) ([cee27f1](https://github.com/elelem-sh/candle/commit/cee27f1f59595512dc5dac03da866a930d780ef9))
