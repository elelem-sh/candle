# frozen_string_literal: true

require_relative "lib/elelem/candle/version"

Gem::Specification.new do |spec|
  spec.name = "elelem-candle"
  spec.version = Elelem::Candle::VERSION
  spec.authors = ["mo khan"]
  spec.email = ["mo@mokhan.ca"]

  spec.summary = "An elelem provider that runs Qwen3 GGUF models on the CPU with candle."
  spec.description = "Runs Qwen3 GGUF models in process with huggingface/candle, with tool calls " \
                     "and a reused prompt cache. Prebuilt for Linux, macOS and Windows."
  spec.homepage = "https://elelem.sh/plugins/candle/"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 4.0.0"
  spec.metadata["allowed_push_host"] = "https://rubygems.org"
  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/elelem-sh/candle"
  spec.metadata["documentation_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "https://github.com/elelem-sh/candle/blob/main/CHANGELOG.md"
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.files = Dir[
    "LICENSE.txt",
    "LICENSE-THIRD-PARTY.txt",
    "lib/**/*.rb",
    "exe/*",
    "Cargo.toml",
    "Cargo.lock",
    "ext/elelem/candle/Cargo.toml",
    "ext/elelem/candle/build.rs",
    "ext/elelem/candle/extconf.rb",
    "ext/elelem/candle/src/**/*.rs"
  ]
  spec.extensions = ["ext/elelem/candle/extconf.rb"]
  spec.bindir = "exe"
  spec.executables = ["elelem-candle"]
  spec.require_paths = ["lib"]

  spec.add_dependency "elelem", ">= 0.12", "< 1"
  spec.add_dependency "rb_sys", "~> 0.9"
end
