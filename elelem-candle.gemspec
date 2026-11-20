# frozen_string_literal: true

require_relative "lib/elelem/candle/version"

Gem::Specification.new do |spec|
  spec.name = "elelem-candle"
  spec.version = Elelem::Candle::VERSION
  spec.authors = ["mo khan"]
  spec.email = ["mo@mokhan.ca"]

  spec.summary = "A candle (huggingface/candle) plugin for elelem."
  spec.description = "A candle (huggingface/candle) plugin for elelem."
  spec.homepage = "https://elelem.sh/plugins/candle/"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 4.0.0"
  spec.metadata["allowed_push_host"] = "https://rubygems.org"
  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/elelem-sh/candle"
  spec.metadata["documentation_uri"] = spec.homepage
  spec.metadata["bug_tracker_uri"] = "https://github.com/elelem-sh/candle/tree/main/.agents/backlog"
  spec.metadata["changelog_uri"] = "https://github.com/elelem-sh/candle/blob/main/CHANGELOG.md"
  spec.metadata["rubygems_mfa_required"] = "true"

  gemspec = File.basename(__FILE__)
  spec.files = IO.popen(%w[git ls-files -z], chdir: __dir__, err: IO::NULL) do |ls|
    ls.readlines("\x0", chomp: true).reject do |f|
      (f == gemspec) || f.start_with?(*%w[bin/ Gemfile .gitignore target/])
    end
  end
  spec.extensions = ["ext/elelem/candle/extconf.rb"]
  spec.bindir = "exe"
  spec.executables = ["elelem-candle"]
  spec.require_paths = ["lib"]

  spec.add_dependency "elelem", ">= 0.12", "< 1"
  spec.add_dependency "rb_sys", "~> 0.9"
end
