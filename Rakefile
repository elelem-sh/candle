# frozen_string_literal: true

require "bundler/gem_tasks"
require "rb_sys/extensiontask"

task build: :compile

GEMSPEC = Gem::Specification.load("elelem-candle.gemspec")

RbSys::ExtensionTask.new("candle", GEMSPEC) do |ext|
  ext.ext_dir = "ext/elelem/candle"
  ext.lib_dir = "lib/elelem/candle"
  ext.cross_compile = true
  ext.cross_platform = %w[x86_64-linux aarch64-linux x86_64-darwin arm64-darwin x64-mingw-ucrt]
end

task default: :compile
