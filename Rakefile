# frozen_string_literal: true

require "bundler/gem_tasks"
require "rb_sys/extensiontask"

task build: :compile

GEMSPEC = Gem::Specification.load("elelem-candle.gemspec")

RbSys::ExtensionTask.new("candle", GEMSPEC) do |ext|
  ext.ext_dir = "ext/elelem/candle"
  ext.lib_dir = "lib/elelem/candle"
end

task default: :compile
