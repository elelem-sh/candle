# frozen_string_literal: true

require "bundler/gem_tasks"
require "rb_sys/extensiontask"

GEMSPEC = Gem::Specification.load("elelem-candle.gemspec")

RbSys::ExtensionTask.new("candle", GEMSPEC) do |ext|
  ext.ext_dir = "ext/elelem/candle"
  ext.lib_dir = "lib/elelem/candle"
end

Rake::Task["lib/elelem/candle/candle.so"].clear_prerequisites if ENV.key?("RUBY_TARGET")

task default: :compile

desc "Write the licenses of the crates the extension links to LICENSE-THIRD-PARTY.txt"
task :licenses do
  sh "cargo", "about", "generate", "--manifest-path", "ext/elelem/candle/Cargo.toml",
     "-c", "about.toml", "-o", "LICENSE-THIRD-PARTY.txt", "about.hbs"
end
