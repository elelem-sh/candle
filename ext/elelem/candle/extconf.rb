# frozen_string_literal: true

require "mkmf"
require "rb_sys/mkmf"

def target_cpu
  ENV.fetch("ELELEM_CANDLE_TARGET_CPU") do
    case ENV.fetch("RUBY_TARGET", nil)
    when nil then "native"
    when /\A(x86_64|x64)-/ then "x86-64-v3"
    end
  end
end

create_rust_makefile("elelem/candle/candle") do |r|
  r.extra_rustflags = ["-C", "target-cpu=#{target_cpu}"] if target_cpu
end
