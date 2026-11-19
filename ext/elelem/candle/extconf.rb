# frozen_string_literal: true

require "mkmf"
require "rb_sys/mkmf"

create_rust_makefile("elelem/candle/candle") do |r|
  r.extra_rustflags = ["-C", "target-cpu=#{ENV.fetch("ELELEM_CANDLE_TARGET_CPU", "native")}"]
end
