# frozen_string_literal: true

require "json"
require "securerandom"
require "elelem"

require_relative "candle/version"
require_relative "candle/native"
Elelem::Candle::Native.load
require_relative "candle/reply"
require_relative "candle/think_stream"
require_relative "candle/provider"

Elelem.configure do |config|
  config.provider(:candle) do |model: nil, **|
    path = model || ENV.fetch("ELELEM_CANDLE_MODEL") do
      raise "no model: pass --model /path/to/model.gguf or set ELELEM_CANDLE_MODEL"
    end
    Elelem::Candle::Provider.build(path: File.expand_path(path))
  end
end
