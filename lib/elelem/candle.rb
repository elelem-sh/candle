# frozen_string_literal: true

require "json"
require "elelem"

require_relative "candle/version"
require "elelem/candle/candle" # native extension (ext/elelem/candle)
require_relative "candle/provider"

Elelem.configure do |config|
  config.provider(:candle) do
    Elelem::Candle::Provider.new(
      model: ENV.fetch("ELELEM_CANDLE_MODEL") { raise "set ELELEM_CANDLE_MODEL=/path/to/model (safetensors dir or hf repo id)" }
    )
  end
end
