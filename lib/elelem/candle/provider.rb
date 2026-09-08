# frozen_string_literal: true

module Elelem
  module Candle
    # Thin Ruby wrapper around the native (Rust/candle) model. The Rust side
    # (ext/elelem/candle/src/lib.rs) does the actual tensor loading/inference;
    # this class just adapts elelem's provider contract to it.
    class Provider
      def initialize(model:, device: ENV.fetch("ELELEM_CANDLE_DEVICE", "cpu"))
        @model = Elelem::Candle::Model.new(model, device)
      end

      def fetch(messages, tools = [], &block)
        # TODO: real chat templating + tool-call parsing, mirroring
        # elelem-llama's provider.rb. For now: single-turn text generation.
        prompt = messages.map { |m| "#{m[:role] || m["role"]}: #{m[:content] || m["content"]}" }.join("\n")

        content = +""
        @model.generate(prompt) do |piece|
          content << piece
          block&.call(type: "thinking", text: piece)
        end

        block&.call(type: "saying", text: content)
        []
      end
    end
  end
end
