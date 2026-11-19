# frozen_string_literal: true

module Elelem
  module Candle
    class Provider
      THINK = %r{\A\s*<think>.*?(?:</think>|\z)}m

      def self.build(path:, env: ENV)
        context = Integer(env.fetch("ELELEM_CANDLE_N_CTX", "8192"))
        model = (@models ||= {})[[path, context]] ||= Model.load(path, context)
        new(model:, thinking: env.fetch("ELELEM_CANDLE_THINKING", "0") == "1", options: options(env))
      end

      def self.options(env)
        seed = Integer(env.fetch("ELELEM_CANDLE_SEED", "-1"))
        {
          max_tokens: Integer(env.fetch("ELELEM_CANDLE_MAX_TOKENS", "0")),
          temperature: Float(env.fetch("ELELEM_CANDLE_TEMP", "0.6")),
          top_k: Integer(env.fetch("ELELEM_CANDLE_TOP_K", "20")),
          top_p: Float(env.fetch("ELELEM_CANDLE_TOP_P", "0.95")),
          seed: seed.negative? ? Random.new_seed % (2**63) : seed,
          chunk: 128
        }
      end

      def initialize(model:, thinking: false, options: self.class.options({}))
        @model = model
        @thinking = thinking
        @options = options
      end

      def fetch(messages, _tools = [], &block)
        prompt = @model.render(JSON.generate(messages), "[]", @thinking)
        text = +""
        stats = @model.generate(prompt, @options, proc { |piece|
          text << piece
          block&.call(type: "thinking", text: piece)
        })
        Elelem.logger.debug { "candle: #{stats}" }
        reply = text.sub(THINK, "").strip
        reply = "[no reply: ran out of output tokens]" if reply.empty? && stats[:stop_reason] == "length"
        block&.call(type: "saying", text: reply) unless reply.empty?
        []
      end
    end
  end
end
