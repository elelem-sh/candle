# frozen_string_literal: true

module Elelem
  module Candle
    class Provider
      THINK = %r{\A\s*<think>.*?(?:</think>|\z)}m
      SAMPLING = {
        thinking: { temperature: 0.6, top_p: 0.95, max_tokens: 2048 },
        direct: { temperature: 0.7, top_p: 0.8, max_tokens: 1024 }
      }.freeze

      def self.build(path:, env: ENV)
        context = Integer(env.fetch("ELELEM_CANDLE_N_CTX", "8192"))
        model = (@models ||= {})[[path, context]] ||= Model.load(path, context)
        thinking = env.fetch("ELELEM_CANDLE_THINKING", "0") == "1"
        new(model:, thinking:, options: options(env, thinking:))
      end

      def self.options(env, thinking: false)
        seed = Integer(env.fetch("ELELEM_CANDLE_SEED", "-1"))
        defaults = thinking ? SAMPLING[:thinking] : SAMPLING[:direct]
        {
          max_tokens: Integer(env.fetch("ELELEM_CANDLE_MAX_TOKENS", defaults[:max_tokens].to_s)),
          temperature: Float(env.fetch("ELELEM_CANDLE_TEMP", defaults[:temperature].to_s)),
          top_k: Integer(env.fetch("ELELEM_CANDLE_TOP_K", "20")),
          top_p: Float(env.fetch("ELELEM_CANDLE_TOP_P", defaults[:top_p].to_s)),
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
