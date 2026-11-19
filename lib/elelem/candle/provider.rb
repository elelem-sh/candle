# frozen_string_literal: true

module Elelem
  module Candle
    class Provider
      STOP = ["<|im_start|>", "<tool_response>"].freeze
      RESERVE = 256
      MINIMUM = 200
      CUT = "\n[... %<removed>d characters removed ...]\n"
      FULL = "[context full: the prompt is %<size>d tokens, limit %<limit>d. " \
             "Start a new session or raise ELELEM_CANDLE_N_CTX]"
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
        @calls = {}
      end

      def fetch(messages, tools = [], &block)
        prompt, size = fit(replay(messages), tools)
        unless prompt
          block&.call(type: "saying", text: format(FULL, size:, limit: @model.context - RESERVE))
          return []
        end

        text = +""
        think = ThinkStream.new
        stats = @model.generate(prompt, @options.merge(stop: tools.empty? ? STOP.take(1) : STOP), proc { |piece|
          text << piece
          shown = think.push(piece)
          block&.call(type: "thinking", text: shown) unless shown.empty?
        })
        Elelem.logger.debug { "candle: #{stats}" }
        reply = Reply.parse(text, thinking: @thinking, stop_reason: stats[:stop_reason], tools:)
        say(reply, stats, &block)
        remember(reply).each { block&.call(it.merge(type: "doing")) }
      end

      private

      def fit(messages, tools)
        loop do
          prompt = @model.render(JSON.generate(messages), JSON.generate(tools), @thinking)
          size = @model.count(prompt)
          over = size - (@model.context - RESERVE)
          return [prompt, size] unless over.positive?

          messages = shorten(messages, over)
          return [nil, size] unless messages
        end
      end

      def shorten(messages, over)
        tail = messages.reverse.take_while { it[:role].to_s == "tool" }
        biggest = tail.max_by { it[:content].to_s.size }
        text = biggest&.dig(:content).to_s
        keep = text.size - (over * 4) - CUT.size - 40
        return if keep < MINIMUM

        cut = "#{text[0, keep / 2]}#{format(CUT, removed: text.size - keep)}#{text[-(keep - (keep / 2))..]}"
        messages.map { it.equal?(biggest) ? it.merge(content: cut) : it }
      end

      def say(reply, stats, &block)
        lines = [reply.content, *reply.errors].reject { it.strip.empty? }
        if lines.empty? && reply.calls.empty? && stats[:stop_reason] == "length"
          lines << "[no reply: ran out of output tokens]"
        end
        block&.call(type: "saying", text: lines.join("\n\n")) unless lines.empty?
      end

      def remember(reply)
        reply.calls.map.with_index do |call, i|
          id = "call_#{SecureRandom.alphanumeric(12)}"
          @calls[id] = { raw: call.raw, reasoning: i.zero? ? reply.reasoning : "" }
          { id:, name: call.name, arguments: call.arguments }
        end
      end

      def replay(messages)
        seen = []
        replayed = messages.map do |message|
          message = message.transform_keys(&:to_sym)
          next message unless message[:tool_calls]

          calls = message[:tool_calls].map { it.transform_keys(&:to_sym) }
          seen.concat(calls.map { it[:id] })
          kept = calls.map { @calls[it[:id]] }
          reasoning = kept.first&.fetch(:reasoning).to_s
          message = message.merge(tool_calls: calls.zip(kept).map { |c, k| k ? c.merge(arguments: k[:raw]) : c })
          reasoning.empty? ? message : message.merge(reasoning_content: reasoning)
        end
        @calls.select! { |id, _| seen.include?(id) }
        replayed
      end
    end
  end
end
