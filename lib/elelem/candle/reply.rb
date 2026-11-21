# frozen_string_literal: true

module Elelem
  module Candle
    class Reply
      Call = Data.define(:name, :arguments, :raw)

      OPEN = "<tool_call>"
      CALL = %r{<tool_call>(.*?)</tool_call>}m
      ARGUMENTS = /\A\s*\{\s*"name"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"(?:arguments|parameters)"\s*:\s*(.*)\}\s*\z/m
      FENCE = /\A```(?:json)?\s*\n(.*?)\n?```\z/m
      TRUNCATED = "[tool call truncated: the reply ran out of output tokens. Raise ELELEM_CANDLE_MAX_TOKENS]"

      attr_reader :reasoning, :content, :calls, :errors

      def self.parse(text, thinking:, stop_reason:, tools: [])
        new(text, thinking:, stop_reason:, names: tools.filter_map { name_of(it) })
      end

      def self.name_of(tool)
        function = tool[:function] || tool["function"] || tool
        function[:name] || function["name"]
      end

      def initialize(text, thinking:, stop_reason:, names:)
        @reasoning, rest = split_thinking(text, thinking)
        @errors = []
        @calls = []
        @content = rest.sub(/\A\n+/, "")
        if @content.include?(OPEN)
          extract(stop_reason)
        elsif stop_reason == "stop" && names.any?
          fallback(names)
        end
      end

      private

      def split_thinking(text, thinking)
        if (m = text.match(%r{\A\s*<think>(.*?)</think>}m))
          [m[1].strip, m.post_match]
        elsif text.match?(/\A\s*<think>/)
          [text.sub(/\A\s*<think>/, "").strip, ""]
        elsif thinking && text.include?("</think>")
          reasoning, _, rest = text.partition("</think>")
          [reasoning.strip, rest]
        else
          ["", text]
        end
      end

      def extract(stop_reason)
        head, _, tail = @content.partition(OPEN)
        @content = head.delete_suffix("\n")
        tail = OPEN + tail
        return @errors << TRUNCATED if tail.gsub(CALL, "").include?(OPEN) || stop_reason != "stop"

        @calls = tail.scan(CALL).flatten.map { call(it) }.uniq { [it.name, it.arguments] }
      end

      def fallback(names)
        body = @content.strip
        body = body[FENCE, 1] || body
        object = parse_json(body)
        return unless object.is_a?(Hash) && names.include?(object["name"])
        return unless object.key?("arguments") || object.key?("parameters")

        @calls << call(body)
        @content = ""
      end

      def call(body)
        object = parse_json(body)
        return Call.new(name: body[/"name"\s*:\s*"([^"]+)"/, 1].to_s, arguments: body.strip, raw: body.strip) unless
          object.is_a?(Hash)

        value = object.fetch("arguments") { object["parameters"] }
        Call.new(name: object["name"].to_s, arguments: arguments_of(value), raw: raw_of(body, value))
      end

      def arguments_of(value)
        case value
        when nil then {}
        when String
          parsed = parse_json(value)
          parsed.is_a?(Hash) ? parsed : value
        else value
        end
      end

      def raw_of(body, value)
        raw = body[ARGUMENTS, 1]&.rstrip
        return raw if raw && parse_json(raw) == value

        JSON.generate(value || {})
      end

      def parse_json(text)
        JSON.parse(text)
      rescue JSON::ParserError
        nil
      end
    end
  end
end
