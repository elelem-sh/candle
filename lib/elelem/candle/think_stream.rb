# frozen_string_literal: true

module Elelem
  module Candle
    class ThinkStream
      OPEN = "<think>"
      CLOSE = "</think>"

      def initialize
        @text = +""
        @sent = 0
      end

      def push(piece)
        @text << piece
        visible = thinking
        return "" if visible.nil? || visible.size <= @sent

        visible[@sent..].tap { @sent = visible.size }
      end

      private

      def thinking
        start = @text.lstrip
        return unless start.start_with?(OPEN)

        body = start.delete_prefix(OPEN).sub(/\A\n+/, "")
        stop = body.index(CLOSE)
        return body[0...stop].rstrip if stop

        body[0...(body.size - held(body))].rstrip
      end

      def held(body)
        (CLOSE.size - 1).downto(1).find { body.end_with?(CLOSE[0, it]) } || 0
      end
    end
  end
end
