# frozen_string_literal: true

class FakeModel
  attr_reader :rendered, :options, :prompt
  attr_accessor :stop_reason, :context

  def initialize(pieces)
    @pieces = pieces
    @stop_reason = "stop"
    @context = 100_000
  end

  def render(*args)
    @rendered = args
    args.take(2).join
  end

  def count(text) = (text.size / 4.0).ceil

  def generate(prompt, options, on_piece)
    @prompt = prompt
    @options = options
    @pieces.each { on_piece.call(it) }
    { stop_reason: }
  end
end
