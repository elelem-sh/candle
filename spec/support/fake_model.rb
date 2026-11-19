# frozen_string_literal: true

class FakeModel
  attr_reader :rendered, :options
  attr_accessor :stop_reason

  def initialize(pieces)
    @pieces = pieces
    @stop_reason = "stop"
  end

  def render(*args)
    @rendered = args
    "prompt"
  end

  def generate(_prompt, options, on_piece)
    @options = options
    @pieces.each { on_piece.call(it) }
    { stop_reason: }
  end
end
