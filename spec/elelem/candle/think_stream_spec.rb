# frozen_string_literal: true

RSpec.describe Elelem::Candle::ThinkStream do
  def stream(pieces)
    think = described_class.new
    pieces.map { think.push(it) }.join
  end

  text = "<think>\nThe user wants <b> a file.\n</think>\n\nHere it is."

  it "shows only the thinking" do
    expect(stream([text])).to eq("The user wants <b> a file.")
  end

  (1...text.size).each do |at|
    it "gives the same text when split at #{at}" do
      expect(stream([text[0, at], text[at..]])).to eq("The user wants <b> a file.")
    end
  end

  it "gives the same text one character at a time" do
    expect(stream(text.chars)).to eq("The user wants <b> a file.")
  end

  it "shows nothing without a think block" do
    expect(stream(["Hello <think> there"])).to eq("")
  end

  it "shows an unclosed think block as it grows" do
    expect(stream(["<think>\nstill ", "going"])).to eq("still going")
  end
end
