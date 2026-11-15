# frozen_string_literal: true

RSpec.describe Elelem::Candle do
  it "has a semantic version" do
    expect(Elelem::Candle::VERSION).to match(/\A\d+\.\d+\.\d+\z/)
  end
end
