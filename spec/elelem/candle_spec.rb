# frozen_string_literal: true

RSpec.describe "candle provider" do
  before { allow(Elelem::Candle::Provider).to receive(:build).and_return(:provider) }

  around do |example|
    saved = ENV.delete("ELELEM_CANDLE_MODEL")
    example.run
  ensure
    ENV["ELELEM_CANDLE_MODEL"] = saved
  end

  it "uses --model" do
    ENV["ELELEM_CANDLE_MODEL"] = "/env.gguf"
    Elelem::Config.build_provider("candle", model: "/flag.gguf")
    expect(Elelem::Candle::Provider).to have_received(:build).with(path: "/flag.gguf")
  end

  it "falls back to ELELEM_CANDLE_MODEL" do
    ENV["ELELEM_CANDLE_MODEL"] = "/env.gguf"
    Elelem::Config.build_provider("candle")
    expect(Elelem::Candle::Provider).to have_received(:build).with(path: "/env.gguf")
  end

  it "names both ways to set the model" do
    expect { Elelem::Config.build_provider("candle") }.to raise_error(/--model.*ELELEM_CANDLE_MODEL/)
  end
end
