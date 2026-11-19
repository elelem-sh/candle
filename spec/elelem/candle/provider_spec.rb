# frozen_string_literal: true

RSpec.describe Elelem::Candle::Provider do
  let(:model) { FakeModel.new(pieces) }
  let(:pieces) { %w[Hel lo] }
  let(:messages) { [{ role: "system", content: "terse" }, { role: "user", content: "hi" }] }

  def fetch(provider = described_class.new(model:))
    events = []
    result = provider.fetch(messages, [{ type: "function" }]) { |event| events << event }
    [events, result]
  end

  it "renders the conversation as JSON without tools" do
    fetch
    expect(model.rendered).to eq([JSON.generate(messages), "[]", false])
  end

  it "passes thinking to the template" do
    fetch(described_class.new(model:, thinking: true))
    expect(model.rendered.last).to be(true)
  end

  it "streams pieces as thinking, then says the reply" do
    events, = fetch
    expect(events).to eq([
                           { type: "thinking", text: "Hel" },
                           { type: "thinking", text: "lo" },
                           { type: "saying", text: "Hello" }
                         ])
  end

  it "returns no tool calls" do
    _, result = fetch
    expect(result).to eq([])
  end

  context "when the model thinks first" do
    let(:pieces) { ["<think>\nhmm\n</think>\n\n", "pong"] }

    it "leaves the thinking out of the reply" do
      events, = fetch
      expect(events.last).to eq(type: "saying", text: "pong")
    end
  end

  context "when output runs out before a reply" do
    let(:pieces) { ["<think>\nstill going"] }

    it "says so" do
      model.stop_reason = "length"
      events, = fetch
      expect(events.last).to eq(type: "saying", text: "[no reply: ran out of output tokens]")
    end
  end

  it "sends sampling options to the model" do
    fetch
    expect(model.options).to include(temperature: 0.7, top_k: 20, top_p: 0.8, max_tokens: 1024, chunk: 128)
  end

  describe ".options" do
    it "reads the environment" do
      options = described_class.options({ "ELELEM_CANDLE_TEMP" => "0", "ELELEM_CANDLE_SEED" => "7",
                                          "ELELEM_CANDLE_MAX_TOKENS" => "64" })
      expect(options).to include(temperature: 0.0, seed: 7, max_tokens: 64)
    end

    it "uses Qwen3's thinking settings when thinking" do
      expect(described_class.options({}, thinking: true)).to include(temperature: 0.6, top_p: 0.95, max_tokens: 2048)
    end

    it "lets the environment lift the output cap" do
      expect(described_class.options({ "ELELEM_CANDLE_MAX_TOKENS" => "0" }, thinking: true)).to include(max_tokens: 0)
    end

    it "picks a random seed by default" do
      expect(described_class.options({})[:seed]).to be_between(0, (2**63) - 1)
    end
  end

  describe ".build" do
    it "loads each model once" do
      allow(Elelem::Candle::Model).to receive(:load).and_return(model)
      2.times { described_class.build(path: "/models/once.gguf", env: {}) }
      expect(Elelem::Candle::Model).to have_received(:load).with("/models/once.gguf", 8192).once
    end

    it "samples for thinking when thinking is on" do
      allow(Elelem::Candle::Model).to receive(:load).and_return(model)
      described_class.build(path: "/models/think.gguf", env: { "ELELEM_CANDLE_THINKING" => "1" }).fetch([])
      expect(model.options).to include(temperature: 0.6, top_p: 0.95)
    end
  end
end
