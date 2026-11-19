# frozen_string_literal: true

RSpec.describe Elelem::Candle::Provider do
  let(:model) { FakeModel.new(pieces) }
  let(:pieces) { %w[Hel lo] }
  let(:messages) { [{ role: "system", content: "terse" }, { role: "user", content: "hi" }] }

  let(:tools) { [{ type: "function", function: { name: "read", description: "Read", parameters: {} } }] }
  let(:provider) { described_class.new(model:) }

  def fetch(provider = self.provider, messages: self.messages, tools: [])
    events = []
    result = provider.fetch(messages, tools) { |event| events << event }
    [events, result]
  end

  def call(body) = "<tool_call>\n#{body}\n</tool_call>"

  it "renders the conversation as JSON without tools" do
    fetch
    expect(model.rendered).to eq([JSON.generate(messages), "[]", false])
  end

  it "renders the tools" do
    fetch(tools:)
    expect(JSON.parse(model.rendered[1])).to eq(JSON.parse(JSON.generate(tools)))
  end

  it "passes thinking to the template" do
    fetch(described_class.new(model:, thinking: true))
    expect(model.rendered.last).to be(true)
  end

  it "says the reply once without streaming it" do
    events, = fetch
    expect(events).to eq([{ type: "saying", text: "Hello" }])
  end

  it "stops on a new turn" do
    fetch
    expect(model.options[:stop]).to eq(["<|im_start|>"])
  end

  it "also stops on a made up tool response when tools are offered" do
    fetch(tools:)
    expect(model.options[:stop]).to eq(["<|im_start|>", "<tool_response>"])
  end

  it "returns no tool calls for a plain reply" do
    _, result = fetch
    expect(result).to eq([])
  end

  context "when the model thinks first" do
    let(:pieces) { ["<think>\nhm", "m\n</thi", "nk>\n\n", "pong"] }

    it "streams the thinking and says the reply" do
      events, = fetch(described_class.new(model:, thinking: true))
      expect(events).to eq([
                             { type: "thinking", text: "hm" },
                             { type: "thinking", text: "m" },
                             { type: "saying", text: "pong" }
                           ])
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

  context "when the model calls a tool" do
    let(:pieces) do
      ["Let me look.\n<tool", "_call>\n{\"name\": \"read\", ", "\"arguments\": {\"path\": \"a\"}}\n</tool_call>"]
    end

    it "says the text and then does the call" do
      events, result = fetch(tools:)
      read = { name: "read", arguments: { "path" => "a" } }
      expect(events.map { it.except(:id) }).to eq([{ type: "saying", text: "Let me look." }, read.merge(type: "doing")])
      expect(result.map { it.except(:id) }).to eq([read])
    end

    it "gives each call a new id" do
      ids = Array.new(2) { fetch(tools:).last.first[:id] }
      expect(ids.uniq.size).to eq(2)
    end

    it "replays the arguments exactly as the model wrote them" do
      _, result = fetch(tools:)
      id = result.first[:id]
      history = messages + [
        { role: "assistant", content: "Let me look.",
          tool_calls: [{ id:, name: "read", arguments: { "path" => "a" } }] },
        { role: "tool", tool_call_id: id, content: "{}" }
      ]
      fetch(messages: history, tools:)
      expect(JSON.parse(model.rendered.first)[2]["tool_calls"].first["arguments"]).to eq('{"path": "a"}')
    end
  end

  context "when the model thinks and then calls a tool" do
    let(:pieces) { ["<think>\nneed a\n</think>\n\n", call('{"name": "read", "arguments": {"path": "a"}}')] }

    it "replays the reasoning" do
      provider = described_class.new(model:, thinking: true)
      _, result = fetch(provider, tools:)
      history = messages + [{ role: "assistant", content: "", tool_calls: result }]
      fetch(provider, messages: history, tools:)
      expect(JSON.parse(model.rendered.first)[2]["reasoning_content"]).to eq("need a")
    end
  end

  context "when the model makes two calls at once" do
    let(:pieces) do
      [call('{"name": "read", "arguments": {"path": "a"}}'), "\n",
       call('{"name": "write", "arguments": {"path": "b", "content": "<content>"}}')]
    end

    it "runs only the first so the next step sees its result" do
      events, result = fetch(tools:)
      expect([events.map { it.except(:id) }, result.map { it[:name] }])
        .to eq([[{ type: "doing", name: "read", arguments: { "path" => "a" } }], ["read"]])
    end
  end

  context "when a call is cut off" do
    let(:pieces) { [call('{"name": "read", "arguments": {"path": "a"}}')] }

    it "reports it and does nothing" do
      model.stop_reason = "length"
      events, result = fetch(tools:)
      expect([events, result]).to eq([[{ type: "saying", text: Elelem::Candle::Reply::TRUNCATED }], []])
    end
  end

  context "when the arguments are not JSON" do
    let(:pieces) { [call('{"name": "read", "arguments": {"path": a}}')] }

    it "passes the raw text on so the toolbox reports it to the model" do
      _, result = fetch(tools:)
      expect(result.first.slice(:name, :arguments))
        .to eq(name: "read", arguments: '{"name": "read", "arguments": {"path": a}}')
    end
  end

  describe "context budget" do
    let(:result) { "x" * 5_000 }
    let(:messages) do
      [{ role: "user", content: "read it" },
       { role: "assistant", content: "", tool_calls: [{ id: "c1", name: "read", arguments: {} }] },
       { role: "tool", tool_call_id: "c1", content: result }]
    end

    it "sends a prompt that fits as it is" do
      fetch
      expect(model.prompt).to include(result)
    end

    it "cuts the middle of the newest tool result to fit" do
      model.context = 1_000
      fetch
      expect([model.count(model.prompt) <= 1_000 - 256, model.prompt]).to match([true, /characters removed/])
    end

    it "says the context is full when cutting is not enough" do
      model.context = 300
      events, result = fetch
      expect([events.first[:text], result]).to match([/\A\[context full: the prompt is \d+ tokens, limit 44/, []])
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
