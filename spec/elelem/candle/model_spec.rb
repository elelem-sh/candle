# frozen_string_literal: true

RSpec.describe Elelem::Candle::Model, if: ENV.fetch("ELELEM_CANDLE_TEST_MODEL", "") != "" do
  subject(:model) { described_class.load(ENV.fetch("ELELEM_CANDLE_TEST_MODEL"), 4096) }

  let(:messages) { [{ role: "user", content: "Reply with the word pong and nothing else." }] }
  let(:options) { { max_tokens: 32, temperature: 0.0, top_k: 20, top_p: 0.95, seed: 1, chunk: 8 } }

  it "renders the chat template" do
    prompt = model.render(JSON.generate(messages), "[]", false)
    expect(prompt).to end_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
  end

  it "replies" do
    text = +""
    stats = model.generate(model.render(JSON.generate(messages), "[]", false), options, proc { text << it })
    expect([text.downcase, stats[:stop_reason]]).to match([/pong/, "stop"])
  end

  it "raises Ruby errors from the block" do
    prompt = model.render(JSON.generate(messages), "[]", false)
    expect { model.generate(prompt, options, proc { raise ArgumentError, "stop" }) }.to raise_error(ArgumentError)
  end
end
