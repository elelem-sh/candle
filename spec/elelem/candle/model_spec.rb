# frozen_string_literal: true

RSpec.describe Elelem::Candle::Model, if: ENV.fetch("ELELEM_CANDLE_TEST_MODEL", "") != "" do
  def self.loaded = @loaded ||= described_class.load(ENV.fetch("ELELEM_CANDLE_TEST_MODEL"), 4096)

  let(:model) { self.class.loaded.tap(&:reset) }

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

  describe "prompt cache" do
    def run(prompt, opts = options)
      text = +""
      [model.generate(prompt, opts, proc { text << it }), text]
    end

    let(:prompt) { model.render(JSON.generate(messages), "[]", false) }

    it "reuses all but the last token of a repeated prompt" do
      first = run(prompt)
      second = run(prompt)
      expect([second[0][:n_reused], second[1]]).to eq([first[0][:n_prompt] - 1, first[1]])
    end

    it "reuses the earlier turn when the conversation grows" do
      stats, reply = run(prompt)
      longer = messages + [{ role: "assistant", content: reply }, { role: "user", content: "Again." }]
      expect(run(model.render(JSON.generate(longer), "[]", false))[0][:n_reused]).to be >= stats[:n_prompt] - 8
    end

    it "matches a fresh run when an earlier message changes" do
      run(prompt)
      changed = model.render(JSON.generate([{ role: "user", content: "Reply with the word ping and nothing else." }]),
                             "[]", false)
      cached = run(changed)[1]
      model.reset
      expect(run(changed)[1]).to eq(cached)
    end

    it "matches a fresh run after an interrupted prefill" do
      long = model.render(JSON.generate([{ role: "user", content: "#{"word " * 400}\nReply with pong." }]), "[]", false)
      worker = Thread.new do
        model.generate(long, options.merge(chunk: 8), proc {})
      rescue Interrupt
        :interrupted
      end
      sleep 0.5
      worker.raise(Interrupt)
      expect(worker.value).to eq(:interrupted)
      partial = run(prompt)[1]
      model.reset
      expect(run(prompt)[1]).to eq(partial)
    end

    it "stops reading a long chunk within a second of an interrupt" do
      long = model.render(JSON.generate([{ role: "user", content: "#{"word " * 900}\nReply with pong." }]), "[]", false)
      worker = Thread.new do
        model.generate(long, options.merge(chunk: 1024), proc {})
      rescue Interrupt
        :interrupted
      end
      sleep 0.5
      worker.raise(Interrupt)
      expect(worker.join(1)&.value).to eq(:interrupted)
    end

    it "matches a fresh run after the block raises" do
      expect { model.generate(prompt, options, proc { raise ArgumentError }) }.to raise_error(ArgumentError)
      partial = run(prompt)[1]
      model.reset
      expect(run(prompt)[1]).to eq(partial)
    end
  end

  it "stops on a caller's stop token" do
    prompt = model.render(JSON.generate(messages), "[]", true)
    text = +""
    stats = model.generate(prompt, options.merge(max_tokens: 400, stop: ["</think>"]), proc { text << it })
    expect([text, stats[:stop_token]]).to match([satisfy { !it.include?("</think>") }, "</think>"])
  end

  it "rejects a stop that is not one token" do
    prompt = model.render(JSON.generate(messages), "[]", false)
    expect { model.generate(prompt, options.merge(stop: ["hello world"]), proc {}) }
      .to raise_error(RuntimeError, /not a single token/)
  end

  it "stops decoding when interrupted" do
    prompt = model.render(JSON.generate([{ role: "user", content: "Count from 1 to 500." }]), "[]", false)
    ready = Queue.new
    worker = Thread.new do
      model.generate(prompt, options.merge(max_tokens: 4000), proc { ready << true })
    rescue Interrupt
      :interrupted
    end
    ready.pop
    worker.raise(Interrupt)
    expect(worker.join(2)&.value).to eq(:interrupted)
  end

  it "raises Ruby errors from the block" do
    prompt = model.render(JSON.generate(messages), "[]", false)
    expect { model.generate(prompt, options, proc { raise ArgumentError, "stop" }) }.to raise_error(ArgumentError)
  end

  it "replays a tool call so the next prompt extends the last one" do
    recorder = Class.new(SimpleDelegator) do
      attr_reader :prompts, :outputs

      def render(*)
        super.tap { (@prompts ||= []) << it }
      end

      def generate(prompt, options, on_piece)
        text = +""
        super(prompt, options, proc { |piece|
          text << piece
          on_piece.call(piece)
        }).tap { (@outputs ||= []) << text }
      end
    end.new(model)
    tools = [{ type: "function", function: { name: "read", description: "Read a file",
                                             parameters: { type: "object", properties: { path: { type: "string" } },
                                                           required: ["path"] } } }]
    provider = Elelem::Candle::Provider.new(model: recorder, thinking: true,
                                            options: options.merge(max_tokens: 600))
    history = [{ role: "user", content: "Use the read tool to read notes.txt." }]
    said = []
    calls = provider.fetch(history, tools) { said << it[:text] if it[:type] == "saying" }
    skip "the model did not call a tool" if calls.empty?

    history += [{ role: "assistant", content: said.join, tool_calls: calls },
                *calls.map { { role: "tool", tool_call_id: it[:id], content: "{\"content\":\"hi\"}" } }]
    provider.fetch(history, tools)
    expect(recorder.prompts[1]).to start_with("#{recorder.prompts[0]}#{recorder.outputs[0]}<|im_end|>\n")
  end
end
