# frozen_string_literal: true

RSpec.describe Elelem::Candle::Reply do
  let(:tools) { [{ type: "function", function: { name: "read" } }, { type: "function", function: { name: "write" } }] }

  def parse(text, thinking: false, stop_reason: "stop", tools: self.tools)
    described_class.parse(text, thinking:, stop_reason:, tools:)
  end

  describe "thinking" do
    [
      ["no think block", "Hello.", false, "", "Hello."],
      ["a closed think block", "<think>\nplan\n</think>\n\nHello.", true, "plan", "Hello."],
      ["an empty think block", "<think>\n\n</think>\n\nHello.", true, "", "Hello."],
      ["an unclosed think block", "<think>\nstill going", true, "still going", ""],
      ["a bare close when thinking", "plan\n</think>\n\nHello.", true, "plan", "Hello."],
      ["a bare close when not thinking", "a </think> b", false, "", "a </think> b"],
      ["leading newlines", "\n\nHello.", false, "", "Hello."]
    ].each do |name, text, thinking, reasoning, content|
      it "splits #{name}" do
        reply = parse(text, thinking:)
        expect([reply.reasoning, reply.content]).to eq([reasoning, content])
      end
    end
  end

  describe "tool calls" do
    call = ->(body) { "<tool_call>\n#{body}\n</tool_call>" }

    [
      ["a closed call", call['{"name": "read", "arguments": {"path": "a.txt"}}'],
       [["read", { "path" => "a.txt" }, '{"path": "a.txt"}']]],
      ["string arguments", call['{"name": "read", "arguments": "{\\"path\\": \\"a.txt\\"}"}'],
       [["read", { "path" => "a.txt" }, '"{\\"path\\": \\"a.txt\\"}"']]],
      ["parameters", call['{"name": "read", "parameters": {"path": "a.txt"}}'],
       [["read", { "path" => "a.txt" }, '{"path": "a.txt"}']]],
      ["missing arguments", call['{"name": "read"}'], [["read", {}, "{}"]]],
      ["array arguments", call['{"name": "read", "arguments": ["a.txt"]}'], [["read", ["a.txt"], '["a.txt"]']]],
      ["bad json", call['{"name": "read", "arguments": {"path": a.txt}}'],
       [["read", '{"name": "read", "arguments": {"path": a.txt}}', '{"name": "read", "arguments": {"path": a.txt}}']]],
      ["two calls", "#{call['{"name": "read", "arguments": {"path": "a"}}']}\n" \
                    "#{call['{"name": "read", "arguments": {"path": "b"}}']}",
       [["read", { "path" => "a" }, '{"path": "a"}'], ["read", { "path" => "b" }, '{"path": "b"}']]],
      ["a duplicate", call['{"name": "read", "arguments": {"path": "a"}}'] * 2,
       [["read", { "path" => "a" }, '{"path": "a"}']]],
      ["an unknown tool", call['{"name": "nope", "arguments": {}}'], [["nope", {}, "{}"]]]
    ].each do |name, text, expected|
      it "reads #{name}" do
        reply = parse(text)
        expect([reply.calls.map { [it.name, it.arguments, it.raw] }, reply.errors]).to eq([expected, []])
      end
    end

    it "keeps the text before the calls as content" do
      reply = parse("Let me look.\n#{call['{"name": "read", "arguments": {"path": "a"}}']}")
      expect(reply.content).to eq("Let me look.")
    end

    it "drops text after the calls" do
      reply = parse("#{call['{"name": "read", "arguments": {"path": "a"}}']}\n<tool_response>\nmade up")
      expect([reply.content, reply.calls.size]).to eq(["", 1])
    end

    it "ignores calls inside the think block" do
      reply = parse("<think>\nI could use <tool_call> here\n</think>\n\nNo tools needed.", thinking: true)
      expect([reply.calls, reply.content]).to eq([[], "No tools needed."])
    end

    it "refuses an unclosed call" do
      reply = parse('<tool_call>\n{"name": "write", "arguments": {"path": "a", "content": "par')
      expect([reply.calls, reply.errors]).to eq([[], [described_class::TRUNCATED]])
    end

    it "refuses closed calls when the reply ran out of tokens" do
      reply = parse(call['{"name": "read", "arguments": {"path": "a"}}'], stop_reason: "length")
      expect([reply.calls, reply.errors]).to eq([[], [described_class::TRUNCATED]])
    end

    it "refuses a closed call followed by an unclosed one" do
      reply = parse("#{call['{"name": "read", "arguments": {"path": "a"}}']}\n<tool_call>\n{\"name\"")
      expect([reply.calls, reply.errors]).to eq([[], [described_class::TRUNCATED]])
    end
  end

  describe "bare JSON" do
    [
      ["a bare call", '{"name": "read", "arguments": {"path": "a"}}', 1],
      ["a fenced call", "```json\n{\"name\": \"read\", \"arguments\": {\"path\": \"a\"}}\n```", 1],
      ["prose around it", 'Try {"name": "read", "arguments": {"path": "a"}} next.', 0],
      ["an unknown name", '{"name": "nope", "arguments": {}}', 0],
      ["two objects", '{"name": "read", "arguments": {}} {"name": "read", "arguments": {}}', 0],
      ["no arguments key", '{"name": "read"}', 0]
    ].each do |name, text, count|
      it "handles #{name}" do
        expect(parse(text).calls.size).to eq(count)
      end
    end

    it "needs tools to be offered" do
      expect(parse('{"name": "read", "arguments": {}}', tools: []).calls).to be_empty
    end

    it "needs a normal stop" do
      expect(parse('{"name": "read", "arguments": {}}', stop_reason: "length").calls).to be_empty
    end

    it "clears the content" do
      expect(parse('{"name": "read", "arguments": {"path": "a"}}').content).to eq("")
    end
  end
end
