# frozen_string_literal: true

RSpec.describe "Elelem::Candle.render_template" do
  root = File.expand_path("../../fixtures/template", __dir__)
  source = File.read(File.join(root, "qwen3.jinja"), encoding: "UTF-8")

  JSON.parse(File.read(File.join(root, "cases.json"), encoding: "UTF-8")).each do |name, input|
    it "renders #{name} like Hugging Face" do
      rendered = Elelem::Candle.render_template(
        source, JSON.generate(input["messages"]), JSON.generate(input["tools"]), input["thinking"]
      )
      expect(rendered).to eq(File.read(File.join(root, "expected", "#{name}.txt"), encoding: "UTF-8"))
    end
  end
end
