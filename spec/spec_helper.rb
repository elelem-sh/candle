# frozen_string_literal: true

require "delegate"
require "elelem/candle"
require_relative "support/fake_model"

RSpec.configure do |config|
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end
end
