# frozen_string_literal: true

RSpec.describe Elelem::Candle::Native do
  it "accepts a CPU with avx2, fma and f16c" do
    expect { described_class.check_cpu(%w[sse4_2 avx2 fma f16c]) }.not_to raise_error
  end

  it "names what an older CPU lacks and how to build from source" do
    expect { described_class.check_cpu(%w[sse4_2 avx]) }
      .to raise_error(LoadError, /lacks avx2, fma, f16c; run gem install elelem-candle --platform ruby/)
  end

  it "finds this CPU's flags" do
    expect(described_class.cpu_flags).to include("avx2")
  end
end
