# frozen_string_literal: true

module Elelem
  module Candle
    module Native
      NEEDS = %w[avx2 fma f16c bmi1 bmi2 movbe].freeze

      def self.load
        check_cpu if prebuilt?
        require_relative "candle"
      end

      def self.prebuilt?
        spec = Gem.loaded_specs["elelem-candle"]
        !spec.nil? && spec.platform.to_s != Gem::Platform::RUBY
      end

      def self.check_cpu(flags = cpu_flags)
        missing = NEEDS - flags
        return if missing.empty?

        raise LoadError, "elelem-candle: this CPU lacks #{missing.join(", ")}; " \
                         "run gem install elelem-candle --platform ruby"
      end

      def self.cpu_flags
        return NEEDS unless RbConfig::CONFIG["host_cpu"].match?(/x86_64|x64/)

        case RbConfig::CONFIG["host_os"]
        when /linux/ then File.read("/proc/cpuinfo")[/^flags\s*:(.*)$/, 1].to_s.split
        when /darwin/ then `sysctl -n machdep.cpu.features machdep.cpu.leaf7_features`.downcase.split
        else NEEDS
        end
      end
    end
  end
end
