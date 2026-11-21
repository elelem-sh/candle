# frozen_string_literal: true

module Elelem
  module Candle
    module Native
      NEEDS = %w[avx2 fma f16c].freeze
      PREBUILT = "#{RUBY_VERSION[/\A\d+\.\d+/]}/candle".freeze

      def self.load
        return require_relative("candle") unless prebuilt?

        check_cpu
        require_relative PREBUILT
      end

      def self.prebuilt? = File.exist?(File.join(__dir__, "#{PREBUILT}.#{RbConfig::CONFIG["DLEXT"]}"))

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
