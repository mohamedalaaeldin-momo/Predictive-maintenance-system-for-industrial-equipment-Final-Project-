import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        panel: {
          DEFAULT: "#14171C",
          raised: "#1B1F26",
        },
        line: "#262B33",
        ink: {
          DEFAULT: "#E8EAED",
          muted: "#8B939F",
          faint: "#5B6270",
        },
        amber: {
          DEFAULT: "#F2A93B",
        },
        cyan: {
          DEFAULT: "#33C7D6",
        },
        ok: "#3ED598",
        danger: "#FF5C5C",
      },
      fontFamily: {
        display: ["var(--font-space-grotesk)", "sans-serif"],
        body: ["var(--font-inter)", "sans-serif"],
        mono: ["var(--font-jbmono)", "monospace"],
      },
      backgroundImage: {
        grid: "linear-gradient(to right, #1d2128 1px, transparent 1px), linear-gradient(to bottom, #1d2128 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
};

export default config;
