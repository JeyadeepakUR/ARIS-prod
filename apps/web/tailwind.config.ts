import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark design system
        bg: "#0a0b0f",
        "bg-2": "#111318",
        "bg-3": "#1a1d26",
        ink: "#e8eaf0",
        "ink-2": "#8b8fa8",
        "ink-3": "#555870",
        accent: "#6366f1",
        "accent-2": "#818cf8",
        bridge: "#f97316",
        success: "#10b981",
        warning: "#f59e0b",
        danger: "#ef4444",
        // Legacy aliases so old code doesn't break immediately
        surface: "#1a1d26",
        sand: "#1a1d26",
        spice: "#818cf8",
        ember: "#f97316",
        pine: "#10b981",
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
      },
      animation: {
        "spin-slow": "spin 3s linear infinite",
        "fade-up": "fade-up 0.3s ease both",
      },
      boxShadow: {
        glow: "0 0 24px rgba(99,102,241,0.3)",
        "glow-bridge": "0 0 24px rgba(249,115,22,0.3)",
        card: "0 1px 3px rgba(0,0,0,0.5), 0 8px 24px rgba(0,0,0,0.3)",
      },
    },
  },
  plugins: [],
};

export default config;