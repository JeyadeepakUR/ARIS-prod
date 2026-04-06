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
        surface: "#f2efe8",
        ink: "#1f1d17",
        sand: "#dfd6c5",
        spice: "#8f4f2b",
        ember: "#d1682f",
        pine: "#2d5a4d",
      },
    },
  },
  plugins: [],
};

export default config;