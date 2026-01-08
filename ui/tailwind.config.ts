import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        primary: {
          50: "#f0f7ff",
          100: "#e0effe",
          200: "#bae0fd",
          300: "#7cc7fb",
          400: "#36a9f7",
          500: "#0c8ce9",
          600: "#0070c7",
          700: "#0159a1",
          800: "#064b85",
          900: "#0a406e",
          950: "#072849",
        },
        secondary: {
          50: "#f4f7fb",
          100: "#e9eff5",
          200: "#cedde9",
          300: "#a3c1d6",
          400: "#729fbe",
          500: "#5083a6",
          600: "#3d698b",
          700: "#335671",
          800: "#2d495e",
          900: "#293f50",
          950: "#1b2935",
        },
        accent: {
          50: "#f2fbf4",
          100: "#e0f8e6",
          200: "#c3f0cf",
          300: "#94e3a9",
          400: "#5dce7c",
          500: "#38b35a",
          600: "#289446",
          700: "#23753a",
          800: "#215d32",
          900: "#1d4d2b",
          950: "#0b2a16",
        },
        success: {
          light: "#4ade80",
          DEFAULT: "#22c55e",
          dark: "#16a34a",
        },
        warning: {
          light: "#fbbf24",
          DEFAULT: "#f59e0b",
          dark: "#d97706",
        },
        error: {
          light: "#f87171",
          DEFAULT: "#ef4444",
          dark: "#dc2626",
        },
        background: {
          primary: "#0f172a",
          secondary: "#1e293b",
          tertiary: "#334155",
        },
        surface: {
          primary: "#1e293b",
          secondary: "#334155",
          tertiary: "#475569",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Menlo", "Monaco", "Courier New", "monospace"],
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-in-out",
        "slide-up": "slideUp 0.3s ease-out",
        "pulse-slow": "pulse 3s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { transform: "translateY(10px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};

export default config;