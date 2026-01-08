/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',

  // Transpile packages that need it
  transpilePackages: [
    'three',
    '@react-three/fiber',
    '@react-three/drei',
  ],

  // Configure webpack for KiCad WASM integration
  webpack: (config, { isServer }) => {
    // Handle WebAssembly modules
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };

    // Ignore fs module in browser
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }

    return config;
  },

  // Environment variables exposed to client
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9080',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:9080',
  },

  // Image optimization
  images: {
    domains: ['localhost', 'api.adverant.ai'],
    unoptimized: process.env.NODE_ENV === 'development',
  },
};

module.exports = nextConfig;