/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['three'],
  
  // Disable static optimization for client-only app
  trailingSlash: true,
  
  // Performance optimizations
  swcMinify: true,
  
  // Compression
  compress: true,
  
  // Enable standalone output for Docker
  output: 'standalone',
  
  // Headers for security and performance
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on'
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload'
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block'
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN'
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(self), geolocation=()'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin'
          }
        ]
      }
    ]
  },
  
  // Environment variables
  env: {
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  
  // Webpack configuration for audio processing
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Audio file handling
    config.module.rules.push({
      test: /\.(mp3|wav|ogg|m4a)$/,
      use: {
        loader: 'file-loader',
        options: {
          publicPath: '/_next/static/audio/',
          outputPath: 'static/audio/',
        },
      },
    });
    
    // WASM support for audio processing
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };
    
    // Opus codec support
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
    };
    
    return config;
  },
  
  // Image optimization
  images: {
    domains: ['localhost'],
    formats: ['image/webp', 'image/avif'],
  },
  
  // API routes timeout
  serverRuntimeConfig: {
    maxDuration: 30,
  },
  
  // Public runtime config
  publicRuntimeConfig: {
    wsUrl: process.env.NEXT_PUBLIC_WS_URL,
    apiUrl: process.env.NEXT_PUBLIC_API_URL,
  }
}

module.exports = nextConfig 