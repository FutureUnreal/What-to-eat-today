/** @type {import('next').NextConfig} */
const nextConfig = {
<<<<<<< HEAD
  experimental: {
    appDir: true,
  },
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*', // 代理到Python后端
      },
    ]
  },
=======
  images: {
    domains: ['localhost', 'via.placeholder.com'],
  },
  
  // Docker 环境配置
  output: 'standalone',
>>>>>>> 3231226 (Reinitialize Git repository)
}

module.exports = nextConfig