/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['localhost', 'via.placeholder.com'],
  },
  
  // Docker 环境配置
  output: 'standalone',
}

module.exports = nextConfig