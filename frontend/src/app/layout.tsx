import type { Metadata, Viewport } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: '今天吃什么 - AI美食推荐助手',
  description: '基于图RAG技术的智能烹饪助手，为您推荐个性化美食和详细烹饪指导',
  keywords: '美食推荐,烹饪助手,AI助手,菜谱,食材',
  authors: [{ name: '今天吃什么团队' }],
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#0ea5e9',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh-CN">
      <body className="antialiased">
        <div id="root">{children}</div>
        <div id="modal-root" />
        <div id="toast-root" />
      </body>
    </html>
  )
}