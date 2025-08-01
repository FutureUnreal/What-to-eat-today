'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { User, Bot, Copy, ThumbsUp, ThumbsDown } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { ChatMessage as ChatMessageType } from '@/types'
import { Button } from '@/components/ui'

interface ChatMessageProps {
  message: ChatMessageType
  onCopy?: (content: string) => void
  onFeedback?: (messageId: string, type: 'like' | 'dislike') => void
  isStreaming?: boolean
  className?: string
}

const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  onCopy,
  onFeedback,
  isStreaming = false,
  className = ''
}) => {
  const isUser = message.role === 'user'
  const isAssistant = message.role === 'assistant'
  
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      onCopy?.(message.content)
    } catch (error) {
      console.error('Failed to copy message:', error)
    }
  }
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex items-start space-x-4 max-w-4xl mx-auto px-4 py-6 ${className}`}
    >
      {/* 头像 */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        isUser 
          ? 'bg-blue-500 text-white' 
          : 'bg-gradient-to-br from-purple-500 to-pink-500 text-white'
      }`}>
        {isUser ? (
          <User className="w-4 h-4" />
        ) : (
          <Bot className="w-4 h-4" />
        )}
      </div>
      
      {/* 消息内容 */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center space-x-2 mb-2">
          <span className="text-sm font-medium text-gray-900">
            {isUser ? '您' : 'AI助手'}
          </span>
          <span className="text-xs text-gray-500">
            {(() => {
              try {
                const date = message.timestamp instanceof Date
                  ? message.timestamp
                  : new Date(message.timestamp);
                return date.toLocaleTimeString('zh-CN', {
                  hour: '2-digit',
                  minute: '2-digit'
                });
              } catch (error) {
                return '刚刚';
              }
            })()}
          </span>
        </div>
        
        <div className={`chat-message ${isUser ? 'user' : 'assistant'} ${
          isStreaming ? 'animate-pulse' : ''
        }`}>
          {isUser ? (
            <p className="text-white">{message.content}</p>
          ) : (
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown
                components={{
                  // 自定义渲染组件
                  p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
                  ul: ({ children }) => <ul className="list-disc pl-6 mb-3">{children}</ul>,
                  ol: ({ children }) => <ol className="list-decimal pl-6 mb-3">{children}</ol>,
                  li: ({ children }) => <li className="mb-1">{children}</li>,
                  h1: ({ children }) => <h1 className="text-xl font-bold mb-3">{children}</h1>,
                  h2: ({ children }) => <h2 className="text-lg font-semibold mb-2">{children}</h2>,
                  h3: ({ children }) => <h3 className="text-base font-medium mb-2">{children}</h3>,
                  code: ({ children, className }) => {
                    const isInline = !className
                    return isInline ? (
                      <code className="bg-gray-100 px-1 py-0.5 rounded text-sm">{children}</code>
                    ) : (
                      <pre className="bg-gray-100 p-3 rounded-lg overflow-x-auto">
                        <code>{children}</code>
                      </pre>
                    )
                  },
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-blue-500 pl-4 italic my-3">
                      {children}
                    </blockquote>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
          
          {/* 流式输入指示器 */}
          {isStreaming && !isUser && (
            <div className="flex items-center space-x-1 mt-2">
              <div className="loading-dots">
                <div></div>
                <div></div>
                <div></div>
              </div>
              <span className="text-xs text-gray-500 ml-2">AI正在思考...</span>
            </div>
          )}
        </div>
        
        {/* 操作按钮 */}
        {isAssistant && !isStreaming && (
          <div className="flex items-center space-x-2 mt-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="text-gray-500 hover:text-gray-700"
            >
              <Copy className="w-4 h-4" />
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onFeedback?.(message.id, 'like')}
              className="text-gray-500 hover:text-green-600"
            >
              <ThumbsUp className="w-4 h-4" />
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onFeedback?.(message.id, 'dislike')}
              className="text-gray-500 hover:text-red-600"
            >
              <ThumbsDown className="w-4 h-4" />
            </Button>
          </div>
        )}
        
        {/* 相关菜谱推荐 */}
        {message.metadata?.recipes && message.metadata.recipes.length > 0 && (
          <div className="mt-4 p-4 bg-blue-50/50 rounded-lg">
            <h4 className="text-sm font-medium text-blue-900 mb-2">
              🍽️ 相关菜谱推荐
            </h4>
            <div className="space-y-2">
              {message.metadata.recipes.slice(0, 3).map((recipe: any, index: number) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-sm text-blue-800">{recipe.name}</span>
                  <Button variant="ghost" size="sm" className="text-blue-600">
                    查看详情
                  </Button>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* 建议问题 */}
        {message.metadata?.suggestions && message.metadata.suggestions.length > 0 && (
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">
              💡 您可能还想问：
            </h4>
            <div className="flex flex-wrap gap-2">
              {message.metadata.suggestions.map((suggestion: string, index: number) => (
                <Button
                  key={index}
                  variant="glass"
                  size="sm"
                  className="text-sm"
                  onClick={() => {
                    // 这里可以触发新的问题
                    console.log('Suggested question:', suggestion)
                  }}
                >
                  {suggestion}
                </Button>
              ))}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  )
}

export default ChatMessage