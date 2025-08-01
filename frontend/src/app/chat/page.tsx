'use client'

import React, { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import { ChefHat, Plus, Sidebar, X, MessageSquare, Trash2 } from 'lucide-react'
import { Button, LoadingSpinner } from '@/components/ui'
import { ChatMessage, ChatInput } from '@/components/chat'
import { useAppStore } from '@/store'
import { useChat } from '@/hooks'

const ChatPage: React.FC = () => {
  const searchParams = useSearchParams()
  const sessionId = searchParams.get('session')
  const initialQuestion = searchParams.get('q')
  
  const {
    chat,
    createChatSession,
    switchChatSession,
    deleteChatSession,
    setSidebarOpen,
    ui
  } = useAppStore()
  
  const {
    currentSession,
    sessions,
    isLoading,
    isStreaming,
    sendMessage,
    stopGeneration,
    copyMessage,
    provideFeedback
  } = useChat()
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [hasInitialized, setHasInitialized] = useState(false)
  
  // 滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }
  
  useEffect(() => {
    scrollToBottom()
  }, [currentSession?.messages])
  
  // 初始化会话
  useEffect(() => {
    if (hasInitialized) return
    
    if (sessionId) {
      // 切换到指定会话
      switchChatSession(sessionId)
    } else if (sessions.length === 0) {
      // 创建新会话
      const newSessionId = createChatSession()
      window.history.replaceState(null, '', `/chat?session=${newSessionId}`)
    } else {
      // 使用最新会话
      switchChatSession(sessions[0].id)
      window.history.replaceState(null, '', `/chat?session=${sessions[0].id}`)
    }
    
    setHasInitialized(true)
  }, [sessionId, sessions, switchChatSession, createChatSession, hasInitialized])
  
  // 发送初始问题
  useEffect(() => {
    if (initialQuestion && currentSession && currentSession.messages.length === 0) {
      sendMessage(initialQuestion)
      // 清除URL中的问题参数
      window.history.replaceState(null, '', `/chat?session=${currentSession.id}`)
    }
  }, [initialQuestion, currentSession, sendMessage])
  
  const handleCreateNewChat = () => {
    const newSessionId = createChatSession()
    window.location.href = `/chat?session=${newSessionId}`
  }
  
  const handleDeleteSession = (sessionIdToDelete: string) => {
    if (sessions.length <= 1) return // 至少保留一个会话
    
    deleteChatSession(sessionIdToDelete)
    
    if (currentSession?.id === sessionIdToDelete) {
      const remainingSessions = sessions.filter(s => s.id !== sessionIdToDelete)
      if (remainingSessions.length > 0) {
        switchChatSession(remainingSessions[0].id)
        window.history.replaceState(null, '', `/chat?session=${remainingSessions[0].id}`)
      }
    }
  }
  
  return (
    <div className="h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* 侧边栏 */}
      <AnimatePresence>
        {ui.sidebarOpen && (
          <>
            {/* 全屏遮罩 */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 z-40"
              onClick={() => setSidebarOpen(false)}
            />

            {/* 侧边栏内容 - 完全 fixed 定位，不影响主布局 */}
            <motion.aside
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{
                type: "tween",
                duration: 0.15,
                ease: "easeOut"
              }}
              className="fixed left-0 top-0 h-full w-80 glass border-r border-white/20 z-50 flex flex-col"
            >
              {/* 侧边栏头部 - 集成导航功能 */}
              <div className="p-4 border-b border-white/10">
                {/* 主标题和返回首页 */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-2">
                    <ChefHat className="w-6 h-6 text-blue-500" />
                    <h1 className="text-lg font-semibold gradient-text">今天吃什么</h1>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => window.location.href = '/'}
                      className="glass-button text-xs"
                    >
                      返回首页
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSidebarOpen(false)}
                      className="lg:hidden"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </div>

                {/* 对话历史标题和新对话按钮 */}
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-sm font-medium text-gray-600">对话历史</h2>
                </div>

                <Button
                  variant="primary"
                  size="sm"
                  onClick={handleCreateNewChat}
                  className="w-full"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  新对话
                </Button>
              </div>
              
              {/* 会话列表 */}
              <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
                <div className="space-y-2">
                  {sessions.map((session) => (
                    <div
                      key={session.id}
                      className={`group relative p-3 rounded-lg cursor-pointer transition-all duration-200 ${
                        currentSession?.id === session.id
                          ? 'bg-blue-100/50 border border-blue-200'
                          : 'hover:bg-white/20'
                      }`}
                      onClick={() => {
                        switchChatSession(session.id)
                        window.history.replaceState(null, '', `/chat?session=${session.id}`)
                        setSidebarOpen(false)
                      }}
                    >
                      <div className="flex items-start space-x-2">
                        <MessageSquare className="w-4 h-4 text-gray-500 mt-0.5 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <h3 className="text-sm font-medium text-gray-900 truncate">
                            {session.title}
                          </h3>
                          <p className="text-xs text-gray-500 mt-1">
                            {session.messages.length} 条消息
                          </p>
                          <p className="text-xs text-gray-400">
                            刚刚
                          </p>
                        </div>
                        
                        {sessions.length > 1 && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDeleteSession(session.id)
                            }}
                            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 h-auto"
                          >
                            <Trash2 className="w-3 h-3 text-red-500" />
                          </Button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
      
      {/* 主聊天区域 */}
      <div className="w-full h-full flex flex-col relative">
        {/* 浮动的侧边栏按钮 */}
        <div className="absolute top-4 left-4 z-10">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(!ui.sidebarOpen)}
            className="glass-button hover:bg-white/10 transition-all duration-150"
          >
            <Sidebar className="w-4 h-4" />
          </Button>
        </div>

        {/* 停止生成按钮（如果正在生成） */}
        {isStreaming && (
          <div className="absolute top-4 right-4 z-10">
            <Button
              variant="secondary"
              size="sm"
              onClick={stopGeneration}
            >
              停止生成
            </Button>
          </div>
        )}
        
        {/* 消息区域 */}
        <div className="flex-1 overflow-y-auto custom-scrollbar pt-16">
          {currentSession && currentSession.messages.length > 0 ? (
            <div className="py-6 px-4">
              {currentSession.messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  onCopy={copyMessage}
                  onFeedback={provideFeedback}
                  isStreaming={isStreaming && message.role === 'assistant' && message === currentSession.messages[currentSession.messages.length - 1]}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center px-4">
              <div className="text-center max-w-md mx-auto">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <ChefHat className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-xl font-semibold text-gray-900 mb-2">
                  开始新的对话
                </h2>
                <p className="text-gray-600 mb-6">
                  问问我今天吃什么，或者分享您的烹饪需求
                </p>
                
                <div className="space-y-2">
                  {[
                    '今天晚餐吃什么？',
                    '推荐一道简单的家常菜',
                    '我想学做甜品'
                  ].map((suggestion, index) => (
                    <Button
                      key={index}
                      variant="glass"
                      size="sm"
                      onClick={() => sendMessage(suggestion)}
                      className="w-full"
                    >
                      {suggestion}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* 输入区域 */}
        <div className="border-t border-white/10 p-4">
          <ChatInput
            onSendMessage={sendMessage}
            disabled={isLoading || isStreaming}
            placeholder={isStreaming ? 'AI正在回复中...' : '请输入您的问题...'}
          />
        </div>
      </div>
    </div>
  )
}

export default ChatPage