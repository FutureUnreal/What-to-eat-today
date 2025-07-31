'use client'

import React, { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Mic, MicOff, Paperclip, Smile } from 'lucide-react'
import { Button } from '@/components/ui'

interface ChatInputProps {
  onSendMessage: (message: string) => void
  onVoiceStart?: () => void
  onVoiceStop?: () => void
  disabled?: boolean
  placeholder?: string
  maxLength?: number
  className?: string
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  onVoiceStart,
  onVoiceStop,
  disabled = false,
  placeholder = '请输入您的问题...',
  maxLength = 1000,
  className = ''
}) => {
  const [message, setMessage] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [isFocused, setIsFocused] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  
  // 自动调整textarea高度
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [message])
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    if (message.trim() && !disabled) {
      onSendMessage(message.trim())
      setMessage('')
      
      // 重置textarea高度
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }
  
  const handleVoiceToggle = () => {
    if (isRecording) {
      setIsRecording(false)
      onVoiceStop?.()
    } else {
      setIsRecording(true)
      onVoiceStart?.()
    }
  }
  
  const canSend = message.trim().length > 0 && !disabled
  
  return (
    <div className={`w-full max-w-4xl mx-auto px-4 ${className}`}>
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className={`glass rounded-2xl border transition-all duration-200 ${
          isFocused ? 'border-blue-400 shadow-lg' : 'border-white/20'
        }`}
      >
        <form onSubmit={handleSubmit} className="flex items-end space-x-3 p-4">
          {/* 附件按钮 */}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="flex-shrink-0 text-gray-500 hover:text-gray-700"
            disabled={disabled}
          >
            <Paperclip className="w-5 h-5" />
          </Button>
          
          {/* 输入区域 */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder={placeholder}
              disabled={disabled}
              maxLength={maxLength}
              rows={1}
              className="w-full resize-none bg-transparent border-none outline-none placeholder-gray-500 text-gray-900 min-h-[24px] max-h-32 overflow-y-auto custom-scrollbar"
            />
            
            {/* 字符计数 */}
            <div className="absolute bottom-0 right-2 text-xs text-gray-400">
              {message.length}/{maxLength}
            </div>
          </div>
          
          {/* 表情按钮 */}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="flex-shrink-0 text-gray-500 hover:text-gray-700"
            disabled={disabled}
          >
            <Smile className="w-5 h-5" />
          </Button>
          
          {/* 语音按钮 */}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={handleVoiceToggle}
            className={`flex-shrink-0 transition-colors ${
              isRecording 
                ? 'text-red-500 hover:text-red-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
            disabled={disabled}
          >
            {isRecording ? (
              <MicOff className="w-5 h-5" />
            ) : (
              <Mic className="w-5 h-5" />
            )}
          </Button>
          
          {/* 发送按钮 */}
          <motion.div
            whileHover={{ scale: canSend ? 1.05 : 1 }}
            whileTap={{ scale: canSend ? 0.95 : 1 }}
          >
            <Button
              type="submit"
              variant="primary"
              size="sm"
              disabled={!canSend}
              className="flex-shrink-0"
            >
              <Send className="w-4 h-4" />
            </Button>
          </motion.div>
        </form>
        
        {/* 语音录制指示器 */}
        {isRecording && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="px-4 pb-4"
          >
            <div className="flex items-center justify-center space-x-2 p-3 bg-red-50/50 rounded-lg">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="text-sm text-red-700">正在录音...</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleVoiceToggle}
                className="text-red-600 hover:text-red-700"
              >
                停止录音
              </Button>
            </div>
          </motion.div>
        )}
      </motion.div>
      
      {/* 快捷提示 */}
      <div className="flex items-center justify-center mt-3 text-xs text-gray-500">
        <span>按 Enter 发送，Shift + Enter 换行</span>
      </div>
    </div>
  )
}

export default ChatInput