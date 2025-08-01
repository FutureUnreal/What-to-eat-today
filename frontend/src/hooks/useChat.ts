import { useCallback, useRef } from 'react'
import { useAppStore } from '@/store'
import { chatApi, apiUtils } from '@/lib/api'

export const useChat = () => {
  const {
    chat,
    addMessage,
    updateMessage,
    setChatLoading,
    setChatStreaming,
    createChatSession,
    updateSessionTitle,
    addToast
  } = useAppStore()
  
  const abortControllerRef = useRef<AbortController | null>(null)
  
  // 发送消息
  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return
    
    try {
      // 确保有活跃的会话
      let sessionId = chat.currentSession?.id
      if (!sessionId) {
        sessionId = createChatSession()
      }
      
      // 添加用户消息
      addMessage(sessionId, {
        role: 'user',
        content: content.trim()
      })

      // 如果这是第一条消息，更新会话标题
      const currentSession = chat.sessions.find(s => s.id === sessionId)
      if (currentSession && currentSession.messages.length === 0) {
        // 截取前30个字符作为标题，避免标题过长
        const title = content.trim().length > 30
          ? content.trim().substring(0, 30) + '...'
          : content.trim()
        updateSessionTitle(sessionId, title)
      }
      
      // 创建助手消息占位符
      const assistantMessageId = addMessage(sessionId, {
        role: 'assistant',
        content: ''
      })
      
      setChatLoading(true)
      setChatStreaming(true)
      
      // 取消之前的请求
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      
      abortControllerRef.current = new AbortController()
      
      let fullResponse = ''
      
      try {
        // 开始流式响应
        const streamGenerator = chatApi.sendMessage(content, sessionId)
        
        for await (const chunk of streamGenerator) {
          fullResponse += chunk
          updateMessage(sessionId, assistantMessageId, fullResponse)
        }
        
        // 如果响应为空，显示错误消息
        if (!fullResponse.trim()) {
          fullResponse = '抱歉，我现在无法回答您的问题。请稍后再试。'
          updateMessage(sessionId, assistantMessageId, fullResponse)
        }
        
      } catch (streamError) {
        console.error('Stream error:', streamError)
        
        // 流式请求失败，尝试普通请求作为后备
        try {
          const fallbackResponse = await chatApi.sendMessage(content, sessionId).next()
          if (fallbackResponse.value) {
            updateMessage(sessionId, assistantMessageId, fallbackResponse.value)
          } else {
            throw new Error('No response from fallback')
          }
        } catch (fallbackError) {
          console.error('Fallback error:', fallbackError)
          updateMessage(
            sessionId,
            assistantMessageId,
            '抱歉，网络连接出现问题。请检查您的网络连接后重试。'
          )
          
          addToast({
            type: 'error',
            title: '发送失败',
            message: '网络连接异常，请重试',
            duration: 5000
          })
        }
      }
      
    } catch (error) {
      console.error('Send message error:', error)
      
      addToast({
        type: 'error',
        title: '发送失败',
        message: apiUtils.handleError(error as any),
        duration: 5000
      })
    } finally {
      setChatLoading(false)
      setChatStreaming(false)
      abortControllerRef.current = null
    }
  }, [
    chat.currentSession?.id,
    chat.sessions,
    addMessage,
    updateMessage,
    setChatLoading,
    setChatStreaming,
    createChatSession,
    updateSessionTitle,
    addToast
  ])
  
  // 停止生成
  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    
    setChatLoading(false)
    setChatStreaming(false)
    
    addToast({
      type: 'info',
      title: '已停止生成',
      duration: 3000
    })
  }, [setChatLoading, setChatStreaming, addToast])
  
  // 重新生成回答
  const regenerateResponse = useCallback(async (messageId: string) => {
    const currentSession = chat.currentSession
    if (!currentSession) return
    
    // 找到要重新生成的消息
    const messageIndex = currentSession.messages.findIndex(m => m.id === messageId)
    if (messageIndex === -1) return
    
    const message = currentSession.messages[messageIndex]
    if (message.role !== 'assistant') return
    
    // 找到对应的用户消息
    const userMessage = currentSession.messages[messageIndex - 1]
    if (!userMessage || userMessage.role !== 'user') return
    
    // 清空助手消息内容
    updateMessage(currentSession.id, messageId, '')
    
    // 重新发送请求
    setChatLoading(true)
    setChatStreaming(true)
    
    try {
      let fullResponse = ''
      const streamGenerator = chatApi.sendMessage(userMessage.content, currentSession.id)
      
      for await (const chunk of streamGenerator) {
        fullResponse += chunk
        updateMessage(currentSession.id, messageId, fullResponse)
      }
      
    } catch (error) {
      console.error('Regenerate error:', error)
      updateMessage(
        currentSession.id,
        messageId,
        '抱歉，重新生成时出现错误。请稍后再试。'
      )
      
      addToast({
        type: 'error',
        title: '重新生成失败',
        message: apiUtils.handleError(error as any),
        duration: 5000
      })
    } finally {
      setChatLoading(false)
      setChatStreaming(false)
    }
  }, [
    chat.currentSession,
    updateMessage,
    setChatLoading,
    setChatStreaming,
    addToast
  ])
  
  // 复制消息
  const copyMessage = useCallback(async (content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      addToast({
        type: 'success',
        title: '已复制到剪贴板',
        duration: 2000
      })
    } catch (error) {
      console.error('Copy error:', error)
      addToast({
        type: 'error',
        title: '复制失败',
        message: '无法访问剪贴板',
        duration: 3000
      })
    }
  }, [addToast])
  
  // 提供反馈
  const provideFeedback = useCallback(async (messageId: string, type: 'like' | 'dislike') => {
    // TODO: 实现反馈API调用
    console.log('Feedback:', messageId, type)
    
    addToast({
      type: 'success',
      title: type === 'like' ? '感谢您的反馈！' : '我们会继续改进',
      duration: 2000
    })
  }, [addToast])
  
  return {
    // 状态
    currentSession: chat.currentSession,
    sessions: chat.sessions,
    isLoading: chat.isLoading,
    isStreaming: chat.isStreaming,
    
    // 方法
    sendMessage,
    stopGeneration,
    regenerateResponse,
    copyMessage,
    provideFeedback
  }
}