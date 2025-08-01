import axios, { AxiosResponse, AxiosError } from 'axios'
import { ApiResponse, Recipe, SearchFilters, SearchResult, UserPreferences } from '@/types'

// 创建axios实例
const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' ? '' : 'http://backend:8000'),
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证token等
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response: AxiosResponse) => {
    return response
  },
  (error: AxiosError) => {
    // 统一错误处理
    console.error('API Error:', error)
    
    if (error.response?.status === 401) {
      // 处理认证失效
      localStorage.removeItem('auth_token')
      window.location.href = '/login'
    }
    
    return Promise.reject(error)
  }
)

// 聊天相关API
export const chatApi = {
  // 发送消息并获取流式响应
  sendMessage: async function* (message: string, sessionId?: string): AsyncGenerator<string> {
    const baseURL = typeof window !== 'undefined' ? '' : 'http://backend:8000'
    const response = await fetch(`${baseURL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        stream: true
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('Failed to get response reader')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            
            if (data === '[DONE]') {
              return
            }

            try {
              const parsed = JSON.parse(data)
              if (parsed.chunk) {
                yield parsed.chunk
              }
            } catch (e) {
              // 忽略解析错误，继续处理下一行
              continue
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  },

  // 获取聊天历史
  getChatHistory: async (sessionId: string): Promise<ApiResponse> => {
    const response = await api.get(`/api/chat/history/${sessionId}`)
    return response.data
  },

  // 创建新的聊天会话
  createSession: async (title?: string): Promise<ApiResponse<{ session_id: string }>> => {
    const response = await api.post('/api/chat/session', { title })
    return response.data
  },

  // 删除聊天会话
  deleteSession: async (sessionId: string): Promise<ApiResponse> => {
    const response = await api.delete(`/api/chat/session/${sessionId}`)
    return response.data
  }
}

// 菜谱相关API
export const recipeApi = {
  // 搜索菜谱
  searchRecipes: async (query: string, filters?: SearchFilters): Promise<ApiResponse<SearchResult>> => {
    const response = await api.post('/api/recipes/search', {
      query,
      filters,
      page: filters?.page || 1,
      page_size: filters?.page_size || 20
    })
    return response.data
  },

  // 获取菜谱详情
  getRecipe: async (recipeId: string): Promise<ApiResponse<Recipe>> => {
    const response = await api.get(`/api/recipes/${recipeId}`)
    return response.data
  },

  // 获取推荐菜谱
  getRecommendations: async (preferences?: UserPreferences): Promise<ApiResponse<Recipe[]>> => {
    const response = await api.post('/api/recipes/recommendations', { preferences })
    return response.data
  },

  // 获取分类菜谱
  getRecipesByCategory: async (category: string, page = 1): Promise<ApiResponse<SearchResult>> => {
    const response = await api.get(`/api/recipes/category/${category}?page=${page}`)
    return response.data
  }
}

// 用户相关API
export const userApi = {
  // 获取用户偏好
  getPreferences: async (): Promise<ApiResponse<UserPreferences>> => {
    const response = await api.get('/api/user/preferences')
    return response.data
  },

  // 更新用户偏好
  updatePreferences: async (preferences: Partial<UserPreferences>): Promise<ApiResponse> => {
    const response = await api.put('/api/user/preferences', preferences)
    return response.data
  },

  // 获取收藏列表
  getFavorites: async (): Promise<ApiResponse<Recipe[]>> => {
    const response = await api.get('/api/user/favorites')
    return response.data
  },

  // 添加收藏
  addFavorite: async (recipeId: string): Promise<ApiResponse> => {
    const response = await api.post('/api/user/favorites', { recipe_id: recipeId })
    return response.data
  },

  // 移除收藏
  removeFavorite: async (recipeId: string): Promise<ApiResponse> => {
    const response = await api.delete(`/api/user/favorites/${recipeId}`)
    return response.data
  },

  // 提交评分
  submitRating: async (recipeId: string, rating: number, review?: string): Promise<ApiResponse> => {
    const response = await api.post('/api/user/ratings', {
      recipe_id: recipeId,
      rating,
      review
    })
    return response.data
  }
}

// 系统相关API
export const systemApi = {
  // 健康检查
  healthCheck: async (): Promise<ApiResponse> => {
    const response = await api.get('/api/health')
    return response.data
  },

  // 获取系统统计
  getStats: async (): Promise<ApiResponse> => {
    const response = await api.get('/api/stats')
    return response.data
  }
}

// 工具函数
export const apiUtils = {
  // 处理API错误
  handleError: (error: AxiosError): string => {
    const errorData = error.response?.data as { message?: string } | undefined
    if (errorData?.message) {
      return errorData.message
    }
    
    if (error.response?.status === 404) {
      return '请求的资源不存在'
    }
    
    if (error.response?.status === 500) {
      return '服务器内部错误，请稍后重试'
    }
    
    if (error.code === 'ECONNABORTED') {
      return '请求超时，请检查网络连接'
    }
    
    return error.message || '未知错误'
  },

  // 重试请求
  retryRequest: async <T>(
    requestFn: () => Promise<T>,
    maxRetries = 3,
    delay = 1000
  ): Promise<T> => {
    let lastError: any
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await requestFn()
      } catch (error) {
        lastError = error
        
        if (i < maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)))
        }
      }
    }
    
    throw lastError
  }
}

export default api