import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { AppState, UserPreferences, ChatSession, Recipe, Timer, Toast } from '@/types'

interface AppStore extends AppState {
  // 用户偏好管理
  updateUserPreferences: (preferences: Partial<UserPreferences>) => void
  addFavorite: (recipeId: string) => void
  removeFavorite: (recipeId: string) => void
  addRating: (recipeId: string, rating: number, review?: string) => void
  
  // 聊天管理
  createChatSession: (title?: string) => string
  switchChatSession: (sessionId: string) => void
  deleteChatSession: (sessionId: string) => void
  updateSessionTitle: (sessionId: string, title: string) => void
  addMessage: (sessionId: string, message: Omit<ChatSession['messages'][0], 'id' | 'timestamp'>) => string
  updateMessage: (sessionId: string, messageId: string, content: string) => void
  setChatLoading: (loading: boolean) => void
  setChatStreaming: (streaming: boolean) => void
  
  // 菜谱管理
  setCurrentRecipe: (recipe: Recipe | null) => void
  setSearchResults: (results: AppState['recipes']['searchResults']) => void
  addRecentRecipe: (recipe: Recipe) => void
  setRecipesLoading: (loading: boolean) => void
  
  // 烹饪管理
  startCooking: (recipe: Recipe) => void
  stopCooking: () => void
  setCurrentStep: (step: number) => void
  completeStep: (stepId: string) => void
  uncompleteStep: (stepId: string) => void
  addTimer: (timer: Omit<Timer, 'id'>) => string
  updateTimer: (timerId: string, updates: Partial<Timer>) => void
  removeTimer: (timerId: string) => void
  
  // UI管理
  setTheme: (theme: 'light' | 'dark') => void
  setSidebarOpen: (open: boolean) => void
  setModalOpen: (open: boolean) => void
  addToast: (toast: Omit<Toast, 'id'>) => string
  removeToast: (toastId: string) => void
  clearToasts: () => void
}

const initialUserPreferences: UserPreferences = {
  dietaryRestrictions: [],
  allergies: [],
  favoriteCuisines: [],
  dislikedIngredients: [],
  spiceLevel: 'medium',
  cookingSkill: 'intermediate',
  mealTypes: [],
  budgetRange: 'medium'
}

export const useAppStore = create<AppStore>()(
  persist(
    (set, get) => ({
      // 初始状态
      user: {
        preferences: initialUserPreferences,
        favorites: [],
        ratings: []
      },
      chat: {
        currentSession: null,
        sessions: [],
        isLoading: false,
        isStreaming: false
      },
      recipes: {
        currentRecipe: null,
        searchResults: null,
        recentRecipes: [],
        isLoading: false
      },
      cooking: {
        activeRecipe: null,
        currentStep: 0,
        timers: [],
        completedSteps: []
      },
      ui: {
        theme: 'light',
        sidebarOpen: false,
        modalOpen: false,
        toasts: []
      },

      // 用户偏好管理
      updateUserPreferences: (preferences) =>
        set((state) => ({
          user: {
            ...state.user,
            preferences: { ...state.user.preferences, ...preferences }
          }
        })),

      addFavorite: (recipeId) =>
        set((state) => {
          const exists = state.user.favorites.some(f => f.recipeId === recipeId)
          if (exists) return state
          
          return {
            user: {
              ...state.user,
              favorites: [
                ...state.user.favorites,
                { recipeId, createdAt: new Date() }
              ]
            }
          }
        }),

      removeFavorite: (recipeId) =>
        set((state) => ({
          user: {
            ...state.user,
            favorites: state.user.favorites.filter(f => f.recipeId !== recipeId)
          }
        })),

      addRating: (recipeId, rating, review) =>
        set((state) => {
          const existingIndex = state.user.ratings.findIndex(r => r.recipeId === recipeId)
          const newRating = { recipeId, rating, review, createdAt: new Date() }
          
          if (existingIndex >= 0) {
            const newRatings = [...state.user.ratings]
            newRatings[existingIndex] = newRating
            return {
              user: { ...state.user, ratings: newRatings }
            }
          } else {
            return {
              user: {
                ...state.user,
                ratings: [...state.user.ratings, newRating]
              }
            }
          }
        }),

      // 聊天管理
      createChatSession: (title) => {
        const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        const newSession: ChatSession = {
          id: sessionId,
          title: title || '新对话',
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date()
        }

        set((state) => ({
          chat: {
            ...state.chat,
            sessions: [newSession, ...state.chat.sessions],
            currentSession: newSession
          }
        }))

        return sessionId
      },

      switchChatSession: (sessionId) =>
        set((state) => {
          const session = state.chat.sessions.find(s => s.id === sessionId)
          return {
            chat: { ...state.chat, currentSession: session || null }
          }
        }),

      deleteChatSession: (sessionId) =>
        set((state) => {
          const newSessions = state.chat.sessions.filter(s => s.id !== sessionId)
          const currentSession = state.chat.currentSession?.id === sessionId
            ? (newSessions[0] || null)
            : state.chat.currentSession

          return {
            chat: {
              ...state.chat,
              sessions: newSessions,
              currentSession
            }
          }
        }),

      updateSessionTitle: (sessionId, title) =>
        set((state) => {
          const sessionIndex = state.chat.sessions.findIndex(s => s.id === sessionId)
          if (sessionIndex === -1) return state

          const newSessions = [...state.chat.sessions]
          newSessions[sessionIndex] = {
            ...newSessions[sessionIndex],
            title,
            updatedAt: new Date()
          }

          return {
            chat: {
              ...state.chat,
              sessions: newSessions,
              currentSession: state.chat.currentSession?.id === sessionId
                ? newSessions[sessionIndex]
                : state.chat.currentSession
            }
          }
        }),

      addMessage: (sessionId, messageData) => {
        const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        const message = {
          ...messageData,
          id: messageId,
          timestamp: new Date()
        }

        set((state) => {
          const sessionIndex = state.chat.sessions.findIndex(s => s.id === sessionId)
          if (sessionIndex === -1) return state

          const newSessions = [...state.chat.sessions]
          newSessions[sessionIndex] = {
            ...newSessions[sessionIndex],
            messages: [...newSessions[sessionIndex].messages, message],
            updatedAt: new Date()
          }

          return {
            chat: {
              ...state.chat,
              sessions: newSessions,
              currentSession: state.chat.currentSession?.id === sessionId
                ? newSessions[sessionIndex]
                : state.chat.currentSession
            }
          }
        })
        
        return messageId
      },

      updateMessage: (sessionId, messageId, content) =>
        set((state) => {
          const sessionIndex = state.chat.sessions.findIndex(s => s.id === sessionId)
          if (sessionIndex === -1) return state

          const session = state.chat.sessions[sessionIndex]
          const messageIndex = session.messages.findIndex(m => m.id === messageId)
          if (messageIndex === -1) return state

          const newSessions = [...state.chat.sessions]
          const newMessages = [...session.messages]
          newMessages[messageIndex] = { ...newMessages[messageIndex], content }
          
          newSessions[sessionIndex] = {
            ...session,
            messages: newMessages,
            updatedAt: new Date()
          }

          return {
            chat: {
              ...state.chat,
              sessions: newSessions,
              currentSession: state.chat.currentSession?.id === sessionId
                ? newSessions[sessionIndex]
                : state.chat.currentSession
            }
          }
        }),

      setChatLoading: (loading) =>
        set((state) => ({
          chat: { ...state.chat, isLoading: loading }
        })),

      setChatStreaming: (streaming) =>
        set((state) => ({
          chat: { ...state.chat, isStreaming: streaming }
        })),

      // 菜谱管理
      setCurrentRecipe: (recipe) =>
        set((state) => ({
          recipes: { ...state.recipes, currentRecipe: recipe }
        })),

      setSearchResults: (results) =>
        set((state) => ({
          recipes: { ...state.recipes, searchResults: results }
        })),

      addRecentRecipe: (recipe) =>
        set((state) => {
          const filtered = state.recipes.recentRecipes.filter(r => r.id !== recipe.id)
          return {
            recipes: {
              ...state.recipes,
              recentRecipes: [recipe, ...filtered].slice(0, 10) // 保持最近10个
            }
          }
        }),

      setRecipesLoading: (loading) =>
        set((state) => ({
          recipes: { ...state.recipes, isLoading: loading }
        })),

      // 烹饪管理
      startCooking: (recipe) =>
        set((state) => ({
          cooking: {
            ...state.cooking,
            activeRecipe: recipe,
            currentStep: 0,
            completedSteps: []
          }
        })),

      stopCooking: () =>
        set((state) => ({
          cooking: {
            ...state.cooking,
            activeRecipe: null,
            currentStep: 0,
            completedSteps: [],
            timers: [] // 清除所有计时器
          }
        })),

      setCurrentStep: (step) =>
        set((state) => ({
          cooking: { ...state.cooking, currentStep: step }
        })),

      completeStep: (stepId) =>
        set((state) => ({
          cooking: {
            ...state.cooking,
            completedSteps: [...state.cooking.completedSteps, stepId]
          }
        })),

      uncompleteStep: (stepId) =>
        set((state) => ({
          cooking: {
            ...state.cooking,
            completedSteps: state.cooking.completedSteps.filter(id => id !== stepId)
          }
        })),

      addTimer: (timerData) => {
        const timerId = `timer_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        const timer = { ...timerData, id: timerId }

        set((state) => ({
          cooking: {
            ...state.cooking,
            timers: [...state.cooking.timers, timer]
          }
        }))

        return timerId
      },

      updateTimer: (timerId, updates) =>
        set((state) => {
          const timerIndex = state.cooking.timers.findIndex(t => t.id === timerId)
          if (timerIndex === -1) return state

          const newTimers = [...state.cooking.timers]
          newTimers[timerIndex] = { ...newTimers[timerIndex], ...updates }

          return {
            cooking: { ...state.cooking, timers: newTimers }
          }
        }),

      removeTimer: (timerId) =>
        set((state) => ({
          cooking: {
            ...state.cooking,
            timers: state.cooking.timers.filter(t => t.id !== timerId)
          }
        })),

      // UI管理
      setTheme: (theme) =>
        set((state) => ({
          ui: { ...state.ui, theme }
        })),

      setSidebarOpen: (open) =>
        set((state) => ({
          ui: { ...state.ui, sidebarOpen: open }
        })),

      setModalOpen: (open) =>
        set((state) => ({
          ui: { ...state.ui, modalOpen: open }
        })),

      addToast: (toastData) => {
        const toastId = `toast_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        const toast = { ...toastData, id: toastId }

        set((state) => ({
          ui: {
            ...state.ui,
            toasts: [...(Array.isArray(state.ui.toasts) ? state.ui.toasts : []), toast]
          }
        }))

        return toastId
      },

      removeToast: (toastId) =>
        set((state) => ({
          ui: {
            ...state.ui,
            toasts: (Array.isArray(state.ui.toasts) ? state.ui.toasts : []).filter(t => t.id !== toastId)
          }
        })),

      clearToasts: () =>
        set((state) => ({
          ui: { ...state.ui, toasts: [] }
        }))
    }),
    {
      name: 'what-to-eat-today-store',
      storage: createJSONStorage(() => localStorage),
      version: 1,
      partialize: (state) => ({
        user: state.user,
        chat: {
          sessions: state.chat.sessions,
          currentSession: state.chat.currentSession
        },
        recipes: {
          recentRecipes: state.recipes.recentRecipes
        },
        ui: {
          theme: state.ui.theme
        }
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          // 确保 toasts 始终是数组
          if (!Array.isArray(state.ui.toasts)) {
            state.ui.toasts = []
          }

          // 安全地恢复日期对象
          const safeDate = (dateValue: any) => {
            if (!dateValue) return new Date()
            if (dateValue instanceof Date) return dateValue
            try {
              const date = new Date(dateValue)
              return isNaN(date.getTime()) ? new Date() : date
            } catch {
              return new Date()
            }
          }

          if (state.chat.currentSession) {
            state.chat.currentSession.createdAt = safeDate(state.chat.currentSession.createdAt)
            state.chat.currentSession.updatedAt = safeDate(state.chat.currentSession.updatedAt)

            // 恢复消息中的时间戳
            if (state.chat.currentSession.messages) {
              state.chat.currentSession.messages = state.chat.currentSession.messages.map(msg => ({
                ...msg,
                timestamp: safeDate(msg.timestamp)
              }))
            }
          }

          // 恢复所有会话的日期对象
          if (state.chat.sessions && Array.isArray(state.chat.sessions)) {
            state.chat.sessions = state.chat.sessions.map(session => ({
              ...session,
              createdAt: safeDate(session.createdAt),
              updatedAt: safeDate(session.updatedAt),
              messages: session.messages ? session.messages.map(msg => ({
                ...msg,
                timestamp: safeDate(msg.timestamp)
              })) : []
            }))
          }
        }
      }
    }
  )
)