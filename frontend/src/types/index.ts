// 基础类型定义
export interface Recipe {
  id: string
  name: string
  description: string
  category: string
  difficulty: 'easy' | 'medium' | 'hard'
  cookingTime: number // 分钟
  prepTime: number // 分钟
  servings: number
  ingredients: Ingredient[]
  steps: CookingStep[]
  tags: string[]
  nutrition?: NutritionInfo
  rating?: number
  imageUrl?: string
  createdAt: string
  updatedAt: string
}

export interface Ingredient {
  id: string
  name: string
  amount: string
  unit: string
  category?: string
  isOptional?: boolean
  alternatives?: string[]
  checked?: boolean // 用于购物清单
}

export interface CookingStep {
  id: string
  stepNumber: number
  title: string
  description: string
  duration?: number // 分钟
  temperature?: number // 摄氏度
  tips?: string[]
  imageUrl?: string
  completed?: boolean
}

export interface NutritionInfo {
  calories: number
  protein: number // 克
  carbs: number // 克
  fat: number // 克
  fiber: number // 克
  sugar: number // 克
  sodium: number // 毫克
}

// 聊天相关类型
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  metadata?: {
    recipes?: Recipe[]
    suggestions?: string[]
    context?: any
  }
}

export interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: Date
  updatedAt: Date
}

// 用户偏好类型
export interface UserPreferences {
  dietaryRestrictions: string[] // 饮食限制
  allergies: string[] // 过敏信息
  favoriteCuisines: string[] // 喜欢的菜系
  dislikedIngredients: string[] // 不喜欢的食材
  spiceLevel: 'mild' | 'medium' | 'hot' // 辣度偏好
  cookingSkill: 'beginner' | 'intermediate' | 'advanced' // 烹饪技能
  mealTypes: string[] // 餐饮类型偏好
  budgetRange: 'low' | 'medium' | 'high' // 预算范围
}

// API响应类型
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  message?: string
  error?: string
}

export interface StreamResponse {
  chunk: string
  done: boolean
  metadata?: any
}

// 搜索和过滤类型
export interface SearchFilters {
  category?: string
  difficulty?: string
  maxCookingTime?: number
  ingredients?: string[]
  tags?: string[]
  rating?: number
  page?: number
  page_size?: number
}

export interface SearchResult {
  recipes: Recipe[]
  total: number
  page: number
  pageSize: number
}

// 计时器类型
export interface Timer {
  id: string
  name: string
  duration: number // 秒
  remaining: number // 秒
  isRunning: boolean
  isCompleted: boolean
  stepId?: string // 关联的烹饪步骤
}

// 收藏和评分类型
export interface UserRating {
  recipeId: string
  rating: number // 1-5
  review?: string
  createdAt: Date
}

export interface UserFavorite {
  recipeId: string
  createdAt: Date
}

// 状态管理类型
export interface AppState {
  // 用户状态
  user: {
    preferences: UserPreferences
    favorites: UserFavorite[]
    ratings: UserRating[]
  }
  
  // 聊天状态
  chat: {
    currentSession: ChatSession | null
    sessions: ChatSession[]
    isLoading: boolean
    isStreaming: boolean
  }
  
  // 菜谱状态
  recipes: {
    currentRecipe: Recipe | null
    searchResults: SearchResult | null
    recentRecipes: Recipe[]
    isLoading: boolean
  }
  
  // 烹饪状态
  cooking: {
    activeRecipe: Recipe | null
    currentStep: number
    timers: Timer[]
    completedSteps: string[]
  }
  
  // UI状态
  ui: {
    theme: 'light' | 'dark'
    sidebarOpen: boolean
    modalOpen: boolean
    toasts: Toast[]
  }
}

export interface Toast {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  duration?: number
  action?: {
    label: string
    onClick: () => void
  }
}

// 组件Props类型
export interface BaseProps {
  className?: string
  children?: React.ReactNode
}

// 路由类型
export type PageParams = {
  [key: string]: string | string[] | undefined
}

export type SearchParams = {
  [key: string]: string | string[] | undefined
}