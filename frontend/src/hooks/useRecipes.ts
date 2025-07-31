import { useCallback, useState } from 'react'
import { useAppStore } from '@/store'
import { recipeApi, userApi, apiUtils } from '@/lib/api'
import { Recipe, SearchFilters, UserPreferences } from '@/types'

export const useRecipes = () => {
  const {
    recipes,
    user,
    setCurrentRecipe,
    setSearchResults,
    addRecentRecipe,
    setRecipesLoading,
    addFavorite,
    removeFavorite,
    addRating,
    addToast
  } = useAppStore()
  
  const [searchQuery, setSearchQuery] = useState('')
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({})
  
  // 搜索菜谱
  const searchRecipes = useCallback(async (
    query: string,
    filters?: SearchFilters,
    page = 1
  ) => {
    if (!query.trim() && !filters) return
    
    setRecipesLoading(true)
    setSearchQuery(query)
    setSearchFilters(filters || {})
    
    try {
      const response = await recipeApi.searchRecipes(query, {
        ...filters,
        page,
        page_size: 20
      })
      
      if (response.success && response.data) {
        setSearchResults(response.data)
      } else {
        throw new Error(response.message || '搜索失败')
      }
    } catch (error) {
      console.error('Search recipes error:', error)
      addToast({
        type: 'error',
        title: '搜索失败',
        message: apiUtils.handleError(error as any),
        duration: 5000
      })
    } finally {
      setRecipesLoading(false)
    }
  }, [setRecipesLoading, setSearchResults, addToast])
  
  // 获取菜谱详情
  const getRecipeDetails = useCallback(async (recipeId: string) => {
    setRecipesLoading(true)
    
    try {
      const response = await recipeApi.getRecipe(recipeId)
      
      if (response.success && response.data) {
        const recipe = response.data
        setCurrentRecipe(recipe)
        addRecentRecipe(recipe)
        return recipe
      } else {
        throw new Error(response.message || '获取菜谱详情失败')
      }
    } catch (error) {
      console.error('Get recipe details error:', error)
      addToast({
        type: 'error',
        title: '获取失败',
        message: apiUtils.handleError(error as any),
        duration: 5000
      })
      return null
    } finally {
      setRecipesLoading(false)
    }
  }, [setRecipesLoading, setCurrentRecipe, addRecentRecipe, addToast])
  
  // 获取推荐菜谱
  const getRecommendations = useCallback(async (preferences?: UserPreferences) => {
    setRecipesLoading(true)
    
    try {
      const response = await recipeApi.getRecommendations(
        preferences || user.preferences
      )
      
      if (response.success && response.data) {
        return response.data
      } else {
        throw new Error(response.message || '获取推荐失败')
      }
    } catch (error) {
      console.error('Get recommendations error:', error)
      addToast({
        type: 'error',
        title: '获取推荐失败',
        message: apiUtils.handleError(error as any),
        duration: 5000
      })
      return []
    } finally {
      setRecipesLoading(false)
    }
  }, [setRecipesLoading, user.preferences, addToast])
  
  
  // 按分类获取菜谱
  const getRecipesByCategory = useCallback(async (category: string, page = 1) => {
    setRecipesLoading(true)
    
    try {
      const response = await recipeApi.getRecipesByCategory(category, page)
      
      if (response.success && response.data) {
        setSearchResults(response.data)
        return response.data
      } else {
        throw new Error(response.message || '获取分类菜谱失败')
      }
    } catch (error) {
      console.error('Get recipes by category error:', error)
      addToast({
        type: 'error',
        title: '获取失败',
        message: apiUtils.handleError(error as any),
        duration: 5000
      })
      return null
    } finally {
      setRecipesLoading(false)
    }
  }, [setRecipesLoading, setSearchResults, addToast])
  
  // 切换收藏状态
  const toggleFavorite = useCallback(async (recipeId: string) => {
    const isFavorited = user.favorites.some(f => f.recipeId === recipeId)
    
    try {
      if (isFavorited) {
        await userApi.removeFavorite(recipeId)
        removeFavorite(recipeId)
        addToast({
          type: 'info',
          title: '已取消收藏',
          duration: 2000
        })
      } else {
        await userApi.addFavorite(recipeId)
        addFavorite(recipeId)
        addToast({
          type: 'success',
          title: '已添加收藏',
          duration: 2000
        })
      }
    } catch (error) {
      console.error('Toggle favorite error:', error)
      addToast({
        type: 'error',
        title: isFavorited ? '取消收藏失败' : '收藏失败',
        message: apiUtils.handleError(error as any),
        duration: 5000
      })
    }
  }, [user.favorites, addFavorite, removeFavorite, addToast])
  
  // 提交评分
  const submitRating = useCallback(async (
    recipeId: string,
    rating: number,
    review?: string
  ) => {
    try {
      await userApi.submitRating(recipeId, rating, review)
      addRating(recipeId, rating, review)
      
      addToast({
        type: 'success',
        title: '评分提交成功',
        message: '感谢您的评分！',
        duration: 3000
      })
    } catch (error) {
      console.error('Submit rating error:', error)
      addToast({
        type: 'error',
        title: '评分提交失败',
        message: apiUtils.handleError(error as any),
        duration: 5000
      })
    }
  }, [addRating, addToast])
  
  // 获取用户收藏
  const getUserFavorites = useCallback(async () => {
    try {
      const response = await userApi.getFavorites()
      
      if (response.success && response.data) {
        return response.data
      } else {
        throw new Error(response.message || '获取收藏列表失败')
      }
    } catch (error) {
      console.error('Get user favorites error:', error)
      return []
    }
  }, [])
  
  // 检查是否已收藏
  const isFavorited = useCallback((recipeId: string) => {
    return user.favorites.some(f => f.recipeId === recipeId)
  }, [user.favorites])
  
  // 获取用户评分
  const getUserRating = useCallback((recipeId: string) => {
    return user.ratings.find(r => r.recipeId === recipeId)
  }, [user.ratings])
  
  // 清除搜索结果
  const clearSearchResults = useCallback(() => {
    setSearchResults(null)
    setSearchQuery('')
    setSearchFilters({})
  }, [setSearchResults])
  
  return {
    // 状态
    currentRecipe: recipes.currentRecipe,
    searchResults: recipes.searchResults,
    recentRecipes: recipes.recentRecipes,
    isLoading: recipes.isLoading,
    searchQuery,
    searchFilters,
    
    // 方法
    searchRecipes,
    getRecipeDetails,
    getRecommendations,
    getRecipesByCategory,
    toggleFavorite,
    submitRating,
    getUserFavorites,
    isFavorited,
    getUserRating,
    clearSearchResults,
    
    // 设置器
    setSearchQuery,
    setSearchFilters
  }
}