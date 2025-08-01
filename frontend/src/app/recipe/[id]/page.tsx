'use client'

import React, { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  ChefHat, Clock, Users, Heart, Share2,
  ArrowLeft, Play, BookOpen, Timer, Thermometer
} from 'lucide-react'
import { Button, Card, CardContent, LoadingSpinner } from '@/components/ui'
import { IngredientList, CookingSteps } from '@/components/recipe'
import { useAppStore } from '@/store'
import { useRecipes } from '@/hooks'
import { Recipe } from '@/types'

const RecipeDetailPage: React.FC = () => {
  const params = useParams()
  const router = useRouter()
  const recipeId = params.id as string
  
  const {
    cooking,
    startCooking,
    stopCooking,
    setCurrentStep,
    addToast
  } = useAppStore()
  
  const {
    currentRecipe,
    isLoading,
    getRecipeDetails,
    toggleFavorite,
    isFavorited
  } = useRecipes()
  
  const [servings, setServings] = useState(1)
  const [activeTab, setActiveTab] = useState<'ingredients' | 'steps' | 'nutrition'>('ingredients')
  
  useEffect(() => {
    if (recipeId) {
      getRecipeDetails(recipeId)
    }
  }, [recipeId, getRecipeDetails])
  
  useEffect(() => {
    if (currentRecipe) {
      setServings(currentRecipe.servings)
    }
  }, [currentRecipe])
  
  const handleStartCooking = () => {
    if (currentRecipe) {
      startCooking(currentRecipe)
      router.push(`/cooking/${currentRecipe.id}`)
    }
  }
  
  const handleShare = async () => {
    if (currentRecipe) {
      try {
        await navigator.share({
          title: currentRecipe.name,
          text: currentRecipe.description,
          url: window.location.href
        })
      } catch (error) {
        // 如果不支持原生分享，复制链接
        await navigator.clipboard.writeText(window.location.href)
        addToast({
          type: 'success',
          title: '链接已复制',
          message: '可以分享给朋友了！',
          duration: 3000
        })
      }
    }
  }
  
  
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="lg" className="mb-4" />
          <p className="text-gray-600">正在加载菜谱详情...</p>
        </div>
      </div>
    )
  }
  
  if (!currentRecipe) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <ChefHat className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">菜谱不存在</h2>
          <p className="text-gray-600 mb-6">抱歉，找不到您要查看的菜谱</p>
          <Button onClick={() => router.push('/')}>
            返回首页
          </Button>
        </div>
      </div>
    )
  }
  
  const difficultyColors = {
    easy: 'text-green-600 bg-green-100',
    medium: 'text-yellow-600 bg-yellow-100',
    hard: 'text-red-600 bg-red-100'
  }
  
  const difficultyLabels = {
    easy: '简单',
    medium: '中等',
    hard: '困难'
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* 顶部导航 */}
      <nav className="glass border-b border-white/20 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Button
              variant="ghost"
              onClick={() => router.back()}
              className="flex items-center"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              返回
            </Button>
            
            <div className="flex items-center space-x-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => toggleFavorite(currentRecipe.id)}
              >
                <Heart
                  className={`w-4 h-4 mr-2 ${
                    isFavorited(currentRecipe.id) ? 'text-red-500 fill-current' : ''
                  }`}
                />
                {isFavorited(currentRecipe.id) ? '已收藏' : '收藏'}
              </Button>
              
              <Button variant="ghost" size="sm" onClick={handleShare}>
                <Share2 className="w-4 h-4 mr-2" />
                分享
              </Button>
              
              <Button
                variant="primary"
                onClick={handleStartCooking}
              >
                <Play className="w-4 h-4 mr-2" />
                开始烹饪
              </Button>
            </div>
          </div>
        </div>
      </nav>
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 菜谱头部信息 */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* 图片区域 */}
            <div className="relative">
              <Card variant="glass" className="overflow-hidden">
                <div className="aspect-video bg-gradient-to-br from-blue-100 to-purple-100">
                  {currentRecipe.imageUrl ? (
                    <img
                      src={currentRecipe.imageUrl}
                      alt={currentRecipe.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <ChefHat className="w-24 h-24 text-gray-400" />
                    </div>
                  )}
                </div>
              </Card>
            </div>
            
            {/* 基本信息 */}
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${difficultyColors[currentRecipe.difficulty]}`}>
                  {difficultyLabels[currentRecipe.difficulty]}
                </span>
                <span className="text-sm text-gray-500">{currentRecipe.category}</span>
              </div>
              
              <h1 className="text-3xl font-bold text-gray-900 mb-4">
                {currentRecipe.name}
              </h1>
              
              <p className="text-gray-600 text-lg mb-6 leading-relaxed">
                {currentRecipe.description}
              </p>
              
              {/* 统计信息 */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <Card variant="glass">
                  <CardContent className="p-4 text-center">
                    <Clock className="w-6 h-6 text-blue-500 mx-auto mb-2" />
                    <p className="text-sm text-gray-600">总用时</p>
                    <p className="text-lg font-semibold">
                      {currentRecipe.cookingTime + currentRecipe.prepTime} 分钟
                    </p>
                  </CardContent>
                </Card>
                
                <Card variant="glass">
                  <CardContent className="p-4 text-center">
                    <Users className="w-6 h-6 text-green-500 mx-auto mb-2" />
                    <p className="text-sm text-gray-600">份数</p>
                    <p className="text-lg font-semibold">{currentRecipe.servings} 人份</p>
                  </CardContent>
                </Card>
              </div>
              {/* 标签 */}
              <div className="flex flex-wrap gap-2">
                {currentRecipe.tags.map((tag, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-blue-100 text-blue-700 text-sm rounded-full"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </motion.section>
        
        {/* 标签页导航 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <div className="flex space-x-1 glass rounded-xl p-1">
            {[
              { id: 'ingredients', label: '食材清单', icon: BookOpen },
              { id: 'steps', label: '烹饪步骤', icon: Timer },
              { id: 'nutrition', label: '营养信息', icon: Thermometer }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-lg transition-all duration-200 ${
                  activeTab === tab.id
                    ? 'bg-white shadow-md text-blue-600'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </div>
        </motion.div>
        
        {/* 标签页内容 */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          {activeTab === 'ingredients' && (
            <IngredientList
              ingredients={currentRecipe.ingredients}
              servings={servings}
              originalServings={currentRecipe.servings}
              onServingsChange={setServings}
              showShoppingList
            />
          )}
          
          {activeTab === 'steps' && (
            <CookingSteps
              steps={currentRecipe.steps}
              currentStep={cooking.currentStep}
              onStepComplete={(stepId) => console.log('Step completed:', stepId)}
              onStepChange={setCurrentStep}
            />
          )}
          
          {activeTab === 'nutrition' && currentRecipe.nutrition && (
            <Card variant="glass">
              <CardContent className="p-6">
                <h3 className="text-xl font-semibold mb-4">营养信息 (每份)</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-orange-500">
                      {Math.round(currentRecipe.nutrition.calories * servings / currentRecipe.servings)}
                    </p>
                    <p className="text-sm text-gray-600">卡路里</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-blue-500">
                      {Math.round(currentRecipe.nutrition.protein * servings / currentRecipe.servings)}g
                    </p>
                    <p className="text-sm text-gray-600">蛋白质</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-green-500">
                      {Math.round(currentRecipe.nutrition.carbs * servings / currentRecipe.servings)}g
                    </p>
                    <p className="text-sm text-gray-600">碳水化合物</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-purple-500">
                      {Math.round(currentRecipe.nutrition.fat * servings / currentRecipe.servings)}g
                    </p>
                    <p className="text-sm text-gray-600">脂肪</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </motion.div>
        
        {/* 底部操作区 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mt-8 text-center"
        >
          <Button
            variant="primary"
            size="lg"
            onClick={handleStartCooking}
            className="px-8 py-4 text-lg"
          >
            <Play className="w-5 h-5 mr-2" />
            开始烹饪
          </Button>
        </motion.div>
      </main>
    </div>
  )
}

export default RecipeDetailPage