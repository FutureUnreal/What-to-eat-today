'use client'

import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { MessageCircle, Search, TrendingUp, Heart, ChefHat, Sparkles, X } from 'lucide-react'
import { Button, Card, CardContent } from '@/components/ui'
import { RecipeCard } from '@/components/recipe'
import { useAppStore } from '@/store'
import { useRecipes } from '@/hooks'
import { Recipe } from '@/types'
import { useRouter } from 'next/navigation'

const HomePage: React.FC = () => {
  const router = useRouter()
  const { createChatSession, ui, setSidebarOpen } = useAppStore()
  const { getRecommendations } = useRecipes()
  
  const [recommendations, setRecommendations] = useState<Recipe[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [sidebarType, setSidebarType] = useState<'search' | 'favorites' | null>(null)
  
  const loadRecommendations = async () => {
    setIsLoading(true)
    try {
      const recs = await getRecommendations()
      setRecommendations(recs.slice(0, 3)) // 只显示3个推荐
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadRecommendations()
  }, [getRecommendations])

  const handleRefreshRecommendations = () => {
    loadRecommendations()
  }
  
  const handleStartChat = () => {
    const sessionId = createChatSession('今天吃什么？')
    window.location.href = `/chat?session=${sessionId}`
  }
  
  const quickQuestions = [
    '今天晚餐吃什么好？',
    '有什么简单易做的家常菜？',
    '适合减肥的低卡菜谱',
    '30分钟内能做完的菜',
    '适合新手的烘焙食谱',
    '下饭的家常菜推荐'
  ]

  // 处理侧边栏
  const handleOpenSidebar = (type: 'search' | 'favorites') => {
    setSidebarType(type)
    setSidebarOpen(true)
  }

  const handleCloseSidebar = () => {
    setSidebarOpen(false)
    setSidebarType(null)
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* 侧边栏 */}
      <AnimatePresence>
        {ui.sidebarOpen && (
          <>
            {/* 移动端遮罩 */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 z-40 lg:hidden"
              onClick={handleCloseSidebar}
            />

            {/* 侧边栏内容 */}
            <motion.aside
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              className="fixed left-0 top-0 h-full w-80 glass border-r border-white/20 z-50 flex flex-col"
            >
              {/* 侧边栏头部 */}
              <div className="p-4 border-b border-white/10">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {sidebarType === 'search' ? (
                      <>
                        <Search className="w-6 h-6 text-blue-500" />
                        <h2 className="text-lg font-semibold gradient-text">搜索菜谱</h2>
                      </>
                    ) : (
                      <>
                        <Heart className="w-6 h-6 text-red-500" />
                        <h2 className="text-lg font-semibold gradient-text">我的收藏</h2>
                      </>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleCloseSidebar}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              {/* 侧边栏内容 */}
              <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
                {sidebarType === 'search' ? (
                  <div className="space-y-4">
                    <div>
                      <input
                        type="text"
                        placeholder="搜索菜谱..."
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    <div className="text-gray-500 text-center py-8">
                      输入关键词搜索菜谱
                    </div>
                  </div>
                ) : (
                  <div className="text-gray-500 text-center py-8">
                    暂无收藏的菜谱
                  </div>
                )}
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
      {/* 头部导航 */}
      <nav className="glass border-b border-white/20 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <ChefHat className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold gradient-text">今天吃什么</h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleOpenSidebar('search')}
              >
                <Search className="w-4 h-4 mr-2" />
                搜索
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleOpenSidebar('favorites')}
              >
                <Heart className="w-4 h-4 mr-2" />
                收藏
              </Button>
            </div>
          </div>
        </div>
      </nav>
      
      {/* 主内容区域 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 欢迎区域 */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className="max-w-3xl mx-auto">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
              className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6"
            >
              <Sparkles className="w-10 h-10 text-white" />
            </motion.div>
            
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-4xl md:text-6xl font-bold gradient-text mb-6"
            >
              今天吃什么？
            </motion.h1>
            
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-xl text-gray-600 mb-8"
            >
              AI美食助手为您推荐个性化菜谱，提供详细烹饪指导
            </motion.p>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Button
                variant="primary"
                size="lg"
                onClick={handleStartChat}
                className="text-lg px-8 py-4"
              >
                <MessageCircle className="w-5 h-5 mr-2" />
                开始对话
              </Button>
            </motion.div>
          </div>
        </motion.section>
        
        {/* 快速问题 */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mb-16"
        >
          <h2 className="text-2xl font-semibold text-gray-900 mb-6 text-center">
            💡 快速开始
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {quickQuestions.map((question, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 + index * 0.1 }}
              >
                <Card
                  variant="glass"
                  hover
                  onClick={() => {
                    const sessionId = createChatSession(question)
                    window.location.href = `/chat?session=${sessionId}&q=${encodeURIComponent(question)}`
                  }}
                  className="cursor-pointer h-full flex items-center justify-center"
                >
                  <div className="p-4 w-full text-center">
                    <p className="text-gray-700 leading-relaxed">{question}</p>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.section>
        
        {/* 推荐菜谱 */}
        {recommendations.length > 0 && (
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="mb-16"
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-semibold text-gray-900 flex items-center">
                <Heart className="w-6 h-6 text-red-500 mr-2" />
                为您推荐
              </h2>
              <Button
                variant="ghost"
                onClick={handleRefreshRecommendations}
                disabled={isLoading}
              >
                换一批
              </Button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recommendations.map((recipe, index) => (
                <motion.div
                  key={recipe.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.9 + index * 0.1 }}
                >
                  <RecipeCard
                    recipe={recipe}
                    onSelect={(recipe) => {
                      window.location.href = `/recipe/${recipe.id}`
                    }}
                  />
                </motion.div>
              ))}
            </div>
          </motion.section>
        )}
        
        
        {/* 加载状态 */}
        {isLoading && (
          <div className="text-center py-12">
            <div className="loading-dots text-blue-500 mb-4">
              <div></div>
              <div></div>
              <div></div>
            </div>
            <p className="text-gray-600">正在为您准备美食推荐...</p>
          </div>
        )}
      </main>
      
      {/* 页脚 */}
      <footer className="glass border-t border-white/20 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-4">
              <ChefHat className="w-5 h-5 text-blue-500" />
              <span className="text-lg font-semibold gradient-text">今天吃什么</span>
            </div>
            <p className="text-gray-600 text-sm">
              基于图RAG技术的智能烹饪助手 · 让美食更简单
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default HomePage