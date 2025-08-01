'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { Clock, Users, ChefHat, Heart } from 'lucide-react'
import { Recipe } from '@/types'
import { Card } from '@/components/ui'

interface RecipeCardProps {
  recipe: Recipe
  onSelect?: (recipe: Recipe) => void
  onFavorite?: (recipeId: string) => void
  isFavorited?: boolean
  className?: string
}

const RecipeCard: React.FC<RecipeCardProps> = ({
  recipe,
  onSelect,
  onFavorite,
  isFavorited = false,
  className = ''
}) => {
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
    <Card
      variant="glass"
      hover
      onClick={() => onSelect?.(recipe)}
      className={`overflow-hidden ${className}`}
    >
      {/* 图片区域 */}
      <div className="relative h-48 bg-gradient-to-br from-blue-100 to-purple-100">
        {recipe.imageUrl ? (
          <img
            src={recipe.imageUrl}
            alt={recipe.name}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <ChefHat className="w-16 h-16 text-gray-400" />
          </div>
        )}
        
        {/* 收藏按钮 */}
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={(e) => {
            e.stopPropagation()
            onFavorite?.(recipe.id)
          }}
          className="absolute top-3 right-3 p-2 rounded-full bg-white/20 backdrop-blur-sm hover:bg-white/30 transition-colors"
        >
          <Heart
            className={`w-5 h-5 ${
              isFavorited ? 'text-red-500 fill-current' : 'text-white'
            }`}
          />
        </motion.button>
        
        {/* 难度标签 */}
        <div className="absolute top-3 left-3">
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${difficultyColors[recipe.difficulty]}`}>
            {difficultyLabels[recipe.difficulty]}
          </span>
        </div>
      </div>
      
      {/* 内容区域 */}
      <div className="p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-2 line-clamp-2">
          {recipe.name}
        </h3>
        
        <p className="text-gray-600 text-sm mb-4 line-clamp-2">
          {recipe.description}
        </p>
        
        {/* 统计信息 */}
        <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <Clock className="w-4 h-4" />
              <span>{recipe.cookingTime + recipe.prepTime}分钟</span>
            </div>
            
            <div className="flex items-center space-x-1">
              <Users className="w-4 h-4" />
              <span>{recipe.servings}人份</span>
            </div>
          </div>
        </div>
        
        {/* 标签 */}
        <div className="flex flex-wrap gap-2">
          {recipe.tags.slice(0, 3).map((tag, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full"
            >
              {tag}
            </span>
          ))}
          {recipe.tags.length > 3 && (
            <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
              +{recipe.tags.length - 3}
            </span>
          )}
        </div>
      </div>
    </Card>
  )
}

export default RecipeCard