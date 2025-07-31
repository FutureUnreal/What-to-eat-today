'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface CardProps {
  children: React.ReactNode
  className?: string
  variant?: 'default' | 'glass' | 'elevated'
  hover?: boolean
  onClick?: () => void
}

const Card: React.FC<CardProps> = ({
  children,
  className = '',
  variant = 'default',
  hover = false,
  onClick
}) => {
  const baseClasses = 'rounded-2xl transition-all duration-300'
  
  const variantClasses = {
    default: 'bg-white shadow-md border border-gray-200',
    glass: 'glass',
    elevated: 'bg-white shadow-xl border border-gray-100'
  }
  
  const hoverClasses = hover ? 'hover:shadow-xl hover:-translate-y-1 cursor-pointer' : ''
  const clickableClasses = onClick ? 'cursor-pointer' : ''
  
  const classes = `${baseClasses} ${variantClasses[variant]} ${hoverClasses} ${clickableClasses} ${className}`
  
  const cardContent = (
    <div className={classes} onClick={onClick}>
      {children}
    </div>
  )
  
  if (hover || onClick) {
    return (
      <motion.div
        whileHover={{ y: hover ? -4 : 0 }}
        whileTap={{ scale: onClick ? 0.98 : 1 }}
      >
        {cardContent}
      </motion.div>
    )
  }
  
  return cardContent
}

export const CardHeader: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => (
  <div className={`p-6 pb-4 ${className}`}>
    {children}
  </div>
)

export const CardContent: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => (
  <div className={`p-6 pt-0 ${className}`}>
    {children}
  </div>
)

export const CardFooter: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ''
}) => (
  <div className={`p-6 pt-4 ${className}`}>
    {children}
  </div>
)

export default Card