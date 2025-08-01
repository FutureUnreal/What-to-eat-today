'use client'

import React, { forwardRef } from 'react'
import { motion } from 'framer-motion'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  icon?: React.ReactNode
  variant?: 'default' | 'glass'
}

const Input = forwardRef<HTMLInputElement, InputProps>(({
  label,
  error,
  icon,
  variant = 'default',
  className = '',
  onChange,
  onFocus,
  onBlur,
  value,
  placeholder,
  type = 'text',
  disabled,
  required,
  ...otherProps
}, ref) => {
  const baseClasses = 'w-full px-4 py-3 rounded-xl border transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-1'
  
  const variantClasses = {
    default: 'bg-white border-gray-300 focus:border-blue-500 focus:ring-blue-500',
    glass: 'glass border-white/20 focus:border-blue-400 focus:ring-blue-400 placeholder-gray-500'
  }
  
  const errorClasses = error ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''
  const iconClasses = icon ? 'pl-12' : ''
  
  const classes = `${baseClasses} ${variantClasses[variant]} ${errorClasses} ${iconClasses} ${className}`
  
  return (
    <div className="space-y-2">
      {label && (
        <label className="block text-sm font-medium text-gray-700">
          {label}
        </label>
      )}
      
      <div className="relative">
        {icon && (
          <div className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400">
            {icon}
          </div>
        )}
        
        <motion.input
          ref={ref}
          className={classes}
          whileFocus={{ scale: 1.01 }}
          onChange={onChange}
          onFocus={onFocus}
          onBlur={onBlur}
          value={value}
          placeholder={placeholder}
          type={type}
          disabled={disabled}
          required={required}
        />
      </div>
      
      {error && (
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-sm text-red-600"
        >
          {error}
        </motion.p>
      )}
    </div>
  )
})

Input.displayName = 'Input'

export default Input