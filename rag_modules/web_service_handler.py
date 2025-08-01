"""
Web服务处理模块
负责处理Web API和静态文件服务
"""

import logging
import json
import time
import concurrent.futures
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WebServiceHandler:
    """
    Web服务处理器
    
    功能：
    1. API路由处理
    2. 静态文件服务
    3. 错误处理
    4. 响应格式化
    """
    
    def __init__(self, rag_system):
        """初始化Web服务处理器"""
        self.rag_system = rag_system
        self.app = None
    
    def setup_flask_app(self):
        """设置Flask应用和路由"""
        try:
            from flask import Flask, request, jsonify, Response
            from flask_cors import CORS
            
            self.app = Flask(__name__)
            CORS(self.app)
            
            # 设置路由
            self._setup_routes()
            
            return self.app
            
        except ImportError as e:
            logger.error(f"Flask导入失败: {e}")
            return None
    
    def _setup_routes(self):
        """设置所有API路由"""
        from flask import request, jsonify, Response, send_from_directory
        
        @self.app.route('/')
        def serve_index():
            """提供主页"""
            return self._serve_static_file('index.html')
        
        @self.app.route('/<path:filename>')
        def serve_static(filename):
            """提供静态文件服务"""
            return self._serve_static_file(filename)
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查端点"""
            return jsonify({
                "status": "healthy",
                "timestamp": str(datetime.now()),
                "service": "RAG System"
            })
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            """聊天API - 普通响应"""
            return self._handle_chat_request()
        
        @self.app.route('/api/chat/stream', methods=['POST'])
        def chat_stream():
            """聊天API - 流式响应"""
            return self._handle_stream_request()
        
        @self.app.route('/api/recipes/recommendations', methods=['POST'])
        def get_recommendations():
            """获取菜谱推荐"""
            return self._handle_recommendations_request()
        
        @self.app.route('/api/recipes/<recipe_id>', methods=['GET'])
        def get_recipe_detail(recipe_id):
            """获取菜谱详情"""
            return self._handle_recipe_detail_request(recipe_id)
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """获取系统统计信息"""
            return self._handle_stats_request()
    
    def _serve_static_file(self, filename):
        """提供静态文件服务"""
        import os
        from flask import send_from_directory
        
        # 安全检查，防止路径遍历攻击
        if '..' in filename or filename.startswith('/'):
            return "Forbidden", 403
        
        # 前端文件路径
        frontend_path = os.path.join(os.getcwd(), 'frontend', 'dist')
        
        try:
            if filename == 'index.html' or filename == '':
                return send_from_directory(frontend_path, 'index.html')
            else:
                return send_from_directory(frontend_path, filename)
        except FileNotFoundError:
            # 如果文件不存在，返回index.html（用于SPA路由）
            return send_from_directory(frontend_path, 'index.html')
    
    def _handle_chat_request(self):
        """处理普通聊天请求"""
        from flask import request, jsonify
        
        try:
            data = request.get_json()
            query = data.get('message', '')
            session_id = data.get('session_id', '')
            
            if not query:
                return jsonify({"error": "消息不能为空"}), 400
            
            # 🚀 并行执行缓存检查和预处理
            cached_response = None
            enhanced_query = query
            
            def check_cache():
                nonlocal cached_response
                cached_response = self.rag_system.cache_manager.check_semantic_cache(query, session_id)
            
            def prepare_query():
                nonlocal enhanced_query
                enhanced_query = self.rag_system.cache_manager.get_context_for_query(session_id, query)
            
            # 并行执行缓存检查和查询预处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_cache = executor.submit(check_cache)
                future_query = executor.submit(prepare_query)
                
                # 等待缓存检查完成
                concurrent.futures.wait([future_cache], timeout=1)
                
                if cached_response:
                    # 缓存命中，取消查询预处理
                    future_query.cancel()
                    self.rag_system.cache_manager.add_to_context(session_id, query, cached_response)
                    return jsonify({
                        "response": cached_response,
                        "query": query,
                        "session_id": session_id,
                        "timestamp": str(datetime.now()),
                        "from_cache": True
                    })
                
                # 缓存未命中，等待查询预处理完成
                concurrent.futures.wait([future_query], timeout=2)
            
            # 缓存未命中，执行完整的RAG流程
            documents, analysis = self.rag_system.query_router.route_query(
                query=enhanced_query,
                top_k=self.rag_system.config.top_k
            )
            # 使用生成模块生成最终答案
            response = self.rag_system.generation_module.generate_adaptive_answer(enhanced_query, documents)
            
            # 将结果添加到会话缓存和上下文
            self.rag_system.cache_manager.add_to_semantic_cache(query, response, session_id)
            self.rag_system.cache_manager.add_to_context(session_id, query, response)
            
            return jsonify({
                "response": response,
                "query": query,
                "timestamp": str(datetime.now())
            })
            
        except Exception as e:
            logger.error(f"Chat API错误: {e}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_stream_request(self):
        """处理流式聊天请求"""
        from flask import request, Response
        
        try:
            data = request.get_json()
            query = data.get('message', '')
            session_id = data.get('session_id', '')
            
            if not query:
                return jsonify({"error": "消息不能为空"}), 400
            
            def generate():
                try:
                    # 🚀 并行执行缓存检查和预处理
                    cached_response = None
                    enhanced_query = query
                    
                    def check_cache():
                        nonlocal cached_response
                        cached_response = self.rag_system.cache_manager.check_semantic_cache(query, session_id)
                    
                    def prepare_query():
                        nonlocal enhanced_query
                        enhanced_query = self.rag_system.cache_manager.get_context_for_query(session_id, query)
                    
                    # 并行执行缓存检查和查询预处理
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        future_cache = executor.submit(check_cache)
                        future_query = executor.submit(prepare_query)
                        
                        # 等待缓存检查完成
                        concurrent.futures.wait([future_cache], timeout=1)
                        
                        if cached_response:
                            # 缓存命中，快速返回
                            future_query.cancel()
                            self.rag_system.cache_manager.add_to_context(session_id, query, cached_response)
                            chunk_size = 3
                            for i in range(0, len(cached_response), chunk_size):
                                chunk = cached_response[i:i+chunk_size]
                                data_obj = {"chunk": chunk, "from_cache": True}
                                yield f"data: {json.dumps(data_obj)}\n\n"
                                time.sleep(0.02)  # 更快的流式响应
                            yield f"data: [DONE]\n\n"
                            return
                        
                        # 缓存未命中，等待查询预处理完成
                        concurrent.futures.wait([future_query], timeout=2)
                    
                    # 缓存未命中，执行完整的RAG流程
                    documents, analysis = self.rag_system.query_router.route_query(
                        query=enhanced_query,
                        top_k=self.rag_system.config.top_k
                    )
                    
                    # 流式生成答案
                    full_response = ""
                    for chunk in self.rag_system.generation_module.generate_adaptive_answer_stream(enhanced_query, documents):
                        full_response += chunk
                        data_obj = {"chunk": chunk}
                        yield f"data: {json.dumps(data_obj)}\n\n"
                    
                    # 将完整结果添加到会话缓存和上下文
                    self.rag_system.cache_manager.add_to_semantic_cache(query, full_response, session_id)
                    self.rag_system.cache_manager.add_to_context(session_id, query, full_response)
                    
                    # 发送结束标记
                    yield f"data: [DONE]\n\n"
                
                except Exception as e:
                    logger.error(f"Stream API错误: {e}")
                    error_msg = f"抱歉，处理您的问题时出现错误：{str(e)}"
                    data_obj = {"chunk": error_msg}
                    yield f"data: {json.dumps(data_obj)}\n\n"
                    yield f"data: [DONE]\n\n"
            
            response = Response(generate(), mimetype='text/event-stream')
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
            
        except Exception as e:
            logger.error(f"Stream API错误: {e}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_recommendations_request(self):
        """处理菜谱推荐请求"""
        from flask import request, jsonify
        
        try:
            data = request.get_json() or {}
            preferences = data.get('preferences', {})
            
            # 获取推荐菜谱
            recipes = self.rag_system.recipe_manager.get_random_recipes_with_images(limit=3)
            
            return jsonify({
                "success": True,
                "data": recipes,
                "message": "推荐获取成功"
            })
            
        except Exception as e:
            logger.error(f"推荐API错误: {e}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_recipe_detail_request(self, recipe_id):
        """处理菜谱详情请求"""
        from flask import jsonify
        
        try:
            recipe = self.rag_system.recipe_manager.get_recipe_by_id(recipe_id)
            if recipe:
                return jsonify({
                    "success": True,
                    "data": recipe
                })
            else:
                return jsonify({"error": "菜谱不存在"}), 404
                
        except Exception as e:
            logger.error(f"菜谱详情API错误: {e}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_stats_request(self):
        """处理统计信息请求"""
        from flask import jsonify
        
        try:
            # 获取系统统计信息
            stats = {
                "cache_stats": self.rag_system.cache_manager.get_session_stats(),
                "route_stats": self.rag_system.query_router.get_route_statistics(),
                "system_info": {
                    "timestamp": str(datetime.now()),
                    "status": "running"
                }
            }
            return jsonify(stats)
            
        except Exception as e:
            logger.error(f"统计API错误: {e}")
            return jsonify({"error": str(e)}), 500
