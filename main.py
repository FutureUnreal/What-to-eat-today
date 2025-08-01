"""
基于图RAG的智能烹饪助手 - 主程序
整合传统检索和图RAG检索，实现真正的图数据优势
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, GraphRAGConfig
from rag_modules import (
    GraphDataPreparationModule,
    MilvusIndexConstructionModule, 
    GenerationIntegrationModule
)
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter, QueryAnalysis

# 加载环境变量
load_dotenv()

class AdvancedGraphRAGSystem:
    """
    图RAG系统
    
    核心特性：
    1. 智能路由：自动选择最适合的检索策略
    2. 双引擎检索：传统混合检索 + 图RAG检索
    3. 图结构推理：多跳遍历、子图提取、关系推理
    4. 查询复杂度分析：深度理解用户意图
    5. 自适应学习：基于反馈优化系统性能
    """
    
    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        # 核心模块
        self.data_module = None
        self.index_module = None
        self.generation_module = None
        
        # 检索引擎
        self.traditional_retrieval = None
        self.graph_rag_retrieval = None
        self.query_router = None
        
        # 系统状态
        self.system_ready = False

        # 🚀 会话级语义缓存系统 - 针对每个聊天窗口独立缓存
        self.session_caches = {}  # 按session_id分组的缓存：{session_id: {query: response}}
        self.session_embeddings = {}  # 按session_id分组的向量：{session_id: {query: embedding}}
        self.session_contexts = {}  # 按session_id分组的上下文：{session_id: [messages]}
        self.cache_threshold = 0.75  # 语义相似度阈值
        self.max_session_cache_size = 50  # 每个会话最大缓存条目数
        self.max_context_length = 10  # 每个会话保留的最大上下文消息数
        
    def initialize_system(self):
        """初始化高级图RAG系统"""
        logger.info("启动高级图RAG系统...")
        
        try:
            # 1. 数据准备模块
            print("初始化数据准备模块...")
            self.data_module = GraphDataPreparationModule(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )
            
            # 2. 向量索引模块
            print("初始化Milvus向量索引...")
            self.index_module = MilvusIndexConstructionModule(
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                collection_name=self.config.milvus_collection_name,
                dimension=self.config.milvus_dimension,
                model_name=self.config.embedding_model
            )
            
            # 3. 生成模块
            print("初始化生成模块...")
            self.generation_module = GenerationIntegrationModule(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # 4. 传统混合检索模块
            print("初始化传统混合检索...")
            self.traditional_retrieval = HybridRetrievalModule(
                config=self.config,
                milvus_module=self.index_module,
                data_module=self.data_module,
                llm_client=self.generation_module.client
            )
            
            # 5. 图RAG检索模块
            print("初始化图RAG检索引擎...")
            self.graph_rag_retrieval = GraphRAGRetrieval(
                config=self.config,
                llm_client=self.generation_module.client
            )
            
            # 6. 智能查询路由器
            print("初始化智能查询路由器...")
            self.query_router = IntelligentQueryRouter(
                traditional_retrieval=self.traditional_retrieval,
                graph_rag_retrieval=self.graph_rag_retrieval,
                llm_client=self.generation_module.client,
                config=self.config
            )
            
            print("✅ 高级图RAG系统初始化完成！")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise
    
    def build_knowledge_base(self):
        """构建知识库（如果需要）"""
        print("\n检查知识库状态...")
        
        try:
            # 检查Milvus集合是否存在
            if self.index_module.has_collection():
                print("✅ 发现已存在的知识库，尝试加载...")
                if self.index_module.load_collection():
                    print("知识库加载成功！")
                    
                    # 重要：即使从已存在的知识库加载，也需要加载图数据以支持图索引
                    print("加载图数据以支持图检索...")
                    self.data_module.load_graph_data()
                    print("构建菜谱文档...")
                    self.data_module.build_recipe_documents()
                    print("进行文档分块...")
                    chunks = self.data_module.chunk_documents(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap
                    )
                    
                    self._initialize_retrievers(chunks)
                    return
                else:
                    print("❌ 知识库加载失败，开始重建...")
            
            print("未找到已存在的集合，开始构建新的知识库...")
            
            # 从Neo4j加载图数据
            print("从Neo4j加载图数据...")
            self.data_module.load_graph_data()
            
            # 构建菜谱文档
            print("构建菜谱文档...")
            self.data_module.build_recipe_documents()
            
            # 进行文档分块
            print("进行文档分块...")
            chunks = self.data_module.chunk_documents(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            # 构建Milvus向量索引
            print("构建Milvus向量索引...")
            if not self.index_module.build_vector_index(chunks):
                raise Exception("构建向量索引失败")
            
            # 初始化检索器
            self._initialize_retrievers(chunks)
            
            # 显示统计信息
            self._show_knowledge_base_stats()
            
            print("✅ 知识库构建完成！")
            
        except Exception as e:
            logger.error(f"知识库构建失败: {e}")
            raise
    
    def _initialize_retrievers(self, chunks: List = None):
        """初始化检索器"""
        print("初始化检索引擎...")
        
        # 如果没有chunks，从数据模块获取
        if chunks is None:
            chunks = self.data_module.chunks or []
        
        # 初始化传统检索器
        self.traditional_retrieval.initialize(chunks)
        
        # 初始化图RAG检索器
        self.graph_rag_retrieval.initialize()
        
        self.system_ready = True
        print("✅ 检索引擎初始化完成！")
    
    def _show_knowledge_base_stats(self):
        """显示知识库统计信息"""
        print(f"\n知识库统计:")
        
        # 数据统计
        stats = self.data_module.get_statistics()
        print(f"   菜谱数量: {stats.get('total_recipes', 0)}")
        print(f"   食材数量: {stats.get('total_ingredients', 0)}")
        print(f"   烹饪步骤: {stats.get('total_cooking_steps', 0)}")
        print(f"   文档数量: {stats.get('total_documents', 0)}")
        print(f"   文本块数: {stats.get('total_chunks', 0)}")
        
        # Milvus统计
        milvus_stats = self.index_module.get_collection_stats()
        print(f"   向量索引: {milvus_stats.get('row_count', 0)} 条记录")
        
        # 图RAG统计
        route_stats = self.query_router.get_route_statistics()
        print(f"   路由统计: 总查询 {route_stats.get('total_queries', 0)} 次")
        
        if stats.get('categories'):
            categories = list(stats['categories'].keys())[:10]
            print(f"   🏷️ 主要分类: {', '.join(categories)}")
    
    def ask_question_with_routing(self, question: str, stream: bool = False, explain_routing: bool = False):
        """
        智能问答：自动选择最佳检索策略
        """
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
            
        print(f"\n❓ 用户问题: {question}")
        
        # 显示路由决策解释（可选）
        if explain_routing:
            explanation = self.query_router.explain_routing_decision(question)
            print(explanation)
        
        start_time = time.time()
        
        try:
            # 1. 智能路由检索
            print("执行智能查询路由...")
            relevant_docs, analysis = self.query_router.route_query(question, self.config.top_k)
            
            # 2. 显示路由信息
            strategy_icons = {
                "hybrid_traditional": "🔍",
                "graph_rag": "🕸️", 
                "combined": "🔄"
            }
            strategy_icon = strategy_icons.get(analysis.recommended_strategy.value, "❓")
            print(f"{strategy_icon} 使用策略: {analysis.recommended_strategy.value}")
            print(f"📊 复杂度: {analysis.query_complexity:.2f}, 关系密集度: {analysis.relationship_intensity:.2f}")
            
            # 3. 显示检索结果信息
            if relevant_docs:
                doc_info = []
                for doc in relevant_docs:
                    recipe_name = doc.metadata.get('recipe_name', '未知内容')
                    search_type = doc.metadata.get('search_type', doc.metadata.get('route_strategy', 'unknown'))
                    score = doc.metadata.get('final_score', doc.metadata.get('relevance_score', 0))
                    doc_info.append(f"{recipe_name}({search_type}, {score:.3f})")
                
                print(f"📋 找到 {len(relevant_docs)} 个相关文档: {', '.join(doc_info[:3])}")
                if len(doc_info) > 3:
                    print(f"    等 {len(relevant_docs)} 个结果...")
            else:
                return "抱歉，没有找到相关的烹饪信息。请尝试其他问题。"
            
            # 4. 生成回答
            print("🎯 智能生成回答...")
            
            if stream:
                try:
                    for chunk_text in self.generation_module.generate_adaptive_answer_stream(question, relevant_docs):
                        print(chunk_text, end="", flush=True)
                    print("\n")
                    result = "流式输出完成"
                except Exception as stream_error:
                    logger.error(f"流式输出过程中出现错误: {stream_error}")
                    print(f"\n⚠️ 流式输出中断，切换到标准模式...")
                    # 使用非流式作为后备
                    result = self.generation_module.generate_adaptive_answer(question, relevant_docs)
            else:
                result = self.generation_module.generate_adaptive_answer(question, relevant_docs)
            
            # 5. 性能统计
            end_time = time.time()
            print(f"\n⏱️ 问答完成，耗时: {end_time - start_time:.2f}秒")
            
            return result, analysis
            
        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return f"抱歉，处理问题时出现错误：{str(e)}", None

    def _get_query_embedding(self, query: str):
        """获取查询的向量表示（用于语义缓存）"""
        try:
            if hasattr(self.index_module, 'embedding_model'):
                # 使用现有的embedding模型
                return self.index_module.embedding_model.encode([query])[0]
            return None
        except Exception as e:
            logger.warning(f"获取查询向量失败: {e}")
            return None

    def _calculate_similarity(self, embedding1, embedding2):
        """计算两个向量的余弦相似度"""
        try:
            import numpy as np
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
        except:
            return 0.0

    def _check_semantic_cache(self, query: str, session_id: str = None):
        """检查会话级语义缓存中是否有相似查询"""
        if not session_id or session_id not in self.session_caches:
            return None

        session_cache = self.session_caches[session_id]
        session_embeddings = self.session_embeddings.get(session_id, {})

        if not session_cache:
            return None

        query_embedding = self._get_query_embedding(query)
        if query_embedding is None:
            return None

        # 查找最相似的缓存查询
        best_similarity = 0
        best_response = None

        for cached_query, cached_data in session_cache.items():
            cached_embedding = session_embeddings.get(cached_query)
            if cached_embedding is not None:
                similarity = self._calculate_similarity(query_embedding, cached_embedding)
                if similarity > best_similarity and similarity >= self.cache_threshold:
                    best_similarity = similarity
                    best_response = cached_data['response']

        if best_response:
            logger.info(f"🎯 会话缓存命中! Session: {session_id}, 相似度: {best_similarity:.3f}")
            return best_response

        return None

    def _add_to_semantic_cache(self, query: str, response: str, session_id: str = None):
        """将查询-答案对添加到会话级语义缓存"""
        try:
            if not session_id:
                return

            # 初始化会话缓存
            if session_id not in self.session_caches:
                self.session_caches[session_id] = {}
                self.session_embeddings[session_id] = {}

            session_cache = self.session_caches[session_id]
            session_embeddings = self.session_embeddings[session_id]

            # 限制会话缓存大小
            if len(session_cache) >= self.max_session_cache_size:
                # 删除最旧的缓存项
                oldest_key = next(iter(session_cache))
                del session_cache[oldest_key]
                del session_embeddings[oldest_key]

            query_embedding = self._get_query_embedding(query)
            if query_embedding is not None:
                session_cache[query] = {
                    'response': response,
                    'timestamp': datetime.now()
                }
                session_embeddings[query] = query_embedding
                logger.info(f"💾 已缓存到会话 {session_id}: {query[:50]}...")
        except Exception as e:
            logger.warning(f"会话缓存添加失败: {e}")

    def _add_to_context(self, session_id: str, user_message: str, ai_response: str):
        """添加消息到会话上下文"""
        try:
            if not session_id:
                return

            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = []

            context = self.session_contexts[session_id]

            # 添加用户消息和AI回复
            context.append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now()
            })
            context.append({
                'role': 'assistant',
                'content': ai_response,
                'timestamp': datetime.now()
            })

            # 限制上下文长度
            if len(context) > self.max_context_length * 2:  # *2 因为每轮对话有两条消息
                context = context[-(self.max_context_length * 2):]
                self.session_contexts[session_id] = context

            logger.info(f"📝 已添加上下文到会话 {session_id}, 当前长度: {len(context)}")
        except Exception as e:
            logger.warning(f"上下文添加失败: {e}")

    def _get_context_for_query(self, session_id: str, current_query: str):
        """获取会话上下文，用于增强当前查询"""
        try:
            if not session_id or session_id not in self.session_contexts:
                return current_query

            context = self.session_contexts[session_id]
            if not context:
                return current_query

            # 构建包含上下文的查询
            context_text = ""
            for msg in context[-6:]:  # 只取最近3轮对话
                role = "用户" if msg['role'] == 'user' else "助手"
                context_text += f"{role}: {msg['content'][:100]}...\n"

            enhanced_query = f"基于以下对话上下文回答问题：\n{context_text}\n当前问题: {current_query}"
            logger.info(f"🔗 已为会话 {session_id} 添加上下文，查询长度: {len(enhanced_query)}")
            return enhanced_query

        except Exception as e:
            logger.warning(f"上下文获取失败: {e}")
            return current_query

    def run_web_service(self):
        """运行Web服务模式"""
        if not self.system_ready:
            print("❌ 系统未就绪，请先构建知识库")
            return
            
        try:
            from flask import Flask, request, jsonify, Response
            from flask_cors import CORS
            import json
            
            app = Flask(__name__)
            CORS(app, origins=["http://localhost", "http://localhost:3000", "http://127.0.0.1:3000"],
                 methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
                 allow_headers=['Content-Type', 'Authorization'])

            # 添加静态文件服务
            @app.route('/static/<path:filename>')
            def serve_static(filename):
                """提供静态文件服务"""
                import os
                from flask import send_from_directory

                # 安全检查，防止路径遍历攻击
                if '..' in filename or filename.startswith('/'):
                    return "Invalid path", 400

                file_path = os.path.join('.', filename)
                if os.path.exists(file_path):
                    directory = os.path.dirname(file_path)
                    filename_only = os.path.basename(file_path)
                    return send_from_directory(directory, filename_only)
                else:
                    return "File not found", 404


            
            @app.route('/health', methods=['GET'])
            def health_check():
                return jsonify({"status": "healthy", "system_ready": self.system_ready})
            
            @app.route('/api/chat', methods=['POST'])
            def chat():
                try:
                    data = request.get_json()
                    query = data.get('message', '')
                    
                    if not query:
                        return jsonify({"error": "消息不能为空"}), 400

                    # 获取会话ID（如果没有则生成一个）
                    import time as time_module
                    session_id = data.get('session_id', f"session_{int(time_module.time())}")

                    # 🚀 首先检查会话级语义缓存
                    cached_response = self._check_semantic_cache(query, session_id)
                    if cached_response:
                        # 即使是缓存命中，也要添加到上下文
                        self._add_to_context(session_id, query, cached_response)
                        return jsonify({
                            "response": cached_response,
                            "query": query,
                            "session_id": session_id,
                            "timestamp": str(datetime.now()),
                            "from_cache": True
                        })

                    # 🔗 获取上下文增强的查询
                    enhanced_query = self._get_context_for_query(session_id, query)

                    # 缓存未命中，执行完整的RAG流程
                    documents, analysis = self.query_router.route_query(
                        query=enhanced_query,
                        top_k=self.config.top_k
                    )
                    # 使用生成模块生成最终答案
                    response = self.generation_module.generate_adaptive_answer(enhanced_query, documents)

                    # 将结果添加到会话缓存和上下文
                    self._add_to_semantic_cache(query, response, session_id)
                    self._add_to_context(session_id, query, response)
                    
                    return jsonify({
                        "response": response,
                        "query": query,
                        "timestamp": str(datetime.now())
                    })
                    
                except Exception as e:
                    logger.error(f"Chat API错误: {e}")
                    return jsonify({"error": str(e)}), 500
            
            @app.route('/api/chat/stream', methods=['POST'])
            def chat_stream():
                try:
                    data = request.get_json()
                    query = data.get('message', '')
                    session_id = data.get('session_id', '')

                    if not query:
                        return jsonify({"error": "消息不能为空"}), 400

                    def generate():
                        try:
                            # 获取会话ID
                            import time as time_module
                            session_id = data.get('session_id', f"session_{int(time_module.time())}")

                            # 🚀 首先检查会话级语义缓存
                            cached_response = self._check_semantic_cache(query, session_id)
                            if cached_response:
                                # 缓存命中，快速返回
                                self._add_to_context(session_id, query, cached_response)
                                import json
                                chunk_size = 3
                                for i in range(0, len(cached_response), chunk_size):
                                    chunk = cached_response[i:i+chunk_size]
                                    data_obj = {"chunk": chunk, "from_cache": True}
                                    yield f"data: {json.dumps(data_obj)}\n\n"
                                    time.sleep(0.02)  # 更快的流式响应
                                yield f"data: [DONE]\n\n"
                                return

                            # 🔗 获取上下文增强的查询
                            enhanced_query = self._get_context_for_query(session_id, query)

                            # 缓存未命中，执行完整的RAG流程
                            documents, analysis = self.query_router.route_query(
                                query=enhanced_query,
                                top_k=self.config.top_k
                            )

                            # 🚀 使用真正的流式生成
                            import json
                            full_response = ""

                            for chunk in self.generation_module.generate_adaptive_answer_stream(enhanced_query, documents):
                                full_response += chunk
                                data_obj = {"chunk": chunk}
                                yield f"data: {json.dumps(data_obj)}\n\n"

                            # 将完整结果添加到会话缓存和上下文
                            self._add_to_semantic_cache(query, full_response, session_id)
                            self._add_to_context(session_id, query, full_response)

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
                    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                    return response

                except Exception as e:
                    logger.error(f"Stream API错误: {e}")
                    return jsonify({"error": str(e)}), 500
            
            @app.route('/api/stats', methods=['GET'])
            def get_stats():
                try:
                    stats = self.data_module.get_statistics()
                    return jsonify(stats)
                except Exception as e:
                    return jsonify({"error": str(e)}), 500

            # 菜谱详情API
            @app.route('/api/recipes/<recipe_id>', methods=['GET'])
            def get_recipe_detail(recipe_id):
                """获取菜谱详情"""
                try:
                    recipe_detail = self._get_recipe_detail_from_db(recipe_id)

                    if recipe_detail:
                        return jsonify({
                            "success": True,
                            "data": recipe_detail,
                            "message": "获取菜谱详情成功"
                        })
                    else:
                        return jsonify({
                            "success": False,
                            "message": "菜谱不存在",
                            "data": None
                        }), 404

                except Exception as e:
                    logger.error(f"获取菜谱详情失败: {e}")
                    return jsonify({
                        "success": False,
                        "message": f"获取菜谱详情失败: {str(e)}",
                        "data": None
                    }), 500

            @app.route('/api/recipes/recommendations', methods=['POST'])
            def get_recommendations():
                try:
                    data = request.get_json() or {}

                    # 兼容不同的请求格式
                    query = data.get('query', '推荐一些菜谱')
                    limit = data.get('limit', 3)
                    preferences = data.get('preferences', {})

                    # 如果有用户偏好，构建更具体的查询
                    if preferences:
                        dietary_restrictions = preferences.get('dietaryRestrictions', [])
                        favorite_cuisines = preferences.get('favoriteCuisines', [])
                        cooking_skill = preferences.get('cookingSkill', 'beginner')

                        if dietary_restrictions or favorite_cuisines:
                            query_parts = ["推荐一些"]
                            if favorite_cuisines:
                                query_parts.append(f"{','.join(favorite_cuisines)}")
                            if dietary_restrictions:
                                query_parts.append(f"适合{','.join(dietary_restrictions)}的")
                            if cooking_skill == 'beginner':
                                query_parts.append("简单易做的")
                            query_parts.append("菜谱")
                            query = "".join(query_parts)

                    # 获取有图片的随机推荐菜谱
                    recipes = self._get_random_recipes_with_images(limit)

                    return jsonify({
                        "success": True,
                        "data": recipes,
                        "total": len(recipes),
                        "query": query
                    })
                    
                except Exception as e:
                    logger.error(f"获取推荐失败: {e}")
                    return jsonify({"error": str(e)}), 500
            


            
            print("🚀 启动Web服务...")
            print(f"📊 健康检查: http://localhost:8000/health")
            print(f"💬 聊天API: http://localhost:8000/api/chat")
            print(f"🌊 流式聊天: http://localhost:8000/api/chat/stream")
            print(f"🍽️ 菜谱推荐: http://localhost:8000/api/recipes/recommendations")
            print(f"📖 菜谱详情: http://localhost:8000/api/recipes/<recipe_id>")
            print(f"📈 统计信息: http://localhost:8000/api/stats")
            
            # 启动Flask应用
            app.run(host='0.0.0.0', port=8000, debug=False)
            
        except ImportError:
            print("❌ Flask未安装，无法启动Web服务")
            print("请运行: pip install flask")
        except Exception as e:
            logger.error(f"Web服务启动失败: {e}")
            print(f"❌ Web服务启动失败: {e}")
    
    def _get_featured_recipes_from_db(self, limit=6):
        """从图数据库获取精选推荐菜谱"""
        try:
            if not hasattr(self, 'graph_rag_retrieval') or not self.graph_rag_retrieval.driver:
                logger.warning("图数据库连接不可用，返回空结果")
                return []

            with self.graph_rag_retrieval.driver.session() as session:
                # 查询评分较高、比较受欢迎的菜谱
                cypher_query = """
                MATCH (r:Recipe)
                WHERE r.nodeId >= '200000000'
                OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:Category)
                WITH r, c
                RETURN
                    r.nodeId as id,
                    r.name as name,
                    COALESCE(r.description, '美味可口的经典菜谱') as description,
                    COALESCE(c.name, r.category, '家常菜') as category,
                    COALESCE(r.difficulty, '★★★') as difficulty_stars,
                    COALESCE(r.cookingTime, 30) as cookingTime,
                    COALESCE(r.prepTime, 15) as prepTime,
                    COALESCE(r.servings, 2) as servings,
                    COALESCE(r.tags, []) as tags,
                    COALESCE(r.rating, 4.5) as rating
                ORDER BY
                    CASE WHEN r.rating IS NOT NULL THEN r.rating ELSE 4.5 END DESC,
                    r.name
                LIMIT $limit
                """

                result = session.run(cypher_query, {"limit": limit})
                recipes = []

                for record in result:
                    # 转换难度星级为前端期望的格式
                    difficulty_stars = record.get('difficulty_stars', '★★★')
                    star_count = difficulty_stars.count('★')
                    if star_count <= 2:
                        difficulty = 'easy'
                    elif star_count <= 3:
                        difficulty = 'medium'
                    else:
                        difficulty = 'hard'

                    recipe = {
                        "id": record.get('id'),
                        "name": record.get('name'),
                        "description": record.get('description'),
                        "category": record.get('category'),
                        "imageUrl": f"https://via.placeholder.com/300x200?text={record.get('name', 'Recipe')}",
                        "cookingTime": int(record.get('cookingTime', 30)),
                        "prepTime": int(record.get('prepTime', 15)),
                        "servings": int(record.get('servings', 2)),
                        "difficulty": difficulty,
                        "rating": float(record.get('rating', 4.5)),
                        "tags": record.get('tags', []),
                        "ingredients": [],
                        "steps": [],
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-01-01T00:00:00Z"
                    }
                    recipes.append(recipe)

                logger.info(f"从数据库获取到 {len(recipes)} 个推荐菜谱")
                return recipes

        except Exception as e:
            logger.error(f"从数据库获取推荐菜谱失败: {e}")
            return []

    def _get_fallback_recommendations(self, limit=6):
        """备用推荐菜谱（当数据库查询失败时使用）"""
        fallback_recipes = [
            {
                "id": "fallback_001",
                "name": "红烧肉",
                "description": "肥瘦相间，入口即化的经典家常菜",
                "category": "家常菜",
                "imageUrl": "https://via.placeholder.com/300x200?text=红烧肉",
                "cookingTime": 60,
                "prepTime": 15,
                "servings": 4,
                "difficulty": "medium",
                "rating": 4.8,
                "tags": ["家常菜", "下饭", "经典"],
                "ingredients": [],
                "steps": [],
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z"
            },
            {
                "id": "fallback_002",
                "name": "西红柿鸡蛋",
                "description": "酸甜开胃，营养丰富的国民菜",
                "category": "家常菜",
                "imageUrl": "https://via.placeholder.com/300x200?text=西红柿鸡蛋",
                "cookingTime": 15,
                "prepTime": 10,
                "servings": 2,
                "difficulty": "easy",
                "rating": 4.6,
                "tags": ["简单", "营养", "下饭"],
                "ingredients": [],
                "steps": [],
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z"
            }
        ]
        return fallback_recipes[:limit]

    def _get_random_recipes_with_images(self, limit=3):
        """从预生成的索引文件获取随机的有图片的菜谱推荐"""
        try:
            import json
            import random
            import os

            # 读取预生成的菜谱索引文件
            index_file = "data/recipes_with_images.json"

            if not os.path.exists(index_file):
                logger.warning(f"菜谱索引文件不存在: {index_file}")
                return self._get_fallback_recommendations(limit)

            with open(index_file, 'r', encoding='utf-8') as f:
                recipes_data = json.load(f)

            if not recipes_data:
                logger.warning("菜谱索引文件为空")
                return self._get_fallback_recommendations(limit)

            # 转换为API格式
            recipes_with_images = []
            for i, recipe_data in enumerate(recipes_data):
                # 生成随机难度
                difficulties = ['easy', 'medium', 'hard']
                difficulty = random.choice(difficulties)

                # 处理图片URL
                image_url = recipe_data.get('image_url', '')
                if image_url and not image_url.startswith('http'):
                    # 如果是相对路径，转换为绝对路径
                    file_path = recipe_data.get('file_path', '')
                    if file_path:
                        # 获取文件所在目录
                        import os
                        file_dir = os.path.dirname(file_path)
                        # 处理相对路径（去掉 ./ 前缀）
                        if image_url.startswith('./'):
                            image_url = image_url[2:]
                        # 构建GitHub LFS媒体URL
                        # 从file_path中提取dishes目录后的路径
                        # 例如: "data\\dishes\\vegetable_dish\\鸡蛋羹\\微波炉鸡蛋羹.md" -> "dishes/vegetable_dish/鸡蛋羹/"
                        if 'dishes' in file_path:
                            # 找到dishes的位置，提取dishes后面的路径
                            dishes_index = file_path.find('dishes')
                            if dishes_index != -1:
                                # 提取从dishes开始到文件名之前的路径
                                path_after_dishes = file_path[dishes_index:].replace('\\', '/')
                                # 移除文件名，只保留目录路径
                                dir_path = '/'.join(path_after_dishes.split('/')[:-1])
                                github_path = f"{dir_path}/{image_url}"
                            else:
                                github_path = image_url
                        else:
                            github_path = image_url

                        # 使用您fork的HowToCook仓库的GitHub LFS媒体URL
                        image_url = f"https://media.githubusercontent.com/media/FutureUnreal/HowToCook/master/{github_path}"
                        logger.info(f"转换后的GitHub图片URL: {image_url}")

                recipe = {
                    "id": f"recipe_{i + 1}",
                    "name": recipe_data.get('name', '未知菜谱'),
                    "description": recipe_data.get('description', '美味可口的经典菜谱'),
                    "category": recipe_data.get('category', '家常菜'),
                    "imageUrl": image_url or f"https://via.placeholder.com/300x200?text={recipe_data.get('name', 'Recipe')}",
                    "cookingTime": recipe_data.get('cooking_time', 30),
                    "prepTime": 15,
                    "servings": 2,
                    "difficulty": difficulty,
                    "tags": recipe_data.get('tags', []),
                    "ingredients": [],
                    "steps": [],
                    "markdownPath": recipe_data.get('file_path', ''),
                    "createdAt": "2024-01-01T00:00:00Z",
                    "updatedAt": "2024-01-01T00:00:00Z"
                }
                recipes_with_images.append(recipe)

            # 随机选择指定数量的菜谱
            if len(recipes_with_images) >= limit:
                selected_recipes = random.sample(recipes_with_images, limit)
            else:
                selected_recipes = recipes_with_images[:limit]
                # 如果不够，用备用数据补充
                if len(selected_recipes) < limit:
                    fallback = self._get_fallback_recommendations(limit - len(selected_recipes))
                    selected_recipes.extend(fallback)

            logger.info(f"从索引文件加载 {len(recipes_with_images)} 个菜谱，返回 {len(selected_recipes)} 个")
            return selected_recipes

        except Exception as e:
            logger.error(f"从索引文件获取菜谱失败: {e}")
            return self._get_fallback_recommendations(limit)



    def _get_recipe_detail_from_db(self, recipe_id):
        """从索引文件和原始文件获取菜谱详情"""
        try:
            import json
            import os

            # 首先尝试从索引文件获取菜谱信息
            index_file = "data/recipes_with_images.json"

            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    recipes_data = json.load(f)

                # 查找对应的菜谱（recipe_id格式为 recipe_N）
                recipe_index = None
                if recipe_id.startswith('recipe_'):
                    try:
                        recipe_index = int(recipe_id.split('_')[1]) - 1
                    except (ValueError, IndexError):
                        pass

                if recipe_index is not None and 0 <= recipe_index < len(recipes_data):
                    recipe_data = recipes_data[recipe_index]

                    # 读取Markdown文件内容
                    markdown_content = None
                    file_path = recipe_data.get('file_path', '')
                    if file_path and os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            markdown_content = f.read()

                    # 处理图片URL - 使用GitHub Raw URL
                    image_url = recipe_data.get('image_url', '')
                    if image_url and not image_url.startswith('http'):
                        if file_path:
                            file_dir = os.path.dirname(file_path)
                            if image_url.startswith('./'):
                                image_url = image_url[2:]
                            # 构建GitHub LFS媒体URL
                            # 从file_path中提取dishes目录后的路径
                            if 'dishes' in file_path:
                                # 找到dishes的位置，提取dishes后面的路径
                                dishes_index = file_path.find('dishes')
                                if dishes_index != -1:
                                    # 提取从dishes开始到文件名之前的路径
                                    path_after_dishes = file_path[dishes_index:].replace('\\', '/')
                                    # 移除文件名，只保留目录路径
                                    dir_path = '/'.join(path_after_dishes.split('/')[:-1])
                                    github_path = f"{dir_path}/{image_url}"
                                else:
                                    github_path = image_url
                            else:
                                github_path = image_url

                            # 使用您fork的HowToCook仓库的GitHub LFS媒体URL
                            github_raw_url = f"https://media.githubusercontent.com/media/FutureUnreal/HowToCook/master/{github_path}"
                            image_url = github_raw_url
                            logger.info(f"详情页GitHub图片URL: {image_url}")

                    # 构建详情数据
                    recipe_detail = {
                        "id": recipe_id,
                        "name": recipe_data.get('name', '未知菜谱'),
                        "description": recipe_data.get('description', '美味可口的经典菜谱'),
                        "category": recipe_data.get('category', '家常菜'),
                        "imageUrl": image_url or f"https://via.placeholder.com/600x400?text={recipe_data.get('name', 'Recipe')}",
                        "cookingTime": recipe_data.get('cooking_time', 30),
                        "prepTime": 15,
                        "servings": 2,
                        "difficulty": "medium",
                        "tags": recipe_data.get('tags', []),
                        "ingredients": [],  # 可以从Markdown解析
                        "steps": [],  # 可以从Markdown解析
                        "markdownContent": markdown_content,
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-01-01T00:00:00Z"
                    }

                    return recipe_detail

            # 如果索引文件方法失败，返回None
            logger.warning(f"无法找到菜谱: {recipe_id}")
            return None

        except Exception as e:
            logger.error(f"获取菜谱详情失败: {e}")
            return None

    def _read_recipe_markdown(self, recipe_name):
        """读取菜谱的原始Markdown文件"""
        try:
            import os
            import glob

            # 在data/dishes目录中搜索匹配的Markdown文件
            dishes_dir = "data/dishes"
            if not os.path.exists(dishes_dir):
                return None

            # 搜索包含菜谱名称的Markdown文件
            for root, dirs, files in os.walk(dishes_dir):
                for file in files:
                    if file.endswith('.md') and recipe_name in file:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            return f.read()

            return None

        except Exception as e:
            logger.error(f"读取菜谱Markdown文件失败: {e}")
            return None

    def _extract_image_from_markdown(self, markdown_content):
        """从Markdown内容中提取图片URL"""
        if not markdown_content:
            return None

        try:
            import re
            # 匹配Markdown图片语法 ![alt](url)
            image_pattern = r'!\[.*?\]\((.*?)\)'
            matches = re.findall(image_pattern, markdown_content)

            if matches:
                # 返回第一个找到的图片URL
                return matches[0]

            return None

        except Exception as e:
            logger.error(f"提取图片URL失败: {e}")
            return None

    def _cleanup(self):
        """清理资源"""
        if self.data_module:
            self.data_module.close()
        if self.traditional_retrieval:
            self.traditional_retrieval.close()
        if self.graph_rag_retrieval:
            self.graph_rag_retrieval.close()
        if self.index_module:
            self.index_module.close()

def main():
    """主函数"""
    try:
        print("启动高级图RAG系统...")
        
        # 创建高级图RAG系统
        rag_system = AdvancedGraphRAGSystem()
        
        # 初始化系统
        rag_system.initialize_system()
        
        # 构建知识库
        rag_system.build_knowledge_base()
        
        # 启动Web服务（Docker环境）
        rag_system.run_web_service()
        
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 系统错误: {e}")

if __name__ == "__main__":
    main() 