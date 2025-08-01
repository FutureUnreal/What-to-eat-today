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
from rag_modules.session_cache_manager import SessionCacheManager
from rag_modules.web_service_handler import WebServiceHandler
from rag_modules.recipe_recommendation import RecipeRecommendationManager

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

        # 会话缓存管理器
        self.cache_manager = None
        
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

            # 7. 会话缓存管理器
            print("初始化会话缓存管理器...")
            self.cache_manager = SessionCacheManager(
                embedding_model=self.index_module.embeddings
            )

            # 8. 菜谱推荐管理器
            print("初始化菜谱推荐管理器...")
            self.recipe_manager = RecipeRecommendationManager()

            # 9. Web服务处理器
            print("初始化Web服务处理器...")
            self.web_handler = WebServiceHandler(self)

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

    def run_web_service(self):
        """运行Web服务模式"""
        if not self.system_ready:
            print("❌ 系统未就绪，请先构建知识库")
            return

        try:
            # 使用Web服务处理器设置Flask应用
            app = self.web_handler.setup_flask_app()
            if not app:
                print("❌ Flask应用初始化失败")
                return

            print("🚀 启动Web服务...")
            print(f"📊 健康检查: http://localhost:8000/health")
            print(f"💬 聊天API: http://localhost:8000/api/chat")
            print(f"🌊 流式聊天: http://localhost:8000/api/chat/stream")
            print(f"🍽️ 菜谱推荐: http://localhost:8000/api/recipes/recommendations")
            print(f"📖 菜谱详情: http://localhost:8000/api/recipes/<recipe_id>")
            print(f"📈 统计信息: http://localhost:8000/api/stats")
            print("=" * 50)

            # 启动Flask应用
            app.run(host='0.0.0.0', port=8000, debug=False)

        except Exception as e:
            logger.error(f"Web服务启动失败: {e}")
            print(f"❌ Web服务启动失败: {e}")

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