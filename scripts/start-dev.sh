#!/bin/bash

# 开发环境启动脚本
set -e

echo "🚀 启动今天吃什么开发环境..."

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker 未运行，请先启动 Docker"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "❌ Docker Compose 未安装"
    exit 1
fi

# 创建必要的目录
mkdir -p data/cypher
mkdir -p nginx/ssl

# 启动基础服务 (Neo4j, Milvus)
echo "📊 启动数据库服务..."
docker-compose -f docker-compose.dev.yml up -d neo4j milvus-etcd milvus-minio milvus-standalone

# 等待数据库服务启动
echo "⏳ 等待数据库服务启动..."
sleep 30

# 检查 Neo4j 连接
echo "🔍 检查 Neo4j 连接..."
max_retries=30
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if docker exec what-to-eat-neo4j-dev cypher-shell -u neo4j -p all-in-rag "RETURN 1" > /dev/null 2>&1; then
        echo "✅ Neo4j 连接成功"
        break
    fi
    
    retry_count=$((retry_count + 1))
    echo "⏳ 等待 Neo4j 启动... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "❌ Neo4j 启动超时"
    exit 1
fi

# 检查 Milvus 连接
echo "🔍 检查 Milvus 连接..."
max_retries=30
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if curl -f http://localhost:9091/healthz > /dev/null 2>&1; then
        echo "✅ Milvus 连接成功"
        break
    fi
    
    retry_count=$((retry_count + 1))
    echo "⏳ 等待 Milvus 启动... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "❌ Milvus 启动超时"
    exit 1
fi

# 导入初始数据 (如果存在)
if [ -f "data/cypher/neo4j_import.cypher" ]; then
    echo "📥 导入 Neo4j 初始数据..."
    docker exec what-to-eat-neo4j-dev cypher-shell -u neo4j -p all-in-rag -f /import/neo4j_import.cypher || echo "⚠️  数据导入失败或数据已存在"
fi

# 启动后端服务
echo "🐍 启动 Python 后端..."
docker-compose -f docker-compose.dev.yml up -d backend

# 等待后端服务启动
echo "⏳ 等待后端服务启动..."
sleep 10

# 检查后端服务
max_retries=30
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "✅ 后端服务启动成功"
        break
    fi
    
    retry_count=$((retry_count + 1))
    echo "⏳ 等待后端服务启动... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "❌ 后端服务启动超时"
    exit 1
fi

echo "🎉 开发环境启动完成！"
echo ""
echo "📋 服务访问地址："
echo "   - Neo4j Browser: http://localhost:7474 (neo4j/all-in-rag)"
echo "   - Milvus Console: http://localhost:9001 (minioadmin/minioadmin)"
echo "   - Python 后端: http://localhost:8000"
echo ""
echo "🔧 前端开发："
echo "   cd frontend && npm install && npm run dev"
echo "   前端地址: http://localhost:3000"
echo ""
echo "📝 查看日志："
echo "   docker-compose -f docker-compose.dev.yml logs -f [service_name]"
echo ""
echo "🛑 停止服务："
echo "   docker-compose -f docker-compose.dev.yml down"