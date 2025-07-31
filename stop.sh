#!/bin/bash

# 今天吃什么 - 统一停止脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo
    print_message $CYAN "==============================================="
    print_message $WHITE "      今天吃什么 - 服务停止"
    print_message $CYAN "==============================================="
    echo
}

print_header

echo "💾 数据保留选项："
echo "   1. 停止服务但保留数据（推荐）"
echo "   2. 停止服务并清理所有数据"
echo

read -p "请选择 (1/2，默认为1): " choice

case $choice in
    2)
        echo
        print_message $RED "⚠️  警告：这将删除所有数据，包括："
        echo "   - Neo4j 图数据库数据"
        echo "   - Milvus 向量数据库数据"
        echo "   - 所有聊天记录和用户偏好"
        echo
        read -p "确认删除所有数据？(y/N): " confirm
        
        if [[ $confirm =~ ^[Yy]$ ]]; then
            print_message $BLUE "🗑️  正在清理所有数据..."
            docker-compose down -v
            print_message $GREEN "✅ 数据清理完成"
        else
            print_message $YELLOW "ℹ️  已取消数据清理，仅停止服务"
            docker-compose down
        fi
        ;;
    *)
        print_message $YELLOW "ℹ️  保留数据，仅停止服务"
        docker-compose down
        ;;
esac

# 清理未使用的Docker资源
print_message $BLUE "🧹 清理未使用的Docker资源..."
docker system prune -f

echo
print_message $GREEN "✅ 服务已停止"
echo
print_message $CYAN "🔄 下次启动："
echo "   ./start.sh (Linux/macOS)"
echo "   start.bat (Windows)"