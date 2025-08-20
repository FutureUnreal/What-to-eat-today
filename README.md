# 🍽️ 今天吃什么 - AI美食推荐助手

本项目是基于 [Datawhale/all-in-rag](https://github.com/datawhalechina/all-in-rag) 教程的完整实战案例，展示了如何使用图RAG技术构建智能烹饪助手，为您推荐个性化美食和详细烹饪指导。

![界面](./view.png)

## ✨ 特性

- 🤖 **智能推荐**: 基于图RAG + 向量检索的双重AI推荐
- 🍳 **详细指导**: 分步骤烹饪指南，新手也能轻松上手
- 💬 **对话交互**: ChatGPT风格的自然语言交互
- 🎨 **现代界面**: 玻璃质感的响应式设计
- 📱 **跨平台**: 支持桌面和移动设备访问

## 🚀 快速开始

### 📋 前置要求

✅ **Docker Desktop** - [下载安装](https://www.docker.com/products/docker-desktop/)
✅ **Node.js 18+** - [下载安装](https://nodejs.org/) (可选，仅开发时需要)

### 🎯 一键启动

**Windows 用户（推荐）:**
```bash
# 1. 克隆项目
git clone https://github.com/FutureUnreal/What-to-eat-today.git
cd What-to-eat-today

# 2. 双击运行或命令行启动
start.bat
```

**Linux/macOS 用户:**
```bash
# 1. 克隆项目
git clone https://github.com/FutureUnreal/What-to-eat-today.git
cd What-to-eat-today

# 2. 给脚本执行权限并启动
chmod +x start.sh stop.sh
./start.sh
```

### 🌐 访问应用

启动完成后，脚本会自动打开浏览器，或手动访问：

- **🏠 应用首页**: http://localhost
- **⚛️ 前端**: http://localhost:3000
- **🐍 后端API**: http://localhost:8000
- **📊 Neo4j**: http://localhost:7474 (neo4j/all-in-rag)
- **🗄️ Milvus**: http://localhost:9001 (minioadmin/minioadmin)

## 🛑 停止服务

```bash
# 停止所有服务
stop.bat        # Windows
./stop.sh       # Linux/macOS

# 或者直接使用 Docker Compose
docker-compose down

# 完全重置（清除所有数据）
docker-compose down -v
```

## 📝 常用命令

```bash
# 查看所有服务状态
docker-compose ps

# 查看特定服务日志
docker-compose logs -f backend
docker-compose logs -f frontend

# 重启特定服务
docker-compose restart backend

# 完全重置（清除所有数据）
docker-compose down -v
```

## 🏗️ 架构说明

### 技术栈

**前端:**
- Next.js 14 (React框架)
- Tailwind CSS (样式框架)
- Zustand (状态管理)
- Framer Motion (动画)

**后端:**
- Python 3.11 + Flask
- Neo4j (图数据库)
- Milvus (向量数据库)
- 图RAG + 混合检索

**部署:**
- Docker + Docker Compose
- Nginx (反向代理)

### 服务架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx (80)    │────│  Frontend(3000) │────│  Backend(8000)  │
│   反向代理        │    │   Next.js应用    │    │   Flask API     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Neo4j (7474)   │────│ Milvus (19530)  │
                       │   图数据库        │    │   向量数据库      │
                       └─────────────────┘    └─────────────────┘
```

## 🔧 开发模式

如果您需要修改前端代码：

```bash
# 1. 停止容器化的前端服务
docker-compose stop frontend

# 2. 本地运行前端开发服务器
cd frontend
npm install
npm run dev

# 3. 访问 http://localhost:3000 进行开发
```

## 🐛 故障排除

### 问题1：Docker 未运行
```
❌ 错误：Docker 未运行
✅ 解决：启动 Docker Desktop
```

### 问题2：端口被占用
```
❌ 错误：端口 80/3000/8000 被占用
✅ 解决：关闭占用端口的程序，或修改 docker-compose.yml 中的端口映射
```

### 问题3：服务启动超时
```
❌ 错误：服务启动超时
✅ 解决：
1. 检查网络连接
2. 查看具体服务日志
3. 重启 Docker Desktop
4. 清理 Docker 缓存：docker system prune -f
```

## 💡 使用技巧

1. **首次启动**：需要下载镜像，可能需要5-10分钟
2. **后续启动**：数据已缓存，通常1-2分钟即可完成
3. **开发调试**：可以单独重启某个服务而不影响其他服务
4. **数据持久化**：除非手动清理，否则数据会永久保存

## 🙏 致谢

本项目的开发得益于以下开源项目和教程：

### 📚 教程项目
- **[Datawhale/all-in-rag](https://github.com/datawhalechina/all-in-rag)** - 大模型应用开发实战：RAG技术全栈指南
  - 本项目是该教程的完整实战案例，展示了RAG技术的实际应用
  - 涵盖数据处理、索引构建、检索优化、生成集成等完整技术栈
  - 通过实际的美食推荐场景，帮助学习者理解图RAG技术的实现细节

### 🍳 菜谱数据
- **[Anduin2017/HowToCook](https://github.com/Anduin2017/HowToCook)** - 程序员在家做饭方法指南
  - 本项目的菜谱数据主要来源于这个优秀的开源项目
  - 该项目用程序员的思维整理了大量实用的菜谱，格式规范、描述清晰
  - 感谢 [@Anduin2017](https://github.com/Anduin2017) 和所有贡献者们的无私分享

### 🛠️ 技术栈
- **Next.js** - React 全栈框架
- **Flask** - Python Web 框架
- **Neo4j** - 图数据库
- **Milvus** - 向量数据库
- **Docker** - 容器化部署
- **Tailwind CSS** - 样式框架

**🍽️ 享受您的美食推荐之旅！** 如果这个项目对您有帮助，请给个⭐️支持一下！

## 📄 许可证

本项目采用 [MIT License](./LICENSE) 开源协议。

### 🔓 使用权限
- ✅ **商业使用** - 可以用于商业项目
- ✅ **修改** - 可以修改源代码
- ✅ **分发** - 可以分发原始或修改后的代码
- ✅ **私人使用** - 可以私人使用
- ✅ **专利使用** - 授予专利使用权

### 📝 使用条件
- **保留版权声明** - 在所有副本中包含原始版权声明和许可证声明
- **保留许可证** - 在所有副本中包含 MIT 许可证

### ⚠️ 免责声明
- 软件按"原样"提供，不提供任何明示或暗示的保证
- 作者不承担任何责任或义务

---

Copyright (c) 2025 FutureUnreal. All rights reserved.