@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 今天吃什么 - Windows启动脚本

echo.
echo ===============================================
echo       今天吃什么 - AI美食推荐助手
echo ===============================================
echo.

REM 检查Docker环境
echo [STEP] 检查Docker环境...
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker未运行，请先启动Docker Desktop
    echo.
    echo [INFO] 请确保：
    echo    1. Docker Desktop已安装并运行
    echo    2. 启用WSL 2引擎（推荐）
    echo.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose未安装
    echo [INFO] Docker Compose通常随Docker Desktop一起安装
    pause
    exit /b 1
)

echo [SUCCESS] Docker环境检查通过

REM 检查环境配置
echo [STEP] 检查环境配置...
if not exist ".env" (
    echo [WARNING] .env文件不存在，正在创建...
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [INFO] 已从.env.example创建.env文件
    ) else (
        echo [ERROR] .env.example文件不存在，无法创建配置文件
        echo [INFO] 请手动创建.env文件并配置必要的环境变量
        pause
        exit /b 1
    )
)

REM 检查API密钥
findstr /C:"OPENAI_API_KEY=sk-" .env >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ⚠️  API密钥未配置或格式不正确
    echo [INFO] 请编辑.env文件，设置您的API密钥：
    echo [INFO]   OPENAI_API_KEY=your_api_key_here
    echo [INFO]   OPENAI_BASE_URL=your_api_base_url
    echo [INFO]   LLM_MODEL=your_model_name
    echo.
    echo [INFO] 支持的API供应商请参考: LLM_CONFIG.md
    echo.
    set /p continue="是否继续启动？(y/N): "
    if /i not "!continue!"=="y" (
        echo [INFO] 请配置API密钥后重新运行
        pause
        exit /b 1
    )
) else (
    echo [SUCCESS] 环境配置检查通过
)

REM 创建必要目录
echo [STEP] 创建必要目录...
if not exist "data\cypher" mkdir "data\cypher"
if not exist "nginx" mkdir "nginx"
if not exist "logs" mkdir "logs"
echo [SUCCESS] 目录创建完成

REM 前端依赖将在Docker容器中自动安装
echo [INFO] 前端依赖将在Docker容器中自动安装

REM 启动服务
echo [STEP] 启动所有服务...

echo [INFO] 拉取Docker镜像...
docker-compose pull

echo [INFO] 构建应用镜像...
docker-compose build

echo [INFO] 启动服务容器...
docker-compose up -d

echo [SUCCESS] 服务启动命令执行完成

REM 等待服务就绪
echo [STEP] 等待服务启动...

set max_retries=60
set retry_count=0

REM 等待后端服务
echo [INFO] 等待后端服务启动...

:check_backend
curl -f http://localhost:8000/health >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] 后端服务启动成功
    goto backend_ready
)

set /a retry_count+=1
if !retry_count! geq !max_retries! (
    echo [ERROR] 后端服务启动超时
    echo [INFO] 查看日志: docker-compose logs backend
    echo [INFO] 常见问题：
    echo [INFO]   - 检查端口8000是否被占用
    echo [INFO]   - 检查Docker内存是否充足
    echo [INFO]   - 检查API密钥配置是否正确
    pause
    exit /b 1
)

echo|set /p="."
timeout /t 2 /nobreak >nul
goto check_backend

:backend_ready

REM 等待Nginx代理服务
echo [INFO] 等待Nginx代理服务启动...
set retry_count=0

:check_nginx
curl -f http://localhost >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] Nginx代理服务启动成功
    goto nginx_ready
)

set /a retry_count+=1
if !retry_count! geq !max_retries! (
    echo [WARNING] Nginx代理服务启动超时
    echo [INFO] 查看日志: docker-compose logs nginx
    echo [INFO] 尝试直接访问前端: http://localhost:3000
    goto nginx_ready
)

echo|set /p="."
timeout /t 2 /nobreak >nul
goto check_nginx

:nginx_ready

REM 等待前端服务
echo [INFO] 等待前端服务启动...
set retry_count=0

:check_frontend
curl -f http://localhost:3000 >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] 前端服务启动成功
    goto frontend_ready
)

set /a retry_count+=1
if !retry_count! geq !max_retries! (
    echo [ERROR] 前端服务启动超时
    echo [INFO] 查看日志: docker-compose logs frontend
    pause
    exit /b 1
)

echo|set /p="."
timeout /t 2 /nobreak >nul
goto check_frontend

:frontend_ready

REM 显示服务信息
echo.
echo ===============================================
echo            🎉 部署完成！
echo ===============================================
echo.

echo 📋 服务访问地址：
echo    🌐 应用首页:     http://localhost
echo    ⚛️  前端应用:     http://localhost:3000
echo    🐍 后端API:      http://localhost:8000
echo    📊 Neo4j浏览器:  http://localhost:7474
echo       用户名: neo4j, 密码: all-in-rag
echo    🗄️  Milvus控制台: http://localhost:9001
echo       用户名: minioadmin, 密码: minioadmin
echo.

echo 📝 管理命令：
echo    查看服务状态: docker-compose ps
echo    查看日志:     docker-compose logs -f [service_name]
echo    重启服务:     docker-compose restart [service_name]
echo    停止服务:     docker-compose down
echo    完全清理:     docker-compose down -v
echo.

echo 💡 开发提示：
echo    - 代码修改后需要重新构建: docker-compose build [service_name]
echo    - 查看实时日志: docker-compose logs -f
echo    - 进入容器调试: docker-compose exec [service_name] bash
echo.

echo [SUCCESS] 🚀 系统启动完成，正在为您打开应用...

REM 打开浏览器
timeout /t 3 /nobreak >nul
start http://localhost

echo.
echo 按任意键退出...
pause >nul