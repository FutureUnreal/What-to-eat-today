@echo off
chcp 65001 >nul

echo.
echo ===============================================
echo       今天吃什么 - 服务停止
echo ===============================================
echo.

echo 💾 数据保留选项：
echo    1. 停止服务但保留数据（推荐）
echo    2. 停止服务并清理所有数据
echo.

set /p choice="请选择 (1/2，默认为1): "

if "%choice%"=="2" (
    echo.
    echo ⚠️  警告：这将删除所有数据，包括：
    echo    - Neo4j 图数据库数据
    echo    - Milvus 向量数据库数据
    echo    - 所有聊天记录和用户偏好
    echo.
    
    set /p confirm="确认删除所有数据？(y/N): "
    if /i "!confirm!"=="y" (
        echo [INFO] 🗑️  正在清理所有数据...
        docker-compose down -v
        echo [SUCCESS] ✅ 数据清理完成
    ) else (
        echo [INFO] ℹ️  已取消数据清理，仅停止服务
        docker-compose down
    )
) else (
    echo [INFO] ℹ️  保留数据，仅停止服务
    docker-compose down
)

echo [INFO] 🧹 清理未使用的Docker资源...
docker system prune -f

echo.
echo [SUCCESS] ✅ 服务已停止
echo.
echo 🔄 下次启动：
echo    start.sh (Linux/macOS)
echo    start.bat (Windows)
echo.

pause