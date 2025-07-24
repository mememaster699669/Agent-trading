@echo off
setlocal enabledelayedexpansion

REM Agent Trading System - Docker Build and Deploy Script (Windows)
REM This script builds and deploys the complete trading system with advanced frameworks

echo 🚀 Agent Trading System - Docker Build ^& Deploy
echo ==================================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker and try again.
    exit /b 1
)

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] docker-compose is not installed. Please install it and try again.
    exit /b 1
)

echo [INFO] Checking system requirements...

REM Check if .env file exists
if not exist .env (
    if exist .env.example (
        echo [WARNING] .env file not found. Copying from .env.example
        copy .env.example .env
    ) else (
        echo [ERROR] .env file not found and no .env.example available
        exit /b 1
    )
)

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist data mkdir data
if not exist logs mkdir logs
if not exist models mkdir models
if not exist cache mkdir cache
if not exist backups mkdir backups
if not exist monitoring mkdir monitoring
if not exist monitoring\grafana mkdir monitoring\grafana
if not exist monitoring\prometheus mkdir monitoring\prometheus

REM Build mode selection
set BUILD_MODE=%1
if "%BUILD_MODE%"=="" set BUILD_MODE=production

if "%BUILD_MODE%"=="clean" (
    echo [INFO] Cleaning up existing containers and images...
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    docker volume prune -f
    echo [SUCCESS] Cleanup completed
    exit /b 0
)

REM Stop existing containers
echo [INFO] Stopping existing containers...
docker-compose down --remove-orphans

REM Build images with cache optimization
echo [INFO] Building Docker images...
set DOCKER_BUILDKIT=1
set COMPOSE_DOCKER_CLI_BUILD=1

REM Build main application
echo [INFO] Building main trading application...
docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1 agent-trading

REM Build dashboard
echo [INFO] Building dashboard application...
docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1 dashboard

REM Start services in correct order
echo [INFO] Starting infrastructure services...
docker-compose up -d redis postgres

REM Wait for infrastructure to be ready
echo [INFO] Waiting for infrastructure services to be ready...
timeout /t 15 /nobreak >nul

REM Start main application
echo [INFO] Starting main trading application...
docker-compose up -d agent-trading

REM Wait for main app to be ready
timeout /t 15 /nobreak >nul

REM Start dashboard
echo [INFO] Starting dashboard...
docker-compose up -d dashboard

REM Start monitoring (optional)
if exist monitoring\prometheus.yml (
    echo [INFO] Starting monitoring services...
    docker-compose up -d prometheus grafana
) else (
    echo [WARNING] Monitoring configuration not found. Skipping Prometheus/Grafana
)

REM Final status check
echo [INFO] Performing final health checks...
timeout /t 20 /nobreak >nul

REM Check service status
echo.
echo [INFO] Service Status:
echo ==============================================
docker-compose ps

echo.
echo [SUCCESS] Deployment completed successfully!
echo ==============================================
echo 🎯 Agent Trading System is now running!
echo.
echo 📊 Dashboard:     http://localhost:8080
echo 🔧 Main API:      http://localhost:8000
echo 📈 Grafana:       http://localhost:3000 (admin/admin)
echo 📊 Prometheus:    http://localhost:9090
echo 🗄️  PostgreSQL:   localhost:5432 (postgres/postgres)
echo 📦 Redis:         localhost:6379
echo.
echo 📁 Logs location: ./logs/
echo 💾 Data location: ./data/
echo.
echo 🔍 To view logs: docker-compose logs -f [service-name]
echo 🛑 To stop:      docker-compose down
echo 🔄 To restart:   docker-compose restart [service-name]
echo.
echo [SUCCESS] Happy Trading! 🚀

pause
