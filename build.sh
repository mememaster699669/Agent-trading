#!/bin/bash

# Agent Trading System - Docker Build and Deploy Script
# This script builds and deploys the complete trading system with advanced frameworks

set -e  # Exit on any error

echo "🚀 Agent Trading System - Docker Build & Deploy"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed. Please install it and try again."
    exit 1
fi

print_status "Checking system requirements..."

# Check available disk space (minimum 10GB)
available_space=$(df . | awk 'NR==2 {print $4}')
if [ $available_space -lt 10485760 ]; then  # 10GB in KB
    print_warning "Low disk space detected. Ensure you have at least 10GB free space."
fi

# Check if .env file exists
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        print_warning ".env file not found. Copying from .env.example"
        cp .env.example .env
    else
        print_error ".env file not found and no .env.example available"
        exit 1
    fi
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data logs models cache backups monitoring/grafana monitoring/prometheus

# Build mode selection
BUILD_MODE=${1:-"production"}

case $BUILD_MODE in
    "dev"|"development")
        print_status "Building in DEVELOPMENT mode..."
        COMPOSE_FILE="docker-compose.yml"
        ;;
    "prod"|"production")
        print_status "Building in PRODUCTION mode..."
        COMPOSE_FILE="docker-compose.yml"
        ;;
    "clean")
        print_status "Cleaning up existing containers and images..."
        docker-compose down --volumes --remove-orphans
        docker system prune -f
        docker volume prune -f
        print_success "Cleanup completed"
        exit 0
        ;;
    *)
        print_error "Invalid build mode. Use: dev, prod, or clean"
        exit 1
        ;;
esac

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose down --remove-orphans

# Build images with cache optimization
print_status "Building Docker images..."
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build main application
print_status "Building main trading application..."
docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1 agent-trading

# Build dashboard
print_status "Building dashboard application..."
docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1 dashboard

# Start services in correct order
print_status "Starting infrastructure services..."
docker-compose up -d redis postgres

# Wait for infrastructure to be ready
print_status "Waiting for infrastructure services to be ready..."
sleep 10

# Check if services are healthy
print_status "Checking service health..."
for i in {1..30}; do
    if docker-compose ps | grep -q "healthy"; then
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

# Start main application
print_status "Starting main trading application..."
docker-compose up -d agent-trading

# Wait for main app to be ready
sleep 15

# Start dashboard
print_status "Starting dashboard..."
docker-compose up -d dashboard

# Start monitoring (optional)
if [ -f monitoring/prometheus.yml ]; then
    print_status "Starting monitoring services..."
    docker-compose up -d prometheus grafana
else
    print_warning "Monitoring configuration not found. Skipping Prometheus/Grafana"
fi

# Final status check
print_status "Performing final health checks..."
sleep 20

# Check service status
echo ""
print_status "Service Status:"
echo "=============================================="
docker-compose ps

# Check if all services are running
if docker-compose ps | grep -q "Exit"; then
    print_error "Some services failed to start. Check logs with: docker-compose logs"
    exit 1
fi

# Display access information
echo ""
print_success "Deployment completed successfully!"
echo "=============================================="
echo "🎯 Agent Trading System is now running!"
echo ""
echo "📊 Dashboard:     http://localhost:8080"
echo "🔧 Main API:      http://localhost:8000"
echo "📈 Grafana:       http://localhost:3000 (admin/admin)"
echo "📊 Prometheus:    http://localhost:9090"
echo "🗄️  PostgreSQL:   localhost:5432 (postgres/postgres)"
echo "📦 Redis:         localhost:6379"
echo ""
echo "📁 Logs location: ./logs/"
echo "💾 Data location: ./data/"
echo ""
echo "🔍 To view logs: docker-compose logs -f [service-name]"
echo "🛑 To stop:      docker-compose down"
echo "🔄 To restart:   docker-compose restart [service-name]"
echo ""
print_success "Happy Trading! 🚀"
