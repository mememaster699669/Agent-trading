#!/bin/bash

# Build and start the Agent Trading System with Dashboard

echo "🚀 Building Agent Trading System with Dashboard..."

# Build the main trading system
echo "📦 Building main trading system..."
docker-compose build agent-trading

# Build the dashboard
echo "📊 Building dashboard..."
docker-compose build dashboard

# Start all services
echo "🔄 Starting all services..."
docker-compose up -d

# Wait a moment for services to start
sleep 10

# Check service status
echo "✅ Checking service status..."
docker-compose ps

echo ""
echo "🎯 Services should be running at:"
echo "  📊 Dashboard: http://localhost:8080"
echo "  🤖 Trading System: http://localhost:8000"  
echo "  📈 Grafana: http://localhost:3000"
echo "  🔍 Prometheus: http://localhost:9090"
echo ""
echo "📋 Dashboard Features:"
echo "  • Real-time BTC price from your trading system"
echo "  • Live trading decisions with confidence levels"
echo "  • System health and performance metrics"
echo "  • Live logs and error tracking"
echo "  • Automatic 5-second refresh"
echo ""
echo "🔧 To view logs:"
echo "  docker-compose logs -f agent-trading    # Trading system logs"
echo "  docker-compose logs -f dashboard        # Dashboard logs"
echo ""
echo "⏹️  To stop all services:"
echo "  docker-compose down"
