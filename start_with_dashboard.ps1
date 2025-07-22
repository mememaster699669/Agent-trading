# Build and start the Agent Trading System with Dashboard

Write-Host "🚀 Building Agent Trading System with Dashboard..." -ForegroundColor Green

# Build the main trading system
Write-Host "📦 Building main trading system..." -ForegroundColor Yellow
docker-compose build agent-trading

# Build the dashboard
Write-Host "📊 Building dashboard..." -ForegroundColor Yellow  
docker-compose build dashboard

# Start all services
Write-Host "🔄 Starting all services..." -ForegroundColor Yellow
docker-compose up -d

# Wait a moment for services to start
Start-Sleep -Seconds 10

# Check service status
Write-Host "✅ Checking service status..." -ForegroundColor Yellow
docker-compose ps

Write-Host ""
Write-Host "🎯 Services should be running at:" -ForegroundColor Green
Write-Host "  📊 Dashboard: http://localhost:8080" -ForegroundColor Cyan
Write-Host "  🤖 Trading System: http://localhost:8000" -ForegroundColor Cyan
Write-Host "  📈 Grafana: http://localhost:3000" -ForegroundColor Cyan
Write-Host "  🔍 Prometheus: http://localhost:9090" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Dashboard Features:" -ForegroundColor Green
Write-Host "  • Real-time BTC price from your trading system" -ForegroundColor White
Write-Host "  • Live trading decisions with confidence levels" -ForegroundColor White
Write-Host "  • System health and performance metrics" -ForegroundColor White
Write-Host "  • Live logs and error tracking" -ForegroundColor White
Write-Host "  • Automatic 5-second refresh" -ForegroundColor White
Write-Host ""
Write-Host "🔧 To view logs:" -ForegroundColor Green
Write-Host "  docker-compose logs -f agent-trading    # Trading system logs" -ForegroundColor White
Write-Host "  docker-compose logs -f dashboard        # Dashboard logs" -ForegroundColor White
Write-Host ""
Write-Host "⏹️  To stop all services:" -ForegroundColor Red
Write-Host "  docker-compose down" -ForegroundColor White
