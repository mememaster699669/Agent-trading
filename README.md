# 🤖 Agent Trading System

Next-generation cryptocurrency trading platform powered by **CrewAI Intelligence** and **ADK Execution**, with comprehensive logging and GCP DevOps integration patterns.

## 🎯 Philosophy: Probabilistic Quantitative Trading

This system abandons the flawed premise of "predicting exact prices" and instead focuses on:

- **Probability Distributions**: Model price ranges and uncertainty
- **Statistical Arbitrage**: Exploit mean reversion and cointegration
- **Risk-Adjusted Returns**: Optimize Sharpe ratios, not raw returns
- **Bayesian Inference**: Continuously update beliefs with new data
- **Regime Detection**: Adapt strategies to market conditions

## 🏗️ Architecture Overview

This system implements a hybrid **CrewAI + ADK** architecture:

### CrewAI Intelligence Layer (Strategic Reasoning)
- **🧠 Market Analysis Agent**: Probabilistic market analysis and pattern recognition
- **🔒 Risk Management Agent**: VaR, CVaR, position sizing, and drawdown controls
- **📊 Quantitative Researcher Agent**: Statistical models and regime detection
- **🎯 Portfolio Optimizer Agent**: Modern portfolio theory with transaction costs

### ADK Execution Layer (Deterministic Operations)
- **⚡ Order Execution Engine**: Smart routing and slippage minimization
- **📊 Data Pipeline**: Real-time 15m candle processing via Binance API
- **🔍 Monitor Agent**: System health and comprehensive performance tracking
- **🛡️ Risk Validation**: Multi-layer risk assessment and safety controls

### Core Infrastructure
- **📝 Comprehensive Logging**: Every action, decision, and error tracked with detailed context
- **🐳 Docker Ready**: Complete containerization for production deployment
- **🔧 Environment Management**: GCP DevOps agent pattern for configuration
- **🔌 LiteLLM Integration**: Custom LLM endpoints with base_url/completion_url support

## 🧮 Quantitative Methodologies

1. **Bayesian State-Space Models**: For price dynamics and uncertainty quantification
2. **GARCH Models**: For volatility forecasting and risk assessment
3. **Kalman Filters**: For adaptive parameter estimation
4. **Copulas**: For dependency modeling between assets
5. **Monte Carlo Simulation**: For risk simulation and scenario analysis
6. **Kelly Criterion**: For optimal position sizing

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- PostgreSQL (via Docker)
- Redis (via Docker)

### Installation

**Windows (PowerShell):**
```powershell
.\install.ps1
```

**Linux/macOS:**
```bash
chmod +x install.sh
./install.sh
```

### Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install crewai litellm pandas numpy ta psycopg2-binary redis ccxt
pip install aiohttp openai anthropic python-dotenv structlog

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure
docker-compose up -d

# Validate setup
python validate_env.py

# Run the system
python run.py
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# LLM Configuration (CRITICAL)
OPENAI_API_KEY=your-openai-api-key-here
MODEL_GPT_4O=gpt-4
LLM_BASE_URL=https://api.openai.com/v1
LLM_COMPLETION_URL=https://api.openai.com/v1/chat/completions

# Trading Configuration
BINANCE_API_KEY=your-binance-api-key
BINANCE_SECRET=your-binance-secret
BINANCE_TESTNET=true
ENABLE_LIVE_TRADING=false

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/agent_trading
REDIS_URL=redis://localhost:6379

# Risk Management
MAX_POSITION_SIZE=10000
MAX_DAILY_LOSS=5000
TRADING_CYCLE_INTERVAL=900
```

## 📊 Key Features

### Real-time Market Analysis
- **15-minute BTC candle processing** with Binance WebSocket
- **Technical indicators** (RSI, MACD, Bollinger Bands, Volume analysis)
- **Probability distributions** for price movements
- **Market regime detection** and adaptive strategies

### CrewAI Intelligence System
```python
# Market Analysis Agent
- Analyzes BTC market data using probabilistic models
- Generates confidence scores for trading signals
- Incorporates technical indicators and market sentiment

# Risk Management Agent  
- Assesses position sizing and risk-reward ratios
- Validates signals against portfolio constraints
- Recommends stop-loss and take-profit levels
```

### ADK Execution Engine
```python
# Order Processing Pipeline
1. Signal Reception → 2. Risk Validation → 3. Position Sizing → 
4. Order Execution → 5. Result Tracking

# Risk Management Features
- Position size limits
- Daily loss limits  
- Portfolio concentration checks
- VaR (Value at Risk) monitoring
```

### Comprehensive Logging System
Every system action is logged with detailed context:

```json
{
  "session_id": "session_1699123456",
  "action_id": "action_0001", 
  "timestamp": "2024-01-15T10:30:00Z",
  "action_type": "trading_decision",
  "status": "completed",
  "details": {
    "symbol": "BTC",
    "signal_type": "buy",
    "confidence": 0.85,
    "reasoning": "Strong bullish momentum with RSI oversold recovery"
  }
}
```

## 🐳 Docker Deployment

### Services
```yaml
services:
  app:           # Main trading application
  postgres:      # Market data storage  
  redis:         # Caching and sessions
  prometheus:    # Metrics collection (optional)
```

### Commands
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services  
docker-compose down

# Rebuild and restart
docker-compose up --build -d
```

## 📈 Trading Workflow

### Continuous Operation
1. **Data Collection** (every 30s): Fetch latest BTC market data
2. **Intelligence Analysis** (every 15m): Generate trading signals with CrewAI
3. **Risk Assessment**: Validate signals against risk parameters
4. **Execution Decision**: Execute approved trades via ADK
5. **Performance Tracking**: Log results and update metrics

### Signal Processing Pipeline
```
Market Data → CrewAI Analysis → Risk Validation → ADK Execution → Result Logging
     ↓              ↓               ↓              ↓              ↓
  Technical     Probabilistic    Multi-layer    Deterministic   Comprehensive
  Indicators    AI Analysis      Risk Checks    Execution       Tracking
```

## 🔍 Monitoring & Debugging

### Real-time Monitoring
```bash
# View live logs
tail -f logs/agent_trading_$(date +%Y%m%d).log

# Action-specific logs
tail -f logs/actions_$(date +%Y%m%d).jsonl | jq .

# System metrics
python -c "from src.logging_system import get_logger; get_logger('main').log_system_metrics({})"
```

### Performance Metrics
- Trading cycle completion times
- Signal generation accuracy  
- Risk assessment effectiveness
- Order execution success rates
- System uptime and error tracking

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test specific components
pytest tests/test_crewai_intelligence.py
pytest tests/test_adk_execution.py  
pytest tests/test_data_pipeline.py

# Integration tests
pytest tests/test_integration.py -v
```

## 📋 System Requirements

### Production
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ for full LLM processing
- **Storage**: 50GB+ for historical data
- **Network**: Stable internet for real-time data

### Development  
- **CPU**: 2+ cores minimum
- **RAM**: 4GB+ sufficient for testing
- **Storage**: 10GB+ for development
- **Python**: 3.9+ with pip

## 🚨 Important Notes

### Security
- Store API keys securely in environment variables
- Use testnet for initial testing
- Enable live trading only after thorough validation

### Risk Management
- Start with small position sizes
- Monitor daily loss limits
- Validate all signals before execution  
- Keep comprehensive logs for analysis

### Performance
- LLM calls may have latency - adjust timeouts accordingly
- Monitor database connections for high-frequency operations
- Use Redis caching to reduce API calls

## 📚 API Documentation

### CrewAI Intelligence
```python
from src.crewai_intelligence import CrewAIIntelligenceSystem

system = CrewAIIntelligenceSystem(config)
signal = system.generate_trading_signal(market_data, portfolio_data)
```

### ADK Execution
```python
from src.adk_execution import ADKExecutionEngine

engine = ADKExecutionEngine(config)
result = await engine.process_signal(trading_signal)
```

### Data Pipeline
```python
from src.dataset import BTCDataManager

data_manager = BTCDataManager(config)  
market_data = await data_manager.fetch_latest_data()
```

### Logging System
```python
from src.logging_system import get_logger

logger = get_logger("MyComponent")
logger.log_trading_decision(symbol="BTC", signal_type="buy", confidence=0.85)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`  
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Documentation**: Check logs for detailed error context

---

## ⚠️ Disclaimer

This system is for educational and research purposes only. Past performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss. Always consult with financial professionals and start with paper trading before using real funds.

**🎯 Built with**: CrewAI • LiteLLM • PostgreSQL • Redis • Docker • Python 3.9+
