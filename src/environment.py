"""
Environment Configuration for Agent Trading System
Centralized environment variable loading and validation with detailed feedback
"""

import os
from dotenv import load_dotenv
import warnings
import logging
from typing import Dict, Any, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('crewai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('ccxt').setLevel(logging.WARNING)
logging.getLogger('pandas').setLevel(logging.WARNING)

print("🚀 Agent Trading System - Environment Initialization")
print("=" * 60)

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# LLM Configuration
# =============================================================================
# OpenAI API Configuration
openai_api_key = os.getenv("OPENAI_API_KEY", "NOT_SET")
llm_api_key = os.getenv("LLM_API_KEY", openai_api_key)  # Support both variants
base_url = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
llm_model = os.getenv("LLM_MODEL", "gpt-4")
llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Model configuration for litellm compatibility
MODEL_GPT_4O = llm_model

# =============================================================================
# Database Configuration
# =============================================================================
database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/agent_trading")
database_pool_size = int(os.getenv("DATABASE_POOL_SIZE", "10"))
database_max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
database_pool_timeout = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))

# =============================================================================
# Redis Configuration
# =============================================================================
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_db = int(os.getenv("REDIS_DB", "0"))
redis_max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))

# =============================================================================
# Binance API Configuration
# =============================================================================
binance_api_key = os.getenv("BINANCE_API_KEY", "NOT_SET")
binance_secret = os.getenv("BINANCE_SECRET", "NOT_SET")
binance_testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# =============================================================================
# BTC Data Configuration
# =============================================================================
primary_symbol = os.getenv("PRIMARY_SYMBOL", "BTCUSDT")
timeframe = os.getenv("TIMEFRAME", "15m")
update_interval = int(os.getenv("UPDATE_INTERVAL", "30"))
historical_days = int(os.getenv("HISTORICAL_DAYS", "30"))

# =============================================================================
# Data Quality Settings
# =============================================================================
min_data_quality = float(os.getenv("MIN_DATA_QUALITY", "0.8"))
max_price_gap = float(os.getenv("MAX_PRICE_GAP", "0.05"))
volume_threshold_percentile = float(os.getenv("VOLUME_THRESHOLD_PERCENTILE", "0.1"))
cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
cache_expiry = int(os.getenv("CACHE_EXPIRY", "3600"))

# =============================================================================
# Trading Configuration
# =============================================================================
min_validation_score = float(os.getenv("MIN_VALIDATION_SCORE", "0.7"))
max_position_size = float(os.getenv("MAX_POSITION_SIZE", "10000"))
max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "5000"))
var_limit = float(os.getenv("VAR_LIMIT", "2000"))
concentration_limit = float(os.getenv("CONCENTRATION_LIMIT", "0.2"))

# =============================================================================
# Docker Configuration
# =============================================================================
app_env = os.getenv("APP_ENV", "development")
container_name = os.getenv("CONTAINER_NAME", "agent-trading-app")
network_name = os.getenv("NETWORK_NAME", "agent-trading-network")

# =============================================================================
# CrewAI Configuration
# =============================================================================
crewai_temperature = float(os.getenv("CREWAI_TEMPERATURE", "0.3"))
crewai_max_iterations = int(os.getenv("CREWAI_MAX_ITERATIONS", "3"))
crewai_memory_enabled = os.getenv("CREWAI_MEMORY_ENABLED", "true").lower() == "true"

# =============================================================================
# ADK Configuration
# =============================================================================
adk_task_timeout = int(os.getenv("ADK_TASK_TIMEOUT", "30"))
adk_retry_attempts = int(os.getenv("ADK_RETRY_ATTEMPTS", "3"))
adk_batch_size = int(os.getenv("ADK_BATCH_SIZE", "10"))

# =============================================================================
# Monitoring Configuration
# =============================================================================
log_level = os.getenv("LOG_LEVEL", "INFO")
metrics_port = int(os.getenv("METRICS_PORT", "8080"))
prometheus_enabled = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"

# =============================================================================
# Set environment variables for downstream libraries
# =============================================================================
if llm_api_key != "NOT_SET":
    os.environ["OPENAI_API_KEY"] = llm_api_key
if base_url:
    os.environ["OPENAI_API_BASE"] = base_url

# =============================================================================
# Configuration Validation and Feedback
# =============================================================================

def validate_environment() -> Dict[str, Any]:
    """Validate environment configuration and return status"""
    
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config_summary": {}
    }
    
    # Critical validations
    if llm_api_key == "NOT_SET":
        validation_results["errors"].append("LLM API Key not set - CrewAI intelligence will not work")
        validation_results["valid"] = False
    
    if binance_api_key == "NOT_SET":
        validation_results["warnings"].append("Binance API Key not set - using mock data only")
    
    if binance_secret == "NOT_SET" and binance_api_key != "NOT_SET":
        validation_results["errors"].append("Binance API Key set but Secret missing")
        validation_results["valid"] = False
    
    # Database validation
    if not database_url.startswith(("postgresql://", "postgres://")):
        validation_results["warnings"].append("Database URL doesn't look like PostgreSQL")
    
    # Redis validation
    if not redis_url.startswith("redis://"):
        validation_results["warnings"].append("Redis URL format may be incorrect")
    
    # Trading parameter validation
    if max_position_size <= 0:
        validation_results["errors"].append("Max position size must be positive")
        validation_results["valid"] = False
    
    if concentration_limit <= 0 or concentration_limit > 1:
        validation_results["errors"].append("Concentration limit must be between 0 and 1")
        validation_results["valid"] = False
    
    # Build configuration summary
    validation_results["config_summary"] = {
        "environment": app_env,
        "trading_mode": "testnet" if binance_testnet else "live",
        "primary_symbol": primary_symbol,
        "timeframe": timeframe,
        "data_source": "binance" if binance_api_key != "NOT_SET" else "mock",
        "llm_model": llm_model,
        "database_configured": database_url != "postgresql://postgres:postgres@localhost:5432/agent_trading",
        "redis_configured": redis_url != "redis://localhost:6379"
    }
    
    return validation_results


def print_environment_status():
    """Print detailed environment status"""
    
    print("\n🔧 LLM Configuration:")
    print(f"   API Key: {'✅ SET' if llm_api_key != 'NOT_SET' else '❌ MISSING (CRITICAL!)'}")
    print(f"   Base URL: {base_url}")
    print(f"   Model: {llm_model}")
    print(f"   Temperature: {llm_temperature}")
    
    print("\n💰 Binance Trading Configuration:")
    print(f"   API Key: {'✅ SET' if binance_api_key != 'NOT_SET' else '⚠️  NOT SET (mock mode)'}")
    print(f"   Secret: {'✅ SET' if binance_secret != 'NOT_SET' else '⚠️  NOT SET (mock mode)'}")
    print(f"   Mode: {'🧪 TESTNET' if binance_testnet else '🔴 LIVE TRADING'}")
    print(f"   Symbol: {primary_symbol}")
    print(f"   Timeframe: {timeframe}")
    
    print("\n🗄️  Database Configuration:")
    print(f"   URL: {database_url}")
    print(f"   Pool Size: {database_pool_size}")
    print(f"   Connection: {'🐘 PostgreSQL' if 'postgres' in database_url else '❓ Unknown'}")
    
    print("\n⚡ Redis Configuration:")
    print(f"   URL: {redis_url}")
    print(f"   Database: {redis_db}")
    print(f"   Max Connections: {redis_max_connections}")
    
    print("\n🛡️  Risk Management:")
    print(f"   Max Position Size: ${max_position_size:,}")
    print(f"   Max Daily Loss: ${max_daily_loss:,}")
    print(f"   VaR Limit: ${var_limit:,}")
    print(f"   Concentration Limit: {concentration_limit:.1%}")
    
    print("\n📊 Data Quality Settings:")
    print(f"   Min Quality Score: {min_data_quality}")
    print(f"   Max Price Gap: {max_price_gap:.1%}")
    print(f"   Cache Enabled: {'✅' if cache_enabled else '❌'}")
    print(f"   Update Interval: {update_interval}s")
    
    print("\n🤖 CrewAI Settings:")
    print(f"   Temperature: {crewai_temperature}")
    print(f"   Max Iterations: {crewai_max_iterations}")
    print(f"   Memory: {'✅ Enabled' if crewai_memory_enabled else '❌ Disabled'}")
    
    print("\n🐳 Docker Configuration:")
    print(f"   Environment: {app_env}")
    print(f"   Container: {container_name}")
    print(f"   Network: {network_name}")
    
    # Validation
    validation = validate_environment()
    
    print("\n🔍 Configuration Validation:")
    if validation["valid"]:
        print("   ✅ All critical settings are valid")
    else:
        print("   ❌ Configuration has errors:")
        for error in validation["errors"]:
            print(f"      • {error}")
    
    if validation["warnings"]:
        print("   ⚠️  Warnings:")
        for warning in validation["warnings"]:
            print(f"      • {warning}")
    
    print("\n" + "=" * 60)
    print("🎯 Agent Trading System Environment Ready")
    if not validation["valid"]:
        print("⚠️  Fix configuration errors before deployment!")
    print("=" * 60)


def get_environment_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of the environment configuration
    Returns a dictionary with all environment settings
    """
    return {
        'llm_config': {
            'api_key_set': bool(openai_api_key and openai_api_key != "NOT_SET"),
            'base_url': base_url,
            'model': llm_model,
            'temperature': llm_temperature
        },
        'binance_config': {
            'api_key_set': bool(binance_api_key and binance_api_key != "NOT_SET"),
            'secret_set': bool(binance_secret and binance_secret != "NOT_SET"),
            'testnet': binance_testnet,
            'symbol': primary_symbol,
            'timeframe': timeframe
        },
        'database_config': {
            'url': database_url,
            'pool_size': database_pool_size,
            'connection_type': 'PostgreSQL' if 'postgresql' in database_url else 'Other'
        },
        'redis_config': {
            'url': redis_url,
            'db': redis_db,
            'max_connections': redis_max_connections
        },
        'risk_config': {
            'max_position_size': max_position_size,
            'max_daily_loss': max_daily_loss,
            'var_limit': var_limit,
            'concentration_limit': concentration_limit
        },
        'data_config': {
            'min_quality_score': min_data_quality,
            'max_price_gap': max_price_gap,
            'cache_enabled': cache_enabled,
            'update_interval': update_interval
        },
        'crewai_config': {
            'temperature': crewai_temperature,
            'max_iterations': crewai_max_iterations,
            'memory_enabled': crewai_memory_enabled
        },
        'docker_config': {
            'environment': app_env,
            'container': container_name,
            'network': network_name
        }
    }


# Auto-run environment status on import
print_environment_status()

# Export validation function for use by other modules
__all__ = ['validate_environment', 'print_environment_status', 'get_environment_summary']
