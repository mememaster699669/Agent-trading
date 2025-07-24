"""
Configuration Management
Centralized configuration for the agent trading system
Integrates with environment.py for comprehensive environment handling
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import environment configuration (this handles dotenv loading)
try:
    from . import environment as env
    ENVIRONMENT_LOADED = True
except ImportError:
    # Fallback if environment.py not available
    from dotenv import load_dotenv
    load_dotenv()
    ENVIRONMENT_LOADED = False


@dataclass
class LLMConfig:
    """LLM configuration"""
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_iterations: int = 3
    memory_enabled: bool = True


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 10000
    max_daily_loss: float = 5000
    max_portfolio_exposure: float = 50000
    var_limit: float = 2000
    concentration_limit: float = 0.2
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10


@dataclass
@dataclass
class TradingConfig:
    """Trading system configuration"""
    symbols: List[str] = None
    update_interval: int = 30
    min_validation_score: float = 0.7
    
    # Trading modes
    enable_paper_trading: bool = True
    enable_live_trading: bool = False
    broker_api_key: Optional[str] = None
    broker_secret: Optional[str] = None
    broker_testnet: bool = True
    
    # Advanced frameworks configuration
    enable_bayesian: bool = True
    enable_quantlib: bool = True
    enable_portfolio_optimization: bool = True
    enable_garch_analysis: bool = True
    enable_ml_frameworks: bool = True
    enable_physics_models: bool = True
    enable_microstructure: bool = True
    
    # Advanced analysis settings
    bayesian_mcmc_draws: int = 2000
    bayesian_tune_steps: int = 1000
    garch_lookback_days: int = 252
    ml_ensemble_models: int = 5
    portfolio_rebalance_freq: str = "weekly"
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTCUSDT"]


@dataclass
class AdvancedFrameworksConfig:
    """Configuration for advanced 5-phase frameworks"""
    # Phase 1: Bayesian
    bayesian_enabled: bool = True
    mcmc_draws: int = 2000
    mcmc_tune: int = 1000
    hierarchical_levels: int = 3
    
    # Phase 2: QuantLib
    quantlib_enabled: bool = True
    risk_free_rate: float = 0.05
    default_volatility: float = 0.25
    option_exercise_style: str = "european"
    
    # Phase 3: Portfolio
    portfolio_enabled: bool = True
    risk_aversion: float = 2.0
    max_weight: float = 0.4
    min_weight: float = 0.05
    rebalance_threshold: float = 0.05
    
    # Phase 4: Time Series
    timeseries_enabled: bool = True
    garch_model: str = "GARCH(1,1)"
    arch_lags: int = 1
    garch_lags: int = 1
    min_periods: int = 100
    
    # Phase 5: ML/AI
    ml_enabled: bool = True
    ensemble_size: int = 5
    train_test_split: float = 0.8
    cross_validation_folds: int = 5
    feature_selection: bool = True
    
    # Physics Models
    physics_enabled: bool = True
    entropy_window: int = 50
    hurst_window: int = 100
    lyapunov_window: int = 50
    
    # Market Microstructure
    microstructure_enabled: bool = True
    tick_analysis: bool = True
    order_book_depth: int = 10
    
    def get_enabled_frameworks(self) -> List[str]:
        """Get list of enabled frameworks"""
        enabled = []
        if self.bayesian_enabled:
            enabled.append("bayesian")
        if self.quantlib_enabled:
            enabled.append("quantlib")
        if self.portfolio_enabled:
            enabled.append("portfolio")
        if self.timeseries_enabled:
            enabled.append("timeseries")
        if self.ml_enabled:
            enabled.append("ml")
        if self.physics_enabled:
            enabled.append("physics")
        if self.microstructure_enabled:
            enabled.append("microstructure")
        return enabled


@dataclass
class DataConfig:
    """Enhanced data configuration for BTC focus"""
    # Binance API
    binance_api_key: Optional[str] = None
    binance_secret: Optional[str] = None
    binance_testnet: bool = True
    
    # Data collection
    primary_symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    update_interval: int = 30  # seconds
    historical_days: int = 30
    
    # Data quality
    min_data_quality: float = 0.8
    max_price_gap: float = 0.05  # 5% max price gap
    volume_threshold_percentile: float = 0.1
    
    # Cache settings
    cache_enabled: bool = True
    cache_expiry: int = 3600  # 1 hour
    
    # Database
    batch_size: int = 1000
    max_retries: int = 3


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    metrics_port: int = 8080
    prometheus_enabled: bool = True
    log_file_path: str = "logs/"
    enable_dashboard: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    enable_legacy_rl: bool = True
    enable_technical_features: bool = True
    enable_microstructure: bool = True
    feature_cache_enabled: bool = True
    max_features: int = 100


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "postgresql://postgres:postgres@localhost:5432/agent_trading"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str = "redis://localhost:6379"
    db: int = 0
    max_connections: int = 10
    socket_timeout: int = 5


@dataclass
class DockerConfig:
    """Docker-specific configuration"""
    app_env: str = "development"
    container_name: str = "agent-trading-app"
    network_name: str = "agent-trading-network"
    data_volume: str = "/app/data"
    logs_volume: str = "/app/logs"
    models_volume: str = "/app/models"


class ConfigManager:
    """Centralized configuration manager with environment integration"""
    
    def __init__(self, use_environment_module: bool = True):
        """
        Initialize configuration manager
        
        Args:
            use_environment_module: If True, use environment.py for enhanced feedback
        """
        self.use_environment_module = use_environment_module and ENVIRONMENT_LOADED
        
        # Load configurations
        self.llm = self._load_llm_config()
        self.risk = self._load_risk_config()
        self.trading = self._load_trading_config()
        self.data = self._load_data_config()
        self.monitoring = self._load_monitoring_config()
        self.features = self._load_feature_config()
        self.database = self._load_database_config()
        self.redis = self._load_redis_config()
        self.docker = self._load_docker_config()
        
        # Load advanced frameworks configuration
        self.advanced_frameworks = self._load_advanced_frameworks_config()
        
        # Validate configuration if environment module is available
        if self.use_environment_module:
            from . import environment as env
            self.validation_results = env.validate_environment()
        else:
            self.validation_results = {"valid": True, "errors": [], "warnings": []}
    
    def _get_env_value(self, key: str, default: Any = None, var_type: type = str) -> Any:
        """Get environment value with proper type conversion"""
        if self.use_environment_module:
            # Use values from environment.py module
            from . import environment as env
            return getattr(env, key.lower(), default)
        else:
            # Fallback to direct os.getenv
            value = os.getenv(key, default)
            if var_type == bool and isinstance(value, str):
                return value.lower() in ('true', 'yes', '1')
            elif var_type in (int, float) and isinstance(value, str):
                return var_type(value)
            return value
    
    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration with environment integration"""
        return LLMConfig(
            model=self._get_env_value("LLM_MODEL", "gpt-4"),
            api_key=self._get_env_value("LLM_API_KEY"),
            base_url=self._get_env_value("LLM_BASE_URL"),
            temperature=self._get_env_value("LLM_TEMPERATURE", 0.3, float),
            max_iterations=self._get_env_value("CREWAI_MAX_ITERATIONS", 3, int),
            memory_enabled=self._get_env_value("CREWAI_MEMORY_ENABLED", True, bool)
        )
    
    def _load_risk_config(self) -> RiskConfig:
        """Load risk configuration with environment integration"""
        return RiskConfig(
            max_position_size=self._get_env_value("MAX_POSITION_SIZE", 10000.0, float),
            max_daily_loss=self._get_env_value("MAX_DAILY_LOSS", 5000.0, float),
            max_portfolio_exposure=self._get_env_value("MAX_PORTFOLIO_EXPOSURE", 50000.0, float),
            var_limit=self._get_env_value("VAR_LIMIT", 2000.0, float),
            concentration_limit=self._get_env_value("CONCENTRATION_LIMIT", 0.2, float)
        )
    
    def _load_trading_config(self) -> TradingConfig:
        """Load trading configuration with environment integration"""
        return TradingConfig(
            symbols=[self._get_env_value("PRIMARY_SYMBOL", "BTCUSDT")],
            update_interval=self._get_env_value("UPDATE_INTERVAL", 30, int),
            min_validation_score=self._get_env_value("MIN_VALIDATION_SCORE", 0.7, float),
            enable_live_trading=self._get_env_value("ENABLE_LIVE_TRADING", False, bool),
            broker_api_key=self._get_env_value("BINANCE_API_KEY"),
            broker_secret=self._get_env_value("BINANCE_SECRET"),
            broker_testnet=self._get_env_value("BINANCE_TESTNET", True, bool)
        )
    
    def _load_data_config(self) -> DataConfig:
        """Load enhanced data configuration with environment integration"""
        return DataConfig(
            binance_api_key=self._get_env_value("BINANCE_API_KEY"),
            binance_secret=self._get_env_value("BINANCE_SECRET"),
            binance_testnet=self._get_env_value("BINANCE_TESTNET", True, bool),
            primary_symbol=self._get_env_value("PRIMARY_SYMBOL", "BTCUSDT"),
            timeframe=self._get_env_value("TIMEFRAME", "15m"),
            update_interval=self._get_env_value("UPDATE_INTERVAL", 30, int),
            historical_days=self._get_env_value("HISTORICAL_DAYS", 30, int),
            min_data_quality=self._get_env_value("MIN_DATA_QUALITY", 0.8, float),
            max_price_gap=self._get_env_value("MAX_PRICE_GAP", 0.05, float),
            volume_threshold_percentile=self._get_env_value("VOLUME_THRESHOLD_PERCENTILE", 0.1, float),
            cache_enabled=self._get_env_value("CACHE_ENABLED", True, bool),
            cache_expiry=self._get_env_value("CACHE_EXPIRY", 3600, int),
            batch_size=self._get_env_value("DATA_BATCH_SIZE", 1000, int),
            max_retries=self._get_env_value("DATA_MAX_RETRIES", 3, int)
        )
    
    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration"""
        return MonitoringConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            metrics_port=int(os.getenv("METRICS_PORT", "8080")),
            prometheus_enabled=os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
        )
    
    def _load_feature_config(self) -> FeatureConfig:
        """Load feature engineering configuration"""
        return FeatureConfig(
            enable_legacy_rl=os.getenv("ENABLE_LEGACY_RL", "true").lower() == "true",
            enable_technical_features=os.getenv("ENABLE_TECHNICAL_FEATURES", "true").lower() == "true",
            enable_microstructure=os.getenv("ENABLE_MICROSTRUCTURE", "true").lower() == "true"
        )
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment"""
        return DatabaseConfig(
            url=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/agent_trading"),
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration from environment"""
        return RedisConfig(
            url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            db=int(os.getenv("REDIS_DB", "0")),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "10")),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
        )
    
    def _load_docker_config(self) -> DockerConfig:
        """Load Docker-specific configuration from environment"""
        return DockerConfig(
            app_env=os.getenv("APP_ENV", "development"),
            container_name=os.getenv("CONTAINER_NAME", "agent-trading-app"),
            network_name=os.getenv("NETWORK_NAME", "agent-trading-network"),
            data_volume=os.getenv("DATA_VOLUME", "/app/data"),
            logs_volume=os.getenv("LOGS_VOLUME", "/app/logs"),
            models_volume=os.getenv("MODELS_VOLUME", "/app/models")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format"""
        return {
            'llm': {
                'model': self.llm.model,
                'api_key': self.llm.api_key,
                'base_url': self.llm.base_url,
                'temperature': self.llm.temperature,
                'max_iterations': self.llm.max_iterations
            },
            'execution': {
                'max_position_size': self.risk.max_position_size,
                'max_daily_loss': self.risk.max_daily_loss,
                'max_portfolio_exposure': self.risk.max_portfolio_exposure,
                'var_limit': self.risk.var_limit,
                'concentration_limit': self.risk.concentration_limit
            },
            'symbols': self.trading.symbols,
            'update_interval': self.trading.update_interval,
            'min_validation_score': self.trading.min_validation_score,
            'data_sources': {
                'financial_api_key': self.data.financial_api_key,
                'binance_api_key': self.data.binance_api_key,
                'enable_real_data': self.data.enable_real_data
            },
            'features': {
                'enable_legacy_rl': self.features.enable_legacy_rl,
                'enable_technical_features': self.features.enable_technical_features,
                'enable_microstructure': self.features.enable_microstructure
            },
            'database': {
                'url': self.database.url,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'pool_timeout': self.database.pool_timeout
            },
            'redis': {
                'url': self.redis.url,
                'db': self.redis.db,
                'max_connections': self.redis.max_connections,
                'socket_timeout': self.redis.socket_timeout
            },
            'docker': {
                'app_env': self.docker.app_env,
                'container_name': self.docker.container_name,
                'network_name': self.docker.network_name,
                'data_volume': self.docker.data_volume,
                'logs_volume': self.docker.logs_volume,
                'models_volume': self.docker.models_volume
            }
        }
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues"""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Check required API keys for live trading
        if self.trading.enable_live_trading:
            if not self.llm.api_key:
                issues['errors'].append("LLM_API_KEY required for live trading")
            
            if not self.data.binance_api_key or not self.data.binance_secret:
                issues['warnings'].append("Binance API keys missing - using mock broker")
        
        # Validate risk parameters
        if self.risk.max_daily_loss <= 0:
            issues['errors'].append("MAX_DAILY_LOSS must be positive")
        
        if self.risk.concentration_limit <= 0 or self.risk.concentration_limit > 1:
            issues['errors'].append("CONCENTRATION_LIMIT must be between 0 and 1")
        
        # Validate trading parameters
        if self.trading.update_interval < 1:
            issues['errors'].append("UPDATE_INTERVAL must be at least 1 second")
        
        if not self.trading.symbols:
            issues['errors'].append("No trading symbols configured")
        
        # Validate score thresholds
        if self.trading.min_validation_score < 0 or self.trading.min_validation_score > 1:
            issues['errors'].append("MIN_VALIDATION_SCORE must be between 0 and 1")
        
        return issues
    
    def _load_advanced_frameworks_config(self) -> AdvancedFrameworksConfig:
        """Load advanced 5-phase frameworks configuration"""
        return AdvancedFrameworksConfig(
            # Phase 1: Bayesian
            bayesian_enabled=self._get_env_value("BAYESIAN_ENABLED", True, bool),
            mcmc_draws=self._get_env_value("MCMC_DRAWS", 2000, int),
            mcmc_tune=self._get_env_value("MCMC_TUNE", 1000, int),
            hierarchical_levels=self._get_env_value("HIERARCHICAL_LEVELS", 3, int),
            
            # Phase 2: QuantLib
            quantlib_enabled=self._get_env_value("QUANTLIB_ENABLED", True, bool),
            risk_free_rate=self._get_env_value("RISK_FREE_RATE", 0.05, float),
            default_volatility=self._get_env_value("DEFAULT_VOLATILITY", 0.25, float),
            option_exercise_style=self._get_env_value("OPTION_EXERCISE_STYLE", "european"),
            
            # Phase 3: Portfolio
            portfolio_enabled=self._get_env_value("PORTFOLIO_ENABLED", True, bool),
            risk_aversion=self._get_env_value("RISK_AVERSION", 2.0, float),
            max_weight=self._get_env_value("MAX_WEIGHT", 0.4, float),
            min_weight=self._get_env_value("MIN_WEIGHT", 0.05, float),
            rebalance_threshold=self._get_env_value("REBALANCE_THRESHOLD", 0.05, float),
            
            # Phase 4: Time Series
            timeseries_enabled=self._get_env_value("TIMESERIES_ENABLED", True, bool),
            garch_model=self._get_env_value("GARCH_MODEL", "GARCH(1,1)"),
            arch_lags=self._get_env_value("ARCH_LAGS", 1, int),
            garch_lags=self._get_env_value("GARCH_LAGS", 1, int),
            min_periods=self._get_env_value("MIN_PERIODS", 100, int),
            
            # Phase 5: ML/AI
            ml_enabled=self._get_env_value("ML_ENABLED", True, bool),
            ensemble_size=self._get_env_value("ENSEMBLE_SIZE", 5, int),
            train_test_split=self._get_env_value("TRAIN_TEST_SPLIT", 0.8, float),
            cross_validation_folds=self._get_env_value("CV_FOLDS", 5, int),
            feature_selection=self._get_env_value("FEATURE_SELECTION", True, bool),
            
            # Physics Models
            physics_enabled=self._get_env_value("PHYSICS_ENABLED", True, bool),
            entropy_window=self._get_env_value("ENTROPY_WINDOW", 50, int),
            hurst_window=self._get_env_value("HURST_WINDOW", 100, int),
            lyapunov_window=self._get_env_value("LYAPUNOV_WINDOW", 50, int),
            
            # Market Microstructure
            microstructure_enabled=self._get_env_value("MICROSTRUCTURE_ENABLED", True, bool),
            tick_analysis=self._get_env_value("TICK_ANALYSIS", True, bool),
            order_book_depth=self._get_env_value("ORDER_BOOK_DEPTH", 10, int)
        )


# Global configuration instance
config = ConfigManager()
