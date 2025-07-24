"""
Dataset Management Module
BTC-focused data collection, cleaning, and feature engineering

This module handles:
- Binance API integration for BTC/USDT 15m candles
- Data cleaning and validation with pandas/numpy
- Feature engineering for quantitative models
- Database storage integration
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from .config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class OHLCVData:
    """OHLCV candle data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"


@dataclass
class ProcessedFeatures:
    """Processed features for quantitative analysis - enhanced for advanced frameworks"""
    timestamp: datetime
    symbol: str
    
    # Basic price features
    price_return: float
    log_return: float
    volatility: float
    
    # Technical features
    sma_20: float
    ema_20: float
    rsi_14: float
    bb_upper: float
    bb_lower: float
    bb_position: float
    
    # Volume features
    volume_sma: float
    volume_ratio: float
    
    # Statistical features
    z_score_20: float
    momentum_5: float
    
    # Advanced features for 5-phase frameworks
    # Bayesian features
    price_level_probability: Optional[float] = None
    regime_probability: Optional[float] = None
    bayesian_confidence: Optional[float] = None
    
    # QuantLib features
    implied_volatility: Optional[float] = None
    option_delta: Optional[float] = None
    option_gamma: Optional[float] = None
    time_value: Optional[float] = None
    
    # Portfolio features
    correlation_btc_eth: Optional[float] = None
    portfolio_beta: Optional[float] = None
    diversification_ratio: Optional[float] = None
    
    # GARCH/Time series features
    conditional_volatility: Optional[float] = None
    volatility_forecast: Optional[float] = None
    arch_effect: Optional[float] = None
    
    # ML features
    ml_prediction: Optional[float] = None
    ensemble_confidence: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    # Physics-based features
    entropy_score: Optional[float] = None
    hurst_exponent: Optional[float] = None
    lyapunov_exponent: Optional[float] = None
    fractal_dimension: Optional[float] = None
    
    # Microstructure features
    bid_ask_spread: Optional[float] = None
    order_flow_imbalance: Optional[float] = None
    market_impact: Optional[float] = None
    
    # Quality metrics
    data_quality: float = 0.0
    feature_confidence: float = 0.0
    frameworks_applied: Optional[List[str]] = None


class BinanceDataCollector:
    """
    Binance API integration for BTC data collection
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.symbol = "BTC/USDT"
        self.timeframe = "15m"
        
        # Initialize Binance connection
        self.exchange = ccxt.binance({
            'apiKey': config.data.binance_api_key,
            'secret': config.data.binance_secret,
            'sandbox': config.data.binance_testnet,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        self.logger = logging.getLogger(f"{__name__}.BinanceCollector")
    
    async def fetch_latest_candles(self, limit: int = 100) -> List[OHLCVData]:
        """
        Fetch latest BTC/USDT 15m candles from Binance
        """
        try:
            # Fetch OHLCV data
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.exchange.fetch_ohlcv,
                self.symbol,
                self.timeframe,
                None,  # since timestamp
                limit
            )
            
            # Convert to OHLCVData objects
            candles = []
            for candle in ohlcv:
                timestamp = datetime.fromtimestamp(candle[0] / 1000)
                
                ohlcv_data = OHLCVData(
                    timestamp=timestamp,
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5])
                )
                candles.append(ohlcv_data)
            
            self.logger.info(f"Fetched {len(candles)} candles from Binance")
            return candles
            
        except Exception as e:
            self.logger.error(f"Failed to fetch Binance data: {e}")
            raise
    
    async def fetch_historical_data(self, days: int = 30) -> List[OHLCVData]:
        """
        Fetch historical BTC data for backtesting
        """
        try:
            # Calculate how many candles we need (96 candles per day for 15m)
            candles_needed = days * 96
            
            # Binance limits to 1000 candles per request
            all_candles = []
            
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            while len(all_candles) < candles_needed:
                batch_size = min(1000, candles_needed - len(all_candles))
                
                ohlcv = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.exchange.fetch_ohlcv,
                    self.symbol,
                    self.timeframe,
                    since,
                    batch_size
                )
                
                if not ohlcv:
                    break
                
                # Convert to OHLCVData objects
                for candle in ohlcv:
                    timestamp = datetime.fromtimestamp(candle[0] / 1000)
                    
                    ohlcv_data = OHLCVData(
                        timestamp=timestamp,
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=float(candle[5])
                    )
                    all_candles.append(ohlcv_data)
                
                # Update since timestamp for next batch
                since = ohlcv[-1][0] + 1
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            self.logger.info(f"Fetched {len(all_candles)} historical candles")
            return all_candles[:candles_needed]
            
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {e}")
            raise


class DataProcessor:
    """
    Pandas/Numpy data processing and feature engineering
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataProcessor")
    
    def clean_ohlcv_data(self, candles: List[OHLCVData]) -> pd.DataFrame:
        """
        Clean and validate OHLCV data using pandas
        """
        try:
            # Convert to DataFrame
            data = []
            for candle in candles:
                data.append({
                    'timestamp': candle.timestamp,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume,
                    'symbol': candle.symbol
                })
            
            df = pd.DataFrame(data)
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Basic validation
            df = self._validate_ohlcv(df)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Fill missing timestamps (if any)
            df = self._fill_missing_timestamps(df)
            
            # Data quality checks
            df['data_quality'] = self._calculate_data_quality(df)
            
            self.logger.info(f"Cleaned {len(df)} candles, avg quality: {df['data_quality'].mean():.3f}")
            return df
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            raise
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data integrity"""
        
        # Remove rows with invalid prices
        df = df[
            (df['open'] > 0) & 
            (df['high'] > 0) & 
            (df['low'] > 0) & 
            (df['close'] > 0) &
            (df['volume'] >= 0)
        ].copy()
        
        # Check OHLC relationships
        valid_ohlc = (
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['high'] >= df['low'])
        )
        
        df = df[valid_ohlc].copy()
        
        # Remove extreme outliers (more than 10% price change in 15 minutes)
        df['price_change'] = df['close'].pct_change()
        df = df[abs(df['price_change']) <= 0.10].copy()
        
        return df.drop('price_change', axis=1)
    
    def _fill_missing_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing 15-minute intervals"""
        
        if len(df) == 0:
            return df
        
        # Create complete time series
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        
        complete_index = pd.date_range(
            start=start_time,
            end=end_time,
            freq='15min'
        )
        
        # Reindex and forward fill missing values
        df_indexed = df.set_index('timestamp')
        df_complete = df_indexed.reindex(complete_index)
        
        # Forward fill OHLC, but mark as interpolated
        df_complete = df_complete.fillna(method='ffill')
        
        return df_complete.reset_index().rename(columns={'index': 'timestamp'})
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate data quality score for each candle"""
        
        quality_scores = pd.Series(1.0, index=df.index)
        
        # Penalize for zero volume
        quality_scores[df['volume'] == 0] *= 0.5
        
        # Penalize for very low volume (below 10th percentile)
        volume_threshold = df['volume'].quantile(0.1)
        quality_scores[df['volume'] < volume_threshold] *= 0.8
        
        # Penalize for price gaps > 1%
        price_gaps = abs(df['open'] / df['close'].shift(1) - 1)
        quality_scores[price_gaps > 0.01] *= 0.9
        
        return quality_scores
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer trading features using numpy/pandas
        """
        try:
            df = df.copy()
            
            # Price-based features
            df['price_return'] = df['close'].pct_change()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volatility (rolling standard deviation)
            df['volatility'] = df['log_return'].rolling(20).std()
            
            # Moving averages
            df['sma_20'] = df['close'].rolling(20).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            
            # RSI
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(df['close'], 20, 2)
            df['bb_upper'] = bb_data['upper']
            df['bb_lower'] = bb_data['lower']
            df['bb_position'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Statistical features
            df['z_score_20'] = (df['close'] - df['sma_20']) / df['close'].rolling(20).std()
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            
            # Feature confidence (based on data availability)
            df['feature_confidence'] = self._calculate_feature_confidence(df)
            
            # Remove initial NaN rows
            df = df.dropna().reset_index(drop=True)
            
            self.logger.info(f"Engineered features for {len(df)} candles")
            return df
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev)
        }
    
    def _calculate_feature_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate confidence score for features"""
        
        confidence = pd.Series(1.0, index=df.index)
        
        # Reduce confidence for recent data (less historical context)
        confidence.iloc[-20:] *= np.linspace(0.8, 1.0, 20)
        
        # Reduce confidence during high volatility periods
        high_vol_mask = df['volatility'] > df['volatility'].quantile(0.9)
        confidence[high_vol_mask] *= 0.9
        
        return confidence


class DatabaseManager:
    """
    PostgreSQL database integration for data storage
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DatabaseManager")
        
        # Initialize Redis for caching
        self.redis_client = redis.from_url(config.redis.url)
        
    def get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(
            self.config.database.url,
            cursor_factory=RealDictCursor
        )
    
    async def store_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Store OHLCV data in database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Prepare data for insertion
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO trading.market_data 
                    (symbol, timestamp, open, high, low, close, volume, timeframe)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """, (
                    row['symbol'],
                    row['timestamp'],
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume'],
                    '15m'
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Stored {len(df)} candles in database")
            return True
            
        except Exception as e:
            self.logger.error(f"Database storage failed: {e}")
            return False
    
    async def get_latest_timestamp(self, symbol: str = "BTCUSDT") -> Optional[datetime]:
        """Get latest timestamp for a symbol"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT MAX(timestamp) as latest_timestamp
                FROM trading.market_data
                WHERE symbol = %s AND timeframe = '15m'
            """, (symbol,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return result['latest_timestamp'] if result else None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest timestamp: {e}")
            return None
    
    async def cache_processed_features(self, df: pd.DataFrame, symbol: str = "BTCUSDT"):
        """Cache processed features in Redis"""
        try:
            # Get latest features
            latest_features = df.iloc[-1].to_dict()
            
            # Store in Redis with 1-hour expiry
            cache_key = f"features:{symbol}:15m"
            self.redis_client.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(latest_features, default=str)
            )
            
            self.logger.info(f"Cached features for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Feature caching failed: {e}")


class BTCDataManager:
    """
    Main BTC data management orchestrator - enhanced for advanced 5-phase frameworks
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.collector = BinanceDataCollector(config)
        self.processor = DataProcessor()
        self.db_manager = DatabaseManager(config)
        self.logger = logging.getLogger(f"{__name__}.BTCDataManager")
        
        # Advanced frameworks integration
        self.advanced_frameworks = {}
        self._initialize_advanced_frameworks()
    
    def _initialize_advanced_frameworks(self):
        """Initialize advanced frameworks for enhanced data processing"""
        try:
            # Try to import and initialize advanced frameworks
            from .quant_models import (
                BayesianTradingFramework,
                QuantLibFinancialEngineering,
                AdvancedPortfolioOptimization,
                AdvancedTimeSeriesAnalysis,
                AdvancedMLTradingFramework,
                AdvancedPhysicsModels,
                MarketMicrostructure
            )
            
            # Initialize frameworks based on configuration
            if self.config.advanced_frameworks.bayesian_enabled:
                self.advanced_frameworks['bayesian'] = BayesianTradingFramework()
                
            if self.config.advanced_frameworks.quantlib_enabled:
                self.advanced_frameworks['quantlib'] = QuantLibFinancialEngineering()
                
            if self.config.advanced_frameworks.portfolio_enabled:
                self.advanced_frameworks['portfolio'] = AdvancedPortfolioOptimization()
                
            if self.config.advanced_frameworks.timeseries_enabled:
                self.advanced_frameworks['timeseries'] = AdvancedTimeSeriesAnalysis()
                
            if self.config.advanced_frameworks.ml_enabled:
                self.advanced_frameworks['ml'] = AdvancedMLTradingFramework()
                
            if self.config.advanced_frameworks.physics_enabled:
                self.advanced_frameworks['physics'] = AdvancedPhysicsModels()
                
            if self.config.advanced_frameworks.microstructure_enabled:
                self.advanced_frameworks['microstructure'] = MarketMicrostructure()
            
            self.logger.info(f"✅ Initialized {len(self.advanced_frameworks)} advanced frameworks for data processing")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Advanced frameworks not available: {e}")
            self.advanced_frameworks = {}
    
    async def initialize(self):
        """Initialize the BTC data manager"""
        self.logger.info("BTCDataManager initialization completed")
        return True
    
    async def fetch_latest_data(self) -> Dict[str, Any]:
        """Fetch latest market data for trading with advanced framework enhancements"""
        try:
            # Fetch latest candles
            candles = await self.collector.fetch_latest_candles(limit=100)  # Increased for advanced analysis
            
            if not candles:
                return {
                    'candles': [],
                    'latest_price': 0,
                    'timeframe': '15m',
                    'symbol': 'BTCUSDT',
                    'advanced_features': {}
                }
            
            # Convert to dict format
            candles_data = []
            for candle in candles:
                candles_data.append({
                    'timestamp': candle.timestamp.isoformat(),
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume
                })
            
            # Calculate basic features
            prices = [c.close for c in candles]
            volumes = [c.volume for c in candles]
            
            # Calculate returns for advanced analysis
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            # Generate advanced features using available frameworks
            advanced_features = await self._generate_advanced_features(prices, returns, volumes)
            
            # Calculate price change
            price_change_24h = 0
            if len(prices) >= 96:  # 24 hours of 15m candles
                price_change_24h = ((prices[-1] - prices[-96]) / prices[-96]) * 100
            
            result = {
                'candles': candles_data,
                'latest_price': candles[-1].close,
                'price_change_24h': price_change_24h,
                'volume_24h': sum(volumes[-96:]) if len(volumes) >= 96 else sum(volumes),
                'timeframe': '15m',
                'symbol': 'BTCUSDT',
                'advanced_features': advanced_features,
                'data_quality_score': self._calculate_data_quality(candles)
            }
            
            self.logger.info(f"Fetched {len(candles)} candles with {len(advanced_features)} advanced features")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching latest data: {e}")
            return {
                'candles': [],
                'latest_price': 0,
                'price_change_24h': 0,
                'volume_24h': 0,
                'timeframe': '15m',
                'symbol': 'BTCUSDT',
                'advanced_features': {},
                'error': str(e)
            }
    
    async def _generate_advanced_features(self, prices: List[float], returns: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Generate advanced features using available frameworks"""
        features = {}
        
        try:
            # Bayesian features
            if 'bayesian' in self.advanced_frameworks and len(prices) > 50:
                bayesian_framework = self.advanced_frameworks['bayesian']
                bayesian_result = bayesian_framework.hierarchical_model_analysis(prices[-50:])
                features['bayesian'] = bayesian_result
            
            # QuantLib features
            if 'quantlib' in self.advanced_frameworks:
                quantlib_framework = self.advanced_frameworks['quantlib']
                current_price = prices[-1]
                option_analysis = quantlib_framework.black_scholes_analysis(
                    spot=current_price,
                    strike=current_price * 1.05,
                    risk_free_rate=self.config.advanced_frameworks.risk_free_rate,
                    volatility=self.config.advanced_frameworks.default_volatility,
                    time_to_expiry=0.25
                )
                features['quantlib'] = option_analysis
            
            # Time series features
            if 'timeseries' in self.advanced_frameworks and len(returns) > 50:
                timeseries_framework = self.advanced_frameworks['timeseries']
                garch_result = timeseries_framework.garch_volatility_modeling(returns[-50:])
                features['timeseries'] = garch_result
            
            # ML features
            if 'ml' in self.advanced_frameworks and len(prices) > 20:
                ml_framework = self.advanced_frameworks['ml']
                
                # Create feature matrix
                feature_matrix = []
                targets = []
                for i in range(10, len(prices)-1):
                    feature_row = prices[i-10:i]
                    target = 1 if prices[i+1] > prices[i] else 0
                    feature_matrix.append(feature_row)
                    targets.append(target)
                
                if len(feature_matrix) > 10:
                    ml_result = ml_framework.ensemble_prediction(feature_matrix, targets)
                    features['ml'] = ml_result
            
            # Physics features
            if 'physics' in self.advanced_frameworks and len(returns) > 20:
                physics_framework = self.advanced_frameworks['physics']
                physics_result = physics_framework.comprehensive_physics_analysis(returns[-50:] if len(returns) >= 50 else returns)
                features['physics'] = physics_result
            
            # Microstructure features
            if 'microstructure' in self.advanced_frameworks:
                microstructure_framework = self.advanced_frameworks['microstructure']
                microstructure_result = microstructure_framework.analyze_market_impact(
                    prices[-20:] if len(prices) >= 20 else prices,
                    volumes[-20:] if len(volumes) >= 20 else volumes
                )
                features['microstructure'] = microstructure_result
            
        except Exception as e:
            self.logger.error(f"Error generating advanced features: {e}")
            features['error'] = str(e)
        
        return features
    
    def _calculate_data_quality(self, candles: List[OHLCVData]) -> float:
        """Calculate data quality score"""
        if not candles:
            return 0.0
        
        quality_score = 1.0
        
        # Check for missing data
        expected_interval = timedelta(minutes=15)
        for i in range(1, len(candles)):
            actual_interval = candles[i].timestamp - candles[i-1].timestamp
            if abs(actual_interval.total_seconds() - expected_interval.total_seconds()) > 60:
                quality_score -= 0.1
        
        # Check for zero volumes
        zero_volume_count = sum(1 for c in candles if c.volume == 0)
        quality_score -= (zero_volume_count / len(candles)) * 0.5
        
        # Check for price anomalies
        prices = [c.close for c in candles]
        if len(prices) > 1:
            price_changes = [abs((prices[i] - prices[i-1]) / prices[i-1]) for i in range(1, len(prices))]
            extreme_changes = sum(1 for change in price_changes if change > 0.1)  # 10% moves
            quality_score -= (extreme_changes / len(price_changes)) * 0.3
        
        return max(0.0, min(1.0, quality_score))
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("BTCDataManager cleanup completed")
        return True
    
    async def update_btc_data(self) -> Dict[str, Any]:
        """
        Complete BTC data update pipeline
        """
        try:
            start_time = time.time()
            
            # 1. Fetch latest data from Binance
            self.logger.info("Fetching latest BTC data from Binance...")
            candles = await self.collector.fetch_latest_candles(limit=100)
            
            if not candles:
                return {'success': False, 'error': 'No data received from Binance'}
            
            # 2. Clean and validate data
            self.logger.info("Processing and cleaning data...")
            df = self.processor.clean_ohlcv_data(candles)
            
            # 3. Engineer features
            df_with_features = self.processor.engineer_features(df)
            
            # 4. Store in database
            await self.db_manager.store_ohlcv_data(df_with_features)
            
            # 5. Cache latest features
            await self.db_manager.cache_processed_features(df_with_features)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'candles_processed': len(df_with_features),
                'latest_timestamp': df_with_features['timestamp'].iloc[-1],
                'latest_price': df_with_features['close'].iloc[-1],
                'data_quality': df_with_features['data_quality'].mean(),
                'processing_time_ms': processing_time * 1000
            }
            
        except Exception as e:
            self.logger.error(f"BTC data update failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_btc_features_for_trading(self) -> Optional[ProcessedFeatures]:
        """
        Get processed BTC features for trading decisions
        """
        try:
            # Try cache first
            cache_key = "features:BTCUSDT:15m"
            cached = self.db_manager.redis_client.get(cache_key)
            
            if cached:
                feature_dict = json.loads(cached)
                return ProcessedFeatures(**feature_dict)
            
            # Fallback to database
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM trading.market_data
                WHERE symbol = 'BTCUSDT' AND timeframe = '15m'
                ORDER BY timestamp DESC
                LIMIT 50
            """)
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return None
            
            # Convert to DataFrame and process
            df = pd.DataFrame(rows)
            df_with_features = self.processor.engineer_features(df)
            
            # Return latest features
            latest = df_with_features.iloc[-1]
            
            return ProcessedFeatures(
                timestamp=latest['timestamp'],
                symbol="BTCUSDT",
                price_return=latest['price_return'],
                log_return=latest['log_return'],
                volatility=latest['volatility'],
                sma_20=latest['sma_20'],
                ema_20=latest['ema_20'],
                rsi_14=latest['rsi_14'],
                bb_upper=latest['bb_upper'],
                bb_lower=latest['bb_lower'],
                bb_position=latest['bb_position'],
                volume_sma=latest['volume_sma'],
                volume_ratio=latest['volume_ratio'],
                z_score_20=latest['z_score_20'],
                momentum_5=latest['momentum_5'],
                data_quality=latest['data_quality'],
                feature_confidence=latest['feature_confidence']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get BTC features: {e}")
            return None


# Main interface for other modules
async def get_btc_data_manager(config: ConfigManager) -> BTCDataManager:
    """Factory function to create BTC data manager"""
    return BTCDataManager(config)
