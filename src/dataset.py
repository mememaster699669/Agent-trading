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
from .config import Config

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
    """Processed features for quantitative analysis"""
    timestamp: datetime
    symbol: str
    
    # Price features
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
    
    # Quality metrics
    data_quality: float
    feature_confidence: float


class BinanceDataCollector:
    """
    Binance API integration for BTC data collection
    """
    
    def __init__(self, config: Config):
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
    
    def __init__(self, config: Config):
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
    Main BTC data management orchestrator
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.collector = BinanceDataCollector(config)
        self.processor = DataProcessor()
        self.db_manager = DatabaseManager(config)
        self.logger = logging.getLogger(f"{__name__}.BTCDataManager")
    
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
async def get_btc_data_manager(config: Config) -> BTCDataManager:
    """Factory function to create BTC data manager"""
    return BTCDataManager(config)
