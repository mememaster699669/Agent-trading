"""
Legacy Integration Module
Integrates existing RL models and technical analysis as supporting components

This module bridges the new quantitative system with existing components
from the bot-gold-VHL system, using them as feature generators rather
than primary decision makers.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# Mock imports (would import from actual legacy system)
# from stable_baselines3 import PPO
# from trading_rl import TradingEnv


class LegacyRLIntegration:
    """
    Integration with existing RL models for pattern recognition
    """
    
    def __init__(self, model_path: str = "rl_model.zip"):
        self.model_path = model_path
        self.model = None
        self.logger = logging.getLogger(f"{__name__}.LegacyRL")
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained RL model"""
        try:
            # Mock implementation - would load actual model
            self.model = MockRLModel()
            self.logger.info("Legacy RL model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load RL model: {e}")
            self.model = None
    
    def get_rl_features(self, price_data: np.ndarray) -> Dict[str, float]:
        """
        Extract RL-based features for pattern recognition
        
        Returns features, not trading decisions
        """
        if self.model is None or len(price_data) < 10:
            return {
                'rl_momentum_score': 0.0,
                'rl_pattern_confidence': 0.0,
                'rl_volatility_regime': 0.0
            }
        
        try:
            # Create environment state
            returns = np.diff(np.log(price_data))
            recent_returns = returns[-10:]  # Last 10 periods
            
            # Mock RL model prediction (would use actual model)
            prediction = self.model.predict_features(recent_returns)
            
            return {
                'rl_momentum_score': float(prediction.get('momentum', 0.0)),
                'rl_pattern_confidence': float(prediction.get('confidence', 0.0)),
                'rl_volatility_regime': float(prediction.get('volatility_regime', 0.0))
            }
            
        except Exception as e:
            self.logger.error(f"RL feature extraction failed: {e}")
            return {
                'rl_momentum_score': 0.0,
                'rl_pattern_confidence': 0.0,
                'rl_volatility_regime': 0.0
            }


class LegacyTechnicalAnalysis:
    """
    Integration with existing technical analysis as feature generators
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.LegacyTA")
    
    def extract_technical_features(self, price_data: np.ndarray, volume_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract technical indicators as features (not signals)
        
        These are used as inputs to the quantitative models, not as direct trading signals
        """
        if len(price_data) < 20:
            return self._empty_features()
        
        try:
            prices = pd.Series(price_data)
            
            # Moving averages (trend features)
            sma_20 = prices.rolling(20).mean().iloc[-1]
            sma_50 = prices.rolling(min(50, len(prices))).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            ma_trend_20 = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
            ma_trend_50 = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
            
            # Volatility features
            returns = prices.pct_change().dropna()
            volatility_20 = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0
            
            # RSI-style momentum (normalized)
            gains = returns[returns > 0].rolling(14).sum().iloc[-1] if len(returns) >= 14 else 0
            losses = abs(returns[returns < 0]).rolling(14).sum().iloc[-1] if len(returns) >= 14 else 0
            rsi_momentum = gains / (gains + losses) if (gains + losses) > 0 else 0.5
            
            # Bollinger Band position
            bb_mean = prices.rolling(20).mean().iloc[-1]
            bb_std = prices.rolling(20).std().iloc[-1]
            bb_position = (current_price - bb_mean) / (2 * bb_std) if bb_std > 0 else 0
            bb_position = np.clip(bb_position, -1, 1)  # Normalize to [-1, 1]
            
            # Volume features (if available)
            volume_features = self._extract_volume_features(volume_data) if volume_data is not None else {}
            
            features = {
                'ma_trend_20': float(ma_trend_20),
                'ma_trend_50': float(ma_trend_50),
                'volatility_20': float(volatility_20),
                'rsi_momentum': float(rsi_momentum),
                'bb_position': float(bb_position),
                'price_momentum_5': float(self._price_momentum(prices, 5)),
                'price_momentum_10': float(self._price_momentum(prices, 10)),
                **volume_features
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Technical feature extraction failed: {e}")
            return self._empty_features()
    
    def _price_momentum(self, prices: pd.Series, periods: int) -> float:
        """Calculate price momentum over N periods"""
        if len(prices) < periods + 1:
            return 0.0
        
        current = prices.iloc[-1]
        past = prices.iloc[-periods-1]
        return (current - past) / past if past > 0 else 0.0
    
    def _extract_volume_features(self, volume_data: np.ndarray) -> Dict[str, float]:
        """Extract volume-based features"""
        if len(volume_data) < 10:
            return {'volume_trend': 0.0, 'volume_spike': 0.0}
        
        volumes = pd.Series(volume_data)
        
        # Volume trend
        recent_avg = volumes.rolling(5).mean().iloc[-1]
        longer_avg = volumes.rolling(min(20, len(volumes))).mean().iloc[-1]
        volume_trend = (recent_avg - longer_avg) / longer_avg if longer_avg > 0 else 0
        
        # Volume spike detection
        volume_std = volumes.rolling(min(20, len(volumes))).std().iloc[-1]
        volume_mean = volumes.rolling(min(20, len(volumes))).mean().iloc[-1]
        current_volume = volumes.iloc[-1]
        
        volume_spike = (current_volume - volume_mean) / volume_std if volume_std > 0 else 0
        volume_spike = np.clip(volume_spike, -3, 3)  # Limit to reasonable range
        
        return {
            'volume_trend': float(volume_trend),
            'volume_spike': float(volume_spike)
        }
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature set"""
        return {
            'ma_trend_20': 0.0,
            'ma_trend_50': 0.0,
            'volatility_20': 0.0,
            'rsi_momentum': 0.5,  # Neutral
            'bb_position': 0.0,
            'price_momentum_5': 0.0,
            'price_momentum_10': 0.0,
            'volume_trend': 0.0,
            'volume_spike': 0.0
        }


class MockRLModel:
    """Mock RL model for testing purposes"""
    
    def predict_features(self, returns: np.ndarray) -> Dict[str, float]:
        """Mock feature prediction"""
        if len(returns) == 0:
            return {'momentum': 0.0, 'confidence': 0.0, 'volatility_regime': 0.0}
        
        # Simple feature calculations
        momentum = np.mean(returns) / (np.std(returns) + 1e-8)
        momentum = np.tanh(momentum * 2)  # Normalize to [-1, 1]
        
        confidence = min(1.0, len(returns) / 10.0)  # Higher confidence with more data
        
        volatility = np.std(returns)
        volatility_regime = np.tanh(volatility * 10)  # Normalize volatility
        
        return {
            'momentum': momentum,
            'confidence': confidence,
            'volatility_regime': volatility_regime
        }


class FeatureEngineer:
    """
    Combine quantitative features with legacy features for enhanced modeling
    """
    
    def __init__(self):
        self.rl_integration = LegacyRLIntegration()
        self.technical_analysis = LegacyTechnicalAnalysis()
        self.logger = logging.getLogger(f"{__name__}.FeatureEngineer")
    
    def engineer_features(self, 
                         symbol: str,
                         price_data: np.ndarray,
                         volume_data: Optional[np.ndarray] = None,
                         additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Comprehensive feature engineering combining all sources
        """
        try:
            features = {}
            
            # 1. Quantitative statistical features
            if len(price_data) >= 2:
                returns = np.diff(np.log(price_data))
                
                # Statistical moments
                features['return_mean'] = float(np.mean(returns))
                features['return_std'] = float(np.std(returns))
                features['return_skew'] = float(self._safe_skew(returns))
                features['return_kurtosis'] = float(self._safe_kurtosis(returns))
                
                # Autocorrelation features
                features['autocorr_1'] = float(self._autocorrelation(returns, 1))
                features['autocorr_5'] = float(self._autocorrelation(returns, 5))
                
                # Volatility clustering
                features['vol_clustering'] = float(self._volatility_clustering(returns))
            
            # 2. Legacy RL features
            rl_features = self.rl_integration.get_rl_features(price_data)
            features.update({f"legacy_{k}": v for k, v in rl_features.items()})
            
            # 3. Technical analysis features
            ta_features = self.technical_analysis.extract_technical_features(price_data, volume_data)
            features.update({f"ta_{k}": v for k, v in ta_features.items()})
            
            # 4. Market microstructure features
            if additional_data:
                micro_features = self._extract_microstructure_features(additional_data)
                features.update({f"micro_{k}": v for k, v in micro_features.items()})
            
            # 5. Feature interactions
            interaction_features = self._create_feature_interactions(features)
            features.update(interaction_features)
            
            # 6. Feature scaling and normalization
            normalized_features = self._normalize_features(features)
            
            self.logger.debug(f"Engineered {len(normalized_features)} features for {symbol}")
            return normalized_features
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed for {symbol}: {e}")
            return {}
    
    def _safe_skew(self, data: np.ndarray) -> float:
        """Calculate skewness safely"""
        if len(data) < 3:
            return 0.0
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            # Fallback calculation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
    
    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis safely"""
        if len(data) < 4:
            return 0.0
        try:
            from scipy.stats import kurtosis
            return kurtosis(data)
        except:
            # Fallback calculation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3
    
    def _autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        if len(data) <= lag:
            return 0.0
        
        try:
            c0 = np.dot(data, data) / len(data)
            c_lag = np.dot(data[:-lag], data[lag:]) / len(data[:-lag])
            return c_lag / c0 if c0 != 0 else 0.0
        except:
            return 0.0
    
    def _volatility_clustering(self, returns: np.ndarray) -> float:
        """Measure volatility clustering (GARCH effect)"""
        if len(returns) < 10:
            return 0.0
        
        try:
            # Simple measure: correlation between |returns| and lagged |returns|
            abs_returns = np.abs(returns)
            if len(abs_returns) <= 1:
                return 0.0
            
            return np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
        except:
            return 0.0
    
    def _extract_microstructure_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract market microstructure features"""
        features = {}
        
        # Bid-ask spread
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else market_data.get('price', 0)
        
        if mid > 0:
            features['bid_ask_spread'] = (ask - bid) / mid if ask > bid else 0.0
        else:
            features['bid_ask_spread'] = 0.0
        
        # Volume-based features
        volume = market_data.get('volume', 0)
        features['volume_normalized'] = min(1.0, volume / 1000000)  # Normalize volume
        
        return features
    
    def _create_feature_interactions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction features"""
        interactions = {}
        
        # Momentum × Volatility
        momentum = features.get('ta_rsi_momentum', 0.5)
        volatility = features.get('ta_volatility_20', 0.0)
        interactions['momentum_vol_interaction'] = (momentum - 0.5) * volatility
        
        # Trend × Volume
        trend = features.get('ta_ma_trend_20', 0.0)
        volume_trend = features.get('ta_volume_trend', 0.0)
        interactions['trend_volume_interaction'] = trend * volume_trend
        
        # RL × Technical
        rl_momentum = features.get('legacy_rl_momentum_score', 0.0)
        ta_momentum = features.get('ta_rsi_momentum', 0.5) - 0.5
        interactions['rl_ta_momentum_consensus'] = rl_momentum * ta_momentum
        
        return interactions
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to reasonable ranges"""
        normalized = {}
        
        for feature_name, value in features.items():
            # Clip extreme values
            if 'momentum' in feature_name or 'trend' in feature_name:
                normalized[feature_name] = np.clip(value, -3.0, 3.0)
            elif 'volatility' in feature_name:
                normalized[feature_name] = np.clip(value, 0.0, 2.0)
            elif 'correlation' in feature_name or 'autocorr' in feature_name:
                normalized[feature_name] = np.clip(value, -1.0, 1.0)
            else:
                # General clipping for other features
                normalized[feature_name] = np.clip(value, -5.0, 5.0)
            
            # Handle NaN/Inf values
            if not np.isfinite(normalized[feature_name]):
                normalized[feature_name] = 0.0
        
        return normalized


class LegacyBridge:
    """
    Main bridge class for integrating all legacy components
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.logger = logging.getLogger(f"{__name__}.LegacyBridge")
    
    def enhance_quantitative_analysis(self, 
                                    symbol: str,
                                    price_data: np.ndarray,
                                    volume_data: Optional[np.ndarray] = None,
                                    market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance quantitative analysis with legacy features
        """
        try:
            # Engineer comprehensive feature set
            features = self.feature_engineer.engineer_features(
                symbol, price_data, volume_data, market_data
            )
            
            # Analyze feature quality
            feature_quality = self._assess_feature_quality(features)
            
            # Generate enhanced insights
            insights = self._generate_insights(features)
            
            return {
                'features': features,
                'feature_quality': feature_quality,
                'insights': insights,
                'feature_count': len(features),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Legacy enhancement failed for {symbol}: {e}")
            return {
                'features': {},
                'feature_quality': {'overall_score': 0.0},
                'insights': {},
                'feature_count': 0,
                'error': str(e)
            }
    
    def _assess_feature_quality(self, features: Dict[str, float]) -> Dict[str, float]:
        """Assess the quality of extracted features"""
        if not features:
            return {'overall_score': 0.0}
        
        # Check for feature diversity
        feature_values = list(features.values())
        
        # Penalize if too many features are zero
        non_zero_ratio = sum(1 for v in feature_values if abs(v) > 1e-6) / len(feature_values)
        
        # Check feature variance
        feature_variance = np.var(feature_values) if len(feature_values) > 1 else 0.0
        variance_score = min(1.0, feature_variance)  # Higher variance often indicates more information
        
        # Check for reasonable value ranges
        extreme_values = sum(1 for v in feature_values if abs(v) > 3.0)
        extreme_ratio = extreme_values / len(feature_values)
        range_score = 1.0 - extreme_ratio  # Penalize too many extreme values
        
        overall_score = (non_zero_ratio * 0.4 + variance_score * 0.3 + range_score * 0.3)
        
        return {
            'overall_score': float(overall_score),
            'non_zero_ratio': float(non_zero_ratio),
            'variance_score': float(variance_score),
            'range_score': float(range_score),
            'feature_count': len(features)
        }
    
    def _generate_insights(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate trading insights from features"""
        insights = {}
        
        # Momentum consensus
        momentum_features = [v for k, v in features.items() if 'momentum' in k.lower()]
        if momentum_features:
            insights['momentum_consensus'] = np.mean(momentum_features)
            insights['momentum_strength'] = np.std(momentum_features)
        
        # Volatility regime
        vol_features = [v for k, v in features.items() if 'volatility' in k.lower() or 'vol' in k.lower()]
        if vol_features:
            insights['volatility_regime'] = np.mean(vol_features)
        
        # Trend consistency
        trend_features = [v for k, v in features.items() if 'trend' in k.lower() or 'ma_' in k.lower()]
        if trend_features:
            insights['trend_consistency'] = 1.0 - np.std(trend_features)  # Lower std = more consistent
        
        # Feature-based confidence
        feature_agreement = self._calculate_feature_agreement(features)
        insights['feature_confidence'] = feature_agreement
        
        return insights
    
    def _calculate_feature_agreement(self, features: Dict[str, float]) -> float:
        """Calculate agreement between different feature types"""
        try:
            # Group features by type
            momentum_features = [v for k, v in features.items() if 'momentum' in k.lower()]
            trend_features = [v for k, v in features.items() if 'trend' in k.lower()]
            volatility_features = [v for k, v in features.items() if 'volatility' in k.lower()]
            
            agreements = []
            
            # Check momentum agreement
            if len(momentum_features) > 1:
                momentum_agreement = 1.0 - np.std(momentum_features) / (np.mean(np.abs(momentum_features)) + 1e-6)
                agreements.append(momentum_agreement)
            
            # Check trend agreement
            if len(trend_features) > 1:
                trend_agreement = 1.0 - np.std(trend_features) / (np.mean(np.abs(trend_features)) + 1e-6)
                agreements.append(trend_agreement)
            
            # Overall agreement
            if agreements:
                return float(np.mean(agreements))
            else:
                return 0.5  # Neutral if insufficient features
                
        except Exception:
            return 0.5
