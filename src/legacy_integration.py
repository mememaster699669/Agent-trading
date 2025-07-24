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


class AdvancedLegacyBridge:
    """
    Enhanced bridge between legacy features and advanced frameworks
    Integrates legacy indicators as feature generators for advanced models
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.logger = logging.getLogger(f"{__name__}.AdvancedLegacyBridge")
        
        # Initialize framework availability flags
        self._check_framework_availability()
    
    def _check_framework_availability(self):
        """Check which advanced frameworks are available"""
        try:
            from .environment import (
                bayesian_enabled, quantlib_enabled, portfolio_enabled,
                timeseries_enabled, ml_enabled, physics_enabled, microstructure_enabled
            )
            self.frameworks_available = {
                'bayesian': bayesian_enabled,
                'quantlib': quantlib_enabled,
                'portfolio': portfolio_enabled,
                'timeseries': timeseries_enabled,
                'ml': ml_enabled,
                'physics': physics_enabled,
                'microstructure': microstructure_enabled
            }
        except ImportError:
            self.frameworks_available = {k: False for k in 
                ['bayesian', 'quantlib', 'portfolio', 'timeseries', 'ml', 'physics', 'microstructure']}
    
    def generate_enhanced_features(self, 
                                 symbol: str,
                                 price_data: np.ndarray,
                                 volume_data: Optional[np.ndarray] = None,
                                 market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive features for advanced frameworks
        """
        try:
            # Start with basic legacy features
            base_features = self.feature_engineer.engineer_features(
                symbol, price_data, volume_data, market_data
            )
            
            enhanced_features = {
                'base_features': base_features,
                'framework_specific_features': {},
                'feature_metadata': {
                    'symbol': symbol,
                    'data_points': len(price_data),
                    'timestamp': datetime.now().isoformat(),
                    'frameworks_used': []
                }
            }
            
            # Generate framework-specific feature sets
            if self.frameworks_available.get('bayesian', False):
                enhanced_features['framework_specific_features']['bayesian'] = \
                    self._generate_bayesian_features(price_data, base_features)
                enhanced_features['feature_metadata']['frameworks_used'].append('bayesian')
            
            if self.frameworks_available.get('quantlib', False):
                enhanced_features['framework_specific_features']['quantlib'] = \
                    self._generate_quantlib_features(price_data, market_data)
                enhanced_features['feature_metadata']['frameworks_used'].append('quantlib')
            
            if self.frameworks_available.get('portfolio', False):
                enhanced_features['framework_specific_features']['portfolio'] = \
                    self._generate_portfolio_features(price_data, base_features)
                enhanced_features['feature_metadata']['frameworks_used'].append('portfolio')
            
            if self.frameworks_available.get('timeseries', False):
                enhanced_features['framework_specific_features']['timeseries'] = \
                    self._generate_timeseries_features(price_data, base_features)
                enhanced_features['feature_metadata']['frameworks_used'].append('timeseries')
            
            if self.frameworks_available.get('ml', False):
                enhanced_features['framework_specific_features']['ml'] = \
                    self._generate_ml_features(price_data, base_features)
                enhanced_features['feature_metadata']['frameworks_used'].append('ml')
            
            if self.frameworks_available.get('physics', False):
                enhanced_features['framework_specific_features']['physics'] = \
                    self._generate_physics_features(price_data, base_features)
                enhanced_features['feature_metadata']['frameworks_used'].append('physics')
            
            if self.frameworks_available.get('microstructure', False) and market_data:
                enhanced_features['framework_specific_features']['microstructure'] = \
                    self._generate_microstructure_features(price_data, market_data)
                enhanced_features['feature_metadata']['frameworks_used'].append('microstructure')
            
            self.logger.debug(f"Generated enhanced features for {symbol} using {len(enhanced_features['feature_metadata']['frameworks_used'])} frameworks")
            
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"Enhanced feature generation failed for {symbol}: {e}")
            return {
                'base_features': {},
                'framework_specific_features': {},
                'feature_metadata': {'error': str(e)}
            }
    
    def _generate_bayesian_features(self, price_data: np.ndarray, base_features: Dict[str, float]) -> Dict[str, Any]:
        """Generate features optimized for Bayesian analysis"""
        if len(price_data) < 10:
            return {}
        
        returns = np.diff(np.log(price_data))
        
        # Prior distributions based on technical indicators
        momentum_prior = {
            'mean': base_features.get('ta_rsi_momentum', 0.5) - 0.5,  # Center around 0
            'std': 0.1,
            'distribution': 'normal'
        }
        
        volatility_prior = {
            'alpha': 2.0,  # Shape parameter for gamma distribution
            'beta': 1.0 / max(base_features.get('ta_volatility_20', 0.01), 0.001),
            'distribution': 'gamma'
        }
        
        trend_strength = abs(base_features.get('ta_ma_trend_20', 0.0))
        regime_prior = {
            'trend_probability': min(0.9, 0.5 + trend_strength),
            'mean_reversion_probability': 1 - min(0.9, 0.5 + trend_strength),
            'distribution': 'categorical'
        }
        
        return {
            'priors': {
                'momentum': momentum_prior,
                'volatility': volatility_prior,
                'regime': regime_prior
            },
            'observations': {
                'returns': returns.tolist()[-20:],  # Last 20 observations
                'volatility_proxy': base_features.get('ta_volatility_20', 0.0),
                'trend_proxy': base_features.get('ta_ma_trend_20', 0.0)
            },
            'hierarchical_structure': {
                'symbol_level': True,
                'timeframe_level': True,
                'regime_level': True
            }
        }
    
    def _generate_quantlib_features(self, price_data: np.ndarray, market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate features for QuantLib pricing models"""
        if len(price_data) < 2:
            return {}
        
        current_price = price_data[-1]
        returns = np.diff(np.log(price_data))
        implied_volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Option pricing parameters
        return {
            'underlying_price': float(current_price),
            'implied_volatility': float(implied_volatility),
            'option_scenarios': [
                {
                    'strike': current_price * (1 + delta),
                    'delta': delta,
                    'expected_move': delta * current_price
                }
                for delta in [-0.05, -0.02, 0.0, 0.02, 0.05]
            ],
            'market_regime': {
                'volatility_regime': 'high' if implied_volatility > 0.3 else 'normal' if implied_volatility > 0.15 else 'low',
                'trend_strength': 'strong' if abs(np.mean(returns[-10:])) > 0.01 else 'weak'
            }
        }
    
    def _generate_portfolio_features(self, price_data: np.ndarray, base_features: Dict[str, float]) -> Dict[str, Any]:
        """Generate features for portfolio optimization"""
        if len(price_data) < 10:
            return {}
        
        returns = np.diff(np.log(price_data))
        
        return {
            'return_characteristics': {
                'expected_return': float(np.mean(returns)),
                'volatility': float(np.std(returns)),
                'sharpe_proxy': float(np.mean(returns) / (np.std(returns) + 1e-8)),
                'downside_deviation': float(np.std(returns[returns < 0])) if len(returns[returns < 0]) > 0 else 0.0
            },
            'correlation_proxies': {
                'momentum_persistence': base_features.get('autocorr_1', 0.0),
                'trend_consistency': base_features.get('ta_ma_trend_20', 0.0),
                'volatility_clustering': base_features.get('vol_clustering', 0.0)
            },
            'risk_factors': {
                'tail_risk': float(np.percentile(returns, 5)) if len(returns) > 0 else 0.0,
                'max_drawdown_proxy': float(min(0.0, np.min(np.cumsum(returns)))),
                'volatility_regime': base_features.get('ta_volatility_20', 0.0)
            }
        }
    
    def _generate_timeseries_features(self, price_data: np.ndarray, base_features: Dict[str, float]) -> Dict[str, Any]:
        """Generate features for GARCH and time series models"""
        if len(price_data) < 20:
            return {}
        
        returns = np.diff(np.log(price_data))
        
        return {
            'volatility_features': {
                'volatility_clustering': base_features.get('vol_clustering', 0.0),
                'heteroscedasticity': float(self._ljung_box_test(returns ** 2)),
                'arch_effects': float(np.corrcoef(returns[:-1] ** 2, returns[1:] ** 2)[0, 1]) if len(returns) > 1 else 0.0
            },
            'return_features': {
                'skewness': base_features.get('return_skew', 0.0),
                'kurtosis': base_features.get('return_kurtosis', 0.0),
                'autocorrelations': [base_features.get('autocorr_1', 0.0), base_features.get('autocorr_5', 0.0)]
            },
            'garch_inputs': {
                'returns': returns.tolist()[-50:],  # Last 50 observations
                'suggested_p': 1,
                'suggested_q': 1
            }
        }
    
    def _generate_ml_features(self, price_data: np.ndarray, base_features: Dict[str, float]) -> Dict[str, Any]:
        """Generate features optimized for ML models"""
        # Comprehensive feature matrix for ML
        feature_matrix = []
        feature_names = []
        
        # Technical features
        for name, value in base_features.items():
            if name.startswith('ta_'):
                feature_matrix.append(value)
                feature_names.append(name)
        
        # Statistical features
        for name, value in base_features.items():
            if name.startswith('return_') or name.startswith('autocorr_'):
                feature_matrix.append(value)
                feature_names.append(name)
        
        # Legacy RL features
        for name, value in base_features.items():
            if name.startswith('legacy_'):
                feature_matrix.append(value)
                feature_names.append(name)
        
        return {
            'feature_matrix': [feature_matrix],  # Single sample
            'feature_names': feature_names,
            'feature_engineering': {
                'scaling_required': True,
                'feature_selection': True,
                'dimensionality_reduction': len(feature_matrix) > 20
            },
            'ensemble_config': {
                'models': ['random_forest', 'xgboost', 'neural_network'],
                'cross_validation': True,
                'hyperparameter_tuning': True
            }
        }
    
    def _generate_physics_features(self, price_data: np.ndarray, base_features: Dict[str, float]) -> Dict[str, Any]:
        """Generate features for physics-based models"""
        if len(price_data) < 30:
            return {}
        
        returns = np.diff(np.log(price_data))
        
        return {
            'entropy_measures': {
                'shannon_entropy': float(self._shannon_entropy(returns)),
                'differential_entropy': float(self._differential_entropy(returns)),
                'market_efficiency': 1.0 - abs(base_features.get('autocorr_1', 0.0))
            },
            'complexity_measures': {
                'hurst_exponent': float(self._estimate_hurst(price_data)),
                'fractal_dimension': float(2.0 - self._estimate_hurst(price_data)),
                'lyapunov_exponent': float(self._estimate_lyapunov(returns))
            },
            'thermodynamic_analogies': {
                'temperature_proxy': base_features.get('ta_volatility_20', 0.0),
                'pressure_proxy': abs(base_features.get('ta_ma_trend_20', 0.0)),
                'phase_transition_indicator': float(base_features.get('vol_clustering', 0.0))
            }
        }
    
    def _generate_microstructure_features(self, price_data: np.ndarray, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate features for microstructure analysis"""
        return {
            'liquidity_measures': {
                'bid_ask_spread': market_data.get('bid_ask_spread', 0.0),
                'volume': market_data.get('volume', 0.0),
                'market_depth': market_data.get('market_depth', 10)
            },
            'order_flow': {
                'buy_sell_ratio': market_data.get('buy_sell_ratio', 1.0),
                'large_order_indicator': market_data.get('large_orders', False),
                'order_book_imbalance': market_data.get('order_imbalance', 0.0)
            },
            'tick_analysis': {
                'tick_direction': market_data.get('tick_direction', 0),
                'trade_size_distribution': market_data.get('trade_sizes', []),
                'price_impact': market_data.get('price_impact', 0.0)
            }
        }
    
    def _ljung_box_test(self, data: np.ndarray, lags: int = 10) -> float:
        """Simple Ljung-Box test statistic for autocorrelation"""
        if len(data) < lags + 1:
            return 0.0
        
        n = len(data)
        autocorrs = [self._autocorrelation(data, lag) for lag in range(1, min(lags + 1, n))]
        
        # Simplified LB statistic
        lb_stat = n * (n + 2) * sum(autocorr ** 2 / (n - lag) for lag, autocorr in enumerate(autocorrs, 1))
        return min(1.0, lb_stat / 20.0)  # Normalize roughly
    
    def _shannon_entropy(self, data: np.ndarray, bins: int = 20) -> float:
        """Calculate Shannon entropy"""
        if len(data) < 2:
            return 0.0
        
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]  # Remove zero bins
        probs = hist / np.sum(hist)
        return -np.sum(probs * np.log2(probs))
    
    def _differential_entropy(self, data: np.ndarray) -> float:
        """Estimate differential entropy"""
        if len(data) < 2:
            return 0.0
        
        # Simple estimation using normal approximation
        return 0.5 * np.log(2 * np.pi * np.e * np.var(data))
    
    def _estimate_hurst(self, data: np.ndarray) -> float:
        """Estimate Hurst exponent using R/S analysis"""
        if len(data) < 10:
            return 0.5
        
        try:
            n = len(data)
            rs_values = []
            
            for size in [10, 20, 50]:
                if size >= n:
                    continue
                
                segments = n // size
                rs_segment = []
                
                for i in range(segments):
                    segment = data[i * size:(i + 1) * size]
                    mean_segment = np.mean(segment)
                    cumulative_deviation = np.cumsum(segment - mean_segment)
                    
                    range_cs = np.max(cumulative_deviation) - np.min(cumulative_deviation)
                    std_segment = np.std(segment)
                    
                    if std_segment > 0:
                        rs_segment.append(range_cs / std_segment)
                
                if rs_segment:
                    rs_values.append((size, np.mean(rs_segment)))
            
            if len(rs_values) < 2:
                return 0.5
            
            # Linear regression in log space
            sizes = np.array([rs[0] for rs in rs_values])
            rs_vals = np.array([rs[1] for rs in rs_values])
            
            log_sizes = np.log(sizes)
            log_rs = np.log(rs_vals + 1e-8)
            
            hurst = np.polyfit(log_sizes, log_rs, 1)[0]
            return np.clip(hurst, 0.0, 1.0)
            
        except:
            return 0.5
    
    def _estimate_lyapunov(self, returns: np.ndarray) -> float:
        """Estimate largest Lyapunov exponent"""
        if len(returns) < 10:
            return 0.0
        
        try:
            # Simple estimation based on divergence of nearby trajectories
            n = len(returns)
            divergences = []
            
            for i in range(n - 5):
                for j in range(i + 1, min(i + 10, n - 5)):
                    initial_distance = abs(returns[i] - returns[j])
                    if initial_distance > 0:
                        final_distance = abs(returns[i + 5] - returns[j + 5])
                        if final_distance > 0:
                            divergences.append(np.log(final_distance / initial_distance) / 5)
            
            return np.mean(divergences) if divergences else 0.0
            
        except:
            return 0.0
    
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

# Export enhanced classes for easy imports
__all__ = [
    'LegacyRLIntegration', 
    'LegacyTechnicalAnalysis', 
    'FeatureEngineer', 
    'LegacyBridge',
    'AdvancedLegacyBridge'  # New enhanced bridge
]

# Factory function for creating the appropriate bridge
def create_legacy_bridge(use_advanced_frameworks: bool = True) -> 'LegacyBridge':
    """
    Factory function to create the appropriate legacy bridge
    
    Args:
        use_advanced_frameworks: If True, returns AdvancedLegacyBridge with 
                               full advanced frameworks integration
    
    Returns:
        Legacy bridge instance
    """
    if use_advanced_frameworks:
        try:
            return AdvancedLegacyBridge()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not create AdvancedLegacyBridge, falling back to basic: {e}")
            return LegacyBridge()
    else:
        return LegacyBridge()
