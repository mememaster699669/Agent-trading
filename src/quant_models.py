"""
Quantitative Logic Trading System
Core Mathematical Models for Probabilistic Trading

This module contains the fundamental quantitative models that replace
traditional technical analysis with proper statistical methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy.spatial.distance import pdist
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ProbabilisticForecast:
    """
    Represents a probabilistic price forecast with uncertainty bounds
    """
    mean: float
    std: float
    confidence_intervals: Dict[float, Tuple[float, float]]  # {confidence: (lower, upper)}
    probability_up: float  # P(price > current_price)
    probability_down: float  # P(price < current_price)
    expected_return: float
    downside_risk: float  # Expected loss if negative
    upside_potential: float  # Expected gain if positive
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for given confidence level"""
        if confidence in self.confidence_intervals:
            return self.confidence_intervals[confidence]
        
        # Calculate on demand
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        lower = self.mean - z_score * self.std
        upper = self.mean + z_score * self.std
        return (lower, upper)


class QuantitativeModels:
    """
    Collection of quantitative trading models based on statistical principles
    """
    
    @staticmethod
    def bayesian_price_model(prices: np.ndarray, 
                           lookback: int = 252,
                           confidence_levels: List[float] = [0.68, 0.95, 0.99]) -> ProbabilisticForecast:
        """
        Bayesian approach to price modeling with uncertainty quantification
        
        Args:
            prices: Historical price series
            lookback: Number of periods for estimation
            confidence_levels: Confidence levels for intervals
            
        Returns:
            ProbabilisticForecast with complete uncertainty characterization
        """
        if len(prices) < lookback:
            lookback = len(prices)
            
        recent_prices = prices[-lookback:]
        returns = np.diff(np.log(recent_prices))
        
        # Bayesian estimation of return parameters
        # Prior: Normal-Gamma conjugate prior
        # Posterior: Updated with observed data
        
        n = len(returns)
        sample_mean = np.mean(returns)
        sample_var = np.var(returns, ddof=1)
        
        # Bayesian posterior parameters (assuming weak prior)
        # This gives us uncertainty about the parameters themselves
        posterior_mean = sample_mean
        posterior_var_of_mean = sample_var / n  # Uncertainty in mean estimate
        posterior_var = sample_var  # Expected variance
        
        # Current price
        current_price = prices[-1]
        
        # Forecast distribution (predictive posterior)
        # Accounts for both parameter uncertainty and inherent randomness
        forecast_variance = posterior_var + posterior_var_of_mean
        forecast_std = np.sqrt(forecast_variance)
        
        # Expected price (geometric Brownian motion approximation)
        expected_log_price = np.log(current_price) + posterior_mean
        expected_price = np.exp(expected_log_price + 0.5 * forecast_variance)
        
        # Confidence intervals
        confidence_intervals = {}
        for conf in confidence_levels:
            alpha = 1 - conf
            z = stats.norm.ppf(1 - alpha/2)
            
            log_lower = expected_log_price - z * forecast_std
            log_upper = expected_log_price + z * forecast_std
            
            confidence_intervals[conf] = (np.exp(log_lower), np.exp(log_upper))
        
        # Probabilities
        prob_up = 1 - stats.norm.cdf(0, posterior_mean, forecast_std)
        prob_down = stats.norm.cdf(0, posterior_mean, forecast_std)
        
        # Expected returns conditional on direction
        truncated_mean_pos = stats.truncnorm.mean(0, np.inf, posterior_mean, forecast_std)
        truncated_mean_neg = stats.truncnorm.mean(-np.inf, 0, posterior_mean, forecast_std)
        
        upside_potential = prob_up * truncated_mean_pos if prob_up > 0 else 0
        downside_risk = prob_down * abs(truncated_mean_neg) if prob_down > 0 else 0
        
        return ProbabilisticForecast(
            mean=expected_price,
            std=forecast_std * current_price,  # Convert to price units
            confidence_intervals=confidence_intervals,
            probability_up=prob_up,
            probability_down=prob_down,
            expected_return=posterior_mean,
            downside_risk=downside_risk,
            upside_potential=upside_potential
        )
    
    @staticmethod
    def regime_detection_hmm(returns: np.ndarray, 
                           n_regimes: int = 3) -> Tuple[np.ndarray, Dict]:
        """
        Hidden Markov Model for market regime detection
        
        Args:
            returns: Return series
            n_regimes: Number of market regimes
            
        Returns:
            Tuple of (regime_probabilities, regime_characteristics)
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            # Fallback to simple regime detection
            return QuantitativeModels._simple_regime_detection(returns, n_regimes)
        
        # Prepare data
        X = returns.reshape(-1, 1)
        
        # Fit Gaussian HMM
        model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full")
        model.fit(X)
        
        # Get regime probabilities
        regime_probs = model.predict_proba(X)
        current_regime_probs = regime_probs[-1]
        
        # Characterize regimes
        regime_stats = {}
        for i in range(n_regimes):
            regime_stats[i] = {
                'mean_return': model.means_[i][0],
                'volatility': np.sqrt(model.covars_[i][0][0]),
                'current_probability': current_regime_probs[i]
            }
        
        return current_regime_probs, regime_stats
    
    @staticmethod
    def _simple_regime_detection(returns: np.ndarray, 
                                n_regimes: int = 3) -> Tuple[np.ndarray, Dict]:
        """Fallback regime detection using rolling statistics"""
        window = min(60, len(returns) // 4)
        rolling_vol = pd.Series(returns).rolling(window).std()
        
        # Simple volatility-based regimes
        vol_quantiles = np.linspace(0, 1, n_regimes + 1)
        vol_thresholds = np.quantile(rolling_vol.dropna(), vol_quantiles)
        
        current_vol = rolling_vol.iloc[-1]
        regime_probs = np.zeros(n_regimes)
        
        # Assign probability based on current volatility
        for i in range(n_regimes):
            if i == 0:  # Low volatility regime
                prob = max(0, 1 - (current_vol - vol_thresholds[i]) / 
                          (vol_thresholds[i+1] - vol_thresholds[i]))
            elif i == n_regimes - 1:  # High volatility regime
                prob = max(0, (current_vol - vol_thresholds[i]) / 
                          (vol_thresholds[i+1] - vol_thresholds[i]))
            else:  # Middle regimes
                prob = max(0, 1 - abs(current_vol - 
                          (vol_thresholds[i] + vol_thresholds[i+1])/2) / 
                          (vol_thresholds[i+1] - vol_thresholds[i]))
        
        # Normalize probabilities
        regime_probs = regime_probs / (regime_probs.sum() + 1e-8)
        
        regime_stats = {
            i: {
                'mean_return': np.mean(returns),
                'volatility': vol_thresholds[i+1],
                'current_probability': regime_probs[i]
            }
            for i in range(n_regimes)
        }
        
        return regime_probs, regime_stats
    
    @staticmethod
    def kelly_position_sizing(expected_return: float,
                            variance: float,
                            max_position: float = 0.25) -> float:
        """
        Kelly Criterion for optimal position sizing
        
        Args:
            expected_return: Expected return of the bet
            variance: Variance of returns
            max_position: Maximum position size (risk management)
            
        Returns:
            Optimal position size as fraction of capital
        """
        if variance <= 0:
            return 0.0
        
        # Kelly formula: f* = μ/σ² (for log-normal returns)
        kelly_fraction = expected_return / variance
        
        # Apply position limits for risk management
        # Never risk more than max_position of capital
        kelly_fraction = np.clip(kelly_fraction, -max_position, max_position)
        
        # Additional safety: scale down if too aggressive
        if abs(kelly_fraction) > 0.1:  # More than 10% seems aggressive
            kelly_fraction *= 0.5
        
        return kelly_fraction
    
    @staticmethod
    def mean_reversion_signal(prices: np.ndarray,
                            lookback: int = 20,
                            z_threshold: float = 2.0) -> Dict:
        """
        Statistical mean reversion signal based on z-score
        
        Args:
            prices: Price series
            lookback: Lookback period for mean calculation
            z_threshold: Z-score threshold for signal
            
        Returns:
            Dictionary with signal strength and statistics
        """
        if len(prices) < lookback + 1:
            return {'signal': 0, 'z_score': 0, 'confidence': 0}
        
        recent_prices = prices[-lookback-1:]
        returns = np.diff(np.log(recent_prices))
        
        # Calculate z-score of most recent return
        mean_return = np.mean(returns[:-1])
        std_return = np.std(returns[:-1], ddof=1)
        
        if std_return == 0:
            return {'signal': 0, 'z_score': 0, 'confidence': 0}
        
        current_return = returns[-1]
        z_score = (current_return - mean_return) / std_return
        
        # Signal strength (mean reversion)
        if z_score > z_threshold:  # Price moved up too much, expect reversion down
            signal = -min(1.0, abs(z_score) / z_threshold - 1)
        elif z_score < -z_threshold:  # Price moved down too much, expect reversion up
            signal = min(1.0, abs(z_score) / z_threshold - 1)
        else:
            signal = 0
        
        # Confidence based on statistical significance
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        confidence = 1 - p_value
        
        return {
            'signal': signal,
            'z_score': z_score,
            'confidence': confidence,
            'p_value': p_value,
            'threshold_used': z_threshold
        }
    
    @staticmethod
    def portfolio_optimization_markowitz(expected_returns: np.ndarray,
                                       covariance_matrix: np.ndarray,
                                       risk_aversion: float = 1.0,
                                       max_weight: float = 0.4) -> np.ndarray:
        """
        Markowitz mean-variance optimization with constraints
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            risk_aversion: Risk aversion parameter (higher = more conservative)
            max_weight: Maximum weight per asset
            
        Returns:
            Optimal portfolio weights
        """
        n_assets = len(expected_returns)
        
        # Objective function: maximize utility = return - (risk_aversion/2) * variance
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds (including short selling constraints if desired)
        bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            # Fallback to equal weights if optimization fails
            return np.ones(n_assets) / n_assets


class RiskMetrics:
    """
    Comprehensive risk measurement and management
    """
    
    @staticmethod
    def value_at_risk(returns: np.ndarray, 
                     confidence: float = 0.05,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Historical returns
            confidence: Confidence level (0.05 = 95% VaR)
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            VaR estimate
        """
        if method == 'historical':
            return np.percentile(returns, confidence * 100)
        
        elif method == 'parametric':
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            z_score = stats.norm.ppf(confidence)
            return mean + z_score * std
        
        elif method == 'monte_carlo':
            # Simple Monte Carlo simulation
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            simulated = np.random.normal(mean, std, 10000)
            return np.percentile(simulated, confidence * 100)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    @staticmethod
    def conditional_value_at_risk(returns: np.ndarray, 
                                confidence: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Historical returns
            confidence: Confidence level
            
        Returns:
            CVaR estimate
        """
        var = RiskMetrics.value_at_risk(returns, confidence, 'historical')
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def maximum_drawdown(prices: np.ndarray) -> Dict:
        """
        Calculate maximum drawdown and related statistics
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with drawdown statistics
        """
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Recovery time
        recovery_idx = None
        for i in range(max_dd_idx, len(prices)):
            if drawdown[i] >= -0.001:  # Within 0.1% of recovery
                recovery_idx = i
                break
        
        recovery_time = recovery_idx - max_dd_idx if recovery_idx else None
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_idx,
            'recovery_time': recovery_time,
            'current_drawdown': drawdown[-1],
            'drawdown_series': drawdown
        }


class AdvancedPhysicsModels:
    """
    @khemkapital-inspired physics-based market analysis
    Implementation of Information Theory, Fractal Memory, and Instability Detection
    """
    
    @staticmethod
    def information_entropy_risk(price_data: np.ndarray, 
                               volume_data: Optional[np.ndarray] = None,
                               bins: int = 20) -> Dict[str, float]:
        """
        Calculate market information entropy to measure system readability
        
        High entropy = Market is unreadable, chaotic, high risk
        Low entropy = Market is readable, structured, lower immediate risk
        
        Args:
            price_data: Price time series
            volume_data: Optional volume data for enhanced analysis
            bins: Number of bins for entropy calculation
            
        Returns:
            Dictionary with entropy metrics and risk assessment
        """
        if len(price_data) < 20:
            return {'entropy': 0.5, 'risk_level': 'medium', 'readability': 0.5}
        
        # Calculate returns
        returns = np.diff(np.log(price_data))
        
        # 1. Return Distribution Entropy
        hist, bin_edges = np.histogram(returns, bins=bins, density=True)
        hist = hist + 1e-8  # Avoid log(0)
        hist = hist / np.sum(hist)  # Normalize
        return_entropy = -np.sum(hist * np.log2(hist))
        
        # 2. Price Change Pattern Entropy
        price_changes = np.diff(price_data)
        price_directions = np.sign(price_changes)
        
        # Calculate entropy of direction sequences (local patterns)
        direction_patterns = []
        pattern_length = min(3, len(price_directions) // 10)
        
        for i in range(len(price_directions) - pattern_length + 1):
            pattern = tuple(price_directions[i:i + pattern_length])
            direction_patterns.append(pattern)
        
        # Count pattern frequencies
        pattern_counts = {}
        for pattern in direction_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Calculate pattern entropy
        total_patterns = len(direction_patterns)
        if total_patterns > 0:
            pattern_probs = np.array(list(pattern_counts.values())) / total_patterns
            pattern_entropy = -np.sum(pattern_probs * np.log2(pattern_probs + 1e-8))
        else:
            pattern_entropy = 0.0
        
        # 3. Volume-weighted entropy (if volume data available)
        volume_entropy = 0.0
        if volume_data is not None and len(volume_data) == len(price_data):
            volume_changes = np.diff(volume_data)
            volume_hist, _ = np.histogram(volume_changes, bins=bins, density=True)
            volume_hist = volume_hist + 1e-8
            volume_hist = volume_hist / np.sum(volume_hist)
            volume_entropy = -np.sum(volume_hist * np.log2(volume_hist))
        
        # Combined entropy measure
        combined_entropy = (return_entropy + pattern_entropy + volume_entropy) / 3.0
        
        # Normalize to [0, 1] range (log2 of bins gives theoretical max)
        max_entropy = np.log2(bins)
        normalized_entropy = min(combined_entropy / max_entropy, 1.0)
        
        # Risk assessment based on entropy
        if normalized_entropy > 0.8:
            risk_level = 'extreme'
            readability = 'unreadable'
        elif normalized_entropy > 0.6:
            risk_level = 'high'
            readability = 'poor'
        elif normalized_entropy > 0.4:
            risk_level = 'medium'
            readability = 'moderate'
        else:
            risk_level = 'low'
            readability = 'good'
        
        return {
            'entropy': normalized_entropy,
            'return_entropy': return_entropy / max_entropy,
            'pattern_entropy': pattern_entropy / max_entropy,
            'volume_entropy': volume_entropy / max_entropy,
            'risk_level': risk_level,
            'readability': readability,
            'information_flow_quality': 1.0 - normalized_entropy  # Higher is better
        }
    
    @staticmethod
    def hurst_exponent_memory(price_data: np.ndarray, 
                            max_lag: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate Hurst exponent to detect fractal memory and traumatic events
        
        H > 0.5: Persistent memory (trending, traumatic events embedded)
        H = 0.5: Random walk (no memory)
        H < 0.5: Anti-persistent (mean reverting)
        
        Args:
            price_data: Price time series
            max_lag: Maximum lag for R/S analysis
            
        Returns:
            Dictionary with Hurst exponent and memory characteristics
        """
        if len(price_data) < 50:
            return {'hurst_exponent': 0.5, 'memory_type': 'insufficient_data', 'trauma_detected': False}
        
        # Calculate log returns
        log_prices = np.log(price_data)
        returns = np.diff(log_prices)
        
        # R/S Analysis (Rescaled Range)
        n = len(returns)
        max_lag = max_lag or min(n // 4, 100)
        
        lags = np.logspace(1, np.log10(max_lag), num=10, dtype=int)
        lags = np.unique(lags)
        lags = lags[lags < n]
        
        rs_values = []
        
        for lag in lags:
            # Split the series into non-overlapping windows
            num_windows = n // lag
            if num_windows < 2:
                continue
                
            rs_window = []
            
            for i in range(num_windows):
                window_returns = returns[i*lag:(i+1)*lag]
                
                # Calculate mean-adjusted cumulative sum
                mean_return = np.mean(window_returns)
                deviations = window_returns - mean_return
                cumulative_deviations = np.cumsum(deviations)
                
                # Range
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                
                # Standard deviation
                S = np.std(window_returns, ddof=1)
                
                # R/S ratio
                if S > 0:
                    rs_window.append(R / S)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        if len(rs_values) < 3:
            return {'hurst_exponent': 0.5, 'memory_type': 'insufficient_data', 'trauma_detected': False}
        
        # Calculate Hurst exponent using linear regression
        # log(R/S) = H * log(n) + constant
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_rs)
        hurst_exponent = slope
        
        # Classify memory type
        if hurst_exponent > 0.6:
            memory_type = 'strong_persistence'
            trauma_detected = True
        elif hurst_exponent > 0.55:
            memory_type = 'moderate_persistence'
            trauma_detected = True
        elif hurst_exponent < 0.4:
            memory_type = 'anti_persistent'
            trauma_detected = False
        elif hurst_exponent < 0.45:
            memory_type = 'weak_anti_persistent'
            trauma_detected = False
        else:
            memory_type = 'random_walk'
            trauma_detected = False
        
        # Additional trauma detection using volatility clustering
        volatility = pd.Series(returns).rolling(window=10).std()
        volatility_changes = np.diff(volatility.dropna())
        extreme_vol_events = np.sum(np.abs(volatility_changes) > 2 * np.std(volatility_changes))
        
        trauma_intensity = min(extreme_vol_events / len(volatility_changes), 1.0) if len(volatility_changes) > 0 else 0
        
        return {
            'hurst_exponent': hurst_exponent,
            'memory_type': memory_type,
            'trauma_detected': trauma_detected,
            'trauma_intensity': trauma_intensity,
            'persistence_strength': abs(hurst_exponent - 0.5),
            'r_squared': r_value**2,
            'statistical_significance': 1 - p_value if p_value < 0.05 else 0
        }
    
    @staticmethod
    def lyapunov_instability_detection(price_data: np.ndarray, 
                                     embedding_dim: int = 3,
                                     time_delay: int = 1) -> Dict[str, float]:
        """
        Calculate Lyapunov exponent to measure system instability propagation
        
        Positive Lyapunov: System amplifies small shocks (chaotic/unstable)
        Zero Lyapunov: System at critical stability threshold
        Negative Lyapunov: System dampens shocks (stable)
        
        Args:
            price_data: Price time series
            embedding_dim: Embedding dimension for phase space reconstruction
            time_delay: Time delay for embedding
            
        Returns:
            Dictionary with instability metrics and risk assessment
        """
        if len(price_data) < 100:
            return {'lyapunov_exponent': 0.0, 'instability_level': 'insufficient_data'}
        
        # Convert to log returns for analysis
        log_prices = np.log(price_data)
        returns = np.diff(log_prices)
        
        # Phase space reconstruction using time delay embedding
        def reconstruct_phase_space(data, m, tau):
            n = len(data)
            reconstructed = np.zeros((n - (m-1)*tau, m))
            
            for i in range(m):
                reconstructed[:, i] = data[i*tau:n - (m-1-i)*tau]
            
            return reconstructed
        
        # Reconstruct phase space
        try:
            phase_space = reconstruct_phase_space(returns, embedding_dim, time_delay)
        except:
            return {'lyapunov_exponent': 0.0, 'instability_level': 'reconstruction_failed'}
        
        if len(phase_space) < 20:
            return {'lyapunov_exponent': 0.0, 'instability_level': 'insufficient_data'}
        
        # Calculate largest Lyapunov exponent using Rosenstein's method
        def lyapunov_rosenstein(trajectories, max_iter=None):
            n_points, n_dim = trajectories.shape
            max_iter = max_iter or min(n_points // 10, 50)
            
            # Find nearest neighbors
            divergence_rates = []
            
            for i in range(n_points - max_iter):
                reference = trajectories[i]
                
                # Find nearest neighbor
                distances = np.linalg.norm(trajectories[i+1:] - reference, axis=1)
                
                if len(distances) == 0:
                    continue
                    
                nearest_idx = np.argmin(distances) + i + 1
                
                if nearest_idx >= n_points - max_iter:
                    continue
                
                # Track divergence over time
                divergences = []
                for j in range(1, min(max_iter, n_points - max(i, nearest_idx))):
                    if i + j >= n_points or nearest_idx + j >= n_points:
                        break
                        
                    current_distance = np.linalg.norm(
                        trajectories[i + j] - trajectories[nearest_idx + j]
                    )
                    
                    if current_distance > 0:
                        divergences.append(np.log(current_distance))
                
                if len(divergences) > 5:  # Need enough points for slope
                    time_steps = np.arange(len(divergences))
                    slope, _, _, _, _ = stats.linregress(time_steps, divergences)
                    divergence_rates.append(slope)
            
            return np.mean(divergence_rates) if divergence_rates else 0.0
        
        # Calculate Lyapunov exponent
        lyapunov_exp = lyapunov_rosenstein(phase_space)
        
        # Alternative: Simple correlation-based instability measure
        # Check how small changes propagate through the system
        correlation_instability = 0.0
        if len(returns) > 20:
            # Measure how much current return depends on previous returns
            autocorr_lags = min(10, len(returns) // 4)
            autocorrelations = [np.corrcoef(returns[:-lag], returns[lag:])[0,1] 
                              for lag in range(1, autocorr_lags + 1)]
            autocorrelations = [corr for corr in autocorrelations if not np.isnan(corr)]
            
            if autocorrelations:
                # High positive autocorrelation at short lags indicates instability
                correlation_instability = np.mean(np.abs(autocorrelations[:3]))
        
        # Combined instability measure
        instability_score = (abs(lyapunov_exp) + correlation_instability) / 2.0
        
        # Classify instability level
        if instability_score > 0.3:
            instability_level = 'extreme'
            systemic_risk = 'high'
        elif instability_score > 0.15:
            instability_level = 'high'
            systemic_risk = 'elevated'
        elif instability_score > 0.05:
            instability_level = 'moderate'
            systemic_risk = 'medium'
        else:
            instability_level = 'low'
            systemic_risk = 'low'
        
        # Shock amplification factor
        shock_amplification = max(1.0, 1.0 + instability_score * 2.0)
        
        return {
            'lyapunov_exponent': lyapunov_exp,
            'instability_score': instability_score,
            'instability_level': instability_level,
            'systemic_risk': systemic_risk,
            'shock_amplification_factor': shock_amplification,
            'correlation_instability': correlation_instability,
            'chaos_detected': lyapunov_exp > 0.01  # Positive Lyapunov indicates chaos
        }
    
    @staticmethod
    def regime_transition_detection(price_data: np.ndarray,
                                  volume_data: Optional[np.ndarray] = None,
                                  lookback: int = 50) -> Dict[str, any]:
        """
        Detect market regime transitions using physics-based indicators
        Combines entropy, memory, and instability for regime classification
        
        Args:
            price_data: Price time series
            volume_data: Optional volume data
            lookback: Lookback period for analysis
            
        Returns:
            Dictionary with regime analysis and transition probabilities
        """
        if len(price_data) < lookback:
            return {'regime': 'insufficient_data', 'transition_probability': 0.0}
        
        recent_data = price_data[-lookback:]
        recent_volume = volume_data[-lookback:] if volume_data is not None else None
        
        # Calculate physics-based metrics
        entropy_metrics = AdvancedPhysicsModels.information_entropy_risk(recent_data, recent_volume)
        memory_metrics = AdvancedPhysicsModels.hurst_exponent_memory(recent_data)
        instability_metrics = AdvancedPhysicsModels.lyapunov_instability_detection(recent_data)
        
        # Regime classification based on combined metrics
        entropy_score = entropy_metrics['entropy']
        memory_score = memory_metrics['hurst_exponent'] - 0.5  # Deviation from random walk
        instability_score = instability_metrics['instability_score']
        
        # Physics-based regime states
        if entropy_score > 0.7 and instability_score > 0.2:
            regime = 'chaos'  # High entropy + high instability = chaotic regime
            stability = 'very_unstable'
        elif memory_score > 0.15 and instability_score > 0.1:
            regime = 'trending_unstable'  # Strong memory + moderate instability
            stability = 'unstable'
        elif memory_score < -0.1:
            regime = 'mean_reverting'  # Anti-persistent behavior
            stability = 'stable'
        elif entropy_score < 0.3 and instability_score < 0.05:
            regime = 'stable_trending'  # Low entropy + low instability
            stability = 'very_stable'
        else:
            regime = 'transitional'  # Mixed signals
            stability = 'moderate'
        
        # Calculate transition probability based on rate of change in metrics
        if len(price_data) > lookback * 2:
            # Compare current metrics to previous period
            prev_data = price_data[-lookback*2:-lookback]
            prev_entropy = AdvancedPhysicsModels.information_entropy_risk(prev_data)['entropy']
            prev_memory = AdvancedPhysicsModels.hurst_exponent_memory(prev_data)['hurst_exponent']
            
            entropy_change = abs(entropy_score - prev_entropy)
            memory_change = abs(memory_score - (prev_memory - 0.5))
            
            transition_probability = min((entropy_change + memory_change) / 2.0, 1.0)
        else:
            transition_probability = 0.5  # Default uncertainty
        
        return {
            'regime': regime,
            'stability': stability,
            'transition_probability': transition_probability,
            'physics_metrics': {
                'entropy_score': entropy_score,
                'memory_deviation': memory_score,
                'instability_score': instability_score
            },
            'risk_assessment': {
                'information_risk': entropy_metrics['risk_level'],
                'memory_risk': 'high' if abs(memory_score) > 0.1 else 'low',
                'systemic_risk': instability_metrics['systemic_risk']
            }
        }


class MarketMicrostructure:
    """
    Market microstructure analysis for execution optimization
    """
    
    @staticmethod
    def estimate_bid_ask_spread(prices: np.ndarray, 
                              volumes: np.ndarray = None) -> float:
        """
        Estimate effective bid-ask spread from price data
        """
        if len(prices) < 2:
            return 0.0
        
        # Roll's model for spread estimation
        price_changes = np.diff(prices)
        
        # Covariance of consecutive price changes (should be negative due to spread)
        if len(price_changes) > 1:
            covariance = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
            spread_estimate = 2 * np.sqrt(-covariance) if covariance < 0 else 0
        else:
            spread_estimate = 0
        
        # Alternative: Use high-frequency price reversals
        reversals = np.abs(price_changes[:-1] + price_changes[1:]) / np.abs(price_changes[:-1])
        reversal_based_spread = np.mean(reversals) * np.mean(np.abs(price_changes))
        
        # Take the more conservative estimate
        return max(spread_estimate, reversal_based_spread)
    
    @staticmethod
    def optimal_execution_twap(total_quantity: float,
                             time_horizon: int,
                             volatility: float,
                             market_impact: float = 0.1) -> np.ndarray:
        """
        Time-Weighted Average Price (TWAP) execution schedule
        
        Args:
            total_quantity: Total quantity to execute
            time_horizon: Number of time periods
            volatility: Price volatility
            market_impact: Market impact parameter
            
        Returns:
            Execution schedule
        """
        # Simple TWAP: equal quantities over time
        # More sophisticated: adjust for volatility and market impact
        
        base_schedule = np.ones(time_horizon) * (total_quantity / time_horizon)
        
        # Adjust for time-varying market conditions (placeholder)
        # In practice, would use real-time volatility and liquidity data
        
        return base_schedule
