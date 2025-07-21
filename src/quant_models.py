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
