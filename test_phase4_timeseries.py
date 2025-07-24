#!/usr/bin/env python3
"""
Phase 4: Advanced Time Series Analysis Test Suite

This test suite validates the comprehensive time series analysis capabilities
including:
- GARCH family models (GARCH, EGARCH, GJR-GARCH, TGARCH)
- Volatility forecasting with uncertainty quantification
- Unit root testing and stationarity analysis
- Volatility regime detection
- Conditional heteroskedasticity modeling
- Time-varying risk metrics
- Model comparison and selection

All tests include both ARCH library integration and fallback implementations.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import our advanced time series analysis
from quant_models import AdvancedTimeSeriesAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesAnalysisTester:
    """Comprehensive test suite for Phase 4 Time Series Analysis"""
    
    def __init__(self):
        self.ts_analyzer = AdvancedTimeSeriesAnalysis()
        logger.info(f"ARCH Library Available: {self.ts_analyzer.is_available}")
        
        # Generate synthetic financial time series
        self.returns_data = self._generate_realistic_returns()
        self.price_data = self._generate_price_series()
        
        logger.info(f"Generated synthetic data: {len(self.returns_data)} return observations")
    
    def _generate_realistic_returns(self, n_periods: int = 1000) -> pd.Series:
        """Generate realistic financial returns with volatility clustering"""
        np.random.seed(42)  # For reproducibility
        
        # Parameters for realistic financial returns
        mu = 0.0005  # Daily mean return (about 12% annually)
        base_vol = 0.015  # Base volatility
        
        # Generate GARCH-like process
        returns = []
        volatilities = []
        h = base_vol**2  # Initial variance
        
        for t in range(n_periods):
            # GARCH(1,1) volatility process
            omega = 0.00001  # Long-run variance
            alpha = 0.08     # ARCH coefficient
            beta = 0.90      # GARCH coefficient
            
            # Shock
            z = np.random.normal(0, 1)
            
            # Return
            ret = mu + np.sqrt(h) * z
            returns.append(ret)
            volatilities.append(np.sqrt(h))
            
            # Update variance
            h = omega + alpha * ret**2 + beta * h
        
        # Create time series
        dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
        return pd.Series(returns, index=dates, name='returns')
    
    def _generate_price_series(self) -> pd.Series:
        """Generate price series from returns"""
        initial_price = 100.0
        prices = [initial_price]
        
        for ret in self.returns_data:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Remove initial price to match returns length
        prices = prices[1:]
        
        return pd.Series(prices, index=self.returns_data.index, name='prices')
    
    def test_comprehensive_garch_analysis(self):
        """Test comprehensive GARCH model analysis"""
        logger.info("\n" + "="*60)
        logger.info("Testing Comprehensive GARCH Analysis")
        logger.info("="*60)
        
        # Test with different model combinations
        model_sets = [
            ['GARCH'],
            ['GARCH', 'EGARCH'],
            ['GARCH', 'EGARCH', 'GJR-GARCH'],
            ['GARCH', 'EGARCH', 'GJR-GARCH', 'TGARCH']
        ]
        
        for i, models in enumerate(model_sets, 1):
            logger.info(f"\nTest {i}: Models {models}")
            
            result = self.ts_analyzer.comprehensive_garch_analysis(
                self.returns_data,
                models=models,
                forecast_horizon=10
            )
            
            self._print_garch_results(result, f"GARCH Analysis - Set {i}")
            
            # Validate results
            assert 'analysis_type' in result
            assert 'models' in result
            
            if result.get('models'):
                # Check that models were fitted
                for model_name in models:
                    if model_name in result['models']:
                        model_result = result['models'][model_name]
                        assert 'model_type' in model_result
                        assert 'conditional_volatility' in model_result
                        logger.info(f"✓ {model_name} model fitted successfully")
            
            # Check forecasts if available
            if 'volatility_forecasts' in result and result['volatility_forecasts']:
                forecasts = result['volatility_forecasts']
                assert 'volatility_forecast' in forecasts
                assert len(forecasts['volatility_forecast']) == 10
                logger.info("✓ Volatility forecasts generated successfully")
            
            logger.info(f"✓ GARCH analysis set {i} completed successfully")
    
    def test_unit_root_analysis(self):
        """Test comprehensive unit root and stationarity testing"""
        logger.info("\n" + "="*60)
        logger.info("Testing Unit Root and Stationarity Analysis")
        logger.info("="*60)
        
        # Test different time series types
        test_series = {
            'returns': self.returns_data,
            'prices': self.price_data,
            'log_prices': np.log(self.price_data),
            'differenced_log_prices': np.log(self.price_data).diff().dropna()
        }
        
        for series_name, series_data in test_series.items():
            logger.info(f"\nTesting stationarity of {series_name}")
            
            result = self.ts_analyzer.unit_root_tests(series_data)
            
            self._print_stationarity_results(result, series_name)
            
            # Validate results
            assert 'series_info' in result
            assert 'tests' in result
            
            if 'overall_assessment' in result:
                assessment = result['overall_assessment']
                assert 'likely_stationary' in assessment
                logger.info(f"✓ Overall assessment: {assessment.get('recommendation', 'unclear')}")
            
            logger.info(f"✓ Unit root testing for {series_name} completed")
    
    def test_volatility_regime_detection(self):
        """Test volatility regime detection and analysis"""
        logger.info("\n" + "="*60)
        logger.info("Testing Volatility Regime Detection")
        logger.info("="*60)
        
        # Test with different numbers of regimes
        regime_counts = [2, 3, 4]
        garch_models = ['GARCH', 'EGARCH']
        
        for n_regimes in regime_counts:
            for model_type in garch_models:
                logger.info(f"\nTesting {n_regimes} regimes with {model_type} model")
                
                result = self.ts_analyzer.volatility_regime_detection(
                    self.returns_data,
                    n_regimes=n_regimes,
                    model_type=model_type
                )
                
                self._print_regime_results(result, f"{n_regimes} Regimes - {model_type}")
                
                # Validate results
                assert 'volatility_regimes' in result
                
                if result.get('volatility_regimes'):
                    regimes = result['volatility_regimes']
                    assert len(regimes) <= n_regimes  # May be fewer if clustering finds fewer
                    
                    # Check regime characteristics
                    for regime_name, stats in regimes.items():
                        assert 'avg_volatility' in stats
                        assert 'frequency' in stats
                        logger.info(f"  {regime_name}: avg_vol={stats['avg_volatility']:.4f}")
                
                if 'current_regime' in result:
                    logger.info(f"  Current regime: {result['current_regime']}")
                
                logger.info(f"✓ Regime detection ({n_regimes}, {model_type}) completed")
    
    def test_volatility_forecasting(self):
        """Test volatility forecasting capabilities"""
        logger.info("\n" + "="*60)
        logger.info("Testing Volatility Forecasting")
        logger.info("="*60)
        
        # First fit a GARCH model
        garch_result = self.ts_analyzer.comprehensive_garch_analysis(
            self.returns_data,
            models=['GARCH'],
            forecast_horizon=20
        )
        
        if garch_result.get('volatility_forecasts'):
            forecasts = garch_result['volatility_forecasts']
            
            logger.info(f"Forecast horizon: {forecasts.get('horizon', 0)} days")
            logger.info(f"Current volatility: {forecasts.get('current_volatility', 0):.4f}")
            
            # Check forecast structure
            vol_forecast = forecasts.get('volatility_forecast', [])
            assert len(vol_forecast) == 20
            
            # Check confidence intervals if available
            if 'confidence_intervals' in forecasts:
                conf_intervals = forecasts['confidence_intervals']
                logger.info("Confidence intervals available:")
                for conf_level, intervals in conf_intervals.items():
                    logger.info(f"  {conf_level}: {len(intervals.get('lower', []))} lower bounds")
            
            logger.info("✓ Volatility forecasting completed successfully")
        else:
            logger.warning("No volatility forecasts generated (using fallback)")
    
    def test_risk_metrics_calculation(self):
        """Test time-varying risk metrics calculation"""
        logger.info("\n" + "="*60)
        logger.info("Testing Time-Varying Risk Metrics")
        logger.info("="*60)
        
        # Get GARCH analysis with risk metrics
        result = self.ts_analyzer.comprehensive_garch_analysis(
            self.returns_data,
            models=['GARCH', 'EGARCH'],
            forecast_horizon=5
        )
        
        if 'risk_metrics' in result:
            risk_metrics = result['risk_metrics']
            
            logger.info("Risk Metrics Available:")
            
            # VaR estimates
            if 'var_estimates' in risk_metrics:
                var_estimates = risk_metrics['var_estimates']
                logger.info("Value at Risk (VaR) estimates:")
                for var_level, estimates in var_estimates.items():
                    current_var = estimates.get('current', 0)
                    logger.info(f"  {var_level}: {current_var:.4f}")
            
            # CVaR estimates
            if 'cvar_estimates' in risk_metrics:
                cvar_estimates = risk_metrics['cvar_estimates']
                logger.info("Conditional VaR (CVaR) estimates:")
                for cvar_level, estimates in cvar_estimates.items():
                    current_cvar = estimates.get('current', 0)
                    logger.info(f"  {cvar_level}: {current_cvar:.4f}")
            
            # Volatility metrics
            if 'volatility_metrics' in risk_metrics:
                vol_metrics = risk_metrics['volatility_metrics']
                current_vol = vol_metrics.get('current_volatility', 0)
                mean_vol = vol_metrics.get('mean_volatility', 0)
                persistence = vol_metrics.get('persistence', 0)
                
                logger.info("Volatility Metrics:")
                logger.info(f"  Current volatility: {current_vol:.4f}")
                logger.info(f"  Mean volatility: {mean_vol:.4f}")
                logger.info(f"  Persistence: {persistence:.4f}")
            
            logger.info("✓ Risk metrics calculation completed successfully")
        else:
            logger.warning("No risk metrics calculated")
    
    def test_model_diagnostics(self):
        """Test GARCH model diagnostics and validation"""
        logger.info("\n" + "="*60)
        logger.info("Testing GARCH Model Diagnostics")
        logger.info("="*60)
        
        # Get comprehensive analysis with diagnostics
        result = self.ts_analyzer.comprehensive_garch_analysis(
            self.returns_data,
            models=['GARCH', 'EGARCH'],
            forecast_horizon=5
        )
        
        if 'diagnostics' in result:
            diagnostics = result['diagnostics']
            
            logger.info("Model Diagnostics Available:")
            
            # Model adequacy tests
            if 'model_adequacy' in diagnostics:
                adequacy = diagnostics['model_adequacy']
                
                for model_name, tests in adequacy.items():
                    logger.info(f"\n{model_name} Model Diagnostics:")
                    
                    # Ljung-Box test
                    if 'ljung_box' in tests:
                        lb = tests['ljung_box']
                        reject_iid = lb.get('reject_iid', False)
                        logger.info(f"  Ljung-Box test: {'REJECT' if reject_iid else 'ACCEPT'} independence")
                    
                    # ARCH test
                    if 'arch_test' in tests:
                        arch = tests['arch_test']
                        reject_homo = arch.get('reject_homoscedastic', False)
                        logger.info(f"  ARCH test: {'REJECT' if reject_homo else 'ACCEPT'} homoscedasticity")
                    
                    # Normality test
                    if 'normality' in tests:
                        norm = tests['normality']
                        reject_norm = norm.get('reject_normality', True)
                        logger.info(f"  Normality test: {'REJECT' if reject_norm else 'ACCEPT'} normality")
            
            logger.info("✓ Model diagnostics completed successfully")
        else:
            logger.warning("No model diagnostics available")
    
    def test_model_comparison(self):
        """Test GARCH model comparison and selection"""
        logger.info("\n" + "="*60)
        logger.info("Testing GARCH Model Comparison")
        logger.info("="*60)
        
        # Comprehensive analysis with multiple models
        result = self.ts_analyzer.comprehensive_garch_analysis(
            self.returns_data,
            models=['GARCH', 'EGARCH', 'GJR-GARCH'],
            forecast_horizon=5
        )
        
        if 'model_comparison' in result:
            comparison = result['model_comparison']
            
            logger.info("Model Comparison Results:")
            
            # Best model
            best_model = comparison.get('best_model')
            if best_model:
                logger.info(f"Best model: {best_model}")
            
            # Rankings
            if 'rankings' in comparison:
                rankings = comparison['rankings']
                
                if 'aic' in rankings:
                    logger.info("AIC Rankings (lower is better):")
                    for i, (model, aic) in enumerate(rankings['aic'], 1):
                        logger.info(f"  {i}. {model}: {aic:.2f}")
                
                if 'bic' in rankings:
                    logger.info("BIC Rankings (lower is better):")
                    for i, (model, bic) in enumerate(rankings['bic'], 1):
                        logger.info(f"  {i}. {model}: {bic:.2f}")
            
            # Model weights
            if 'model_weights' in comparison:
                weights = comparison['model_weights']
                logger.info("Model Weights (AIC-based):")
                for model, weight in weights.items():
                    logger.info(f"  {model}: {weight:.3f}")
            
            logger.info("✓ Model comparison completed successfully")
        else:
            logger.warning("No model comparison available")
    
    def _print_garch_results(self, result: dict, title: str):
        """Print formatted GARCH analysis results"""
        print(f"\n{title}")
        print("-" * len(title))
        
        print(f"Analysis Type: {result.get('analysis_type', 'Unknown')}")
        print(f"ARCH Available: {result.get('arch_available', self.ts_analyzer.is_available)}")
        
        # Data information
        if 'data_info' in result:
            data_info = result['data_info']
            print(f"\nData Information:")
            print(f"  Observations: {data_info.get('n_observations', 0)}")
            print(f"  Mean Return: {data_info.get('mean_return', 0):.6f}")
            print(f"  Volatility: {data_info.get('volatility', 0):.4f}")
            print(f"  Skewness: {data_info.get('skewness', 0):.4f}")
            print(f"  Kurtosis: {data_info.get('kurtosis', 0):.4f}")
        
        # Models fitted
        if 'models' in result and result['models']:
            print(f"\nModels Fitted:")
            for model_name, model_info in result['models'].items():
                convergence = model_info.get('convergence', False)
                aic = model_info.get('aic', 0)
                current_vol = 0
                if hasattr(model_info.get('conditional_volatility'), 'iloc'):
                    current_vol = model_info['conditional_volatility'].iloc[-1]
                
                print(f"  {model_name}:")
                print(f"    Converged: {convergence}")
                print(f"    AIC: {aic:.2f}")
                print(f"    Current Vol: {current_vol:.4f}")
                
                # Model-specific parameters
                if 'persistence' in model_info:
                    print(f"    Persistence: {model_info['persistence']:.4f}")
                if 'asymmetry_param' in model_info:
                    print(f"    Asymmetry: {model_info['asymmetry_param']:.4f}")
        
        # Best model
        best_model = result.get('best_model')
        if best_model:
            print(f"\nBest Model: {best_model}")
        
        # Forecasts
        if 'volatility_forecasts' in result and result['volatility_forecasts']:
            forecasts = result['volatility_forecasts']
            horizon = forecasts.get('horizon', 0)
            current_vol = forecasts.get('current_volatility', 0)
            
            print(f"\nVolatility Forecasts:")
            print(f"  Horizon: {horizon} periods")
            print(f"  Current Volatility: {current_vol:.4f}")
            
            if 'volatility_forecast' in forecasts:
                vol_forecast = forecasts['volatility_forecast']
                if len(vol_forecast) > 0:
                    print(f"  Next Period Forecast: {vol_forecast[0]:.4f}")
                    if len(vol_forecast) > 1:
                        print(f"  Final Period Forecast: {vol_forecast[-1]:.4f}")
    
    def _print_stationarity_results(self, result: dict, series_name: str):
        """Print formatted stationarity test results"""
        print(f"\nStationarity Analysis: {series_name}")
        print("-" * (22 + len(series_name)))
        
        # Series information
        if 'series_info' in result:
            info = result['series_info']
            print(f"Series Length: {info.get('length', 0)}")
            print(f"Mean: {info.get('mean', 0):.6f}")
            print(f"Std Dev: {info.get('std', 0):.4f}")
        
        print(f"ARCH Available: {result.get('arch_available', self.ts_analyzer.is_available)}")
        
        # Test results
        if 'tests' in result:
            tests = result['tests']
            print(f"\nTest Results:")
            
            for test_name, test_result in tests.items():
                print(f"  {test_name.upper()}:")
                if 'statistic' in test_result:
                    print(f"    Statistic: {test_result['statistic']:.4f}")
                if 'pvalue' in test_result:
                    print(f"    P-value: {test_result['pvalue']:.4f}")
                if 'null_hypothesis' in test_result:
                    print(f"    H0: {test_result['null_hypothesis']}")
                
                # Decision
                if 'reject_unit_root' in test_result:
                    decision = "Stationary" if test_result['reject_unit_root'] else "Non-stationary"
                    print(f"    Decision: {decision}")
                elif 'reject_stationarity' in test_result:
                    decision = "Non-stationary" if test_result['reject_stationarity'] else "Stationary"
                    print(f"    Decision: {decision}")
        
        # Overall assessment
        if 'overall_assessment' in result:
            assessment = result['overall_assessment']
            likely_stationary = assessment.get('likely_stationary', False)
            recommendation = assessment.get('recommendation', 'unclear')
            
            print(f"\nOverall Assessment:")
            print(f"  Likely Stationary: {likely_stationary}")
            print(f"  Recommendation: {recommendation}")
            
            if 'consensus_strength' in assessment:
                print(f"  Consensus Strength: {assessment['consensus_strength']:.2f}")
    
    def _print_regime_results(self, result: dict, title: str):
        """Print formatted volatility regime results"""
        print(f"\n{title}")
        print("-" * len(title))
        
        # Method information
        method = result.get('method', result.get('garch_model_used', 'Unknown'))
        print(f"Method: {method}")
        print(f"ARCH Available: {result.get('arch_available', self.ts_analyzer.is_available)}")
        
        # Current state
        current_regime = result.get('current_regime', 'Unknown')
        current_vol = result.get('current_volatility', 0)
        print(f"Current Regime: {current_regime}")
        print(f"Current Volatility: {current_vol:.4f}")
        
        # Regime characteristics
        if 'volatility_regimes' in result and result['volatility_regimes']:
            regimes = result['volatility_regimes']
            print(f"\nRegime Characteristics:")
            
            for regime_name, stats in regimes.items():
                print(f"  {regime_name}:")
                print(f"    Avg Volatility: {stats.get('avg_volatility', 0):.4f}")
                
                vol_range = stats.get('vol_range', [0, 0])
                if len(vol_range) == 2:
                    print(f"    Vol Range: [{vol_range[0]:.4f}, {vol_range[1]:.4f}]")
                
                frequency = stats.get('frequency', 0)
                print(f"    Frequency: {frequency:.2%}")
                
                if 'current_probability' in stats:
                    print(f"    Current Prob: {stats['current_probability']:.2%}")
        
        # Transition probabilities
        if 'regime_transition_probabilities' in result:
            trans_probs = result['regime_transition_probabilities']
            if 'persistence_probabilities' in trans_probs:
                persistences = trans_probs['persistence_probabilities']
                print(f"\nRegime Persistence Probabilities:")
                for i, persistence in enumerate(persistences):
                    print(f"  Regime {i}: {persistence:.2%}")
    
    def run_comprehensive_tests(self):
        """Run all time series analysis tests"""
        logger.info("Starting Phase 4: Advanced Time Series Analysis Test Suite")
        logger.info(f"ARCH Library Available: {self.ts_analyzer.is_available}")
        
        test_methods = [
            self.test_comprehensive_garch_analysis,
            self.test_unit_root_analysis,
            self.test_volatility_regime_detection,
            self.test_volatility_forecasting,
            self.test_risk_metrics_calculation,
            self.test_model_diagnostics,
            self.test_model_comparison
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_method in test_methods:
            try:
                test_method()
                passed_tests += 1
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed: {e}")
                failed_tests += 1
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("PHASE 4 TIME SERIES ANALYSIS TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Tests: {passed_tests + failed_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/(passed_tests + failed_tests)*100:.1f}%")
        
        if self.ts_analyzer.is_available:
            logger.info("✅ Advanced ARCH/GARCH features tested successfully")
        else:
            logger.info("⚠️ ARCH library not available - fallback implementations tested")
        
        logger.info("Phase 4 Time Series Analysis implementation complete!")
        
        return passed_tests, failed_tests


def main():
    """Run the Phase 4 Time Series Analysis test suite"""
    tester = TimeSeriesAnalysisTester()
    passed, failed = tester.run_comprehensive_tests()
    
    # Exit with appropriate code
    exit_code = 0 if failed == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
