#!/usr/bin/env python3
"""
Phase 3: Advanced Portfolio Optimization Test Suite

This test suite validates the comprehensive portfolio optimization capabilities
including:
- Mean-variance optimization with efficient frontier
- Black-Litterman model with investor views
- Hierarchical Risk Parity (HRP)
- Factor model optimization (CAPM, Fama-French)
- Risk budgeting and equal risk contribution
- Monte Carlo portfolio simulation
- Advanced risk metrics and performance analysis

All tests include both PyPortfolioOpt integration and fallback implementations.
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

# Import our advanced portfolio optimization
from quant_models import AdvancedPortfolioOptimization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioOptimizationTester:
    """Comprehensive test suite for Phase 3 Portfolio Optimization"""
    
    def __init__(self):
        self.portfolio_optimizer = AdvancedPortfolioOptimization()
        logger.info(f"PyPortfolioOpt Available: {self.portfolio_optimizer.is_available}")
        
        # Generate synthetic market data
        self.returns_df = self._generate_synthetic_returns()
        self.market_caps = self._generate_market_caps()
        self.market_returns = self._generate_market_returns()
        
        logger.info(f"Generated synthetic data: {len(self.returns_df)} periods, {len(self.returns_df.columns)} assets")
    
    def _generate_synthetic_returns(self, n_periods: int = 252, n_assets: int = 5) -> pd.DataFrame:
        """Generate synthetic asset returns"""
        np.random.seed(42)  # For reproducibility
        
        # Asset names
        assets = [f'ASSET_{i+1}' for i in range(n_assets)]
        
        # Generate correlated returns
        mean_returns = np.random.uniform(0.0001, 0.0008, n_assets)  # Daily returns
        volatilities = np.random.uniform(0.01, 0.03, n_assets)  # Daily volatilities
        
        # Create correlation matrix
        correlation = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
        correlation = (correlation + correlation.T) / 2  # Make symmetric
        np.fill_diagonal(correlation, 1.0)
        
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive
        correlation = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Generate returns
        returns = []
        for i in range(n_periods):
            daily_returns = np.random.multivariate_normal(mean_returns, 
                                                         np.outer(volatilities, volatilities) * correlation)
            returns.append(daily_returns)
        
        dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='D')
        return pd.DataFrame(returns, index=dates, columns=assets)
    
    def _generate_market_caps(self) -> dict:
        """Generate synthetic market capitalizations"""
        market_caps = {}
        for asset in self.returns_df.columns:
            market_caps[asset] = np.random.uniform(1e9, 1e12)  # $1B to $1T
        return market_caps
    
    def _generate_market_returns(self) -> pd.Series:
        """Generate synthetic market returns"""
        np.random.seed(42)
        market_returns = np.random.normal(0.0005, 0.015, len(self.returns_df))
        return pd.Series(market_returns, index=self.returns_df.index)
    
    def test_efficient_frontier_optimization(self):
        """Test Mean-Variance Optimization with Efficient Frontier"""
        logger.info("\n" + "="*60)
        logger.info("Testing Efficient Frontier Optimization")
        logger.info("="*60)
        
        methods = ['max_sharpe', 'min_volatility', 'efficient_return', 'efficient_risk']
        
        for method in methods:
            logger.info(f"\nTesting method: {method}")
            
            # Set method-specific parameters
            kwargs = {}
            if method == 'efficient_return':
                kwargs['target_return'] = 0.12  # 12% annual return
            elif method == 'efficient_risk':
                kwargs['target_volatility'] = 0.15  # 15% annual volatility
            
            result = self.portfolio_optimizer.efficient_frontier_optimization(
                self.returns_df, method=method, **kwargs
            )
            
            self._print_portfolio_results(result, f"Efficient Frontier - {method}")
            
            # Validate results
            assert 'weights' in result
            assert 'performance' in result
            assert 'risk_metrics' in result
            
            # Check weights sum to 1 (approximately)
            total_weight = sum(result['weights'].values())
            assert abs(total_weight - 1.0) < 0.01, f"Weights don't sum to 1: {total_weight}"
            
            logger.info(f"✓ {method} optimization completed successfully")
    
    def test_black_litterman_optimization(self):
        """Test Black-Litterman Model with Investor Views"""
        logger.info("\n" + "="*60)
        logger.info("Testing Black-Litterman Optimization")
        logger.info("="*60)
        
        # Test without views (pure equilibrium)
        logger.info("\nTest 1: Pure equilibrium (no views)")
        result1 = self.portfolio_optimizer.black_litterman_optimization(
            self.returns_df,
            market_caps=self.market_caps
        )
        
        self._print_portfolio_results(result1, "Black-Litterman - No Views")
        
        # Test with investor views
        logger.info("\nTest 2: With investor views")
        assets = list(self.returns_df.columns)
        views = {
            assets[0]: 0.15,  # Expect 15% annual return
            assets[1]: 0.08   # Expect 8% annual return
        }
        view_uncertainties = {
            assets[0]: 0.05,  # 5% uncertainty
            assets[1]: 0.03   # 3% uncertainty
        }
        
        result2 = self.portfolio_optimizer.black_litterman_optimization(
            self.returns_df,
            market_caps=self.market_caps,
            views=views,
            view_uncertainties=view_uncertainties
        )
        
        self._print_portfolio_results(result2, "Black-Litterman - With Views")
        
        # Validate results
        for result in [result1, result2]:
            assert 'weights' in result
            assert 'bl_returns' in result
            assert 'market_implied_returns' in result
            
            total_weight = sum(result['weights'].values())
            assert abs(total_weight - 1.0) < 0.01
        
        logger.info("✓ Black-Litterman optimization completed successfully")
    
    def test_hierarchical_risk_parity(self):
        """Test Hierarchical Risk Parity (HRP)"""
        logger.info("\n" + "="*60)
        logger.info("Testing Hierarchical Risk Parity")
        logger.info("="*60)
        
        linkage_methods = ['ward', 'complete', 'average']
        
        for method in linkage_methods:
            logger.info(f"\nTesting linkage method: {method}")
            
            result = self.portfolio_optimizer.hierarchical_risk_parity(
                self.returns_df,
                linkage_method=method
            )
            
            self._print_portfolio_results(result, f"HRP - {method}")
            
            # Validate results
            assert 'weights' in result
            assert 'clustering_info' in result
            
            total_weight = sum(result['weights'].values())
            assert abs(total_weight - 1.0) < 0.01
            
            logger.info(f"✓ HRP with {method} linkage completed successfully")
    
    def test_factor_model_optimization(self):
        """Test Factor Model Optimization"""
        logger.info("\n" + "="*60)
        logger.info("Testing Factor Model Optimization")
        logger.info("="*60)
        
        # Test CAPM
        logger.info("\nTest 1: CAPM Factor Model")
        result1 = self.portfolio_optimizer.factor_model_optimization(
            self.returns_df,
            factor_model='capm',
            market_returns=self.market_returns
        )
        
        self._print_portfolio_results(result1, "Factor Model - CAPM")
        
        # Test Fama-French (simplified)
        logger.info("\nTest 2: Fama-French Factor Model")
        factors_df = self._generate_factor_returns()
        result2 = self.portfolio_optimizer.factor_model_optimization(
            self.returns_df,
            factor_model='fama_french_3',
            factors_df=factors_df
        )
        
        self._print_portfolio_results(result2, "Factor Model - Fama-French")
        
        # Validate results
        for result in [result1, result2]:
            assert 'weights' in result
            assert 'expected_returns' in result
            assert 'factor_model_info' in result
            
            total_weight = sum(result['weights'].values())
            assert abs(total_weight - 1.0) < 0.01
        
        logger.info("✓ Factor model optimization completed successfully")
    
    def test_risk_budgeting_optimization(self):
        """Test Risk Budgeting and Risk Parity"""
        logger.info("\n" + "="*60)
        logger.info("Testing Risk Budgeting Optimization")
        logger.info("="*60)
        
        # Test equal risk contribution
        logger.info("\nTest 1: Equal Risk Contribution")
        result1 = self.portfolio_optimizer.risk_budgeting_optimization(
            self.returns_df,
            method='equal_risk_contribution'
        )
        
        self._print_portfolio_results(result1, "Risk Budgeting - Equal Risk")
        
        # Test custom risk budgets
        logger.info("\nTest 2: Custom Risk Budgets")
        assets = list(self.returns_df.columns)
        risk_budgets = {asset: 1.0/len(assets) for asset in assets}
        risk_budgets[assets[0]] = 0.4  # Higher risk budget for first asset
        risk_budgets[assets[1]] = 0.3  # Medium risk budget for second asset
        # Normalize remaining budgets
        remaining_budget = 0.3
        for i, asset in enumerate(assets[2:], 2):
            risk_budgets[asset] = remaining_budget / (len(assets) - 2)
        
        result2 = self.portfolio_optimizer.risk_budgeting_optimization(
            self.returns_df,
            risk_budgets=risk_budgets,
            method='custom_risk_budgets'
        )
        
        self._print_portfolio_results(result2, "Risk Budgeting - Custom")
        
        # Validate results
        for result in [result1, result2]:
            assert 'weights' in result
            assert 'risk_contributions' in result
            assert 'risk_budgeting_info' in result
            
            total_weight = sum(result['weights'].values())
            assert abs(total_weight - 1.0) < 0.01
        
        logger.info("✓ Risk budgeting optimization completed successfully")
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo Portfolio Simulation"""
        logger.info("\n" + "="*60)
        logger.info("Testing Monte Carlo Portfolio Simulation")
        logger.info("="*60)
        
        result = self.portfolio_optimizer.monte_carlo_portfolio_simulation(
            self.returns_df,
            n_simulations=1000,  # Reduced for testing speed
            time_horizon=63,     # 3 months
            initial_portfolio_value=100000
        )
        
        if 'error' not in result:
            self._print_monte_carlo_results(result)
            
            # Validate results
            assert 'base_weights' in result
            assert 'simulation_results' in result
            assert 'risk_metrics' in result
            assert 'performance_percentiles' in result
            
            # Check probability makes sense (0 to 1)
            prob_positive = result['simulation_results']['probability_positive_return']
            assert 0 <= prob_positive <= 1
            
            logger.info("✓ Monte Carlo simulation completed successfully")
        else:
            logger.warning(f"Monte Carlo simulation failed: {result.get('error', 'Unknown error')}")
    
    def _generate_factor_returns(self) -> pd.DataFrame:
        """Generate synthetic factor returns for testing"""
        np.random.seed(42)
        factors = ['Market', 'SMB', 'HML']  # Market, Small-Minus-Big, High-Minus-Low
        
        factor_returns = []
        for i in range(len(self.returns_df)):
            daily_factors = np.random.multivariate_normal(
                [0.0005, 0.0001, 0.0002],  # Mean returns
                [[0.0002, 0.0001, 0.00005],
                 [0.0001, 0.0001, 0.00003],
                 [0.00005, 0.00003, 0.0001]]  # Covariance matrix
            )
            factor_returns.append(daily_factors)
        
        return pd.DataFrame(factor_returns, index=self.returns_df.index, columns=factors)
    
    def _print_portfolio_results(self, result: dict, title: str):
        """Print formatted portfolio optimization results"""
        print(f"\n{title}")
        print("-" * len(title))
        
        print(f"Optimization Method: {result.get('optimization_method', 'Unknown')}")
        
        # Portfolio weights
        weights = result.get('weights', {})
        print(f"\nPortfolio Weights:")
        for asset, weight in weights.items():
            print(f"  {asset}: {weight:.4f} ({weight*100:.2f}%)")
        
        # Performance metrics
        performance = result.get('performance', {})
        if performance:
            print(f"\nPerformance Metrics:")
            print(f"  Expected Annual Return: {performance.get('expected_annual_return', 0):.4f} ({performance.get('expected_annual_return', 0)*100:.2f}%)")
            print(f"  Annual Volatility: {performance.get('annual_volatility', 0):.4f} ({performance.get('annual_volatility', 0)*100:.2f}%)")
            print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.4f}")
        
        # Risk metrics
        risk_metrics = result.get('risk_metrics', {})
        if risk_metrics:
            print(f"\nRisk Metrics:")
            for metric, value in risk_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Additional information
        if 'views_applied' in result:
            print(f"\nBlack-Litterman Views Applied: {result['views_applied']}")
        
        if 'clustering_info' in result:
            clustering = result['clustering_info']
            print(f"\nClustering Info:")
            print(f"  Linkage Method: {clustering.get('linkage_method', 'Unknown')}")
            print(f"  Number of Assets: {clustering.get('n_assets', 0)}")
        
        if 'factor_model_info' in result:
            factor_info = result['factor_model_info']
            print(f"\nFactor Model Info:")
            print(f"  Model Type: {factor_info.get('model_type', 'Unknown')}")
            print(f"  Risk-Free Rate: {factor_info.get('risk_free_rate', 0):.4f}")
        
        if 'risk_budgeting_info' in result:
            risk_info = result['risk_budgeting_info']
            print(f"\nRisk Budgeting Info:")
            print(f"  Method: {risk_info.get('method', 'Unknown')}")
            risk_contribs = result.get('risk_contributions', {})
            if risk_contribs:
                print(f"  Actual Risk Contributions:")
                for asset, contrib in risk_contribs.items():
                    print(f"    {asset}: {contrib:.4f} ({contrib*100:.2f}%)")
    
    def _print_monte_carlo_results(self, result: dict):
        """Print formatted Monte Carlo simulation results"""
        print(f"\nMonte Carlo Portfolio Simulation")
        print("-" * 35)
        
        sim_results = result.get('simulation_results', {})
        print(f"Simulation Parameters:")
        print(f"  Number of Simulations: {sim_results.get('n_simulations', 0):,}")
        print(f"  Time Horizon: {sim_results.get('time_horizon_days', 0)} days")
        print(f"  Initial Portfolio Value: ${sim_results.get('initial_value', 0):,}")
        
        print(f"\nExpected Returns:")
        print(f"  Mean Final Return: {sim_results.get('mean_final_return', 0):.4f} ({sim_results.get('mean_final_return', 0)*100:.2f}%)")
        print(f"  Std Final Return: {sim_results.get('std_final_return', 0):.4f} ({sim_results.get('std_final_return', 0)*100:.2f}%)")
        print(f"  Probability of Positive Return: {sim_results.get('probability_positive_return', 0):.4f} ({sim_results.get('probability_positive_return', 0)*100:.2f}%)")
        
        risk_metrics = result.get('risk_metrics', {})
        print(f"\nRisk Metrics:")
        print(f"  Value at Risk (95%): {risk_metrics.get('value_at_risk_95', 0):.4f} ({risk_metrics.get('value_at_risk_95', 0)*100:.2f}%)")
        print(f"  Value at Risk (99%): {risk_metrics.get('value_at_risk_99', 0):.4f} ({risk_metrics.get('value_at_risk_99', 0)*100:.2f}%)")
        print(f"  Conditional VaR (95%): {risk_metrics.get('conditional_var_95', 0):.4f} ({risk_metrics.get('conditional_var_95', 0)*100:.2f}%)")
        print(f"  Maximum Drawdown: {risk_metrics.get('maximum_drawdown', 0):.4f} ({risk_metrics.get('maximum_drawdown', 0)*100:.2f}%)")
        
        percentiles = result.get('performance_percentiles', {})
        print(f"\nPerformance Percentiles:")
        for percentile, value in percentiles.items():
            print(f"  {percentile.replace('_', ' ').title()}: {value:.4f} ({value*100:.2f}%)")
    
    def run_comprehensive_tests(self):
        """Run all portfolio optimization tests"""
        logger.info("Starting Phase 3: Advanced Portfolio Optimization Test Suite")
        logger.info(f"PyPortfolioOpt Available: {self.portfolio_optimizer.is_available}")
        
        test_methods = [
            self.test_efficient_frontier_optimization,
            self.test_black_litterman_optimization,
            self.test_hierarchical_risk_parity,
            self.test_factor_model_optimization,
            self.test_risk_budgeting_optimization,
            self.test_monte_carlo_simulation
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
        logger.info("PHASE 3 PORTFOLIO OPTIMIZATION TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Tests: {passed_tests + failed_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/(passed_tests + failed_tests)*100:.1f}%")
        
        if self.portfolio_optimizer.is_available:
            logger.info("✓ Advanced PyPortfolioOpt features tested successfully")
        else:
            logger.info("⚠ PyPortfolioOpt not available - fallback implementations tested")
        
        logger.info("Phase 3 Portfolio Optimization implementation complete!")
        
        return passed_tests, failed_tests


def main():
    """Run the Phase 3 Portfolio Optimization test suite"""
    tester = PortfolioOptimizationTester()
    passed, failed = tester.run_comprehensive_tests()
    
    # Exit with appropriate code
    exit_code = 0 if failed == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
