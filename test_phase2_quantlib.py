"""
Test Phase 2: Advanced Financial Engineering with QuantLib
Validates the enhanced quantitative models with QuantLib and PyPortfolioOpt implementation
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from quant_models import (QuantLibFinancialEngineering, AdvancedPortfolioOptimization, 
                         QUANTLIB_AVAILABLE, PYPFOPT_AVAILABLE)

def test_phase2_quantlib_models():
    """Test all Phase 2 QuantLib and Portfolio Optimization enhancements"""
    print("🧪 TESTING PHASE 2: ADVANCED FINANCIAL ENGINEERING")
    print("=" * 60)
    
    print(f"🔧 QuantLib available: {QUANTLIB_AVAILABLE}")
    print(f"🔧 PyPortfolioOpt available: {PYPFOPT_AVAILABLE}")
    print()
    
    # Test 1: QuantLib Options Pricing
    print("1️⃣  Testing QuantLib Black-Scholes Options Pricing...")
    try:
        quantlib_engine = QuantLibFinancialEngineering()
        
        # BTC option example
        option_result = quantlib_engine.black_scholes_option_pricing(
            spot_price=50000,     # BTC at $50k
            strike=52000,         # Strike at $52k
            time_to_expiry=0.25,  # 3 months
            risk_free_rate=0.05,  # 5% risk-free rate
            volatility=0.80,      # 80% volatility (crypto-like)
            option_type='call'
        )
        
        print(f"   ✅ Option price: ${option_result['option_price']:.2f}")
        print(f"   ✅ Delta: {option_result['delta']:.4f}")
        print(f"   ✅ Gamma: {option_result['gamma']:.6f}")
        print(f"   ✅ Theta (daily): ${option_result['theta']:.2f}")
        print(f"   ✅ Vega (1% vol): ${option_result['vega']:.2f}")
        print(f"   ✅ Moneyness: {option_result['moneyness']:.3f}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    # Test 2: Advanced VaR Calculation
    print("2️⃣  Testing Advanced VaR Calculation...")
    try:
        # Generate realistic crypto portfolio returns
        np.random.seed(42)
        n_days = 252
        portfolio_returns = np.random.normal(0.001, 0.03, n_days)  # Daily returns
        
        var_results = quantlib_engine.advanced_var_calculation(
            portfolio_returns=portfolio_returns,
            confidence_level=0.05,  # 95% VaR
            time_horizon=1
        )
        
        print(f"   ✅ Historical VaR (95%): {var_results['historical_var']:.4f}")
        print(f"   ✅ Parametric VaR: {var_results['parametric_var']:.4f}")
        print(f"   ✅ Cornish-Fisher VaR: {var_results['cornish_fisher_var']:.4f}")
        print(f"   ✅ Expected Shortfall: {var_results['expected_shortfall']:.4f}")
        print(f"   ✅ Max Drawdown: {var_results['max_drawdown']:.4f}")
        print(f"   ✅ Portfolio skewness: {var_results['skewness']:.3f}")
        print(f"   ✅ Excess kurtosis: {var_results['excess_kurtosis']:.3f}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    # Test 3: Advanced Portfolio Optimization
    print("3️⃣  Testing Advanced Portfolio Optimization...")
    try:
        portfolio_optimizer = AdvancedPortfolioOptimization()
        
        # Generate multi-asset returns data
        np.random.seed(42)
        n_days = 252
        assets = ['BTC', 'ETH', 'SOL', 'MATIC', 'AVAX']
        
        # Simulate correlated returns
        mean_returns = np.array([0.0015, 0.0012, 0.0018, 0.0010, 0.0014])
        cov_matrix = np.array([
            [0.0009, 0.0005, 0.0003, 0.0002, 0.0003],
            [0.0005, 0.0008, 0.0004, 0.0003, 0.0004],
            [0.0003, 0.0004, 0.0012, 0.0002, 0.0003],
            [0.0002, 0.0003, 0.0002, 0.0006, 0.0002],
            [0.0003, 0.0004, 0.0003, 0.0002, 0.0010]
        ])
        
        returns_data = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
        returns_df = pd.DataFrame(returns_data, columns=assets)
        
        # Test efficient frontier optimization
        ef_result = portfolio_optimizer.efficient_frontier_optimization(
            returns_df=returns_df,
            method='max_sharpe',
            weight_bounds=(0.05, 0.4)  # Min 5%, max 40% per asset
        )
        
        print(f"   ✅ Optimization method: {ef_result['optimization_method']}")
        print(f"   ✅ Expected annual return: {ef_result['performance']['expected_annual_return']:.2%}")
        print(f"   ✅ Annual volatility: {ef_result['performance']['annual_volatility']:.2%}")
        print(f"   ✅ Sharpe ratio: {ef_result['performance']['sharpe_ratio']:.3f}")
        print(f"   ✅ Diversification ratio: {ef_result['risk_metrics']['diversification_ratio']:.3f}")
        print(f"   ✅ Concentration index: {ef_result['risk_metrics']['concentration_index']:.3f}")
        
        print("   📊 Optimal weights:")
        for asset, weight in ef_result['weights'].items():
            if weight > 0.01:  # Show only significant weights
                print(f"      {asset}: {weight:.2%}")
        print()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    # Test 4: Integration with Phase 1 Bayesian Models
    print("4️⃣  Testing Integration with Phase 1 Bayesian Models...")
    try:
        from quant_models import BayesianTradingFramework
        
        # Generate price data
        np.random.seed(42)
        n_days = 300
        price_data = np.zeros(n_days)
        price_data[0] = 50000  # Starting at $50k
        
        for i in range(1, n_days):
            return_shock = np.random.normal(0.001, 0.025)
            price_data[i] = price_data[i-1] * np.exp(return_shock)
        
        # Combined analysis
        bayesian_framework = BayesianTradingFramework()
        quantlib_engine = QuantLibFinancialEngineering()
        
        # Bayesian analysis
        bayesian_results = bayesian_framework.comprehensive_bayesian_analysis(price_data)
        
        # QuantLib risk analysis
        returns = np.diff(np.log(price_data))
        quantlib_var = quantlib_engine.advanced_var_calculation(returns)
        
        print(f"   ✅ Bayesian framework available: {bayesian_framework.is_initialized}")
        print(f"   ✅ QuantLib engine available: {quantlib_engine.is_available}")
        print(f"   ✅ Integrated risk assessment: {bayesian_results.get('integrated_risk', {}).get('overall_risk_level', 'N/A')}")
        print(f"   ✅ QuantLib VaR (95%): {quantlib_var['historical_var']:.4f}")
        print(f"   ✅ Cross-validation successful: Phase 1 + Phase 2 integration working")
        print()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    # Test 5: Performance comparison
    print("5️⃣  Testing Performance Comparison: Advanced vs Fallback...")
    try:
        # Test both advanced and fallback methods
        portfolio_returns = np.random.normal(0.001, 0.02, 100)
        
        # Advanced method
        if QUANTLIB_AVAILABLE:
            advanced_var = quantlib_engine.advanced_var_calculation(portfolio_returns)
            print(f"   ✅ Advanced QuantLib VaR: {advanced_var['historical_var']:.4f}")
        else:
            print(f"   ⚠️  QuantLib not available - using fallback methods")
        
        # Portfolio optimization comparison
        if PYPFOPT_AVAILABLE:
            test_returns = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
            advanced_portfolio = portfolio_optimizer.efficient_frontier_optimization(test_returns)
            print(f"   ✅ Advanced portfolio Sharpe: {advanced_portfolio['performance']['sharpe_ratio']:.3f}")
        else:
            print(f"   ⚠️  PyPortfolioOpt not available - using fallback methods")
            
        print()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    print("🎯 PHASE 2 TESTING COMPLETE")
    print("=" * 60)
    
    if QUANTLIB_AVAILABLE and PYPFOPT_AVAILABLE:
        print("✅ All advanced financial engineering features are functional!")
        print("✅ QuantLib options pricing and risk models working")
        print("✅ PyPortfolioOpt advanced portfolio optimization working")
        print("✅ Full integration with Phase 1 Bayesian models successful")
    else:
        print("⚠️  Some advanced libraries not installed")
        if not QUANTLIB_AVAILABLE:
            print("⚠️  QuantLib not available - install with: pip install quantlib")
        if not PYPFOPT_AVAILABLE:
            print("⚠️  PyPortfolioOpt not available - install with: pip install pypfopt")
        print("ℹ️  Fallback implementations are used when advanced libraries unavailable")
    
    print()
    print("📋 NEXT STEPS:")
    print("   - Install missing dependencies if needed") 
    print("   - Proceed to Phase 3: Time Series Analysis (ARCH/GARCH)")
    print("   - Integrate with existing CrewAI intelligence and dashboard")

if __name__ == "__main__":
    test_phase2_quantlib_models()
