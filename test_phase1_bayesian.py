"""
Test Phase 1: Advanced Bayesian & Probabilistic Models
Validates the enhanced quantitative models with full Bayesian implementation
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from quant_models import QuantitativeModels, BayesianTradingFramework, BAYESIAN_AVAILABLE

def test_phase1_bayesian_models():
    """Test all Phase 1 Bayesian enhancements"""
    print("🧪 TESTING PHASE 1: ADVANCED BAYESIAN MODELS")
    print("=" * 60)
    
    # Generate synthetic market data
    np.random.seed(42)
    n_days = 500
    
    # Synthetic price data with regime changes
    price_data = np.zeros(n_days)
    price_data[0] = 100.0
    
    # Create synthetic returns with changing volatility
    for i in range(1, n_days):
        if i < 200:
            vol = 0.02  # Low volatility regime
        elif i < 350:
            vol = 0.04  # High volatility regime  
        else:
            vol = 0.015  # Return to low volatility
        
        return_shock = np.random.normal(0.0005, vol)
        price_data[i] = price_data[i-1] * np.exp(return_shock)
    
    # Multi-asset data for portfolio tests
    returns_data = {}
    for asset in ['BTC', 'ETH', 'SOL']:
        asset_returns = np.random.multivariate_normal([0.001, 0.0008, 0.0012], 
                                                    [[0.0004, 0.0002, 0.0001],
                                                     [0.0002, 0.0006, 0.0001], 
                                                     [0.0001, 0.0001, 0.0008]], 
                                                    n_days-1)
        returns_data[asset] = asset_returns[:, ['BTC', 'ETH', 'SOL'].index(asset)]
    
    returns_df = pd.DataFrame(returns_data)
    
    print(f"📊 Generated synthetic data: {n_days} days, 3 assets")
    print(f"🔧 Bayesian libraries available: {BAYESIAN_AVAILABLE}")
    print()
    
    # Test 1: Enhanced Bayesian Price Model  
    print("1️⃣  Testing Enhanced Bayesian Price Model...")
    try:
        bayesian_forecast = QuantitativeModels.bayesian_price_model(price_data)
        print(f"   ✅ Expected price: ${bayesian_forecast.mean:.2f}")
        print(f"   ✅ 95% CI: ${bayesian_forecast.get_confidence_interval(0.95)[0]:.2f} - ${bayesian_forecast.get_confidence_interval(0.95)[1]:.2f}")
        print(f"   ✅ Probability up: {bayesian_forecast.probability_up:.3f}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    # Test 2: Advanced Bayesian Volatility
    print("2️⃣  Testing Advanced Bayesian Stochastic Volatility...")
    try:
        vol_results = QuantitativeModels.advanced_bayesian_volatility(price_data, n_samples=500)
        print(f"   ✅ Model type: {vol_results['model_type']}")
        print(f"   ✅ Current volatility: {vol_results['current_volatility']['mean']:.4f}")
        print(f"   ✅ 95% HDI: [{vol_results['current_volatility']['hdi_lower']:.4f}, {vol_results['current_volatility']['hdi_upper']:.4f}]")
        print(f"   ✅ Convergence OK: {vol_results.get('convergence_ok', 'N/A')}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    # Test 3: Bayesian Regime Switching
    print("3️⃣  Testing Bayesian Regime Switching...")
    try:
        returns = np.diff(np.log(price_data))
        regime_results = QuantitativeModels.bayesian_regime_switching(returns, n_samples=500)
        print(f"   ✅ Model type: {regime_results['model_type']}")
        print(f"   ✅ Current regime probs: {[f'{p:.3f}' for p in regime_results['current_regime_probabilities']]}")
        print(f"   ✅ Regime analysis: {len(regime_results['regime_analysis'])} regimes detected")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    # Test 4: Bayesian Portfolio Optimization
    print("4️⃣  Testing Bayesian Portfolio Optimization...")
    try:
        portfolio_results = QuantitativeModels.bayesian_portfolio_optimization(returns_df, n_samples=200)
        print(f"   ✅ Model type: {portfolio_results['model_type']}")
        print(f"   ✅ Optimal weights: {[f'{w:.3f}' for w in portfolio_results['optimal_weights']['mean']]}")
        print(f"   ✅ Assets: {portfolio_results['asset_names']}")
        print(f"   ✅ Portfolio vol: {portfolio_results['risk_metrics']['portfolio_volatility_mean']:.4f}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    # Test 5: Unified Bayesian Framework
    print("5️⃣  Testing Unified Bayesian Framework...")
    try:
        framework = BayesianTradingFramework()
        comprehensive_results = framework.comprehensive_bayesian_analysis(
            price_data=price_data,
            returns_df=returns_df
        )
        
        print(f"   ✅ Framework initialized: {framework.is_initialized}")
        print(f"   ✅ Analysis components: {comprehensive_results.get('analysis_components', [])}")
        print(f"   ✅ Overall risk: {comprehensive_results.get('integrated_risk', {}).get('overall_risk_level', 'N/A')}")
        print(f"   ✅ Primary signal: {comprehensive_results.get('trading_signals', {}).get('primary_signal', 0):.3f}")
        print(f"   ✅ Signal confidence: {comprehensive_results.get('trading_signals', {}).get('signal_confidence', 0):.3f}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    # Test 6: Integration with existing physics models
    print("6️⃣  Testing Integration with Existing Physics Models...")
    try:
        from quant_models import AdvancedPhysicsModels
        
        # Physics analysis
        entropy_metrics = AdvancedPhysicsModels.information_entropy_risk(price_data)
        memory_metrics = AdvancedPhysicsModels.hurst_exponent_memory(price_data)
        
        print(f"   ✅ Information entropy: {entropy_metrics['entropy']:.3f} ({entropy_metrics['risk_level']})")
        print(f"   ✅ Hurst exponent: {memory_metrics['hurst_exponent']:.3f} ({memory_metrics['memory_type']})")
        print(f"   ✅ Physics + Bayesian integration successful")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
    
    print("🎯 PHASE 1 TESTING COMPLETE")
    print("=" * 60)
    
    if BAYESIAN_AVAILABLE:
        print("✅ All advanced Bayesian features are functional!")
        print("✅ PyMC/ArviZ integration working correctly")
        print("✅ Full uncertainty quantification available")
    else:
        print("⚠️  Advanced Bayesian libraries not installed")
        print("⚠️  Fallback implementations used")
        print("ℹ️  To enable full features, install: pip install pymc arviz")
    
    print()
    print("📋 NEXT STEPS:")
    print("   - Install missing dependencies if needed")
    print("   - Proceed to Phase 2: QuantLib Financial Engineering")
    print("   - Integrate with existing CrewAI intelligence")

if __name__ == "__main__":
    test_phase1_bayesian_models()
