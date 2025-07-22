#!/usr/bin/env python3
"""
Test Script for Physics-Based Risk Models
Validates the @khemkapital methodology implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from src.quant_models import AdvancedPhysicsModels
import json
from datetime import datetime

def generate_test_data():
    """Generate realistic price data for testing"""
    # Generate multiple market scenarios
    scenarios = {
        'stable_market': generate_brownian_motion(1000, 0.0001, 0.02),
        'volatile_market': generate_brownian_motion(1000, 0.0002, 0.08),
        'trending_market': generate_brownian_motion(1000, 0.0008, 0.03),
        'crash_scenario': generate_crash_scenario(),
        'random_walk': generate_brownian_motion(1000, 0.0, 0.04)
    }
    return scenarios

def generate_brownian_motion(n_periods, drift, volatility, initial_price=43000):
    """Generate geometric Brownian motion"""
    dt = 1/365  # Daily periods
    prices = [initial_price]
    
    for _ in range(n_periods - 1):
        random_shock = np.random.normal(0, 1)
        price_change = drift * dt + volatility * np.sqrt(dt) * random_shock
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    return np.array(prices)

def generate_crash_scenario():
    """Generate a market crash scenario with trauma"""
    # Normal period
    normal_period = generate_brownian_motion(800, 0.0002, 0.02, 43000)
    
    # Crash period (sudden drop)
    crash_start = normal_period[-1]
    crash_period = []
    current_price = crash_start
    
    # Sharp decline over 20 days
    for i in range(20):
        decline = np.random.normal(-0.05, 0.02)  # Average 5% daily decline
        current_price *= (1 + decline)
        crash_period.append(current_price)
    
    # Recovery period
    recovery_start = crash_period[-1]
    recovery_period = generate_brownian_motion(180, 0.001, 0.04, recovery_start)
    
    return np.concatenate([normal_period, crash_period, recovery_period])

def test_physics_models():
    """Test all physics-based models"""
    print("🔬 Testing @khemkapital Physics-Based Risk Models")
    print("=" * 60)
    
    scenarios = generate_test_data()
    results = {}
    
    for scenario_name, price_data in scenarios.items():
        print(f"\n📊 Analyzing {scenario_name.upper().replace('_', ' ')}")
        print("-" * 40)
        
        # Test Information Entropy
        entropy_results = AdvancedPhysicsModels.information_entropy_risk(price_data)
        print(f"🔍 Information Entropy: {entropy_results['entropy']:.3f}")
        print(f"   Risk Level: {entropy_results['risk_level']}")
        print(f"   Market Readability: {entropy_results['readability']}")
        
        # Test Hurst Exponent (Memory)
        memory_results = AdvancedPhysicsModels.hurst_exponent_memory(price_data)
        print(f"🧠 Hurst Exponent: {memory_results['hurst_exponent']:.3f}")
        print(f"   Memory Type: {memory_results['memory_type']}")
        print(f"   Trauma Detected: {memory_results['trauma_detected']}")
        
        # Test Lyapunov Exponent (Instability)
        instability_results = AdvancedPhysicsModels.lyapunov_instability_detection(price_data)
        print(f"⚡ Instability Score: {instability_results['instability_score']:.3f}")
        print(f"   Instability Level: {instability_results['instability_level']}")
        print(f"   Systemic Risk: {instability_results['systemic_risk']}")
        
        # Test Regime Detection
        regime_results = AdvancedPhysicsModels.regime_transition_detection(price_data)
        print(f"🌊 Market Regime: {regime_results['regime']}")
        print(f"   Stability: {regime_results['stability']}")
        print(f"   Transition Probability: {regime_results['transition_probability']:.3f}")
        
        # Store results
        results[scenario_name] = {
            'entropy': entropy_results,
            'memory': memory_results,
            'instability': instability_results,
            'regime': regime_results,
            'combined_risk_score': (
                entropy_results['entropy'] + 
                abs(memory_results['hurst_exponent'] - 0.5) * 2 + 
                instability_results['instability_score'] * 5
            ) / 3
        }
    
    # Summary comparison
    print("\n📈 PHYSICS RISK COMPARISON ACROSS SCENARIOS")
    print("=" * 60)
    print(f"{'Scenario':<20} {'Entropy':<10} {'Hurst':<8} {'Instability':<12} {'Risk Score':<10}")
    print("-" * 60)
    
    for scenario, data in results.items():
        entropy = data['entropy']['entropy']
        hurst = data['memory']['hurst_exponent']
        instability = data['instability']['instability_score']
        risk_score = data['combined_risk_score']
        
        print(f"{scenario:<20} {entropy:<10.3f} {hurst:<8.3f} {instability:<12.3f} {risk_score:<10.3f}")
    
    # Save results
    save_test_results(results)
    
    return results

def save_test_results(results):
    """Save test results to file"""
    try:
        output = {
            'test_timestamp': datetime.now().isoformat(),
            'physics_model_test_results': results,
            'summary': {
                'models_tested': ['information_entropy', 'hurst_exponent', 'lyapunov_instability', 'regime_detection'],
                'scenarios_tested': list(results.keys()),
                'test_status': 'completed'
            }
        }
        
        with open('physics_models_test_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to: physics_models_test_results.json")
        
    except Exception as e:
        print(f"❌ Error saving test results: {e}")

def demonstrate_risk_amplification():
    """Demonstrate how physics models detect risk amplification"""
    print("\n🚨 RISK AMPLIFICATION DEMONSTRATION")
    print("=" * 50)
    
    # Create a scenario with building instability
    base_price = 43000
    n_periods = 500
    
    # Start stable, then become increasingly unstable
    prices = [base_price]
    volatility = 0.02  # Start with 2% volatility
    
    for i in range(n_periods - 1):
        # Gradually increase volatility (building instability)
        progress = i / n_periods
        current_vol = volatility * (1 + progress * 3)  # Up to 8% volatility
        
        # Add some momentum (creates memory)
        momentum = 0.7 if i > 0 else 0
        prev_return = (prices[-1] - prices[-2]) / prices[-2] if i > 0 else 0
        
        random_shock = np.random.normal(0, 1)
        base_return = momentum * prev_return + current_vol * random_shock / np.sqrt(365)
        
        new_price = prices[-1] * (1 + base_return)
        prices.append(new_price)
    
    price_data = np.array(prices)
    
    # Analyze with physics models
    entropy_analysis = AdvancedPhysicsModels.information_entropy_risk(price_data)
    memory_analysis = AdvancedPhysicsModels.hurst_exponent_memory(price_data)
    instability_analysis = AdvancedPhysicsModels.lyapunov_instability_detection(price_data)
    
    print(f"📊 Building Instability Scenario Analysis:")
    print(f"   Final Entropy Risk: {entropy_analysis['risk_level']} ({entropy_analysis['entropy']:.3f})")
    print(f"   Memory Persistence: H = {memory_analysis['hurst_exponent']:.3f} ({memory_analysis['memory_type']})")
    print(f"   System Instability: {instability_analysis['instability_level']} ({instability_analysis['instability_score']:.3f})")
    
    # Calculate risk amplification
    base_var = np.std(np.diff(np.log(price_data[:100])))  # Early period volatility
    final_var = np.std(np.diff(np.log(price_data[-100:])))  # Late period volatility
    
    amplification_factor = final_var / base_var if base_var > 0 else 1.0
    
    print(f"   Volatility Amplification: {amplification_factor:.2f}x")
    print(f"   Physics Risk Amplification: {instability_analysis['shock_amplification_factor']:.2f}x")

if __name__ == "__main__":
    try:
        print("🚀 Starting Physics-Based Risk Models Test")
        print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Run main tests
        results = test_physics_models()
        
        # Demonstrate risk amplification
        demonstrate_risk_amplification()
        
        print("\n✅ All physics model tests completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. Review test results in physics_models_test_results.json")
        print("   2. Run your trading system to see physics models in action")
        print("   3. Check dashboard for real-time physics risk metrics")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
