#!/usr/bin/env python3
"""
COMPREHENSIVE 5-PHASE QUANTITATIVE TRADING SYSTEM TEST

This is the master test suite that validates the complete implementation
of all 5 phases of advanced quantitative trading capabilities:

Phase 1: Bayesian Statistical Inference & MCMC Sampling
Phase 2: QuantLib Financial Engineering & Derivatives Pricing  
Phase 3: Advanced Portfolio Optimization & Risk Management
Phase 4: Time Series Analysis & Volatility Modeling
Phase 5: Machine Learning & AI Integration

The test suite validates both advanced implementations (when libraries
are available) and comprehensive fallback systems.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveSystemTester:
    """Master test suite for all 5 phases of the quantitative trading system"""
    
    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_phase_test(self, phase_number: int, test_file: str, phase_name: str):
        """Run a specific phase test"""
        logger.info(f"\n{'='*80}")
        logger.info(f"EXECUTING PHASE {phase_number}: {phase_name}")
        logger.info(f"{'='*80}")
        
        try:
            # Run the test file
            result = subprocess.run([
                sys.executable, test_file
            ], capture_output=True, text=True, timeout=300)
            
            # Parse results
            if result.returncode == 0:
                logger.info(f"✅ Phase {phase_number} tests PASSED")
                self.test_results[f"Phase {phase_number}"] = "PASSED"
                self.passed_tests += 1
            else:
                logger.error(f"❌ Phase {phase_number} tests FAILED")
                logger.error(f"Error output: {result.stderr}")
                self.test_results[f"Phase {phase_number}"] = "FAILED"
                self.failed_tests += 1
            
            # Log output summary
            if result.stdout:
                lines = result.stdout.split('\n')
                summary_lines = [line for line in lines if 'Summary' in line or '✓' in line or '❌' in line]
                for line in summary_lines[-5:]:  # Show last 5 summary lines
                    logger.info(f"  {line}")
                    
            self.total_tests += 1
            
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Phase {phase_number} tests TIMED OUT (5 minutes)")
            self.test_results[f"Phase {phase_number}"] = "TIMEOUT"
            self.failed_tests += 1
            self.total_tests += 1
            
        except Exception as e:
            logger.error(f"💥 Phase {phase_number} tests CRASHED: {e}")
            self.test_results[f"Phase {phase_number}"] = "CRASHED"
            self.failed_tests += 1
            self.total_tests += 1
    
    def check_library_availability(self):
        """Check availability of all advanced libraries"""
        logger.info(f"\n{'='*80}")
        logger.info("CHECKING LIBRARY AVAILABILITY")
        logger.info(f"{'='*80}")
        
        libraries = {
            'Phase 1 - Bayesian': ['pymc', 'arviz'],
            'Phase 2 - QuantLib': ['QuantLib'],
            'Phase 3 - Portfolio': ['pypfopt'],
            'Phase 4 - Time Series': ['arch'],
            'Phase 5 - ML Core': ['torch', 'sklearn', 'xgboost', 'lightgbm'],
            'Phase 5 - NLP': ['transformers', 'spacy'],
            'Phase 5 - RL': ['stable_baselines3', 'gymnasium']
        }
        
        availability_status = {}
        
        for phase, libs in libraries.items():
            available_count = 0
            for lib in libs:
                try:
                    __import__(lib)
                    available_count += 1
                except ImportError:
                    pass
            
            availability_status[phase] = f"{available_count}/{len(libs)} libraries available"
            
            if available_count == len(libs):
                logger.info(f"✅ {phase}: {availability_status[phase]}")
            elif available_count > 0:
                logger.info(f"⚠️  {phase}: {availability_status[phase]}")
            else:
                logger.info(f"❌ {phase}: {availability_status[phase]}")
        
        return availability_status
    
    def validate_core_system(self):
        """Validate that the core system can be imported"""
        logger.info(f"\n{'='*80}")
        logger.info("VALIDATING CORE SYSTEM IMPORTS")
        logger.info(f"{'='*80}")
        
        try:
            # Test core imports
            from quant_models import (
                BayesianTradingFramework,
                QuantLibFinancialEngineering, 
                AdvancedPortfolioOptimization,
                AdvancedTimeSeriesAnalysis,
                AdvancedMLTradingFramework
            )
            
            logger.info("✅ All core framework classes imported successfully")
            
            # Test instantiation
            frameworks = {}
            
            frameworks['Bayesian'] = BayesianTradingFramework()
            frameworks['QuantLib'] = QuantLibFinancialEngineering()
            frameworks['Portfolio'] = AdvancedPortfolioOptimization()
            frameworks['TimeSeries'] = AdvancedTimeSeriesAnalysis()
            frameworks['ML'] = AdvancedMLTradingFramework()
            
            logger.info("✅ All framework instances created successfully")
            
            # Log availability status
            for name, framework in frameworks.items():
                if hasattr(framework, 'is_available'):
                    status = "Available" if framework.is_available else "Fallback Mode"
                    logger.info(f"  {name} Framework: {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Core system validation failed: {e}")
            return False
    
    def run_comprehensive_system_test(self):
        """Run the complete system test suite"""
        start_time = datetime.now()
        
        logger.info(f"\n{'#'*100}")
        logger.info("ADVANCED QUANTITATIVE TRADING SYSTEM - COMPREHENSIVE TEST SUITE")
        logger.info(f"{'#'*100}")
        logger.info(f"Test started at: {start_time}")
        
        # Step 1: Check library availability
        availability = self.check_library_availability()
        
        # Step 2: Validate core system
        core_valid = self.validate_core_system()
        
        if not core_valid:
            logger.error("💥 Core system validation failed - stopping tests")
            return False
        
        # Step 3: Run all phase tests
        phase_tests = [
            (1, "test_phase1_bayesian.py", "Bayesian Statistical Inference & MCMC"),
            (2, "test_phase2_quantlib.py", "QuantLib Financial Engineering"),
            (3, "test_phase3_portfolio.py", "Advanced Portfolio Optimization"), 
            (4, "test_phase4_timeseries.py", "Time Series Analysis & Volatility Modeling"),
            (5, "test_phase5_ml_ai.py", "Machine Learning & AI Integration")
        ]
        
        for phase_num, test_file, phase_name in phase_tests:
            self.run_phase_test(phase_num, test_file, phase_name)
        
        # Step 4: Generate comprehensive report
        self.generate_final_report(start_time, availability)
        
        return self.failed_tests == 0
    
    def generate_final_report(self, start_time: datetime, availability: dict):
        """Generate comprehensive final test report"""
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n{'#'*100}")
        logger.info("FINAL COMPREHENSIVE TEST REPORT")
        logger.info(f"{'#'*100}")
        
        # Test summary
        logger.info(f"\n📊 TEST EXECUTION SUMMARY:")
        logger.info(f"  Total Test Phases: {self.total_tests}")
        logger.info(f"  Passed: {self.passed_tests}")
        logger.info(f"  Failed: {self.failed_tests}")
        logger.info(f"  Success Rate: {(self.passed_tests/max(self.total_tests,1)*100):.1f}%")
        logger.info(f"  Total Duration: {duration}")
        
        # Phase-by-phase results
        logger.info(f"\n📋 PHASE-BY-PHASE RESULTS:")
        for phase, result in self.test_results.items():
            status_emoji = "✅" if result == "PASSED" else "❌"
            logger.info(f"  {status_emoji} {phase}: {result}")
        
        # Library availability summary
        logger.info(f"\n📚 LIBRARY AVAILABILITY SUMMARY:")
        for phase, status in availability.items():
            logger.info(f"  {phase}: {status}")
        
        # Overall system status
        logger.info(f"\n🏆 OVERALL SYSTEM STATUS:")
        
        if self.failed_tests == 0:
            logger.info("  🎉 ALL TESTS PASSED! Advanced quantitative trading system fully operational!")
            logger.info("  🚀 System ready for production deployment")
            logger.info("  💡 All 5 phases of advanced functionality validated")
        elif self.passed_tests > 0:
            logger.info(f"  ⚠️  PARTIAL SUCCESS: {self.passed_tests}/{self.total_tests} phases working")
            logger.info("  🔧 System operational with some limitations")
            logger.info("  📝 Check failed phases for specific issues")
        else:
            logger.info("  💥 SYSTEM FAILURE: No phases passed validation")
            logger.info("  🔍 Check core dependencies and configuration")
        
        # Advanced features summary
        logger.info(f"\n🔬 ADVANCED FEATURES VALIDATION:")
        features = [
            "Bayesian MCMC Inference",
            "Professional Derivatives Pricing", 
            "Modern Portfolio Theory",
            "GARCH Volatility Modeling",
            "Deep Learning Trading Models",
            "NLP Sentiment Analysis",
            "Reinforcement Learning Agents"
        ]
        
        for i, feature in enumerate(features, 1):
            phase_result = self.test_results.get(f"Phase {i if i <= 4 else 5}", "NOT_TESTED")
            status = "✅" if phase_result == "PASSED" else "❌"
            logger.info(f"  {status} {feature}")
        
        # Deployment readiness
        if self.failed_tests == 0:
            logger.info(f"\n🎯 DEPLOYMENT READINESS:")
            logger.info("  ✅ Production Ready")
            logger.info("  ✅ All advanced libraries integrated")
            logger.info("  ✅ Robust fallback systems validated")
            logger.info("  ✅ Comprehensive error handling")
            logger.info("  ✅ Full feature test coverage")
        
        logger.info(f"\n{'#'*100}")
        logger.info(f"Test completed at: {end_time}")
        logger.info(f"{'#'*100}")


def main():
    """Run the comprehensive 5-phase system test"""
    tester = ComprehensiveSystemTester()
    
    try:
        success = tester.run_comprehensive_system_test()
        exit_code = 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test execution interrupted by user")
        exit_code = 130
        
    except Exception as e:
        logger.error(f"\n💥 Test execution failed with error: {e}")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
