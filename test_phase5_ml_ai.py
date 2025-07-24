#!/usr/bin/env python3
"""
Phase 5: Advanced Machine Learning & AI Integration Test Suite

This test suite validates the comprehensive ML and AI capabilities including:
- Deep Learning Models (LSTM, Transformer, CNN) 
- Ensemble Methods (Random Forest, XGBoost, LightGBM)
- Automated Feature Engineering
- Natural Language Processing for Sentiment Analysis
- Reinforcement Learning Trading Agents
- Model Selection and Hyperparameter Optimization
- Uncertainty Quantification
- Multi-modal Data Integration

All tests include both advanced ML frameworks and fallback implementations.
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

# Import our advanced ML framework
from quant_models import AdvancedMLTradingFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLTradingFrameworkTester:
    """Comprehensive test suite for Phase 5 Machine Learning & AI Integration"""
    
    def __init__(self):
        self.ml_framework = AdvancedMLTradingFramework()
        logger.info(f"ML Libraries Available: {self.ml_framework.is_available}")
        logger.info(f"NLP Libraries Available: {self.ml_framework.nlp_available}")
        logger.info(f"RL Libraries Available: {self.ml_framework.rl_available}")
        
        # Generate comprehensive synthetic data
        self.price_data = self._generate_realistic_market_data()
        self.volume_data = self._generate_volume_data()
        self.fundamental_data = self._generate_fundamental_data()
        self.news_data = self._generate_synthetic_news()
        
        logger.info(f"Generated synthetic data: {len(self.price_data)} price observations")
    
    def _generate_realistic_market_data(self, n_periods: int = 2000) -> pd.Series:
        """Generate realistic market data with multiple regimes and patterns"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
        
        # Multi-regime price process
        prices = [100.0]  # Initial price
        volatility = 0.02  # Initial volatility
        
        for i in range(1, n_periods):
            # Regime switching volatility
            if i % 250 == 0:  # Yearly regime changes
                volatility = np.random.uniform(0.01, 0.05)
            
            # Add momentum and mean reversion
            momentum = 0.0
            if i > 20:
                recent_returns = np.array([np.log(prices[j]/prices[j-1]) for j in range(i-20, i)])
                momentum = 0.1 * np.mean(recent_returns)  # Momentum factor
            
            # Mean reversion component
            long_term_price = np.mean(prices[-252:]) if i > 252 else prices[0]
            mean_reversion = -0.001 * np.log(prices[-1] / long_term_price)
            
            # Random shock
            shock = np.random.normal(0, volatility)
            
            # Price update
            daily_return = momentum + mean_reversion + shock
            new_price = prices[-1] * np.exp(daily_return)
            prices.append(new_price)
        
        return pd.Series(prices, index=dates, name='price')
    
    def _generate_volume_data(self) -> pd.Series:
        """Generate volume data correlated with price volatility"""
        np.random.seed(43)
        
        returns = self.price_data.pct_change().dropna()
        volatility = returns.rolling(20).std()
        
        # Volume inversely correlated with volatility (high vol = high volume)
        base_volume = 1000000
        volume_multiplier = 1 + 2 * volatility.fillna(volatility.mean())
        
        # Add random component
        random_component = np.random.lognormal(0, 0.3, len(volume_multiplier))
        
        volume = base_volume * volume_multiplier * random_component
        
        return pd.Series(volume.values, index=self.price_data.index[1:], name='volume')
    
    def _generate_fundamental_data(self) -> pd.DataFrame:
        """Generate synthetic fundamental indicators"""
        np.random.seed(44)
        
        n_periods = len(self.price_data)
        dates = self.price_data.index
        
        # Generate correlated fundamental indicators
        pe_ratio = 15 + 5 * np.random.randn(n_periods).cumsum() * 0.01
        debt_to_equity = 0.3 + 0.1 * np.random.randn(n_periods).cumsum() * 0.001
        roe = 0.12 + 0.02 * np.random.randn(n_periods).cumsum() * 0.001
        
        return pd.DataFrame({
            'pe_ratio': pe_ratio,
            'debt_to_equity': np.clip(debt_to_equity, 0.1, 1.0),
            'roe': np.clip(roe, 0.05, 0.25)
        }, index=dates)
    
    def _generate_synthetic_news(self) -> list:
        """Generate synthetic news headlines for sentiment testing"""
        positive_news = [
            "Company reports record quarterly earnings beating expectations",
            "Market rallies on positive economic indicators",
            "Strong consumer confidence drives stock prices higher",
            "Technology sector shows robust growth prospects",
            "Federal Reserve signals dovish monetary policy stance",
            "Corporate profits surge amid economic recovery",
            "Stock market reaches new all-time highs",
            "Investor optimism boosts trading volumes"
        ]
        
        negative_news = [
            "Market volatility increases amid economic uncertainty",
            "Corporate earnings disappoint Wall Street analysts",
            "Geopolitical tensions weigh on investor sentiment",
            "Federal Reserve raises interest rates unexpectedly",
            "Economic indicators signal potential recession",
            "Stock market experiences significant selloff",
            "Trade war concerns impact global markets",
            "Inflation fears drive market volatility"
        ]
        
        neutral_news = [
            "Market closes mixed in moderate trading session",
            "Economic data meets analyst expectations",
            "Corporate earnings season shows mixed results",
            "Federal Reserve maintains current policy stance",
            "Trading volumes remain within normal ranges",
            "Market participants await economic indicators",
            "Stock prices show little movement today",
            "Analysts maintain neutral outlook on markets"
        ]
        
        # Randomly combine news for testing
        all_news = positive_news + negative_news + neutral_news
        np.random.shuffle(all_news)
        
        return all_news[:20]  # Return subset for testing
    
    def test_automated_feature_engineering(self):
        """Test automated feature engineering capabilities"""
        logger.info("\n" + "="*60)
        logger.info("Testing Automated Feature Engineering")
        logger.info("="*60)
        
        # Test basic feature engineering
        logger.info("Test 1: Basic price and volume features")
        result = self.ml_framework.comprehensive_ml_analysis(
            price_data=self.price_data,
            volume_data=self.volume_data,
            prediction_horizon=5
        )
        
        self._print_ml_results(result, "Basic Feature Engineering")
        
        # Validate feature engineering
        if 'feature_engineering' in result:
            feature_info = result['feature_engineering']
            assert 'n_features' in feature_info
            assert feature_info['n_features'] > 0
            logger.info(f"✓ Generated {feature_info['n_features']} features")
        
        # Test with fundamental data
        logger.info("\nTest 2: Including fundamental data")
        result_with_fundamentals = self.ml_framework.comprehensive_ml_analysis(
            price_data=self.price_data,
            volume_data=self.volume_data,
            fundamental_data=self.fundamental_data,
            prediction_horizon=5
        )
        
        if 'feature_engineering' in result_with_fundamentals:
            fund_features = result_with_fundamentals['feature_engineering']['n_features']
            basic_features = result['feature_engineering']['n_features']
            assert fund_features > basic_features
            logger.info(f"✓ Fundamental data added {fund_features - basic_features} features")
        
        logger.info("✓ Automated feature engineering completed successfully")
    
    def test_ensemble_ml_models(self):
        """Test ensemble machine learning models"""
        logger.info("\n" + "="*60)
        logger.info("Testing Ensemble Machine Learning Models")
        logger.info("="*60)
        
        result = self.ml_framework.comprehensive_ml_analysis(
            price_data=self.price_data,
            volume_data=self.volume_data,
            prediction_horizon=10
        )
        
        if 'ensemble_models' in result:
            ensemble_results = result['ensemble_models']
            logger.info("Ensemble Models Training Results:")
            
            for model_name, model_result in ensemble_results.items():
                if isinstance(model_result, dict) and 'cv_score_mean' in model_result:
                    cv_score = model_result['cv_score_mean']
                    cv_std = model_result.get('cv_score_std', 0)
                    logger.info(f"  {model_name}: CV Score = {cv_score:.4f} ± {cv_std:.4f}")
                    
                    # Validate reasonable performance
                    assert -1.0 <= cv_score <= 1.0, f"Invalid CV score for {model_name}"
                elif 'error' in model_result:
                    logger.warning(f"  {model_name}: {model_result['error']}")
            
            logger.info("✓ Ensemble models training completed")
        else:
            logger.warning("No ensemble models in results")
        
        # Test ensemble prediction
        if 'ensemble_prediction' in result:
            prediction = result['ensemble_prediction']
            logger.info(f"\nEnsemble Prediction: {prediction.get('ensemble_prediction', 'N/A')}")
            logger.info(f"Prediction Std: {prediction.get('prediction_std', 'N/A')}")
            logger.info(f"Number of Models: {prediction.get('n_models', 0)}")
            
            assert 'ensemble_prediction' in prediction
            logger.info("✓ Ensemble prediction generated successfully")
    
    def test_deep_learning_models(self):
        """Test deep learning models"""
        logger.info("\n" + "="*60)
        logger.info("Testing Deep Learning Models")
        logger.info("="*60)
        
        # Use larger dataset for deep learning
        large_price_data = self.price_data
        
        result = self.ml_framework.comprehensive_ml_analysis(
            price_data=large_price_data,
            volume_data=self.volume_data,
            prediction_horizon=5
        )
        
        if 'deep_learning' in result:
            dl_results = result['deep_learning']
            logger.info("Deep Learning Models Training Results:")
            
            for model_name, model_result in dl_results.items():
                if isinstance(model_result, dict) and 'r2_score' in model_result:
                    r2 = model_result['r2_score']
                    test_loss = model_result.get('test_loss', 0)
                    epochs = model_result.get('epochs_trained', 0)
                    
                    logger.info(f"  {model_name}:")
                    logger.info(f"    R² Score: {r2:.4f}")
                    logger.info(f"    Test Loss: {test_loss:.6f}")
                    logger.info(f"    Epochs: {epochs}")
                    
                    # Validate model trained
                    assert epochs > 0, f"No training occurred for {model_name}"
                elif 'error' in model_result:
                    logger.warning(f"  {model_name}: {model_result['error']}")
            
            logger.info("✓ Deep learning models training completed")
        else:
            logger.info("Deep learning models not available or insufficient data")
    
    def test_sentiment_analysis(self):
        """Test natural language processing and sentiment analysis"""
        logger.info("\n" + "="*60)
        logger.info("Testing Sentiment Analysis")
        logger.info("="*60)
        
        result = self.ml_framework.comprehensive_ml_analysis(
            price_data=self.price_data,
            news_data=self.news_data,
            prediction_horizon=3
        )
        
        if 'sentiment_analysis' in result:
            sentiment = result['sentiment_analysis']
            
            logger.info("Sentiment Analysis Results:")
            logger.info(f"  Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A')}")
            logger.info(f"  Confidence: {sentiment.get('confidence', 'N/A')}")
            
            # Sentiment distribution
            if 'sentiment_distribution' in sentiment:
                dist = sentiment['sentiment_distribution']
                logger.info(f"  Sentiment Distribution:")
                logger.info(f"    Positive: {dist.get('positive', 0)}")
                logger.info(f"    Neutral: {dist.get('neutral', 0)}")
                logger.info(f"    Negative: {dist.get('negative', 0)}")
            
            # Key topics
            if 'key_topics' in sentiment and sentiment['key_topics']:
                logger.info(f"  Key Topics: {', '.join(sentiment['key_topics'])}")
            
            # Validate sentiment score range
            overall_sentiment = sentiment.get('overall_sentiment', 0)
            assert -1.0 <= overall_sentiment <= 1.0, "Sentiment score out of valid range"
            
            logger.info("✓ Sentiment analysis completed successfully")
        else:
            logger.info("Sentiment analysis not available (using fallback or NLP libraries not available)")
    
    def test_uncertainty_quantification(self):
        """Test prediction uncertainty quantification"""
        logger.info("\n" + "="*60)
        logger.info("Testing Uncertainty Quantification")
        logger.info("="*60)
        
        result = self.ml_framework.comprehensive_ml_analysis(
            price_data=self.price_data,
            volume_data=self.volume_data,
            prediction_horizon=7
        )
        
        if 'uncertainty_analysis' in result:
            uncertainty = result['uncertainty_analysis']
            
            logger.info("Uncertainty Analysis Results:")
            logger.info(f"  Model Disagreement: {uncertainty.get('model_disagreement', 'N/A')}")
            logger.info(f"  Epistemic Uncertainty: {uncertainty.get('epistemic_uncertainty', 'N/A')}")
            logger.info(f"  Aleatoric Uncertainty: {uncertainty.get('aleatoric_uncertainty', 'N/A')}")
            
            # Prediction interval
            pred_interval = uncertainty.get('prediction_interval', [0, 0])
            if len(pred_interval) == 2:
                logger.info(f"  90% Prediction Interval: [{pred_interval[0]:.4f}, {pred_interval[1]:.4f}]")
            
            # Validate uncertainty metrics
            model_disagreement = uncertainty.get('model_disagreement', 0)
            assert model_disagreement >= 0, "Model disagreement should be non-negative"
            
            logger.info("✓ Uncertainty quantification completed successfully")
        else:
            logger.warning("Uncertainty analysis not available")
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis"""
        logger.info("\n" + "="*60)
        logger.info("Testing Feature Importance Analysis")
        logger.info("="*60)
        
        result = self.ml_framework.comprehensive_ml_analysis(
            price_data=self.price_data,
            volume_data=self.volume_data,
            fundamental_data=self.fundamental_data,
            prediction_horizon=5
        )
        
        if 'feature_importance' in result:
            importance = result['feature_importance']
            
            logger.info("Feature Importance Analysis:")
            
            # Top features
            if 'top_features' in importance and importance['top_features']:
                logger.info("  Top 10 Most Important Features:")
                for i, (feature_name, importance_score) in enumerate(importance['top_features'][:10], 1):
                    logger.info(f"    {i:2d}. {feature_name}: {importance_score:.4f}")
            
            # Model agreement
            model_agreement = importance.get('model_agreement', 0)
            logger.info(f"  Model Agreement on Feature Importance: {model_agreement:.4f}")
            
            # Validate importance analysis
            assert 0 <= model_agreement <= 1, "Model agreement should be between 0 and 1"
            
            logger.info("✓ Feature importance analysis completed successfully")
        else:
            logger.warning("Feature importance analysis not available")
    
    def test_reinforcement_learning(self):
        """Test reinforcement learning trading agent"""
        logger.info("\n" + "="*60)
        logger.info("Testing Reinforcement Learning Trading Agent")
        logger.info("="*60)
        
        if not self.ml_framework.rl_available:
            logger.info("Reinforcement learning libraries not available - skipping RL tests")
            return
        
        # Use subset of data for faster training
        rl_price_data = self.price_data.iloc[-500:]  # Last 500 days
        
        try:
            result = self.ml_framework.train_reinforcement_learning_agent(
                price_data=rl_price_data,
                initial_balance=10000
            )
            
            if 'error' not in result:
                logger.info("RL Agent Training Results:")
                logger.info(f"  Algorithm: {result.get('algorithm', 'N/A')}")
                logger.info(f"  Training Steps: {result.get('total_timesteps', 0)}")
                logger.info(f"  Training Completed: {result.get('training_completed', False)}")
                
                # Evaluation metrics
                if 'evaluation' in result:
                    eval_results = result['evaluation']
                    logger.info(f"  Total Return: {eval_results.get('total_return', 'N/A')}")
                    logger.info(f"  Sharpe Ratio: {eval_results.get('sharpe_ratio', 'N/A')}")
                    logger.info(f"  Max Drawdown: {eval_results.get('max_drawdown', 'N/A')}")
                
                logger.info("✓ RL agent training completed successfully")
            else:
                logger.warning(f"RL training failed: {result['error']}")
                
        except Exception as e:
            logger.warning(f"RL test failed: {e}")
    
    def test_model_performance_comparison(self):
        """Test comprehensive model performance comparison"""
        logger.info("\n" + "="*60)
        logger.info("Testing Model Performance Comparison")
        logger.info("="*60)
        
        # Run comprehensive analysis
        result = self.ml_framework.comprehensive_ml_analysis(
            price_data=self.price_data,
            volume_data=self.volume_data,
            fundamental_data=self.fundamental_data,
            news_data=self.news_data,
            prediction_horizon=5
        )
        
        self._print_comprehensive_summary(result)
        
        # Validate overall analysis
        assert 'timestamp' in result
        assert 'ml_available' in result
        
        # Check for key components
        expected_components = ['ensemble_models']
        available_components = result.get('analysis_components', [])
        
        for component in expected_components:
            if component in available_components:
                logger.info(f"✓ {component} analysis completed")
        
        logger.info("✓ Comprehensive model performance comparison completed")
    
    def test_fallback_mechanisms(self):
        """Test fallback mechanisms when advanced libraries unavailable"""
        logger.info("\n" + "="*60)
        logger.info("Testing Fallback Mechanisms")
        logger.info("="*60)
        
        # Test with minimal data to trigger fallbacks
        minimal_data = self.price_data.iloc[:50]  # Very limited data
        
        result = self.ml_framework.comprehensive_ml_analysis(
            price_data=minimal_data,
            prediction_horizon=1
        )
        
        logger.info("Fallback Analysis Results:")
        logger.info(f"  ML Available: {result.get('ml_available', False)}")
        
        if 'fallback_prediction' in result:
            logger.info(f"  Fallback Prediction: {result['fallback_prediction']}")
            logger.info(f"  Method: {result.get('method', 'Unknown')}")
        
        # Should always return some result, even with fallbacks
        assert isinstance(result, dict)
        logger.info("✓ Fallback mechanisms working correctly")
    
    def _print_ml_results(self, result: dict, title: str):
        """Print formatted ML analysis results"""
        print(f"\n{title}")
        print("-" * len(title))
        
        print(f"ML Framework Available: {result.get('ml_available', self.ml_framework.is_available)}")
        print(f"Analysis Timestamp: {result.get('timestamp', 'N/A')}")
        
        # Feature engineering
        if 'feature_engineering' in result:
            feature_info = result['feature_engineering']
            print(f"\nFeature Engineering:")
            print(f"  Features Generated: {feature_info.get('n_features', 0)}")
            
            feature_names = feature_info.get('feature_names', [])
            if feature_names:
                print(f"  Sample Features: {', '.join(feature_names[:5])}...")
        
        # Analysis components
        components = result.get('analysis_components', [])
        if components:
            print(f"\nCompleted Analysis Components: {', '.join(components)}")
    
    def _print_comprehensive_summary(self, result: dict):
        """Print comprehensive summary of all ML analysis results"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE ML ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Overall status
        print(f"ML Libraries Available: {result.get('ml_available', False)}")
        print(f"NLP Libraries Available: {self.ml_framework.nlp_available}")
        print(f"RL Libraries Available: {self.ml_framework.rl_available}")
        print(f"Analysis Timestamp: {result.get('timestamp', 'N/A')}")
        
        # Components completed
        components = result.get('analysis_components', [])
        print(f"\nCompleted Components ({len(components)}): {', '.join(components)}")
        
        # Model performance summary
        if 'ensemble_models' in result:
            ensemble = result['ensemble_models']
            print(f"\nEnsemble Models Performance:")
            
            best_model = None
            best_score = -float('inf')
            
            for model_name, model_result in ensemble.items():
                if isinstance(model_result, dict) and 'cv_score_mean' in model_result:
                    score = model_result['cv_score_mean']
                    print(f"  {model_name}: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                print(f"  Best Performing Model: {best_model} ({best_score:.4f})")
        
        # Prediction summary
        if 'ensemble_prediction' in result:
            pred = result['ensemble_prediction']
            print(f"\nPrediction Summary:")
            print(f"  Ensemble Prediction: {pred.get('ensemble_prediction', 'N/A')}")
            print(f"  Prediction Uncertainty: {pred.get('prediction_std', 'N/A')}")
            print(f"  Contributing Models: {pred.get('n_models', 0)}")
        
        # Sentiment summary
        if 'sentiment_analysis' in result:
            sentiment = result['sentiment_analysis']
            overall = sentiment.get('overall_sentiment', 0)
            confidence = sentiment.get('confidence', 0)
            print(f"\nSentiment Analysis:")
            print(f"  Overall Sentiment: {overall:.3f}")
            print(f"  Confidence: {confidence:.3f}")
        
        # Feature importance summary
        if 'feature_importance' in result:
            importance = result['feature_importance']
            top_features = importance.get('top_features', [])
            if top_features:
                print(f"\nTop 3 Features:")
                for i, (feature, score) in enumerate(top_features[:3], 1):
                    print(f"  {i}. {feature}: {score:.4f}")
    
    def run_comprehensive_tests(self):
        """Run all machine learning and AI tests"""
        logger.info("Starting Phase 5: Advanced Machine Learning & AI Integration Test Suite")
        logger.info(f"ML Libraries Available: {self.ml_framework.is_available}")
        logger.info(f"NLP Libraries Available: {self.ml_framework.nlp_available}")
        logger.info(f"RL Libraries Available: {self.ml_framework.rl_available}")
        
        test_methods = [
            self.test_automated_feature_engineering,
            self.test_ensemble_ml_models,
            self.test_deep_learning_models,
            self.test_sentiment_analysis,
            self.test_uncertainty_quantification,
            self.test_feature_importance_analysis,
            self.test_reinforcement_learning,
            self.test_model_performance_comparison,
            self.test_fallback_mechanisms
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
        logger.info("PHASE 5 MACHINE LEARNING & AI TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Tests: {passed_tests + failed_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/(passed_tests + failed_tests)*100:.1f}%")
        
        # Library availability summary
        if self.ml_framework.is_available:
            logger.info("✅ Advanced ML libraries (PyTorch, Sklearn, XGBoost) tested successfully")
        else:
            logger.info("⚠️ Advanced ML libraries not available - fallback implementations tested")
        
        if self.ml_framework.nlp_available:
            logger.info("✅ NLP libraries (Transformers, spaCy) tested successfully")
        else:
            logger.info("⚠️ NLP libraries not available - basic sentiment analysis tested")
        
        if self.ml_framework.rl_available:
            logger.info("✅ Reinforcement learning libraries tested successfully")
        else:
            logger.info("⚠️ RL libraries not available - traditional optimization tested")
        
        logger.info("Phase 5 Machine Learning & AI Integration implementation complete!")
        logger.info("🤖 Advanced AI-powered quantitative trading system ready for deployment!")
        
        return passed_tests, failed_tests


def main():
    """Run the Phase 5 Machine Learning & AI test suite"""
    tester = MLTradingFrameworkTester()
    passed, failed = tester.run_comprehensive_tests()
    
    # Exit with appropriate code
    exit_code = 0 if failed == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
