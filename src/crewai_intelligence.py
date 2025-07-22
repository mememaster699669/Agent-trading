"""
CrewAI Intelligence Layer with LiteLLM Integration
Probabilistic decision-making and strategic reasoning
Based on GCP Agent LLM configuration pattern
Integrates advanced quantitative models for sophisticated market analysis
"""

import os
import re
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import litellm

# Import our enhanced logging system
from .logging_system import get_logger
from .config import ConfigManager
from .environment import MODEL_GPT_4O, is_production

# Import quantitative models for advanced analysis
from .quant_models import (
    QuantitativeModels, 
    RiskMetrics, 
    ProbabilisticForecast,
    MarketMicrostructure,
    AdvancedPhysicsModels
)

# Initialize logger for this module
logger = get_logger("CrewAIIntelligence")

class LiteLLMConfig:
    """
    LiteLLM configuration based on GCP agent pattern
    Supports custom base URLs and completion endpoints
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.setup_litellm()
        logger.info("LiteLLM configuration initialized")
    
    def setup_litellm(self):
        """Setup litellm with custom configuration"""
        
        # Get model configuration from environment
        model_name = os.getenv("MODEL_GPT_4O", "gpt-4")
        base_url = os.getenv("LLM_BASE_URL", None)
        completion_url = os.getenv("LLM_COMPLETION_URL", None)
        api_key = os.getenv("OPENAI_API_KEY", None)
        
        logger.log_action(
            "llm_config_setup",
            {
                "model_name": model_name,
                "base_url": base_url,
                "completion_url": completion_url,
                "has_api_key": bool(api_key),
                "description": "Setting up LiteLLM configuration"
            },
            status="started"
        )
        
        # Configure litellm settings
        if base_url:
            litellm.api_base = base_url
            logger.info(f"LiteLLM base URL set to: {base_url}")
        
        if completion_url:
            # For custom completion endpoints
            litellm.completion_url = completion_url
            logger.info(f"LiteLLM completion URL set to: {completion_url}")
        
        if api_key:
            litellm.api_key = api_key
            logger.info("LiteLLM API key configured")
        
        # Set model for CrewAI
        self.model_name = model_name
        
        # Configure additional litellm settings
        litellm.set_verbose = not is_production()  # Verbose in development
        
        logger.log_action(
            "llm_config_setup",
            {
                "model_name": model_name,
                "base_url": base_url,
                "completion_url": completion_url,
                "description": "LiteLLM configuration completed"
            },
            status="completed"
        )
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for CrewAI agents"""
        return {
            "model": self.model_name,
            "base_url": litellm.api_base,
            "api_key": litellm.api_key
        }

class QuantitativeAnalysisAgent:
    """
    Advanced Quantitative Market Analysis Agent
    
    Personality: Sophisticated quant researcher with deep expertise in:
    - Bayesian inference and probabilistic modeling
    - Market regime detection using Hidden Markov Models
    - Statistical arbitrage and mean reversion strategies
    - Risk-adjusted return optimization
    
    This agent embodies the philosophy of probabilistic trading over prediction,
    focusing on uncertainty quantification and statistical significance.
    """
    
    def __init__(self, llm_config: LiteLLMConfig):
        self.llm_config = llm_config
        self.agent = self._create_agent()
        self.quant_models = QuantitativeModels()
        self.risk_metrics = RiskMetrics()
        logger.info("Quantitative Analysis Agent initialized with advanced mathematical models")
    
    def _create_agent(self) -> Agent:
        """Create quantitative analysis agent with specialized personality"""
        
        model_config = self.llm_config.get_model_config()
        
        return Agent(
            role='Senior Quantitative Researcher',
            goal="""Develop probabilistic models for BTC market dynamics, focusing on uncertainty 
            quantification rather than point predictions. Generate statistically rigorous trading 
            signals with confidence intervals and risk assessments.""",
            backstory="""You are a PhD-level quantitative researcher with 15+ years of experience 
            in statistical finance. You specialize in Bayesian methods, regime detection, and 
            probabilistic modeling. You believe that markets are inherently uncertain and that 
            successful trading comes from modeling this uncertainty rather than making precise 
            predictions.
            
            Your expertise includes:
            - Bayesian state-space models for price dynamics
            - Hidden Markov Models for regime detection  
            - Kelly Criterion for optimal position sizing
            - Modern portfolio theory with transaction costs
            - Statistical arbitrage and mean reversion
            - Value-at-Risk and tail risk modeling
            
            You always provide confidence intervals, statistical significance tests, and 
            uncertainty bounds with your analysis. You prefer mathematically grounded 
            approaches over heuristic technical indicators.""",
            verbose=True,
            allow_delegation=False,
            llm=model_config["model"]
        )
    
    def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced probabilistic market analysis using quantitative models
        
        Combines AI reasoning with rigorous mathematical models:
        1. Bayesian price forecasting with uncertainty quantification
        2. Regime detection using HMM
        3. Mean reversion signal analysis
        4. Statistical significance testing
        """
        
        start_time = time.time()
        
        action_id = logger.log_action(
            "quantitative_analysis",
            {
                "data_points": len(market_data.get('candles', [])),
                "symbol": market_data.get('symbol', 'BTC'),
                "timeframe": market_data.get('timeframe', '15m'),
                "models_used": ["Bayesian", "HMM", "MeanReversion"],
                "description": "Advanced quantitative analysis with probabilistic models"
            },
            status="started"
        )
        
        try:
            # Extract price data
            candles = market_data.get('candles', [])
            if len(candles) < 50:  # Need minimum data for statistical analysis
                logger.warning("Insufficient data for robust quantitative analysis")
                return self._fallback_analysis(market_data)
            
            # Convert to numpy arrays for quantitative analysis
            prices = np.array([float(candle.get('close', 0)) for candle in candles])
            volumes = np.array([float(candle.get('volume', 0)) for candle in candles])
            returns = np.diff(np.log(prices))
            
            # 1. Bayesian Probabilistic Forecast
            logger.log_action("quantitative_analysis", {"step": "bayesian_forecast"}, status="progress")
            
            probabilistic_forecast = self.quant_models.bayesian_price_model(
                prices=prices,
                lookback=min(252, len(prices)),
                confidence_levels=[0.68, 0.95, 0.99]
            )
            
            # 2. Market Regime Detection
            logger.log_action("quantitative_analysis", {"step": "regime_detection"}, status="progress")
            
            regime_probs, regime_stats = self.quant_models.regime_detection_hmm(
                returns=returns,
                n_regimes=3
            )
            
            # 3. Mean Reversion Analysis
            logger.log_action("quantitative_analysis", {"step": "mean_reversion"}, status="progress")
            
            mean_reversion = self.quant_models.mean_reversion_signal(
                prices=prices,
                lookback=20,
                z_threshold=2.0
            )
            
            # 4. Risk Metrics Calculation
            logger.log_action("quantitative_analysis", {"step": "risk_metrics"}, status="progress")
            
            var_95 = self.risk_metrics.value_at_risk(returns, confidence=0.05, method='historical')
            cvar_95 = self.risk_metrics.conditional_value_at_risk(returns, confidence=0.05)
            max_dd = self.risk_metrics.maximum_drawdown(prices)
            
            # 5. Market Microstructure Analysis
            estimated_spread = MarketMicrostructure.estimate_bid_ask_spread(prices, volumes)
            
            # Prepare quantitative summary for AI agent
            quant_summary = {
                "probabilistic_forecast": {
                    "expected_price": probabilistic_forecast.mean,
                    "price_std": probabilistic_forecast.std,
                    "probability_up": probabilistic_forecast.probability_up,
                    "probability_down": probabilistic_forecast.probability_down,
                    "confidence_intervals": probabilistic_forecast.confidence_intervals,
                    "expected_return": probabilistic_forecast.expected_return,
                    "upside_potential": probabilistic_forecast.upside_potential,
                    "downside_risk": probabilistic_forecast.downside_risk
                },
                "regime_analysis": {
                    "current_regime_probabilities": regime_probs.tolist(),
                    "regime_characteristics": regime_stats,
                    "dominant_regime": int(np.argmax(regime_probs))
                },
                "mean_reversion": {
                    "signal_strength": mean_reversion['signal'],
                    "z_score": mean_reversion['z_score'], 
                    "confidence": mean_reversion['confidence'],
                    "p_value": mean_reversion['p_value']
                },
                "risk_metrics": {
                    "var_95": var_95,
                    "cvar_95": cvar_95,
                    "max_drawdown": max_dd['max_drawdown'],
                    "current_drawdown": max_dd['current_drawdown']
                },
                "market_structure": {
                    "estimated_spread": estimated_spread,
                    "recent_volatility": np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0
                }
            }
            
            # 6. AI Agent Analysis with Quantitative Context
            logger.log_action("quantitative_analysis", {"step": "ai_synthesis"}, status="progress")
            
            task = Task(
                description=f"""
                As a senior quantitative researcher, analyze the following BTC market data using 
                your expertise in probabilistic modeling and statistical finance:
                
                MARKET DATA SUMMARY:
                - Symbol: {market_data.get('symbol', 'BTC')}
                - Current Price: ${prices[-1]:,.2f}
                - Data Points: {len(prices)} candles
                - Timeframe: {market_data.get('timeframe', '15m')}
                
                QUANTITATIVE ANALYSIS RESULTS:
                
                1. BAYESIAN PROBABILISTIC FORECAST:
                - Expected Price: ${probabilistic_forecast.mean:,.2f}
                - Price Uncertainty (1σ): ±${probabilistic_forecast.std:,.2f}
                - Probability of Upward Move: {probabilistic_forecast.probability_up:.1%}
                - Probability of Downward Move: {probabilistic_forecast.probability_down:.1%}
                - 95% Confidence Interval: ${probabilistic_forecast.get_confidence_interval(0.95)[0]:,.2f} - ${probabilistic_forecast.get_confidence_interval(0.95)[1]:,.2f}
                - Expected Return: {probabilistic_forecast.expected_return:.4f}
                - Upside Potential: {probabilistic_forecast.upside_potential:.4f}
                - Downside Risk: {probabilistic_forecast.downside_risk:.4f}
                
                2. MARKET REGIME ANALYSIS (HMM):
                - Current Regime Probabilities: {[f'{p:.1%}' for p in regime_probs]}
                - Dominant Regime: {['Low Vol', 'Medium Vol', 'High Vol'][int(np.argmax(regime_probs))]}
                - Regime Characteristics: {regime_stats}
                
                3. MEAN REVERSION ANALYSIS:
                - Signal Strength: {mean_reversion['signal']:.3f} (-1 to +1 scale)
                - Z-Score: {mean_reversion['z_score']:.2f}
                - Statistical Confidence: {mean_reversion['confidence']:.1%}
                - P-Value: {mean_reversion['p_value']:.4f}
                
                4. RISK METRICS:
                - 95% Value-at-Risk: {var_95:.4f}
                - 95% Conditional VaR: {cvar_95:.4f}
                - Maximum Drawdown: {max_dd['max_drawdown']:.1%}
                - Current Drawdown: {max_dd['current_drawdown']:.1%}
                
                5. MARKET MICROSTRUCTURE:
                - Estimated Bid-Ask Spread: {estimated_spread:.6f}
                - Recent Volatility (Annualized): {np.std(returns[-20:]) * np.sqrt(252):.1%}
                
                PROVIDE YOUR EXPERT ANALYSIS INCLUDING:
                
                1. PROBABILISTIC ASSESSMENT:
                   - Interpretation of the Bayesian forecast and uncertainty bounds
                   - Assessment of directional probabilities and their reliability
                   - Commentary on the confidence intervals and what they imply
                
                2. REGIME-BASED STRATEGY:
                   - Current market regime interpretation
                   - How regime probabilities should influence trading approach
                   - Regime transition risks and opportunities
                
                3. STATISTICAL ARBITRAGE OPPORTUNITIES:
                   - Mean reversion signal interpretation and strength
                   - Statistical significance of current price deviations
                   - Expected holding period and profit potential
                
                4. RISK ASSESSMENT:
                   - VaR and CVaR interpretation for position sizing
                   - Drawdown analysis and portfolio protection
                   - Tail risk considerations
                
                5. EXECUTION RECOMMENDATIONS:
                   - Optimal position sizing using Kelly Criterion principles
                   - Entry/exit timing based on statistical models
                   - Risk management parameters (stop-loss, take-profit)
                
                FORMAT YOUR RESPONSE AS:
                - Overall Market Assessment (Bull/Bear/Neutral with confidence)
                - Statistical Confidence Score (0-100%)
                - Recommended Action (Strong Buy/Buy/Hold/Sell/Strong Sell)
                - Position Size Recommendation (% of portfolio)
                - Risk Management Parameters
                - Key Uncertainties and Model Limitations
                
                Remember: Focus on probabilistic thinking, not point predictions. 
                Quantify uncertainty and provide statistical reasoning for all recommendations.
                """,
                agent=self.agent,
                expected_output="Comprehensive quantitative analysis with probabilistic recommendations"
            )
            
            # Execute AI analysis
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            ai_analysis = crew.kickoff()
            
            processing_time = time.time() - start_time
            
            # Extract structured insights from AI analysis
            confidence_score = self._extract_confidence_score(str(ai_analysis))
            market_assessment = self._extract_market_assessment(str(ai_analysis))
            position_size = self._extract_position_size(str(ai_analysis))
            
            # Combine quantitative models with AI insights
            final_analysis = {
                "ai_analysis": str(ai_analysis),
                "quantitative_models": quant_summary,
                "confidence_score": confidence_score,
                "market_assessment": market_assessment,
                "position_size_recommendation": position_size,
                "statistical_significance": mean_reversion['confidence'],
                "regime_based_confidence": float(np.max(regime_probs)),
                "processing_time": processing_time,
                "timestamp": time.time(),
                "model_version": "bayesian_hmm_v1.0"
            }
            
            # Log comprehensive results
            logger.log_llm_interaction(
                prompt_type="quantitative_analysis",
                model=self.llm_config.model_name,
                prompt_length=len(task.description),
                response_length=len(str(ai_analysis)),
                processing_time=processing_time
            )
            
            logger.log_action(
                "quantitative_analysis",
                {
                    "confidence_score": confidence_score,
                    "market_assessment": market_assessment,
                    "statistical_significance": mean_reversion['confidence'],
                    "regime_confidence": float(np.max(regime_probs)),
                    "models_used": ["Bayesian", "HMM", "MeanReversion", "VaR"],
                    "description": "Quantitative analysis completed with AI synthesis"
                },
                status="completed"
            )
            
            return final_analysis
            
        except Exception as e:
            logger.log_error(
                "quantitative_analysis_error",
                str(e),
                {
                    "market_data_keys": list(market_data.keys()),
                    "data_points": len(market_data.get('candles', []))
                }
            )
            return self._fallback_analysis(market_data)
    
    def _fallback_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when insufficient data or errors occur"""
        return {
            "ai_analysis": "Insufficient data for robust quantitative analysis",
            "confidence_score": 0.3,
            "market_assessment": "neutral",
            "position_size_recommendation": 0.01,  # Very conservative
            "statistical_significance": 0.0,
            "regime_based_confidence": 0.33,  # Equal probability across regimes
            "processing_time": 0.1,
            "timestamp": time.time(),
            "model_version": "fallback_v1.0"
        }
    
    def _extract_confidence_score(self, analysis_text: str) -> float:
        """Extract confidence score from AI analysis"""
        import re
        
        # Look for confidence percentages
        confidence_matches = re.findall(r'confidence[:\s]*(\d+(?:\.\d+)?)\s*%', analysis_text.lower())
        if confidence_matches:
            return float(confidence_matches[0]) / 100.0
        
        # Look for statistical confidence
        stat_confidence_matches = re.findall(r'statistical confidence[:\s]*(\d+(?:\.\d+)?)', analysis_text.lower())
        if stat_confidence_matches:
            return float(stat_confidence_matches[0]) / 100.0 if float(stat_confidence_matches[0]) > 1 else float(stat_confidence_matches[0])
        
        # Default based on content quality
        if 'high confidence' in analysis_text.lower():
            return 0.8
        elif 'medium confidence' in analysis_text.lower():
            return 0.6
        elif 'low confidence' in analysis_text.lower():
            return 0.4
        
        return 0.7  # Default moderate confidence
    
    def _extract_market_assessment(self, analysis_text: str) -> str:
        """Extract market assessment from AI analysis"""
        text_lower = analysis_text.lower()
        
        if 'strong bull' in text_lower or 'strongly bullish' in text_lower:
            return 'strong_bullish'
        elif 'bull' in text_lower or 'bullish' in text_lower:
            return 'bullish'
        elif 'strong bear' in text_lower or 'strongly bearish' in text_lower:
            return 'strong_bearish'
        elif 'bear' in text_lower or 'bearish' in text_lower:
            return 'bearish'
        else:
            return 'neutral'
    
    def _extract_position_size(self, analysis_text: str) -> float:
        """Extract position size recommendation from AI analysis"""
        import re
        
        # Look for percentage recommendations
        percentage_matches = re.findall(r'position size[:\s]*(\d+(?:\.\d+)?)\s*%', analysis_text.lower())
        if percentage_matches:
            return float(percentage_matches[0]) / 100.0
        
        # Look for portfolio fraction
        fraction_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*of\s*portfolio', analysis_text.lower())
        if fraction_matches:
            return float(fraction_matches[0]) / 100.0
        
        # Conservative default
        return 0.02  # 2%

class AdvancedRiskManagementAgent:
    """
    Advanced Risk Management Agent with Quantitative Risk Models
    
    Personality: Conservative risk manager with expertise in:
    - Value-at-Risk (VaR) and Conditional VaR modeling
    - Portfolio optimization using Modern Portfolio Theory
    - Kelly Criterion for optimal position sizing
    - Tail risk assessment and stress testing
    - Dynamic hedging and correlation analysis
    
    This agent embodies prudent risk management philosophy with mathematical rigor,
    focusing on capital preservation and risk-adjusted returns.
    """
    
    def __init__(self, llm_config: LiteLLMConfig):
        self.llm_config = llm_config
        self.agent = self._create_agent()
        self.quant_models = QuantitativeModels()
        self.risk_metrics = RiskMetrics()
        self.physics_models = AdvancedPhysicsModels()
        logger.info("Advanced Risk Management Agent initialized with quantitative risk models and physics-based analysis")
    
    def _create_agent(self) -> Agent:
        """Create advanced risk management agent with specialized expertise"""
        
        model_config = self.llm_config.get_model_config()
        
        return Agent(
            role='Chief Risk Officer & Portfolio Manager',
            goal="""Provide comprehensive risk assessment and portfolio optimization recommendations 
            using advanced quantitative risk models. Ensure all trading decisions align with 
            risk-adjusted return objectives and portfolio constraints.""",
            backstory="""You are a seasoned Chief Risk Officer with an MBA in Finance and CFA 
            certification, with 20+ years of experience managing risk for institutional trading 
            operations. You have deep expertise in quantitative risk management and have survived 
            multiple market crises including the 2008 financial crisis and COVID-19 market volatility.
            
            Your core competencies include:
            - Value-at-Risk (VaR) and Expected Shortfall (CVaR) modeling
            - Portfolio optimization using Markowitz mean-variance framework
            - Kelly Criterion and optimal capital allocation
            - Stress testing and scenario analysis
            - Correlation analysis and regime-dependent risk modeling
            - Regulatory risk management (Basel III, etc.)
            - Behavioral finance and risk psychology
            
            Your philosophy is "capital preservation first, returns second." You believe that 
            superior risk management, not market timing, is the key to long-term trading success. 
            You always consider tail risks, model uncertainty, and the potential for regime changes.
            
            You are known for asking tough questions like:
            - "What's the worst-case scenario?"
            - "How much can we afford to lose?"
            - "What if our models are wrong?"
            - "How correlated are our positions?"
            
            You provide detailed mathematical justification for all risk recommendations.""",
            verbose=True,
            allow_delegation=False,
            llm=model_config["model"]
        )
    
    def assess_risk(self, signal_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive risk assessment using advanced quantitative models
        
        Combines AI reasoning with sophisticated risk metrics:
        1. VaR and CVaR calculation across multiple methodologies
        2. Kelly Criterion position sizing with risk constraints
        3. Portfolio optimization with correlation analysis
        4. Stress testing and scenario analysis
        5. Risk-adjusted return projections
        """
        
        start_time = time.time()
        
        action_id = logger.log_action(
            "advanced_risk_assessment",
            {
                "signal_type": signal_data.get('signal_type', 'unknown'),
                "signal_confidence": signal_data.get('confidence_score', 0),
                "portfolio_value": portfolio_data.get('total_value', 0),
                "risk_models": ["VaR", "CVaR", "Kelly", "Markowitz"],
                "description": "Advanced quantitative risk assessment"
            },
            status="started"
        )
        
        try:
            # Extract market data for risk calculations
            market_analysis = signal_data.get('quantitative_models', {})
            probabilistic_forecast = market_analysis.get('probabilistic_forecast', {})
            
            # Simulate historical returns for risk calculations
            # In practice, this would come from your actual historical data
            expected_return = probabilistic_forecast.get('expected_return', 0.001)
            downside_risk = probabilistic_forecast.get('downside_risk', 0.02)
            upside_potential = probabilistic_forecast.get('upside_potential', 0.02)
            
            # Generate synthetic return distribution for risk modeling
            # (In production, use actual historical returns)
            n_simulations = 1000
            returns_mean = expected_return
            returns_std = (downside_risk + upside_potential) / 2  # Approximate volatility
            simulated_returns = np.random.normal(returns_mean, returns_std, n_simulations)
            
            # 1. Advanced VaR Analysis
            logger.log_action("advanced_risk_assessment", {"step": "var_analysis"}, status="progress")
            
            var_historical = self.risk_metrics.value_at_risk(simulated_returns, 0.05, 'historical')
            var_parametric = self.risk_metrics.value_at_risk(simulated_returns, 0.05, 'parametric')
            var_monte_carlo = self.risk_metrics.value_at_risk(simulated_returns, 0.05, 'monte_carlo')
            
            cvar_95 = self.risk_metrics.conditional_value_at_risk(simulated_returns, 0.05)
            
            # 2. Kelly Criterion Position Sizing
            logger.log_action("advanced_risk_assessment", {"step": "kelly_sizing"}, status="progress")
            
            variance = returns_std ** 2
            kelly_fraction = self.quant_models.kelly_position_sizing(
                expected_return=expected_return,
                variance=variance,
                max_position=0.25  # Maximum 25% position
            )
            
            # 3. Portfolio Risk Metrics
            current_portfolio_value = portfolio_data.get('total_value', 100000)
            available_cash = portfolio_data.get('available_cash', 50000)
            current_exposure = portfolio_data.get('crypto_exposure', 0.3)
            
            # Calculate position size recommendations
            kelly_position_value = abs(kelly_fraction) * current_portfolio_value
            conservative_position_value = kelly_position_value * 0.5  # 50% of Kelly for safety
            
            # 4. Risk-Adjusted Metrics
            sharpe_ratio = expected_return / returns_std if returns_std > 0 else 0
            sortino_ratio = expected_return / downside_risk if downside_risk > 0 else 0
            
            # 4.5. PHYSICS-BASED RISK ANALYSIS (@khemkapital methodology)
            logger.log_action("advanced_risk_assessment", {"step": "physics_risk_analysis"}, status="progress")
            
            # Extract price data for physics analysis
            # In practice, this would be actual market data
            market_data = signal_data.get('market_data', {})
            price_data = market_data.get('price_history', [])
            volume_data = market_data.get('volume_history', None)
            
            # If no actual data, simulate realistic price series for demonstration
            if not price_data or len(price_data) < 50:
                # Generate realistic price simulation
                base_price = market_data.get('current_price', 43000)
                n_periods = 100
                dt = 1/365  # Daily periods
                drift = expected_return * dt
                volatility = returns_std * np.sqrt(dt)
                
                price_simulation = [base_price]
                for _ in range(n_periods - 1):
                    random_shock = np.random.normal(0, 1)
                    price_change = drift + volatility * random_shock
                    new_price = price_simulation[-1] * (1 + price_change)
                    price_simulation.append(new_price)
                
                price_data = np.array(price_simulation)
            else:
                price_data = np.array(price_data)
            
            # Calculate physics-based risk metrics
            try:
                # Information Entropy Analysis
                entropy_analysis = self.physics_models.information_entropy_risk(
                    price_data, volume_data
                )
                
                # Hurst Exponent (Fractal Memory)
                memory_analysis = self.physics_models.hurst_exponent_memory(price_data)
                
                # Lyapunov Exponent (Instability Detection)
                instability_analysis = self.physics_models.lyapunov_instability_detection(price_data)
                
                # Regime Transition Detection
                regime_analysis = self.physics_models.regime_transition_detection(
                    price_data, volume_data
                )
                
                # Combine physics metrics into risk score
                entropy_risk_weight = entropy_analysis['entropy']  # 0-1 scale
                memory_risk_weight = abs(memory_analysis['hurst_exponent'] - 0.5) * 2  # 0-1 scale
                instability_risk_weight = min(instability_analysis['instability_score'] * 5, 1.0)  # Scale to 0-1
                
                physics_risk_score = (entropy_risk_weight + memory_risk_weight + instability_risk_weight) / 3.0
                
                # Risk amplification factor based on physics
                risk_amplification = 1.0 + physics_risk_score * 0.5  # 1.0 to 1.5x multiplier
                
                physics_metrics = {
                    'entropy_analysis': entropy_analysis,
                    'memory_analysis': memory_analysis,  
                    'instability_analysis': instability_analysis,
                    'regime_analysis': regime_analysis,
                    'combined_physics_risk_score': physics_risk_score,
                    'risk_amplification_factor': risk_amplification
                }
                
                logger.log_action(
                    "physics_risk_analysis", 
                    {
                        "entropy_risk": entropy_analysis['risk_level'],
                        "memory_type": memory_analysis['memory_type'],
                        "instability_level": instability_analysis['instability_level'],
                        "regime": regime_analysis['regime'],
                        "physics_risk_score": physics_risk_score
                    }, 
                    status="completed"
                )
                
            except Exception as e:
                logger.log_error("physics_risk_analysis_error", str(e))
                # Fallback to conservative physics risk
                physics_metrics = {
                    'entropy_analysis': {'risk_level': 'medium', 'entropy': 0.5},
                    'memory_analysis': {'memory_type': 'random_walk', 'hurst_exponent': 0.5},
                    'instability_analysis': {'instability_level': 'moderate', 'instability_score': 0.1},
                    'regime_analysis': {'regime': 'transitional', 'stability': 'moderate'},
                    'combined_physics_risk_score': 0.5,
                    'risk_amplification_factor': 1.25
                }
            
            # 5. Stress Testing
            logger.log_action("advanced_risk_assessment", {"step": "stress_testing"}, status="progress")
            
            # Scenario analysis
            var_95 = var_historical
            scenarios = {
                "base_case": {"return": expected_return, "probability": 0.6},
                "stress_case": {"return": var_95, "probability": 0.05},
                "extreme_case": {"return": cvar_95, "probability": 0.01},
                "bull_case": {"return": upside_potential, "probability": 0.34}
            }
            
            # Calculate scenario-weighted expected return
            scenario_weighted_return = sum(
                scenario["return"] * scenario["probability"] 
                for scenario in scenarios.values()
            )
            
            # Prepare comprehensive risk summary for AI agent
            risk_summary = {
                "var_analysis": {
                    "var_95_historical": var_historical,
                    "var_95_parametric": var_parametric,
                    "var_95_monte_carlo": var_monte_carlo,
                    "cvar_95": cvar_95,
                    "var_confidence_interval": (var_parametric * 0.8, var_parametric * 1.2)
                },
                "position_sizing": {
                    "kelly_fraction": kelly_fraction,
                    "kelly_position_value": kelly_position_value,
                    "conservative_position_value": conservative_position_value,
                    "max_position_pct": min(abs(kelly_fraction), 0.1) * 100  # Cap at 10%
                },
                "portfolio_metrics": {
                    "current_exposure": current_exposure,
                    "available_cash": available_cash,
                    "proposed_exposure": (conservative_position_value / current_portfolio_value),
                    "diversification_ratio": 1.0 - current_exposure  # Simplified
                },
                "risk_adjusted_returns": {
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                    "expected_return": expected_return,
                    "volatility": returns_std,
                    "scenario_weighted_return": scenario_weighted_return
                },
                "physics_based_risk": physics_metrics,
                "stress_scenarios": scenarios,
                "risk_limits": {
                    "max_single_position": 0.1,  # 10%
                    "max_sector_exposure": 0.3,   # 30%
                    "var_limit": 0.05,           # 5% daily VaR limit
                    "drawdown_limit": 0.15       # 15% max drawdown
                }
            }
            
            # 6. AI Agent Risk Assessment
            logger.log_action("advanced_risk_assessment", {"step": "ai_risk_synthesis"}, status="progress")
            
            task = Task(
                description=f"""
                As Chief Risk Officer, conduct a comprehensive risk assessment for the proposed 
                BTC trading position using your expertise in quantitative risk management:
                
                SIGNAL INFORMATION:
                - Signal Type: {signal_data.get('market_assessment', 'unknown')}
                - Signal Confidence: {signal_data.get('confidence_score', 0):.1%}
                - Expected Return: {expected_return:.4f}
                - Statistical Significance: {signal_data.get('statistical_significance', 0):.1%}
                
                PORTFOLIO CONTEXT:
                - Total Portfolio Value: ${current_portfolio_value:,.2f}
                - Available Cash: ${available_cash:,.2f}
                - Current Crypto Exposure: {current_exposure:.1%}
                - Cash Utilization: {(current_portfolio_value - available_cash) / current_portfolio_value:.1%}
                
                QUANTITATIVE RISK ANALYSIS:
                
                1. VALUE-AT-RISK ANALYSIS:
                - 95% VaR (Historical): {var_historical:.4f}
                - 95% VaR (Parametric): {var_parametric:.4f}
                - 95% VaR (Monte Carlo): {var_monte_carlo:.4f}
                - 95% Conditional VaR: {cvar_95:.4f}
                - VaR Confidence Range: {var_parametric * 0.8:.4f} to {var_parametric * 1.2:.4f}
                
                2. POSITION SIZING ANALYSIS:
                - Kelly Criterion Fraction: {kelly_fraction:.4f}
                - Kelly Position Value: ${kelly_position_value:,.2f}
                - Conservative Position (50% Kelly): ${conservative_position_value:,.2f}
                - Recommended Max Position: {min(abs(kelly_fraction), 0.1) * 100:.1f}% of portfolio
                
                3. RISK-ADJUSTED METRICS:
                - Expected Sharpe Ratio: {sharpe_ratio:.2f}
                - Expected Sortino Ratio: {sortino_ratio:.2f}
                - Return/Risk Ratio: {expected_return / returns_std if returns_std > 0 else 0:.2f}
                - Volatility: {returns_std:.1%}
                
                4. STRESS TEST SCENARIOS:
                - Base Case (60% prob): {scenarios['base_case']['return']:.4f}
                - Stress Case (5% prob): {scenarios['stress_case']['return']:.4f}
                - Extreme Case (1% prob): {scenarios['extreme_case']['return']:.4f}
                - Bull Case (34% prob): {scenarios['bull_case']['return']:.4f}
                - Scenario-Weighted Return: {scenario_weighted_return:.4f}
                
                5. PHYSICS-BASED RISK ANALYSIS (@khemkapital methodology):
                - Information Entropy Risk: {physics_metrics['entropy_analysis']['risk_level']} (Score: {physics_metrics['entropy_analysis']['entropy']:.3f})
                - Market Readability: {physics_metrics['entropy_analysis'].get('readability', 'unknown')}
                - Fractal Memory (Hurst): {physics_metrics['memory_analysis']['memory_type']} (H = {physics_metrics['memory_analysis']['hurst_exponent']:.3f})
                - Traumatic Events Detected: {physics_metrics['memory_analysis'].get('trauma_detected', False)}
                - System Instability: {physics_metrics['instability_analysis']['instability_level']} (Score: {physics_metrics['instability_analysis']['instability_score']:.3f})
                - Market Regime: {physics_metrics['regime_analysis']['regime']} (Stability: {physics_metrics['regime_analysis']['stability']})
                - Combined Physics Risk Score: {physics_metrics['combined_physics_risk_score']:.3f}/1.0
                - Risk Amplification Factor: {physics_metrics['risk_amplification_factor']:.2f}x
                
                6. RISK LIMITS COMPLIANCE:
                - Max Single Position Limit: 10% (Current proposal: {conservative_position_value / current_portfolio_value:.1%})
                - VaR Limit: 5% daily (Current estimate: {abs(var_parametric):.1%})
                - Sector Exposure Limit: 30% (Current + Proposed: {current_exposure + (conservative_position_value / current_portfolio_value):.1%})
                
                PROVIDE YOUR EXPERT RISK ASSESSMENT INCLUDING:
                
                1. RISK APPROVAL DECISION:
                   - Overall risk rating (Low/Medium/High/Extreme)
                   - Approve/Reject/Conditional approval
                   - Key risk factors and concerns
                
                2. POSITION SIZE RECOMMENDATION:
                   - Optimal position size ($ amount and % of portfolio)
                   - Risk-adjusted position sizing rationale
                   - Maximum acceptable position given current portfolio
                
                3. RISK MANAGEMENT PARAMETERS:
                   - Stop-loss level (based on VaR analysis)
                   - Take-profit targets (risk-reward optimization)
                   - Position monitoring triggers
                   - Portfolio rebalancing requirements
                
                4. SCENARIO PLANNING:
                   - Best-case scenario profit/loss
                   - Worst-case scenario profit/loss
                   - Most likely outcome range
                   - Exit strategy for each scenario
                
                5. RISK MONITORING REQUIREMENTS:
                   - Key risk metrics to monitor
                   - Warning indicators for position adjustment
                   - Portfolio heat map and concentration risks
                   - Correlation monitoring with existing positions
                
                6. REGULATORY AND COMPLIANCE:
                   - Risk limit compliance status
                   - Documentation requirements
                   - Escalation procedures if limits exceeded
                
                FORMAT YOUR RESPONSE WITH:
                - RISK DECISION: [APPROVED/REJECTED/CONDITIONAL]
                - RISK RATING: [LOW/MEDIUM/HIGH/EXTREME]
                - POSITION SIZE: $X (Y% of portfolio)
                - STOP LOSS: $X (Z% below entry)
                - TAKE PROFIT: $X (Z% above entry)
                - MAX LOSS: $X (absolute dollar amount)
                - MONITORING: [Key metrics to watch]
                
                Remember: Your primary responsibility is capital preservation. Be conservative 
                with position sizing and always consider tail risks and model uncertainty.
                """,
                agent=self.agent,
                expected_output="Comprehensive risk assessment with quantitative backing and clear recommendations"
            )
            
            # Execute AI risk assessment
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            ai_risk_assessment = crew.kickoff()
            
            processing_time = time.time() - start_time
            
            # Extract structured recommendations from AI assessment
            risk_decision = self._extract_risk_decision(str(ai_risk_assessment))
            risk_rating = self._extract_risk_rating(str(ai_risk_assessment))
            recommended_position_size = self._extract_recommended_position_size(str(ai_risk_assessment))
            stop_loss_level = self._extract_stop_loss(str(ai_risk_assessment))
            take_profit_level = self._extract_take_profit(str(ai_risk_assessment))
            
            # Final risk assessment result
            final_risk_assessment = {
                "ai_assessment": str(ai_risk_assessment),
                "risk_decision": risk_decision,
                "risk_rating": risk_rating,
                "quantitative_analysis": risk_summary,
                "recommendations": {
                    "position_size_usd": recommended_position_size,
                    "position_size_pct": recommended_position_size / current_portfolio_value,
                    "kelly_fraction": kelly_fraction,
                    "conservative_sizing": conservative_position_value,
                    "stop_loss_level": stop_loss_level,
                    "take_profit_level": take_profit_level,
                    "max_acceptable_loss": abs(cvar_95) * recommended_position_size
                },
                "risk_metrics": {
                    "var_95": var_parametric,
                    "cvar_95": cvar_95,
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                    "scenario_weighted_return": scenario_weighted_return
                },
                "compliance_status": {
                    "position_limit_ok": recommended_position_size / current_portfolio_value <= 0.1,
                    "var_limit_ok": abs(var_parametric) <= 0.05,
                    "exposure_limit_ok": (current_exposure + recommended_position_size / current_portfolio_value) <= 0.3
                },
                "processing_time": processing_time,
                "timestamp": time.time(),
                "risk_model_version": "advanced_quant_v1.0"
            }
            
            # Log comprehensive results
            logger.log_llm_interaction(
                prompt_type="advanced_risk_assessment",
                model=self.llm_config.model_name,
                prompt_length=len(task.description),
                response_length=len(str(ai_risk_assessment)),
                processing_time=processing_time
            )
            
            logger.log_action(
                "advanced_risk_assessment",
                {
                    "risk_decision": risk_decision,
                    "risk_rating": risk_rating,
                    "position_size_usd": recommended_position_size,
                    "kelly_fraction": kelly_fraction,
                    "var_95": var_parametric,
                    "sharpe_ratio": sharpe_ratio,
                    "compliance_ok": all(final_risk_assessment["compliance_status"].values()),
                    "description": "Advanced risk assessment completed"
                },
                status="completed"
            )
            
            return final_risk_assessment
            
        except Exception as e:
            logger.log_error(
                "advanced_risk_assessment_error",
                str(e),
                {
                    "signal_data": signal_data,
                    "portfolio_data": portfolio_data
                }
            )
            return self._fallback_risk_assessment(signal_data, portfolio_data)
    
    def _fallback_risk_assessment(self, signal_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative fallback when risk models fail"""
        return {
            "risk_decision": "rejected",
            "risk_rating": "high",
            "recommendations": {
                "position_size_usd": 1000.0,  # Very small position
                "position_size_pct": 0.01,
                "stop_loss_level": 0.95,
                "take_profit_level": 1.05
            },
            "risk_metrics": {
                "var_95": -0.05,
                "sharpe_ratio": 0.0
            },
            "processing_time": 0.1,
            "timestamp": time.time(),
            "risk_model_version": "fallback_conservative_v1.0"
        }
    
    def _extract_risk_decision(self, assessment_text: str) -> str:
        """Extract risk decision from AI assessment"""
        text_lower = assessment_text.lower()
        if 'approved' in text_lower and 'rejected' not in text_lower:
            return 'approved'
        elif 'conditional' in text_lower:
            return 'conditional'
        else:
            return 'rejected'
    
    def _extract_risk_rating(self, assessment_text: str) -> str:
        """Extract risk rating from AI assessment"""
        text_lower = assessment_text.lower()
        if 'extreme' in text_lower:
            return 'extreme'
        elif 'high' in text_lower:
            return 'high'
        elif 'medium' in text_lower:
            return 'medium'
        else:
            return 'low'
    
    def _extract_recommended_position_size(self, assessment_text: str) -> float:
        """Extract recommended position size from AI assessment"""
        import re
        
        # Look for dollar amounts
        dollar_matches = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)', assessment_text)
        if dollar_matches:
            return float(dollar_matches[0].replace(',', ''))
        
        # Look for percentage of portfolio
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*of\s*portfolio', assessment_text.lower())
        if pct_matches:
            return float(pct_matches[0]) / 100.0 * 100000  # Assume $100k portfolio
        
        return 2000.0  # Conservative default $2000
    
    def _extract_stop_loss(self, assessment_text: str) -> float:
        """Extract stop loss level from AI assessment"""
        import re
        
        # Look for stop loss percentages
        stop_matches = re.findall(r'stop\s*loss[:\s]*(\d+(?:\.\d+)?)\s*%', assessment_text.lower())
        if stop_matches:
            return 1.0 - (float(stop_matches[0]) / 100.0)  # Convert to multiplier
        
        return 0.95  # Default 5% stop loss
    
    def _extract_take_profit(self, assessment_text: str) -> float:
        """Extract take profit level from AI assessment"""
        import re
        
        # Look for take profit percentages
        profit_matches = re.findall(r'take\s*profit[:\s]*(\d+(?:\.\d+)?)\s*%', assessment_text.lower())
        if profit_matches:
            return 1.0 + (float(profit_matches[0]) / 100.0)  # Convert to multiplier
        
        return 1.10  # Default 10% take profit

class CrewAIIntelligenceSystem:
    """
    Advanced CrewAI Intelligence Coordination System
    
    Orchestrates sophisticated quantitative agents with specialized expertise:
    - Quantitative Analysis Agent: Advanced probabilistic modeling and statistical analysis
    - Advanced Risk Management Agent: Comprehensive risk assessment with quantitative models
    
    This system embodies the philosophy of combining AI reasoning with rigorous
    quantitative methods for superior trading intelligence.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.llm_config = LiteLLMConfig(config_manager)
        
        # Initialize specialized agents with quantitative expertise
        self.quantitative_agent = QuantitativeAnalysisAgent(self.llm_config)
        self.risk_agent = AdvancedRiskManagementAgent(self.llm_config)
        
        logger.info("CrewAI Intelligence System initialized with advanced quantitative agents")
        logger.log_action(
            "intelligence_system_initialization",
            {
                "agents_created": ["QuantitativeAnalysisAgent", "AdvancedRiskManagementAgent"],
                "llm_model": self.llm_config.model_name,
                "quantitative_models": ["Bayesian", "HMM", "Kelly", "VaR", "Markowitz"],
                "description": "Advanced CrewAI Intelligence System with quantitative specialization ready"
            },
            status="completed"
        )
    
    def generate_trading_signal(self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive trading signal using advanced quantitative analysis
        
        Process:
        1. Advanced quantitative market analysis with probabilistic models
        2. Comprehensive risk assessment with mathematical backing
        3. Signal synthesis and final recommendation
        4. Confidence scoring and uncertainty quantification
        """
        
        logger.info(f"Generating advanced trading signal for {market_data.get('symbol', 'BTC')}")
        
        signal_start_time = time.time()
        
        action_id = logger.log_action(
            "advanced_signal_generation",
            {
                "symbol": market_data.get('symbol', 'BTC'),
                "market_data_points": len(market_data.get('candles', [])),
                "portfolio_value": portfolio_data.get('total_value', 0),
                "agents_involved": ["QuantitativeAnalysisAgent", "AdvancedRiskManagementAgent"],
                "description": "Generating trading signal with advanced quantitative models"
            },
            status="started"
        )
        
        try:
            # Step 1: Advanced Quantitative Market Analysis
            logger.info("Executing advanced quantitative market analysis...")
            
            quantitative_analysis = self.quantitative_agent.analyze_market_data(market_data)
            
            # Step 2: Comprehensive Risk Assessment
            logger.info("Conducting comprehensive risk assessment...")
            
            # Prepare signal data for risk assessment
            signal_data = {
                'signal_type': 'buy',  # This will be determined from quantitative analysis
                'confidence_score': quantitative_analysis.get('confidence_score', 0.7),
                'market_assessment': quantitative_analysis.get('market_assessment', 'neutral'),
                'symbol': market_data.get('symbol', 'BTC'),
                'entry_price': market_data.get('latest_price', 0),
                'quantitative_models': quantitative_analysis.get('quantitative_models', {}),
                'statistical_significance': quantitative_analysis.get('statistical_significance', 0)
            }
            
            risk_assessment = self.risk_agent.assess_risk(signal_data, portfolio_data)
            
            # Step 3: Signal Synthesis and Final Decision
            logger.info("Synthesizing final trading recommendation...")
            
            final_signal = self._synthesize_final_signal(
                quantitative_analysis, 
                risk_assessment, 
                market_data, 
                portfolio_data
            )
            
            signal_processing_time = time.time() - signal_start_time
            
            # Add comprehensive metadata
            final_signal.update({
                'timestamp': time.time(),
                'processing_time': signal_processing_time,
                'system_version': 'advanced_quant_v1.0',
                'agents_used': ['QuantitativeAnalysisAgent', 'AdvancedRiskManagementAgent'],
                'models_applied': ['Bayesian', 'HMM', 'Kelly', 'VaR', 'CVaR', 'Markowitz'],
                'data_quality_score': self._assess_data_quality(market_data),
                'model_uncertainty': self._estimate_model_uncertainty(quantitative_analysis, risk_assessment)
            })
            
            # Comprehensive logging
            logger.log_trading_decision(
                symbol=final_signal['symbol'],
                signal_type=final_signal['recommended_action'],
                confidence=final_signal['final_confidence'],
                reasoning=final_signal.get('reasoning', 'Advanced quantitative analysis'),
                position_size=final_signal.get('position_size_pct', 0.02)
            )
            
            logger.log_action(
                "advanced_signal_generation",
                {
                    "final_action": final_signal['recommended_action'],
                    "confidence_score": final_signal['final_confidence'],
                    "position_size_pct": final_signal.get('position_size_pct', 0),
                    "risk_rating": risk_assessment.get('risk_rating', 'unknown'),
                    "quantitative_confidence": quantitative_analysis.get('confidence_score', 0),
                    "risk_approved": risk_assessment.get('risk_decision') == 'approved',
                    "processing_time_ms": signal_processing_time * 1000,
                    "description": "Advanced trading signal generation completed"
                },
                status="completed"
            )
            
            return final_signal
            
        except Exception as e:
            logger.log_error(
                "advanced_signal_generation_error",
                str(e),
                {
                    "market_data_keys": list(market_data.keys()),
                    "portfolio_data_keys": list(portfolio_data.keys()),
                    "agents_attempted": ["QuantitativeAnalysisAgent", "AdvancedRiskManagementAgent"]
                }
            )
            
            # Return conservative fallback signal
            return self._fallback_signal(market_data, portfolio_data)
    
    def _synthesize_final_signal(self, 
                               quantitative_analysis: Dict[str, Any],
                               risk_assessment: Dict[str, Any],
                               market_data: Dict[str, Any],
                               portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize final trading signal from quantitative analysis and risk assessment
        
        Combines:
        - Quantitative model confidence scores
        - Risk management approvals and constraints
        - Portfolio context and position sizing
        - Statistical significance and uncertainty measures
        """
        
        # Extract key metrics
        quant_confidence = quantitative_analysis.get('confidence_score', 0.7)
        market_assessment = quantitative_analysis.get('market_assessment', 'neutral')
        statistical_significance = quantitative_analysis.get('statistical_significance', 0.7)
        regime_confidence = quantitative_analysis.get('regime_based_confidence', 0.33)
        
        risk_decision = risk_assessment.get('risk_decision', 'rejected')
        risk_rating = risk_assessment.get('risk_rating', 'high')
        recommended_position = risk_assessment.get('recommendations', {}).get('position_size_pct', 0.01)
        
        # Decision synthesis logic
        if risk_decision == 'rejected':
            final_action = 'hold'
            final_confidence = 0.3
            reasoning = "Risk management rejected the signal due to excessive risk"
            
        elif risk_decision == 'conditional':
            final_action = 'cautious_buy' if market_assessment in ['bullish', 'strong_bullish'] else 'hold'
            final_confidence = quant_confidence * 0.7  # Reduced due to conditional approval
            reasoning = f"Conditional approval with {risk_rating} risk rating"
            
        elif risk_decision == 'approved':
            # Map market assessment to actions
            if market_assessment == 'strong_bullish' and quant_confidence > 0.8:
                final_action = 'strong_buy'
                final_confidence = min(0.95, (quant_confidence + statistical_significance + regime_confidence) / 3)
            elif market_assessment in ['bullish', 'strong_bullish']:
                final_action = 'buy'
                final_confidence = (quant_confidence + statistical_significance) / 2
            elif market_assessment in ['bearish', 'strong_bearish']:
                final_action = 'sell'
                final_confidence = (quant_confidence + statistical_significance) / 2
            else:
                final_action = 'hold'
                final_confidence = quant_confidence * 0.8
            
            reasoning = f"Approved by risk management with {risk_rating} risk rating. Quantitative confidence: {quant_confidence:.1%}, Statistical significance: {statistical_significance:.1%}"
        
        else:
            final_action = 'hold'
            final_confidence = 0.5
            reasoning = "Unknown risk decision, defaulting to hold"
        
        # Position sizing (use risk management recommendation)
        if final_action in ['strong_buy', 'buy', 'cautious_buy']:
            position_size_pct = recommended_position
        else:
            position_size_pct = 0.0
        
        return {
            'symbol': market_data.get('symbol', 'BTC'),
            'recommended_action': final_action,
            'final_confidence': final_confidence,
            'position_size_pct': position_size_pct,
            'reasoning': reasoning,
            'quantitative_analysis': quantitative_analysis,
            'risk_assessment': risk_assessment,
            'synthesis_metrics': {
                'quant_confidence': quant_confidence,
                'statistical_significance': statistical_significance,
                'regime_confidence': regime_confidence,
                'risk_decision': risk_decision,
                'risk_rating': risk_rating,
                'market_assessment': market_assessment
            },
            'entry_conditions': {
                'price_target': market_data.get('latest_price', 0),
                'stop_loss': risk_assessment.get('recommendations', {}).get('stop_loss_level', 0.95),
                'take_profit': risk_assessment.get('recommendations', {}).get('take_profit_level', 1.10),
                'max_loss_usd': risk_assessment.get('recommendations', {}).get('max_acceptable_loss', 1000)
            }
        }
    
    def _assess_data_quality(self, market_data: Dict[str, Any]) -> float:
        """Assess the quality of market data for analysis"""
        
        candles = market_data.get('candles', [])
        
        if len(candles) >= 200:
            return 0.95  # Excellent data quality
        elif len(candles) >= 100:
            return 0.85  # Good data quality
        elif len(candles) >= 50:
            return 0.70  # Acceptable data quality
        else:
            return 0.50  # Poor data quality
    
    def _estimate_model_uncertainty(self, 
                                  quantitative_analysis: Dict[str, Any], 
                                  risk_assessment: Dict[str, Any]) -> float:
        """Estimate overall model uncertainty"""
        
        # Factors that increase uncertainty
        uncertainty_factors = []
        
        # Low statistical significance
        stat_sig = quantitative_analysis.get('statistical_significance', 1.0)
        if stat_sig < 0.6:
            uncertainty_factors.append(0.3)
        
        # High risk rating
        risk_rating = risk_assessment.get('risk_rating', 'low')
        if risk_rating in ['high', 'extreme']:
            uncertainty_factors.append(0.4)
        
        # Low regime confidence
        regime_conf = quantitative_analysis.get('regime_based_confidence', 1.0)
        if regime_conf < 0.5:
            uncertainty_factors.append(0.2)
        
        # Calculate overall uncertainty
        if uncertainty_factors:
            return min(0.8, sum(uncertainty_factors))
        else:
            return 0.1  # Base model uncertainty
    
    def _fallback_signal(self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative fallback signal when analysis fails"""
        
        return {
            'symbol': market_data.get('symbol', 'BTC'),
            'recommended_action': 'hold',
            'final_confidence': 0.3,
            'position_size_pct': 0.0,
            'reasoning': 'Analysis failed, defaulting to conservative hold position',
            'quantitative_analysis': {},
            'risk_assessment': {'risk_decision': 'rejected', 'risk_rating': 'extreme'},
            'timestamp': time.time(),
            'system_version': 'fallback_v1.0',
            'data_quality_score': 0.1,
            'model_uncertainty': 0.9
        }
