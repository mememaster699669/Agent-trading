"""
Agent Trading System - Main Entry Point
Hybrid CrewAI + ADK Architecture with Comprehensive Logging
Based on GCP DevOps Agent pattern
"""

import asyncio
import time
import signal
import sys
from typing import Dict, Any, Optional
import os
import json
from pathlib import Path

# Import our modules
from .config import ConfigManager
from .environment import validate_environment, get_environment_summary
from .logging_system import setup_agent_logging, get_logger
from .dataset import BTCDataManager
from .crewai_intelligence import CrewAIIntelligenceSystem
from .adk_execution import ExecutionLayer

# Import all advanced 5-phase frameworks for complete system integration
try:
    from .quant_models import (
        BayesianTradingFramework,
        QuantLibFinancialEngineering,
        AdvancedPortfolioOptimization,
        AdvancedTimeSeriesAnalysis,
        AdvancedMLTradingFramework,
        AdvancedPhysicsModels,
        MarketMicrostructure
    )
    ADVANCED_FRAMEWORKS_AVAILABLE = True
    print("✅ Advanced 5-phase frameworks available for main system")
except ImportError as e:
    print(f"⚠️ Advanced frameworks not available: {e}")
    ADVANCED_FRAMEWORKS_AVAILABLE = False

# Setup logging first thing
loggers = setup_agent_logging()
main_logger = loggers["main"]
data_logger = loggers["data"] 
crewai_logger = loggers["crewai"]
adk_logger = loggers["adk"]

class AgentTradingSystem:
    """
    Main Agent Trading System
    Orchestrates CrewAI Intelligence + ADK Execution with comprehensive logging
    """
    
    def __init__(self):
        """Initialize the agent trading system"""
        
        main_logger.info("Initializing Agent Trading System...")
        
        # Validate environment first
        self._validate_environment()
        
        # Initialize configuration
        self.config = ConfigManager()
        main_logger.info("Configuration manager initialized")
        
        # Initialize components
        self.data_manager = None
        self.intelligence_system = None
        self.execution_engine = None
        self.running = False
        
        # Initialize all advanced 5-phase frameworks for main system
        self.advanced_frameworks = {}
        self.framework_status = {}
        self._initialize_advanced_frameworks()
        
        # System metrics
        self.start_time = time.time()
        self.cycles_completed = 0
        self.errors_encountered = 0
        
        # Dashboard data tracking
        self.latest_decision = None
        self.latest_price = 0
        self.price_change_24h = 0
        self.volume_24h = 0
        self.total_return = 0.0
        self.total_pnl = 0.0
        self.win_rate = 0
        self.current_risk_level = "LOW"
        self.current_var = 0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.is_running = False
        
        # Performance tracking
        self.trades_history = []
        self.daily_returns = []
        
        main_logger.log_action(
            "system_initialization", 
            {
                "config_loaded": True,
                "environment_validated": True,
                "description": "Agent Trading System initialization started"
            },
            status="started"
        )
    
    def _initialize_advanced_frameworks(self):
        """Initialize all advanced 5-phase frameworks for main system orchestration"""
        if not ADVANCED_FRAMEWORKS_AVAILABLE:
            main_logger.warning("⚠️ Advanced frameworks not available - using basic system only")
            return
        
        main_logger.info("Initializing advanced 5-phase frameworks for main system...")
        
        try:
            # Phase 1: Bayesian Framework
            main_logger.info("Initializing Phase 1: Bayesian Trading Framework")
            self.advanced_frameworks['bayesian'] = BayesianTradingFramework()
            self.framework_status['bayesian'] = self.advanced_frameworks['bayesian'].is_available
            
            # Phase 2: QuantLib Framework
            main_logger.info("Initializing Phase 2: QuantLib Financial Engineering")
            self.advanced_frameworks['quantlib'] = QuantLibFinancialEngineering()
            self.framework_status['quantlib'] = self.advanced_frameworks['quantlib'].is_available
            
            # Phase 3: Portfolio Optimization Framework
            main_logger.info("Initializing Phase 3: Advanced Portfolio Optimization")
            self.advanced_frameworks['portfolio'] = AdvancedPortfolioOptimization()
            self.framework_status['portfolio'] = self.advanced_frameworks['portfolio'].is_available
            
            # Phase 4: Time Series Framework
            main_logger.info("Initializing Phase 4: Advanced Time Series Analysis")
            self.advanced_frameworks['timeseries'] = AdvancedTimeSeriesAnalysis()
            self.framework_status['timeseries'] = self.advanced_frameworks['timeseries'].is_available
            
            # Phase 5: ML/AI Framework
            main_logger.info("Initializing Phase 5: Advanced ML Trading Framework")
            self.advanced_frameworks['ml'] = AdvancedMLTradingFramework()
            self.framework_status['ml'] = self.advanced_frameworks['ml'].is_available
            
            # Additional Physics and Microstructure Models
            main_logger.info("Initializing Advanced Physics Models")
            self.advanced_frameworks['physics'] = AdvancedPhysicsModels()
            self.framework_status['physics'] = self.advanced_frameworks['physics'].is_available
            
            main_logger.info("Initializing Market Microstructure Models")
            self.advanced_frameworks['microstructure'] = MarketMicrostructure()
            self.framework_status['microstructure'] = self.advanced_frameworks['microstructure'].is_available
            
            # Log framework status
            available_count = sum(1 for status in self.framework_status.values() if status)
            total_count = len(self.framework_status)
            
            main_logger.info(f"✅ Advanced frameworks initialized: {available_count}/{total_count} available")
            main_logger.info(f"Framework availability: {self.framework_status}")
            
            main_logger.log_action(
                "advanced_frameworks_init",
                {
                    "frameworks_available": available_count,
                    "total_frameworks": total_count,
                    "framework_status": self.framework_status,
                    "description": "Advanced 5-phase frameworks initialization completed"
                },
                status="completed"
            )
            
        except Exception as e:
            main_logger.error(f"⚠️ Could not initialize advanced frameworks: {e}")
            main_logger.log_error(
                "advanced_frameworks_init_failed",
                str(e),
                {"error_type": type(e).__name__}
            )
            self.advanced_frameworks = {}
            self.framework_status = {}
    
    def _validate_environment(self):
        """Validate environment configuration"""
        
        main_logger.info("Validating environment configuration...")
        
        validation_result = validate_environment()
        env_issues = validation_result.get("errors", [])
        env_warnings = validation_result.get("warnings", [])
        
        # Log warnings
        if env_warnings:
            main_logger.warning("Environment warnings:")
            for warning in env_warnings:
                main_logger.warning(f"  - {warning}")
        
        # Check for errors
        if env_issues:
            main_logger.error("Environment validation failed:")
            for issue in env_issues:
                main_logger.error(f"  - {issue}")
            
            main_logger.log_error(
                "environment_validation_failed",
                f"{len(env_issues)} environment issues found",
                {"issues": env_issues}
            )
            
            if len(env_issues) > 5:  # Too many critical issues
                raise RuntimeError("Environment validation failed with critical issues")
        else:
            main_logger.info("Environment validation passed ✅")
        
        # Log environment summary
        env_summary = validation_result.get("config_summary", {})
        main_logger.info("Environment Summary:")
        for key, value in env_summary.items():
            main_logger.info(f"  {key}: {value}")
        
        # Log advanced frameworks status
        frameworks_status = validation_result.get("advanced_frameworks_status", {})
        if frameworks_status.get("enabled_frameworks"):
            main_logger.info(f"Advanced frameworks enabled: {', '.join(frameworks_status['enabled_frameworks'])}")
        if frameworks_status.get("configuration_issues"):
            main_logger.warning("Framework configuration issues:")
            for issue in frameworks_status["configuration_issues"]:
                main_logger.warning(f"  - {issue}")
    
    async def initialize_components(self):
        """Initialize all system components"""
        
        main_logger.info("Initializing system components...")
        
        try:
            # Initialize data manager
            data_logger.info("Initializing BTC Data Manager...")
            action_id = data_logger.log_action(
                "data_manager_init",
                {"symbol": "BTC", "description": "Initializing BTC data pipeline"},
                status="started"
            )
            
            self.data_manager = BTCDataManager(self.config)
            await self.data_manager.initialize()
            
            data_logger.log_action(
                "data_manager_init",
                {"description": "BTC data manager initialized successfully"},
                status="completed"
            )
            
            # Initialize CrewAI intelligence system
            crewai_logger.info("Initializing CrewAI Intelligence System...")
            action_id = crewai_logger.log_action(
                "crewai_init",
                {"description": "Initializing CrewAI agents and LLM configuration"},
                status="started"
            )
            
            self.intelligence_system = CrewAIIntelligenceSystem(self.config)
            
            crewai_logger.log_action(
                "crewai_init", 
                {"description": "CrewAI Intelligence System initialized successfully"},
                status="completed"
            )
            
            # Initialize ADK execution engine
            adk_logger.info("Initializing ADK Execution Engine...")
            action_id = adk_logger.log_action(
                "adk_init",
                {"description": "Initializing ADK execution and risk management"},
                status="started"
            )
            
            from .adk_execution import ExecutionLayer
            self.execution_engine = ExecutionLayer(self.config)
            
            adk_logger.log_action(
                "adk_init",
                {"description": "ADK Execution Engine initialized successfully"}, 
                status="completed"
            )
            
            main_logger.log_action(
                "system_initialization",
                {
                    "components": ["BTCDataManager", "CrewAIIntelligenceSystem", "ExecutionLayer"],
                    "description": "All components initialized successfully"
                },
                status="completed"
            )
            
            main_logger.info("All system components initialized successfully ✅")
            
        except Exception as e:
            main_logger.log_error(
                "component_initialization_failed",
                str(e),
                {
                    "data_manager": self.data_manager is not None,
                    "intelligence_system": self.intelligence_system is not None,
                    "execution_engine": self.execution_engine is not None
                }
            )
            raise
    
    async def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        
        cycle_start = time.time()
        cycle_id = f"cycle_{self.cycles_completed + 1:04d}"
        
        main_logger.info(f"Starting trading cycle {cycle_id}")
        
        action_id = main_logger.log_action(
            "trading_cycle",
            {
                "cycle_id": cycle_id,
                "cycle_number": self.cycles_completed + 1,
                "description": f"Executing trading cycle {cycle_id}"
            },
            status="started"
        )
        
        try:
            # Step 1: Fetch and process market data
            data_logger.info("Fetching latest BTC market data...")
            market_data = await self.data_manager.fetch_latest_data()
            
            # Update dashboard price data
            self.latest_price = market_data.get('latest_price', 0)
            self.price_change_24h = market_data.get('price_change_24h', 0)
            self.volume_24h = market_data.get('volume_24h', 0)
            
            data_logger.log_data_operation(
                operation="fetch_latest",
                symbol="BTC",
                data_info={
                    "candles_count": len(market_data.get('candles', [])),
                    "latest_price": market_data.get('latest_price', 'N/A'),
                    "timeframe": market_data.get('timeframe', '15m')
                }
            )
            
            # Step 2: Generate intelligence signals with advanced frameworks integration
            crewai_logger.info("Generating trading signals with CrewAI and advanced frameworks...")
            portfolio_data = await self._get_portfolio_data()
            
            # Enhance with advanced framework analysis if available
            advanced_analysis = await self._run_advanced_analysis(market_data)
            
            trading_signal = self.intelligence_system.generate_trading_signal(
                market_data, portfolio_data, advanced_analysis
            )
            
            # Update dashboard decision data
            self.latest_decision = {
                'action': trading_signal.get('action', 'HOLD'),
                'confidence': trading_signal.get('final_confidence', 0),
                'reasoning': trading_signal.get('reasoning', 'AI analysis completed'),
                'timestamp': time.time()
            }
            
            # Write comprehensive decision and analysis data to file for dashboard
            dashboard_data = {
                'market_data': {
                    'price': self.latest_price,
                    'change_24h': self.price_change_24h,
                    'volume_24h': self.volume_24h,
                    'timestamp': time.time()
                },
                'decision': self.latest_decision
            }
            
            # Include advanced analysis results if available
            if advanced_analysis and advanced_analysis.get("frameworks_used"):
                dashboard_data['advanced_analysis'] = {
                    'frameworks_used': advanced_analysis.get("frameworks_used", []),
                    'frameworks_count': len(advanced_analysis.get("frameworks_used", [])),
                    'analysis_summary': advanced_analysis.get("analysis", {}),
                    'timestamp': advanced_analysis.get("timestamp", time.time())
                }
                main_logger.info(f"Dashboard data enhanced with {len(advanced_analysis.get('frameworks_used', []))} advanced framework results")
            
            self._write_dashboard_data(dashboard_data)
            
            # Step 3: Execute trades with ADK
            adk_logger.info("Processing signals with ADK Execution Engine...")
            execution_result = await self.execution_engine.process_signal(trading_signal)
            
            # Log execution result
            if execution_result.get('executed'):
                adk_logger.log_execution_result(
                    order_id=execution_result.get('order_id', 'unknown'),
                    execution_report=execution_result
                )
            
            # Update cycle metrics
            cycle_time = time.time() - cycle_start
            self.cycles_completed += 1
            
            # Update dashboard performance metrics
            self._update_performance_metrics(execution_result, trading_signal)
            
            main_logger.log_action(
                "trading_cycle",
                {
                    "cycle_id": cycle_id,
                    "cycle_time_ms": cycle_time * 1000,
                    "signal_confidence": trading_signal.get('final_confidence', 0),
                    "action_taken": execution_result.get('action', 'none'),
                    "description": f"Trading cycle {cycle_id} completed successfully"
                },
                status="completed"
            )
            
            main_logger.info(f"Cycle {cycle_id} completed in {cycle_time:.2f}s")
            
            return execution_result
            
        except Exception as e:
            self.errors_encountered += 1
            
            main_logger.log_error(
                "trading_cycle_failed",
                str(e),
                {
                    "cycle_id": cycle_id,
                    "cycle_number": self.cycles_completed + 1,
                    "error_count": self.errors_encountered
                }
            )
            
            main_logger.error(f"Trading cycle {cycle_id} failed: {e}")
            raise
    
    def _update_performance_metrics(self, execution_result: Dict[str, Any], trading_signal: Dict[str, Any]):
        """Update performance metrics for dashboard"""
        try:
            # Update risk level based on current conditions
            confidence = trading_signal.get('final_confidence', 0)
            if confidence > 80:
                self.current_risk_level = "HIGH"
            elif confidence > 50:
                self.current_risk_level = "MEDIUM"
            else:
                self.current_risk_level = "LOW"
            
            # Calculate win rate (simplified)
            if execution_result.get('executed'):
                pnl = execution_result.get('pnl', 0)
                self.trades_history.append(pnl)
                
                # Keep only recent trades for win rate calculation
                if len(self.trades_history) > 100:
                    self.trades_history = self.trades_history[-100:]
                
                # Calculate win rate
                winning_trades = sum(1 for pnl in self.trades_history if pnl > 0)
                self.win_rate = int((winning_trades / len(self.trades_history)) * 100) if self.trades_history else 0
                
                # Update total P&L
                self.total_pnl = sum(self.trades_history)
                
                # Calculate total return percentage
                if self.total_pnl != 0:
                    initial_capital = 10000  # Assume $10k initial
                    self.total_return = (self.total_pnl / initial_capital) * 100
            
            # Calculate max drawdown (simplified - without numpy)
            if len(self.trades_history) > 10:
                try:
                    returns = self.trades_history[-30:]  # Last 30 trades
                    # Simple VaR calculation (5th percentile)
                    sorted_returns = sorted(returns)
                    var_index = max(0, int(len(sorted_returns) * 0.05))
                    self.current_var = int(abs(sorted_returns[var_index]))
                    
                    # Simple max drawdown calculation
                    cumulative = 0
                    max_value = 0
                    max_drawdown = 0
                    for ret in returns:
                        cumulative += ret
                        max_value = max(max_value, cumulative)
                        drawdown = max_value - cumulative
                        max_drawdown = max(max_drawdown, drawdown)
                    self.max_drawdown = max_drawdown
                except Exception as calc_error:
                    main_logger.error(f"Error in calculations: {calc_error}")
            
            # Write updated performance data to dashboard
            self._write_dashboard_data({})
            
        except Exception as e:
            main_logger.error(f"Error updating performance metrics: {e}")
    
    def _write_dashboard_data(self, data: Dict[str, Any]):
        """Write data to files for dashboard consumption"""
        try:
            # Create data directory if it doesn't exist
            data_dir = Path("/app/data")
            data_dir.mkdir(exist_ok=True)
            
            # Write market data
            if 'market_data' in data:
                market_file = data_dir / "latest_market_data.json"
                with open(market_file, 'w') as f:
                    json.dump(data['market_data'], f)
            
            # Write decision data
            if 'decision' in data:
                decision_file = data_dir / "latest_decision.json"
                with open(decision_file, 'w') as f:
                    json.dump(data['decision'], f)
            
            # Write performance data
            performance_data = {
                'total_pnl': getattr(self, 'total_pnl', 0),
                'win_rate': getattr(self, 'win_rate', 0),
                'total_return': getattr(self, 'total_return', 0),
                'current_risk_level': getattr(self, 'current_risk_level', 'LOW'),
                'current_var': getattr(self, 'current_var', 0),
                'max_drawdown': getattr(self, 'max_drawdown', 0),
                'cycles_completed': self.cycles_completed,
                'errors_encountered': self.errors_encountered,
                'uptime_seconds': time.time() - self.start_time,
                'timestamp': time.time()
            }
            performance_file = data_dir / "latest_performance.json"
            with open(performance_file, 'w') as f:
                json.dump(performance_data, f)
                
        except Exception as e:
            main_logger.error(f"Error writing dashboard data: {e}")
    
    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get current portfolio information"""
        
        # This would interface with your exchange/broker API
        # For now, return mock data
        return {
            "total_value": 10000.0,
            "available_cash": 5000.0,
            "crypto_exposure": 0.3,
            "positions": {
                "BTC": {"quantity": 0.1, "value": 5000.0}
            }
        }
    
    async def run_continuous(self, cycle_interval: int = 900):  # 15 minutes default
        """Run the trading system continuously"""
        
        main_logger.info(f"Starting continuous trading with {cycle_interval}s intervals")
        
        main_logger.log_action(
            "continuous_trading_start",
            {
                "cycle_interval_seconds": cycle_interval,
                "expected_cycles_per_day": 24 * 60 * 60 / cycle_interval,
                "description": "Starting continuous trading operation"
            },
            status="started"
        )
        
        self.running = True
        self.is_running = True  # For dashboard
        
        try:
            while self.running:
                try:
                    # Execute trading cycle
                    await self.run_trading_cycle()
                    
                    # Log system metrics periodically
                    if self.cycles_completed % 10 == 0:
                        await self._log_system_metrics()
                    
                    # Wait for next cycle
                    main_logger.info(f"Waiting {cycle_interval}s until next cycle...")
                    await asyncio.sleep(cycle_interval)
                    
                except Exception as e:
                    main_logger.error(f"Error in trading cycle: {e}")
                    
                    # Exponential backoff on errors
                    backoff_time = min(300, 30 * (2 ** min(self.errors_encountered, 5)))
                    main_logger.warning(f"Backing off for {backoff_time}s due to error")
                    await asyncio.sleep(backoff_time)
                    
        except KeyboardInterrupt:
            main_logger.info("Received shutdown signal")
        finally:
            self.running = False
            self.is_running = False  # For dashboard
            main_logger.info("Continuous trading stopped")
            await self.shutdown()
    
    async def _run_advanced_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive analysis using all available advanced frameworks"""
        
        if not self.advanced_frameworks:
            main_logger.info("No advanced frameworks available - using basic analysis")
            return {"status": "basic_analysis_only"}
        
        main_logger.info("Running advanced 5-phase analysis...")
        analysis_results = {
            "timestamp": time.time(),
            "frameworks_used": [],
            "analysis": {}
        }
        
        try:
            # Extract price data for analysis
            candles = market_data.get('candles', [])
            if not candles:
                main_logger.warning("No price data available for advanced analysis")
                return {"status": "no_data", "error": "No candle data available"}
            
            # Convert candles to arrays for analysis
            prices = [float(candle.get('close', 0)) for candle in candles[-100:]]  # Last 100 periods
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            # Phase 1: Bayesian Analysis
            if self.framework_status.get('bayesian', False):
                try:
                    main_logger.info("Running Bayesian hierarchical analysis...")
                    bayesian_framework = self.advanced_frameworks['bayesian']
                    bayesian_result = bayesian_framework.hierarchical_model_analysis(prices)
                    analysis_results["analysis"]["bayesian"] = bayesian_result
                    analysis_results["frameworks_used"].append("bayesian")
                    main_logger.info("✅ Bayesian analysis completed")
                except Exception as e:
                    main_logger.error(f"Bayesian analysis failed: {e}")
                    analysis_results["analysis"]["bayesian"] = {"error": str(e)}
            
            # Phase 2: QuantLib Derivatives Analysis
            if self.framework_status.get('quantlib', False):
                try:
                    main_logger.info("Running QuantLib derivatives analysis...")
                    quantlib_framework = self.advanced_frameworks['quantlib']
                    current_price = prices[-1] if prices else 50000
                    quantlib_result = quantlib_framework.black_scholes_option_pricing(
                        spot_price=current_price, 
                        strike=current_price * 1.05,  # 5% OTM call
                        risk_free_rate=0.05, 
                        volatility=0.3, 
                        time_to_expiry=0.25
                    )
                    analysis_results["analysis"]["quantlib"] = quantlib_result
                    analysis_results["frameworks_used"].append("quantlib")
                    main_logger.info("✅ QuantLib analysis completed")
                except Exception as e:
                    main_logger.error(f"QuantLib analysis failed: {e}")
                    analysis_results["analysis"]["quantlib"] = {"error": str(e)}
            
            # Phase 3: Portfolio Optimization
            if self.framework_status.get('portfolio', False):
                try:
                    main_logger.info("Running portfolio optimization analysis...")
                    portfolio_framework = self.advanced_frameworks['portfolio']
                    symbols = ['BTC', 'ETH', 'SPY', 'GOLD']  # Multi-asset analysis
                    expected_returns = [0.15, 0.12, 0.08, 0.05]  # Example expected returns
                    portfolio_result = portfolio_framework.comprehensive_portfolio_analysis(symbols, expected_returns)
                    analysis_results["analysis"]["portfolio"] = portfolio_result
                    analysis_results["frameworks_used"].append("portfolio")
                    main_logger.info("✅ Portfolio optimization completed")
                except Exception as e:
                    main_logger.error(f"Portfolio analysis failed: {e}")
                    analysis_results["analysis"]["portfolio"] = {"error": str(e)}
            
            # Phase 4: GARCH Time Series Analysis
            if self.framework_status.get('timeseries', False) and len(returns) > 50:
                try:
                    main_logger.info("Running GARCH volatility analysis...")
                    timeseries_framework = self.advanced_frameworks['timeseries']
                    garch_result = timeseries_framework.garch_volatility_modeling(returns)
                    analysis_results["analysis"]["timeseries"] = garch_result
                    analysis_results["frameworks_used"].append("timeseries")
                    main_logger.info("✅ GARCH analysis completed")
                except Exception as e:
                    main_logger.error(f"Time series analysis failed: {e}")
                    analysis_results["analysis"]["timeseries"] = {"error": str(e)}
            
            # Phase 5: ML/AI Predictions
            if self.framework_status.get('ml', False) and len(prices) > 20:
                try:
                    main_logger.info("Running ML ensemble predictions...")
                    ml_framework = self.advanced_frameworks['ml']
                    
                    # Create feature matrix from price data
                    features = []
                    targets = []
                    for i in range(10, len(prices)-1):
                        # Use last 10 prices as features
                        feature_row = prices[i-10:i]
                        target = 1 if prices[i+1] > prices[i] else 0  # Binary target: price up/down
                        features.append(feature_row)
                        targets.append(target)
                    
                    if len(features) > 10:
                        ml_result = ml_framework.ensemble_prediction(features, targets)
                        analysis_results["analysis"]["ml"] = ml_result
                        analysis_results["frameworks_used"].append("ml")
                        main_logger.info("✅ ML ensemble analysis completed")
                except Exception as e:
                    main_logger.error(f"ML analysis failed: {e}")
                    analysis_results["analysis"]["ml"] = {"error": str(e)}
            
            # Physics-based Risk Analysis
            if self.framework_status.get('physics', False):
                try:
                    main_logger.info("Running physics-based risk analysis...")
                    physics_framework = self.advanced_frameworks['physics']
                    physics_result = physics_framework.comprehensive_physics_analysis(returns if returns else [0.01])
                    analysis_results["analysis"]["physics"] = physics_result
                    analysis_results["frameworks_used"].append("physics")
                    main_logger.info("✅ Physics analysis completed")
                except Exception as e:
                    main_logger.error(f"Physics analysis failed: {e}")
                    analysis_results["analysis"]["physics"] = {"error": str(e)}
            
            # Market Microstructure Analysis
            if self.framework_status.get('microstructure', False):
                try:
                    main_logger.info("Running market microstructure analysis...")
                    microstructure_framework = self.advanced_frameworks['microstructure']
                    # Use volume data if available
                    volumes = [float(candle.get('volume', 1000000)) for candle in candles[-50:]]
                    microstructure_result = microstructure_framework.analyze_market_impact(prices[-50:], volumes)
                    analysis_results["analysis"]["microstructure"] = microstructure_result
                    analysis_results["frameworks_used"].append("microstructure")
                    main_logger.info("✅ Microstructure analysis completed")
                except Exception as e:
                    main_logger.error(f"Microstructure analysis failed: {e}")
                    analysis_results["analysis"]["microstructure"] = {"error": str(e)}
            
            # Log comprehensive results
            frameworks_count = len(analysis_results["frameworks_used"])
            main_logger.info(f"✅ Advanced analysis completed using {frameworks_count} frameworks")
            main_logger.log_action(
                "advanced_analysis",
                {
                    "frameworks_used": analysis_results["frameworks_used"],
                    "frameworks_count": frameworks_count,
                    "data_points_analyzed": len(prices),
                    "description": f"Comprehensive 5-phase analysis using {frameworks_count} advanced frameworks"
                },
                status="completed"
            )
            
            return analysis_results
            
        except Exception as e:
            main_logger.error(f"Advanced analysis failed: {e}")
            main_logger.log_error(
                "advanced_analysis_failed",
                str(e),
                {"error_type": type(e).__name__}
            )
            return {"status": "error", "error": str(e), "frameworks_used": []}
    
    async def _log_system_metrics(self):
        """Log system performance metrics"""
        
        uptime = time.time() - self.start_time
        
        metrics = {
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "cycles_completed": self.cycles_completed,
            "errors_encountered": self.errors_encountered,
            "success_rate": (self.cycles_completed - self.errors_encountered) / max(self.cycles_completed, 1) * 100,
            "average_cycle_time": uptime / max(self.cycles_completed, 1)
        }
        
        main_logger.log_system_metrics(metrics)
        
        main_logger.info("System Metrics:")
        main_logger.info(f"  Uptime: {metrics['uptime_hours']:.1f} hours")
        main_logger.info(f"  Cycles: {metrics['cycles_completed']}")
        main_logger.info(f"  Success Rate: {metrics['success_rate']:.1f}%")
        main_logger.info(f"  Avg Cycle Time: {metrics['average_cycle_time']:.1f}s")
    
    async def shutdown(self):
        """Graceful system shutdown"""
        
        main_logger.info("Initiating system shutdown...")
        
        self.running = False
        
        # Final metrics log
        await self._log_system_metrics()
        
        # Cleanup components
        if self.data_manager:
            await self.data_manager.cleanup()
            data_logger.info("Data manager cleaned up")
        
        if self.execution_engine:
            await self.execution_engine.cleanup()
            adk_logger.info("Execution engine cleaned up")
        
        main_logger.log_action(
            "system_shutdown",
            {
                "total_uptime_seconds": time.time() - self.start_time,
                "total_cycles": self.cycles_completed,
                "total_errors": self.errors_encountered,
                "description": "Agent Trading System shutdown completed"
            },
            status="completed"
        )
        
        main_logger.info("Agent Trading System shutdown completed ✅")

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    main_logger.info(f"Received signal {signum}, initiating shutdown...")
    # The main loop will handle the actual shutdown
    raise KeyboardInterrupt()

async def main():
    """Main entry point"""
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create and initialize system
        trading_system = AgentTradingSystem()
        await trading_system.initialize_components()
        
        # Get cycle interval from environment
        cycle_interval = int(os.getenv("TRADING_CYCLE_INTERVAL", "900"))  # 15 minutes default
        
        # Run continuously
        await trading_system.run_continuous(cycle_interval)
        
    except Exception as e:
        main_logger.critical(f"Fatal error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
