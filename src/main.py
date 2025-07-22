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

# Import our modules
from .config import ConfigManager
from .environment import validate_environment, get_environment_summary
from .logging_system import setup_agent_logging, get_logger
from .dataset import BTCDataManager
from .crewai_intelligence import CrewAIIntelligenceSystem
from .adk_execution import ExecutionLayer

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
        
        # System metrics
        self.start_time = time.time()
        self.cycles_completed = 0
        self.errors_encountered = 0
        
        main_logger.log_action(
            "system_initialization", 
            {
                "config_loaded": True,
                "environment_validated": True,
                "description": "Agent Trading System initialization started"
            },
            status="started"
        )
    
    def _validate_environment(self):
        """Validate environment configuration"""
        
        main_logger.info("Validating environment configuration...")
        
        env_issues = validate_environment()
        
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
        env_summary = get_environment_summary()
        main_logger.info("Environment Summary:")
        for key, value in env_summary.items():
            main_logger.info(f"  {key}: {value}")
    
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
            
            data_logger.log_data_operation(
                operation="fetch_latest",
                symbol="BTC",
                data_info={
                    "candles_count": len(market_data.get('candles', [])),
                    "latest_price": market_data.get('latest_price', 'N/A'),
                    "timeframe": market_data.get('timeframe', '15m')
                }
            )
            
            # Step 2: Generate intelligence signals
            crewai_logger.info("Generating trading signals with CrewAI...")
            portfolio_data = await self._get_portfolio_data()
            
            trading_signal = self.intelligence_system.generate_trading_signal(
                market_data, portfolio_data
            )
            
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
            await self.shutdown()
    
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
