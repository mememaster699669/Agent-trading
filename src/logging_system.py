"""
Advanced Logging System for Agent Trading
Comprehensive logging to track every action, decision, and error
Based on GCP DevOps Agent logging pattern
"""

import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import os

# Configure logging format similar to GCP agent
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

class AgentTradingLogger:
    """
    Comprehensive logging system for agent trading operations
    Tracks every action, decision, and data flow
    """
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Set logging level
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Prevent duplicate logs
        if self.logger.handlers:
            return
        
        # Create formatters
        formatter = logging.Formatter(LOGGING_FORMAT)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        self._setup_file_logging(formatter, level)
        
        # Initialize action tracking
        self.action_counter = 0
        self.session_start = time.time()
        self.session_id = f"session_{int(self.session_start)}"
        
        self.log_system_start()
    
    def _setup_file_logging(self, formatter, level):
        """Setup file logging with rotation"""
        try:
            # Create logs directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Daily log file
            log_file = log_dir / f"agent_trading_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)
            
            # Action-specific log file for detailed tracking
            action_log_file = log_dir / f"actions_{datetime.now().strftime('%Y%m%d')}.jsonl"
            self.action_log_file = action_log_file
            
        except Exception as e:
            self.logger.warning(f"Could not setup file logging: {e}")
            self.action_log_file = None
    
    def log_system_start(self):
        """Log system initialization"""
        self.logger.info("=" * 80)
        self.logger.info(f"AGENT TRADING SYSTEM STARTED - Session ID: {self.session_id}")
        self.logger.info(f"Logger: {self.name}")
        self.logger.info(f"Start Time: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)
    
    def log_action(self, action_type: str, details: Dict[str, Any], 
                   status: str = "started", error: Optional[str] = None):
        """
        Log a specific action with detailed context
        Similar to GCP agent action tracking
        """
        self.action_counter += 1
        
        action_data = {
            "session_id": self.session_id,
            "action_id": f"action_{self.action_counter:04d}",
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "status": status,
            "details": details,
            "logger_name": self.name
        }
        
        if error:
            action_data["error"] = error
        
        # Log to console/file
        if status == "started":
            self.logger.info(f"ACTION START [{action_data['action_id']}] {action_type}: {details.get('description', 'No description')}")
        elif status == "completed":
            self.logger.info(f"ACTION COMPLETE [{action_data['action_id']}] {action_type}: Success")
        elif status == "failed":
            self.logger.error(f"ACTION FAILED [{action_data['action_id']}] {action_type}: {error}")
        elif status == "progress":
            self.logger.info(f"ACTION PROGRESS [{action_data['action_id']}] {action_type}: {details.get('progress', 'Working...')}")
        
        # Log to action file (JSONL format for easy parsing)
        if self.action_log_file:
            try:
                with open(self.action_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(action_data, default=str) + '\n')
            except Exception as e:
                self.logger.warning(f"Could not write to action log: {e}")
        
        return action_data["action_id"]
    
    def log_data_operation(self, operation: str, symbol: str, data_info: Dict[str, Any]):
        """Log data operations (fetching, processing, storing)"""
        details = {
            "operation": operation,
            "symbol": symbol,
            "data_info": data_info,
            "description": f"{operation} data for {symbol}"
        }
        return self.log_action("data_operation", details)
    
    def log_trading_decision(self, symbol: str, signal_type: str, confidence: float, 
                           reasoning: str, position_size: float):
        """Log trading decisions with full context"""
        details = {
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "position_size": position_size,
            "description": f"Trading decision: {signal_type} {symbol} (confidence: {confidence:.2f})"
        }
        return self.log_action("trading_decision", details)
    
    def log_risk_check(self, order_id: str, risk_checks: Dict[str, Any], approved: bool):
        """Log risk management checks"""
        details = {
            "order_id": order_id,
            "risk_checks": risk_checks,
            "approved": approved,
            "description": f"Risk check for order {order_id}: {'APPROVED' if approved else 'REJECTED'}"
        }
        return self.log_action("risk_check", details)
    
    def log_llm_interaction(self, prompt_type: str, model: str, prompt_length: int, 
                           response_length: int, processing_time: float):
        """Log LLM interactions for debugging"""
        details = {
            "prompt_type": prompt_type,
            "model": model,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "processing_time_ms": processing_time * 1000,
            "description": f"LLM {prompt_type} call to {model}"
        }
        return self.log_action("llm_interaction", details)
    
    def log_execution_result(self, order_id: str, execution_report: Dict[str, Any]):
        """Log order execution results"""
        details = {
            "order_id": order_id,
            "execution_report": execution_report,
            "description": f"Order execution: {order_id}"
        }
        return self.log_action("order_execution", details)
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any]):
        """Log errors with full context"""
        details = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context,
            "description": f"Error in {error_type}: {error_message}"
        }
        return self.log_action("error", details, status="failed", error=error_message)
    
    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system performance metrics"""
        details = {
            "metrics": metrics,
            "uptime_seconds": time.time() - self.session_start,
            "description": "System performance metrics"
        }
        return self.log_action("system_metrics", details)
    
    # =============================================================================
    # Advanced Frameworks Logging Methods
    # =============================================================================
    
    def log_bayesian_analysis(self, analysis_type: str, model_params: Dict[str, Any], 
                             mcmc_stats: Dict[str, Any], convergence_info: Dict[str, Any]):
        """Log Bayesian analysis operations"""
        details = {
            "analysis_type": analysis_type,
            "model_params": model_params,
            "mcmc_stats": mcmc_stats,
            "convergence_info": convergence_info,
            "description": f"Bayesian {analysis_type} analysis with MCMC"
        }
        return self.log_action("bayesian_analysis", details)
    
    def log_quantlib_pricing(self, instrument_type: str, pricing_params: Dict[str, Any], 
                            valuation_results: Dict[str, Any], greeks: Optional[Dict[str, Any]] = None):
        """Log QuantLib pricing operations"""
        details = {
            "instrument_type": instrument_type,
            "pricing_params": pricing_params,
            "valuation_results": valuation_results,
            "greeks": greeks or {},
            "description": f"QuantLib pricing for {instrument_type}"
        }
        return self.log_action("quantlib_pricing", details)
    
    def log_portfolio_optimization(self, optimization_type: str, constraints: Dict[str, Any], 
                                  portfolio_metrics: Dict[str, Any], optimization_results: Dict[str, Any]):
        """Log portfolio optimization operations"""
        details = {
            "optimization_type": optimization_type,
            "constraints": constraints,
            "portfolio_metrics": portfolio_metrics,
            "optimization_results": optimization_results,
            "description": f"Portfolio optimization: {optimization_type}"
        }
        return self.log_action("portfolio_optimization", details)
    
    def log_garch_analysis(self, model_type: str, model_params: Dict[str, Any], 
                          volatility_forecast: Dict[str, Any], diagnostic_tests: Dict[str, Any]):
        """Log GARCH volatility modeling"""
        details = {
            "model_type": model_type,
            "model_params": model_params,
            "volatility_forecast": volatility_forecast,
            "diagnostic_tests": diagnostic_tests,
            "description": f"GARCH volatility analysis: {model_type}"
        }
        return self.log_action("garch_analysis", details)
    
    def log_ml_prediction(self, model_type: str, feature_count: int, prediction_results: Dict[str, Any], 
                         model_performance: Dict[str, Any], ensemble_info: Optional[Dict[str, Any]] = None):
        """Log ML/AI prediction operations"""
        details = {
            "model_type": model_type,
            "feature_count": feature_count,
            "prediction_results": prediction_results,
            "model_performance": model_performance,
            "ensemble_info": ensemble_info or {},
            "description": f"ML prediction using {model_type}"
        }
        return self.log_action("ml_prediction", details)
    
    def log_physics_analysis(self, analysis_type: str, physics_params: Dict[str, Any], 
                           entropy_metrics: Dict[str, Any], complexity_measures: Dict[str, Any]):
        """Log physics-based market analysis"""
        details = {
            "analysis_type": analysis_type,
            "physics_params": physics_params,
            "entropy_metrics": entropy_metrics,
            "complexity_measures": complexity_measures,
            "description": f"Physics analysis: {analysis_type}"
        }
        return self.log_action("physics_analysis", details)
    
    def log_microstructure_analysis(self, analysis_type: str, market_data_summary: Dict[str, Any], 
                                   order_flow_metrics: Dict[str, Any], liquidity_measures: Dict[str, Any]):
        """Log market microstructure analysis"""
        details = {
            "analysis_type": analysis_type,
            "market_data_summary": market_data_summary,
            "order_flow_metrics": order_flow_metrics,
            "liquidity_measures": liquidity_measures,
            "description": f"Microstructure analysis: {analysis_type}"
        }
        return self.log_action("microstructure_analysis", details)
    
    def log_framework_availability(self, framework_status: Dict[str, Any]):
        """Log advanced frameworks availability and configuration"""
        details = {
            "framework_status": framework_status,
            "enabled_count": len(framework_status.get("enabled_frameworks", [])),
            "disabled_count": len(framework_status.get("disabled_frameworks", [])),
            "description": f"Advanced frameworks status: {framework_status.get('enabled_frameworks', [])}"
        }
        return self.log_action("framework_availability", details)
    
    def log_comprehensive_analysis(self, analysis_results: Dict[str, Any], frameworks_used: list, 
                                  execution_time: float, confidence_score: float):
        """Log comprehensive multi-framework analysis results"""
        details = {
            "analysis_results": analysis_results,
            "frameworks_used": frameworks_used,
            "execution_time_ms": execution_time * 1000,
            "confidence_score": confidence_score,
            "frameworks_count": len(frameworks_used),
            "description": f"Comprehensive analysis using {len(frameworks_used)} frameworks"
        }
        return self.log_action("comprehensive_analysis", details)
    
    def info(self, message: str):
        """Standard info logging"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Standard warning logging"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Standard error logging"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Standard debug logging"""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Standard critical logging"""
        self.logger.critical(message)


# Global logger instances (similar to GCP agent pattern)
_loggers = {}

def get_logger(name: str, log_level: str = None) -> AgentTradingLogger:
    """
    Get or create a logger instance
    Similar to GCP agent logging pattern
    """
    if name not in _loggers:
        if log_level is None:
            log_level = os.getenv("LOG_LEVEL", "INFO")
        _loggers[name] = AgentTradingLogger(name, log_level)
    return _loggers[name]

# Pre-configure common loggers
def setup_agent_logging():
    """Setup all agent loggers with consistent configuration"""
    
    # Main system logger
    main_logger = get_logger("AgentTradingSystem")
    
    # Component-specific loggers
    data_logger = get_logger("DataPipeline")
    crewai_logger = get_logger("CrewAIIntelligence") 
    adk_logger = get_logger("ADKExecution")
    risk_logger = get_logger("RiskManagement")
    quant_logger = get_logger("QuantitativeModels")
    
    # Advanced frameworks loggers
    bayesian_logger = get_logger("BayesianFramework")
    quantlib_logger = get_logger("QuantLibFramework")
    portfolio_logger = get_logger("PortfolioFramework")
    timeseries_logger = get_logger("TimeSeriesFramework")
    ml_logger = get_logger("MLFramework")
    physics_logger = get_logger("PhysicsFramework")
    microstructure_logger = get_logger("MicrostructureFramework")
    
    main_logger.info("Agent logging system initialized")
    main_logger.info(f"Active loggers: {list(_loggers.keys())}")
    
    # Log framework availability
    try:
        from .environment import validate_environment
        env_status = validate_environment()
        if "advanced_frameworks_status" in env_status:
            main_logger.log_framework_availability(env_status["advanced_frameworks_status"])
    except Exception as e:
        main_logger.warning(f"Could not log framework availability: {e}")
    
    return {
        "main": main_logger,
        "data": data_logger,
        "crewai": crewai_logger,
        "adk": adk_logger,
        "risk": risk_logger,
        "quant": quant_logger,
        "bayesian": bayesian_logger,
        "quantlib": quantlib_logger,
        "portfolio": portfolio_logger,
        "timeseries": timeseries_logger,
        "ml": ml_logger,
        "physics": physics_logger,
        "microstructure": microstructure_logger
    }

# Export for easy imports
__all__ = ['AgentTradingLogger', 'get_logger', 'setup_agent_logging']
