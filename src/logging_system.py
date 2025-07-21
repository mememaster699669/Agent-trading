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
    
    main_logger.info("Agent logging system initialized")
    main_logger.info(f"Active loggers: {list(_loggers.keys())}")
    
    return {
        "main": main_logger,
        "data": data_logger,
        "crewai": crewai_logger,
        "adk": adk_logger,
        "risk": risk_logger,
        "quant": quant_logger
    }

# Export for easy imports
__all__ = ['AgentTradingLogger', 'get_logger', 'setup_agent_logging']
