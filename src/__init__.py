"""
Agent Trading System
Hybrid CrewAI Intelligence + ADK Execution Architecture

This package implements a sophisticated quantitative trading system that combines:
- CrewAI for strategic intelligence and probabilistic reasoning
- ADK for deterministic execution and risk management  
- Advanced quantitative models for statistical trading logic
"""

__version__ = "0.1.0"
__author__ = "Agent Trading Team"
__description__ = "Hybrid CrewAI + ADK Quantitative Trading System"

from .config import config
from .main import AgentTradingSystem

# Version info
VERSION = (0, 1, 0)

def get_version():
    """Get version string"""
    return ".".join(str(v) for v in VERSION)

# Validate configuration on import
validation_result = config.validate_config()
if validation_result['errors']:
    import warnings
    warnings.warn(f"Configuration errors detected: {validation_result['errors']}")

__all__ = [
    "AgentTradingSystem",
    "config",
    "get_version"
]
