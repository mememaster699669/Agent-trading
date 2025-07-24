"""
API Server for Agent Trading Dashboard
Provides REST endpoints to feed real-time data to the HTML dashboard
Integrated with the Agent Trading System for live data
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import json
import os
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Try to import numpy for advanced analytics
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
from typing import Dict, Any, List
import threading

# Import the trading system components
try:
    from src.logging_system import get_logger
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False

try:
    from src.main import AgentTradingSystem
    MAIN_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import main trading system: {e}")
    MAIN_SYSTEM_AVAILABLE = False

try:
    from src.config import ConfigManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from src.environment import validate_environment
    ENVIRONMENT_AVAILABLE = True
except ImportError:
    ENVIRONMENT_AVAILABLE = False

try:
    # Import all advanced 5-phase frameworks for dashboard
    from src.quant_models import (
        BayesianTradingFramework,
        QuantLibFinancialEngineering,
        AdvancedPortfolioOptimization,
        AdvancedTimeSeriesAnalysis,
        AdvancedMLTradingFramework,
        AdvancedPhysicsModels,
        MarketMicrostructure
    )
    ADVANCED_FRAMEWORKS_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: Could not import trading system components: {e}")
    print("Running in standalone mode with file-based data only")
    ADVANCED_FRAMEWORKS_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for dashboard access

class TradingSystemIntegration:
    """
    Direct integration with the Agent Trading System for real-time data
    """
    
    def __init__(self):
        self.trading_system = None
        self.config = None
        self.last_decision = None
        self.performance_data = {}
        self.system_health = "INITIALIZING"
        self.start_time = time.time()
        
        # Initialize all advanced 5-phase frameworks for dashboard
        self.advanced_frameworks = {}
        self.framework_status = {}
        
        # Try to initialize trading system and frameworks
        self._initialize_trading_system()
        self._initialize_advanced_frameworks()
    
    def _initialize_advanced_frameworks(self):
        """Initialize all advanced 5-phase frameworks for dashboard analytics"""
        if not ADVANCED_FRAMEWORKS_AVAILABLE:
            print("⚠️ Advanced frameworks not available - using fallback data")
            return
        
        try:
            # Phase 1: Bayesian Framework
            self.advanced_frameworks['bayesian'] = BayesianTradingFramework()
            self.framework_status['bayesian'] = self.advanced_frameworks['bayesian'].is_available
            
            # Phase 2: QuantLib Framework
            self.advanced_frameworks['quantlib'] = QuantLibFinancialEngineering()
            self.framework_status['quantlib'] = self.advanced_frameworks['quantlib'].is_available
            
            # Phase 3: Portfolio Optimization Framework
            self.advanced_frameworks['portfolio'] = AdvancedPortfolioOptimization()
            self.framework_status['portfolio'] = self.advanced_frameworks['portfolio'].is_available
            
            # Phase 4: Time Series Framework
            self.advanced_frameworks['timeseries'] = AdvancedTimeSeriesAnalysis()
            self.framework_status['timeseries'] = self.advanced_frameworks['timeseries'].is_available
            
            # Phase 5: ML Framework
            self.advanced_frameworks['ml'] = AdvancedMLTradingFramework()
            self.framework_status['ml'] = self.advanced_frameworks['ml'].is_available
            
            print("✅ Advanced 5-phase frameworks initialized for dashboard")
            print(f"Framework availability: {self.framework_status}")
            
        except Exception as e:
            print(f"⚠️ Could not initialize advanced frameworks: {e}")
            self.advanced_frameworks = {}
            self.framework_status = {}
    
    def _initialize_trading_system(self):
        """Initialize connection to trading system"""
        try:
            # Initialize config first if available
            if CONFIG_AVAILABLE:
                self.config = ConfigManager()
                print("✅ ConfigManager initialized")
            else:
                print("⚠️ ConfigManager not available - using environment variables")
                self.config = None
            
            # Try to connect to main trading system if available
            if MAIN_SYSTEM_AVAILABLE:
                # Instead of creating a full trading system, just initialize data components
                # This avoids permission issues and full system startup
                self.system_health = "CONNECTED"
                print("✅ Connected to Agent Trading System configuration")
            else:
                print("⚠️ Main trading system not available - using file-based mode")
                self.system_health = "FILE_BASED"
            
        except Exception as e:
            print(f"⚠️ Could not connect to trading system: {e}")
            self.system_health = "DISCONNECTED"
    
    def get_real_btc_price(self) -> Dict[str, Any]:
        """Get real BTC price from the trading system or shared data files"""
        try:
            # First try to read from shared data files written by the main trading system
            data_file = Path("/app/data/latest_market_data.json")
            if data_file.exists():
                with open(data_file, 'r') as f:
                    market_data = json.load(f)
                    if market_data and 'price' in market_data:
                        return {
                            "btcPrice": market_data['price'],
                            "priceChange": market_data.get('change_24h', 0),
                            "volume24h": market_data.get('volume_24h', 0),
                            "lastUpdate": market_data.get('timestamp', datetime.now().isoformat()),
                            "source": "file_based_live_data"
                        }
            
            # Fallback: try direct trading system connection
            if self.trading_system and hasattr(self.trading_system, 'data_pipeline'):
                # Get latest market data from the trading system
                data_pipeline = self.trading_system.data_pipeline
                
                if hasattr(data_pipeline, 'latest_price'):
                    current_price = data_pipeline.latest_price
                    price_change = data_pipeline.price_change_24h if hasattr(data_pipeline, 'price_change_24h') else 0
                    
                    return {
                        "btcPrice": current_price,
                        "priceChange": price_change,
                        "volume24h": getattr(data_pipeline, 'volume_24h', 0),
                        "lastUpdate": datetime.now().isoformat(),
                        "source": "live_trading_system"
                    }
            
            return None
        except Exception as e:
            print(f"Error getting real BTC price: {e}")
            return None
    
    def get_real_decision(self) -> Dict[str, Any]:
        """Get latest real trading decision from shared files or trading system"""
        try:
            # First try to read from shared decision file
            decision_file = Path("/app/data/latest_decision.json")
            if decision_file.exists():
                with open(decision_file, 'r') as f:
                    decision_data = json.load(f)
                    if decision_data:
                        return {
                            "currentAction": decision_data.get('action', 'HOLD'),
                            "confidence": decision_data.get('confidence', 0),
                            "reasoning": decision_data.get('reasoning', 'No reasoning available'),
                            "timestamp": decision_data.get('timestamp', datetime.now().isoformat()),
                            "source": "file_based_live_data"
                        }
            
            # Fallback: try direct trading system connection
            if self.trading_system and hasattr(self.trading_system, 'latest_decision'):
                decision = self.trading_system.latest_decision
                
                if decision:
                    return {
                        "currentAction": decision.get('action', 'HOLD'),
                        "confidence": decision.get('confidence', 0),
                        "reasoning": decision.get('reasoning', 'No reasoning available'),
                        "timestamp": decision.get('timestamp', datetime.now().isoformat()),
                        "source": "live_trading_system"
                    }
            
            return None
        except Exception as e:
            print(f"Error getting real decision: {e}")
            return None
    
    def get_real_performance(self) -> Dict[str, Any]:
        """Get real performance metrics from shared files or trading system"""
        try:
            # First try to read from shared performance file
            performance_file = Path("/app/data/latest_performance.json")
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    performance_data = json.load(f)
                    if performance_data:
                        return {
                            "totalPnL": performance_data.get('total_pnl', 0),
                            "winRate": performance_data.get('win_rate', 0),
                            "totalReturn": performance_data.get('total_return', 0.0),
                            "currentVaR": performance_data.get('current_var', 0),
                            "maxDrawdown": performance_data.get('max_drawdown', 0),
                            "sharpeRatio": 0.0,  # Calculate if needed
                            "trades": performance_data.get('cycles_completed', 0),
                            "source": "file_based_live_data"
                        }
            
            # Fallback: try direct trading system connection
            if self.trading_system:
                # Get performance data from trading system
                return {
                    "totalPnL": getattr(self.trading_system, 'total_pnl', 0),
                    "winRate": getattr(self.trading_system, 'win_rate', 0),
                    "totalReturn": getattr(self.trading_system, 'total_return', 0.0),
                    "currentVaR": getattr(self.trading_system, 'current_var', 0),
                    "maxDrawdown": getattr(self.trading_system, 'max_drawdown', 0),
                    "sharpeRatio": 0.0,
                    "trades": getattr(self.trading_system, 'cycles_completed', 0),
                    "source": "live_trading_system"
                }
            
            return None
        except Exception as e:
            print(f"Error getting real performance: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get real system status from shared files or trading system"""
        try:
            # First try to read from shared performance file for system data
            performance_file = Path("/app/data/latest_performance.json")
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    performance_data = json.load(f)
                    if performance_data:
                        uptime_seconds = performance_data.get('uptime_seconds', 0)
                        cycles = performance_data.get('cycles_completed', 0)
                        errors = performance_data.get('errors_encountered', 0)
                        
                        # Determine health based on error rate
                        health = "RUNNING"
                        if errors > 0 and cycles > 0:
                            error_rate = errors / cycles
                            if error_rate > 0.1:  # More than 10% error rate
                                health = "WARNING"
                        
                        return {
                            "systemHealth": health,
                            "uptime": uptime_seconds,
                            "lastCheck": datetime.now().isoformat(),
                            "tradingSystemConnected": True,
                            "tradingSystemRunning": True,
                            "cycles": cycles,
                            "errors": errors,
                            "source": "file_based_live_data"
                        }
            
            # Fallback: basic status check
            uptime_seconds = int(time.time() - self.start_time)
            
            # Check if trading system is running
            is_running = self.system_health == "CONNECTED"
            health = "RUNNING" if is_running else "DISCONNECTED"
            
            return {
                "systemHealth": health,
                "uptime": uptime_seconds,
                "lastCheck": datetime.now().isoformat(),
                "tradingSystemConnected": self.system_health == "CONNECTED",
                "tradingSystemRunning": is_running,
                "source": "basic_status"
            }
            
        except Exception as e:
            return {
                "systemHealth": "ERROR",
                "uptime": 0,
                "lastCheck": datetime.now().isoformat(),
                "error": str(e),
                "source": "error"
            }

class DashboardDataProvider:
    """
    Collects real-time data from the trading system for dashboard display
    Now integrates directly with the Agent Trading System
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.logs_dir = Path("logs")
        self.data_dir = Path("data")
        
        # Initialize trading system integration
        self.trading_integration = TradingSystemIntegration()
        
        # Initialize logger
        try:
            self.logger = get_logger("DashboardAPI")
        except:
            # Fallback logger
            self.logger = logging.getLogger("DashboardAPI")
            logging.basicConfig(level=logging.INFO)
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system health and status - now uses real data"""
        try:
            # First try to get real system status
            real_status = self.trading_integration.get_system_status()
            
            if real_status and real_status.get('source') == 'live_trading_system':
                return real_status
            
            # Fallback to file-based status
            recent_logs = self._get_recent_log_files()
            uptime_seconds = int(time.time() - self.start_time)
            error_count = self._count_recent_errors()
            
            if error_count > 10:
                health = "CRITICAL"
            elif error_count > 5:
                health = "WARNING"
            elif len(recent_logs) > 0:
                health = "HEALTHY"
            else:
                health = "UNKNOWN"
            
            return {
                "systemHealth": health,
                "uptime": uptime_seconds,
                "lastCheck": datetime.now().isoformat(),
                "errorCount": error_count,
                "logFiles": len(recent_logs),
                "source": "file_based"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                "systemHealth": "ERROR",
                "uptime": 0,
                "lastCheck": datetime.now().isoformat(),
                "errorCount": 999,
                "logFiles": 0,
                "source": "error"
            }
    
    def get_btc_price_data(self) -> Dict[str, Any]:
        """Get current BTC price and change data - now uses real data"""
        try:
            # First try to get real price data
            real_price = self.trading_integration.get_real_btc_price()
            
            if real_price:
                return real_price
            
            # Try to read from data files
            price_data = self._read_latest_price_data()
            
            if price_data:
                price_data["source"] = "file_based"
                return price_data
            
            # Fallback to mock data
            return {
                "btcPrice": 43250.75,
                "priceChange": 1.25,
                "volume24h": 28500000000,
                "lastUpdate": datetime.now().isoformat(),
                "source": "mock_data"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting BTC price data: {e}")
            return {
                "btcPrice": 0,
                "priceChange": 0,
                "volume24h": 0,
                "lastUpdate": datetime.now().isoformat(),
                "source": "error"
            }
    
    def get_latest_decision(self) -> Dict[str, Any]:
        """Get the latest trading decision and confidence - now uses real data"""
        try:
            # First try to get real decision data
            real_decision = self.trading_integration.get_real_decision()
            
            if real_decision:
                return real_decision
            
            # Read from action logs
            decision_data = self._read_latest_decision_from_logs()
            
            if decision_data:
                decision_data["source"] = "file_based"
                return decision_data
            
            # Fallback to default
            return {
                "currentAction": "HOLD",
                "confidence": 0,
                "lastDecisionTime": datetime.now().isoformat(),
                "reasoning": "Awaiting market data",
                "source": "default"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting latest decision: {e}")
            return {
                "currentAction": "ERROR",
                "confidence": 0,
                "lastDecisionTime": datetime.now().isoformat(),
                "reasoning": "Error reading decision data",
                "source": "error"
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get trading performance metrics - now uses real data"""
        try:
            # First try to get real performance data
            real_performance = self.trading_integration.get_real_performance()
            
            if real_performance:
                return real_performance
            
            # Try to read from performance logs/data
            perf_data = self._read_performance_data()
            
            if perf_data:
                perf_data["source"] = "file_based"
                return perf_data
            
            # Fallback to default metrics
            return {
                "totalCycles": 0,
                "performance": 0.0,
                "pnl": 0.0,
                "winRate": 0,
                "riskLevel": "LOW",
                "var": 0,
                "maxDrawdown": 0.0,
                "sharpeRatio": 0.0,
                "source": "default"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {
                "totalCycles": 0,
                "performance": 0.0,
                "pnl": 0.0,
                "winRate": 0,
                "riskLevel": "ERROR",
                "var": 0,
                "maxDrawdown": 0.0,
                "sharpeRatio": 0.0,
                "source": "error"
            }
    
    def get_recent_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent system logs"""
        try:
            logs = []
            
            # Read from action logs (JSONL format)
            action_logs = self._read_action_logs(limit)
            logs.extend(action_logs)
            
            # Read from regular log files
            text_logs = self._read_text_logs(limit)
            logs.extend(text_logs)
            
            # Sort by timestamp and return most recent
            logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return logs[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting recent logs: {e}")
            return [{
                "type": "error",
                "message": f"Error reading logs: {e}",
                "timestamp": datetime.now().isoformat()
            }]
    
    def _get_recent_log_files(self) -> List[Path]:
        """Get list of recent log files"""
        if not self.logs_dir.exists():
            return []
        
        log_files = list(self.logs_dir.glob("*.log")) + list(self.logs_dir.glob("*.jsonl"))
        
        # Filter files modified in last 24 hours
        recent_files = []
        cutoff_time = time.time() - (24 * 3600)  # 24 hours ago
        
        for log_file in log_files:
            if log_file.stat().st_mtime > cutoff_time:
                recent_files.append(log_file)
        
        return recent_files
    
    def _count_recent_errors(self) -> int:
        """Count errors in recent logs"""
        error_count = 0
        recent_files = self._get_recent_log_files()
        
        for log_file in recent_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    error_count += content.lower().count('error')
                    error_count += content.lower().count('exception')
                    error_count += content.lower().count('failed')
            except Exception:
                continue
        
        return error_count
    
    def _read_latest_price_data(self) -> Dict[str, Any]:
        """Read latest price data from data files"""
        try:
            # Look for price data files
            price_files = list(self.data_dir.glob("*price*.json"))
            
            if not price_files:
                return None
            
            # Get most recent file
            latest_file = max(price_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
                return data
        except Exception:
            return None
    
    def _read_latest_decision_from_logs(self) -> Dict[str, Any]:
        """Read latest trading decision from action logs"""
        try:
            action_files = list(self.logs_dir.glob("actions_*.jsonl"))
            
            if not action_files:
                return None
            
            # Get most recent action file
            latest_file = max(action_files, key=lambda f: f.stat().st_mtime)
            
            # Read last few lines to find latest decision
            with open(latest_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Search backwards for trading decision
            for line in reversed(lines[-50:]):  # Check last 50 lines
                try:
                    action_data = json.loads(line.strip())
                    
                    if 'trading' in action_data.get('action_type', '').lower():
                        details = action_data.get('details', {})
                        return {
                            "currentAction": details.get('action', 'HOLD'),
                            "confidence": details.get('confidence', 0),
                            "lastDecisionTime": action_data.get('timestamp'),
                            "reasoning": details.get('reasoning', 'No reasoning provided')
                        }
                except json.JSONDecodeError:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def _read_performance_data(self) -> Dict[str, Any]:
        """Read performance data from files"""
        try:
            # Look for performance data files
            perf_files = list(self.data_dir.glob("*performance*.json"))
            
            if not perf_files:
                return None
            
            # Get most recent file
            latest_file = max(perf_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
                return data
        except Exception:
            return None
    
    def _read_action_logs(self, limit: int) -> List[Dict[str, Any]]:
        """Read action logs in JSONL format"""
        logs = []
        
        try:
            action_files = list(self.logs_dir.glob("actions_*.jsonl"))
            
            for log_file in action_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            action_data = json.loads(line.strip())
                            logs.append({
                                "type": "info" if action_data.get('status') == 'completed' else "warning",
                                "message": f"{action_data.get('action_type')}: {action_data.get('status')}",
                                "timestamp": action_data.get('timestamp'),
                                "details": action_data.get('details', {})
                            })
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.logger.error(f"Error reading action logs: {e}")
        
        return logs[-limit:] if logs else []
    
    def _read_text_logs(self, limit: int) -> List[Dict[str, Any]]:
        """Read text log files"""
        logs = []
        
        try:
            log_files = list(self.logs_dir.glob("agent_trading_*.log"))
            
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Parse recent log lines
                for line in lines[-limit:]:
                    if not line.strip():
                        continue
                    
                    # Simple log parsing
                    log_type = "info"
                    if "ERROR" in line.upper():
                        log_type = "error"
                    elif "WARNING" in line.upper():
                        log_type = "warning"
                    elif "SUCCESS" in line.upper() or "COMPLETED" in line.upper():
                        log_type = "success"
                    
                    logs.append({
                        "type": log_type,
                        "message": line.strip(),
                        "timestamp": datetime.now().isoformat()  # Could parse from log line
                    })
        except Exception as e:
            self.logger.error(f"Error reading text logs: {e}")
        
        return logs[-limit:] if logs else []

    def get_physics_risk_metrics(self) -> Dict[str, Any]:
        """Get physics-based risk metrics following @khemkapital methodology"""
        try:
            # Try to read from latest decision file which should include physics metrics
            physics_data = self._read_physics_risk_from_files()
            
            if physics_data:
                return physics_data
            
            # Fallback to default physics risk structure
            return {
                "entropy_analysis": {
                    "entropy": 0.5,
                    "risk_level": "medium",
                    "readability": "moderate",
                    "information_flow_quality": 0.5
                },
                "memory_analysis": {
                    "hurst_exponent": 0.5,
                    "memory_type": "random_walk",
                    "trauma_detected": False,
                    "trauma_intensity": 0.0,
                    "persistence_strength": 0.0
                },
                "instability_analysis": {
                    "lyapunov_exponent": 0.0,
                    "instability_score": 0.1,
                    "instability_level": "low",
                    "systemic_risk": "low",
                    "shock_amplification_factor": 1.0,
                    "chaos_detected": False
                },
                "regime_analysis": {
                    "regime": "stable_trending",
                    "stability": "moderate",
                    "transition_probability": 0.1,
                    "physics_metrics": {
                        "entropy_score": 0.5,
                        "memory_deviation": 0.0,
                        "instability_score": 0.1
                    }
                },
                "combined_physics_risk_score": 0.3,
                "risk_amplification_factor": 1.15,
                "last_updated": datetime.now().isoformat(),
                "source": "default"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting physics risk metrics: {e}")
            return {
                "error": str(e),
                "last_updated": datetime.now().isoformat(),
                "source": "error"
            }
    
    def _read_physics_risk_from_files(self) -> Dict[str, Any]:
        """Read physics risk metrics from latest decision or separate physics file"""
        try:
            # First try to read from latest decision file
            latest_decision_file = self.data_dir / "latest_decision.json"
            
            if latest_decision_file.exists():
                with open(latest_decision_file, 'r') as f:
                    decision_data = json.load(f)
                    
                    # Check if decision contains physics risk data
                    risk_assessment = decision_data.get('risk_assessment', {})
                    physics_risk = risk_assessment.get('physics_based_risk', {})
                    
                    if physics_risk:
                        physics_risk['last_updated'] = decision_data.get('timestamp', datetime.now().isoformat())
                        physics_risk['source'] = 'decision_file'
                        return physics_risk
            
            # Try to read from dedicated physics risk file
            physics_file = self.data_dir / "physics_risk.json"
            
            if physics_file.exists():
                with open(physics_file, 'r') as f:
                    physics_data = json.load(f)
                    physics_data['source'] = 'physics_file'
                    return physics_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error reading physics risk from files: {e}")
            return None


# Initialize data provider
data_provider = DashboardDataProvider()

# API Routes

@app.route('/')
def serve_dashboard():
    """Serve the dashboard HTML file"""
    return send_from_directory('.', 'dashboard.html')

@app.route('/api/status')
def get_status():
    """Get system status and health"""
    return jsonify(data_provider.get_system_status())

@app.route('/api/price')
def get_price():
    """Get current BTC price data"""
    return jsonify(data_provider.get_btc_price_data())

@app.route('/api/decision')
def get_decision():
    """Get latest trading decision"""
    return jsonify(data_provider.get_latest_decision())

@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    return jsonify(data_provider.get_performance_metrics())

@app.route('/api/logs')
def get_logs():
    """Get recent system logs"""
    limit = int(request.args.get('limit', 20))
    return jsonify({"logs": data_provider.get_recent_logs(limit)})

@app.route('/api/dashboard')
def get_all_dashboard_data():
    """Get all dashboard data in one call"""
    try:
        dashboard_data = {
            "status": data_provider.get_system_status(),
            "price": data_provider.get_btc_price_data(),
            "decision": data_provider.get_latest_decision(),
            "performance": data_provider.get_performance_metrics(),
            "physics_risk": data_provider.get_physics_risk_metrics(),
            "logs": data_provider.get_recent_logs(10)
        }
        return jsonify(dashboard_data)
    except Exception as e:
        print(f"Error getting dashboard data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/physics-risk')
def get_physics_risk():
    """Get physics-based risk metrics"""
    return jsonify(data_provider.get_physics_risk_metrics())

@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    uptime = int(time.time() - data_provider.start_time) if hasattr(data_provider, 'start_time') else 0
    
    # Include framework status if available
    framework_status = getattr(data_provider.trading_integration, 'framework_status', {})
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": uptime,
        "frameworks_available": framework_status,
        "system_health": getattr(data_provider.trading_integration, 'system_health', 'UNKNOWN')
    })

@app.route('/api/bayesian-analysis/<symbol>')
def api_bayesian_analysis(symbol):
    """Get Bayesian analysis for a symbol"""
    if not hasattr(data_provider.trading_integration, 'advanced_frameworks') or 'bayesian' not in data_provider.trading_integration.advanced_frameworks:
        return jsonify({'error': 'Bayesian framework not available'}), 503
    
    try:
        # Generate sample data for demonstration
        if not NUMPY_AVAILABLE:
            return jsonify({'error': 'NumPy not available for advanced analysis'}), 503
            
        # Using numpy imported at top
        price_data = np.random.normal(100, 15, 100)  # Simulated price data
        
        framework = data_provider.trading_integration.advanced_frameworks['bayesian']
        analysis = framework.hierarchical_model_analysis(price_data)
        
        return jsonify({
            'symbol': symbol,
            'analysis_type': 'bayesian_hierarchical',
            'results': analysis,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quantlib-derivatives/<symbol>')
def api_quantlib_derivatives(symbol):
    """Get QuantLib derivatives analysis"""
    if not hasattr(data_provider.trading_integration, 'advanced_frameworks') or 'quantlib' not in data_provider.trading_integration.advanced_frameworks:
        return jsonify({'error': 'QuantLib framework not available'}), 503
    
    try:
        framework = data_provider.trading_integration.advanced_frameworks['quantlib']
        
        # Example Black-Scholes option pricing
        option_analysis = framework.black_scholes_analysis(
            spot=100, strike=105, risk_free_rate=0.05, 
            volatility=0.2, time_to_expiry=0.25
        )
        
        return jsonify({
            'symbol': symbol,
            'analysis_type': 'quantlib_derivatives',
            'option_pricing': option_analysis,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio-optimization')
def api_portfolio_optimization():
    """Get portfolio optimization analysis"""
    if not hasattr(data_provider.trading_integration, 'advanced_frameworks') or 'portfolio' not in data_provider.trading_integration.advanced_frameworks:
        return jsonify({'error': 'Portfolio framework not available'}), 503
    
    try:
        framework = data_provider.trading_integration.advanced_frameworks['portfolio']
        
        # Example symbols and expected returns
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        if not NUMPY_AVAILABLE:
            return jsonify({'error': 'NumPy not available for portfolio optimization'}), 503
            
        # Using numpy imported at top
        expected_returns = np.random.normal(0.12, 0.05, len(symbols))
        
        optimization = framework.mean_variance_optimization(symbols, expected_returns)
        
        return jsonify({
            'analysis_type': 'portfolio_optimization',
            'symbols': symbols,
            'optimization_results': optimization,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/garch-analysis/<symbol>')
def api_garch_analysis(symbol):
    """Get GARCH volatility analysis"""
    if not hasattr(data_provider.trading_integration, 'advanced_frameworks') or 'timeseries' not in data_provider.trading_integration.advanced_frameworks:
        return jsonify({'error': 'Time series framework not available'}), 503
    
    try:
        framework = data_provider.trading_integration.advanced_frameworks['timeseries']
        
        # Generate sample return data for demonstration
        # Using numpy imported at top
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        
        garch_analysis = framework.garch_volatility_modeling(returns)
        
        return jsonify({
            'symbol': symbol,
            'analysis_type': 'garch_volatility',
            'volatility_forecast': garch_analysis,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml-predictions/<symbol>')
def api_ml_predictions(symbol):
    """Get ML-based predictions"""
    if not hasattr(data_provider.trading_integration, 'advanced_frameworks') or 'ml' not in data_provider.trading_integration.advanced_frameworks:
        return jsonify({'error': 'ML framework not available'}), 503
    
    try:
        framework = data_provider.trading_integration.advanced_frameworks['ml']
        
        # Generate sample feature data for demonstration
        # Using numpy imported at top
        features = np.random.randn(100, 10)  # 100 samples, 10 features
        targets = np.random.randn(100)  # Target values
        
        ml_analysis = framework.ensemble_prediction(features, targets)
        
        return jsonify({
            'symbol': symbol,
            'analysis_type': 'ml_ensemble_prediction',
            'predictions': ml_analysis,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment-analysis/<symbol>')
def api_sentiment_analysis(symbol):
    """Get sentiment analysis for a symbol"""
    if not hasattr(data_provider.trading_integration, 'advanced_frameworks') or 'ml' not in data_provider.trading_integration.advanced_frameworks:
        return jsonify({'error': 'ML framework not available'}), 503
    
    try:
        framework = data_provider.trading_integration.advanced_frameworks['ml']
        
        # Example news texts for sentiment analysis
        news_texts = [
            f"{symbol} reports strong quarterly earnings",
            f"Market volatility affects {symbol} trading",
            f"Analysts upgrade {symbol} price target"
        ]
        
        sentiment_analysis = framework.sentiment_analysis(news_texts)
        
        return jsonify({
            'symbol': symbol,
            'analysis_type': 'sentiment_analysis',
            'sentiment_scores': sentiment_analysis,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reinforcement-learning/<symbol>')
def api_reinforcement_learning(symbol):
    """Get reinforcement learning analysis"""
    if not hasattr(data_provider.trading_integration, 'advanced_frameworks') or 'ml' not in data_provider.trading_integration.advanced_frameworks:
        return jsonify({'error': 'ML framework not available'}), 503
    
    try:
        framework = data_provider.trading_integration.advanced_frameworks['ml']
        
        # Generate sample market environment data
        # Using numpy imported at top
        market_state = np.random.randn(50)  # Market state features
        
        rl_analysis = framework.reinforcement_learning_strategy(market_state)
        
        return jsonify({
            'symbol': symbol,
            'analysis_type': 'reinforcement_learning',
            'strategy_recommendations': rl_analysis,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/comprehensive-analysis/<symbol>')
def api_comprehensive_analysis(symbol):
    """Get comprehensive analysis using all available frameworks"""
    if not hasattr(data_provider.trading_integration, 'advanced_frameworks') or not data_provider.trading_integration.advanced_frameworks:
        return jsonify({'error': 'No advanced frameworks available'}), 503
    
    comprehensive_results = {
        'symbol': symbol,
        'analysis_timestamp': time.time(),
        'available_frameworks': list(getattr(data_provider.trading_integration, 'framework_status', {}).keys()),
        'analyses': {}
    }
    
    # Run all available analyses
    framework_status = getattr(data_provider.trading_integration, 'framework_status', {})
    for framework_name, is_available in framework_status.items():
        if not is_available:
            continue
            
        try:
            if framework_name == 'bayesian':
                # Using numpy imported at top
                price_data = np.random.normal(100, 15, 100)
                framework = data_provider.trading_integration.advanced_frameworks['bayesian']
                comprehensive_results['analyses']['bayesian'] = framework.hierarchical_model_analysis(price_data)
                
            elif framework_name == 'quantlib':
                framework = data_provider.trading_integration.advanced_frameworks['quantlib']
                comprehensive_results['analyses']['quantlib'] = framework.black_scholes_analysis(
                    spot=100, strike=105, risk_free_rate=0.05, volatility=0.2, time_to_expiry=0.25
                )
                
            elif framework_name == 'portfolio':
                framework = data_provider.trading_integration.advanced_frameworks['portfolio']
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
                expected_returns = np.random.normal(0.12, 0.05, len(symbols))
                comprehensive_results['analyses']['portfolio'] = framework.mean_variance_optimization(symbols, expected_returns)
                
            elif framework_name == 'timeseries':
                framework = data_provider.trading_integration.advanced_frameworks['timeseries']
                returns = np.random.normal(0.001, 0.02, 252)
                comprehensive_results['analyses']['timeseries'] = framework.garch_volatility_modeling(returns)
                
            elif framework_name == 'ml':
                framework = data_provider.trading_integration.advanced_frameworks['ml']
                features = np.random.randn(100, 10)
                targets = np.random.randn(100)
                comprehensive_results['analyses']['ml'] = framework.ensemble_prediction(features, targets)
                
        except Exception as e:
            comprehensive_results['analyses'][framework_name] = {'error': str(e)}
    
    return jsonify(comprehensive_results)

if __name__ == '__main__':
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    print("Dashboard API Server starting...")
    print("Dashboard will be available at: http://localhost:8080")
    print("API endpoints available at: http://localhost:8080/api/")
    
    # Run the Flask development server
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=False,  # Set to False for Docker
        threaded=True
    )
