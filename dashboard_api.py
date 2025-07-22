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
from typing import Dict, Any, List
import threading

# Import the trading system components
try:
    from src.logging_system import get_logger
    from src.main import AgentTradingSystem
    from src.config import ConfigManager
    from src.environment import validate_environment
except ImportError as e:
    print(f"Warning: Could not import trading system components: {e}")
    print("Running in standalone mode with file-based data only")

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
        
        # Try to initialize trading system
        self._initialize_trading_system()
    
    def _initialize_trading_system(self):
        """Initialize connection to trading system"""
        try:
            # Validate environment first
            validate_environment()
            
            # Initialize config
            self.config = ConfigManager()
            
            # Create trading system instance (but don't start it)
            self.trading_system = AgentTradingSystem(self.config)
            
            self.system_health = "CONNECTED"
            print("✅ Connected to Agent Trading System")
            
        except Exception as e:
            print(f"⚠️ Could not connect to trading system: {e}")
            self.system_health = "DISCONNECTED"
    
    def get_real_btc_price(self) -> Dict[str, Any]:
        """Get real BTC price from the trading system"""
        try:
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
        """Get latest real trading decision"""
        try:
            if self.trading_system and hasattr(self.trading_system, 'latest_decision'):
                decision = self.trading_system.latest_decision
                
                if decision:
                    return {
                        "currentAction": decision.get('action', 'HOLD'),
                        "confidence": decision.get('confidence', 0),
                        "lastDecisionTime": decision.get('timestamp', datetime.now().isoformat()),
                        "reasoning": decision.get('reasoning', 'AI analysis completed'),
                        "source": "live_trading_system"
                    }
            
            return None
        except Exception as e:
            print(f"Error getting real decision: {e}")
            return None
    
    def get_real_performance(self) -> Dict[str, Any]:
        """Get real performance metrics"""
        try:
            if self.trading_system:
                # Get performance data from trading system
                performance = {
                    "totalCycles": getattr(self.trading_system, 'total_cycles', 0),
                    "performance": getattr(self.trading_system, 'total_return', 0.0),
                    "pnl": getattr(self.trading_system, 'total_pnl', 0.0),
                    "winRate": getattr(self.trading_system, 'win_rate', 0),
                    "riskLevel": getattr(self.trading_system, 'current_risk_level', 'LOW'),
                    "var": getattr(self.trading_system, 'current_var', 0),
                    "maxDrawdown": getattr(self.trading_system, 'max_drawdown', 0.0),
                    "sharpeRatio": getattr(self.trading_system, 'sharpe_ratio', 0.0),
                    "source": "live_trading_system"
                }
                
                return performance
            
            return None
        except Exception as e:
            print(f"Error getting real performance: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get real system status"""
        try:
            uptime_seconds = int(time.time() - self.start_time)
            
            # Check if trading system is running
            is_running = False
            if self.trading_system:
                is_running = getattr(self.trading_system, 'is_running', False)
            
            # Determine health status
            if self.system_health == "CONNECTED" and is_running:
                health = "HEALTHY"
            elif self.system_health == "CONNECTED":
                health = "IDLE"
            else:
                health = "DISCONNECTED"
            
            return {
                "systemHealth": health,
                "uptime": uptime_seconds,
                "lastCheck": datetime.now().isoformat(),
                "tradingSystemConnected": self.system_health == "CONNECTED",
                "tradingSystemRunning": is_running,
                "source": "live_trading_system"
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
            "logs": data_provider.get_recent_logs(10)
        }
        return jsonify(dashboard_data)
    except Exception as e:
        print(f"Error getting dashboard data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": int(time.time() - data_provider.start_time)
    })

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
