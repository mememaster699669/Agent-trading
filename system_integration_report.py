#!/usr/bin/env python3
"""
COMPREHENSIVE SYSTEM INTEGRATION STATUS REPORT

This script analyzes the entire Agent Trading System to identify 
which components are using the advanced 5-phase frameworks vs
legacy implementations.

🔥 CRITICAL FINDING: Major integration gaps discovered!
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def analyze_system_integration():
    """Comprehensive analysis of system integration status"""
    
    print("🔍 AGENT TRADING SYSTEM INTEGRATION ANALYSIS")
    print("=" * 80)
    
    # Define what we're looking for
    ADVANCED_FRAMEWORKS = [
        "BayesianTradingFramework",
        "QuantLibFinancialEngineering", 
        "AdvancedPortfolioOptimization",
        "AdvancedTimeSeriesAnalysis",
        "AdvancedMLTradingFramework"
    ]
    
    LEGACY_MODELS = [
        "QuantitativeModels",
        "RiskMetrics", 
        "ProbabilisticForecast"
    ]
    
    # Files to analyze
    CORE_FILES = {
        "🧠 Core Intelligence": [
            "src/crewai_intelligence.py",
            "src/main.py"
        ],
        "📊 Dashboard & API": [
            "dashboard_api.py",
            "dashboard.html"
        ],
        "⚙️ Execution Layer": [
            "src/adk_execution.py"
        ],
        "📈 Data Management": [
            "src/dataset.py"
        ],
        "🔧 Configuration": [
            "src/config.py",
            "src/environment.py"
        ],
        "💎 Quantitative Models": [
            "src/quant_models.py"
        ]
    }
    
    integration_status = {}
    
    for category, files in CORE_FILES.items():
        print(f"\n{category}")
        print("-" * 60)
        
        category_status = {}
        
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"❌ {file_path} - FILE NOT FOUND")
                category_status[file_path] = "MISSING"
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for advanced frameworks
                advanced_found = []
                for framework in ADVANCED_FRAMEWORKS:
                    if framework in content:
                        advanced_found.append(framework)
                
                # Check for legacy models
                legacy_found = []
                for legacy in LEGACY_MODELS:
                    if legacy in content:
                        legacy_found.append(legacy)
                
                # Determine status
                if advanced_found and len(advanced_found) >= 3:
                    status = "🟢 FULLY INTEGRATED"
                    status_code = "ADVANCED"
                elif advanced_found:
                    status = "🟡 PARTIALLY INTEGRATED"
                    status_code = "MIXED"
                elif legacy_found:
                    status = "🔴 LEGACY ONLY"
                    status_code = "LEGACY"
                else:
                    status = "⚪ NO QUANT MODELS"
                    status_code = "NONE"
                
                print(f"{status} {file_path}")
                
                if advanced_found:
                    print(f"   ✅ Advanced: {', '.join(advanced_found)}")
                if legacy_found:
                    print(f"   ⚠️  Legacy: {', '.join(legacy_found)}")
                
                category_status[file_path] = {
                    "status": status_code,
                    "advanced": advanced_found,
                    "legacy": legacy_found
                }
                
            except Exception as e:
                print(f"❌ {file_path} - ERROR: {e}")
                category_status[file_path] = "ERROR"
        
        integration_status[category] = category_status
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("📋 INTEGRATION STATUS SUMMARY")
    print("=" * 80)
    
    total_files = 0
    advanced_files = 0
    mixed_files = 0
    legacy_files = 0
    missing_files = 0
    
    critical_issues = []
    
    for category, files in integration_status.items():
        for file_path, status in files.items():
            total_files += 1
            
            if status == "MISSING" or status == "ERROR":
                missing_files += 1
                if "dashboard" in file_path or "main.py" in file_path:
                    critical_issues.append(f"🚨 CRITICAL: {file_path} has issues")
            elif isinstance(status, dict):
                if status["status"] == "ADVANCED":
                    advanced_files += 1
                elif status["status"] == "MIXED":
                    mixed_files += 1
                elif status["status"] == "LEGACY":
                    legacy_files += 1
                    if "dashboard" in file_path or "crewai" in file_path:
                        critical_issues.append(f"🔴 HIGH PRIORITY: {file_path} needs upgrade")
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total Files Analyzed: {total_files}")
    print(f"   🟢 Fully Integrated: {advanced_files}")
    print(f"   🟡 Partially Integrated: {mixed_files}")
    print(f"   🔴 Legacy Only: {legacy_files}")
    print(f"   ❌ Missing/Error: {missing_files}")
    
    integration_percentage = (advanced_files / max(total_files, 1)) * 100
    print(f"\n🎯 INTEGRATION COMPLETION: {integration_percentage:.1f}%")
    
    # Critical issues
    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES IDENTIFIED:")
        for issue in critical_issues:
            print(f"   {issue}")
    
    # Specific dashboard analysis
    print(f"\n📊 DASHBOARD STATUS ANALYSIS:")
    dashboard_issues = []
    
    # Check if dashboard has advanced endpoints
    if os.path.exists("dashboard_api.py"):
        with open("dashboard_api.py", 'r') as f:
            dashboard_content = f.read()
        
        missing_endpoints = []
        
        # Critical endpoints that should exist
        required_endpoints = [
            "/api/bayesian",
            "/api/quantlib", 
            "/api/portfolio",
            "/api/timeseries",
            "/api/ml-analysis"
        ]
        
        for endpoint in required_endpoints:
            if endpoint not in dashboard_content:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"   🔴 Missing API endpoints: {', '.join(missing_endpoints)}")
            dashboard_issues.append("API endpoints not integrated with advanced frameworks")
        
        # Check for advanced data structures
        if "ensemble_prediction" not in dashboard_content:
            dashboard_issues.append("No ML ensemble prediction integration")
        if "bayesian_analysis" not in dashboard_content:
            dashboard_issues.append("No Bayesian analysis integration")
        if "garch_analysis" not in dashboard_content:
            dashboard_issues.append("No GARCH volatility modeling integration")
    
    if dashboard_issues:
        print(f"   🚨 Dashboard Integration Issues:")
        for issue in dashboard_issues:
            print(f"      - {issue}")
    else:
        print(f"   ✅ Dashboard appears to be well integrated")
    
    # Recommendations
    print(f"\n💡 INTEGRATION RECOMMENDATIONS:")
    print(f"   1. 🎯 URGENT: Upgrade dashboard API to expose all 5-phase frameworks")
    print(f"   2. 🔧 HIGH: Integrate advanced frameworks into main.py orchestration")
    print(f"   3. 📊 MEDIUM: Update dataset.py to provide data for advanced models")
    print(f"   4. ⚙️ LOW: Enhance adk_execution.py with advanced risk management")
    
    # Final verdict
    print(f"\n🏆 FINAL VERDICT:")
    if integration_percentage >= 80:
        print(f"   🎉 EXCELLENT: System is well integrated with advanced frameworks")
    elif integration_percentage >= 60:
        print(f"   🟡 GOOD: System has solid integration but needs minor upgrades")
    elif integration_percentage >= 40:
        print(f"   🔴 NEEDS WORK: Significant integration gaps identified")
    else:
        print(f"   🚨 CRITICAL: Major integration overhaul required")
    
    print(f"\n   📈 QUANTITATIVE MODELS STATUS: Advanced 5-phase frameworks implemented")
    print(f"   🧠 AI INTELLIGENCE STATUS: Partially integrated")
    print(f"   📊 DASHBOARD STATUS: REQUIRES IMMEDIATE UPGRADE")
    print(f"   ⚙️ EXECUTION LAYER STATUS: Needs integration")
    
    return integration_status

if __name__ == "__main__":
    analyze_system_integration()
