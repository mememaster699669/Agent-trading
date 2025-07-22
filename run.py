#!/usr/bin/env python3
"""
Agent Trading System - Main Entry Point
Simple wrapper to start the main application
"""

import asyncio

if __name__ == "__main__":
    from src.main import main
    asyncio.run(main())