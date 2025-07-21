#!/bin/bash
# Docker startup script for Agent Trading System

set -e

echo "Starting Agent Trading System..."
echo "Environment: ${APP_ENV:-development}"
echo "Python Path: ${PYTHONPATH:-/app}"

# Wait for dependencies
echo "Waiting for database and Redis..."
python -c "
import time
import psycopg2
import redis
import os

# Wait for PostgreSQL
while True:
    try:
        conn = psycopg2.connect(os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/agent_trading'))
        conn.close()
        print('PostgreSQL is ready!')
        break
    except psycopg2.OperationalError:
        print('PostgreSQL is not ready, waiting...')
        time.sleep(5)

# Wait for Redis
while True:
    try:
        r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379'))
        r.ping()
        print('Redis is ready!')
        break
    except:
        print('Redis is not ready, waiting...')
        time.sleep(5)
"

# Create necessary directories
mkdir -p /app/data /app/logs /app/models

# Run database migrations if needed
echo "Running database initialization..."
# Add migration commands here if needed

# Start the application
echo "Starting Agent Trading System..."
exec python run.py
