-- Agent Trading System Database Initialization
-- PostgreSQL initialization script

-- Create database if not exists (handled by docker-compose)
-- CREATE DATABASE IF NOT EXISTS agent_trading;

-- Use the database
\c agent_trading;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS risk;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Create tables for trading data
CREATE TABLE IF NOT EXISTS trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    size DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed', 'pending')),
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create tables for market data
CREATE TABLE IF NOT EXISTS trading.market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, timestamp, timeframe)
);

-- Create tables for signals
CREATE TABLE IF NOT EXISTS analytics.signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('buy', 'sell', 'hold')),
    strength DECIMAL(5, 4) NOT NULL CHECK (strength BETWEEN 0 AND 1),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Create tables for risk metrics
CREATE TABLE IF NOT EXISTS risk.portfolio_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_value DECIMAL(20, 8) NOT NULL,
    total_pnl DECIMAL(20, 8) NOT NULL,
    max_drawdown DECIMAL(5, 4),
    sharpe_ratio DECIMAL(8, 4),
    var_95 DECIMAL(20, 8),
    beta DECIMAL(8, 4),
    metadata JSONB
);

-- Create tables for system monitoring
CREATE TABLE IF NOT EXISTS monitoring.system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON trading.positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON trading.positions(opened_at);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON trading.market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON trading.market_data(timeframe);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON analytics.signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON analytics.signals(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_expires_at ON analytics.signals(expires_at);

CREATE INDEX IF NOT EXISTS idx_portfolio_metrics_timestamp ON risk.portfolio_metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_system_events_type ON monitoring.system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_severity ON monitoring.system_events(severity);
CREATE INDEX IF NOT EXISTS idx_system_events_created_at ON monitoring.system_events(created_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
CREATE TRIGGER update_positions_updated_at 
    BEFORE UPDATE ON trading.positions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA risk TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO postgres;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA trading TO postgres;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO postgres;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA risk TO postgres;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA monitoring TO postgres;
