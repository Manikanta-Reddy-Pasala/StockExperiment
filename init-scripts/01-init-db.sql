-- Database initialization script
-- This script ensures all tables are created properly

-- Create the trading_system database if it doesn't exist
-- (This is handled by POSTGRES_DB environment variable)

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables (these will be created by SQLAlchemy, but we can add them here as backup)
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    last_activity TIMESTAMP
);

-- Create admin user if it doesn't exist
-- Password hash for 'admin123' using bcrypt
INSERT INTO users (username, email, password_hash, first_name, last_name, is_active, is_admin, created_at)
VALUES (
    'admin',
    'admin@trading-system.com',
    '$2b$12$C4TAPNHIUChvMlPrxow22u4evaMMKVqdWlAZ7m6ZpQUovjg0fF7JW', -- admin123
    'System',
    'Administrator',
    TRUE,
    TRUE,
    CURRENT_TIMESTAMP
) ON CONFLICT (username) DO NOTHING;

-- Create other essential tables (these will be created by SQLAlchemy)
CREATE TABLE IF NOT EXISTS instruments (
    id SERIAL PRIMARY KEY,
    exchange_token VARCHAR(50) UNIQUE NOT NULL,
    tradingsymbol VARCHAR(50) NOT NULL,
    name VARCHAR(100),
    exchange VARCHAR(20),
    instrument_type VARCHAR(20),
    segment VARCHAR(20),
    lot_size INTEGER DEFAULT 1,
    tick_size DECIMAL(10,4) DEFAULT 0.05,
    expiry_date DATE,
    strike_price DECIMAL(10,2),
    option_type VARCHAR(2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    order_id VARCHAR(50) UNIQUE NOT NULL,
    parent_order_id VARCHAR(50),
    exchange_order_id VARCHAR(50),
    tradingsymbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(20),
    instrument_token VARCHAR(50),
    product VARCHAR(10),
    order_type VARCHAR(10),
    transaction_type VARCHAR(10),
    quantity INTEGER,
    disclosed_quantity INTEGER,
    price DECIMAL(10,2),
    trigger_price DECIMAL(10,2),
    average_price DECIMAL(10,2),
    filled_quantity INTEGER,
    pending_quantity INTEGER,
    order_status VARCHAR(20),
    status_message TEXT,
    tag VARCHAR(100),
    placed_at TIMESTAMP,
    placed_by VARCHAR(50),
    variety VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    order_id VARCHAR(50) REFERENCES orders(order_id),
    exchange_order_id VARCHAR(50),
    tradingsymbol VARCHAR(50),
    exchange VARCHAR(20),
    instrument_token VARCHAR(50),
    transaction_type VARCHAR(10),
    quantity INTEGER,
    price DECIMAL(10,2),
    filled_quantity INTEGER,
    order_price DECIMAL(10,2),
    trade_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    tradingsymbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(20),
    instrument_token VARCHAR(50),
    product VARCHAR(10),
    quantity INTEGER,
    overnight_quantity INTEGER,
    multiplier INTEGER,
    average_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    last_price DECIMAL(10,2),
    value DECIMAL(15,2),
    pnl DECIMAL(15,2),
    m2m DECIMAL(15,2),
    unrealised DECIMAL(15,2),
    realised DECIMAL(15,2),
    buy_quantity INTEGER,
    buy_price DECIMAL(10,2),
    buy_value DECIMAL(15,2),
    buy_m2m DECIMAL(15,2),
    sell_quantity INTEGER,
    sell_price DECIMAL(10,2),
    sell_value DECIMAL(15,2),
    sell_m2m DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS suggested_stocks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    symbol VARCHAR(20) NOT NULL,
    name VARCHAR(100),
    recommendation VARCHAR(20),
    target_price DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS configurations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    key VARCHAR(100) NOT NULL,
    value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, key)
);

-- User Strategy Settings table
CREATE TABLE IF NOT EXISTS user_strategy_settings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_enabled BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 1,
    custom_parameters TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, strategy_name)
);

CREATE TABLE IF NOT EXISTS logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS broker_configurations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    broker_name VARCHAR(50) NOT NULL,
    client_id VARCHAR(100),
    access_token TEXT,
    refresh_token TEXT,
    api_key VARCHAR(200),
    api_secret TEXT,
    redirect_url VARCHAR(500),
    app_type VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    is_connected BOOLEAN DEFAULT FALSE,
    last_connection_test TIMESTAMP,
    connection_status VARCHAR(20),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, broker_name)
);

-- Stock Strategy Tables
CREATE TABLE IF NOT EXISTS stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    sector VARCHAR(100),
    market_cap DECIMAL(15,2),
    market_cap_category VARCHAR(20),
    current_price DECIMAL(10,2),
    volume BIGINT,
    pe_ratio DECIMAL(8,2),
    pb_ratio DECIMAL(8,2),
    roe DECIMAL(8,4),
    debt_to_equity DECIMAL(8,4),
    dividend_yield DECIMAL(8,4),
    beta DECIMAL(8,4),
    is_active BOOLEAN DEFAULT TRUE,
    is_tradeable BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stock_prices (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    date DATE NOT NULL,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    adjusted_close DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(stock_id, date)
);

CREATE TABLE IF NOT EXISTS strategy_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    config_json TEXT,
    large_cap_allocation DECIMAL(5,4) DEFAULT 0.0,
    mid_cap_allocation DECIMAL(5,4) DEFAULT 0.0,
    small_cap_allocation DECIMAL(5,4) DEFAULT 0.0,
    risk_level VARCHAR(20),
    max_position_size DECIMAL(5,4),
    max_sector_allocation DECIMAL(5,4),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategy_stock_selections (
    id SERIAL PRIMARY KEY,
    strategy_type_id INTEGER REFERENCES strategy_types(id),
    stock_id INTEGER REFERENCES stocks(id),
    selection_date DATE NOT NULL,
    selection_price DECIMAL(10,2),
    selection_reason TEXT,
    ml_confidence DECIMAL(5,4),
    expected_return DECIMAL(8,4),
    risk_score DECIMAL(5,4),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_type_id, stock_id, selection_date)
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    prediction_date DATE NOT NULL,
    rf_predicted_price DECIMAL(10,2),
    xgb_predicted_price DECIMAL(10,2),
    lstm_predicted_price DECIMAL(10,2),
    final_predicted_price DECIMAL(10,2),
    predicted_change_percent DECIMAL(8,4),
    confidence DECIMAL(5,4),
    signal VARCHAR(10),
    prediction_horizon_days INTEGER,
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(stock_id, prediction_date, prediction_horizon_days)
);

CREATE TABLE IF NOT EXISTS portfolio_strategies (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    strategy_name VARCHAR(100) NOT NULL,
    strategy_type VARCHAR(20) NOT NULL,
    total_capital DECIMAL(15,2) NOT NULL,
    allocated_capital DECIMAL(15,2) DEFAULT 0.0,
    available_capital DECIMAL(15,2),
    large_cap_allocation DECIMAL(5,4) DEFAULT 0.6,
    mid_cap_allocation DECIMAL(5,4) DEFAULT 0.3,
    small_cap_allocation DECIMAL(5,4) DEFAULT 0.1,
    max_position_size DECIMAL(5,4) DEFAULT 0.05,
    max_sector_allocation DECIMAL(5,4) DEFAULT 0.20,
    stop_loss_percentage DECIMAL(5,4) DEFAULT 0.10,
    rebalance_frequency_days INTEGER DEFAULT 30,
    last_rebalance_date TIMESTAMP,
    next_rebalance_date TIMESTAMP,
    initial_value DECIMAL(15,2),
    current_value DECIMAL(15,2),
    total_return DECIMAL(15,2),
    return_percentage DECIMAL(8,4),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_positions (
    id SERIAL PRIMARY KEY,
    portfolio_strategy_id INTEGER REFERENCES portfolio_strategies(id),
    stock_id INTEGER REFERENCES stocks(id),
    quantity INTEGER NOT NULL,
    average_price DECIMAL(10,2) NOT NULL,
    current_price DECIMAL(10,2),
    market_value DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    realized_pnl DECIMAL(15,2),
    entry_date DATE NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default strategy types
INSERT INTO strategy_types (
    name, display_name, description, config_json,
    large_cap_allocation, mid_cap_allocation, small_cap_allocation,
    risk_level, max_position_size, max_sector_allocation, is_active
) VALUES 
(
    'default_risk',
    'Default Risk (Balanced)',
    'Balanced portfolio with 60% large cap, 30% mid cap, and 10% small cap allocation. Suitable for moderate risk investors.',
    '{"approach": "balanced", "rebalance_frequency": 30, "ml_confidence_threshold": 0.65}',
    0.60, 0.30, 0.10,
    'medium', 0.05, 0.20, TRUE
),
(
    'high_risk',
    'High Risk (Small Cap Focus)',
    'Aggressive portfolio with 80% small cap and 20% mid cap allocation. Suitable for high risk investors seeking higher returns.',
    '{"approach": "aggressive", "rebalance_frequency": 15, "ml_confidence_threshold": 0.60}',
    0.00, 0.20, 0.80,
    'high', 0.08, 0.30, TRUE
) ON CONFLICT (name) DO NOTHING;

-- Insert default strategy settings for admin user
INSERT INTO user_strategy_settings (user_id, strategy_name, is_active, is_enabled, priority, custom_parameters)
VALUES 
(
    (SELECT id FROM users WHERE username = 'admin'),
    'default_risk',
    TRUE,
    TRUE,
    1,
    '{}'
),
(
    (SELECT id FROM users WHERE username = 'admin'),
    'high_risk',
    TRUE,
    TRUE,
    2,
    '{}'
) ON CONFLICT (user_id, strategy_name) DO NOTHING;

-- Portfolio Performance Tracking Tables (Broker-Aware)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    broker_name VARCHAR(50) NOT NULL,
    snapshot_date DATE NOT NULL,
    portfolio_value DECIMAL(15,2) NOT NULL,
    cash_balance DECIMAL(15,2) DEFAULT 0.0,
    total_invested DECIMAL(15,2) DEFAULT 0.0,
    total_pnl DECIMAL(15,2) DEFAULT 0.0,
    day_pnl DECIMAL(15,2) DEFAULT 0.0,
    return_percent DECIMAL(8,4) DEFAULT 0.0,
    holdings_data JSONB,
    positions_data JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, broker_name, snapshot_date)
);

CREATE TABLE IF NOT EXISTS portfolio_performance_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    broker_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    portfolio_value DECIMAL(15,2) NOT NULL,
    daily_return DECIMAL(8,4) DEFAULT 0.0,
    cumulative_return DECIMAL(8,4) DEFAULT 0.0,
    drawdown DECIMAL(8,4) DEFAULT 0.0,
    volatility DECIMAL(8,4) DEFAULT 0.0,
    sharpe_ratio DECIMAL(8,4) DEFAULT 0.0,
    max_drawdown DECIMAL(8,4) DEFAULT 0.0,
    win_rate DECIMAL(5,2) DEFAULT 0.0,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    best_day DECIMAL(8,4) DEFAULT 0.0,
    worst_day DECIMAL(8,4) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, broker_name, date)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_stocks_market_cap_category ON stocks(market_cap_category);
CREATE INDEX IF NOT EXISTS idx_stock_prices_stock_date ON stock_prices(stock_id, date);
CREATE INDEX IF NOT EXISTS idx_strategy_stock_selections_strategy ON strategy_stock_selections(strategy_type_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_stock_date ON ml_predictions(stock_id, prediction_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_strategies_user ON portfolio_strategies(user_id);
CREATE INDEX IF NOT EXISTS idx_user_strategy_settings_user ON user_strategy_settings(user_id);
CREATE INDEX IF NOT EXISTS idx_user_strategy_settings_active ON user_strategy_settings(user_id, is_active, is_enabled);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_portfolio ON portfolio_positions(portfolio_strategy_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_user_broker_date ON portfolio_snapshots(user_id, broker_name, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_user_broker_date ON portfolio_performance_history(user_id, broker_name, date);
