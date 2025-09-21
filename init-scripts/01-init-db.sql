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
-- Note: Stocks table contains only verified stocks with live API data
-- Verification is handled in symbol_master table before stock creation
CREATE TABLE IF NOT EXISTS stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    exchange VARCHAR(20) NOT NULL DEFAULT 'NSE',
    sector VARCHAR(100),
    market_cap DOUBLE PRECISION,  -- in crores, matches SQLAlchemy Float
    market_cap_category VARCHAR(20),
    current_price DOUBLE PRECISION,  -- matches SQLAlchemy Float
    volume BIGINT,  -- Handle high volume stocks like IDEA (2.6B+ volume)
    pe_ratio DOUBLE PRECISION,  -- matches SQLAlchemy Float
    pb_ratio DOUBLE PRECISION,  -- matches SQLAlchemy Float
    roe DOUBLE PRECISION,  -- matches SQLAlchemy Float
    debt_to_equity DOUBLE PRECISION,  -- matches SQLAlchemy Float
    dividend_yield DOUBLE PRECISION,  -- matches SQLAlchemy Float
    beta DOUBLE PRECISION,  -- matches SQLAlchemy Float
    -- Volatility and risk metrics
    atr_14 DOUBLE PRECISION,  -- Average True Range (14-day period)
    atr_percentage DOUBLE PRECISION,  -- ATR as percentage of current price
    historical_volatility_1y DOUBLE PRECISION,  -- Annualized historical volatility
    bid_ask_spread DOUBLE PRECISION,  -- Estimated bid-ask spread
    avg_daily_volume_20d DOUBLE PRECISION,  -- 20-day average daily volume
    avg_daily_turnover DOUBLE PRECISION,  -- Average daily turnover in crores
    trades_per_day INTEGER,  -- Average trades per day
    volatility_last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- For tracking volatility updates
    is_active BOOLEAN DEFAULT TRUE,
    is_tradeable BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for stocks table (matches SQLAlchemy model)
CREATE INDEX IF NOT EXISTS ix_stocks_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS ix_stocks_market_cap_category ON stocks(market_cap_category);
CREATE INDEX IF NOT EXISTS ix_stocks_is_active ON stocks(is_active);

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
    user_id INTEGER REFERENCES users(id) NOT NULL,
    strategy_type_id INTEGER REFERENCES strategy_types(id),
    stock_id INTEGER REFERENCES stocks(id),
    selection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    selection_price DECIMAL(10,2) NOT NULL,
    selection_score DECIMAL(10,4),
    recommended_quantity INTEGER,
    recommended_allocation DECIMAL(10,4),
    position_size_rationale TEXT,
    target_price DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    expected_return DECIMAL(10,4),
    risk_score DECIMAL(10,4),
    status VARCHAR(20) DEFAULT 'selected',
    execution_date TIMESTAMP,
    exit_date TIMESTAMP,
    current_price DECIMAL(10,2),
    unrealized_pnl DECIMAL(15,2),
    realized_pnl DECIMAL(15,2),
    selection_reason TEXT,
    algorithm_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    user_id INTEGER REFERENCES users(id),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_horizon_days INTEGER DEFAULT 30,
    current_price DECIMAL(10,2) NOT NULL,
    rf_predicted_price DECIMAL(10,2),
    xgb_predicted_price DECIMAL(10,2),
    lstm_predicted_price DECIMAL(10,2),
    ensemble_predicted_price DECIMAL(10,2),
    prediction_confidence DECIMAL(10,4),
    model_accuracy DECIMAL(10,4),
    prediction_std DECIMAL(10,4),
    signal VARCHAR(10),
    signal_strength DECIMAL(10,4),
    expected_return DECIMAL(10,4),
    risk_reward_ratio DECIMAL(10,4),
    model_version VARCHAR(50),
    features_used TEXT,
    training_data_period VARCHAR(20),
    actual_price DECIMAL(10,2),
    prediction_error DECIMAL(10,4),
    is_validated BOOLEAN DEFAULT FALSE,
    validation_date TIMESTAMP,
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
    entry_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    entry_price DECIMAL(10,2) NOT NULL,
    quantity INTEGER NOT NULL,
    investment_amount DECIMAL(15,2) NOT NULL,
    current_price DECIMAL(10,2),
    current_value DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    unrealized_pnl_percentage DECIMAL(10,4),
    target_price DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    position_allocation DECIMAL(10,4),
    exit_date TIMESTAMP,
    exit_price DECIMAL(10,2),
    realized_pnl DECIMAL(15,2),
    realized_pnl_percentage DECIMAL(10,4),
    status VARCHAR(20) DEFAULT 'active',
    entry_reason TEXT,
    exit_reason TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Symbol Master table for raw broker data (fytoken as primary key)
-- This table stores all symbols from Fyers API and handles verification
-- Only verified symbols (is_fyers_verified = true) are promoted to stocks table
CREATE TABLE IF NOT EXISTS symbol_master (
    fytoken VARCHAR(50) PRIMARY KEY NOT NULL,  -- Fyers unique token as primary key
    symbol VARCHAR(50) NOT NULL,
    name VARCHAR(200) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    segment VARCHAR(20) NOT NULL,
    instrument_type VARCHAR(20) NOT NULL,
    lot_size INTEGER DEFAULT 1,
    tick_size DOUBLE PRECISION DEFAULT 0.05,  -- matches SQLAlchemy Float
    isin VARCHAR(20),
    data_source VARCHAR(20) DEFAULT 'fyers',
    source_updated VARCHAR(20),
    download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_equity BOOLEAN DEFAULT TRUE,
    -- Verification columns - validates symbols work with Fyers API quotes
    is_fyers_verified BOOLEAN DEFAULT FALSE,
    verification_date TIMESTAMP,
    verification_error TEXT,
    last_quote_check TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Unique constraint to prevent duplicate symbol-exchange combinations
    CONSTRAINT _symbol_exchange_uc UNIQUE(symbol, exchange)
);

-- Indexes for symbol_master table (matches SQLAlchemy model)
CREATE INDEX IF NOT EXISTS ix_symbol_master_symbol ON symbol_master(symbol);
CREATE INDEX IF NOT EXISTS ix_symbol_master_exchange ON symbol_master(exchange);
CREATE INDEX IF NOT EXISTS ix_symbol_master_is_active ON symbol_master(is_active);
CREATE INDEX IF NOT EXISTS ix_symbol_master_is_equity ON symbol_master(is_equity);
CREATE INDEX IF NOT EXISTS ix_symbol_master_is_fyers_verified ON symbol_master(is_fyers_verified);

-- Market Data Snapshots table
CREATE TABLE IF NOT EXISTS market_data_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date VARCHAR(10) NOT NULL,
    nifty_50 DECIMAL(10,2),
    sensex DECIMAL(10,2),
    nifty_midcap DECIMAL(10,2),
    nifty_smallcap DECIMAL(10,2),
    total_stocks_tracked INTEGER,
    large_cap_avg_change DECIMAL(10,4),
    mid_cap_avg_change DECIMAL(10,4),
    small_cap_avg_change DECIMAL(10,4),
    total_volume BIGINT,
    advance_decline_ratio DECIMAL(10,4),
    data_source VARCHAR(20) DEFAULT 'fyers',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(snapshot_date)
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

-- ML Training and Model Management Tables
CREATE TABLE IF NOT EXISTS ml_training_jobs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    duration VARCHAR(10) NOT NULL DEFAULT '1y',
    status VARCHAR(20) DEFAULT 'pending' NOT NULL,
    progress DECIMAL(5,2) DEFAULT 0.0,
    accuracy DECIMAL(5,4),
    use_technical_indicators BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_trained_models (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    training_job_id INTEGER REFERENCES ml_training_jobs(id),
    symbol VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    accuracy DECIMAL(5,4),
    model_version VARCHAR(50),
    model_file_path TEXT,
    scaler_file_path TEXT,
    feature_columns TEXT,
    target_column VARCHAR(100),
    training_start_date TIMESTAMP,
    training_end_date TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, symbol, model_type, is_active)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_stocks_market_cap_category ON stocks(market_cap_category);
CREATE INDEX IF NOT EXISTS idx_strategy_stock_selections_strategy ON strategy_stock_selections(strategy_type_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_stock_date ON ml_predictions(stock_id, prediction_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_strategies_user ON portfolio_strategies(user_id);
CREATE INDEX IF NOT EXISTS idx_user_strategy_settings_user ON user_strategy_settings(user_id);
CREATE INDEX IF NOT EXISTS idx_user_strategy_settings_active ON user_strategy_settings(user_id, is_active, is_enabled);
CREATE INDEX IF NOT EXISTS idx_portfolio_positions_portfolio ON portfolio_positions(portfolio_strategy_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_user_broker_date ON portfolio_snapshots(user_id, broker_name, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_user_broker_date ON portfolio_performance_history(user_id, broker_name, date);
CREATE INDEX IF NOT EXISTS idx_ml_training_jobs_user_status ON ml_training_jobs(user_id, status);
CREATE INDEX IF NOT EXISTS idx_ml_training_jobs_symbol ON ml_training_jobs(symbol);
CREATE INDEX IF NOT EXISTS idx_ml_trained_models_user_symbol ON ml_trained_models(user_id, symbol);
CREATE INDEX IF NOT EXISTS idx_ml_trained_models_active ON ml_trained_models(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_symbol_master_symbol ON symbol_master(symbol);
CREATE INDEX IF NOT EXISTS idx_symbol_master_exchange ON symbol_master(exchange);
CREATE INDEX IF NOT EXISTS idx_symbol_master_active ON symbol_master(is_active, is_equity);
CREATE INDEX IF NOT EXISTS idx_symbol_master_verified ON symbol_master(is_fyers_verified);
CREATE INDEX IF NOT EXISTS idx_market_data_snapshots_date ON market_data_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_stocks_active ON stocks(is_active, is_tradeable);
CREATE INDEX IF NOT EXISTS idx_stocks_market_cap ON stocks(market_cap);
-- Volatility indexes for screening performance
CREATE INDEX IF NOT EXISTS idx_stocks_atr_percentage ON stocks(atr_percentage);
CREATE INDEX IF NOT EXISTS idx_stocks_beta ON stocks(beta);
CREATE INDEX IF NOT EXISTS idx_stocks_historical_volatility ON stocks(historical_volatility_1y);
CREATE INDEX IF NOT EXISTS idx_stocks_avg_volume_20d ON stocks(avg_daily_volume_20d);
CREATE INDEX IF NOT EXISTS idx_stocks_volatility_last_updated ON stocks(volatility_last_updated);
