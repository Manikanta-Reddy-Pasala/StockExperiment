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
