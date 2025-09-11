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
    instrument_id INTEGER REFERENCES instruments(id),
    order_type VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'PENDING',
    order_id VARCHAR(50),
    exchange_order_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    order_id INTEGER REFERENCES orders(id),
    instrument_id INTEGER REFERENCES instruments(id),
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    trade_id VARCHAR(50),
    exchange_trade_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    instrument_id INTEGER REFERENCES instruments(id),
    quantity INTEGER NOT NULL DEFAULT 0,
    average_price DECIMAL(10,2) DEFAULT 0,
    unrealized_pnl DECIMAL(15,2) DEFAULT 0,
    realized_pnl DECIMAL(15,2) DEFAULT 0,
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
