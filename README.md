# Automated Trading System

**High-performance automated trading system with triple ML models, dual strategies, and complete automation.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Quick Start

```bash
# 1. Start all services (Docker)
./run.sh dev

# 2. Access web interface
http://localhost:5001

# 3. Access admin dashboard (admin users only)
http://localhost:5001/admin

# 4. Check system status
./tools/check_all_schedulers.sh
```

**First run:** Initial pipeline takes ~30 minutes to populate data for 2,259 stocks.

**Subsequent runs:** Fully automated - no manual intervention needed!

---

## 📋 Table of Contents

- [Features](#-features)
- [System Overview](#-system-overview)
- [Installation](#-installation)
- [Architecture](#-architecture)
- [Database Schema](#️-database-schema)
- [API Endpoints](#-api-endpoints)
- [Automation](#-automation)
- [Machine Learning](#-machine-learning)
- [Configuration](#️-configuration)
- [Usage Examples](#-usage-examples)
- [Monitoring](#-monitoring)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

---

## ✨ Features

### 🤖 Complete Automation
- **100% Automated Data Collection** - Runs daily at 9 PM after market close
- **Automated ML Training** - Trains 3 models daily at 10 PM with fresh data
- **Automated Stock Selection** - Generates daily top 50 picks × 3 models × 2 strategies
- **Automated Trading Execution** - Places orders at 9:20 AM with stop-loss/targets
- **Automated CSV Exports** - Daily backups for analysis and reporting
- **Self-Healing** - Saga pattern with retry logic (3 attempts, 60s delay)
- **Zero Manual Intervention** - Set it and forget it!

### 🧠 Triple Machine Learning Models
- **Traditional ML** (Random Forest + XGBoost ensemble)
  - Walk-forward cross-validation (5 splits)
  - 25-30 features (technical + fundamental)
  - Fast training (~1-2 minutes)
- **Raw LSTM** (Deep learning)
  - Per-symbol models with 60-day lookback
  - Predicts 14 days ahead
  - OHLCV sequence modeling
- **Kronos** (K-line tokenization)
  - Candlestick pattern recognition
  - Advanced technical analysis
  - Experimental cutting-edge model

### 📊 Comprehensive Data Pipeline
- **6-Step Saga Pattern** - Symbols → Stocks → History → Indicators → Metrics → Validation
- **2,259+ NSE Stocks** - All NSE-listed stocks with complete data
- **1-Year Historical Data** - 820K+ OHLCV records
- **Technical Indicators** - RSI, MACD, SMA, EMA, ATR, Bollinger Bands
- **Real-Time Updates** - Market data refresh during trading hours
- **Robust Error Handling** - Retry logic with exponential backoff

### 🎯 Dual Strategy System
- **DEFAULT_RISK** (Conservative)
  - Large-cap stocks (>20,000 Cr market cap)
  - PE ratio 5-40, price ₹100-10,000
  - Target gain: 7%, Stop loss: 5%
  - Good liquidity requirements
- **HIGH_RISK** (Aggressive)
  - Small/Mid-cap stocks (1,000-20,000 Cr)
  - Broader criteria, lower score threshold
  - Target gain: 12%, Stop loss: 10%
  - Higher volatility tolerance

### 🌐 API & Interface
- **REST API** - Fast, cached endpoints with ML predictions
- **Admin Dashboard** - Manual task triggers with real-time monitoring
- **Web Interface** - Portfolio management and trading controls
- **Multi-Broker Support** - Unified broker service (Fyers, Zerodha, Simulator)
- **Real-Time Status** - Live task monitoring and logs
- **Auto-Trading** - Automated order placement with risk management

### 📈 Analytics & Insights
- **6 Model Combinations** - 3 models × 2 strategies = comprehensive coverage
- **ML Prediction Scores** - 0-1 score for opportunity ranking
- **Confidence Levels** - Model confidence in predictions
- **Risk Scores** - Lower = safer investment
- **Performance Tracking** - Daily P&L snapshots
- **Ollama AI Enhancement** - Optional market intelligence layer

---

## 📊 System Overview

### Daily Automation Schedule

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTOMATED DAILY SCHEDULE                   │
└─────────────────────────────────────────────────────────────┘

🌅 Morning:
  06:00 AM → Symbol Master Update (Monday Only)
             • Refresh NSE symbols from Fyers API
             • ~2,300 stocks
             • Duration: 1-2 minutes

  09:20 AM → Auto-Trading Execution (Daily)
             • Check AI market sentiment
             • Apply weekly limits
             • Place orders with stop-loss/targets
             • Duration: 2-3 minutes

🌆 Evening (Daily After Market Close):
  06:00 PM → Performance Tracking
             • Update order performance
             • Create daily snapshots
             • Check stop-loss/targets
             • Close orders if needed
             • Duration: 1-2 minutes

  09:00 PM → Data Pipeline (6-step saga)
             • Update all stock prices
             • Fetch 1-year historical OHLCV
             • Calculate technical indicators
             • Compute volatility metrics
             • Validate data quality
             • Duration: 20-30 minutes
             • Records: 2,259 stocks updated

  09:30 PM → Fill Missing Data & Business Logic (Parallel)
             • Populate adj_close, liquidity
             • Calculate ATR, volatility
             • EPS, Book Value, PEG Ratio, ROA
             • Operating/Net/Profit Margins
             • Current/Quick Ratios, Debt to Equity
             • Duration: 5-10 minutes

  10:00 PM → CSV Export & Data Validation (Parallel)
             • Export 4 CSV files (stocks, history, indicators, picks)
             • Validate data quality
             • Duration: 2-3 minutes

🌙 Late Evening:
  10:00 PM → ML Model Training (3 Models)
             • Model 1: Traditional ML (RF + XGBoost)
             • Model 2: Raw LSTM (Deep Learning)
             • Model 3: Kronos (K-line Tokenization)
             • Duration: 5-10 minutes total

  10:15 PM → Daily Stock Selection (Triple Model × Dual Strategy)
             • Run for all 3 models
             • Apply both strategies (DEFAULT_RISK, HIGH_RISK)
             • Generate 6 combinations total
             • Optional: Ollama AI enhancement
             • Save to daily_suggested_stocks table
             • Duration: 3-5 minutes

  03:00 AM → Cleanup Old Data (Sunday Only)
             • Remove snapshots > 90 days
             • Remove CSV exports > 30 days
             • Duration: < 1 minute

Total Daily Automation Time: ~45-60 minutes
Total Manual Intervention: ZERO! 🎉
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────┘

External APIs:
┌──────────────┐
│  Fyers API   │──┐
└──────────────┘  │
                  │
┌──────────────┐  │
│ Zerodha API  │──┤
└──────────────┘  │
                  ↓
         ┌────────────────┐
         │  Flask App     │ ← HTTP Requests
         │  (Port 5001)   │
         └────────────────┘
                  ↓
         ┌────────────────┐
         │  Dragonfly     │
         │  (Redis Cache) │
         └────────────────┘
                  ↓
    ┌─────────────────────────────┐
    │    PostgreSQL DB (15 tables)│
    │  ┌──────────────────────┐   │
    │  │ stocks (2,259)       │   │
    │  │ historical_data      │   │
    │  │ tech_indicators      │   │
    │  │ daily_suggested      │   │
    │  │ pipeline_tracking    │   │
    │  │ orders, trades       │   │
    │  │ auto_trading_settings│   │
    │  └──────────────────────┘   │
    └─────────────────────────────┘
                  ↑
    ┌─────────────┴─────────────┐
    │                           │
┌───────────────┐      ┌──────────────┐
│ Data Scheduler│      │ ML Scheduler │
│  (9 PM daily) │      │ (10 PM daily)│
└───────────────┘      └──────────────┘

ML Models Storage:
┌────────────────────────────┐
│ Traditional ML:            │
│ ml_models/                 │
│ ├── rf_price_model.pkl     │
│ └── xgb_price_model.pkl    │
│                            │
│ Raw LSTM:                  │
│ ml_models/raw_ohlcv_lstm/  │
│ └── {symbol}/              │
│     └── lstm_model.h5      │
│                            │
│ Kronos:                    │
│ ml_models/kronos/          │
│ └── kronos_model.pkl       │
└────────────────────────────┘

CSV Exports:
┌────────────┐        ┌────────────┐
│  /exports  │        │   /logs    │
│ (CSV files)│        │ (App logs) │
└────────────┘        └────────────┘
```

---

## 🔧 Installation

### Prerequisites

- **Operating System:** Linux, macOS, or Windows (WSL2)
- **Docker:** Version 20.10 or higher
- **Docker Compose:** Version 2.0 or higher
- **Python:** 3.10 or higher (for local scripts)
- **Git:** For cloning the repository
- **Fyers API Credentials:** Client ID and Access Token

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/StockExperiment.git
cd StockExperiment
```

### Step 2: Configure Fyers API

Create `.env` file in root:

```bash
# Fyers API Credentials
FYERS_CLIENT_ID=your_client_id_here
FYERS_ACCESS_TOKEN=your_access_token_here

# Database Configuration (default - don't change unless needed)
DATABASE_URL=postgresql://trader:trader_password@database:5432/trading_system
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_password
POSTGRES_DB=trading_system

# Redis Configuration
REDIS_HOST=dragonfly
REDIS_PORT=6379

# Application Settings
FLASK_ENV=development
FLASK_DEBUG=1
LOG_LEVEL=INFO
```

### Step 3: Start Docker Services

```bash
# Development mode (with auto-reload)
./run.sh dev

# Production mode
./run.sh prod
```

### Step 4: Wait for Initial Setup

```bash
# Monitor initial pipeline (takes ~30 minutes)
docker compose logs -f trading_system

# Check when ready
./tools/check_all_schedulers.sh
```

### Step 5: Access Application

```bash
# Web Interface
http://localhost:5001

# Admin Dashboard
http://localhost:5001/admin

# API Documentation
http://localhost:5001/api/suggested-stocks/

# Database (PostgreSQL)
docker exec -it trading_system_db psql -U trader -d trading_system
```

---

## 🏗️ Architecture

### Technology Stack

**Backend:**
- **Python 3.10+** - Core language
- **Flask 3.0** - Web framework
- **SQLAlchemy 2.0** - ORM for database
- **PostgreSQL 15** - Primary database (11 tables)
- **Dragonfly (Redis)** - Caching layer
- **Scikit-learn** - Random Forest models
- **XGBoost** - Gradient boosting
- **TensorFlow 2.16+** - LSTM models
- **Pandas/NumPy** - Data processing
- **Schedule** - Task scheduling

**Frontend:**
- **Bootstrap 5** - UI framework
- **Chart.js** - Data visualization
- **JavaScript ES6** - Client-side logic
- **Jinja2** - Template engine

**DevOps:**
- **Docker & Docker Compose** - 5 containers orchestration
- **Git** - Version control
- **GitHub Actions** - CI/CD (optional)

### Directory Structure

```
/StockExperiment
├── README.md                     # This file
├── CLAUDE.md                     # AI assistant instructions
├── .env                          # Environment variables (create this)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker services config
│
├── app.py                        # Flask application entry
├── run.py                        # Application launcher
├── run.sh                        # Docker startup script
├── config.py                     # Application configuration
│
├── run_pipeline.py               # Data pipeline orchestrator
├── data_scheduler.py             # Data automation (9 PM)
├── scheduler.py                  # ML automation (10 PM)
│
├── /src                          # Source code
│   ├── __init__.py
│   │
│   ├── /models                   # Database models (SQLAlchemy ORM)
│   │   ├── database.py           # DB connection, session management
│   │   ├── models.py             # Core tables (users, orders, trades)
│   │   ├── stock_models.py       # Stock, SymbolMaster models
│   │   └── historical_models.py  # HistoricalData, TechnicalIndicators
│   │
│   ├── /services                 # Business logic
│   │   ├── /data                 # Data pipeline services
│   │   │   ├── pipeline_saga.py  # 6-step data saga
│   │   │   ├── suggested_stocks_saga.py  # 7-step stock selection
│   │   │   ├── historical_data_service.py # Smart data fetching
│   │   │   └── fyers_symbol_service.py  # Symbol management
│   │   │
│   │   ├── /brokers              # Broker integrations
│   │   │   └── /core
│   │   │       └── unified_broker_service.py # Multi-broker abstraction
│   │   │
│   │   ├── /ml                   # Machine learning
│   │   │   ├── enhanced_stock_predictor.py # Traditional ML (RF + XGBoost)
│   │   │   ├── raw_lstm_prediction_service.py # Raw LSTM models
│   │   │   ├── kronos_prediction_service.py # Kronos K-line model
│   │   │   └── training_service.py # Model training orchestration
│   │   │
│   │   └── /trading              # Trading services
│   │       ├── auto_trading_service.py # Automated trading
│   │       └── order_performance_tracking_service.py # P&L tracking
│   │
│   ├── /web                      # Flask routes & templates
│   │   ├── app.py                # Flask app factory
│   │   ├── admin_routes.py       # Admin dashboard routes
│   │   ├── /routes               # API routes
│   │   │   └── suggested_stocks_routes.py
│   │   └── /templates            # HTML templates
│   │       ├── base.html         # Base template
│   │       ├── dashboard.html    # Main dashboard
│   │       └── /admin            # Admin templates
│   │           └── dashboard.html
│   │
│   └── /utils                    # Helper utilities
│       ├── logger.py             # Logging setup
│       └── helpers.py            # Common utilities
│
├── /config                       # Configuration files
│   ├── stock_filters.yaml        # Stock screening criteria
│   ├── database.yaml             # Database settings
│   └── broker_config.yaml        # Broker settings
│
├── /ml_models                    # Trained ML models
│   ├── rf_price_model.pkl        # Random Forest price model
│   ├── xgb_price_model.pkl       # XGBoost price model
│   ├── /raw_ohlcv_lstm           # Per-symbol LSTM models
│   │   └── {SYMBOL}/lstm_model.h5
│   └── /kronos                   # Kronos models
│       └── kronos_model.pkl
│
├── /tools                        # Utility scripts
│   ├── README.md                 # Tools documentation
│   ├── train_ml_model.py         # Manual ML training
│   ├── batch_train_lstm_top_stocks.py # LSTM training for large-caps
│   ├── batch_train_lstm_small_mid_cap.py # LSTM for small/mid-caps
│   ├── generate_kronos_predictions.py # Kronos predictions
│   └── check_all_schedulers.sh   # Complete system status
│
├── /logs                         # Application logs
│   ├── data_scheduler.log        # Data pipeline logs
│   ├── scheduler.log             # ML scheduler logs
│   └── app.log                   # Flask app logs
│
└── /exports                      # CSV exports (daily)
    ├── stocks_YYYY-MM-DD.csv
    ├── historical_30d_YYYY-MM-DD.csv
    ├── technical_indicators_YYYY-MM-DD.csv
    └── suggested_stocks_YYYY-MM-DD.csv
```

### Docker Services

```yaml
services:
  database:
    image: postgres:15
    container_name: trading_system_db
    ports: ["5432:5432"]
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: trader_password
      POSTGRES_DB: trading_system

  dragonfly:
    image: docker.dragonflydb.io/dragonflydb/dragonfly
    container_name: trading_system_redis
    ports: ["6379:6379"]
    volumes:
      - dragonfly_data:/data

  trading_system:
    build: .
    container_name: trading_system_app
    ports: ["5001:5001"]
    depends_on: [database, dragonfly]
    volumes:
      - .:/app
      - ./logs:/app/logs
      - ./exports:/app/exports
      - ./ml_models:/app/ml_models
    environment:
      DATABASE_URL: postgresql://trader:trader_password@database:5432/trading_system
      REDIS_HOST: dragonfly

  data_scheduler:
    build: .
    container_name: trading_system_data_scheduler
    command: python data_scheduler.py
    depends_on: [database, dragonfly]
    volumes:
      - .:/app
      - ./logs:/app/logs
      - ./exports:/app/exports

  ml_scheduler:
    build: .
    container_name: trading_system_ml_scheduler
    command: python scheduler.py
    depends_on: [database, dragonfly]
    volumes:
      - .:/app
      - ./logs:/app/logs
      - ./ml_models:/app/ml_models
```

---

## 🗄️ Database Schema

### Table Overview (11+ Tables)

| Table | Records | Description |
|-------|---------|-------------|
| **stocks** | 2,259 | Current prices, fundamentals, ratios |
| **historical_data** | ~820,000 | 1-year OHLCV data |
| **technical_indicators** | ~820,000 | RSI, MACD, SMA, EMA, ATR |
| **daily_suggested_stocks** | Growing | Daily picks (3 models × 2 strategies) |
| **pipeline_tracking** | Variable | Pipeline saga status |
| **symbol_master** | ~2,259 | Complete NSE symbol list |
| **users** | Variable | User accounts |
| **strategies** | Variable | Trading strategies |
| **orders** | Variable | Order history with model/strategy |
| **trades** | Variable | Executed trades |
| **positions** | Variable | Current positions |
| **broker_configurations** | Variable | API credentials |
| **auto_trading_settings** | Variable | Per-user trading preferences |

### Core Tables Schema

#### `stocks` Table
```sql
CREATE TABLE stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) UNIQUE NOT NULL,
    stock_name VARCHAR(200),

    -- Price & Market Data
    current_price DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    volume BIGINT,
    avg_volume_30d BIGINT,

    -- Fundamental Ratios
    pe_ratio DOUBLE PRECISION,
    pb_ratio DOUBLE PRECISION,
    roe DOUBLE PRECISION,
    eps DOUBLE PRECISION,
    beta DOUBLE PRECISION,
    debt_to_equity DOUBLE PRECISION,

    -- Growth & Profitability
    revenue_growth DOUBLE PRECISION,
    earnings_growth DOUBLE PRECISION,
    operating_margin DOUBLE PRECISION,
    net_margin DOUBLE PRECISION,
    profit_margin DOUBLE PRECISION,

    -- Valuation
    book_value DOUBLE PRECISION,
    peg_ratio DOUBLE PRECISION,
    roa DOUBLE PRECISION,

    -- Liquidity
    current_ratio DOUBLE PRECISION,
    quick_ratio DOUBLE PRECISION,

    -- Volatility
    historical_volatility_1y DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,

    -- Metadata
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap_category VARCHAR(20),
    data_source VARCHAR(50), -- 'real' or 'estimated_enhanced'
    last_updated TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `daily_suggested_stocks` Table (Enhanced)
```sql
CREATE TABLE daily_suggested_stocks (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    stock_name VARCHAR(200),
    current_price DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,

    -- Model & Strategy
    model_type VARCHAR(50),  -- 'traditional', 'raw_lstm', 'kronos'
    strategy VARCHAR(50) NOT NULL, -- 'default_risk', 'high_risk'
    selection_score DOUBLE PRECISION,
    rank INTEGER,

    -- ML Predictions
    ml_prediction_score DOUBLE PRECISION,  -- 0-1 (higher = better)
    ml_price_target DOUBLE PRECISION,      -- Predicted price
    ml_confidence DOUBLE PRECISION,        -- Model confidence 0-1
    ml_risk_score DOUBLE PRECISION,        -- Risk score 0-1 (lower = safer)

    -- Technical Indicators (snapshot)
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,

    -- Fundamental Metrics (snapshot)
    pe_ratio DOUBLE PRECISION,
    pb_ratio DOUBLE PRECISION,
    roe DOUBLE PRECISION,
    eps DOUBLE PRECISION,
    beta DOUBLE PRECISION,

    -- Growth & Profitability
    revenue_growth DOUBLE PRECISION,
    earnings_growth DOUBLE PRECISION,
    operating_margin DOUBLE PRECISION,

    -- Trading Signals
    target_price DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    recommendation VARCHAR(20),
    reason TEXT,

    -- Metadata
    sector VARCHAR(100),
    market_cap_category VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(date, symbol, strategy, model_type)  -- Allows upsert
);
CREATE INDEX idx_daily_suggested_date ON daily_suggested_stocks(date DESC);
CREATE INDEX idx_daily_suggested_ml_score ON daily_suggested_stocks(ml_prediction_score DESC);
CREATE INDEX idx_daily_suggested_model ON daily_suggested_stocks(model_type);
```

#### `pipeline_tracking` Table
```sql
CREATE TABLE pipeline_tracking (
    id SERIAL PRIMARY KEY,
    pipeline_id VARCHAR(100) UNIQUE NOT NULL,
    step VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL, -- 'pending', 'in_progress', 'completed', 'failed', 'retrying'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    records_processed INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🌐 API Endpoints

### Main Endpoint: Suggested Stocks

```http
GET /api/suggested-stocks/
```

**Query Parameters:**
- `strategy` - Trading strategy: `default_risk`, `high_risk` (default: `default_risk`)
- `model_type` - ML model: `traditional`, `raw_lstm`, `kronos` (optional, returns all if not specified)
- `limit` - Number of stocks to return (default: `50`, max: `100`)
- `sector` - Filter by sector (optional)
- `search` - Search by symbol or name (optional)
- `sort_by` - Sort field: `ml_prediction_score`, `market_cap`, `pe_ratio` (default: `ml_prediction_score`)
- `order` - Sort order: `asc`, `desc` (default: `desc`)

**Example Request:**
```bash
curl "http://localhost:5001/api/suggested-stocks/?strategy=default_risk&model_type=traditional&limit=10"
```

**Example Response:**
```json
{
  "success": true,
  "count": 10,
  "strategy": "default_risk",
  "model_type": "traditional",
  "generated_at": "2025-10-14T22:15:00",
  "stocks": [
    {
      "symbol": "RELIANCE",
      "stock_name": "Reliance Industries Ltd",
      "rank": 1,
      "current_price": 2450.50,
      "market_cap": 16500000.0,
      "sector": "Energy",
      "model_type": "traditional",

      "ml_prediction_score": 0.78,
      "ml_price_target": 2650.20,
      "ml_confidence": 0.85,
      "ml_risk_score": 0.12,

      "technical_indicators": {
        "rsi_14": 58.5,
        "macd": 12.3,
        "sma_50": 2400.0,
        "sma_200": 2300.0
      },

      "fundamentals": {
        "pe_ratio": 24.5,
        "pb_ratio": 2.8,
        "roe": 12.5,
        "eps": 100.5,
        "beta": 1.15,
        "revenue_growth": 8.5,
        "earnings_growth": 10.2,
        "operating_margin": 15.3
      },

      "trading_signals": {
        "target_price": 2625.0,
        "stop_loss": 2328.0,
        "recommendation": "BUY"
      }
    }
  ]
}
```

### Admin Dashboard Endpoints

#### Get System Status
```http
GET /admin/system/status
```

**Response:**
```json
{
  "success": true,
  "status": {
    "stocks": {
      "total": 2259,
      "with_price": 2250,
      "with_market_cap": 2240,
      "last_updated": "2025-10-14T21:30:00"
    },
    "historical_data": {
      "symbols": 2259,
      "records": 820574,
      "latest_date": "2025-10-14"
    },
    "ml_models": {
      "traditional": "trained",
      "raw_lstm": "84 symbols trained",
      "kronos": "trained"
    },
    "daily_snapshots": {
      "total": 15000,
      "unique_dates": 90,
      "latest_date": "2025-10-14",
      "model_combinations": 6
    }
  }
}
```

#### Trigger Tasks
```http
POST /admin/trigger/pipeline           # Run data pipeline
POST /admin/trigger/fill-data          # Fill missing data
POST /admin/trigger/business-logic     # Calculate metrics
POST /admin/trigger/ml-training        # Train all ML models
POST /admin/trigger/csv-export         # Export CSV files
POST /admin/trigger/all                # Run all tasks sequentially
```

---

## 🤖 Automation

### Data Scheduler (`data_scheduler.py`)

**Schedule:**
- Symbol Master: Monday 6:00 AM
- Data Pipeline: Daily 9:00 PM
- Fill Data: Daily 9:30 PM (parallel)
- Business Logic: Daily 9:30 PM (parallel)
- CSV Export: Daily 10:00 PM (parallel)
- Data Validation: Daily 10:00 PM (parallel)

### ML Scheduler (`scheduler.py`)

**Schedule:**
- Auto-Trading: Daily 9:20 AM (5 min after market opens)
- Performance Tracking: Daily 6:00 PM (after market close)
- ML Training (3 models): Daily 10:00 PM
- Daily Snapshot: Daily 10:15 PM (6 combinations)
- Cleanup: Sunday 3:00 AM

### Manual Task Execution

**Via Admin Dashboard:**
```
http://localhost:5001/admin
→ Click task buttons (Pipeline, ML Training, etc.)
→ Monitor progress in real-time
→ Retry failed steps
```

**Via Command Line:**
```bash
# Run data pipeline
python3 run_pipeline.py

# Train ML models
python3 tools/train_ml_model.py

# Train LSTM models
python3 tools/batch_train_lstm_top_stocks.py
python3 tools/batch_train_lstm_small_mid_cap.py

# Generate Kronos predictions
python3 tools/generate_kronos_predictions.py

# Fill missing data
python3 fill_data_sql.py
python3 fix_business_logic.py
```

---

## 🧠 Machine Learning

### Triple Model Architecture

#### 1. Traditional ML (Enhanced)
- **Algorithms:** Random Forest + XGBoost ensemble
- **Features:** 25-30 (technical + fundamental + chaos features)
- **Training:** Walk-forward cross-validation (5 splits)
- **Output:** Price predictions, risk assessment
- **Performance:** R² 0.15-0.35 (good for stocks)
- **Training Time:** 1-2 minutes

#### 2. Raw LSTM (Deep Learning)
- **Architecture:** LSTM neural networks
- **Input:** 60-day OHLCV sequences
- **Prediction:** 14 days ahead
- **Training:** Per-symbol models
- **Coverage:** Top liquid stocks
- **Training Time:** 5-10 minutes per batch

#### 3. Kronos (K-line Tokenization)
- **Approach:** Candlestick pattern recognition
- **Features:** K-line tokens, technical patterns
- **Innovation:** Experimental cutting-edge model
- **Coverage:** All stocks
- **Training Time:** 2-3 minutes

### Dual Strategy System

#### DEFAULT_RISK (Conservative)
```yaml
target_market_cap: "> 20,000 Cr"
pe_ratio: "5-40"
price_range: "₹100-10,000"
liquidity: "High"
target_gain: "7%"
stop_loss: "5%"
typical_stocks: "RELIANCE, TCS, HDFC, INFY"
```

#### HIGH_RISK (Aggressive)
```yaml
target_market_cap: "1,000-20,000 Cr"
pe_ratio: "Broader range"
price_range: "Flexible"
liquidity: "Medium"
target_gain: "12%"
stop_loss: "10%"
typical_stocks: "Mid/Small-cap growth stocks"
```

### Model Performance Metrics

**Total Combinations:** 6 (3 models × 2 strategies)

Each combination provides:
- `ml_prediction_score` (0-1): Higher = better opportunity
- `ml_price_target`: Expected price
- `ml_confidence` (0-1): Model confidence
- `ml_risk_score` (0-1): Lower = safer

### Feature Engineering

**Technical Features:**
- Price momentum, volume patterns
- RSI, MACD, Bollinger Bands
- SMA/EMA crossovers
- ATR volatility

**Fundamental Features:**
- PE, PB, ROE, ROA ratios
- Growth metrics (revenue, earnings)
- Profitability margins
- Liquidity ratios

**Engineered Features:**
- Chaos features (fractal dimension)
- Market regime indicators
- Sector relative strength
- Sentiment scores (optional)

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Fyers API (Required)
FYERS_CLIENT_ID=your_fyers_client_id
FYERS_ACCESS_TOKEN=your_fyers_access_token

# Database (Default - don't change unless needed)
DATABASE_URL=postgresql://trader:trader_password@database:5432/trading_system
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_password
POSTGRES_DB=trading_system

# Redis
REDIS_HOST=dragonfly
REDIS_PORT=6379

# Flask
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your_secret_key_here

# Logging
LOG_LEVEL=INFO

# Screening Parameters (Optional)
SCREENING_QUOTES_RATE_LIMIT_DELAY=0.2
VOLATILITY_MAX_WORKERS=5
VOLATILITY_MAX_STOCKS=500
```

### Stock Filters (`config/stock_filters.yaml`)

```yaml
stage_1_filters:  # Market data screening
  tradeability:
    minimum_price: 50.0
    maximum_price: 10000.0
    minimum_daily_volume: 50000
    minimum_daily_turnover_inr: 50000000

stage_2_filters:  # Business logic screening
  filtering_thresholds:
    minimum_total_score: 25
  fundamental_ratios:
    max_pe_ratio: 50.0
    min_roe: 5.0
    max_debt_equity: 2.0
```

---

## 💼 Usage Examples

### Example 1: Get Top Conservative Picks (Traditional ML)

```bash
curl "http://localhost:5001/api/suggested-stocks/?strategy=default_risk&model_type=traditional&limit=10"
```

### Example 2: Get Aggressive LSTM Picks

```bash
curl "http://localhost:5001/api/suggested-stocks/?strategy=high_risk&model_type=raw_lstm&limit=20"
```

### Example 3: Get All Kronos Predictions

```bash
curl "http://localhost:5001/api/suggested-stocks/?model_type=kronos&limit=50"
```

### Example 4: Query Database for Model Comparison

```sql
-- Connect to database
docker exec -it trading_system_db psql -U trader -d trading_system

-- Compare model performance
SELECT
    model_type,
    strategy,
    COUNT(*) as stocks,
    AVG(ml_prediction_score) as avg_score,
    AVG(ml_confidence) as avg_confidence
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
GROUP BY model_type, strategy
ORDER BY avg_score DESC;
```

---

## 📊 Monitoring

### System Status

```bash
# Complete system status
./tools/check_all_schedulers.sh

# Docker containers
docker compose ps

# View logs (real-time)
docker compose logs -f
docker compose logs -f data_scheduler
docker compose logs -f ml_scheduler
docker compose logs -f trading_system
```

### Database Queries

```sql
-- Model coverage
SELECT
    model_type,
    COUNT(DISTINCT symbol) as unique_stocks,
    COUNT(*) as total_records
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
GROUP BY model_type;

-- Today's top picks across all models
SELECT
    symbol,
    stock_name,
    model_type,
    strategy,
    ml_prediction_score,
    ml_price_target,
    rank
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
ORDER BY ml_prediction_score DESC
LIMIT 20;

-- Pipeline status
SELECT * FROM pipeline_tracking
ORDER BY updated_at DESC
LIMIT 10;
```

---

## 🔧 Troubleshooting

### Common Issues

#### Pipeline Fails at HISTORICAL_DATA
```bash
# Rate limiting issue - increase delay
export SCREENING_QUOTES_RATE_LIMIT_DELAY=0.5

# Reduce parallel workers
export VOLATILITY_MAX_WORKERS=3

# Restart pipeline
python3 run_pipeline.py
```

#### ML Training Fails
```bash
# Check data availability
docker exec -it trading_system_db psql -U trader -d trading_system -c "SELECT COUNT(*) FROM historical_data;"

# Check disk space for models
df -h ml_models/

# Manual training
python3 tools/train_ml_model.py
```

#### Scheduler Not Running
```bash
# Check process
docker compose ps

# Restart schedulers
docker compose restart ml_scheduler data_scheduler

# Check logs
docker compose logs ml_scheduler | tail -50
```

---

## 📈 Performance

### Metrics

- **Data Pipeline:** 20-30 minutes (2,259 stocks)
- **Traditional ML Training:** 1-2 minutes
- **LSTM Training:** 5-10 minutes (batch)
- **Kronos Training:** 2-3 minutes
- **Daily Snapshot:** 3-5 minutes (6 combinations)
- **API Response:** < 100ms (with cache)
- **Database Size:** ~1.6M records, ~500MB
- **Memory Usage:** ~500MB (Flask), ~1GB (PostgreSQL)

### Optimization Tips

1. **Saga Pattern:** Automatic retry prevents failures
2. **Parallel Processing:** Data tasks run in parallel
3. **Model Caching:** Models loaded once, cached in memory
4. **Database Indexes:** Optimized for common queries
5. **Redis Caching:** API responses cached

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[README.md](README.md)** | This file - comprehensive overview |
| **[CLAUDE.md](CLAUDE.md)** | AI assistant instructions & codebase guide |
| **[AUTOMATION.md](docs/AUTOMATION.md)** | Complete automation guide |
| **[ML_GUIDE.md](docs/ML_GUIDE.md)** | Machine learning models documentation |
| **[STRUCTURE.md](docs/STRUCTURE.md)** | Project structure & data flow |
| **[tools/README.md](tools/README.md)** | Utility scripts documentation |

---

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/StockExperiment.git
cd StockExperiment

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start development server
./run.sh dev
```

### Code Style

- **Python:** PEP 8, type hints preferred
- **SQL:** Uppercase keywords, snake_case tables
- **Saga Pattern:** Use for multi-step operations
- **Sessions:** Always use context managers
- **Models:** Lazy loading with caching

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🎉 Summary

✅ **100% Automated** - Data collection, ML training, stock selection, trading execution
✅ **Triple ML Models** - Traditional (RF+XGBoost), Raw LSTM, Kronos
✅ **Dual Strategies** - Conservative (DEFAULT_RISK) and Aggressive (HIGH_RISK)
✅ **6 Combinations Daily** - 3 models × 2 strategies = comprehensive coverage
✅ **2,259+ NSE Stocks** - Complete fundamental and technical data
✅ **Saga Pattern** - Reliable multi-step operations with retry logic
✅ **Auto-Trading** - Automated order placement with risk management
✅ **Performance Tracking** - Daily P&L snapshots and order monitoring
✅ **Multi-Broker Support** - Unified service for Fyers, Zerodha, Simulator
✅ **Production-Ready** - Docker, logging, error handling, monitoring

**Zero manual intervention required!** 🚀

---

## 📞 Support

### Getting Help

1. Check documentation in `/docs` folder
2. Review logs: `docker compose logs -f`
3. Check system status: `./tools/check_all_schedulers.sh`
4. Search issues on GitHub
5. Create new issue with logs and details

### Common Commands

```bash
# Start system
./run.sh dev

# Stop system
docker compose down

# Restart services
docker compose restart data_scheduler ml_scheduler trading_system

# View logs
docker compose logs -f [service_name]

# Check status
./tools/check_all_schedulers.sh

# Manual tasks
python3 run_pipeline.py
python3 tools/train_ml_model.py

# Database access
docker exec -it trading_system_db psql -U trader -d trading_system
```

---

**Built with ❤️ for automated stock trading and analysis**

Last Updated: October 14, 2025