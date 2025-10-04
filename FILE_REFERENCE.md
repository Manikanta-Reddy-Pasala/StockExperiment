# Project Files Reference

## 📁 Root Directory Files

### Production Scripts
- **`app.py`** - Main Flask web application entry point
- **`run.py`** - Application launcher with environment configuration
- **`run.sh`** - Shell script to start containers and application
- **`config.py`** - Application configuration and settings
- **`run_pipeline.py`** - Main data pipeline orchestration (6-step saga)

### Setup & Configuration
- **`configure_fyers.py`** - Interactive script to configure Fyers API credentials
- **`check_status.sh`** - System health check (containers, database, API status)

### Utility Scripts
- **`fill_data_sql.py`** - Populate missing data fields (adj_close, liquidity, ATR, volatility)
- **`fix_business_logic.py`** - Calculate derived financial metrics (EPS, margins, ratios, growth)

### Documentation
- **`README.md`** - Project overview and quick start guide
- **`PRODUCTION_SETUP.md`** - Complete production setup instructions with Fyers API
- **`DEVELOPMENT.md`** - Development guidelines and architecture
- **`AGENTS.md`** - AI agent patterns and best practices
- **`FILE_REFERENCE.md`** - This file - reference for all project files

---

## 🗂️ Key Directories

### `/src` - Source Code
- `/models` - SQLAlchemy models (stocks, historical_data, technical_indicators)
- `/services` - Business logic and API services
  - `/data` - Data pipeline services (saga pattern)
  - `/broker` - Fyers API integration
- `/web` - Flask web routes and templates
- `/utils` - Helper utilities and logging

### `/config` - Configuration Files
- `stock_filters.yaml` - Stock screening criteria and thresholds
- `database.yaml` - Database connection settings
- Other YAML configs

### `/docker` - Docker Configuration
- `docker-compose.yml` - Container orchestration
- Database initialization scripts

---

## 🔄 Typical Workflow

### Initial Setup
1. Start containers: `./run.sh dev`
2. Configure Fyers: `python3 configure_fyers.py`
3. Run pipeline: `python3 run_pipeline.py`
4. Fill missing data: `python3 fill_data_sql.py && python3 fix_business_logic.py`

### Daily Operations
1. Check status: `./check_status.sh`
2. Update data: `python3 run_pipeline.py`
3. Access API: `http://localhost:5001/api/suggested-stocks/`

### Maintenance
1. Re-fill data: `python3 fill_data_sql.py`
2. Recalculate metrics: `python3 fix_business_logic.py`
3. Check logs: `docker compose logs trading_system`

---

## 📊 Data Pipeline Flow

```
run_pipeline.py
    ↓
1. SYMBOL_MASTER (fetch NSE symbols)
    ↓
2. STOCKS (fetch prices, calculate market cap)
    ↓
3. HISTORICAL_DATA (1-year OHLCV data)
    ↓
4. TECHNICAL_INDICATORS (RSI, MACD, SMA, EMA, ATR)
    ↓
5. COMPREHENSIVE_METRICS (volatility calculations)
    ↓
6. PIPELINE_VALIDATION (data quality checks)
```

After pipeline:
- Run `fill_data_sql.py` to populate additional fields
- Run `fix_business_logic.py` to calculate derived metrics

---

## 🚀 API Endpoints

### Main Endpoint
- **GET** `/api/suggested-stocks/`
  - Query params: `strategy`, `limit`, `sector`, `search`, `sort_by`
  - Returns: Filtered & scored stock suggestions with complete financial data

### Data Includes
- Price & Market Data (current_price, market_cap, volume)
- Fundamental Ratios (PE, PB, ROE, debt_to_equity)
- Derived Metrics (EPS, book_value, beta, peg_ratio)
- Growth Metrics (revenue_growth, earnings_growth)
- Profitability (operating_margin, net_margin, profit_margin)
- Liquidity (current_ratio, quick_ratio)
- Risk Metrics (beta, volatility)
- Trading Signals (target_price, stop_loss, recommendation)

---

## 📝 Notes

- All test files and mock data scripts have been removed
- Pipeline uses real Fyers API data only
- Database contains 2,259 active stocks with complete data
- Technical indicators updated during pipeline runs
- Business logic automatically calculates 14+ derived fields
