# Project Files Reference

## 📁 Root Directory Files

### Production Scripts
- **`app.py`** - Main Flask web application entry point
- **`run.py`** - Application launcher with environment configuration
- **`run.sh`** - Shell script to start containers and application
- **`config.py`** - Application configuration and settings
- **`run_pipeline.py`** - Main data pipeline orchestration (6-step saga)

### Automation Scripts
- **`data_scheduler.py`** - Data pipeline scheduler (9 PM daily: pipeline → fill → calc → export)
- **`scheduler.py`** - ML scheduler (2 AM daily: train → snapshot → cleanup)

### Tools & Utilities (`/tools`)
- **`train_ml_model.py`** - Manual ML model training script
- **`check_scheduler.sh`** - ML scheduler status check
- **`check_all_schedulers.sh`** - Complete system status (both schedulers + database + exports)
- **`README.md`** - Tools directory documentation

### Documentation
- **`README.md`** - Project overview and quick start guide
- **`PRODUCTION_SETUP.md`** - Complete production setup instructions with Fyers API
- **`DEVELOPMENT.md`** - Development guidelines and architecture
- **`AGENTS.md`** - AI agent patterns and best practices
- **`COMPLETE_AUTOMATION_GUIDE.md`** - Complete automation overview (data + ML)
- **`DATA_SCHEDULER.md`** - Data pipeline scheduler (CSV, history, calculations)
- **`SCHEDULER.md`** - ML scheduler (training & daily snapshots)
- **`ML_IMPLEMENTATION_SUMMARY.md`** - ML technical implementation details
- **`FILE_REFERENCE.md`** - This file - reference for all project files

---

## 🗂️ Key Directories

### `/src` - Source Code
- `/models` - SQLAlchemy models (stocks, historical_data, technical_indicators, daily_suggested_stocks)
- `/services` - Business logic and API services
  - `/data` - Data pipeline services (saga pattern, daily snapshots)
  - `/broker` - Fyers API integration
  - `/ml` - Machine learning models (stock prediction, risk assessment)
- `/web` - Flask web routes and templates
- `/utils` - Helper utilities and logging

### `/tools` - Utility Scripts & Monitoring
- `train_ml_model.py` - Manual ML training
- `check_scheduler.sh` - ML scheduler status
- `check_all_schedulers.sh` - Complete system status
- `README.md` - Tools documentation

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
