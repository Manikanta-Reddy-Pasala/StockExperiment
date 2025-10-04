# Project Structure

## 📁 Directory Layout

```
/StockExperiment
├── README.md                    # Quick start guide
├── app.py                       # Flask application entry point
├── run.py                       # Application launcher
├── run.sh                       # Docker startup script
├── config.py                    # Application configuration
├── run_pipeline.py              # Data pipeline orchestrator (6-step saga)
├── data_scheduler.py            # Data automation (9 PM daily)
├── scheduler.py                 # ML automation (2 AM daily)
│
├── /src                         # Source code
│   ├── /models                  # Database models (SQLAlchemy)
│   ├── /services               # Business logic
│   │   ├── /data               # Data pipeline, sagas
│   │   ├── /broker             # Fyers API integration
│   │   └── /ml                 # Machine learning models
│   ├── /web                    # Flask routes & templates
│   └── /utils                  # Helpers, logging
│
├── /config                      # YAML configurations
│   └── stock_filters.yaml      # Screening criteria
│
├── /docker                      # Docker files
│   └── docker-compose.yml      # Container orchestration
│
├── /tools                       # Utility scripts
│   ├── train_ml_model.py       # Manual ML training
│   ├── check_scheduler.sh      # ML scheduler status
│   └── check_all_schedulers.sh # Complete system status
│
├── /docs                        # Documentation
│   ├── AUTOMATION.md           # Complete automation guide
│   ├── ML_GUIDE.md             # Machine learning guide
│   └── STRUCTURE.md            # This file
│
├── /logs                        # Application logs
│   ├── data_scheduler.log      # Data pipeline logs
│   └── scheduler.log           # ML scheduler logs
│
└── /exports                     # CSV exports (daily)
    ├── stocks_YYYY-MM-DD.csv
    ├── historical_30d_YYYY-MM-DD.csv
    └── suggested_stocks_YYYY-MM-DD.csv
```

---

## 🗄️ Database Tables

### Core Tables (Created by SQLAlchemy)

#### `stocks`
Primary stock data with fundamentals
- symbol, stock_name, current_price, market_cap
- pe_ratio, pb_ratio, roe, eps, beta
- revenue_growth, earnings_growth, margins
- Last updated: Real-time via pipeline

#### `historical_data`
1-year OHLCV data for all stocks
- symbol, date, open, high, low, close, volume
- Records: ~2,259 stocks × 365 days = ~820K rows

#### `technical_indicators`
Calculated indicators for each stock/day
- rsi_14, macd, signal_line, macd_histogram
- sma_50, sma_200, ema_12, ema_26, atr_14
- Records: Same as historical_data

#### `daily_suggested_stocks`
ML predictions & daily top picks
- date, symbol, strategy, rank
- ml_prediction_score, ml_price_target
- ml_confidence, ml_risk_score
- Records: 50 stocks/day × 90 days = ~4,500 rows

#### `pipeline_execution_tracking`
Pipeline saga execution logs
- pipeline_id, step_name, status
- start_time, end_time, error_message

### User & Trading Tables
- `users` - User accounts
- `strategies` - Trading strategies
- `orders` - Order history
- `trades` - Executed trades
- `positions` - Current positions
- `broker_configurations` - API credentials

---

## 🔄 Data Flow

### Daily Pipeline (9 PM - 10:30 PM)

```
data_scheduler.py
    ↓
run_pipeline.py (6-step saga)
    ↓
Step 1: SYMBOL_MASTER
    → Fetch NSE symbols from Fyers
    → Store in stocks table
    ↓
Step 2: STOCKS
    → Fetch current prices, market cap
    → Update stocks table
    ↓
Step 3: HISTORICAL_DATA
    → Fetch 1-year OHLCV for all stocks
    → Store in historical_data table
    ↓
Step 4: TECHNICAL_INDICATORS
    → Calculate RSI, MACD, SMA, EMA, ATR
    → Store in technical_indicators table
    ↓
Step 5: COMPREHENSIVE_METRICS
    → Calculate volatility metrics
    → Update stocks table
    ↓
Step 6: PIPELINE_VALIDATION
    → Verify data quality
    → Log to pipeline_execution_tracking
    ↓
fill_data_sql.py
    → Populate missing fields
    ↓
fix_business_logic.py
    → Calculate derived metrics
    ↓
export_daily_csv()
    → Export to /exports
```

### ML Pipeline (2 AM - 3 AM)

```
scheduler.py
    ↓
Step 1: train_ml_models()
    → Fetch 365 days historical + technical
    → Train Random Forest models
    → Store in memory
    ↓
Step 2: update_daily_snapshot()
    → Run suggested_stocks_saga
        ↓
        Step 1-5: Filter & score stocks
        Step 6: Apply ML predictions
        Step 7: Save top 50 to daily_suggested_stocks
    ↓
Step 3: cleanup_old_snapshots() (Sunday only)
    → Remove snapshots > 90 days old
```

---

## 🚀 Key Scripts

### Production Scripts (Root)

**`run.sh`**
- Starts Docker containers
- Launches Flask app
- Usage: `./run.sh dev` or `./run.sh prod`

**`run_pipeline.py`**
- Orchestrates 6-step data pipeline
- Called by data_scheduler.py
- Duration: 20-30 minutes

**`data_scheduler.py`**
- Automated data collection (9 PM daily)
- Runs: pipeline → fill → calc → export
- Docker service: `data_scheduler`

**`scheduler.py`**
- Automated ML training (2 AM daily)
- Runs: train → snapshot → cleanup
- Docker service: `ml_scheduler`

### Utility Scripts (/tools)

**`train_ml_model.py`**
- Manual ML model training
- Usage: `python3 tools/train_ml_model.py`
- Duration: 1-2 minutes

**`check_all_schedulers.sh`**
- Complete system status check
- Shows: containers, database, exports, logs
- Usage: `./tools/check_all_schedulers.sh`

**`check_scheduler.sh`**
- ML scheduler status only
- Usage: `./tools/check_scheduler.sh`

---

## 📊 Configuration Files

### `/config/stock_filters.yaml`
Stock screening criteria:
```yaml
min_price: 50.0
max_price: 5000.0
min_market_cap_crores: 500
min_volume: 100000
min_rsi: 30
max_rsi: 70
```

### `.env`
Environment variables:
```bash
FYERS_CLIENT_ID=your_client_id
FYERS_ACCESS_TOKEN=your_access_token
DATABASE_URL=postgresql://...
```

### `docker-compose.yml`
Services configuration:
- database (PostgreSQL)
- dragonfly (Redis)
- trading_system (Flask API)
- data_scheduler (Automation)
- ml_scheduler (ML Automation)

---

## 🌐 API Endpoints

### Main Endpoints

**`GET /api/suggested-stocks/`**
- Returns top stocks with ML predictions
- Query params: `strategy`, `limit`, `sector`, `sort_by`
- Response: Stocks with ML scores

**`GET /admin/`**
- Admin dashboard (admin users only)
- Manual task triggers
- Real-time status monitoring

**`GET /admin/system/status`**
- System statistics (JSON)
- Stock counts, latest updates

**`POST /admin/trigger/{task_type}`**
- Manual task execution
- Types: pipeline, fill-data, business-logic, ml-training, csv-export, all

---

## 📝 Logs

### Log Files (`/logs`)

**`data_scheduler.log`**
- Data pipeline execution logs
- CSV exports
- Data quality checks

**`scheduler.log`**
- ML training logs
- Daily snapshot creation
- Cleanup operations

**`app.log`** (if configured)
- Flask application logs
- API requests
- Errors

### Viewing Logs

```bash
# Real-time (all services)
docker compose logs -f

# Specific service
docker compose logs -f data_scheduler
docker compose logs -f ml_scheduler

# Last 100 lines
docker compose logs --tail=100 data_scheduler
```

---

## 🎯 Quick Reference

### Start System
```bash
./run.sh dev
```

### Check Status
```bash
./tools/check_all_schedulers.sh
```

### Manual Tasks
```bash
# Data pipeline
python3 run_pipeline.py

# ML training
python3 tools/train_ml_model.py

# Via Admin Dashboard
http://localhost:5001/admin
```

### Database Access
```bash
# PostgreSQL
docker exec -it trading_system_db_dev psql -U trader -d trading_system

# Redis
docker exec -it trading_system_redis redis-cli
```

---

## 📚 Documentation

- **`README.md`** (root) - Quick start guide
- **`docs/AUTOMATION.md`** - Complete automation guide
- **`docs/ML_GUIDE.md`** - Machine learning guide
- **`docs/STRUCTURE.md`** - This file
- **`tools/README.md`** - Tools documentation

---

## 🐳 Docker Services

```yaml
services:
  database:
    image: postgres:15
    port: 5432
    data: /var/lib/postgresql/data

  dragonfly:
    image: docker.dragonflydb.io/dragonflydb/dragonfly
    port: 6379
    purpose: Redis cache

  trading_system:
    build: .
    port: 5001
    purpose: Flask API

  data_scheduler:
    command: python data_scheduler.py
    purpose: Data automation

  ml_scheduler:
    command: python scheduler.py
    purpose: ML automation
```

---

## 🎯 Development Workflow

1. **Start containers:** `./run.sh dev`
2. **Access API:** `http://localhost:5001`
3. **Access Admin:** `http://localhost:5001/admin`
4. **Check logs:** `docker compose logs -f`
5. **Run pipeline manually:** `python3 run_pipeline.py`
6. **Train ML manually:** `python3 tools/train_ml_model.py`

---

## ✅ File Count Summary

- **Root Python scripts:** 6 (production)
- **Tools scripts:** 3 (utilities)
- **Documentation:** 4 files
- **Source code:** ~50+ Python files
- **Config files:** 3-4 YAML files
- **Docker files:** 1 docker-compose.yml

**Total:** Clean, organized, production-ready structure! 🎉
