# Complete Automation Guide

## 🎯 Overview

The trading system runs **100% automated** with two schedulers handling all tasks:

1. **Data Scheduler** - Daily data updates after market close (9-10:30 PM)
2. **ML Scheduler** - Machine learning training and predictions (2-3 AM)

---

## 📅 Daily Schedule

### Morning (Monday Only)
```
06:00 AM - Symbol Master Update (refresh NSE symbols)
```

### After Market Close
```
21:00 PM - Data Pipeline (6-step saga: 20-30 min)
21:30 PM - Fill Missing Data (2-5 min)
21:45 PM - Calculate Business Logic (2-5 min)
22:00 PM - Export CSV Files (1-2 min)
22:30 PM - Data Quality Validation
```

### Early Morning (Next Day)
```
02:00 AM - ML Model Training (1-2 min)
02:15 AM - Daily Snapshot - Top 50 stocks with ML (2-3 min)
03:00 AM - Cleanup Old Data (Sunday only)
```

---

## 🚀 Quick Start

### Start All Services
```bash
docker compose up -d
```

### Check System Status
```bash
./tools/check_all_schedulers.sh
```

### View Logs
```bash
# All services
docker compose logs -f

# Data scheduler only
docker compose logs -f data_scheduler

# ML scheduler only
docker compose logs -f ml_scheduler
```

---

## 📊 Data Scheduler (`data_scheduler.py`)

### What It Does

Runs **6 automated tasks** daily after market close:

#### 1. Symbol Master Update (Monday 6 AM)
- Refreshes NSE symbols from Fyers API
- ~2,300 symbols
- Weekly update to catch new listings

#### 2. Data Pipeline (Daily 9 PM)
**6-Step Saga:**
1. Symbol Master (if needed)
2. Stocks (prices, market cap)
3. Historical Data (1-year OHLCV)
4. Technical Indicators (RSI, MACD, SMA, EMA, ATR)
5. Comprehensive Metrics (volatility)
6. Pipeline Validation

**Duration:** 20-30 minutes
**Records:** ~2,259 stocks updated

#### 3. Fill Missing Data (Daily 9:30 PM)
Populates:
- `adj_close`
- `liquidity`
- `atr_14`
- `historical_volatility_1y`
- Volume averages

**Duration:** 2-5 minutes

#### 4. Business Logic Calculations (Daily 9:45 PM)
Calculates 14+ derived fields:
- EPS, Book Value, PEG Ratio, ROA
- Operating/Net/Profit Margins
- Current/Quick Ratios
- Debt to Equity
- Revenue/Earnings Growth

**Duration:** 2-5 minutes

#### 5. CSV Export (Daily 10 PM)
Exports 4 files:
- `stocks_YYYY-MM-DD.csv`
- `historical_30d_YYYY-MM-DD.csv`
- `technical_indicators_YYYY-MM-DD.csv`
- `suggested_stocks_YYYY-MM-DD.csv`

**Location:** `/exports`
**Retention:** 30 days
**Duration:** 1-2 minutes

#### 6. Data Quality Validation (Daily 10:30 PM)
Checks:
- Record counts
- Missing data percentages
- Data freshness
- Anomalies

### Configuration

Edit `data_scheduler.py`:
```python
# Schedule times
schedule.every().monday.at("06:00").do(update_symbol_master)
schedule.every().day.at("21:00").do(run_data_pipeline)
schedule.every().day.at("21:30").do(fill_missing_data)
schedule.every().day.at("21:45").do(calculate_business_logic)
schedule.every().day.at("22:00").do(export_daily_csv)
schedule.every().day.at("22:30").do(validate_data_quality)
```

### Manual Execution

Via Admin Dashboard:
```
http://localhost:5001/admin
```

Or via command line:
```bash
python3 run_pipeline.py
```

---

## 🤖 ML Scheduler (`scheduler.py`)

### What It Does

Runs **3 automated ML tasks** daily in early morning:

#### 1. ML Model Training (Daily 2 AM)
- Trains Random Forest price prediction model
- Trains Random Forest risk assessment model
- Uses 365 days of historical data
- 25-30 features (technical + fundamental)
- 100 trees, max_depth=10
- Parallel processing (n_jobs=-1)

**Duration:** 1-2 minutes
**Output:**
- Price R² score
- Risk R² score
- Model ready for predictions

#### 2. Daily Snapshot (Daily 2:15 AM)
- Runs suggested stocks saga
- Applies ML predictions to all stocks
- Selects top 50 stocks
- Saves to `daily_suggested_stocks` table
- Upserts (replaces same-day data)

**Duration:** 2-3 minutes
**Output:** Top 50 stocks with ML scores

#### 3. Cleanup Old Data (Sunday 3 AM)
- Removes snapshots older than 90 days
- Keeps database size manageable
- Runs weekly

**Duration:** < 1 minute

### Configuration

Edit `scheduler.py`:
```python
# Schedule times
schedule.every().day.at("02:00").do(train_ml_models)
schedule.every().day.at("02:15").do(update_daily_snapshot)
schedule.every().sunday.at("03:00").do(cleanup_old_snapshots)
```

### Manual Execution

Via Admin Dashboard:
```
http://localhost:5001/admin
```

Or via command line:
```bash
python3 tools/train_ml_model.py
```

---

## 🐳 Docker Services

```yaml
services:
  database:           # PostgreSQL
  dragonfly:          # Redis cache
  trading_system:     # Flask API (port 5001)
  data_scheduler:     # Data pipeline automation
  ml_scheduler:       # ML training automation
```

### Service Management

```bash
# Start all
docker compose up -d

# Stop all
docker compose down

# Restart scheduler
docker compose restart data_scheduler
docker compose restart ml_scheduler

# View status
docker compose ps

# View logs
docker compose logs -f data_scheduler
docker compose logs -f ml_scheduler
```

---

## 📊 What Gets Automated

### ✅ Data Collection
- [x] NSE symbol list refresh
- [x] Real-time stock prices
- [x] Market capitalization
- [x] Historical OHLCV (1 year)
- [x] Fundamental ratios (PE, PB, ROE, etc.)

### ✅ Calculations
- [x] Technical indicators (RSI, MACD, SMA, EMA, ATR)
- [x] Volatility metrics (historical, realized)
- [x] Volume averages
- [x] Derived metrics (EPS, margins, growth)
- [x] Risk metrics (Beta, Sharpe ratio)

### ✅ Machine Learning
- [x] Model training (Random Forest)
- [x] Price prediction (2-week targets)
- [x] Risk assessment
- [x] Confidence scoring

### ✅ Stock Selection
- [x] Multi-stage filtering
- [x] Strategy-based scoring
- [x] ML-enhanced ranking
- [x] Daily top 50 picks

### ✅ Data Export
- [x] CSV backups
- [x] Database snapshots
- [x] Quality reports

### ✅ Maintenance
- [x] Cleanup old snapshots (90 days)
- [x] Cleanup old CSVs (30 days)
- [x] Data validation

---

## 🔧 Troubleshooting

### Scheduler Not Running
```bash
# Check container status
docker compose ps data_scheduler ml_scheduler

# Restart
docker compose restart data_scheduler ml_scheduler

# View errors
docker compose logs data_scheduler | grep ERROR
docker compose logs ml_scheduler | grep ERROR
```

### Missing Data
```bash
# Re-run pipeline manually
docker compose exec data_scheduler python3 run_pipeline.py

# Fill data
docker compose exec data_scheduler python3 fill_data_sql.py

# Business logic
docker compose exec data_scheduler python3 fix_business_logic.py
```

### No CSV Files
```bash
# Create directory
mkdir -p exports

# Set permissions
chmod 755 exports

# Run export manually
docker compose exec data_scheduler python3 -c "from data_scheduler import export_daily_csv; export_daily_csv()"
```

### ML Models Not Training
```bash
# Check ML scheduler logs
docker compose logs ml_scheduler | tail -50

# Manual training
python3 tools/train_ml_model.py

# Check database has enough data
docker exec trading_system_db_dev psql -U trader -d trading_system -c "SELECT COUNT(*) FROM historical_data;"
```

---

## 📊 Monitoring

### Daily Health Check
```bash
./tools/check_all_schedulers.sh
```

**What it shows:**
- Container status (Running/Stopped)
- Recent scheduler activity
- Database statistics
- Today's ML predictions
- CSV export status
- Log file sizes

### Database Status
```sql
-- Latest updates
SELECT
    'Stocks' as table_name,
    COUNT(*) as records,
    MAX(last_updated) as last_update
FROM stocks;

-- Today's ML picks
SELECT symbol, stock_name, ml_prediction_score, ml_price_target
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
ORDER BY rank;

-- Data coverage
SELECT
    COUNT(*) as total_stocks,
    COUNT(current_price) as with_price,
    COUNT(ml_prediction_score) as with_ml
FROM stocks s
LEFT JOIN daily_suggested_stocks dss
    ON s.symbol = dss.symbol
    AND dss.date = CURRENT_DATE;
```

---

## 💡 Tips for Production

1. **Monitor First Week**: Check logs daily for the first week
2. **Disk Space**: Keep 10GB+ free for exports and logs
3. **API Limits**: Don't run manual tasks during scheduled times
4. **Backup**: Export CSVs are your backup
5. **Timezone**: Ensure IST for correct market timing
6. **Alerts**: Set up notifications for critical failures

---

## 📝 Environment Variables

```bash
# .env file
FYERS_CLIENT_ID=your_client_id
FYERS_ACCESS_TOKEN=your_access_token
DATABASE_URL=postgresql://trader:trader_password@database:5432/trading_system
```

---

## ✅ Success Metrics

### Data Quality
- Stocks coverage: **99%+** with prices
- Historical data: **1 year** OHLCV
- Technical indicators: **100%** calculated
- Missing fields: **<1%**

### Automation
- Uptime: **99%+** (restarts on failure)
- On-time execution: **100%**
- Failed tasks: **<1%**

### Performance
- Pipeline duration: **20-30 minutes**
- ML training: **1-2 minutes**
- Total daily downtime: **0 minutes**

---

## 🎉 Summary

You now have:
- ✅ **Fully automated** data collection (9 PM daily)
- ✅ **Automated** calculations & metrics (9:30-9:45 PM)
- ✅ **Automated** CSV exports (10 PM daily)
- ✅ **Automated** ML training (2 AM daily)
- ✅ **Automated** stock selection (2:15 AM daily)
- ✅ **Automated** cleanup (Sunday 3 AM)

**Zero manual intervention required!** 🚀

Just check the logs and review the results each morning.
