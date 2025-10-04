# Data Pipeline Scheduler

## Overview

The data scheduler automates **all data collection, calculations, and exports** on a daily basis after market close.

## Complete Schedule

### Morning Tasks
| Time | Task | Description |
|------|------|-------------|
| **6:00 AM** | Symbol Master Update | Refresh NSE symbol list (Monday only) |

### After Market Close (Evening Tasks)
| Time | Task | Description |
|------|------|-------------|
| **9:00 PM** | Data Pipeline | 6-step saga: symbols → stocks → history → indicators → metrics → validation |
| **9:30 PM** | Fill Missing Data | adj_close, liquidity, ATR, volatility, volume averages |
| **9:45 PM** | Business Logic | Calculate EPS, margins, ratios, growth metrics (14+ fields) |
| **10:00 PM** | CSV Export | Export stocks, history, indicators, suggested stocks |
| **10:30 PM** | Data Quality Check | Validate completeness and generate report |

### Machine Learning Tasks
| Time | Task | Description |
|------|------|-------------|
| **2:00 AM** | ML Training | Train Random Forest models with latest data |
| **2:15 AM** | Daily Snapshot | Generate top 50 stocks with ML predictions |
| **3:00 AM** | Cleanup (Sunday) | Delete old snapshots and CSV files |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Complete Daily Workflow                     │
└──────────────────────────────────────────────────────────────┘

Monday 6:00 AM:
┌─────────────────────┐
│ Symbol Master Update│  → Refresh NSE symbols from Fyers
└─────────────────────┘

Daily 9:00 PM (After Market Close):
┌─────────────────────┐
│  Data Pipeline      │  → 6-Step Saga
│  1. SYMBOL_MASTER   │     - Fetch NSE symbols
│  2. STOCKS          │     - Update prices, market cap
│  3. HISTORICAL_DATA │     - Fetch 1-year OHLCV
│  4. TECHNICAL_IND   │     - Calculate RSI, MACD, SMA, EMA
│  5. METRICS         │     - Calculate volatility metrics
│  6. VALIDATION      │     - Validate data quality
└─────────────────────┘

Daily 9:30 PM:
┌─────────────────────┐
│ Fill Missing Data   │  → SQL bulk updates
│  - adj_close        │
│  - liquidity_score  │
│  - ATR (14-day)     │
│  - volatility (1y)  │
│  - volume averages  │
└─────────────────────┘

Daily 9:45 PM:
┌─────────────────────┐
│ Business Logic Calc │  → Calculate 14+ derived fields
│  - EPS, Book Value  │
│  - PEG, ROA         │
│  - Margins          │
│  - Ratios           │
│  - Growth metrics   │
│  - Beta             │
└─────────────────────┘

Daily 10:00 PM:
┌─────────────────────┐
│   CSV Export        │  → Create daily backups
│  - stocks.csv       │     (2,259 stocks)
│  - history_30d.csv  │     (last 30 days)
│  - indicators.csv   │     (latest)
│  - suggested.csv    │     (today's picks)
└─────────────────────┘

Daily 10:30 PM:
┌─────────────────────┐
│ Data Quality Check  │  → Validate & report
│  - Coverage %       │
│  - Completeness     │
│  - Date ranges      │
└─────────────────────┘

Daily 2:00 AM (Next Day):
┌─────────────────────┐
│   ML Training       │  → Train prediction models
│  - Random Forest    │
│  - 365 days data    │
│  - 25+ features     │
└─────────────────────┘

Daily 2:15 AM:
┌─────────────────────┐
│  Daily Snapshot     │  → Top 50 stocks with ML
│  - ML predictions   │
│  - Save to DB       │
│  - Upsert logic     │
└─────────────────────┘
```

## Files Generated

### Daily CSV Exports (`/exports` directory)

1. **stocks_YYYY-MM-DD.csv**
   - All 2,259 stocks with complete fundamental data
   - Includes: price, market cap, ratios, metrics

2. **historical_30d_YYYY-MM-DD.csv**
   - Last 30 days of OHLCV data for all stocks
   - Useful for recent price analysis

3. **technical_indicators_YYYY-MM-DD.csv**
   - Latest technical indicators for all stocks
   - RSI, MACD, SMA, EMA, ATR

4. **suggested_stocks_YYYY-MM-DD.csv**
   - Today's top stock picks with ML predictions
   - Ready for trading decisions

### Log Files

- `logs/data_scheduler.log` - Data pipeline logs
- `logs/scheduler.log` - ML scheduler logs

## Running the Schedulers

### Production (Docker)

```bash
# Start all services
docker compose up -d

# View data scheduler logs
docker compose logs -f data_scheduler

# View ML scheduler logs
docker compose logs -f ml_scheduler

# Check status
docker compose ps
```

### Development (Local)

```bash
# Run data scheduler
python3 data_scheduler.py

# Run ML scheduler
python3 scheduler.py
```

## Manual Task Execution

### Run Data Pipeline Manually
```bash
python3 run_pipeline.py
```

### Fill Missing Data Manually
```bash
python3 fill_data_sql.py
```

### Calculate Business Logic Manually
```bash
python3 fix_business_logic.py
```

### Train ML Models Manually
```bash
python3 train_ml_model.py
```

## Configuration

### Change Schedule Times

Edit `data_scheduler.py`:

```python
# Symbol master (default: Monday 6 AM)
schedule.every().monday.at("06:00").do(update_symbol_master)

# Data pipeline (default: 9 PM)
schedule.every().day.at("21:00").do(run_data_pipeline)

# Fill missing data (default: 9:30 PM)
schedule.every().day.at("21:30").do(fill_missing_data)

# Business logic (default: 9:45 PM)
schedule.every().day.at("21:45").do(calculate_business_logic)

# CSV export (default: 10 PM)
schedule.every().day.at("22:00").do(export_daily_csv)

# Quality check (default: 10:30 PM)
schedule.every().day.at("22:30").do(validate_data_quality)
```

### Change CSV Retention Period

Edit `data_scheduler.py`:

```python
# Default: 30 days
cleanup_old_csv_files(export_dir, keep_days=30)

# Change to 60 days
cleanup_old_csv_files(export_dir, keep_days=60)
```

## Monitoring

### Check Scheduler Status

```bash
# Data scheduler status
docker compose ps data_scheduler

# ML scheduler status
docker compose ps ml_scheduler

# View recent logs
docker compose logs --tail=50 data_scheduler
docker compose logs --tail=50 ml_scheduler
```

### Check CSV Exports

```bash
# List exported files
ls -lh exports/

# Latest exports
ls -lt exports/ | head -10

# Check file sizes
du -sh exports/*.csv
```

### Verify Data Updates

```sql
-- Check latest stock updates
SELECT MAX(last_updated) as latest_update FROM stocks;

-- Check historical data coverage
SELECT 
    COUNT(DISTINCT symbol) as symbols,
    MAX(date) as latest_date,
    MIN(date) as earliest_date
FROM historical_data;

-- Check technical indicators
SELECT 
    COUNT(DISTINCT symbol) as symbols_with_indicators,
    MAX(date) as latest_indicators
FROM technical_indicators;

-- Today's suggested stocks
SELECT COUNT(*) FROM daily_suggested_stocks WHERE date = CURRENT_DATE;
```

## Troubleshooting

### Data Pipeline Failures

**Issue**: Pipeline times out or fails

**Solutions**:
```bash
# Check Fyers API credentials
docker compose exec data_scheduler env | grep FYERS

# Check database connection
docker compose exec data_scheduler python3 -c "from src.models.database import get_database_manager; db = get_database_manager(); print('✅ Connected')"

# Check available disk space
df -h

# Increase timeout in data_scheduler.py
# Change timeout=3600 to timeout=7200 (2 hours)
```

### CSV Export Failures

**Issue**: CSV files not created

**Solutions**:
```bash
# Check exports directory exists
mkdir -p exports

# Check permissions
chmod 755 exports

# Check pandas installation
docker compose exec data_scheduler python3 -c "import pandas; print(pandas.__version__)"
```

### Missing Data Issues

**Issue**: Data fields still empty after fill_data_sql.py

**Causes**:
- Pipeline didn't run completely
- Fyers API rate limits
- Insufficient source data

**Solutions**:
```bash
# Re-run pipeline manually
docker compose exec data_scheduler python3 run_pipeline.py

# Then fill data
docker compose exec data_scheduler python3 fill_data_sql.py

# Then business logic
docker compose exec data_scheduler python3 fix_business_logic.py
```

## Performance Optimization

### Pipeline Duration

Typical timings (2,259 stocks):
- **Data Pipeline**: 15-30 minutes
- **Fill Missing Data**: 2-5 minutes
- **Business Logic**: 2-5 minutes
- **CSV Export**: 1-2 minutes
- **Total**: ~25-45 minutes

### Memory Usage

- Data Pipeline: ~500MB
- ML Training: ~1GB
- CSV Export: ~200MB

### Database Growth

- Historical data: ~70K rows/day
- Technical indicators: ~2.3K rows/day
- Daily snapshots: 50 rows/day
- Total growth: ~1MB/day

## Best Practices

1. **Monitor First Week**: Check logs daily to ensure smooth operation
2. **Disk Space**: Keep at least 10GB free for exports and logs
3. **API Limits**: Fyers API has rate limits - don't run manual tasks during scheduled times
4. **Backup**: Export directory contains valuable CSV backups
5. **Timezone**: Ensure server timezone is IST for correct market timing

## Daily Checklist

✅ Morning (check yesterday's run):
- [ ] Check data_scheduler logs for errors
- [ ] Verify CSV exports in `/exports`
- [ ] Check suggested_stocks table has today's data
- [ ] Review data quality report in logs

✅ Weekly (Monday morning):
- [ ] Verify symbol master updated (6 AM)
- [ ] Check total symbol count hasn't dropped significantly

✅ Monthly:
- [ ] Review disk space usage
- [ ] Archive old CSV files if needed
- [ ] Check ML model performance metrics

## Integration with ML Scheduler

The data scheduler works in tandem with ML scheduler:

```
9:00 PM - 10:30 PM:  Data Collection & Processing
                      ↓
2:00 AM:             ML Training (uses fresh data)
                      ↓
2:15 AM:             Daily Snapshot (applies ML to stocks)
```

This ensures ML models always train on the most recent data from the previous day's market.

## Support

For issues:
1. Check logs: `docker compose logs data_scheduler`
2. Review DATA_SCHEDULER.md (this file)
3. Test manually: `python3 run_pipeline.py`
4. Verify database connectivity and API credentials
