# Trading System Scheduler

## Overview

The scheduler automatically runs ML training and updates daily stock snapshots at scheduled times.

## Scheduled Tasks

| Task | Schedule | Description |
|------|----------|-------------|
| **ML Model Training** | Daily at 2:00 AM | Trains Random Forest models with latest historical data (365 days) |
| **Daily Snapshot Update** | Daily at 2:15 AM | Generates fresh stock recommendations with ML predictions |
| **Cleanup Old Snapshots** | Weekly (Sunday) at 3:00 AM | Deletes snapshots older than 90 days |

## How It Works

### 1. ML Model Training (2:00 AM)
- Fetches 1 year of historical stock data
- Trains price prediction model (Random Forest Regressor)
- Trains risk assessment model (predicts max drawdown)
- Features used: 25+ technical indicators + fundamentals
- Models are persisted in memory for the day

### 2. Daily Snapshot Update (2:15 AM)
- Runs suggested stocks saga with default_risk strategy
- Applies ML predictions to top 50 stocks
- Saves to `daily_suggested_stocks` table
- **Upsert logic**: Replaces same-day data if run multiple times

### 3. Cleanup (Sunday 3:00 AM)
- Removes snapshots older than 90 days
- Keeps database size manageable

## Running the Scheduler

### Development (Local)
```bash
# Install dependencies
pip install schedule

# Run scheduler
python3 scheduler.py
```

### Production (Docker Compose)
```bash
# Start all services including scheduler
docker compose up -d

# View scheduler logs
docker compose logs -f scheduler

# Check scheduler status
docker compose ps scheduler
```

### Manual Tasks

Run ML training manually:
```bash
python3 train_ml_model.py
```

Update daily snapshot manually:
```bash
python3 test_ml_integration.py
```

## Logs

Scheduler logs are written to:
- `logs/scheduler.log` (file)
- STDOUT (console)

View logs:
```bash
# Docker
docker compose logs -f scheduler

# Local
tail -f logs/scheduler.log
```

## Configuration

### Changing Schedule Times

Edit `scheduler.py`:
```python
# Change ML training time (default: 02:00)
schedule.every().day.at("02:00").do(train_ml_models)

# Change snapshot update time (default: 02:15)
schedule.every().day.at("02:15").do(update_daily_snapshot)

# Change cleanup day/time (default: Sunday 03:00)
schedule.every().sunday.at("03:00").do(cleanup_old_snapshots)
```

### Changing Snapshot Retention

Edit `scheduler.py`:
```python
# Default: 90 days
deleted = snapshot_service.delete_old_snapshots(keep_days=90)

# Change to 30 days
deleted = snapshot_service.delete_old_snapshots(keep_days=30)
```

### Running Tasks on Startup

Uncomment these lines in `scheduler.py` to run tasks immediately when scheduler starts:
```python
# Run immediately on startup (optional - for testing)
logger.info("Running initial ML training...")
train_ml_models()
time.sleep(5)
logger.info("Running initial snapshot update...")
update_daily_snapshot()
```

## Monitoring

### Check if Scheduler is Running
```bash
docker compose ps scheduler
```

### View Recent Snapshots
```sql
SELECT date, COUNT(*) as stocks, 
       COUNT(ml_prediction_score) as with_ml_scores
FROM daily_suggested_stocks
GROUP BY date
ORDER BY date DESC
LIMIT 7;
```

### Check ML Training Success
```bash
# Look for these log messages:
grep "ML Training Complete" logs/scheduler.log
grep "Price Model R²" logs/scheduler.log
```

## Troubleshooting

### Scheduler Not Running

1. Check if container is running:
```bash
docker compose ps scheduler
```

2. Check logs for errors:
```bash
docker compose logs scheduler
```

3. Restart scheduler:
```bash
docker compose restart scheduler
```

### ML Training Failures

Common issues:
- **Insufficient data**: Need at least 100 training samples
- **Database connection**: Check DATABASE_URL environment variable
- **Memory issues**: ML training requires ~1GB RAM

Solutions:
```bash
# Check database connection
docker exec trading_system_scheduler python3 -c "from src.models.database import get_database_manager; db = get_database_manager(); print('✅ Database connected')"

# Check available data
docker exec trading_system_db_dev psql -U trader -d trading_system -c "SELECT COUNT(*) FROM historical_data;"
```

### Snapshot Update Failures

Check Fyers API credentials:
```bash
# View environment variables
docker compose exec scheduler env | grep FYERS
```

## Architecture

```
scheduler.py
├── train_ml_models()           # 2:00 AM daily
│   ├── StockMLPredictor.train()
│   └── Saves models in memory
│
├── update_daily_snapshot()      # 2:15 AM daily
│   ├── SuggestedStocksSaga.execute()
│   │   ├── Step 1-5: Stock filtering
│   │   ├── Step 6: ML predictions
│   │   └── Step 7: Daily snapshot save
│   └── DailySnapshotService.save()
│
└── cleanup_old_snapshots()      # Sunday 3:00 AM
    └── DailySnapshotService.delete_old_snapshots(90 days)
```

## Best Practices

1. **Timezone**: Scheduler uses server time. Set TZ environment variable if needed.
2. **API Limits**: 2:15 AM avoids market hours (no rate limiting issues)
3. **Gap Between Tasks**: 15-minute gap ensures ML training completes before snapshot
4. **Restart Policy**: `restart: unless-stopped` ensures scheduler survives system reboots
5. **Monitoring**: Check logs daily for the first week to ensure smooth operation

## Future Enhancements

Potential additions:
- Email/Slack notifications on task completion/failure
- Performance metrics dashboard
- Configurable schedule via environment variables
- Multi-strategy snapshots (conservative, aggressive, balanced)
- Backtest historical snapshots for ML model accuracy
