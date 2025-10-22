# Deployment Instructions - Production Release

## 🚀 New Features Ready for Production

### 1. ML Predictions on ALL Stocks (Not Just Filtered Subset)
- **Script:** `tools/generate_ml_all_stocks.py`
- **What it does:** Predicts ALL ~2,259 NSE stocks with all 3 models
- **Schedule:** Daily at 6:30 AM (after training, before daily snapshot)
- **Duration:** ~10-15 minutes
- **Output:** `ml_predictions` table with comprehensive predictions

### 2. Token Expiry Monitoring (Dashboard + Auto-refresh)
- **UI:** Fyers dashboard shows expiry time and countdown
- **Alerts:** Visual warnings when token expires soon (<12 hours)
- **Auto-refresh:** Background thread checks every 30 minutes
- **Scheduled checks:** Every 6 hours with detailed logging

### 3. Paper Trading (Already Implemented)
- **Status:** Fully working, just needed documentation
- **Default:** All users in paper mode by default
- **Guides:** `PAPER_TRADING_GUIDE.md` and `ML_ALL_STOCKS_GUIDE.md`

---

## 📋 Pre-Deployment Checklist

Before deploying to production, verify:

- [ ] Docker Desktop is installed and running
- [ ] PostgreSQL data volume exists (or will be created fresh)
- [ ] Fyers API credentials are configured
- [ ] Port 5001 is available (web app)
- [ ] Port 5432 is available (database)
- [ ] Sufficient disk space (~2GB minimum)
- [ ] ML models are trained (or will be trained on first run)

---

## 🔧 Deployment Steps

### Step 1: Start Docker Desktop

```bash
# On macOS
open -a Docker

# Wait for Docker to fully start (check menu bar icon)
# Should show "Docker Desktop is running"
```

### Step 2: Navigate to Project Directory

```bash
cd /Users/manip/Documents/codeRepo/poc/StockExperiment
```

### Step 3: Pull Latest Changes (if deploying from remote)

```bash
# If changes were pushed to git
git pull origin main
```

### Step 4: Start Production Services

```bash
# Option 1: Full production mode (recommended)
./run.sh start

# Option 2: Development mode (for testing)
./run.sh dev
```

**Expected output:**
```
[INFO] Creating necessary directories...
[INFO] Starting Trading System with Docker Compose...
Creating network "stockexperiment_default" ...
Creating volume "stockexperiment_postgres_data" ...
Creating stockexperiment_database_1 ...
Creating stockexperiment_dragonfly_1 ...
Creating stockexperiment_trading_system_1 ...
Creating stockexperiment_ml_scheduler_1 ...
Creating stockexperiment_data_scheduler_1 ...

[INFO] Trading System is starting...
[INFO] Web UI: http://localhost:5001
[INFO] Admin Panel: http://localhost:5001/admin
[INFO] Database: localhost:5432
```

### Step 5: Verify Services Are Running

```bash
docker compose ps
```

**Expected output:**
```
NAME                                  STATUS    PORTS
stockexperiment_database_1            Up        0.0.0.0:5432->5432/tcp
stockexperiment_dragonfly_1           Up        0.0.0.0:6379->6379/tcp
stockexperiment_trading_system_1      Up        0.0.0.0:5001->5001/tcp
stockexperiment_ml_scheduler_1        Up
stockexperiment_data_scheduler_1      Up
```

### Step 6: Check Logs (First 5 Minutes)

```bash
# Check all logs
docker compose logs -f

# Or check specific services
docker compose logs -f ml_scheduler
docker compose logs -f trading_system
```

**Look for:**
- ✅ "Database initialized successfully"
- ✅ "Flask app is running"
- ✅ "ML scheduler started"
- ✅ "Token monitoring initialized"

### Step 7: Access Web Interface

Open browser and navigate to:
```
http://localhost:5001
```

**Login credentials:**
- Username: `admin` (or your configured username)
- Password: Your configured password

### Step 8: Verify Token Monitoring

Navigate to Fyers broker page:
```
http://localhost:5001/brokers/fyers
```

**Check:**
- [ ] "Token Expires" shows date/time
- [ ] "Time Remaining" shows hours (color-coded)
- [ ] Warning banner appears if <12 hours remaining
- [ ] "Refresh Token" button is present

### Step 9: Verify Paper Trading Mode

```bash
# Connect to database
docker exec -it stockexperiment_database_1 psql -U trader -d trading_system

# Check user settings
SELECT id, username, is_mock_trading_mode FROM users;

# Should show: is_mock_trading_mode = TRUE (default)
```

### Step 10: Run ML Predictions on ALL Stocks (Manual Test)

```bash
# Execute inside trading_system container
docker exec -it stockexperiment_trading_system_1 python3 tools/generate_ml_all_stocks.py
```

**Expected output:**
```
================================================================================
GENERATING ML PREDICTIONS FOR ALL STOCKS - ALL 3 MODELS
================================================================================
Model 1: Traditional ML - 2,180 predictions saved
Model 2: Raw LSTM - 432 predictions saved
Model 3: Kronos - 1,795 predictions saved
Total: 4,407 predictions
Duration: 245 seconds (4.1 minutes)
✅ ML PREDICTION GENERATION COMPLETE!
```

### Step 11: Verify ML Predictions in Database

```bash
docker exec -it stockexperiment_database_1 psql -U trader -d trading_system -c "
  SELECT model_type, COUNT(*) as predictions
  FROM ml_predictions
  WHERE prediction_date = CURRENT_DATE
  GROUP BY model_type
  ORDER BY model_type;
"
```

**Expected output:**
```
 model_type  | predictions
-------------+-------------
 kronos      |        1795
 raw_lstm    |         432
 traditional |        2180
```

---

## 📅 Automated Schedule (Daily Tasks)

Once deployed, the system runs these tasks automatically:

### Morning Schedule (Before Market Opens)
```
06:00 AM - Train ALL 3 ML models
           ├─ Traditional ML (RF + XGBoost)
           ├─ Raw LSTM (Deep Learning)
           └─ Kronos (K-line Tokenization)

06:30 AM - Generate ML predictions for ALL stocks ← NEW!
           ├─ Traditional: ~2,180 stocks
           ├─ LSTM: ~432 stocks
           └─ Kronos: ~1,850 stocks

07:00 AM - Generate daily snapshot (top 50 filtered stocks)
           └─ Uses pre-computed predictions from 6:30 AM
```

### Trading Hours
```
09:20 AM - Execute auto-trading (if enabled)
           ├─ Check market sentiment
           ├─ Select top strategies
           ├─ Create orders (paper or live mode)
           └─ Track performance
```

### Evening Schedule (After Market Close)
```
06:00 PM - Update order performance
           ├─ Calculate P&L
           ├─ Check stop-loss/target
           └─ Create daily snapshots
```

### Night Schedule (Data Pipeline)
```
09:00 PM - Run 6-step data pipeline
           ├─ Update symbol master
           ├─ Fetch historical data
           ├─ Calculate indicators
           └─ Validate data quality
```

### Token Monitoring (Continuous)
```
Every 30 min - Auto-refresh check (background)
Every 6 hours - Full status check with logging
12 hours before expiry - Warning logged
```

---

## 🔍 Post-Deployment Verification

### 1. Check Scheduler Logs (Token Monitoring)

```bash
docker compose logs ml_scheduler | grep -i "token"
```

**Should see:**
```
✅ Initializing Token Monitoring
✅ Auto-refresh started for user 1
✅ User 1: Token valid for 18.5 hours
```

### 2. Check ML Prediction Schedule

```bash
docker compose logs ml_scheduler | grep -A 5 "ML Predictions"
```

**Should see:**
```
- ML Predictions (ALL STOCKS): Daily at 06:30 AM IST
  → Predict ALL ~2,259 stocks with all 3 models
  → Saves to ml_predictions table
```

### 3. Check Paper Trading Orders

```bash
docker exec -it stockexperiment_database_1 psql -U trader -d trading_system -c "
  SELECT COUNT(*) as mock_orders, COUNT(*) FILTER (WHERE is_mock_order = FALSE) as real_orders
  FROM orders;
"
```

**Should see:**
```
 mock_orders | real_orders
-------------+-------------
          45 |           0
```
(If is_mock_trading_mode is TRUE for all users)

### 4. Test Web UI Features

**Navigate to each page and verify:**
- [ ] Dashboard loads without errors
- [ ] Fyers broker page shows token expiry
- [ ] Orders page shows paper trading orders
- [ ] Performance tracking works
- [ ] ML predictions are visible (if generated)

---

## 🐛 Troubleshooting

### Issue: Services won't start

```bash
# Check if ports are in use
lsof -i :5001  # Web app port
lsof -i :5432  # Database port

# If occupied, kill the process or change ports in docker-compose.yml
```

### Issue: Database connection errors

```bash
# Check database is running
docker compose ps database

# Check database logs
docker compose logs database

# Restart database
docker compose restart database
```

### Issue: ML predictions script fails

```bash
# Check if models are trained
docker exec -it stockexperiment_trading_system_1 ls -lh ml_models/

# If missing, train manually
docker exec -it stockexperiment_trading_system_1 python3 tools/train_ml_model.py
```

### Issue: Token monitoring not working

```bash
# Check scheduler logs
docker compose logs ml_scheduler | grep -i token

# Restart scheduler
docker compose restart ml_scheduler

# Check token status in database
docker exec -it stockexperiment_database_1 psql -U trader -d trading_system -c "
  SELECT broker_name, is_connected, access_token IS NOT NULL as has_token
  FROM broker_configurations;
"
```

### Issue: Paper trading orders going to broker

```bash
# CRITICAL: Verify paper mode is enabled
docker exec -it stockexperiment_database_1 psql -U trader -d trading_system -c "
  SELECT username, is_mock_trading_mode FROM users;
"

# If FALSE (live trading), change back to TRUE (paper)
docker exec -it stockexperiment_database_1 psql -U trader -d trading_system -c "
  UPDATE users SET is_mock_trading_mode = TRUE WHERE id = 1;
"
```

---

## 📊 Monitoring Production

### View Real-Time Logs

```bash
# All services
docker compose logs -f

# ML scheduler only
docker compose logs -f ml_scheduler

# Data pipeline only
docker compose logs -f data_scheduler

# Web app only
docker compose logs -f trading_system
```

### Check System Health

```bash
# Service status
docker compose ps

# Resource usage
docker stats

# Disk usage
docker system df
```

### Database Queries for Monitoring

```bash
# Connect to database
docker exec -it stockexperiment_database_1 psql -U trader -d trading_system

# Check data freshness
SELECT 'stocks' as table_name, COUNT(*) as records FROM stocks
UNION ALL
SELECT 'historical_data', COUNT(*) FROM historical_data
UNION ALL
SELECT 'ml_predictions', COUNT(*) FROM ml_predictions WHERE prediction_date = CURRENT_DATE;

# Check last ML predictions run
SELECT
    model_type,
    COUNT(*) as predictions,
    MAX(created_at) as last_run
FROM ml_predictions
WHERE prediction_date = CURRENT_DATE
GROUP BY model_type;

# Check auto-trading executions
SELECT execution_date, status, orders_created, total_amount_invested
FROM auto_trading_executions
ORDER BY execution_date DESC
LIMIT 10;
```

---

## 🔄 Restarting Services

```bash
# Restart all services
./run.sh restart

# Or restart specific services
docker compose restart ml_scheduler
docker compose restart trading_system
docker compose restart data_scheduler
```

---

## 🛑 Stopping Production

```bash
# Stop all services (keeps data)
./run.sh stop

# Or stop and remove everything (including data)
./run.sh cleanup
```

---

## 📚 Documentation References

- **Paper Trading:** `PAPER_TRADING_GUIDE.md`
- **ML Predictions:** `ML_ALL_STOCKS_GUIDE.md`
- **General Documentation:** `CLAUDE.md`
- **Deployment Guide:** `DEPLOYMENT.md`

---

## ✅ Deployment Checklist Summary

- [ ] Docker Desktop running
- [ ] Services started (`./run.sh start`)
- [ ] All 5 containers running (`docker compose ps`)
- [ ] Web UI accessible (http://localhost:5001)
- [ ] Token monitoring active (check Fyers page)
- [ ] Paper trading mode enabled (check users table)
- [ ] ML predictions scheduled (check scheduler logs)
- [ ] Database healthy (check connection)
- [ ] Logs are clean (no critical errors)

---

## 🎉 Post-Deployment

Once deployed successfully:

1. **Monitor first 24 hours** - Watch logs for any errors
2. **Verify scheduled tasks** - Check that 6:00 AM, 6:30 AM, 7:00 AM tasks run
3. **Test token refresh** - Wait for token to expire and verify auto-refresh
4. **Review ML predictions** - Check quality of predictions for all stocks
5. **Monitor paper trading** - Verify orders are created correctly

---

**Deployed By:** Claude Code
**Deployment Date:** 2025-10-22
**Version:** v2.0 - All Stocks ML Predictions + Token Monitoring
**Commit:** `44cd4e5` - Add ML predictions for ALL stocks + Token monitoring + Paper trading docs
