# Tools Directory

This directory contains utility scripts and monitoring tools for the trading system.

## 📊 Monitoring Tools

### `check_scheduler.sh`
**Purpose:** Check ML scheduler status and recent activity

**Usage:**
```bash
./tools/check_scheduler.sh
```

**What it checks:**
- ML scheduler container status
- Recent training activity
- Daily snapshot generation
- Cleanup operations
- Database statistics
- Log file sizes

---

### `check_all_schedulers.sh`
**Purpose:** Complete system status check (data + ML schedulers)

**Usage:**
```bash
./tools/check_all_schedulers.sh
```

**What it checks:**
- Both scheduler containers (data + ML)
- Database statistics (stocks, history, indicators, snapshots)
- Today's suggested stocks with ML scores
- Recent CSV exports
- Log file sizes
- Quick summary

**Recommended:** Run this every morning to verify system health

---

## 🤖 Manual Execution Tools

### `train_ml_model.py`
**Purpose:** Manually train ML models

**Usage:**
```bash
python3 tools/train_ml_model.py
```

**What it does:**
- Trains Random Forest models with 365 days of historical data
- Creates price prediction model (2-week targets)
- Creates risk assessment model (drawdown prediction)
- Displays R² scores and sample predictions
- Duration: 1-2 minutes

**When to use:**
- After major data updates
- When ML predictions seem off
- For testing model improvements
- Manual override of scheduled training

---

## 📁 Directory Structure

```
tools/
├── README.md                     # This file
├── check_scheduler.sh            # ML scheduler status
├── check_all_schedulers.sh       # Complete system status
└── train_ml_model.py             # Manual ML training
```

---

## 🔧 Future Tools (Planned)

These tools are referenced in admin dashboard but run via Flask API:

- `configure_fyers.py` - Interactive Fyers API credential setup
- `fill_data_sql.py` - Populate missing data fields
- `fix_business_logic.py` - Calculate derived metrics
- `run_pipeline.py` - Main data pipeline orchestration

Currently, these are triggered via:
- Admin Dashboard UI (`/admin`)
- Automated schedulers (data_scheduler.py, scheduler.py)
- Docker services

---

## 💡 Quick Reference

**Daily Health Check:**
```bash
./tools/check_all_schedulers.sh
```

**Manual ML Training:**
```bash
python3 tools/train_ml_model.py
```

**Check ML Scheduler Only:**
```bash
./tools/check_scheduler.sh
```

---

## 📝 Notes

- All tools are safe to run manually
- They don't interfere with automated schedulers
- Tools provide read-only status or one-time operations
- For continuous automation, use the admin dashboard instead
