# ML Training Guide - Status & Monitoring

**Quick Status Check:** `python3 tools/check_ml_status.py`

---

## Current Status

❌ **ML models are NOT trained**

The training you started earlier timed out after 10 minutes. The process was interrupted before completing all 5 cross-validation folds.

**What happened:**
- Training started and progressed through Fold 4/5
- Command timed out at 10-minute mark
- Training was interrupted, models were NOT saved
- System reverted to "not trained" state

---

## Training Options

### Option 1: Train in Foreground (Watch Progress)

```bash
python3 tools/train_ml_models.py
```

**Pros:**
- See real-time progress
- Know exactly when it's done
- See any errors immediately

**Cons:**
- Terminal must stay open
- Takes 10-15 minutes
- Can't do other work in that terminal

**Expected Output:**
```
================================================================================
ENHANCED ML MODEL TRAINING
================================================================================
Started at: 2025-10-07 21:55:30

This will take 10-15 minutes due to:
  • 5-fold walk-forward cross-validation
  • Training on ~500k samples
  • RF + XGBoost ensemble (4 models total)
  • Chaos theory feature engineering

Training with walk-forward cross-validation...
Fold 1/5 - Price R²: 0.041, Risk R²: 0.110
Fold 2/5 - Price R²: 0.052, Risk R²: 0.116
Fold 3/5 - Price R²: 0.046, Risk R²: 0.150
Fold 4/5 - Price R²: 0.057, Risk R²: 0.104
Fold 5/5 - Price R²: 0.048, Risk R²: 0.112

✅ TRAINING COMPLETE!
⏱️  Training Time: 12.3 minutes
```

---

### Option 2: Train in Background (Recommended)

```bash
# Start training in background
nohup python3 tools/train_ml_models.py > training.log 2>&1 &

# Get the process ID
echo $!
# Output: 12345 (remember this number)
```

**Monitor progress:**
```bash
# Watch live updates
tail -f training.log

# Check last 20 lines
tail -20 training.log

# Search for specific info
grep "Fold" training.log
grep "COMPLETE" training.log
```

**Check if still running:**
```bash
# Check by process ID (replace 12345 with your PID)
ps -p 12345

# Or search for training process
ps aux | grep train_ml_models
```

**When complete:**
```bash
# Verify models are trained
python3 tools/check_ml_status.py
```

---

### Option 3: Wait for Scheduled Training

**Next scheduled run:** Today at **10:00 PM** (22:00)

The ML scheduler runs automatically every night after the data pipeline completes.

**Monitor scheduler:**
```bash
# Check scheduler logs
docker logs trading_system_ml_scheduler --tail 50 --follow

# Check if training is scheduled
docker logs trading_system_ml_scheduler | grep "ML Training"
```

**Pros:**
- Fully automated
- No manual intervention needed
- Models stay fresh with nightly updates

**Cons:**
- Must wait until 10:00 PM
- Can't test saga flows until then

---

### Option 4: Trigger via Docker Scheduler

```bash
# Manually trigger training via scheduler container
docker exec trading_system_ml_scheduler python3 -c \
  'from scheduler import train_ml_models; train_ml_models()'
```

**Monitor in real-time:**
```bash
# In another terminal, watch logs
docker logs trading_system_ml_scheduler --follow
```

---

## Monitoring Commands Reference

### Check ML Status
```bash
# Quick status check
python3 tools/check_ml_status.py

# Check from Python directly
python3 -c "
from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor

with get_database_manager().get_session() as session:
    predictor = EnhancedStockPredictor(session)
    if predictor.rf_price_model:
        print('✅ Models are trained')
    else:
        print('❌ Models NOT trained')
"
```

### Monitor Background Training
```bash
# Watch training log in real-time
tail -f training.log

# Check training progress
grep -i "fold\|complete\|error" training.log

# Check if process is running
ps aux | grep train_ml_models | grep -v grep
```

### Monitor Docker Scheduler
```bash
# Watch scheduler logs
docker logs trading_system_ml_scheduler --follow

# Check recent scheduler activity
docker logs trading_system_ml_scheduler --tail 100

# Check scheduled times
docker logs trading_system_ml_scheduler | grep "Scheduled Tasks" -A 5
```

### Check Training Performance
```bash
# After training completes, check stats
python3 -c "
from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor

with get_database_manager().get_session() as session:
    predictor = EnhancedStockPredictor(session)
    if hasattr(predictor, 'training_stats'):
        stats = predictor.training_stats
        print(f'Samples: {stats.get(\"samples\")}')
        print(f'Features: {stats.get(\"features\")}')
        print(f'Price R²: {stats.get(\"price_r2\", 0):.3f}')
        print(f'CV R²: {stats.get(\"cv_price_r2\", 0):.3f}')
"
```

---

## Training Timeline

**Typical Training Process:**

| Time | Activity | Status |
|------|----------|--------|
| 0:00 | Start training | Preparing data |
| 0:30 | Data loaded | 494k samples |
| 1:00 | Features engineered | 40 features + chaos |
| 2:00 | Fold 1/5 complete | R² ~0.04 |
| 4:00 | Fold 2/5 complete | R² ~0.05 |
| 6:00 | Fold 3/5 complete | R² ~0.05 |
| 8:00 | Fold 4/5 complete | R² ~0.06 |
| 10:00 | Fold 5/5 complete | R² ~0.05 |
| 12:00 | Training complete | Models saved ✅ |

**Total time:** 10-15 minutes (depends on system)

---

## After Training Completes

### 1. Verify Models
```bash
python3 tools/check_ml_status.py
```

Expected output:
```
✅ ML MODELS ARE TRAINED
📊 Model Information:
  ✓ Price Model: Trained
  ✓ Risk Model: Trained
```

### 2. Test Risk Profiles
```bash
python3 tools/test_risk_profiles.py
```

This will:
- Test DEFAULT_RISK strategy (15 stocks)
- Test HIGH_RISK strategy (15 stocks)
- Compare the two strategies
- Validate they produce different results
- Check ML predictions are included

**Expected time:** 2-3 minutes

### 3. Full Saga Test
```bash
python3 tools/test_complete_saga_flow.py
```

This validates all 7 saga steps end-to-end.

**Expected time:** 5 minutes

---

## Troubleshooting

### Training Hangs or Takes Too Long

**Check system resources:**
```bash
# CPU usage
top -l 1 | grep "CPU usage"

# Memory usage
top -l 1 | grep PhysMem

# Check if database is responding
docker exec trading_system_db_dev pg_isready
```

**Reduce training load:**
Edit `tools/train_ml_models.py` and change:
```python
# From:
stats = predictor.train_with_walk_forward(lookback_days=365, n_splits=5)

# To (faster but less accurate):
stats = predictor.train_with_walk_forward(lookback_days=180, n_splits=3)
```

### Training Fails with Error

**Check database connection:**
```bash
python3 -c "
from src.models.database import get_database_manager
db = get_database_manager()
with db.get_session() as session:
    print('✅ Database connected')
"
```

**Check data availability:**
```bash
python3 -c "
from src.models.database import get_database_manager
from sqlalchemy import text

with get_database_manager().get_session() as session:
    result = session.execute(text('SELECT COUNT(*) FROM historical_data'))
    count = result.scalar()
    print(f'Historical data rows: {count:,}')

    if count > 100000:
        print('✅ Sufficient data for training')
    else:
        print('❌ Insufficient data - run data pipeline first')
"
```

### Models Don't Save

**Check if models are being created:**
```bash
# During training, check memory objects
ps aux | grep train_ml_models

# After training, verify predictor state
python3 tools/check_ml_status.py
```

---

## Quick Command Reference

```bash
# STATUS CHECKS
python3 tools/check_ml_status.py                    # Check if models trained
ps aux | grep train_ml                              # Check if training running
tail -f training.log                                 # Monitor background training

# TRAINING
python3 tools/train_ml_models.py                    # Train in foreground
nohup python3 tools/train_ml_models.py > training.log 2>&1 &  # Train in background

# TESTING (after training)
python3 tools/test_risk_profiles.py                 # Test DEFAULT vs HIGH risk
python3 tools/test_complete_saga_flow.py            # Full saga validation

# SCHEDULER
docker logs trading_system_ml_scheduler --follow    # Watch scheduler
docker logs trading_system_ml_scheduler --tail 50   # Recent logs
```

---

## Recommended Workflow

**For immediate testing:**
```bash
# 1. Train in background
nohup python3 tools/train_ml_models.py > training.log 2>&1 &

# 2. Monitor progress (in another terminal)
tail -f training.log

# 3. Wait for "TRAINING COMPLETE" message (10-15 min)

# 4. Verify models
python3 tools/check_ml_status.py

# 5. Test risk profiles
python3 tools/test_risk_profiles.py
```

**For production:**
```bash
# Just let the scheduler handle it
# Training runs automatically at 10:00 PM daily
# No manual intervention needed

# Check scheduler status
docker logs trading_system_ml_scheduler | grep "ML Training"
```

---

**Last Updated:** October 7, 2025
**Current Status:** Models not trained - ready to train
