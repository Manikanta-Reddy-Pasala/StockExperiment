# Scheduler Update Summary - Dual Strategy Support

## What Was Changed

### Problem
The dual model view page was designed to show 4 combinations:
1. Traditional Model + Default Risk ✅ (Had 10 stocks)
2. Traditional Model + High Risk ❌ (Had 0 stocks)
3. Raw LSTM Model + Default Risk ✅ (Had 5 stocks)
4. Raw LSTM Model + High Risk ❌ (Had 0 stocks)

**Root Cause**: The scheduler (`scheduler.py`) was only running `default_risk` strategy, never `high_risk`.

### Solution Implemented

#### 1. Updated `scheduler.py`

**File**: `scheduler.py`

**Changes**:
- Modified `update_daily_snapshot()` function (lines 71-129)
- **Before**: Only ran `strategies=['default_risk']`
- **After**: Runs BOTH `strategies=['default_risk', 'high_risk']` in a loop

**Key Changes**:
```python
# OLD CODE (line 82)
strategies=['default_risk'],

# NEW CODE (lines 78-79)
strategies_to_run = ['default_risk', 'high_risk']
total_stocks_stored = 0

# Loop through both strategies (lines 84-120)
for strategy in strategies_to_run:
    result = orchestrator.execute_suggested_stocks_saga(
        user_id=1,
        strategies=[strategy],
        limit=50
    )
```

**Benefits**:
- ✅ Now generates predictions for BOTH risk levels daily
- ✅ Populates all 4 model/risk combinations automatically
- ✅ Better logging showing individual strategy results
- ✅ Continues if one strategy fails (doesn't break the whole job)

#### 2. Created Manual Generation Script

**File**: `tools/generate_high_risk_predictions.py`

**Purpose**: Generate high_risk predictions immediately without waiting for scheduler

**Features**:
- ✅ Shows current database status before generation
- ✅ Generates high_risk predictions for Traditional model
- ✅ Applies ML predictions and saves to database
- ✅ Verifies data after generation
- ✅ Shows which model/risk combinations are available

---

## How to Use

### Option 1: Run Manual Script (Immediate Results)

Generate high_risk predictions right now:

```bash
python3 tools/generate_high_risk_predictions.py
```

**What it does**:
1. Shows current data status
2. Generates Traditional + High Risk predictions (up to 50 stocks)
3. Applies ML predictions
4. Saves to `daily_suggested_stocks` table
5. Verifies all 4 combinations are available

**Expected Output**:
```
✅ Traditional + default_risk: 10 stocks
✅ Traditional + high_risk:    XX stocks (newly generated)
✅ Raw LSTM + default_risk:     5 stocks
❌ Raw LSTM + high_risk:        0 stocks (needs separate generation)
```

### Option 2: Wait for Scheduler (Automatic)

The scheduler will now automatically generate BOTH strategies daily at **10:15 PM**.

**Scheduler Configuration**:
```
- ML Training:           Daily at 10:00 PM
- Daily Snapshot Update: Daily at 10:15 PM
  → Strategies: DEFAULT_RISK + HIGH_RISK (both)
- Cleanup Old Snapshots: Weekly (Sunday) at 03:00 AM
```

---

## Verification

### Check Database Status

```bash
python3 << 'EOF'
from src.models.database import get_database_manager
from sqlalchemy import text

db = get_database_manager()

with db.get_session() as session:
    query = text("""
        SELECT
            model_type,
            strategy,
            COUNT(*) as count,
            ROUND(AVG(ml_prediction_score)::numeric, 4) as avg_score
        FROM daily_suggested_stocks
        WHERE date = CURRENT_DATE
        GROUP BY model_type, strategy
        ORDER BY model_type, strategy
    """)

    result = session.execute(query)
    rows = result.fetchall()

    print(f"{'Model':<15} {'Strategy':<20} {'Count':<8} {'Avg Score'}")
    print("-" * 60)
    for row in rows:
        print(f"{row[0]:<15} {row[1]:<20} {row[2]:<8} {row[3]}")
EOF
```

### Expected Result (After Running Script)

```
Model           Strategy             Count    Avg Score
------------------------------------------------------------
raw_lstm        raw_lstm             5        0.2680
traditional     default_risk         10       0.5602
traditional     high_risk            XX       0.XXXX  ← NEW!
```

---

## Dual Model View Page

After generating high_risk predictions, the **Suggested Stocks** page will show:

```
┌─────────────────────────────────┬─────────────────────────────────┐
│   TRADITIONAL MODEL             │   RAW LSTM MODEL                │
├─────────────────────────────────┼─────────────────────────────────┤
│ [Default Risk] [High Risk]      │ [Default Risk] [High Risk]      │
│                                 │                                 │
│ Default: 10 stocks ✅           │ Default: 5 stocks ✅            │
│ High:    XX stocks ✅ (NEW!)    │ High:    0 stocks ❌            │
└─────────────────────────────────┴─────────────────────────────────┘
```

**Note**: Raw LSTM + High Risk still needs to be generated separately (different model/data source).

---

## Files Modified

1. **scheduler.py** (lines 71-129, 207)
   - Updated `update_daily_snapshot()` to run both strategies
   - Updated scheduler startup message

2. **tools/generate_high_risk_predictions.py** (NEW)
   - Manual script for immediate high_risk generation
   - Includes verification and status checking

---

## Next Steps

### Immediate Actions

1. **Generate high_risk predictions now**:
   ```bash
   python3 tools/generate_high_risk_predictions.py
   ```

2. **Verify dual model view page**:
   - Login to web app
   - Navigate to Suggested Stocks page
   - Should see Traditional Model with both Default and High Risk tabs
   - High Risk tab should have stocks (if script ran successfully)

3. **Clear browser cache**:
   ```
   Ctrl+Shift+R (Windows/Linux)
   Cmd+Shift+R (Mac)
   ```

### Future Enhancements (Optional)

1. **Generate Raw LSTM + High Risk**:
   - Requires training Raw LSTM model with high_risk strategy
   - Can use similar script but for raw_lstm model

2. **Monitor Scheduler**:
   - Check `logs/scheduler.log` after 10:15 PM
   - Verify both strategies ran successfully

3. **Add More Strategies**:
   - Can extend to support additional risk profiles
   - Update `strategies_to_run` list in scheduler

---

## Troubleshooting

### If high_risk predictions don't appear:

1. **Check if script ran successfully**:
   ```bash
   # Look for errors in output
   python3 tools/generate_high_risk_predictions.py
   ```

2. **Check database directly**:
   ```bash
   python3 << 'EOF'
   from src.models.database import get_database_manager
   from sqlalchemy import text

   db = get_database_manager()
   with db.get_session() as session:
       query = text("SELECT COUNT(*) FROM daily_suggested_stocks WHERE strategy = 'high_risk' AND date = CURRENT_DATE")
       count = session.execute(query).scalar()
       print(f"High risk stocks today: {count}")
   EOF
   ```

3. **Check ML models are trained**:
   ```bash
   ls -lh ml_models/
   # Should see: rf_price_model.pkl, rf_risk_model.pkl, metadata.pkl
   ```

4. **Check Flask logs**:
   ```bash
   # Look for errors when accessing /api/suggested-stocks/dual-model-view
   tail -f logs/app.log
   ```

---

## Summary

✅ **Scheduler now runs BOTH strategies** (default_risk + high_risk)
✅ **Manual script available** for immediate generation
✅ **Dual model view page ready** to show all combinations
⏳ **Waiting for**: high_risk predictions to be generated (run script or wait for scheduler)

**Status**: Ready to use! Just run the manual script to populate high_risk data immediately.
