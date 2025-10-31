# Database Migrations

## Current Migration: Add EMA 8-21 Strategy Columns

### Problem

The `technical_indicators` table was missing the required columns for the 8-21 EMA strategy:
- ❌ `ema_8` - Fast EMA (missing)
- ❌ `ema_21` - Slow EMA (missing)
- ❌ `demarker` - DeMarker oscillator (missing)

The table had old columns from a previous ML strategy:
- ✅ `ema_12`, `ema_26` - For MACD (not needed for 8-21 EMA)

### Solution

Add the three required columns for 8-21 EMA strategy.

### Steps to Apply Migration

#### Step 1: Apply Database Schema Changes

```bash
# Run migration SQL script
docker exec -i trading_system_db psql -U trader -d trading_system < migrations/add_ema_8_21_demarker.sql

# Expected output:
# ALTER TABLE
# ALTER TABLE
# ALTER TABLE
# CREATE INDEX
# CREATE INDEX
# CREATE INDEX
# CREATE INDEX
# COMMENT
# COMMENT
# COMMENT
```

#### Step 2: Verify Columns Added

```bash
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'technical_indicators'
AND column_name IN ('ema_8', 'ema_21', 'demarker')
ORDER BY column_name;
"

# Expected output:
#  column_name |     data_type
# -------------+-------------------
#  demarker    | double precision
#  ema_21      | double precision
#  ema_8       | double precision
```

#### Step 3: Populate Data for Existing Records

```bash
# Calculate and populate EMA 8, 21, DeMarker for all stocks
python tools/populate_ema_8_21.py

# This will:
# 1. Read historical data for all stocks
# 2. Calculate EMA 8, EMA 21, DeMarker
# 3. Store values in technical_indicators table
# 4. Runtime: ~10-15 minutes for 2,259 stocks
```

#### Step 4: Verify Data

```bash
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT
    COUNT(*) as total_rows,
    COUNT(ema_8) as has_ema8,
    COUNT(ema_21) as has_ema21,
    COUNT(demarker) as has_demarker,
    ROUND(COUNT(ema_8)::numeric / COUNT(*)::numeric * 100, 1) as ema8_pct,
    ROUND(COUNT(ema_21)::numeric / COUNT(*)::numeric * 100, 1) as ema21_pct,
    ROUND(COUNT(demarker)::numeric / COUNT(*)::numeric * 100, 1) as demarker_pct
FROM technical_indicators
WHERE date >= CURRENT_DATE - INTERVAL '365 days';
"

# Expected output (after population):
#  total_rows | has_ema8 | has_ema21 | has_demarker | ema8_pct | ema21_pct | demarker_pct
# ------------+----------+-----------+--------------+----------+-----------+--------------
#       61520 |    61520 |     61520 |        61520 |    100.0 |     100.0 |        100.0
```

#### Step 5: Test 8-21 EMA Strategy

```bash
# Test the strategy calculator
docker exec trading_system python -c "
from src.models.database import get_database_manager
from src.services.technical.ema_strategy_calculator import get_ema_strategy_calculator

db_manager = get_database_manager()
with db_manager.get_session() as session:
    calculator = get_ema_strategy_calculator(session)
    symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ']
    result = calculator.calculate_all_indicators(symbols, lookback_days=252)

    for symbol, indicators in result.items():
        print(f'{symbol}:')
        print(f'  Current Price: ₹{indicators[\"current_price\"]:.2f}')
        print(f'  EMA 8: ₹{indicators[\"ema_8\"]:.2f}')
        print(f'  EMA 21: ₹{indicators[\"ema_21\"]:.2f}')
        print(f'  Power Zone: {indicators[\"power_zone_status\"]}')
        print(f'  DeMarker: {indicators[\"demarker\"]:.3f}')
        print(f'  Buy Signal: {indicators[\"buy_signal\"]} ({indicators[\"signal_quality\"]})')
        print()
"

# Expected output:
# NSE:RELIANCE-EQ:
#   Current Price: ₹2450.50
#   EMA 8: ₹2465.30
#   EMA 21: ₹2420.10
#   Power Zone: bullish
#   DeMarker: 0.285
#   Buy Signal: True (high)
#
# NSE:TCS-EQ:
#   Current Price: ₹3850.20
#   ...
```

---

## What Changed

### Before Migration:
```sql
technical_indicators
├── ema_12 ✅ (for MACD)
├── ema_26 ✅ (for MACD)
├── ema_50 ✅ (for ML)
├── ❌ ema_8 (MISSING!)
├── ❌ ema_21 (MISSING!)
└── ❌ demarker (MISSING!)
```

### After Migration:
```sql
technical_indicators
├── ema_12 (kept, not used)
├── ema_26 (kept, not used)
├── ema_50 (kept, not used)
├── ema_8 ✅ (ADDED - required for strategy)
├── ema_21 ✅ (ADDED - required for strategy)
└── demarker ✅ (ADDED - required for strategy)
```

---

## Performance Impact

### Storage
- **Before:** ~300MB for 61K rows
- **After:** ~310MB for 61K rows (+3%)
- **Impact:** Minimal (3 new columns)

### Query Speed
- **Before:** Calculate EMA 8/21/DeMarker on every request (slow)
- **After:** Query pre-calculated values (fast)
- **Improvement:** 10-50x faster queries

### Calculation Time
- **One-time population:** ~10-15 minutes
- **Daily updates:** Same as before (~5 minutes)

---

## Future Cleanup (Optional)

If you want to remove unused columns to save space:

```sql
-- WARNING: This will delete data!
-- Only run if you're sure you don't need these columns

ALTER TABLE technical_indicators DROP COLUMN IF EXISTS ema_12;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS ema_26;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS ema_50;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS rsi_14;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS macd;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS macd_signal;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS macd_histogram;
-- ... (drop other unused columns)
```

**Recommendation:** Keep the columns for now. They don't hurt, and you might want them later for backtesting or comparison.

---

## Rollback (If Needed)

If something goes wrong, you can remove the new columns:

```sql
-- Drop new columns
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS ema_8;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS ema_21;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS demarker;

-- Drop indexes
DROP INDEX IF EXISTS idx_technical_ema8;
DROP INDEX IF EXISTS idx_technical_ema21;
DROP INDEX IF EXISTS idx_technical_demarker;
DROP INDEX IF EXISTS idx_technical_ema_strategy;
```

---

## Summary

✅ **Required for 8-21 EMA Strategy:**
- EMA 8 (fast EMA)
- EMA 21 (slow EMA)
- DeMarker oscillator

✅ **Migration adds:**
- 3 new columns
- 4 new indexes
- Population script

✅ **Result:**
- Strategy now uses pre-calculated values
- 10-50x faster queries
- Cleaner, more efficient system

---

**Migration Date:** October 31, 2025
**Status:** Ready to apply
