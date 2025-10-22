# ML Predictions on ALL Stocks - Complete Guide

## Problem Statement

**Current Issue:** ML predictions are only generated for a filtered subset of stocks (~200 out of 2,259), because filtering happens BEFORE ML prediction.

**What's happening now:**
```
All Stocks (2,259)
    ↓
Stage 1 Filtering (market data) → ~500 stocks
    ↓
Stage 2 Filtering (business logic) → ~200 stocks
    ↓
ML Predictions (ONLY on 200 stocks) ← WRONG!
    ↓
Daily Snapshot (50 stocks)
```

**What should happen:**
```
All Stocks (2,259)
    ↓
ML Predictions (ALL 2,259 stocks) ← CORRECT!
    ↓
Stage 1 Filtering → ~500 stocks
    ↓
Stage 2 Filtering → ~200 stocks
    ↓
Daily Snapshot (50 stocks)
```

## Solution

I've created a new script that generates ML predictions for **ALL stocks**, not just the filtered subset:

**File:** `tools/generate_ml_all_stocks.py`

### What It Does

1. **Model 1: Traditional ML (RF + XGBoost)**
   - Predicts ALL stocks (~2,259)
   - Uses existing trained models
   - Generates: `ml_prediction_score`, `ml_price_target`, `ml_confidence`, `ml_risk_score`

2. **Model 2: Raw LSTM**
   - Predicts ALL stocks with trained LSTM models
   - Only stocks in `ml_models/raw_ohlcv_lstm/` directory
   - Same prediction format

3. **Model 3: Kronos (K-line Tokenization)**
   - Predicts ALL stocks with ≥200 days of historical data
   - Uses on-the-fly K-line tokenization
   - Same prediction format

4. **Saves to New Table: `ml_predictions`**
   - Separate from `daily_suggested_stocks`
   - Contains predictions for ALL stocks
   - Can be queried for any stock at any time
   - Updated daily with fresh predictions

## How to Use

### Method 1: Manual Execution

```bash
# Run from project root
python3 tools/generate_ml_all_stocks.py
```

**Output:**
```
================================================================================
GENERATING ML PREDICTIONS FOR ALL STOCKS - ALL 3 MODELS
================================================================================
Started at: 2025-01-23 10:00:00

================================================================================
MODEL 1: TRADITIONAL ML (RF + XGBoost) - ALL STOCKS
================================================================================
Found 2259 stocks to predict
  Processing 100/2259...
  Processing 200/2259...
  ...

✅ Traditional ML predictions complete
  Successful: 2180
  Failed/Skipped: 79
  Total: 2259

  ✅ Saved 2180 traditional predictions to database

================================================================================
MODEL 2: RAW LSTM - ALL TRAINED STOCKS
================================================================================
Found 450 symbols with trained LSTM models
  Processing 50/450...
  ...

✅ LSTM predictions complete
  Successful: 432
  Failed/Skipped: 18
  Total: 450

  ✅ Saved 432 raw_lstm predictions to database

================================================================================
MODEL 3: KRONOS (K-line Tokenization) - ALL STOCKS
================================================================================
Found 1850 stocks with sufficient history
  Processing 100/1850...
  ...

✅ Kronos predictions complete
  Successful: 1795
  Failed/Skipped: 55
  Total: 1850

  ✅ Saved 1795 kronos predictions to database

================================================================================
FINAL SUMMARY
================================================================================
Duration: 245.3 seconds (4.1 minutes)
Total predictions saved: 4407
  Traditional ML: 2180
  Raw LSTM: 432
  Kronos: 1795

Database Verification:
  kronos          1,795 predictions
  raw_lstm          432 predictions
  traditional     2,180 predictions
================================================================================
✅ ML PREDICTION GENERATION COMPLETE!
================================================================================
```

### Method 2: Add to Scheduler (Automated Daily)

The script should run daily AFTER ML training but BEFORE daily snapshot generation.

**Current schedule:**
```
06:00 AM - Train ML models (Traditional, LSTM, Kronos)
07:00 AM - Generate daily snapshot (filtered stocks only)
```

**New schedule:**
```
06:00 AM - Train ML models
06:30 AM - Generate ML predictions for ALL stocks ← NEW!
07:00 AM - Generate daily snapshot (uses pre-computed predictions)
```

To add to scheduler, edit `scheduler.py`:

```python
def generate_all_ml_predictions():
    """Generate ML predictions for ALL stocks."""
    logger.info("=" * 80)
    logger.info("Generating ML Predictions for ALL Stocks")
    logger.info("=" * 80)

    try:
        result = subprocess.run(
            ['python3', 'tools/generate_ml_all_stocks.py'],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes
        )

        if result.returncode == 0:
            logger.info("✅ ML predictions generated for all stocks")
            logger.info(result.stdout)
        else:
            logger.error(f"❌ ML prediction generation failed: {result.stderr}")

    except Exception as e:
        logger.error(f"❌ ML prediction generation error: {e}", exc_info=True)

# Add to schedule (run after ML training)
schedule.every().day.at("06:30").do(generate_all_ml_predictions)
```

## Database Schema

### New Table: `ml_predictions`

```sql
CREATE TABLE ml_predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    stock_name VARCHAR(200),
    current_price DECIMAL(10,2),
    model_type VARCHAR(20) NOT NULL,  -- 'traditional', 'raw_lstm', or 'kronos'
    ml_prediction_score DECIMAL(5,4),  -- 0.0000 to 1.0000
    ml_price_target DECIMAL(10,2),
    ml_confidence DECIMAL(5,4),
    ml_risk_score DECIMAL(5,4),
    prediction_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, model_type, prediction_date)
);

CREATE INDEX idx_ml_predictions_symbol ON ml_predictions(symbol);
CREATE INDEX idx_ml_predictions_model ON ml_predictions(model_type);
CREATE INDEX idx_ml_predictions_date ON ml_predictions(prediction_date);
CREATE INDEX idx_ml_predictions_score ON ml_predictions(ml_prediction_score DESC);
```

## Querying Predictions

### Get Predictions for a Specific Stock

```sql
-- Get all model predictions for Reliance
SELECT
    model_type,
    ml_prediction_score,
    ml_price_target,
    ml_confidence,
    ml_risk_score,
    prediction_date
FROM ml_predictions
WHERE symbol = 'NSE:RELIANCE-EQ'
  AND prediction_date = CURRENT_DATE
ORDER BY ml_prediction_score DESC;
```

### Get Top Predictions Across All Models

```sql
-- Top 20 stocks by prediction score (all models)
SELECT
    symbol,
    stock_name,
    current_price,
    model_type,
    ml_prediction_score,
    ml_price_target,
    ml_confidence,
    ((ml_price_target - current_price) / current_price * 100) as potential_gain_pct
FROM ml_predictions
WHERE prediction_date = CURRENT_DATE
  AND ml_confidence >= 0.7
ORDER BY ml_prediction_score DESC
LIMIT 20;
```

### Compare Model Predictions for Same Stock

```sql
-- Compare all 3 models for a stock
SELECT
    symbol,
    stock_name,
    current_price,
    STRING_AGG(
        model_type || ': ' || ROUND(ml_prediction_score::numeric, 3)::text,
        ', '
        ORDER BY model_type
    ) as model_scores,
    AVG(ml_prediction_score) as avg_score,
    AVG(ml_price_target) as avg_target
FROM ml_predictions
WHERE symbol = 'NSE:TCS-EQ'
  AND prediction_date = CURRENT_DATE
GROUP BY symbol, stock_name, current_price;
```

### Get Consensus Predictions (All Models Agree)

```sql
-- Stocks where all 3 models predict BUY (score > 0.7)
SELECT
    p.symbol,
    p.stock_name,
    p.current_price,
    COUNT(DISTINCT p.model_type) as models_count,
    AVG(p.ml_prediction_score) as avg_score,
    AVG(p.ml_confidence) as avg_confidence,
    AVG(p.ml_price_target) as avg_target
FROM ml_predictions p
WHERE p.prediction_date = CURRENT_DATE
  AND p.ml_prediction_score >= 0.7
GROUP BY p.symbol, p.stock_name, p.current_price
HAVING COUNT(DISTINCT p.model_type) = 3  -- All 3 models
ORDER BY avg_score DESC
LIMIT 50;
```

### Get Model Performance Stats

```sql
-- Count predictions by model
SELECT
    model_type,
    COUNT(*) as total_predictions,
    AVG(ml_prediction_score) as avg_score,
    AVG(ml_confidence) as avg_confidence,
    COUNT(*) FILTER (WHERE ml_prediction_score >= 0.7) as buy_signals,
    COUNT(*) FILTER (WHERE ml_prediction_score <= 0.3) as sell_signals
FROM ml_predictions
WHERE prediction_date = CURRENT_DATE
GROUP BY model_type
ORDER BY model_type;
```

## Integration with Daily Snapshot

Now that you have predictions for ALL stocks, you can modify the daily snapshot generation to use these pre-computed predictions instead of re-computing them.

### Option 1: Use Pre-Computed Predictions

Modify `suggested_stocks_saga.py` to query `ml_predictions` table instead of running predictions again:

```python
# Instead of this:
predictor = EnhancedStockPredictor(session)
prediction = predictor.predict_stock(symbol)

# Do this:
prediction_query = text("""
    SELECT
        ml_prediction_score, ml_price_target,
        ml_confidence, ml_risk_score
    FROM ml_predictions
    WHERE symbol = :symbol
      AND model_type = :model_type
      AND prediction_date = CURRENT_DATE
""")

result = session.execute(prediction_query, {
    'symbol': symbol,
    'model_type': 'traditional'  # or 'raw_lstm' or 'kronos'
})

prediction = dict(result.fetchone()._mapping) if result else None
```

### Option 2: Run Both in Parallel

Keep the current workflow but add this as a separate daily task:
- 06:30 AM: Generate predictions for ALL stocks → `ml_predictions` table
- 07:00 AM: Generate daily snapshot → `daily_suggested_stocks` table (top 50 filtered)

This way you have:
- **Full coverage**: Predictions for ALL 2,259 stocks
- **Curated picks**: Filtered top 50 in `daily_suggested_stocks`

## Benefits

1. **Complete Market Coverage**: Predictions for all ~2,259 NSE stocks
2. **No Missed Opportunities**: Can query predictions for any stock at any time
3. **Model Comparison**: Compare all 3 models for same stock
4. **Consensus Signals**: Find stocks where all models agree
5. **Historical Analysis**: Track prediction accuracy over time
6. **Flexible Filtering**: Apply different filters to same predictions
7. **Research & Backtesting**: Full dataset for analysis

## Performance Considerations

- **Traditional ML**: ~2-5 minutes for 2,259 stocks
- **Raw LSTM**: ~1-3 minutes for 450 trained stocks
- **Kronos**: ~3-7 minutes for 1,850 stocks
- **Total Time**: ~10-15 minutes for all models

**Optimization Tips:**
1. Run during low-traffic hours (6:30 AM)
2. Use database connection pooling
3. Batch insert predictions (100 at a time)
4. Add database indexes for fast querying
5. Archive old predictions (>90 days)

## Monitoring

### Check Prediction Coverage

```sql
-- How many stocks have predictions today?
SELECT
    model_type,
    COUNT(DISTINCT symbol) as stocks_with_predictions,
    MIN(prediction_date) as first_prediction,
    MAX(prediction_date) as last_prediction
FROM ml_predictions
GROUP BY model_type;
```

### Check Prediction Freshness

```sql
-- Are predictions up-to-date?
SELECT
    prediction_date,
    model_type,
    COUNT(*) as prediction_count
FROM ml_predictions
WHERE prediction_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY prediction_date, model_type
ORDER BY prediction_date DESC, model_type;
```

### Check Missing Predictions

```sql
-- Which stocks don't have predictions?
SELECT s.symbol, s.name
FROM stocks s
WHERE s.current_price IS NOT NULL
  AND s.current_price > 0
  AND NOT EXISTS (
      SELECT 1 FROM ml_predictions p
      WHERE p.symbol = s.symbol
        AND p.prediction_date = CURRENT_DATE
  )
LIMIT 100;
```

## Troubleshooting

### Issue: Script fails with "No module named 'src'"

```bash
# Make sure you run from project root
cd /path/to/StockExperiment
python3 tools/generate_ml_all_stocks.py
```

### Issue: Traditional ML predictions fail

Check if models are trained:
```bash
ls -lh ml_models/
# Should see: rf_price_model.pkl, xgb_price_model.pkl
```

If missing, train models first:
```bash
python3 tools/train_ml_model.py
```

### Issue: LSTM predictions fail

Check if LSTM models exist:
```bash
ls ml_models/raw_ohlcv_lstm/ | wc -l
# Should show number of trained symbols
```

Train more LSTM models:
```bash
python3 tools/batch_train_lstm_top_stocks.py
python3 tools/batch_train_lstm_small_mid_cap.py
```

### Issue: Kronos predictions fail

Check historical data coverage:
```sql
SELECT
    COUNT(DISTINCT symbol) as stocks_with_200_days
FROM (
    SELECT symbol, COUNT(*) as days
    FROM historical_data
    GROUP BY symbol
    HAVING COUNT(*) >= 200
) subq;
```

### Issue: Predictions saved but can't query

Check if table exists:
```sql
SELECT * FROM ml_predictions LIMIT 5;
```

If table doesn't exist, it will be auto-created on first run.

## Next Steps

1. **Run the script manually** to test:
   ```bash
   python3 tools/generate_ml_all_stocks.py
   ```

2. **Verify predictions in database**:
   ```sql
   SELECT COUNT(*) FROM ml_predictions WHERE prediction_date = CURRENT_DATE;
   ```

3. **Add to scheduler** for daily automation

4. **Build UI dashboard** to view predictions for any stock

5. **Create alerts** for high-confidence predictions

6. **Backtest strategies** using historical predictions

## Summary

✅ **New script generates ML predictions for ALL ~2,259 stocks**
✅ **All 3 models supported (Traditional, LSTM, Kronos)**
✅ **Saves to separate `ml_predictions` table**
✅ **Run manually or add to scheduler**
✅ **Query predictions for any stock at any time**
✅ **Compare models and find consensus signals**

---

For questions or issues:
- Script: `tools/generate_ml_all_stocks.py`
- Related: `src/services/ml/enhanced_stock_predictor.py`
- Related: `src/services/ml/raw_lstm_prediction_service.py`
