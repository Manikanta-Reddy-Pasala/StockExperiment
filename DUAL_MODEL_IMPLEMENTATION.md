# Dual Model Implementation Complete! 🎉

## Overview

Your system now supports **TWO ML approaches** for stock suggestions, allowing side-by-side comparison and letting data decide the winner!

---

## ✅ What's Been Implemented

### 1. Database Schema Update

**Added `model_type` column to `daily_suggested_stocks` table:**
- `'traditional'` - Your existing feature-engineered ensemble
- `'raw_lstm'` - New raw OHLCV simple LSTM

**Migration:** Already executed successfully ✅
```sql
ALTER TABLE daily_suggested_stocks
ADD COLUMN model_type VARCHAR(20) DEFAULT 'traditional'
```

### 2. Raw LSTM Prediction Service

**File:** `src/services/ml/raw_lstm_prediction_service.py`

**Features:**
- ✅ Loads trained raw OHLCV LSTM models
- ✅ Generates predictions with triple barrier probabilities
- ✅ Converts to same format as traditional model
- ✅ Saves to `daily_suggested_stocks` with `model_type='raw_lstm'`
- ✅ Batch prediction support
- ✅ Model caching for performance

**Usage:**
```python
from src.services.ml.raw_lstm_prediction_service import get_raw_lstm_prediction_service

# Get service
service = get_raw_lstm_prediction_service()

# Predict single stock
prediction = service.predict_single_stock('NSE:RELIANCE-EQ')

# Batch predict
symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:INFY-EQ']
predictions = service.batch_predict(symbols)

# Save to database
service.save_to_suggested_stocks(predictions, strategy='raw_lstm')
```

---

## 📊 How Both Models Co-exist

### Database Structure

```
daily_suggested_stocks
├── model_type = 'traditional'  ← Your existing model
│   ├── strategy = 'balanced'
│   ├── strategy = 'growth'
│   └── strategy = 'value'
│
└── model_type = 'raw_lstm'     ← New raw OHLCV model
    ├── strategy = 'raw_lstm'
    ├── strategy = 'balanced_raw'
    └── strategy = 'growth_raw'
```

### Both Models Generate:
- `ml_prediction_score` (0-1)
- `ml_price_target` (₹)
- `ml_confidence` (0-1)
- `ml_risk_score` (0-1)
- `recommendation` (BUY/HOLD/SELL)
- All fundamental metrics
- Same table structure!

---

## 🚀 Next Steps to Populate Both Models

### Option 1: Manual Training & Prediction

```bash
# Step 1: Train raw LSTM models for top stocks
for symbol in RELIANCE TCS INFY HDFCBANK WIPRO; do
    python tools/train_raw_ohlcv_lstm.py --symbol NSE:${symbol}-EQ
done

# Step 2: Run prediction service
python -c "
from src.services.ml.raw_lstm_prediction_service import get_raw_lstm_prediction_service

service = get_raw_lstm_prediction_service()

symbols = [
    'NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:INFY-EQ',
    'NSE:HDFCBANK-EQ', 'NSE:WIPRO-EQ'
]

predictions = service.batch_predict(symbols)
service.save_to_suggested_stocks(predictions, strategy='raw_lstm')

print(f'✓ Saved {len(predictions)} raw LSTM predictions')
"
```

### Option 2: Update Scheduler (Recommended)

Update `scheduler.py` to train and predict with BOTH models:

```python
# In scheduler.py

def train_both_models():
    """Train both traditional and raw LSTM models."""
    logger.info("Training BOTH models...")

    # 1. Train traditional model (existing)
    train_traditional_models()

    # 2. Train raw LSTM models
    from src.services.ml.raw_ohlcv_lstm import RawOHLCVLSTM
    from src.services.ml.data_service import get_raw_ohlcv_data

    top_stocks = get_top_liquid_stocks(limit=50)

    for symbol in top_stocks:
        try:
            df = get_raw_ohlcv_data(symbol, period='3y')
            model = RawOHLCVLSTM(hidden_size=8, window_length=100)
            model.train(df, epochs=30, verbose=0)
            model.save(f'ml_models/raw_ohlcv_lstm/{symbol}')
            logger.info(f"✓ Trained raw LSTM for {symbol}")
        except Exception as e:
            logger.error(f"✗ Failed {symbol}: {e}")

def update_suggested_stocks_dual():
    """Generate suggestions from BOTH models."""
    logger.info("Generating suggestions from BOTH models...")

    # 1. Traditional model (existing)
    update_traditional_suggestions()

    # 2. Raw LSTM model
    from src.services.ml.raw_lstm_prediction_service import get_raw_lstm_prediction_service

    service = get_raw_lstm_prediction_service()
    stocks = get_all_stock_symbols()

    predictions = service.batch_predict(stocks[:100])  # Top 100
    service.save_to_suggested_stocks(predictions, strategy='raw_lstm')

    logger.info(f"✓ Generated {len(predictions)} raw LSTM suggestions")

# Schedule both
schedule.every().day.at("02:00").do(train_both_models)
schedule.every().day.at("02:30").do(update_suggested_stocks_dual)
```

### Option 3: Separate Schedulers

Keep them independent for easier debugging:

```bash
# Traditional model (existing)
schedule.every().day.at("02:00").do(train_traditional_models)
schedule.every().day.at("02:30").do(update_traditional_suggestions)

# Raw LSTM model (new)
schedule.every().day.at("03:00").do(train_raw_lstm_models)
schedule.every().day.at("03:30").do(update_raw_lstm_suggestions)
```

---

## 🔍 Querying Both Models

### Get Traditional Model Suggestions
```sql
SELECT *
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
  AND model_type = 'traditional'
ORDER BY ml_prediction_score DESC
LIMIT 10;
```

### Get Raw LSTM Suggestions
```sql
SELECT *
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
  AND model_type = 'raw_lstm'
ORDER BY ml_prediction_score DESC
LIMIT 10;
```

### Compare Both Models Side-by-Side
```sql
SELECT
    t.symbol,
    t.stock_name,
    t.ml_prediction_score as traditional_score,
    r.ml_prediction_score as raw_lstm_score,
    t.ml_price_target as traditional_target,
    r.ml_price_target as raw_lstm_target,
    t.recommendation as traditional_rec,
    r.recommendation as raw_lstm_rec
FROM daily_suggested_stocks t
LEFT JOIN daily_suggested_stocks r
    ON t.symbol = r.symbol
    AND t.date = r.date
    AND r.model_type = 'raw_lstm'
WHERE t.date = CURRENT_DATE
  AND t.model_type = 'traditional'
ORDER BY t.rank
LIMIT 20;
```

### Find Agreement Between Models
```sql
SELECT
    t.symbol,
    t.stock_name,
    t.ml_prediction_score as trad_score,
    r.ml_prediction_score as lstm_score,
    ABS(t.ml_prediction_score - r.ml_prediction_score) as score_diff
FROM daily_suggested_stocks t
JOIN daily_suggested_stocks r
    ON t.symbol = r.symbol
    AND t.date = r.date
WHERE t.date = CURRENT_DATE
  AND t.model_type = 'traditional'
  AND r.model_type = 'raw_lstm'
  AND t.recommendation = r.recommendation  -- Both agree
ORDER BY (t.ml_prediction_score + r.ml_prediction_score) DESC
LIMIT 10;
```

---

## 📈 API Endpoint Example

Add to your routes:

```python
# In src/web/routes/ml/ml_routes.py

@bp.route('/api/suggested-stocks/compare', methods=['GET'])
def compare_models():
    """Compare traditional vs raw LSTM suggestions."""
    date = request.args.get('date', datetime.now().date())
    limit = int(request.args.get('limit', 20))

    with db.get_session() as session:
        query = text("""
            SELECT
                t.symbol,
                t.stock_name,
                t.current_price,
                t.ml_prediction_score as traditional_score,
                r.ml_prediction_score as raw_lstm_score,
                t.ml_confidence as traditional_confidence,
                r.ml_confidence as raw_lstm_confidence,
                t.recommendation as traditional_rec,
                r.recommendation as raw_lstm_rec,
                t.ml_price_target as traditional_target,
                r.ml_price_target as raw_lstm_target
            FROM daily_suggested_stocks t
            LEFT JOIN daily_suggested_stocks r
                ON t.symbol = r.symbol
                AND t.date = r.date
                AND r.model_type = 'raw_lstm'
            WHERE t.date = :date
              AND t.model_type = 'traditional'
            ORDER BY t.rank
            LIMIT :limit
        """)

        result = session.execute(query, {'date': date, 'limit': limit})
        comparisons = [dict(row._mapping) for row in result]

    return jsonify({
        'success': True,
        'date': str(date),
        'count': len(comparisons),
        'comparisons': comparisons
    })
```

**Test it:**
```bash
curl "http://localhost:5001/api/suggested-stocks/compare?limit=10"
```

---

## 🎯 Model Selection Strategies

### 1. Always Use Best Performer
```python
# Auto-select model with higher score
best_model = 'traditional' if trad_score > lstm_score else 'raw_lstm'
```

### 2. Ensemble Both
```python
# Average predictions
ensemble_score = (trad_score * 0.5) + (lstm_score * 0.5)
```

### 3. Require Agreement
```python
# Only suggest if both models agree
if trad_rec == lstm_rec == 'BUY':
    high_confidence_buy = True
```

### 4. Use Different Models for Different Stocks
```python
# Traditional for stable large-caps
# Raw LSTM for volatile mid-caps
if market_cap > 50000:  # Crores
    use_model = 'traditional'
else:
    use_model = 'raw_lstm'
```

---

## 📊 Performance Tracking

Create a comparison dashboard:

```python
def compare_model_performance():
    """Compare actual vs predicted performance for both models."""
    query = text("""
        SELECT
            model_type,
            AVG(CASE WHEN actual_return > 0 AND recommendation = 'BUY' THEN 1 ELSE 0 END) as accuracy,
            AVG(ABS(ml_price_target - actual_price)) as mae,
            COUNT(*) as predictions
        FROM daily_suggested_stocks
        WHERE date BETWEEN :start_date AND :end_date
        GROUP BY model_type
    """)
    # Analyze results
```

---

## 🔧 Files Modified/Created

### New Files
1. ✅ `tools/migrate_add_model_type.py` - Database migration
2. ✅ `src/services/ml/raw_lstm_prediction_service.py` - Prediction service
3. ✅ `DUAL_MODEL_IMPLEMENTATION.md` - This file

### Files to Update (Optional)
4. ⏳ `scheduler.py` - Add raw LSTM training
5. ⏳ `src/web/routes/ml/ml_routes.py` - Add comparison API
6. ⏳ `src/services/data/suggested_stocks_saga.py` - Dual model support

---

## 🧪 Quick Test

Test the raw LSTM prediction service:

```bash
python3 -c "
from src.services.ml.raw_lstm_prediction_service import get_raw_lstm_prediction_service

service = get_raw_lstm_prediction_service()

# Assuming you have a trained model for ADANIPOWER
prediction = service.predict_single_stock('NSE:ADANIPOWER-EQ')

if prediction:
    print(f\"✓ Prediction generated!\")
    print(f\"  Symbol: {prediction['symbol']}\")
    print(f\"  Score: {prediction['ml_prediction_score']:.4f}\")
    print(f\"  Target: ₹{prediction['ml_price_target']:.2f}\")
    print(f\"  Recommendation: {prediction['recommendation']}\")
    print(f\"  Model Type: {prediction['model_type']}\")
else:
    print(\"✗ No trained model available yet. Run training first.\")
"
```

---

## 📝 Summary

### What You Have Now:

✅ **Database supports both models** - `model_type` column added
✅ **Raw LSTM prediction service** - Ready to generate suggestions
✅ **Compatible APIs** - Both models use same structure
✅ **Migration completed** - Existing data marked as 'traditional'

### What You Need to Do:

1. **Train raw LSTM models** for your top stocks:
   ```bash
   python tools/train_raw_ohlcv_lstm.py --symbol NSE:RELIANCE-EQ
   ```

2. **Generate raw LSTM predictions**:
   ```python
   python src/services/ml/raw_lstm_prediction_service.py
   ```

3. **Update scheduler** (optional):
   - Add raw LSTM training to daily schedule
   - Add raw LSTM prediction generation

4. **Create comparison UI** (optional):
   - Show both models side-by-side
   - Highlight agreements/disagreements
   - Track performance over time

---

## 🎉 Benefits

1. **Scientific Comparison** - Same data, two approaches, let results decide
2. **Redundancy** - If one model fails, other still works
3. **Ensemble Potential** - Combine predictions for better results
4. **Flexibility** - Use different models for different stocks
5. **Innovation** - Test cutting-edge research on Indian markets

---

**Your system is now ready for dual-model stock suggestions!** 🚀

Train a few models, generate predictions, and compare the results!
