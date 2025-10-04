# Machine Learning Guide

## 🤖 Overview

The trading system uses **Random Forest** machine learning models to:
- Predict 2-week price targets
- Assess risk (max drawdown)
- Score stock opportunities (0-1)
- Provide confidence levels

**Training:** Daily at 2:00 AM (automated)
**Duration:** 1-2 minutes
**Data:** 365 days of historical + technical indicators

---

## 📊 ML Models

### 1. Price Prediction Model
**Purpose:** Predict price change% in 2 weeks

**Algorithm:** Random Forest Regressor
```python
RandomForestRegressor(
    n_estimators=100,      # 100 trees
    max_depth=10,          # Shallow trees (fast)
    min_samples_split=20,  # Prevents overfitting
    min_samples_leaf=10,   # Larger leaves (faster)
    n_jobs=-1              # Parallel processing
)
```

**Output:**
- `ml_prediction_score` (0-1): Higher = better opportunity
- `ml_price_target`: Predicted price in 2 weeks
- `ml_confidence` (0-1): Model confidence

### 2. Risk Assessment Model
**Purpose:** Predict maximum drawdown in 2 weeks

**Algorithm:** Random Forest Regressor (same config)

**Output:**
- `ml_risk_score` (0-1): Lower = safer
- `predicted_drawdown_pct`: Expected max loss%

---

## 🎯 Features Used (25-30 total)

### Price & Market
- `current_price`
- `market_cap`
- `volume`

### Fundamental Ratios
- `pe_ratio`, `pb_ratio`, `roe`, `eps`
- `beta`, `debt_to_equity`

### Growth & Profitability
- `revenue_growth`, `earnings_growth`
- `operating_margin`, `net_margin`

### Volatility
- `historical_volatility_1y`
- `atr_14`, `atr_percentage`

### Technical Indicators
- `rsi_14`
- `macd`, `signal_line`, `macd_histogram`
- `sma_50`, `sma_200`
- `ema_12`, `ema_26`

### Engineered Features
- `sma_ratio` = sma_50 / sma_200
- `ema_diff` = ema_12 - ema_26
- `price_vs_sma50` = (price - sma_50) / sma_50
- `price_vs_sma200` = (price - sma_200) / sma_200

---

## 📈 Training Process

### Data Preparation
```python
# Fetch 365 days of historical data
# Join: stocks + historical_data + technical_indicators
# Features: 25-30 columns
# Samples: ~600,000-700,000 rows
```

### Feature Engineering
```python
# Calculate targets
future_price = price in 14 days
price_change_pct = (future_price - current_price) / current_price * 100

min_price_14d = MIN(price) in next 14 days
max_drawdown_pct = (min_price_14d - current_price) / current_price * 100
```

### Model Training
```python
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train price model
price_model.fit(X_scaled, price_change_pct)

# Train risk model
risk_model.fit(X_scaled, max_drawdown_pct)
```

### Evaluation
```python
# R² Score (higher = better)
# Typical values:
# - Price R²: 0.15-0.35 (stock prediction is hard!)
# - Risk R²: 0.20-0.40
```

**Why Low R²?**
- Stock markets are inherently noisy
- Many external factors not in features
- R² > 0.2 is actually good for stock prediction!

---

## 🎲 Making Predictions

### Single Stock Prediction
```python
from src.services.ml.stock_predictor import StockMLPredictor

predictor = StockMLPredictor(db_session)
predictor.train(lookback_days=365)

prediction = predictor.predict(stock_data)

# Returns:
{
    'ml_prediction_score': 0.75,      # 0-1 score
    'ml_price_target': 1250.50,       # ₹ price
    'ml_confidence': 0.82,            # 0-1 confidence
    'ml_risk_score': 0.15,            # Lower = safer
    'predicted_change_pct': 8.5,      # +8.5%
    'predicted_drawdown_pct': -3.2    # Max -3.2% drawdown
}
```

### Batch Predictions (All Stocks)
```python
from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator

saga = SuggestedStocksSagaOrchestrator(db_session)
result = saga.execute(strategy='balanced', limit=50)

# Step 6: ML Prediction (applies to all stocks)
# Step 7: Daily Snapshot (saves top 50 to DB)
```

---

## 💾 Daily Storage

### Database Table
```sql
daily_suggested_stocks
├── date (YYYY-MM-DD)
├── symbol
├── ml_prediction_score (0-1)
├── ml_price_target (₹)
├── ml_confidence (0-1)
├── ml_risk_score (0-1)
└── ... (technical + fundamental data)
```

### Upsert Logic
- **Unique Key:** `(date, symbol, strategy)`
- **Behavior:** Replaces same-day data if re-run
- **Retention:** 90 days (cleaned up weekly)

---

## 🤖 Automated Training

### Scheduler (`scheduler.py`)
```python
# Daily at 2:00 AM
schedule.every().day.at("02:00").do(train_ml_models)

# Daily at 2:15 AM
schedule.every().day.at("02:15").do(update_daily_snapshot)

# Sunday at 3:00 AM
schedule.every().sunday.at("03:00").do(cleanup_old_snapshots)
```

### Manual Training
```bash
# Command line
python3 tools/train_ml_model.py

# Admin dashboard
http://localhost:5001/admin
→ Click "Train Models (1-2 min)"
```

---

## 📊 Performance Metrics

### Training Time
- **Data fetch:** 20-30 seconds
- **Preprocessing:** 10-15 seconds
- **Price model:** 25-35 seconds
- **Risk model:** 25-35 seconds
- **Total:** ~90-120 seconds

### Model Accuracy
- **Price R²:** 0.15-0.35 (typical for stock prediction)
- **Risk R²:** 0.20-0.40
- **Feature importance:** Price, RSI, MACD, volatility top 4

### Prediction Quality
- **High confidence (>0.7):** More reliable
- **Low risk (<0.3):** Safer bets
- **High score (>0.6):** Better opportunities

---

## 🔧 Customization

### Adjust Training Data Window
```python
# In scheduler.py or train_ml_model.py
predictor.train(lookback_days=730)  # Use 2 years instead of 1
```

### Increase Model Complexity (Slower but More Accurate)
```python
RandomForestRegressor(
    n_estimators=500,      # More trees (5x slower)
    max_depth=20,          # Deeper trees
    min_samples_split=10,  # More splits
    n_jobs=-1
)
# Training time: 5-10 minutes instead of 1-2
```

### Add More Features
Edit `src/services/ml/stock_predictor.py`:
```python
feature_cols = [
    # Existing features...
    'sector_performance',    # New feature
    'news_sentiment',        # New feature
    'options_implied_vol',   # New feature
]
```

---

## 📈 Using ML Predictions

### API Response
```json
GET /api/suggested-stocks/?strategy=balanced&limit=10

{
  "stocks": [
    {
      "symbol": "RELIANCE",
      "current_price": 2450.50,
      "ml_prediction_score": 0.78,
      "ml_price_target": 2650.20,
      "ml_confidence": 0.85,
      "ml_risk_score": 0.12,
      "recommendation": "BUY"
    }
  ]
}
```

### Interpretation

**High Opportunity (Score > 0.7):**
- Model predicts good upside
- Use with high confidence (>0.7)

**Low Risk (Score < 0.3):**
- Limited downside expected
- Good for conservative strategies

**Combined Score:**
- **Score >0.7 + Risk <0.3 + Confidence >0.7** = Strong buy signal
- **Score <0.4 + Risk >0.5** = Avoid

---

## ⚠️ Important Notes

### ML Limitations
1. **Not Financial Advice**: ML predictions are estimates, not guarantees
2. **Past Performance**: No guarantee of future results
3. **Market Conditions**: Models trained on historical data may not capture black swan events
4. **Use with Caution**: Always combine ML with fundamental analysis

### Best Practices
1. **Diversify**: Don't rely solely on ML scores
2. **Risk Management**: Use stop losses
3. **Monitor**: Check daily snapshots for consistency
4. **Update**: Retrain models when market conditions change significantly
5. **Validate**: Backtest predictions vs actual results

---

## 🛠️ Troubleshooting

### Low R² Scores
**Normal!** Stock prediction R² of 0.2-0.3 is actually good.

### Training Failures
```bash
# Check data availability
docker exec trading_system_db_dev psql -U trader -d trading_system -c "SELECT COUNT(*) FROM historical_data;"

# Need at least 100 samples with 14-day future prices
# With 2,259 stocks × 365 days = plenty of data
```

### Inconsistent Predictions
- Market regime changed → Retrain
- Missing technical indicators → Run pipeline
- Outlier stocks → Check data quality

---

## 📚 Further Reading

- **Random Forest:** Ensemble learning method (Wikipedia)
- **Feature Engineering:** Adding domain knowledge to features
- **Backtesting:** Testing strategies on historical data
- **Risk Management:** Position sizing, stop losses, diversification

---

## 🎉 Summary

✅ **Automated ML training** (daily at 2 AM)
✅ **Two models:** Price prediction + Risk assessment
✅ **25-30 features:** Technical + Fundamental
✅ **Fast training:** 1-2 minutes
✅ **Daily snapshots:** Top 50 stocks saved to DB
✅ **API integration:** `/api/suggested-stocks/` with ML scores
✅ **Admin dashboard:** Manual training available

**Ready to use!** Check `/admin` for manual controls.
