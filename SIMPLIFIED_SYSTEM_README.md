# 📊 Simplified Technical Indicator Stock Trading System

## Overview

This branch (`feature/simplified-technical-screener`) represents a **major simplification** of the stock trading system. We've removed all ML complexity and replaced it with a pure **technical analysis approach** based on proven indicators.

---

## 🎯 What Changed

### ❌ **Removed (19,353 lines deleted)**

1. **All ML Models:**
   - Random Forest (RF) price and risk models
   - XGBoost models
   - LSTM deep learning models
   - Kronos K-line tokenization models
   - All model training infrastructure

2. **ML Services Directory:** `src/services/ml/` (29 files)
   - enhanced_stock_predictor.py
   - raw_lstm_prediction_service.py
   - kronos_prediction_service.py
   - portfolio_optimizer.py
   - ai_stock_analyst.py (OpenAI integration)
   - All other ML-related services

3. **ML Training Tools:** 20+ tools in `tools/`
   - train_ml_models.py
   - batch_train_lstm_*.py
   - generate_ml_all_stocks.py
   - test_all_models_saga.py

4. **ML Model Files:** `ml_models/` directory
   - rf_price_model.pkl, xgb_price_model.pkl
   - LSTM models for 5 stocks
   - Feature scalers and metadata

5. **Database Columns:**
   - `ml_prediction_score`
   - `ml_price_target`
   - `ml_confidence`
   - `ml_risk_score`
   - `model_type` (traditional/raw_lstm/kronos)

---

### ✅ **Added (1,412 lines added)**

1. **Technical Indicators Service:** `src/services/technical/indicators_calculator.py`
   - **RS Rating (1-99):** Relative Strength vs NIFTY 50
     - Quarterly performance with weighted scoring
     - Q1: 40%, Q2: 20%, Q3: 20%, Q4: 20%
   - **Wave Indicators:** EMA-based momentum
     - HLC3 calculation (avg of high, low, close)
     - Fast Wave: 12-period EMA of deviation
     - Slow Wave: 3-period MA of Fast Wave
     - Delta: Fast Wave - Slow Wave
   - **Buy/Sell Signals:** Wave crossover detection
     - Buy: Fast Wave > Slow Wave AND Delta > 0
     - Sell: Fast Wave < Slow Wave AND Delta < 0

2. **Database Schema Updates:** `init-scripts/02-add-technical-indicators.sql`
   ```sql
   -- New columns in stocks table
   rs_rating DECIMAL(10, 4)
   fast_wave DECIMAL(10, 4)
   slow_wave DECIMAL(10, 4)
   delta DECIMAL(10, 4)
   buy_signal BOOLEAN
   sell_signal BOOLEAN
   indicators_last_updated TIMESTAMP

   -- Same columns in daily_suggested_stocks table
   ```

3. **Simplified Scheduler:** `scheduler.py`
   - **10:00 PM:** Calculate technical indicators for all stocks
   - **10:15 PM:** Generate daily stock picks using technical signals
   - **Removed:** All ML training (6:00 AM), ML predictions (6:30 AM)

4. **Updated Saga Pipeline:** `suggested_stocks_saga.py`
   - Step 6: **Technical Indicators** (replaces ML Prediction)
     - Fetches indicators from stocks table
     - Calculates composite technical score
     - Sorts and ranks by technical score
   - Step 7: **Daily Snapshot** (updated)
     - Saves technical indicators instead of ML predictions

---

## 🧮 Technical Score Formula

The system ranks stocks using a **Composite Technical Score (0-100)**:

```python
# Base score from RS Rating (0-60 points)
base_score = rs_rating * 0.6

# Delta contribution (-40 to +40 points)
delta_contribution = clamp(delta * 100, -40, 40)

# Signal bonuses
buy_bonus = 10 if buy_signal else 0
sell_penalty = -10 if sell_signal else 0

# Final composite score
composite_score = clamp(base_score + delta_contribution + buy_bonus + sell_penalty, 0, 100)
```

**Example:**
- Stock with RS Rating = 80, Delta = 0.15, Buy Signal = True
- Score = (80 * 0.6) + (0.15 * 100) + 10 = 48 + 15 + 10 = **73 points**

---

## 📅 New Daily Schedule

### **Data Scheduler** (`data_scheduler.py`)
| Time | Task | Description |
|------|------|-------------|
| 6:00 AM (Mon) | Symbol Master Update | Fetch ~2,259 NSE symbols from Fyers |
| 9:00 PM | Data Pipeline (6-step saga) | OHLCV data, technical indicators, fundamentals |
| 9:30 PM | Fill Missing Data | Business logic calculations |
| 10:00 PM | CSV Export | Export data for analysis |

### **ML Scheduler** (`scheduler.py`) → **Simplified Scheduler**
| Time | Task | Description |
|------|------|-------------|
| ~~6:00 AM~~ | ~~ML Training~~ | ❌ **REMOVED** |
| ~~6:30 AM~~ | ~~ML Predictions~~ | ❌ **REMOVED** |
| ~~7:00 AM~~ | ~~Daily Snapshot~~ | ❌ **REMOVED** |
| **10:00 PM** | **Technical Indicators** | ✅ **NEW** - Calculate RS Rating, Waves, Signals |
| **10:15 PM** | **Daily Stock Picks** | ✅ **UPDATED** - Select top 50 stocks per strategy using technical score |
| 9:20 AM | Auto-Trading | Place orders (if enabled) |
| 6:00 PM | Performance Tracking | Update order performance |
| 3:00 AM (Sun) | Cleanup | Delete old snapshots (>90 days) |
| Every 6 hours | Token Status Check | Monitor Fyers token expiry |

---

## 🚀 How to Use the Simplified System

### 1. **Run Database Migration**

```bash
# Apply the new schema (adds technical indicator columns, removes ML columns)
docker exec -it trading_system_db psql -U trader -d trading_system -f /docker-entrypoint-initdb.d/02-add-technical-indicators.sql
```

### 2. **Start the Simplified System**

```bash
# Start all services
./run.sh prod

# Or just the scheduler
docker compose up -d
docker compose logs -f
```

### 3. **First-Time Setup**

The system will:
1. **10:00 PM:** Calculate technical indicators for all stocks (first run might take 20-30 minutes)
2. **10:15 PM:** Generate daily stock picks using technical scores
3. **Next day 9:20 AM:** Auto-place orders (if enabled)

---

## 📊 Technical Indicators Explained

### **RS Rating (Relative Strength)**

Compares a stock's performance against NIFTY 50 over the last year:
- **1-33:** Weak performers (avoid)
- **34-66:** Average performers
- **67-99:** Strong performers (buy candidates)

**Calculation:**
```python
# Get quarterly returns for stock and NIFTY
stock_returns = [Q1, Q2, Q3, Q4]  # Last 4 quarters (63 days each)
nifty_returns = [Q1, Q2, Q3, Q4]

# Calculate relative performance
relative_performance = [stock[i] - nifty[i] for i in range(4)]

# Apply weights (recent quarters matter more)
weights = [0.4, 0.2, 0.2, 0.2]
rs_rating = weighted_sum(relative_performance, weights)

# Normalize to 1-99 scale
```

### **Wave Indicators (Momentum)**

Based on EMA deviations to detect trend changes:

**Fast Wave:**
- Measures short-term momentum
- Reacts quickly to price changes
- Positive = bullish momentum

**Slow Wave:**
- Smoothed version of Fast Wave
- Filters out noise
- Confirms trend direction

**Delta:**
- Difference between Fast and Slow
- **Delta > 0:** Bullish (Fast above Slow)
- **Delta < 0:** Bearish (Fast below Slow)

### **Buy/Sell Signals**

Generated from wave crossovers:
- **Buy Signal:** Fast Wave crosses above Slow Wave (golden cross)
- **Sell Signal:** Fast Wave crosses below Slow Wave (death cross)

---

## 🔄 Migration from ML System

### **For Existing Users:**

1. **Backup your database:**
   ```bash
   docker exec trading_system_db pg_dump -U trader trading_system > backup_$(date +%Y%m%d).sql
   ```

2. **Switch to simplified branch:**
   ```bash
   git checkout feature/simplified-technical-screener
   ```

3. **Rebuild containers:**
   ```bash
   docker compose down
   docker compose build
   docker compose up -d
   ```

4. **Run migration:**
   ```bash
   docker exec -it trading_system_db psql -U trader -d trading_system -f /docker-entrypoint-initdb.d/02-add-technical-indicators.sql
   ```

5. **Wait for first indicator calculation:**
   - Check logs: `docker compose logs -f`
   - First run at 10:00 PM will calculate indicators for all stocks

---

## 📈 Benefits of Simplified System

### **1. Faster Execution**
- ❌ ML Training: 10-15 minutes
- ❌ LSTM Training: 5-10 minutes
- ✅ **Technical Indicators:** 2-3 minutes

### **2. Simpler Codebase**
- ❌ 29 ML service files
- ❌ 20+ training tools
- ❌ 19,353 lines of ML code
- ✅ **1 technical service** with clear, understandable logic

### **3. More Interpretable**
- ❌ Black-box ML predictions
- ❌ Difficult to explain why a stock was selected
- ✅ **Clear technical reasons:** "High RS Rating (85), strong bullish delta (0.25), buy signal triggered"

### **4. Lower Maintenance**
- ❌ Model retraining required
- ❌ Feature engineering complexity
- ❌ TensorFlow/scikit-learn dependencies
- ✅ **Simple calculations** based on price data

### **5. Better for Beginners**
- ❌ Requires ML/stats knowledge
- ❌ Hard to debug model issues
- ✅ **Based on proven technical analysis** principles anyone can understand

---

## 🧪 Testing the System

### **Verify Technical Indicators:**

```sql
-- Check if indicators are calculated
SELECT symbol, rs_rating, fast_wave, slow_wave, delta, buy_signal, sell_signal, indicators_last_updated
FROM stocks
WHERE indicators_last_updated IS NOT NULL
ORDER BY rs_rating DESC
LIMIT 10;
```

### **Check Daily Picks:**

```sql
-- View today's stock picks
SELECT date, symbol, stock_name, strategy, rank, selection_score, rs_rating, delta, buy_signal
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
ORDER BY strategy, rank;
```

### **Manual Indicator Calculation:**

```bash
# Calculate indicators for all stocks
docker exec -it trading_system python3 -c "
from src.models.database import get_database_manager
from src.services.technical.indicators_calculator import get_indicators_calculator

db_manager = get_database_manager()
with db_manager.get_session() as session:
    calculator = get_indicators_calculator(session)

    # Calculate for specific stock
    indicators = calculator.calculate_all_indicators('NSE:RELIANCE-EQ')
    print(indicators)
"
```

---

## 🐛 Troubleshooting

### **No indicators calculated:**
```bash
# Check scheduler logs
docker compose logs -f | grep "Technical Indicators"

# Manually trigger calculation
docker exec -it trading_system python3 scheduler.py
```

### **Database errors after migration:**
```bash
# Verify schema changes
docker exec -it trading_system_db psql -U trader -d trading_system -c "\d daily_suggested_stocks"

# Check for rs_rating, fast_wave, slow_wave, delta, buy_signal, sell_signal columns
```

### **Import errors:**
```bash
# Rebuild containers to ensure clean state
docker compose down
docker compose build --no-cache
docker compose up -d
```

---

## 📚 Further Reading

- **RS Rating:** Similar to IBD's Relative Strength Rating
- **EMA-based Momentum:** Inspired by MACD but using wave indicators
- **Technical Analysis:** Based on the article "How to Build a Stock Screener with Custom Technical Indicators"

---

## 🎓 Next Steps

1. ✅ **Phase 1:** Remove ML code and create technical indicator service
2. ✅ **Phase 2:** Update saga and snapshot service
3. ⏳ **Phase 3:** Update UI to display new indicators
4. ⏳ **Phase 4:** Update documentation (CLAUDE.md, README.md)
5. ⏳ **Phase 5:** Testing and validation

---

## 🤝 Contributing

If you want to add more technical indicators (e.g., Bollinger Bands, Stochastic Oscillator), follow this pattern:

1. Add indicator calculation to `indicators_calculator.py`
2. Add column to database schema
3. Update composite score formula in `suggested_stocks_saga.py`
4. Update UI to display the indicator

---

**🤖 Generated with Claude Code**

Co-Authored-By: Claude <noreply@anthropic.com>
