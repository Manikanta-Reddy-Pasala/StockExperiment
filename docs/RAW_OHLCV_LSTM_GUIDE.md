# Raw OHLCV LSTM Model - Research-Based Approach

## Overview

This implementation is based on groundbreaking research from a 2024 Korean stock prediction study that challenged conventional wisdom in financial ML. The key finding:

> **"A simple LSTM model trained on raw OHLCV data matches the performance of heavily engineered XGBoost models with 100+ technical indicators"**

### Research Results

| Model | Features | F1 Score | Key Insight |
|-------|----------|----------|-------------|
| **Simple LSTM** | Raw OHLCV (5 features) | **0.4312** | Hidden size=8, Window=100 |
| **XGBoost** | 100+ technical indicators | 0.4316 | Complex feature engineering |
| **Dummy Classifier** | Baseline | 0.1852 | Random guessing |

**Conclusion:** Adding close price to OHLCV increased F1 from 0.4170 to 0.4297, and full OHLCV reached 0.4312 - matching XGBoost!

---

## Why This Matters

### Traditional Approach (What You Were Doing)
```python
# Your current implementation
- 70+ engineered features (RSI, MACD, SMA, EMA, Bollinger Bands, etc.)
- Complex ensemble models (Random Forest + XGBoost + LSTM)
- Heavy preprocessing and feature scaling
- Risk of overfitting to specific indicators
```

### Research-Based Approach (What We Implemented)
```python
# New implementation
- Only 5 raw features: Open, High, Low, Close, Volume
- Simple LSTM with 8 hidden units
- Minimal preprocessing (just scaling)
- Let the model learn representations automatically
```

### Benefits
✅ **Simpler** - Less code, fewer dependencies
✅ **Faster** - No feature engineering overhead
✅ **More Robust** - Less overfitting, better generalization
✅ **Information Preservation** - No signal loss from transformations
✅ **Easier to Maintain** - No brittle technical indicator calculations

---

## Triple Barrier Labeling

Instead of simple "will price go up or down?", we use **Triple Barrier Method** from López de Prado:

### Concept

Define 3 barriers:
1. **Upper Barrier** (take-profit): +9% price move
2. **Lower Barrier** (stop-loss): -9% price move
3. **Time Barrier**: 29 days maximum holding period

**Label = which barrier is hit FIRST**

```
Profit (1):   Upper barrier hit first → Stock rose +9%
Neutral (0):  Time barrier hit first → No significant move in 29 days
Loss (-1):    Lower barrier hit first → Stock fell -9%
```

### Why This is Better

❌ **Traditional Labeling:**
```python
# Simple future price
y = (price_tomorrow - price_today) / price_today
# Problems:
# - Ignores stop-loss behavior
# - Ignores profit-taking
# - Look-ahead bias
# - Unrealistic for trading
```

✅ **Triple Barrier Labeling:**
```python
# Realistic trading targets
# - Accounts for risk management
# - Matches real trading behavior
# - More balanced labels
# - Better for strategy development
```

---

## Implementation Guide

### 1. Basic Training

```python
from src.services.ml.data_service import get_raw_ohlcv_data
from src.services.ml.raw_ohlcv_lstm import RawOHLCVLSTM

# Fetch raw OHLCV data (no feature engineering!)
df = get_raw_ohlcv_data(
    symbol='RELIANCE',
    period='3y',
    user_id=1
)

# Create model with research-optimal parameters
model = RawOHLCVLSTM(
    hidden_size=8,      # Paper's finding: 8 is optimal (not 128!)
    window_length=100,  # Use 100 days of history
    use_full_ohlcv=True # Use all 5 OHLCV features
)

# Train
metrics = model.train(df, epochs=50, verbose=1)

# Results
print(f"F1 Macro: {metrics['f1_macro']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Save
model.save('ml_models/raw_ohlcv_lstm/RELIANCE')
```

### 2. Making Predictions

```python
# Load trained model
model = RawOHLCVLSTM()
model.load('ml_models/raw_ohlcv_lstm/RELIANCE')

# Get new data
df_new = get_raw_ohlcv_data('RELIANCE', period='1M')

# Predict
predictions = model.predict(df_new)
probabilities = model.predict(df_new, return_probabilities=True)

# Interpret
# 0 = Loss expected
# 1 = Neutral (no significant move)
# 2 = Profit expected
```

### 3. Using Triple Barrier Labeling

```python
from src.services.ml.triple_barrier_labeling import TripleBarrierLabeler

# Create labeler
labeler = TripleBarrierLabeler(
    upper_barrier=9.0,   # +9% profit target
    lower_barrier=9.0,   # -9% stop loss
    time_horizon=29      # 29 days max holding
)

# Apply to data
df_labeled = labeler.apply_triple_barrier(df, price_col='close')

# Check results
print(df_labeled[['close', 'tbl_label', 'tbl_barrier_touched', 'tbl_return']].head())

# Get ML-ready format
df_clean, labels = labeler.apply_for_ml(df, return_multiclass=True)
# labels: 0=Loss, 1=Neutral, 2=Profit
```

### 4. Auto-Tune Barrier Parameters

```python
from src.services.ml.triple_barrier_labeling import create_balanced_barriers

# Find optimal barriers for balanced labels
upper, lower, horizon = create_balanced_barriers(df)

print(f"Optimal barriers: {upper}%, {lower}%, {horizon} days")

# Use in labeler
labeler = TripleBarrierLabeler(upper, lower, horizon)
```

### 5. Compare Approaches (A/B Testing)

```python
from src.services.ml.model_comparison import compare_traditional_vs_raw_lstm

# Compare your traditional approach vs research approach
results = compare_traditional_vs_raw_lstm(
    symbol='RELIANCE',
    period='3y',
    user_id=1
)

# Check ml_models/comparison/ for detailed report
```

---

## Command Line Usage

### Train on Single Stock
```bash
# Basic training (uses research-optimal parameters)
python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE

# Custom parameters
python tools/train_raw_ohlcv_lstm.py \
    --symbol TCS \
    --hidden-size 8 \
    --window 100 \
    --epochs 50
```

### Grid Search for Best Hyperparameters
```bash
python tools/train_raw_ohlcv_lstm.py \
    --symbol INFY \
    --grid-search \
    --max-trials 20
```

### Compare with Your Traditional Approach
```bash
python tools/train_raw_ohlcv_lstm.py \
    --symbol HDFCBANK \
    --compare
```

---

## Configuration

Edit `config/triple_barrier_config.yaml`:

```yaml
# Default (from research)
default:
  upper_barrier: 9.0
  lower_barrier: 9.0
  time_horizon: 29

# Model architecture
lstm_model:
  hidden_size: 8          # Small is better!
  window_length: 100
  use_full_ohlcv: true
  epochs: 100
  batch_size: 32
```

---

## Architecture Details

### LSTM Model Structure

```
Input: (batch_size, 100, 5)
  ↓
LSTM Layer (8 units)
  ↓
Dropout (0.2)
  ↓
Dense (3 units, softmax)
  ↓
Output: [P(Loss), P(Neutral), P(Profit)]
```

**Total Parameters:** ~1,000 (vs 10,000+ in complex models)

### Why So Simple?

The research showed that:
- **Hidden size 8** outperformed 16, 32, 64, 128, 256
- **Window 100** was optimal (50 too short, 150+ diminishing returns)
- **Raw data** matched feature-engineered approaches

**"Simpler models with proper data labeling beat complex models"**

---

## Expected Performance

### Research Benchmarks (Korean Stocks)
- LSTM (raw OHLCV): **F1 = 0.4312**
- XGBoost (indicators): F1 = 0.4316
- Baseline: F1 = 0.1852

### Your Results (Indian Stocks)

Run comparison to see how it performs on NSE stocks!

```bash
python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE --compare
```

---

## Troubleshooting

### Issue: "Insufficient data"
```python
# Need at least 300+ samples after labeling
# Solution: Use longer period
df = get_raw_ohlcv_data('SYMBOL', period='5y')  # Instead of '1y'
```

### Issue: "Poor performance (F1 < 0.3)"
```python
# Try auto-tuning barriers for Indian market
upper, lower, horizon = create_balanced_barriers(df)
```

### Issue: "Overfitting (train >> test)"
```python
# Increase dropout or reduce model size
model = RawOHLCVLSTM(hidden_size=4, dropout=0.3)
```

---

## Key Takeaways

1. **Don't Over-Engineer**: Raw data can match engineered features
2. **Smaller Can Be Better**: Hidden size 8 > 128
3. **Better Labels Matter**: Triple barrier > simple returns
4. **Test on Your Market**: Korean findings may differ for Indian stocks
5. **A/B Test Everything**: Compare approaches scientifically

---

## Next Steps

1. **Train on a stock:**
   ```bash
   python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE
   ```

2. **Compare approaches:**
   ```bash
   python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE --compare
   ```

3. **Optimize for Indian market:**
   ```bash
   python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE --grid-search
   ```

4. **Check the comparison report:**
   ```
   cat ml_models/comparison/comparison_RELIANCE.md
   ```

---

## References

- **Research Paper**: Korean Stock Prediction with LSTM and Triple Barrier Labeling (2024)
- **Triple Barrier Method**: "Advances in Financial Machine Learning" by Marcos López de Prado
- **Implementation**: `/src/services/ml/raw_ohlcv_lstm.py`, `/src/services/ml/triple_barrier_labeling.py`

---

**Built with ❤️ for smarter, simpler stock prediction**
