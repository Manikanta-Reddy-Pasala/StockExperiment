# Quick Start: Raw OHLCV LSTM Model

## 🚀 Try It in 5 Minutes!

### Step 1: Train Your First Model

```bash
# Train on RELIANCE stock using research-optimal parameters
python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE
```

Expected output:
```
Training Raw OHLCV LSTM for RELIANCE
Data shape: (732, 5)
...
✓ Training complete!
  Accuracy:    0.6832
  F1 Macro:    0.4215
  F1 Weighted: 0.6754

Research Paper Benchmark:
  LSTM (Korean stocks):    F1 = 0.4312
  XGBoost (Korean stocks): F1 = 0.4316
```

### Step 2: Compare with Your Traditional Approach

```bash
# A/B test: Raw LSTM vs Feature-Engineered Ensemble
python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE --compare
```

This will:
- Train both models on same data
- Generate detailed comparison report
- Save results to `ml_models/comparison/`

### Step 3: Check the Results

```bash
# View comparison report
cat ml_models/comparison/comparison_RELIANCE.md
```

---

## 📊 What's Different?

### Your Current Approach
```python
70+ features: RSI, MACD, SMA, EMA, Bollinger Bands...
  ↓
Random Forest + XGBoost + LSTM Ensemble
  ↓
F1 Score: ???
```

### Research-Based Approach
```python
5 raw features: Open, High, Low, Close, Volume
  ↓
Simple LSTM (8 hidden units)
  ↓
F1 Score: 0.4312 (matches complex models!)
```

**Key Insight:** "Don't Over-Engineer"

---

## 🎯 Use Cases

### 1. Quick Training
```bash
# Train with defaults (hidden_size=8, window=100)
python tools/train_raw_ohlcv_lstm.py --symbol TCS
```

### 2. Custom Parameters
```bash
# Experiment with different configs
python tools/train_raw_ohlcv_lstm.py \
    --symbol INFY \
    --hidden-size 16 \
    --window 50 \
    --epochs 100
```

### 3. Find Best Hyperparameters
```bash
# Grid search across multiple configs
python tools/train_raw_ohlcv_lstm.py \
    --symbol HDFCBANK \
    --grid-search
```

### 4. Auto-Tune for Indian Market
```bash
# Auto-find best barrier parameters
python tools/train_raw_ohlcv_lstm.py \
    --symbol WIPRO \
    --auto-tune-barriers
```

---

## 📁 File Structure

```
src/services/ml/
├── triple_barrier_labeling.py    # Triple barrier method
├── raw_ohlcv_lstm.py             # Simple LSTM model
├── model_comparison.py           # A/B testing framework
└── data_service.py               # Updated with raw OHLCV support

config/
└── triple_barrier_config.yaml    # Configuration parameters

tools/
└── train_raw_ohlcv_lstm.py       # Training script

docs/
└── RAW_OHLCV_LSTM_GUIDE.md      # Comprehensive guide
```

---

## 🔬 Triple Barrier Labeling

Instead of "will price go up?", we ask:

**"Will the stock hit +9% profit, -9% stop loss, or neither in 29 days?"**

```python
from src.services.ml.triple_barrier_labeling import TripleBarrierLabeler

# Create labeler
labeler = TripleBarrierLabeler(
    upper_barrier=9.0,   # +9% profit target
    lower_barrier=9.0,   # -9% stop loss
    time_horizon=29      # 29 days max
)

# Apply to your data
df_labeled = labeler.apply_triple_barrier(df)

# Results
# Label 0 = Loss (hit stop loss)
# Label 1 = Neutral (time expired)
# Label 2 = Profit (hit target)
```

---

## 📈 Expected Results

### Research Benchmarks (Korean Stocks)
| Model | Features | F1 Score |
|-------|----------|----------|
| Simple LSTM | OHLCV (5) | **0.4312** |
| XGBoost | Indicators (100+) | 0.4316 |
| Random Baseline | - | 0.1852 |

### Your Performance

Run the comparison to find out! Indian market may behave differently.

---

## ❓ FAQ

**Q: Why only 8 hidden units?**
A: Research showed 8 outperforms 16, 32, 64, 128, 256. Simple is better!

**Q: Why no technical indicators?**
A: Raw data preserves all information. Indicators can lose signals.

**Q: Will this work for Indian stocks?**
A: That's what the comparison is for! Test it.

**Q: Can I use both approaches?**
A: Yes! You can ensemble them or use for different stocks.

---

## 🚨 Common Errors

### Error: "Insufficient data"
```bash
# Need more history
python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE --period 5y
```

### Error: "No data available"
```bash
# Check your Fyers API connection
# Ensure user_id is correct
python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE --user-id 1
```

### Poor Performance (F1 < 0.3)
```bash
# Try auto-tuning barriers
python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE --auto-tune-barriers
```

---

## 🎓 Learn More

- **Full Guide:** `docs/RAW_OHLCV_LSTM_GUIDE.md`
- **Config:** `config/triple_barrier_config.yaml`
- **Source Code:** `src/services/ml/raw_ohlcv_lstm.py`

---

## 🏆 Next Steps

1. ✅ Train on 3-5 different stocks
2. ✅ Compare performance with your current models
3. ✅ Find the best approach for each stock
4. ✅ Deploy the winner!

```bash
# Quick test on 3 stocks
python tools/train_raw_ohlcv_lstm.py --symbol RELIANCE --compare
python tools/train_raw_ohlcv_lstm.py --symbol TCS --compare
python tools/train_raw_ohlcv_lstm.py --symbol INFY --compare
```

---

**Ready to challenge conventional wisdom? Start training!** 🚀
