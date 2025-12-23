# Technical Indicators Needed for 8-21 EMA Strategy

## ✅ Required (Core Strategy)

### 1. EMA 8 (8-day Exponential Moving Average)
- **Purpose:** Fast-moving average to capture short-term momentum
- **Calculation:** `EMA = (Close × α) + (Previous_EMA × (1 - α))` where `α = 2/(8+1) = 0.222`
- **Usage:** When price > EMA 8 > EMA 21 = Bullish power zone
- **Storage:** `technical_indicators.ema_8` column

### 2. EMA 21 (21-day Exponential Moving Average)
- **Purpose:** Slow-moving average representing institutional holding period
- **Calculation:** `EMA = (Close × α) + (Previous_EMA × (1 - α))` where `α = 2/(21+1) = 0.091`
- **Usage:** Dynamic support/resistance level
- **Storage:** `technical_indicators.ema_21` column

### 3. DeMarker Oscillator (14-period)
- **Purpose:** Identifies overbought/oversold conditions for precise entry timing
- **Calculation:**
  ```
  DeMax = High(today) - High(yesterday) if > 0, else 0
  DeMin = Low(yesterday) - Low(today) if > 0, else 0
  DeMax_SMA = SMA(DeMax, 14)
  DeMin_SMA = SMA(DeMin, 14)
  DeMarker = DeMax_SMA / (DeMax_SMA + DeMin_SMA)
  ```
- **Range:** 0 to 1
  - **< 0.30:** Oversold (ideal buy opportunity during pullbacks) ✅
  - **0.30-0.70:** Neutral zone
  - **> 0.70:** Overbought (avoid new entries)
- **Storage:** `technical_indicators.demarker` column

### 4. Fibonacci Extension Targets
- **Purpose:** Dynamic profit target calculation
- **Calculation:** Based on recent swing high/low
  ```
  Swing Range = Swing High - Swing Low
  Target 127.2% = Current Price + (Range × 0.272)
  Target 161.8% = Current Price + (Range × 0.618)  [PRIMARY TARGET]
  Target 200.0% = Current Price + (Range × 1.000)
  Target 261.8% = Current Price + (Range × 1.618)
  ```
- **Storage:** Calculated on-the-fly (not stored in database)
- **Why not stored:** Values change based on recent price action

---

## 🔧 Optional (For Context & Confirmation)

### 5. Volume (Already in historical_data table)
- **Purpose:** Confirm breakouts and trend strength
- **Usage:** Higher volume on EMA crossovers = stronger signal
- **Storage:** `historical_data.volume` (already exists)

### 6. SMA 50 (50-day Simple Moving Average)
- **Purpose:** Medium-term trend confirmation
- **Usage:** Price > SMA 50 confirms uptrend
- **Storage:** `technical_indicators.sma_50` (already exists)

### 7. SMA 200 (200-day Simple Moving Average)
- **Purpose:** Major trend identification (bull vs bear market)
- **Usage:** Price > SMA 200 = bull market context
- **Storage:** `technical_indicators.sma_200` (already exists)

---

## ❌ Not Needed (Currently Calculated but Unused)

These indicators are calculated and stored but **NOT used** by the 8-21 EMA strategy:

### Moving Averages (Unused)
- `sma_5`, `sma_10`, `sma_20`, `sma_100` - Not referenced
- `ema_12`, `ema_26` - Old MACD EMAs, not used
- `ema_50` - Not used in 8-21 strategy

### Momentum Indicators (Unused)
- `rsi_14` - RSI not used (DeMarker is superior for this strategy)
- `macd`, `macd_signal`, `macd_histogram` - MACD not used

### Volatility Indicators (Unused)
- `atr_14`, `atr_percentage` - ATR not used (could be useful for position sizing)
- `bb_upper`, `bb_middle`, `bb_lower`, `bb_width` - Bollinger Bands not used

### Trend Indicators (Unused)
- `adx_14` - ADX not used

### Volume Indicators (Unused)
- `obv` - On-Balance Volume not used
- `volume_sma_20`, `volume_ratio` - Not used

### Custom Indicators (Unused)
- `price_momentum_5d`, `price_momentum_20d` - Not used
- `volatility_rank` - Not used

**Total Unused:** ~20 columns (can be removed or kept for future use)

---

## 📊 Database Storage Summary

### Current State (Before Migration)
```
technical_indicators table:
✅ Has: ema_12, ema_26, ema_50, sma_5, sma_10, sma_20, sma_50, sma_100, sma_200
✅ Has: rsi_14, macd, macd_signal, macd_histogram
✅ Has: atr_14, bb_upper, bb_middle, bb_lower, adx_14, obv
❌ Missing: ema_8, ema_21, demarker
```

### Required State (After Migration)
```
technical_indicators table:
✅ Must have: ema_8, ema_21, demarker
✅ Nice to have: sma_50, sma_200, volume (for context)
❌ Optional: All other indicators (kept but not used)
```

---

## 🚀 Quick Implementation Checklist

### Step 1: Add Missing Columns
```bash
# Run migration
docker exec -i trading_system_db psql -U trader -d trading_system < migrations/add_ema_8_21_demarker.sql
```

### Step 2: Populate Data
```bash
# Calculate and store values
python tools/populate_ema_8_21.py
```

### Step 3: Verify
```bash
# Check data exists
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT COUNT(*), COUNT(ema_8), COUNT(ema_21), COUNT(demarker)
FROM technical_indicators
WHERE date >= CURRENT_DATE - INTERVAL '365 days';
"
```

### Step 4: Update Calculation Service
Modify `src/services/data/technical_indicators_service.py` to include:
```python
# Add to _calculate_all_indicators method:
indicators['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
indicators['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
indicators['demarker'] = self._calculate_demarker(df, period=14)
```

---

## 📈 Performance Impact

### Storage
- **3 new columns:** ~10MB additional storage
- **Total database size:** ~500MB → ~510MB (+2%)

### Query Speed
- **Before:** Calculate on-the-fly (slow)
- **After:** Query pre-calculated values (fast)
- **Improvement:** 10-50x faster

### Calculation Time
- **One-time population:** 10-15 minutes
- **Daily updates:** 5-7 minutes (same as before)

---

## 💡 Recommendations

### Do Now:
1. ✅ Add `ema_8`, `ema_21`, `demarker` columns (migration)
2. ✅ Populate historical data (run script)
3. ✅ Update daily calculation service

### Do Later (Optional):
1. Remove unused columns to save space
2. Add composite indexes for faster queries
3. Archive old data (>2 years)

### Don't Do:
1. ❌ Don't delete `sma_50`, `sma_200` - useful for context
2. ❌ Don't delete `volume` - needed for confirmation
3. ❌ Don't delete old columns yet - keep for backtesting

---

## 🎯 Summary

**For 8-21 EMA Strategy, you only need 3 stored indicators:**
1. **EMA 8** - Fast trend
2. **EMA 21** - Slow trend
3. **DeMarker** - Entry timing

**Plus 1 calculated indicator:**
4. **Fibonacci Extensions** - Target prices (calculated on-the-fly)

Everything else is **optional** or **unused**!

---

**Last Updated:** October 31, 2025
**Migration Status:** Ready to apply
