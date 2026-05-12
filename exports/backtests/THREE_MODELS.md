# 3 Models — Production-Ready Trading Stack

_Updated: 2026-05-12 | Capital: ₹10,00,000_

## Three Recommended Models

| # | Model | Type | Bars | Best ROI/yr | DD% | Trades/yr |
|---|-------|------|------|-----------:|----:|----------:|
| **1** | **EMA 200/400** | Swing | 1H | **+53.26%** | 13.06 | 119 |
| **2** | **ORB-60** | Day trade | 15m | +14.70% | **5.60** | 643 |
| **3** | **EMA 9/21 filtered** | Short swing | 1H | +6.62% | 9.80 | 37 |

## Each model — what it does, when it fires, when it exits

### Model 1: EMA 200/400 (Swing — multi-week holds)

**When BUY fires:**
1. EMA(200) crosses ABOVE EMA(400) on 1H bar
2. Price retraces back to EMA(200) (retest1 marked)
3. Bar high breaks above retest1.high → BUY entry

**Stop loss:** EMA(400) close (trails as EMA rises)
**Partial:** +5% target → sell 50%, move SL to entry
**Final target:** +10%
**Max hold:** No time stop — exits on TARGET/STOP

**Reverse (SELL):** Mirror logic — EMA 200 cross below EMA 400.

**3-yr backtest (N50, max=2, ₹10L):**
- 2023-24: +98.13% / 13.06% DD / 179 trades
- 2024-25: +54.88% / 13.06% DD / 125 trades
- 2025-26: +6.77% / 13.01% DD / 54 trades

### Model 2: ORB-60 (Day Trade — same-day exit)

**When BUY fires (09:15-15:30 IST only):**
1. ORB high = max(high) over 09:15-10:14 (first hour, 4× 15m bars)
2. After 10:15: any 15m bar closes above ORB high
3. AND volume of that bar > 1.5× ORB-window avg volume
4. → BUY entry at bar close

**Stop loss:** ORB low (opposite side)
**Target:** entry + (daily ATR(14) × 1.5)
**EOD force-close:** 15:20 IST (no overnight)
**One entry per direction per day**

**3-yr backtest (N50, max=2, ₹10L):**
- 2023-24: +9.96% / 5.36% DD / 1141 trades
- 2024-25: +28.44% / 5.60% DD / 668 trades
- 2025-26: +5.69% / 2.32% DD / 121 trades

### Model 3: EMA 9/21 Filtered (Short Swing — 1-5 day holds)

**When BUY fires:**
1. EMA(9) crosses ABOVE EMA(21) on 1H bar
2. **Filter 1:** Gap between EMAs ≥ 0.3% on cross (no touching crosses)
3. **Filter 2:** Bar volume > 1.5× 20-bar avg
4. **Filter 3:** Higher-timeframe SMA-200 trend matches (BUY only if close > daily SMA-200)
5. Retest of EMA(21) → break high → ENTRY

**Stop loss:** EMA(21) close
**Partial:** +5% → sell half
**Target:** +10%

**3-yr backtest (N50, max=2, ₹10L):**
- 2023-24: +18.08% / 9.79% DD / 57 trades
- 2024-25: +8.24% / 9.80% DD / 40 trades
- 2025-26: -6.46% / 9.80% DD / 13 trades

## Detailed trade ledgers (every buy/sell logged)

| Model | Trades | Wins | Losses | Win% | Total P&L | Ledger |
|-------|-------:|-----:|-------:|-----:|----------:|--------|
| 1: EMA 200/400 | 61 | 27 | 34 | 44.3% | ₹+51,047 | [ledgers/MODEL_1_EMA_200_400.md](ledgers/MODEL_1_EMA_200_400.md) |
| 2: ORB-60 | 121 | TBD | TBD | TBD | TBD | [ledgers/MODEL_2_ORB_60.md](ledgers/MODEL_2_ORB_60.md) |
| 3: EMA 9/21 filtered | 16 | TBD | TBD | TBD | TBD | [ledgers/MODEL_3_EMA_9_21_FILT.md](ledgers/MODEL_3_EMA_9_21_FILT.md) |

Each ledger has columns: # / Entry Date / Entry Time / Exit Date / Exit Time / Symbol / Qty / Buy ₹ / Sell ₹ / P&L ₹ / P&L % / Bars / Reason / Cash After ₹.

Also CSV format in `exports/backtests/ledgers/MODEL_*.csv` for spreadsheet analysis.

## Three production approaches

### Approach A — Single model (simplest)
Pick the one that fits your risk profile:
- Max ROI: **Model 1** (EMA 200/400 swing)
- Min DD: **Model 2** (ORB-60 day trade)
- Hybrid: **Model 3** (EMA 9/21 filtered)

### Approach B — Multi-sleeve (diversified)
Run all 3 models on separate capital sleeves:
- ₹4L → Model 1 swing
- ₹4L → Model 2 day trade
- ₹2L → Model 3 short swing
- Different time horizons = uncorrelated returns
- Theoretical combined: ~+30%/yr at 8% DD

### Approach C — Regime-aware switching (smart)

Single capital pool. Switch model based on market regime:

```python
# Determine regime daily (Nifty 50 trend + ATR)
nifty_above_200dma = (nifty_close > nifty_sma_200)
nifty_atr_pct = atr14 / nifty_close * 100

if nifty_above_200dma and nifty_atr_pct < 1.0:
    use_model = "EMA_200_400"   # bull + calm = trending → swing wins
elif nifty_above_200dma and nifty_atr_pct >= 1.5:
    use_model = "ORB_60"        # bull + volatile = day trade
elif not nifty_above_200dma:
    use_model = "ORB_60"        # bear = stay defensive
else:
    use_model = "EMA_9_21_FILT" # neutral = short swing
```

**Untested.** Would need separate backtest. Theoretical: combines each model's
best regime, reduces overall DD.

## Regime-aware selector — Conceptual

Implementation outline (not yet built):

1. **Daily morning routine:**
   - Fetch Nifty 50 daily close + EMA50 + EMA200 + ATR14
   - Compute regime: bull_calm / bull_vol / neutral / bear
   - Pick model based on regime
   - Write `signals/active_model_<date>.json`

2. **Signal generator uses active model:**
   - Reads `active_model.json`
   - Runs only that strategy's evaluator
   - Skips other models

3. **Position monitor:**
   - Continues to monitor open positions from previous model
   - Only NEW entries respect today's active_model

**Risk:** model switching mid-trade could orphan positions. Solution:
allow current model's open trades to run to exit, only NEW entries
follow today's active model.

## What's missing (to be honest)

1. **Sentiment-based selection** — user mentioned. We've avoided news/NLP
   (no free reliable feed in India). Could use price-based proxies:
   - VIX historical (Fyers blocks INDIAVIX — used Nifty 50 ATR instead)
   - PUT/CALL ratio (Nifty options OI — Fyers partial)
   - Advance/Decline ratio (NSE bhavcopy daily)
   - FII/DII flows (NSE fiidiiTradeReact)
   None requires NLP. All scriptable.

2. **Multi-year ORB-60 fine-tune** — current implementation is v1, simple.
   Could improve with:
   - Volume confirmation tuning (1.5× too strict for low-volume days)
   - Anchored VWAP exit (better than ATR target)
   - Time-of-day filter (skip 14:00-15:00 chop)

3. **EMA 9/21 selector universe** — Phase 7 showed selector top-10 +
   EMA 9/21 = +33.32%. Multi-year selector + EMA 9/21 untested.

## Decision needed

1. **Single model or multi-sleeve?**
2. **Build regime-aware selector next?**
3. **Paper-trade Model 1 alone first to validate?**

## Files

- `tools/backtests/trade_ledger.py` — per-trade ledger generator
- `exports/backtests/ledgers/MODEL_{1,2,3}_*.{md,csv}` — detailed trade logs
- `exports/backtests/THREE_MODELS.md` — this file
- `exports/backtests/SUMMARY.md` — top-level summary
- `tools/live/` — production scripts (signal, paper, fyers executors)
