# Positional Swing Models (Not M3 Momentum) — Backtest

_Goal: positional model (weeks-to-months hold) beating +16%/yr, different driver than M3. Window: May 2023 → May 2026. N100 universe. ₹10L capital._

## All 4 Tested

| Model | Driver | CAGR | MaxDD | Sharpe | M3 Corr | Trades | Avg Hold |
|-------|--------|----:|-----:|------:|-------:|------:|--------:|
| **M4 mtf_trend** ⭐ | Multi-TF trend (daily+weekly EMAs) | **+61.8%** | -32.8% | **1.61** | **+0.25** | 65 | 83d |
| M1 weekly_momentum_q | 12-week return rank, quarterly | +68.7% | -31.0% | 1.30 | +0.74 | 65 | 83d |
| M2 52w_breakout_pos | 250d high + trail SL | +47.7% | -25.5% | 1.47 | +0.95 | 211 | 20d |
| M3 bb_squeeze | Bollinger band contraction → breakout | +13.9% | -43.8% | 0.63 | -0.57 | 166 | 25d |

## Winner: M4 Multi-Timeframe Trend

```
ENTRY: stock simultaneously meets all 4 conditions
  - Daily close > 50-EMA
  - Daily 50-EMA > 200-EMA
  - Weekly close > 50-EMA (weekly trend confirmation)
  - Ranked top-5 by EMA50/EMA200 ratio

HOLD: 90 calendar days fixed
REBALANCE: quarterly (Jan/Apr/Jul/Oct)
POSITIONS: top-5 equal-weight
EXIT: 90-day timer OR drops out of top-5 next quarter
```

**Why it works:** Multi-TF filter eliminates ~half the candidates that pure 60d momentum (M3) picks. Stocks must show strength on BOTH daily AND weekly → fewer false positives.

## Why Each Loser Rejected

| Model | Why Rejected |
|-------|--------------|
| M1 weekly_momentum_q | Same driver as M3, just slower. Corr 0.74 = redundant. No diversification value. |
| M2 52w_breakout_pos | Corr 0.95 with M3 — same regime trigger different label. Avg hold 20d (trail SL fires) = not really positional. |
| M3 bb_squeeze | Genuinely uncorrelated (-0.57) but CAGR 13.9% < threshold AND MaxDD -43.8% catastrophic. |

## M4 vs Model B (the prior pick)

| | M4 mtf_trend | Model B (low-vol quarterly) |
|--|-------------|-----------------------------|
| CAGR | +61.8% | +31.7% |
| Sharpe | 1.61 | **1.98** |
| MaxDD | -32.8% | **-3.5%** |
| M3 Corr | +0.25 | +0.045 |
| Driver | Trend (slower) | Low-vol anomaly |

**Trade-off:** M4 has 2× the CAGR but 9× the DD. Model B is safer; M4 swings harder.

## Recommended Allocation Options (paired with M3)

### Option 1: M3 + M4 (aggressive)
```
M3: 60% (₹6L) — momentum rotation
M4: 40% (₹4L) — multi-TF trend
Expected: ~76% CAGR, blended Sharpe ~1.3, DD ~20-25%
```

### Option 2: M3 + B (conservative, original)
```
M3: 60% (₹6L) — momentum rotation
B:  40% (₹4L) — low vol
Expected: ~65% CAGR, blended Sharpe ~1.7, DD <8%
```

### Option 3: Triple (M3 + M4 + B)
```
M3: 50% (₹5L) — momentum
M4: 25% (₹2.5L) — trend (different timeframe)
B:  25% (₹2.5L) — low vol stabilizer
Expected: ~58% CAGR, blended Sharpe ~1.7, DD ~10-15%
```

## Bottom Line

**M4 mtf_trend is the only NEW positional model that meaningfully beats +16%/yr AND has acceptable correlation with M3.**

- If max CAGR matters → 60/40 M3+M4
- If lowest DD matters → 60/40 M3+B
- If want both → 50/25/25 M3+M4+B

## Implementation Notes

- **Rebalance frequency:** quarterly (4×/yr) = lowest automation cost
- **Reuses existing momrot infrastructure** — same universe file, same trade path; only swap ranking function (~50 LOC)
- **Capacity:** N100 large caps, ₹10L position size zero slippage; scales to ₹10Cr+
- **Risk:** long-only equity, no leverage, MaxDD -33% — same scale as M3 alone

## Files

```
exports/backtests/POSITIONAL_ALT_MODELS.md
remote: /tmp/positional_backtest.py + /tmp/positional_results.json
```
