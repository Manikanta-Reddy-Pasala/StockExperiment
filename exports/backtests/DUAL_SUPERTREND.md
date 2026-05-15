# Dual Supertrend — Backtest on N100

_Window: May 2023 → May 2026. Capital: ₹10L. Max 5 concurrent. 0.13% RT cost._

## Strategy Spec

**Dual Supertrend dynamic** on daily bars:
- **Fast Supertrend**: period=7, multiplier=3
- **Slow Supertrend**: period=10, multiplier=4

```
ENTRY: BOTH fast AND slow Supertrends turn bullish (close above both lines)
HOLD:  while slow Supertrend stays bullish (slow line = trailing stop)
EXIT:  BOTH fast AND slow Supertrends flip bearish (close below both)
```

Supertrend formula:
```
HL2 = (high + low) / 2
upper_band = HL2 + multiplier × ATR(period)
lower_band = HL2 − multiplier × ATR(period)
if close > prior_upper → trend = bullish (line = lower_band)
if close < prior_lower → trend = bearish (line = upper_band)
```

## Published Claim vs Reality

| Period | Claim | **Reality (strategy)** | Reality (executed) |
|--------|------:|----------------------:|-------------------:|
| 2023-24 (bull) | 60-70% | **59.7%** ✓ (barely) | 50.0% |
| 2024-25 (mixed) | 15-25% | 40.4% | 50.0% |
| 2025-26 | varies | 43.6% | 36.4% |
| 3-yr lifetime | -- | -- | **42.3%** |

**Verdict on 60-70% bull claim:** Marketing-grade truth. Hits 59.7% ONLY at lower bound in strongest bull year on ALL signals. Drops to 40-44% in mixed regimes. Executed win rate = 42% (5-slot cap binding).

## Full 3-yr Stats

| Metric | Value |
|--------|-----:|
| Total signals | 622 |
| Executed (5-slot cap) | 52 |
| Skipped (no slot) | 570 |
| Executed win rate | **42.3%** |
| Avg trade P&L | +6.07% |
| Avg holding | ~100 days |
| **Total ROI 3-yr** | **+77.89%** |
| **CAGR** | **+22.64%** |
| Sharpe (daily) | 1.12 |
| MaxDD | -22.65% |

## Top 10 Stocks Where It Works

| Symbol | Trades | Win% | Total P&L% | Avg/trade | Avg Hold |
|--------|------:|----:|----------:|---------:|--------:|
| **ANGELONE** | 8 | 62.5 | +1059% | +132% | 48d |
| **GVT&D** | 3 | **100.0** | +795% | +265% | 206d |
| **MCX** | 6 | 66.7 | +596% | +99% | 88d |
| GALLANTT | 8 | 37.5 | +373% | +47% | 69d |
| WOCKPHARMA | 6 | 50.0 | +356% | +59% | 93d |
| POWERINDIA | 6 | 50.0 | +224% | +37% | 80d |
| COCHINSHIP | 9 | 55.6 | +203% | +23% | 50d |
| GRSE | 8 | 50.0 | +173% | +22% | 53d |
| NETWEB | 6 | 66.7 | +164% | +27% | 68d |
| VEDL | 6 | 50.0 | +152% | +25% | 79d |

**Theme:** mid/large-cap megatrend riders (defense, fintech, exchange, EV, power infra). Doesn't work on banks (HDFCBANK, KOTAK, BANDHAN), FMCG (TITAN), or pharma defensives.

## Comparison vs M3 Momentum Rotation

| Strategy | CAGR | MaxDD | Sharpe | Win% | Avg Hold |
|----------|----:|-----:|------:|----:|--------:|
| **M3 Momentum Rotation** ⭐ | **+87.0%** | **-6%** | high | n/a (rotation) | 30d |
| Dual Supertrend | +22.6% | -22.6% | 1.12 | 42% | 100d |

Dual Supertrend = **4× worse CAGR, 4× worse DD** than M3.

## Pros + Cons

| Pros | Cons |
|------|------|
| Simple ATR-based logic, no lookahead | 92% of signals starved by 5-slot cap |
| ~100d avg hold = low operational burden | Long holds tie up capital |
| Sharpe 1.12 acceptable | Win rate collapses 60% → 40% in non-bull regimes |
| Beats Nifty buy-hold | -22.6% DD vs M3's -6% |
| Works great on individual megatrend names | Doesn't match 60-70% all-condition claim |

## Final Recommendation

**Do NOT deploy as primary engine.** M3 momentum rotation already dominates on every metric (CAGR, DD, Sharpe).

Acceptable use:
- **Long-hold confirmation overlay** on M3-picked names (if M3 says BUY + Dual Supertrend is bullish, higher conviction)
- **Standalone for "set and forget" investors** — 100-day avg hold = touch portfolio quarterly, accept lower returns
- NOT a 60-70% win rate engine outside strongest bull years

## Files

```
exports/backtests/DUAL_SUPERTREND.md
tools/backtests/dual_supertrend.py
remote: /app/logs/dual_supertrend/dual_supertrend_report.json + summary.txt
```
