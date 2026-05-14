# 5%/Trade Profit-Booking — Short-Term Strategy Backtest

_Goal: ≥5% target per trade, MAJORITY (>55%) trades hit target, short-term (1-15 days). Universe: Nifty 100. Window: May 2024 → May 2026 (2 yrs). Capital: ₹10L. Max 5 concurrent. 0.13% round-trip cost._

## Strategies Tested

| # | Strategy | Setup |
|---|----------|-------|
| 1 | 52w_high_breakout | Close > 250d high + vol > 1.5× avg. SL -3%, target +5%, max 15d |
| 2 | ema20_pullback | Uptrend (close > 50ema, 20>50) + low touches 20ema + close above. SL -2%, target +5%, max 10d |
| 3 | gap_up_continuation | Open > yesterday close × 1.02 + above 20ema. SL -2%, target +5%, max 5d |
| 4 | bull_engulfing_support | Bullish engulfing near 50ema. SL -2%, target +5%, max 7d |
| 5 | higher_high_breakout | Today high > 20d high + close > open + vol > 1.3×. SL -2%, target +5%, max 7d |

## Results

| Strategy | Trades | Win% | **5%-Hit%** | Total ROI | Avg/mo | MaxDD | Sharpe |
|----------|------:|----:|----------:|---------:|------:|-----:|------:|
| **52w_high_breakout** ⭐ | 330 | 45.2 | **42.7** | +31.0% | +1.29 | 12.0 | **1.73** |
| ema20_pullback | 685 | 36.1 | 30.2 | **+36.0%** | +1.50 | 11.3 | 1.11 |
| higher_high_breakout | 739 | 36.7 | 27.5 | +22.8% | +0.95 | 18.6 | 0.80 |
| bull_engulfing_support | 396 | 38.1 | 23.5 | +13.3% | +0.55 | 9.4 | 0.93 |
| gap_up_continuation | 553 | 28.4 | 25.7 | -19.2% | -0.80 | 35.8 | -1.34 |

## ❌ HONEST: 5% with Majority Hits = NOT Achievable

**No strategy delivered majority (>55%) 5%-target hits. Best = 42.7%.**

Why: requiring exact +5% before -2% to -3% stop on noisy daily bars = inherently minority outcome. R/R 1.67:1 caps win rate ~45% on simple breakouts.

## Winner: 52w_high_breakout

- 42.7% target hits (best)
- Sharpe 1.73 (best by big margin)
- 12% MaxDD (manageable)
- ~14 trades/month (good frequency)
- 31% total ROI over 2 yrs (~1.3%/mo)

Highest 5%-target rate AND highest Sharpe. Trend continuation: stocks at new 52-wk highs on volume = asymmetric upside.

## Top 10 Stocks Where It Works Best

| Stock | Trades | 5% Hits | Hit% | Total P&L |
|-------|------:|------:|----:|---------:|
| **KAYNES** | 7 | 7 | **100%** | ₹78,532 |
| **SHRIRAMFIN** | 3 | 3 | **100%** | ₹34,222 |
| **HFCL** | 3 | 3 | **100%** | ₹36,260 |
| COCHINSHIP | 5 | 4 | 80% | ₹37,137 |
| ABB | 4 | 3 | 75% | ₹25,654 |
| M&M | 4 | 3 | 75% | ₹24,140 |
| ARE&M | 4 | 3 | 75% | ₹23,991 |
| SUZLON | 4 | 3 | 75% | ₹24,367 |
| DIXON | 6 | 4 | 66.7% | ₹27,714 |
| ADANIPOWER | 6 | 4 | 66.7% | ₹29,328 |

Theme: high-beta growth stocks + recent IPO winners (KAYNES, HFCL) + defense/EV/power infra.

## What Would Actually Work for Majority Hits

To get >55% hit rate at 5%:
1. **Lower target to 2-3%** (faster hits) — but then per-trade alpha small
2. **Volatility-scaled targets** (1× ATR not fixed 5%) — adaptive
3. **Intraday bars** (1h/15m) — finer entries, more setups
4. **Stricter setup filters** — fewer trades, higher quality

## Real-World Implementation Notes

| Aspect | Detail |
|--------|--------|
| Cost | 0.13% RT already baked. Real Zerodha ~0.12% — backtest slightly conservative. |
| Slippage | Breakouts have wider spreads. +0.1% on entry realistic. Net Sharpe ~1.4-1.5. |
| Automation | Trivial — EOD scan, 1-2 bracket orders pre-open. Fits existing scheduler infra. |
| Capacity | Top stocks > ₹50cr/day ADV. ₹10L position zero impact. |
| Trades/month | ~14 — manageable, brokerage drag ~1.8%/yr |

## Recommendation

**Don't use as primary income engine — 1.3%/mo (~16%/yr) is far below M3's 87%/yr.**

Acceptable as:
- **Diversifier alongside M3** (separate alpha source — trend continuation)
- **Paper-trade first** for 1-2 months to validate before committing capital

NOT acceptable as standalone replacement for M3.

## Files

```
exports/backtests/PROFIT_BOOK_5PCT.md
remote: /tmp/backtest_5pct.py + /tmp/backtest_5pct_results.json
```
