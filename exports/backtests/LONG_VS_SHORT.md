# Momentum Rotation — Long vs Short Backtest

_3-year window May 2023 → May 2026, N100 universe, monthly rebalance, top-N=5, max-concurrent=1, ₹10L capital. ₹5L per side for long-short._

## Verdict: Long-only wins by huge margin

| Mode | 3-yr Avg ROI/yr | 3-yr Compound | Worst-yr MaxDD |
|------|-------------:|-------------:|------:|
| ✅ **long_only** | **+62.94%** | **+236.74%** | 24.78% |
| ❌ short_only | -35.77% | -85.79% | 81.31% |
| ❌ long_short (50/50) | +13.59% | +42.32% | 34.84% |
| ❌ turncoat (short ex-leaders) | +21.67% | +69.21% | 16.14% |

## Per-Year ROI

| Year | long_only | short_only | long_short | turncoat |
|------|--------:|--------:|--------:|--------:|
| 2023-24 | +132.26% | -80.62% | +25.86% | +26.54% |
| 2024-25 | +72.52%  | -26.68% | +22.87% | +48.33% |
| 2025-26 | -15.96%  | 0.00%   | -7.97%  | -9.85% |

## Why Shorts Destroy Returns

1. **2023-25 was bull market.** Bottom-momentum stocks mean-reverted hard with the tide → short side caught the rallies as losses.
2. **Long-short kills asymmetry.** Hedging out market beta removes the asymmetric upside that makes momentum work.
3. **Turncoat best of shorts** (-leaders fall) but still negative in yr 3.

## Real-World Blockers (Even If Backtest Looked Good)

1. Indian retail **cannot short equity delivery (CNC)** — only intraday MIS.
2. Monthly shorts require stock futures (F&O list ~180 names, not all N100) or SLB → margin 20-30%, roll cost, liquidity issues.
3. Backtest assumed ideal short fills; real fills slip on illiquid bottom-N.

## Caveat — Same Lookahead Bias

Universe = 2026-05-13 N100 retroactive across 2023-2025. Inflates all numbers (survivor bias). Validated walk-forward long-only baseline (`MODEL3_TRADE_LEDGER.md`) is +86.93%/yr — slightly different from +62.94% here due to top-N + cap variant.

## Recommendation

**Stay long-only.** Adding shorts strictly destroys returns on this dataset AND tradeability blocks live implementation. >50%/yr threshold already met by long-only.

## Files

```
tools/backtests/momrot_long_short_backtest.py
tools/backtests/run_long_short_all.sh
exports/backtests/long_short_results/*.json
```
