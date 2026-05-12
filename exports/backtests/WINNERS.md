# Monthly + Yearly Winners — ₹2,00,000 ₹2L Backtest

Source: 48 backtests in `yearly_filter/` + `yearly_nofilter/`. Cache-only
mode, 3 years, max_concurrent=2.

## Monthly P&L Wins (out of 37 months)

| Universe / Filter | ema_9_21 | ema_200_400 | swing_pullback | orb_15min |
|---|---:|---:|---:|---:|
| N50 filter | **16 (43%)** | 15 (41%) | 4 (11%) | 0 |
| N500 filter | **20 (54%)** | 9 (24%) | 5 (14%) | 0 |

**ema_9_21 wins more individual months** (43-54% of the time) — but doesn't win every month.

## Yearly ROI — All 3 Years Positive Check

Only configurations where **every** year ended positive on ₹2L:

| Config | 2023-24 | 2024-25 | 2025-26 | 3/3 +ve? |
|---|---:|---:|---:|:---:|
| **ema_200_400 N50 (filter ON or OFF)** | **+98.04%** | **+54.47%** | **+5.41%** | **✓** |
| **ema_200_400 N500 nofilter** | +21.25% | +1.05% | +2.52% | ✓ |
| ema_9_21 N50 filter | +62.37% | +27.33% | -11.11% | ✗ |
| ema_9_21 N500 nofilter | +118.24% | -22.78% | -20.46% | ✗ |
| swing N50 / N500 | +5.91% / +12.81% | -7.50% / -16.21% | -1.09% / +0.13% | ✗ |

## Overall Winner

**EMA 200/400 on Nifty 50** = the only model that combines:
- ✅ 3/3 years positive (+98%, +54%, +5%)
- ✅ Lowest worst-year MDD: **13.02%**
- ✅ 3-year avg ROI: **+52.64%**
- ✅ Wins 15/37 (41%) of months — close to ema_9_21's 43%

## Trade-offs

- **ema_200_400 N50**: most consistent, all-year winner, moderate volatility. **Best for paper/live ₹2L deploy.**
- **ema_9_21 N500 nofilter**: highest peak year (+118%), wins most months — but loses 2 of 3 years. High variance, regime-dependent.
- **swing_pullback**: low signal density (~37 trades / yr), negative 3-yr avg both universes. Skip.
- **orb_15min**: no 5m bar Postgres cache → backtest can't compute → 0% across the board. Skip until 5m cache table added.

## Honest caveat

ema_200_400 best-year +98% reflects a strong 2023-24 bull regime. Don't extrapolate that as a future expectation. The +5% on 2025-26 may be closer to baseline. Plan for **annualized ~10-15% net** after slippage/STT in a normal market.
