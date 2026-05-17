# n20_daily_v2_minret10

**v2 of `n20_daily_30d_mc1_uptrend`** with loss-minimization filter applied.

## What's different from v1

| Knob | v1 (n20_daily) | **v2 (this)** |
|---|---|---|
| Universe | Top-20 ADV + uptrend filter | Same |
| Lookback | 30 days | Same |
| Position | top-1 (max_concurrent=1) | Same |
| Rebalance | Daily | Same |
| **New filter** | — | **Min 30d return ≥ 10%** (skip weak momentum picks) |

If the rank-1 stock has 30d return < 10%, **sit in cash** instead of entering a weak setup. Strategy waits for stronger momentum.

## Stock pick logic (plain English)

1. Build universe (per day): top-20 N500 stocks by 20-day ADV
2. Apply uptrend filter: keep only stocks where close > 200d SMA
3. Rank remaining by 30-day return
4. **NEW**: drop any stock with 30d return < 10%
5. Pick rank-1 from filtered set (if any); else hold cash
6. Rebalance daily

## Why this filter

Loss-minimization sweep tested 11 filter variants (hard SL, trailing SL, no-reentry blocks, min-hold periods, min-return thresholds). Most were neutral or harmful — daily rotation already exits losers fast, hard SL never fires before next-day rotation, re-entry blocks miss winners.

**Only winning filter**: skipping weak-momentum picks (30d return below 10%). Avoids low-conviction entries.

## Backtest result (₹10L, 2023-05-15 → 2026-05-12)

| Metric | v1 baseline | **v2 (min 30d ≥ 10%)** | Δ |
|---|---:|---:|---:|
| Final NAV | ₹1.70 Cr | **₹1.65 Cr** | -₹5L |
| CAGR | +157.11% | **+154.72%** | -2.4pp |
| Max DD | 50.61% | **48.04%** | **-2.6pp** ✅ |
| Calmar | 3.10 | **3.22** | +0.12 ✅ |
| Trades | 134 | **127** | -7 (less churn) |
| WR | 47.8% | 45.7% | -2.1pp |

**Marginal improvement** — slightly lower DD, slightly higher Calmar, 7 fewer trades (less cost drag). CAGR drops marginally because some weak-momentum picks would have worked but didn't reach 10% threshold.

## All other filters tested (DID NOT HELP)

| Filter | Result |
|---|---|
| Hard SL -5% | No change (rotation exits faster than SL fires) |
| Hard SL -7% | No change |
| Trail SL -10% from peak | Never fires (winners exit on rotation) |
| No re-entry 7 days after loss | -₹43L damage (misses winning re-entries) |
| No re-entry 14 days after loss | -₹103L damage |
| Min 30d return ≥ 20% | Too strict, -₹58L damage |
| Min hold 5 trading days | -₹140L damage (kills daily rotation edge) |

**Lesson**: daily rotation IS the loss filter. Extra SL/timing rules harm more than help.

## Files

| File | Purpose |
|---|---|
| `backtest.py` | Standalone reproducer (v2 winning config) |
| `trade_ledger.json` | 127 trades + summary |

See `exports/models/n20_daily_v2_minret10/SUMMARY.md` + `TRADE_LEDGER.md` for full ledger.
