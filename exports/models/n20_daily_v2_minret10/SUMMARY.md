# n20_daily_v2_minret10 — SUMMARY

**v2 of n20_daily_30d_mc1_uptrend** with loss-minimization filter: skip rank-1 picks with 30d return < 10%.

## Stock pick logic (plain English)

1. **Universe (per day)**: top-20 N500 stocks by 20-day ADV
2. **Uptrend filter**: keep only stocks where close > 200d SMA
3. **Rank by 30d return** (highest first)
4. **NEW v2 filter**: drop any candidate with 30d return < 10%
5. **Pick top-1** from filtered set; if empty, **hold cash**
6. **Rebalance daily** (re-rank + rotate)

## Key knobs

| Knob | Value |
|---|---|
| Universe size | 20 |
| Uptrend filter | close > 200d SMA |
| **Momentum threshold (NEW)** | **30d return ≥ 10%** |
| Lookback | 30 days |
| Position | top-1, max_concurrent=1 |
| Rebalance | Daily |
| Cash policy | Sit in cash if no candidate meets threshold |

## Headline result (₹10L, 2023-05-15 → 2026-05-12)

| Metric | v1 baseline | **v2 (this)** | Δ |
|---|---:|---:|---:|
| Final NAV | ₹1.70 Cr | **₹16,527,560** | -₹5L |
| CAGR | +157.11% | **+154.72%** | -2.4pp |
| Max DD | 50.61% | **48.04%** | **-2.6pp** ✅ |
| Calmar | 3.10 | **3.22** | +0.12 ✅ |
| Trades | 134 | **127** | -7 (less churn) |
| WR | 47.8% | 45.7% | -2.1pp |

## Returns by NSE cap segment

| Cap | Trades | Wins | Losses | WR | Total PnL ₹ |
|---|---:|---:|---:|---:|---:|
| **Large** | 48 | 24 | 24 | 50% | +8,611,057 |
| **Mid** | 43 | 22 | 21 | 51% | +6,035,054 |
| **Small** | 36 | 12 | 24 | 33% | +709,218 |

## Yearly money flow

| Year | Open | Close | ROI | Trades |
|---|---:|---:|---:|---:|
| 2023-24 | ₹1,000,000 | ₹3,922,517 | **+292.25%** | 45 |
| 2024-25 | ₹3,922,517 | ₹11,882,358 | **+202.93%** | 35 |
| 2025-26 | ₹11,882,358 | ₹16,355,330 | **+37.64%** | 47 |

## Why other loss filters were rejected

Sweep tested 11 filter variants. Most were neutral or harmful:

| Filter | Result | Why it failed |
|---|---|---|
| Hard SL -5% / -7% | No change | Daily rotation exits losers before SL fires |
| Trail SL -10% from peak | Never fires | Winners exit on rotation, not trail |
| No re-entry 7-14d after loss | -₹43L to -₹103L | Misses winning re-entries (some losses are temporary) |
| Min 30d return ≥ 20% | -₹58L | Too strict, misses moderate-momentum winners |
| Min hold 5 days | -₹140L | Kills daily-rotation edge (can't exit fast) |
| SL + min-hold + re-entry combo | -₹76L | Each filter overlaps, compounds harm |

**Insight**: daily rotation IS the loss filter. Extra SL/timing rules harm more than help.
**Only winning filter**: skip weak-momentum picks (min 30d return ≥ 10%). Avoids low-conviction entries.

## Caveats

- Marginal improvement (+0.12 Calmar). Baseline v1 already near-optimal for daily rotation strategy.
- 50% Max DD still high. Cap-segment filters (drop Mid+Small → Large only) cut DD harder if needed (-25% DD at -16pp CAGR cost).
- 127 trades / 3yr = ~42 round-trips/yr. STT + brokerage drag ~3-5%/yr.
- Min-return threshold (10%) is hardcoded — could be parameter for further tuning.