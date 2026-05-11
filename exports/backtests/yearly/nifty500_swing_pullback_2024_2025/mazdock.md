# Mazagoan Dock Shipbuilders Ltd. (MAZDOCK)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 2657.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 3.45% / 8.33%
- **Sum % (uncompounded):** 24.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 3 | 3 | 3.45% | 24.1% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 3 | 3 | 3.45% | 24.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 3 | 3 | 3.45% | 24.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 05:30:00 | 2387.40 | 1672.96 | 2277.36 | Stage2 pullback-breakout RSI=55 vol=2.5x ATR=129.19 |
| Stop hit — per-position SL triggered | 2024-09-06 05:30:00 | 2193.61 | 1684.28 | 2271.06 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-10-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 05:30:00 | 2214.18 | 1771.32 | 2110.37 | Stage2 pullback-breakout RSI=56 vol=4.5x ATR=95.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 05:30:00 | 2404.52 | 1800.46 | 2160.88 | T1 booked 50% @ 2404.52 |
| Stop hit — per-position SL triggered | 2024-10-22 05:30:00 | 2214.18 | 1803.24 | 2153.10 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2024-11-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 05:30:00 | 2117.07 | 1852.95 | 2053.32 | Stage2 pullback-breakout RSI=55 vol=2.1x ATR=88.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 05:30:00 | 2293.47 | 1860.77 | 2088.72 | T1 booked 50% @ 2293.47 |
| Target hit | 2024-12-20 05:30:00 | 2362.05 | 1947.20 | 2389.82 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-01-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 05:30:00 | 2438.45 | 2001.74 | 2268.68 | Stage2 pullback-breakout RSI=60 vol=2.0x ATR=116.61 |
| Stop hit — per-position SL triggered | 2025-01-22 05:30:00 | 2263.54 | 2008.04 | 2277.66 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2025-04-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 05:30:00 | 2661.30 | 2154.43 | 2486.51 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=145.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 05:30:00 | 2951.62 | 2207.44 | 2662.79 | T1 booked 50% @ 2951.62 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-04 05:30:00 | 2387.40 | 2024-09-06 05:30:00 | 2193.61 | STOP_HIT | 1.00 | -8.12% |
| BUY | retest1 | 2024-10-10 05:30:00 | 2214.18 | 2024-10-21 05:30:00 | 2404.52 | PARTIAL | 0.50 | 8.60% |
| BUY | retest1 | 2024-10-10 05:30:00 | 2214.18 | 2024-10-22 05:30:00 | 2214.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-26 05:30:00 | 2117.07 | 2024-11-28 05:30:00 | 2293.47 | PARTIAL | 0.50 | 8.33% |
| BUY | retest1 | 2024-11-26 05:30:00 | 2117.07 | 2024-12-20 05:30:00 | 2362.05 | TARGET_HIT | 0.50 | 11.57% |
| BUY | retest1 | 2025-01-20 05:30:00 | 2438.45 | 2025-01-22 05:30:00 | 2263.54 | STOP_HIT | 1.00 | -7.17% |
| BUY | retest1 | 2025-04-15 05:30:00 | 2661.30 | 2025-04-29 05:30:00 | 2951.62 | PARTIAL | 0.50 | 10.91% |
