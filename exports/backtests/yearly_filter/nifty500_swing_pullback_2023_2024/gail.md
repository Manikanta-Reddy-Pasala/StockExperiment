# GAIL (India) Ltd. (GAIL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 163.22
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 2
- **Target hits / Stop hits / Partials:** 2 / 3 / 4
- **Avg / median % per leg:** 2.95% / 4.44%
- **Sum % (uncompounded):** 26.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 7 | 77.8% | 2 | 3 | 4 | 2.95% | 26.5% |
| BUY @ 2nd Alert (retest1) | 9 | 7 | 77.8% | 2 | 3 | 4 | 2.95% | 26.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 7 | 77.8% | 2 | 3 | 4 | 2.95% | 26.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 106.45 | 100.88 | 105.81 | Stage2 pullback-breakout RSI=52 vol=2.3x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 00:00:00 | 109.99 | 101.09 | 106.48 | T1 booked 50% @ 109.99 |
| Stop hit — per-position SL triggered | 2023-07-17 00:00:00 | 108.60 | 101.66 | 107.94 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 121.35 | 105.58 | 116.17 | Stage2 pullback-breakout RSI=64 vol=2.7x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 00:00:00 | 127.07 | 105.93 | 117.42 | T1 booked 50% @ 127.07 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 121.35 | 107.07 | 121.06 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2023-10-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 00:00:00 | 129.35 | 109.87 | 123.21 | Stage2 pullback-breakout RSI=65 vol=3.2x ATR=3.08 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 124.74 | 110.99 | 125.82 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 131.90 | 114.04 | 125.41 | Stage2 pullback-breakout RSI=69 vol=6.4x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 137.79 | 114.54 | 127.90 | T1 booked 50% @ 137.79 |
| Target hit | 2023-12-20 00:00:00 | 137.75 | 117.64 | 137.86 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 189.10 | 146.63 | 180.33 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=6.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 00:00:00 | 201.48 | 147.99 | 183.65 | T1 booked 50% @ 201.48 |
| Target hit | 2024-05-06 00:00:00 | 197.80 | 156.76 | 200.54 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 106.45 | 2023-07-06 00:00:00 | 109.99 | PARTIAL | 0.50 | 3.32% |
| BUY | retest1 | 2023-07-03 00:00:00 | 106.45 | 2023-07-17 00:00:00 | 108.60 | STOP_HIT | 0.50 | 2.02% |
| BUY | retest1 | 2023-09-01 00:00:00 | 121.35 | 2023-09-05 00:00:00 | 127.07 | PARTIAL | 0.50 | 4.71% |
| BUY | retest1 | 2023-09-01 00:00:00 | 121.35 | 2023-09-13 00:00:00 | 121.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-12 00:00:00 | 129.35 | 2023-10-20 00:00:00 | 124.74 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest1 | 2023-11-30 00:00:00 | 131.90 | 2023-12-04 00:00:00 | 137.79 | PARTIAL | 0.50 | 4.47% |
| BUY | retest1 | 2023-11-30 00:00:00 | 131.90 | 2023-12-20 00:00:00 | 137.75 | TARGET_HIT | 0.50 | 4.44% |
| BUY | retest1 | 2024-04-03 00:00:00 | 189.10 | 2024-04-08 00:00:00 | 201.48 | PARTIAL | 0.50 | 6.55% |
| BUY | retest1 | 2024-04-03 00:00:00 | 189.10 | 2024-05-06 00:00:00 | 197.80 | TARGET_HIT | 0.50 | 4.60% |
