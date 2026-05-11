# JM Financial Ltd. (JMFINANCIL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 140.69
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 4
- **Avg / median % per leg:** 2.88% / 0.00%
- **Sum % (uncompounded):** 34.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 1 | 7 | 4 | 2.88% | 34.6% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 1 | 7 | 4 | 2.88% | 34.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 5 | 41.7% | 1 | 7 | 4 | 2.88% | 34.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 00:00:00 | 75.50 | 68.16 | 73.44 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=2.02 |
| Stop hit — per-position SL triggered | 2023-07-24 00:00:00 | 72.48 | 68.51 | 73.72 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-07-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 00:00:00 | 76.80 | 68.74 | 73.95 | Stage2 pullback-breakout RSI=65 vol=6.6x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 00:00:00 | 80.82 | 69.06 | 75.44 | T1 booked 50% @ 80.82 |
| Stop hit — per-position SL triggered | 2023-08-04 00:00:00 | 76.80 | 69.23 | 75.86 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2023-08-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 00:00:00 | 78.25 | 69.66 | 75.33 | Stage2 pullback-breakout RSI=57 vol=6.4x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-21 00:00:00 | 84.24 | 69.89 | 76.38 | T1 booked 50% @ 84.24 |
| Stop hit — per-position SL triggered | 2023-08-23 00:00:00 | 78.25 | 70.07 | 76.83 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 86.10 | 70.64 | 77.88 | Stage2 pullback-breakout RSI=69 vol=3.2x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 00:00:00 | 93.34 | 71.24 | 81.28 | T1 booked 50% @ 93.34 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 86.10 | 71.98 | 84.09 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2023-10-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 00:00:00 | 88.45 | 74.03 | 85.50 | Stage2 pullback-breakout RSI=59 vol=2.4x ATR=3.33 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 87.10 | 75.37 | 87.36 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 88.70 | 76.00 | 84.81 | Stage2 pullback-breakout RSI=58 vol=5.1x ATR=3.73 |
| Stop hit — per-position SL triggered | 2023-11-07 00:00:00 | 83.11 | 76.16 | 84.68 | SL hit (bars_held=2) |

### Cycle 7 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 85.15 | 76.60 | 83.76 | Stage2 pullback-breakout RSI=53 vol=1.6x ATR=3.23 |
| Stop hit — per-position SL triggered | 2023-12-01 00:00:00 | 84.85 | 77.30 | 83.91 | Time-stop (10d <3%) |

### Cycle 8 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 86.90 | 77.40 | 84.20 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 00:00:00 | 92.11 | 78.44 | 87.77 | T1 booked 50% @ 92.11 |
| Target hit | 2024-02-09 00:00:00 | 103.80 | 86.60 | 106.80 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-14 00:00:00 | 75.50 | 2023-07-24 00:00:00 | 72.48 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest1 | 2023-07-28 00:00:00 | 76.80 | 2023-08-02 00:00:00 | 80.82 | PARTIAL | 0.50 | 5.24% |
| BUY | retest1 | 2023-07-28 00:00:00 | 76.80 | 2023-08-04 00:00:00 | 76.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-17 00:00:00 | 78.25 | 2023-08-21 00:00:00 | 84.24 | PARTIAL | 0.50 | 7.66% |
| BUY | retest1 | 2023-08-17 00:00:00 | 78.25 | 2023-08-23 00:00:00 | 78.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-01 00:00:00 | 86.10 | 2023-09-06 00:00:00 | 93.34 | PARTIAL | 0.50 | 8.40% |
| BUY | retest1 | 2023-09-01 00:00:00 | 86.10 | 2023-09-12 00:00:00 | 86.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-06 00:00:00 | 88.45 | 2023-10-20 00:00:00 | 87.10 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest1 | 2023-11-03 00:00:00 | 88.70 | 2023-11-07 00:00:00 | 83.11 | STOP_HIT | 1.00 | -6.30% |
| BUY | retest1 | 2023-11-16 00:00:00 | 85.15 | 2023-12-01 00:00:00 | 84.85 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-12-04 00:00:00 | 86.90 | 2023-12-15 00:00:00 | 92.11 | PARTIAL | 0.50 | 6.00% |
| BUY | retest1 | 2023-12-04 00:00:00 | 86.90 | 2024-02-09 00:00:00 | 103.80 | TARGET_HIT | 0.50 | 19.45% |
