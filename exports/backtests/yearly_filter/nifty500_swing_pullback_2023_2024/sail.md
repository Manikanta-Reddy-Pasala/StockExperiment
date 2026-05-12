# Steel Authority of India Ltd. (SAIL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 181.90
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 2
- **Target hits / Stop hits / Partials:** 4 / 3 / 6
- **Avg / median % per leg:** 6.08% / 3.75%
- **Sum % (uncompounded):** 78.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 11 | 84.6% | 4 | 3 | 6 | 6.08% | 79.0% |
| BUY @ 2nd Alert (retest1) | 13 | 11 | 84.6% | 4 | 3 | 6 | 6.08% | 79.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 11 | 84.6% | 4 | 3 | 6 | 6.08% | 79.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 87.50 | 83.60 | 84.90 | Stage2 pullback-breakout RSI=67 vol=1.6x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 00:00:00 | 90.54 | 83.82 | 86.15 | T1 booked 50% @ 90.54 |
| Stop hit — per-position SL triggered | 2023-07-20 00:00:00 | 89.90 | 84.33 | 88.52 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-07-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 00:00:00 | 92.25 | 84.52 | 89.12 | Stage2 pullback-breakout RSI=68 vol=1.8x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 00:00:00 | 95.71 | 84.96 | 90.92 | T1 booked 50% @ 95.71 |
| Stop hit — per-position SL triggered | 2023-08-02 00:00:00 | 92.25 | 85.04 | 91.12 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2023-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 00:00:00 | 91.40 | 85.68 | 88.48 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 00:00:00 | 95.30 | 85.84 | 89.48 | T1 booked 50% @ 95.30 |
| Target hit | 2023-09-12 00:00:00 | 93.75 | 86.76 | 94.26 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-11-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 00:00:00 | 90.35 | 87.50 | 87.67 | Stage2 pullback-breakout RSI=59 vol=2.2x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 94.23 | 87.82 | 89.97 | T1 booked 50% @ 94.23 |
| Target hit | 2024-01-17 00:00:00 | 113.10 | 94.37 | 114.07 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-01-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 00:00:00 | 119.10 | 95.52 | 114.22 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 00:00:00 | 128.34 | 96.83 | 117.54 | T1 booked 50% @ 128.34 |
| Target hit | 2024-02-12 00:00:00 | 122.75 | 99.14 | 125.74 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-03-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 00:00:00 | 133.00 | 102.67 | 126.20 | Stage2 pullback-breakout RSI=58 vol=2.4x ATR=6.24 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 123.64 | 105.15 | 130.30 | SL hit (bars_held=8) |

### Cycle 7 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 129.00 | 106.25 | 127.64 | Stage2 pullback-breakout RSI=52 vol=1.7x ATR=6.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 00:00:00 | 142.28 | 108.00 | 131.96 | T1 booked 50% @ 142.28 |
| Target hit | 2024-05-09 00:00:00 | 153.05 | 118.21 | 156.12 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 87.50 | 2023-07-10 00:00:00 | 90.54 | PARTIAL | 0.50 | 3.47% |
| BUY | retest1 | 2023-07-03 00:00:00 | 87.50 | 2023-07-20 00:00:00 | 89.90 | STOP_HIT | 0.50 | 2.74% |
| BUY | retest1 | 2023-07-25 00:00:00 | 92.25 | 2023-08-01 00:00:00 | 95.71 | PARTIAL | 0.50 | 3.75% |
| BUY | retest1 | 2023-07-25 00:00:00 | 92.25 | 2023-08-02 00:00:00 | 92.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-30 00:00:00 | 91.40 | 2023-09-01 00:00:00 | 95.30 | PARTIAL | 0.50 | 4.26% |
| BUY | retest1 | 2023-08-30 00:00:00 | 91.40 | 2023-09-12 00:00:00 | 93.75 | TARGET_HIT | 0.50 | 2.57% |
| BUY | retest1 | 2023-11-20 00:00:00 | 90.35 | 2023-12-04 00:00:00 | 94.23 | PARTIAL | 0.50 | 4.29% |
| BUY | retest1 | 2023-11-20 00:00:00 | 90.35 | 2024-01-17 00:00:00 | 113.10 | TARGET_HIT | 0.50 | 25.18% |
| BUY | retest1 | 2024-01-25 00:00:00 | 119.10 | 2024-02-02 00:00:00 | 128.34 | PARTIAL | 0.50 | 7.76% |
| BUY | retest1 | 2024-01-25 00:00:00 | 119.10 | 2024-02-12 00:00:00 | 122.75 | TARGET_HIT | 0.50 | 3.06% |
| BUY | retest1 | 2024-03-01 00:00:00 | 133.00 | 2024-03-13 00:00:00 | 123.64 | STOP_HIT | 1.00 | -7.04% |
| BUY | retest1 | 2024-03-21 00:00:00 | 129.00 | 2024-04-02 00:00:00 | 142.28 | PARTIAL | 0.50 | 10.29% |
| BUY | retest1 | 2024-03-21 00:00:00 | 129.00 | 2024-05-09 00:00:00 | 153.05 | TARGET_HIT | 0.50 | 18.64% |
