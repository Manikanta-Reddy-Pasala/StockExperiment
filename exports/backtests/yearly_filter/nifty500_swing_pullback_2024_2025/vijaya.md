# Vijaya Diagnostic Centre Ltd. (VIJAYA)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 1278.90
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 4
- **Avg / median % per leg:** 2.14% / 0.97%
- **Sum % (uncompounded):** 21.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 1 | 5 | 4 | 2.14% | 21.4% |
| BUY @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 1 | 5 | 4 | 2.14% | 21.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 5 | 50.0% | 1 | 5 | 4 | 2.14% | 21.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 00:00:00 | 816.85 | 670.63 | 784.02 | Stage2 pullback-breakout RSI=62 vol=10.4x ATR=30.58 |
| Stop hit — per-position SL triggered | 2024-07-18 00:00:00 | 770.98 | 677.30 | 783.61 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2024-08-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 00:00:00 | 809.25 | 690.44 | 787.42 | Stage2 pullback-breakout RSI=60 vol=2.0x ATR=28.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 00:00:00 | 866.40 | 696.72 | 800.51 | T1 booked 50% @ 866.40 |
| Stop hit — per-position SL triggered | 2024-08-16 00:00:00 | 809.25 | 699.20 | 804.36 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2024-09-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 00:00:00 | 959.35 | 755.31 | 908.71 | Stage2 pullback-breakout RSI=63 vol=7.5x ATR=37.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 00:00:00 | 1033.56 | 777.47 | 962.48 | T1 booked 50% @ 1033.56 |
| Target hit | 2024-10-21 00:00:00 | 968.70 | 786.29 | 974.78 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 00:00:00 | 1025.20 | 807.58 | 971.03 | Stage2 pullback-breakout RSI=61 vol=2.3x ATR=45.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 00:00:00 | 1115.66 | 810.69 | 985.16 | T1 booked 50% @ 1115.66 |
| Stop hit — per-position SL triggered | 2024-11-13 00:00:00 | 1025.20 | 817.84 | 1002.22 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 00:00:00 | 1147.90 | 827.58 | 1024.90 | Stage2 pullback-breakout RSI=66 vol=3.3x ATR=58.51 |
| Stop hit — per-position SL triggered | 2024-12-05 00:00:00 | 1147.10 | 859.25 | 1108.47 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-12-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 00:00:00 | 1096.40 | 888.46 | 1083.72 | Stage2 pullback-breakout RSI=53 vol=1.8x ATR=48.68 |
| Stop hit — per-position SL triggered | 2025-01-01 00:00:00 | 1023.38 | 895.23 | 1076.20 | SL hit (bars_held=4) |

### Cycle 7 — BUY (started 2025-01-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 00:00:00 | 1113.10 | 898.93 | 1077.50 | Stage2 pullback-breakout RSI=57 vol=4.4x ATR=51.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 00:00:00 | 1215.20 | 904.26 | 1094.28 | T1 booked 50% @ 1215.20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-09 00:00:00 | 816.85 | 2024-07-18 00:00:00 | 770.98 | STOP_HIT | 1.00 | -5.62% |
| BUY | retest1 | 2024-08-06 00:00:00 | 809.25 | 2024-08-13 00:00:00 | 866.40 | PARTIAL | 0.50 | 7.06% |
| BUY | retest1 | 2024-08-06 00:00:00 | 809.25 | 2024-08-16 00:00:00 | 809.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-30 00:00:00 | 959.35 | 2024-10-15 00:00:00 | 1033.56 | PARTIAL | 0.50 | 7.74% |
| BUY | retest1 | 2024-09-30 00:00:00 | 959.35 | 2024-10-21 00:00:00 | 968.70 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2024-11-07 00:00:00 | 1025.20 | 2024-11-08 00:00:00 | 1115.66 | PARTIAL | 0.50 | 8.82% |
| BUY | retest1 | 2024-11-07 00:00:00 | 1025.20 | 2024-11-13 00:00:00 | 1025.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-21 00:00:00 | 1147.90 | 2024-12-05 00:00:00 | 1147.10 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest1 | 2024-12-26 00:00:00 | 1096.40 | 2025-01-01 00:00:00 | 1023.38 | STOP_HIT | 1.00 | -6.66% |
| BUY | retest1 | 2025-01-03 00:00:00 | 1113.10 | 2025-01-07 00:00:00 | 1215.20 | PARTIAL | 0.50 | 9.17% |
