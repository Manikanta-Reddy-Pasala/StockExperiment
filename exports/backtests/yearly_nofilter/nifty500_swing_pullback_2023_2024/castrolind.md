# Castrol India Ltd. (CASTROLIND)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 183.76
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
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 4
- **Target hits / Stop hits / Partials:** 0 / 7 / 4
- **Avg / median % per leg:** 2.60% / 2.35%
- **Sum % (uncompounded):** 28.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 7 | 63.6% | 0 | 7 | 4 | 2.60% | 28.6% |
| BUY @ 2nd Alert (retest1) | 11 | 7 | 63.6% | 0 | 7 | 4 | 2.60% | 28.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 7 | 63.6% | 0 | 7 | 4 | 2.60% | 28.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 00:00:00 | 149.00 | 126.25 | 144.58 | Stage2 pullback-breakout RSI=60 vol=2.8x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 00:00:00 | 157.90 | 126.54 | 145.59 | T1 booked 50% @ 157.90 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 149.00 | 127.63 | 148.38 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-10-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 00:00:00 | 143.20 | 129.90 | 140.36 | Stage2 pullback-breakout RSI=54 vol=1.5x ATR=4.23 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 136.86 | 130.90 | 141.44 | SL hit (bars_held=8) |

### Cycle 3 — BUY (started 2023-11-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 00:00:00 | 140.50 | 131.82 | 136.30 | Stage2 pullback-breakout RSI=60 vol=3.7x ATR=3.28 |
| Stop hit — per-position SL triggered | 2023-12-11 00:00:00 | 140.55 | 132.62 | 138.80 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-12-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 00:00:00 | 148.25 | 132.89 | 139.68 | Stage2 pullback-breakout RSI=69 vol=6.4x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 00:00:00 | 155.76 | 133.48 | 142.40 | T1 booked 50% @ 155.76 |
| Stop hit — per-position SL triggered | 2023-12-21 00:00:00 | 148.25 | 133.66 | 143.28 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 189.25 | 143.88 | 176.06 | Stage2 pullback-breakout RSI=64 vol=2.3x ATR=7.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 00:00:00 | 205.11 | 145.85 | 182.05 | T1 booked 50% @ 205.11 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 193.70 | 148.92 | 189.40 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-02-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 00:00:00 | 207.70 | 150.87 | 192.66 | Stage2 pullback-breakout RSI=65 vol=2.7x ATR=9.70 |
| Stop hit — per-position SL triggered | 2024-03-01 00:00:00 | 213.65 | 156.06 | 200.75 | Time-stop (10d <3%) |

### Cycle 7 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 201.40 | 163.56 | 196.97 | Stage2 pullback-breakout RSI=53 vol=1.6x ATR=8.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 00:00:00 | 218.20 | 166.36 | 203.80 | T1 booked 50% @ 218.20 |
| Stop hit — per-position SL triggered | 2024-04-19 00:00:00 | 201.40 | 169.11 | 207.84 | SL hit (bars_held=12) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-05 00:00:00 | 149.00 | 2023-09-06 00:00:00 | 157.90 | PARTIAL | 0.50 | 5.97% |
| BUY | retest1 | 2023-09-05 00:00:00 | 149.00 | 2023-09-12 00:00:00 | 149.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-12 00:00:00 | 143.20 | 2023-10-25 00:00:00 | 136.86 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest1 | 2023-11-24 00:00:00 | 140.50 | 2023-12-11 00:00:00 | 140.55 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest1 | 2023-12-14 00:00:00 | 148.25 | 2023-12-20 00:00:00 | 155.76 | PARTIAL | 0.50 | 5.07% |
| BUY | retest1 | 2023-12-14 00:00:00 | 148.25 | 2023-12-21 00:00:00 | 148.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-29 00:00:00 | 189.25 | 2024-02-02 00:00:00 | 205.11 | PARTIAL | 0.50 | 8.38% |
| BUY | retest1 | 2024-01-29 00:00:00 | 189.25 | 2024-02-12 00:00:00 | 193.70 | STOP_HIT | 0.50 | 2.35% |
| BUY | retest1 | 2024-02-16 00:00:00 | 207.70 | 2024-03-01 00:00:00 | 213.65 | STOP_HIT | 1.00 | 2.86% |
| BUY | retest1 | 2024-04-01 00:00:00 | 201.40 | 2024-04-09 00:00:00 | 218.20 | PARTIAL | 0.50 | 8.34% |
| BUY | retest1 | 2024-04-01 00:00:00 | 201.40 | 2024-04-19 00:00:00 | 201.40 | STOP_HIT | 0.50 | 0.00% |
