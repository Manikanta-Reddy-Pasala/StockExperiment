# The New India Assurance Company Ltd. (NIACL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 160.70
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 2 / 6 / 3
- **Avg / median % per leg:** 4.87% / 2.21%
- **Sum % (uncompounded):** 53.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 2 | 6 | 3 | 4.87% | 53.6% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 6 | 3 | 4.87% | 53.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 6 | 54.5% | 2 | 6 | 3 | 4.87% | 53.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 00:00:00 | 118.95 | 109.75 | 117.61 | Stage2 pullback-breakout RSI=54 vol=2.9x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 00:00:00 | 124.95 | 109.88 | 118.16 | T1 booked 50% @ 124.95 |
| Target hit | 2023-08-16 00:00:00 | 123.30 | 112.07 | 123.65 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 00:00:00 | 131.20 | 112.63 | 124.60 | Stage2 pullback-breakout RSI=66 vol=8.5x ATR=4.14 |
| Stop hit — per-position SL triggered | 2023-09-05 00:00:00 | 134.10 | 114.35 | 128.63 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-09-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 00:00:00 | 143.20 | 115.81 | 131.36 | Stage2 pullback-breakout RSI=68 vol=6.9x ATR=5.49 |
| Stop hit — per-position SL triggered | 2023-10-03 00:00:00 | 139.40 | 118.14 | 136.87 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 148.50 | 123.95 | 140.90 | Stage2 pullback-breakout RSI=62 vol=2.6x ATR=5.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 00:00:00 | 159.99 | 124.93 | 145.26 | T1 booked 50% @ 159.99 |
| Target hit | 2023-12-20 00:00:00 | 213.85 | 143.07 | 215.64 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-01-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 00:00:00 | 227.95 | 150.02 | 216.36 | Stage2 pullback-breakout RSI=61 vol=1.8x ATR=10.45 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 221.20 | 156.69 | 218.68 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-01-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 00:00:00 | 242.90 | 157.55 | 220.99 | Stage2 pullback-breakout RSI=68 vol=4.4x ATR=10.80 |
| Stop hit — per-position SL triggered | 2024-01-23 00:00:00 | 226.70 | 159.05 | 223.20 | SL hit (bars_held=2) |

### Cycle 7 — BUY (started 2024-02-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 00:00:00 | 265.65 | 165.50 | 234.90 | Stage2 pullback-breakout RSI=70 vol=3.1x ATR=13.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 00:00:00 | 292.06 | 169.02 | 248.01 | T1 booked 50% @ 292.06 |
| Stop hit — per-position SL triggered | 2024-02-12 00:00:00 | 265.65 | 170.99 | 251.59 | SL hit (bars_held=5) |

### Cycle 8 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 244.70 | 199.17 | 230.43 | Stage2 pullback-breakout RSI=60 vol=3.4x ATR=10.97 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 228.25 | 200.87 | 234.23 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-21 00:00:00 | 118.95 | 2023-07-24 00:00:00 | 124.95 | PARTIAL | 0.50 | 5.05% |
| BUY | retest1 | 2023-07-21 00:00:00 | 118.95 | 2023-08-16 00:00:00 | 123.30 | TARGET_HIT | 0.50 | 3.66% |
| BUY | retest1 | 2023-08-22 00:00:00 | 131.20 | 2023-09-05 00:00:00 | 134.10 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest1 | 2023-09-15 00:00:00 | 143.20 | 2023-10-03 00:00:00 | 139.40 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest1 | 2023-11-16 00:00:00 | 148.50 | 2023-11-21 00:00:00 | 159.99 | PARTIAL | 0.50 | 7.74% |
| BUY | retest1 | 2023-11-16 00:00:00 | 148.50 | 2023-12-20 00:00:00 | 213.85 | TARGET_HIT | 0.50 | 44.01% |
| BUY | retest1 | 2024-01-04 00:00:00 | 227.95 | 2024-01-18 00:00:00 | 221.20 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest1 | 2024-01-19 00:00:00 | 242.90 | 2024-01-23 00:00:00 | 226.70 | STOP_HIT | 1.00 | -6.67% |
| BUY | retest1 | 2024-02-05 00:00:00 | 265.65 | 2024-02-08 00:00:00 | 292.06 | PARTIAL | 0.50 | 9.94% |
| BUY | retest1 | 2024-02-05 00:00:00 | 265.65 | 2024-02-12 00:00:00 | 265.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-26 00:00:00 | 244.70 | 2024-05-03 00:00:00 | 228.25 | STOP_HIT | 1.00 | -6.72% |
