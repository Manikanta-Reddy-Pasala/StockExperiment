# HEROMOTOCO (HEROMOTOCO)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 5225.50
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
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 7.88% / 4.14%
- **Sum % (uncompounded):** 55.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 1 | 3 | 3 | 7.88% | 55.1% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 1 | 3 | 3 | 7.88% | 55.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 1 | 3 | 3 | 7.88% | 55.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 00:00:00 | 3029.80 | 2685.39 | 2857.92 | Stage2 pullback-breakout RSI=69 vol=2.7x ATR=68.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 00:00:00 | 3166.58 | 2690.10 | 2886.52 | T1 booked 50% @ 3166.58 |
| Stop hit — per-position SL triggered | 2023-07-18 00:00:00 | 3116.40 | 2726.71 | 3020.44 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 00:00:00 | 3114.65 | 2825.94 | 3001.69 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=61.97 |
| Stop hit — per-position SL triggered | 2023-09-21 00:00:00 | 3021.69 | 2830.01 | 3006.80 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 00:00:00 | 3100.85 | 2852.24 | 3012.60 | Stage2 pullback-breakout RSI=58 vol=2.1x ATR=68.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 00:00:00 | 3238.27 | 2870.01 | 3081.65 | T1 booked 50% @ 3238.27 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 3100.85 | 2878.79 | 3103.27 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2023-11-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 00:00:00 | 3280.05 | 2915.93 | 3135.77 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=67.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 00:00:00 | 3415.96 | 2933.67 | 3213.42 | T1 booked 50% @ 3415.96 |
| Target hit | 2024-02-20 00:00:00 | 4663.35 | 3550.42 | 4675.96 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-05-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 00:00:00 | 4764.90 | 3944.46 | 4518.46 | Stage2 pullback-breakout RSI=65 vol=4.3x ATR=130.88 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-04 00:00:00 | 3029.80 | 2023-07-05 00:00:00 | 3166.58 | PARTIAL | 0.50 | 4.51% |
| BUY | retest1 | 2023-07-04 00:00:00 | 3029.80 | 2023-07-18 00:00:00 | 3116.40 | STOP_HIT | 0.50 | 2.86% |
| BUY | retest1 | 2023-09-18 00:00:00 | 3114.65 | 2023-09-21 00:00:00 | 3021.69 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest1 | 2023-10-11 00:00:00 | 3100.85 | 2023-10-19 00:00:00 | 3238.27 | PARTIAL | 0.50 | 4.43% |
| BUY | retest1 | 2023-10-11 00:00:00 | 3100.85 | 2023-10-25 00:00:00 | 3100.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-16 00:00:00 | 3280.05 | 2023-11-22 00:00:00 | 3415.96 | PARTIAL | 0.50 | 4.14% |
| BUY | retest1 | 2023-11-16 00:00:00 | 3280.05 | 2024-02-20 00:00:00 | 4663.35 | TARGET_HIT | 0.50 | 42.17% |
