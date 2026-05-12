# Gabriel India Ltd. (GABRIEL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1127.90
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 6.20% / 5.57%
- **Sum % (uncompounded):** 43.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 6.20% | 43.4% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 6.20% | 43.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 2 | 3 | 2 | 6.20% | 43.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 00:00:00 | 208.50 | 172.36 | 198.59 | Stage2 pullback-breakout RSI=70 vol=2.4x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 00:00:00 | 220.10 | 174.55 | 206.05 | T1 booked 50% @ 220.10 |
| Target hit | 2023-09-12 00:00:00 | 292.05 | 200.72 | 292.41 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 00:00:00 | 365.55 | 243.38 | 337.55 | Stage2 pullback-breakout RSI=67 vol=2.8x ATR=14.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 00:00:00 | 394.63 | 248.61 | 351.05 | T1 booked 50% @ 394.63 |
| Target hit | 2023-12-20 00:00:00 | 394.70 | 283.57 | 406.59 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-02-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 00:00:00 | 399.85 | 312.99 | 384.92 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=14.34 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 378.33 | 314.35 | 384.26 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 359.85 | 321.44 | 335.52 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=17.12 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 334.18 | 324.82 | 349.87 | SL hit (bars_held=9) |

### Cycle 5 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 379.05 | 327.25 | 352.69 | Stage2 pullback-breakout RSI=65 vol=4.9x ATR=14.35 |
| Stop hit — per-position SL triggered | 2024-05-08 00:00:00 | 357.53 | 330.20 | 363.41 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-20 00:00:00 | 208.50 | 2023-07-27 00:00:00 | 220.10 | PARTIAL | 0.50 | 5.57% |
| BUY | retest1 | 2023-07-20 00:00:00 | 208.50 | 2023-09-12 00:00:00 | 292.05 | TARGET_HIT | 0.50 | 40.07% |
| BUY | retest1 | 2023-11-09 00:00:00 | 365.55 | 2023-11-15 00:00:00 | 394.63 | PARTIAL | 0.50 | 7.95% |
| BUY | retest1 | 2023-11-09 00:00:00 | 365.55 | 2023-12-20 00:00:00 | 394.70 | TARGET_HIT | 0.50 | 7.97% |
| BUY | retest1 | 2024-02-07 00:00:00 | 399.85 | 2024-02-09 00:00:00 | 378.33 | STOP_HIT | 1.00 | -5.38% |
| BUY | retest1 | 2024-04-01 00:00:00 | 359.85 | 2024-04-15 00:00:00 | 334.18 | STOP_HIT | 1.00 | -7.13% |
| BUY | retest1 | 2024-04-29 00:00:00 | 379.05 | 2024-05-08 00:00:00 | 357.53 | STOP_HIT | 1.00 | -5.68% |
