# Hindustan Copper Ltd. (HINDCOPPER)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 567.30
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
| TARGET_HIT | 3 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 3 / 2 / 3
- **Avg / median % per leg:** 14.93% / 10.61%
- **Sum % (uncompounded):** 119.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 14.93% | 119.4% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 14.93% | 119.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 3 | 2 | 3 | 14.93% | 119.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 00:00:00 | 122.00 | 110.10 | 115.44 | Stage2 pullback-breakout RSI=69 vol=2.8x ATR=3.15 |
| Stop hit — per-position SL triggered | 2023-07-13 00:00:00 | 117.27 | 110.51 | 116.67 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-07-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 00:00:00 | 122.60 | 110.73 | 117.53 | Stage2 pullback-breakout RSI=64 vol=1.5x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 00:00:00 | 129.43 | 112.05 | 122.97 | T1 booked 50% @ 129.43 |
| Target hit | 2023-08-18 00:00:00 | 140.75 | 116.54 | 141.20 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 162.05 | 132.79 | 148.96 | Stage2 pullback-breakout RSI=65 vol=6.4x ATR=5.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 00:00:00 | 173.34 | 135.57 | 157.92 | T1 booked 50% @ 173.34 |
| Target hit | 2024-02-09 00:00:00 | 270.15 | 180.51 | 278.26 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 00:00:00 | 269.80 | 190.38 | 262.53 | Stage2 pullback-breakout RSI=54 vol=2.4x ATR=14.18 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 248.52 | 197.34 | 266.02 | SL hit (bars_held=9) |

### Cycle 5 — BUY (started 2024-03-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 00:00:00 | 282.55 | 202.79 | 267.65 | Stage2 pullback-breakout RSI=57 vol=1.7x ATR=14.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 00:00:00 | 312.52 | 205.40 | 273.92 | T1 booked 50% @ 312.52 |
| Target hit | 2024-05-09 00:00:00 | 357.55 | 240.12 | 362.98 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 00:00:00 | 122.00 | 2023-07-13 00:00:00 | 117.27 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest1 | 2023-07-17 00:00:00 | 122.60 | 2023-07-31 00:00:00 | 129.43 | PARTIAL | 0.50 | 5.57% |
| BUY | retest1 | 2023-07-17 00:00:00 | 122.60 | 2023-08-18 00:00:00 | 140.75 | TARGET_HIT | 0.50 | 14.80% |
| BUY | retest1 | 2023-11-13 00:00:00 | 162.05 | 2023-11-29 00:00:00 | 173.34 | PARTIAL | 0.50 | 6.97% |
| BUY | retest1 | 2023-11-13 00:00:00 | 162.05 | 2024-02-09 00:00:00 | 270.15 | TARGET_HIT | 0.50 | 66.71% |
| BUY | retest1 | 2024-02-29 00:00:00 | 269.80 | 2024-03-13 00:00:00 | 248.52 | STOP_HIT | 1.00 | -7.89% |
| BUY | retest1 | 2024-03-26 00:00:00 | 282.55 | 2024-04-01 00:00:00 | 312.52 | PARTIAL | 0.50 | 10.61% |
| BUY | retest1 | 2024-03-26 00:00:00 | 282.55 | 2024-05-09 00:00:00 | 357.55 | TARGET_HIT | 0.50 | 26.54% |
