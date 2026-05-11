# Reliance Power Ltd. (RPOWER)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 28.04
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 3
- **Avg / median % per leg:** 5.82% / 8.06%
- **Sum % (uncompounded):** 40.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 2 | 2 | 3 | 5.82% | 40.8% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 2 | 3 | 5.82% | 40.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 2 | 2 | 3 | 5.82% | 40.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 00:00:00 | 29.69 | 25.57 | 28.16 | Stage2 pullback-breakout RSI=60 vol=2.3x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 00:00:00 | 32.08 | 25.79 | 29.23 | T1 booked 50% @ 32.08 |
| Target hit | 2024-08-13 00:00:00 | 30.34 | 26.33 | 30.78 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-09-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 00:00:00 | 34.61 | 27.53 | 31.34 | Stage2 pullback-breakout RSI=66 vol=2.2x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 00:00:00 | 37.45 | 27.72 | 32.42 | T1 booked 50% @ 37.45 |
| Target hit | 2024-10-17 00:00:00 | 41.74 | 30.60 | 43.24 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 00:00:00 | 43.77 | 32.17 | 42.46 | Stage2 pullback-breakout RSI=55 vol=3.0x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-11-11 00:00:00 | 40.45 | 32.34 | 42.10 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-12-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 00:00:00 | 43.14 | 33.11 | 39.17 | Stage2 pullback-breakout RSI=61 vol=1.6x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 00:00:00 | 47.15 | 33.82 | 41.99 | T1 booked 50% @ 47.15 |
| Stop hit — per-position SL triggered | 2024-12-19 00:00:00 | 43.14 | 34.32 | 43.44 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-25 00:00:00 | 29.69 | 2024-07-31 00:00:00 | 32.08 | PARTIAL | 0.50 | 8.06% |
| BUY | retest1 | 2024-07-25 00:00:00 | 29.69 | 2024-08-13 00:00:00 | 30.34 | TARGET_HIT | 0.50 | 2.19% |
| BUY | retest1 | 2024-09-19 00:00:00 | 34.61 | 2024-09-23 00:00:00 | 37.45 | PARTIAL | 0.50 | 8.20% |
| BUY | retest1 | 2024-09-19 00:00:00 | 34.61 | 2024-10-17 00:00:00 | 41.74 | TARGET_HIT | 0.50 | 20.60% |
| BUY | retest1 | 2024-11-07 00:00:00 | 43.77 | 2024-11-11 00:00:00 | 40.45 | STOP_HIT | 1.00 | -7.58% |
| BUY | retest1 | 2024-12-05 00:00:00 | 43.14 | 2024-12-13 00:00:00 | 47.15 | PARTIAL | 0.50 | 9.30% |
| BUY | retest1 | 2024-12-05 00:00:00 | 43.14 | 2024-12-19 00:00:00 | 43.14 | STOP_HIT | 0.50 | 0.00% |
