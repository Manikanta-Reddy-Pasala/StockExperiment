# Cochin Shipyard Ltd. (COCHINSHIP)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1724.80
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 0
- **Target hits / Stop hits / Partials:** 2 / 0 / 3
- **Avg / median % per leg:** 20.35% / 10.87%
- **Sum % (uncompounded):** 101.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 5 | 100.0% | 2 | 0 | 3 | 20.35% | 101.8% |
| BUY @ 2nd Alert (retest1) | 5 | 5 | 100.0% | 2 | 0 | 3 | 20.35% | 101.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 5 | 100.0% | 2 | 0 | 3 | 20.35% | 101.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 00:00:00 | 343.95 | 270.30 | 328.94 | Stage2 pullback-breakout RSI=64 vol=3.4x ATR=12.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 00:00:00 | 368.42 | 271.63 | 336.08 | T1 booked 50% @ 368.42 |
| Target hit | 2023-09-22 00:00:00 | 501.08 | 321.36 | 501.93 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 502.15 | 368.12 | 494.12 | Stage2 pullback-breakout RSI=52 vol=2.0x ATR=23.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 00:00:00 | 548.73 | 378.91 | 511.78 | T1 booked 50% @ 548.73 |
| Target hit | 2024-01-08 00:00:00 | 646.83 | 453.65 | 652.53 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-03-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 00:00:00 | 890.45 | 604.27 | 844.17 | Stage2 pullback-breakout RSI=57 vol=4.7x ATR=48.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 00:00:00 | 987.24 | 632.71 | 886.04 | T1 booked 50% @ 987.24 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-14 00:00:00 | 343.95 | 2023-08-16 00:00:00 | 368.42 | PARTIAL | 0.50 | 7.11% |
| BUY | retest1 | 2023-08-14 00:00:00 | 343.95 | 2023-09-22 00:00:00 | 501.08 | TARGET_HIT | 0.50 | 45.68% |
| BUY | retest1 | 2023-11-06 00:00:00 | 502.15 | 2023-11-15 00:00:00 | 548.73 | PARTIAL | 0.50 | 9.28% |
| BUY | retest1 | 2023-11-06 00:00:00 | 502.15 | 2024-01-08 00:00:00 | 646.83 | TARGET_HIT | 0.50 | 28.81% |
| BUY | retest1 | 2024-03-15 00:00:00 | 890.45 | 2024-04-02 00:00:00 | 987.24 | PARTIAL | 0.50 | 10.87% |
