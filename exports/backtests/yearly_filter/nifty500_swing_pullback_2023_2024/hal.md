# Hindustan Aeronautics Ltd. (HAL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 4762.50
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 9.60% / 4.44%
- **Sum % (uncompounded):** 48.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 9.60% | 48.0% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 9.60% | 48.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 9.60% | 48.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 00:00:00 | 1939.43 | 1552.36 | 1905.29 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=44.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 00:00:00 | 2028.41 | 1579.11 | 1928.99 | T1 booked 50% @ 2028.41 |
| Stop hit — per-position SL triggered | 2023-08-29 00:00:00 | 1939.43 | 1590.19 | 1935.38 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2023-11-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 00:00:00 | 2017.15 | 1719.39 | 1904.47 | Stage2 pullback-breakout RSI=65 vol=2.9x ATR=44.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-10 00:00:00 | 2106.70 | 1729.01 | 1940.99 | T1 booked 50% @ 2106.70 |
| Target hit | 2024-01-23 00:00:00 | 2890.50 | 2111.77 | 2916.91 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-02-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 00:00:00 | 3097.20 | 2201.81 | 2960.74 | Stage2 pullback-breakout RSI=66 vol=3.0x ATR=89.13 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 2963.51 | 2209.41 | 2961.18 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-14 00:00:00 | 1939.43 | 2023-08-24 00:00:00 | 2028.41 | PARTIAL | 0.50 | 4.59% |
| BUY | retest1 | 2023-08-14 00:00:00 | 1939.43 | 2023-08-29 00:00:00 | 1939.43 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-07 00:00:00 | 2017.15 | 2023-11-10 00:00:00 | 2106.70 | PARTIAL | 0.50 | 4.44% |
| BUY | retest1 | 2023-11-07 00:00:00 | 2017.15 | 2024-01-23 00:00:00 | 2890.50 | TARGET_HIT | 0.50 | 43.30% |
| BUY | retest1 | 2024-02-08 00:00:00 | 3097.20 | 2024-02-09 00:00:00 | 2963.51 | STOP_HIT | 1.00 | -4.32% |
