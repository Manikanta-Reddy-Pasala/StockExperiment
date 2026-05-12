# Ceat Ltd. (CEATLTD)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3266.00
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
- **Avg / median % per leg:** 3.41% / 4.66%
- **Sum % (uncompounded):** 17.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.41% | 17.0% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.41% | 17.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 3.41% | 17.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 00:00:00 | 2194.75 | 1972.48 | 2142.26 | Stage2 pullback-breakout RSI=55 vol=11.3x ATR=70.33 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 2089.25 | 1980.07 | 2148.91 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 2171.90 | 2011.46 | 2115.39 | Stage2 pullback-breakout RSI=61 vol=4.8x ATR=50.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 00:00:00 | 2273.07 | 2020.36 | 2156.93 | T1 booked 50% @ 2273.07 |
| Target hit | 2024-01-17 00:00:00 | 2423.00 | 2117.16 | 2426.24 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-01-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 00:00:00 | 2594.75 | 2125.89 | 2450.49 | Stage2 pullback-breakout RSI=69 vol=3.8x ATR=72.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 00:00:00 | 2740.61 | 2133.73 | 2494.63 | T1 booked 50% @ 2740.61 |
| Stop hit — per-position SL triggered | 2024-02-02 00:00:00 | 2594.75 | 2178.21 | 2605.39 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-17 00:00:00 | 2194.75 | 2023-10-23 00:00:00 | 2089.25 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest1 | 2023-12-01 00:00:00 | 2171.90 | 2023-12-07 00:00:00 | 2273.07 | PARTIAL | 0.50 | 4.66% |
| BUY | retest1 | 2023-12-01 00:00:00 | 2171.90 | 2024-01-17 00:00:00 | 2423.00 | TARGET_HIT | 0.50 | 11.56% |
| BUY | retest1 | 2024-01-19 00:00:00 | 2594.75 | 2024-01-20 00:00:00 | 2740.61 | PARTIAL | 0.50 | 5.62% |
| BUY | retest1 | 2024-01-19 00:00:00 | 2594.75 | 2024-02-02 00:00:00 | 2594.75 | STOP_HIT | 0.50 | 0.00% |
