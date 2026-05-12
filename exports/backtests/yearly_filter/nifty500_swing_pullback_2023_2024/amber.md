# Amber Enterprises India Ltd. (AMBER)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 8614.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 3.58% / 6.17%
- **Sum % (uncompounded):** 25.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 3.58% | 25.0% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 3.58% | 25.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 2 | 3 | 2 | 3.58% | 25.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 00:00:00 | 2504.25 | 2100.90 | 2335.41 | Stage2 pullback-breakout RSI=70 vol=4.1x ATR=92.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 00:00:00 | 2689.00 | 2137.39 | 2476.95 | T1 booked 50% @ 2689.00 |
| Target hit | 2023-09-13 00:00:00 | 2844.55 | 2273.10 | 2872.65 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 3348.90 | 2476.71 | 2990.37 | Stage2 pullback-breakout RSI=69 vol=2.7x ATR=130.85 |
| Stop hit — per-position SL triggered | 2023-11-09 00:00:00 | 3152.62 | 2500.62 | 3064.87 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2023-11-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 00:00:00 | 3385.30 | 2547.58 | 3138.95 | Stage2 pullback-breakout RSI=62 vol=2.0x ATR=147.39 |
| Stop hit — per-position SL triggered | 2023-11-28 00:00:00 | 3164.22 | 2581.52 | 3177.73 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2023-12-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 00:00:00 | 3321.30 | 2646.71 | 3141.69 | Stage2 pullback-breakout RSI=61 vol=6.0x ATR=124.12 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 3135.12 | 2662.17 | 3147.65 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-01-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 00:00:00 | 3189.80 | 2704.67 | 3128.37 | Stage2 pullback-breakout RSI=55 vol=2.0x ATR=98.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 00:00:00 | 3386.74 | 2716.18 | 3157.12 | T1 booked 50% @ 3386.74 |
| Target hit | 2024-02-12 00:00:00 | 3696.60 | 2993.45 | 4050.92 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-05-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 00:00:00 | 3969.00 | 3287.16 | 3756.15 | Stage2 pullback-breakout RSI=67 vol=1.5x ATR=128.61 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-04 00:00:00 | 2504.25 | 2023-08-17 00:00:00 | 2689.00 | PARTIAL | 0.50 | 7.38% |
| BUY | retest1 | 2023-08-04 00:00:00 | 2504.25 | 2023-09-13 00:00:00 | 2844.55 | TARGET_HIT | 0.50 | 13.59% |
| BUY | retest1 | 2023-11-06 00:00:00 | 3348.90 | 2023-11-09 00:00:00 | 3152.62 | STOP_HIT | 1.00 | -5.86% |
| BUY | retest1 | 2023-11-20 00:00:00 | 3385.30 | 2023-11-28 00:00:00 | 3164.22 | STOP_HIT | 1.00 | -6.53% |
| BUY | retest1 | 2023-12-15 00:00:00 | 3321.30 | 2023-12-20 00:00:00 | 3135.12 | STOP_HIT | 1.00 | -5.61% |
| BUY | retest1 | 2024-01-04 00:00:00 | 3189.80 | 2024-01-08 00:00:00 | 3386.74 | PARTIAL | 0.50 | 6.17% |
| BUY | retest1 | 2024-01-04 00:00:00 | 3189.80 | 2024-02-12 00:00:00 | 3696.60 | TARGET_HIT | 0.50 | 15.89% |
