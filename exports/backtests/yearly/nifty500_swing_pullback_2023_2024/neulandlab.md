# Neuland Laboratories Ltd. (NEULANDLAB)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 17666.00
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
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 2
- **Target hits / Stop hits / Partials:** 4 / 2 / 4
- **Avg / median % per leg:** 9.23% / 8.59%
- **Sum % (uncompounded):** 92.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 4 | 2 | 4 | 9.23% | 92.3% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 4 | 2 | 4 | 9.23% | 92.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 8 | 80.0% | 4 | 2 | 4 | 9.23% | 92.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 05:30:00 | 2965.55 | 2013.16 | 2856.60 | Stage2 pullback-breakout RSI=60 vol=2.3x ATR=129.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 05:30:00 | 3224.58 | 2108.17 | 2953.07 | T1 booked 50% @ 3224.58 |
| Target hit | 2023-08-30 05:30:00 | 3812.15 | 2527.83 | 3814.06 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-17 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 05:30:00 | 4015.90 | 2865.61 | 3827.64 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=150.95 |
| Stop hit — per-position SL triggered | 2023-10-23 05:30:00 | 3789.47 | 2910.86 | 3888.05 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-11-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 05:30:00 | 4038.45 | 2991.92 | 3876.94 | Stage2 pullback-breakout RSI=60 vol=1.5x ATR=156.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 05:30:00 | 4352.19 | 3008.06 | 3947.10 | T1 booked 50% @ 4352.19 |
| Target hit | 2023-12-11 05:30:00 | 5054.70 | 3464.18 | 5075.12 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 05:30:00 | 5592.30 | 3726.98 | 5247.39 | Stage2 pullback-breakout RSI=68 vol=1.5x ATR=195.15 |
| Stop hit — per-position SL triggered | 2024-01-08 05:30:00 | 5299.57 | 3781.20 | 5327.26 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-01-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 05:30:00 | 6095.80 | 4012.63 | 5613.76 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=261.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-29 05:30:00 | 6619.31 | 4037.77 | 5701.88 | T1 booked 50% @ 6619.31 |
| Target hit | 2024-03-06 05:30:00 | 6610.60 | 4716.17 | 6886.66 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-04-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 05:30:00 | 6665.40 | 5024.70 | 6351.71 | Stage2 pullback-breakout RSI=57 vol=2.5x ATR=291.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 05:30:00 | 7248.10 | 5070.10 | 6528.66 | T1 booked 50% @ 7248.10 |
| Target hit | 2024-05-09 05:30:00 | 7146.85 | 5393.75 | 7193.88 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-04 05:30:00 | 2965.55 | 2023-07-18 05:30:00 | 3224.58 | PARTIAL | 0.50 | 8.73% |
| BUY | retest1 | 2023-07-04 05:30:00 | 2965.55 | 2023-08-30 05:30:00 | 3812.15 | TARGET_HIT | 0.50 | 28.55% |
| BUY | retest1 | 2023-10-17 05:30:00 | 4015.90 | 2023-10-23 05:30:00 | 3789.47 | STOP_HIT | 1.00 | -5.64% |
| BUY | retest1 | 2023-11-06 05:30:00 | 4038.45 | 2023-11-07 05:30:00 | 4352.19 | PARTIAL | 0.50 | 7.77% |
| BUY | retest1 | 2023-11-06 05:30:00 | 4038.45 | 2023-12-11 05:30:00 | 5054.70 | TARGET_HIT | 0.50 | 25.16% |
| BUY | retest1 | 2024-01-03 05:30:00 | 5592.30 | 2024-01-08 05:30:00 | 5299.57 | STOP_HIT | 1.00 | -5.23% |
| BUY | retest1 | 2024-01-25 05:30:00 | 6095.80 | 2024-01-29 05:30:00 | 6619.31 | PARTIAL | 0.50 | 8.59% |
| BUY | retest1 | 2024-01-25 05:30:00 | 6095.80 | 2024-03-06 05:30:00 | 6610.60 | TARGET_HIT | 0.50 | 8.45% |
| BUY | retest1 | 2024-04-12 05:30:00 | 6665.40 | 2024-04-16 05:30:00 | 7248.10 | PARTIAL | 0.50 | 8.74% |
| BUY | retest1 | 2024-04-12 05:30:00 | 6665.40 | 2024-05-09 05:30:00 | 7146.85 | TARGET_HIT | 0.50 | 7.22% |
