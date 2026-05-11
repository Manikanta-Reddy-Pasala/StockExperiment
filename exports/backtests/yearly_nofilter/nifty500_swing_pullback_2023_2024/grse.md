# Garden Reach Shipbuilders & Engineers Ltd. (GRSE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 2944.00
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 2 / 4 / 3
- **Avg / median % per leg:** 5.02% / 6.81%
- **Sum % (uncompounded):** 45.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 2 | 4 | 3 | 5.02% | 45.1% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 2 | 4 | 3 | 5.02% | 45.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 6 | 66.7% | 2 | 4 | 3 | 5.02% | 45.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 00:00:00 | 616.30 | 467.47 | 568.02 | Stage2 pullback-breakout RSI=70 vol=4.8x ATR=24.97 |
| Stop hit — per-position SL triggered | 2023-07-14 00:00:00 | 578.85 | 471.36 | 575.89 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-08-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 00:00:00 | 657.50 | 496.87 | 602.89 | Stage2 pullback-breakout RSI=68 vol=7.6x ATR=25.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 00:00:00 | 708.47 | 499.65 | 619.41 | T1 booked 50% @ 708.47 |
| Target hit | 2023-09-13 00:00:00 | 798.95 | 556.53 | 799.11 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 00:00:00 | 809.95 | 640.99 | 776.83 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=27.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 00:00:00 | 865.13 | 659.53 | 815.80 | T1 booked 50% @ 865.13 |
| Stop hit — per-position SL triggered | 2023-12-08 00:00:00 | 832.40 | 663.11 | 820.15 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-12-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 00:00:00 | 849.60 | 674.80 | 828.34 | Stage2 pullback-breakout RSI=60 vol=2.8x ATR=25.34 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 811.59 | 676.11 | 826.27 | SL hit (bars_held=1) |

### Cycle 5 — BUY (started 2024-01-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 00:00:00 | 905.50 | 700.59 | 857.60 | Stage2 pullback-breakout RSI=68 vol=3.4x ATR=29.37 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 861.45 | 711.40 | 870.61 | SL hit (bars_held=6) |

### Cycle 6 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 807.40 | 752.52 | 783.02 | Stage2 pullback-breakout RSI=54 vol=3.1x ATR=34.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 00:00:00 | 876.70 | 754.53 | 796.14 | T1 booked 50% @ 876.70 |
| Target hit | 2024-05-07 00:00:00 | 913.40 | 788.14 | 930.87 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-11 00:00:00 | 616.30 | 2023-07-14 00:00:00 | 578.85 | STOP_HIT | 1.00 | -6.08% |
| BUY | retest1 | 2023-08-14 00:00:00 | 657.50 | 2023-08-16 00:00:00 | 708.47 | PARTIAL | 0.50 | 7.75% |
| BUY | retest1 | 2023-08-14 00:00:00 | 657.50 | 2023-09-13 00:00:00 | 798.95 | TARGET_HIT | 0.50 | 21.51% |
| BUY | retest1 | 2023-11-21 00:00:00 | 809.95 | 2023-12-06 00:00:00 | 865.13 | PARTIAL | 0.50 | 6.81% |
| BUY | retest1 | 2023-11-21 00:00:00 | 809.95 | 2023-12-08 00:00:00 | 832.40 | STOP_HIT | 0.50 | 2.77% |
| BUY | retest1 | 2023-12-19 00:00:00 | 849.60 | 2023-12-20 00:00:00 | 811.59 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest1 | 2024-01-10 00:00:00 | 905.50 | 2024-01-18 00:00:00 | 861.45 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest1 | 2024-04-01 00:00:00 | 807.40 | 2024-04-03 00:00:00 | 876.70 | PARTIAL | 0.50 | 8.58% |
| BUY | retest1 | 2024-04-01 00:00:00 | 807.40 | 2024-05-07 00:00:00 | 913.40 | TARGET_HIT | 0.50 | 13.13% |
