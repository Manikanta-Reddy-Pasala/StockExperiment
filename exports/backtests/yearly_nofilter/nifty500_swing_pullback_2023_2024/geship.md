# Great Eastern Shipping Co. Ltd. (GESHIP)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 1588.30
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 1 / 6 / 2
- **Avg / median % per leg:** 2.38% / 0.63%
- **Sum % (uncompounded):** 21.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 1 | 6 | 2 | 2.38% | 21.4% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 1 | 6 | 2 | 2.38% | 21.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 1 | 6 | 2 | 2.38% | 21.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 00:00:00 | 771.15 | 673.96 | 765.55 | Stage2 pullback-breakout RSI=52 vol=2.8x ATR=29.14 |
| Stop hit — per-position SL triggered | 2023-08-31 00:00:00 | 764.50 | 682.77 | 765.15 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 00:00:00 | 815.50 | 687.44 | 771.17 | Stage2 pullback-breakout RSI=67 vol=9.5x ATR=25.18 |
| Stop hit — per-position SL triggered | 2023-09-22 00:00:00 | 811.15 | 699.76 | 799.27 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-09-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 00:00:00 | 851.10 | 702.48 | 806.23 | Stage2 pullback-breakout RSI=64 vol=2.3x ATR=29.64 |
| Stop hit — per-position SL triggered | 2023-10-11 00:00:00 | 856.50 | 716.13 | 831.47 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-11-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 00:00:00 | 819.00 | 739.29 | 794.06 | Stage2 pullback-breakout RSI=57 vol=3.4x ATR=24.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 00:00:00 | 867.44 | 742.97 | 807.44 | T1 booked 50% @ 867.44 |
| Target hit | 2024-01-15 00:00:00 | 953.55 | 796.89 | 956.50 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-01-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 00:00:00 | 1007.70 | 810.24 | 965.32 | Stage2 pullback-breakout RSI=64 vol=2.7x ATR=31.08 |
| Stop hit — per-position SL triggered | 2024-02-01 00:00:00 | 961.08 | 817.68 | 976.27 | SL hit (bars_held=4) |

### Cycle 6 — BUY (started 2024-02-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 00:00:00 | 1005.90 | 842.81 | 957.83 | Stage2 pullback-breakout RSI=62 vol=3.9x ATR=34.87 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 953.60 | 854.22 | 974.27 | SL hit (bars_held=8) |

### Cycle 7 — BUY (started 2024-04-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 00:00:00 | 1022.50 | 882.36 | 991.43 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=40.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 00:00:00 | 1103.30 | 890.21 | 1013.20 | T1 booked 50% @ 1103.30 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 1040.75 | 899.16 | 1036.16 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-17 00:00:00 | 771.15 | 2023-08-31 00:00:00 | 764.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest1 | 2023-09-07 00:00:00 | 815.50 | 2023-09-22 00:00:00 | 811.15 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2023-09-26 00:00:00 | 851.10 | 2023-10-11 00:00:00 | 856.50 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest1 | 2023-11-23 00:00:00 | 819.00 | 2023-11-30 00:00:00 | 867.44 | PARTIAL | 0.50 | 5.91% |
| BUY | retest1 | 2023-11-23 00:00:00 | 819.00 | 2024-01-15 00:00:00 | 953.55 | TARGET_HIT | 0.50 | 16.43% |
| BUY | retest1 | 2024-01-25 00:00:00 | 1007.70 | 2024-02-01 00:00:00 | 961.08 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest1 | 2024-02-29 00:00:00 | 1005.90 | 2024-03-12 00:00:00 | 953.60 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest1 | 2024-04-18 00:00:00 | 1022.50 | 2024-04-25 00:00:00 | 1103.30 | PARTIAL | 0.50 | 7.90% |
| BUY | retest1 | 2024-04-18 00:00:00 | 1022.50 | 2024-05-03 00:00:00 | 1040.75 | STOP_HIT | 0.50 | 1.78% |
