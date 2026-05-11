# Ipca Laboratories Ltd. (IPCALAB)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1551.60
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
- **Avg / median % per leg:** 0.82% / 1.20%
- **Sum % (uncompounded):** 7.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 1 | 6 | 2 | 0.82% | 7.4% |
| BUY @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 1 | 6 | 2 | 0.82% | 7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 1 | 6 | 2 | 0.82% | 7.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 00:00:00 | 930.75 | 825.00 | 892.20 | Stage2 pullback-breakout RSI=68 vol=3.9x ATR=22.36 |
| Stop hit — per-position SL triggered | 2023-09-21 00:00:00 | 897.22 | 827.62 | 897.52 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 935.65 | 832.57 | 905.39 | Stage2 pullback-breakout RSI=65 vol=1.8x ATR=23.10 |
| Stop hit — per-position SL triggered | 2023-10-19 00:00:00 | 946.90 | 846.47 | 938.22 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-10-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-23 00:00:00 | 998.80 | 848.87 | 943.79 | Stage2 pullback-breakout RSI=68 vol=8.3x ATR=25.05 |
| Stop hit — per-position SL triggered | 2023-10-26 00:00:00 | 961.22 | 851.60 | 951.50 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-01-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 00:00:00 | 1144.35 | 946.92 | 1104.12 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=28.59 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 1101.47 | 955.49 | 1110.49 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2024-01-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 00:00:00 | 1132.60 | 963.98 | 1107.38 | Stage2 pullback-breakout RSI=57 vol=2.1x ATR=31.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 00:00:00 | 1195.06 | 973.68 | 1119.66 | T1 booked 50% @ 1195.06 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 1160.65 | 984.36 | 1147.20 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-02-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 00:00:00 | 1241.45 | 990.24 | 1157.20 | Stage2 pullback-breakout RSI=67 vol=3.1x ATR=43.78 |
| Stop hit — per-position SL triggered | 2024-02-28 00:00:00 | 1175.79 | 1007.66 | 1188.29 | SL hit (bars_held=8) |

### Cycle 7 — BUY (started 2024-03-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 00:00:00 | 1196.05 | 1034.45 | 1177.63 | Stage2 pullback-breakout RSI=55 vol=2.5x ATR=38.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 00:00:00 | 1273.60 | 1047.78 | 1206.50 | T1 booked 50% @ 1273.60 |
| Target hit | 2024-05-07 00:00:00 | 1293.30 | 1099.15 | 1314.54 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-15 00:00:00 | 930.75 | 2023-09-21 00:00:00 | 897.22 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest1 | 2023-09-29 00:00:00 | 935.65 | 2023-10-19 00:00:00 | 946.90 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest1 | 2023-10-23 00:00:00 | 998.80 | 2023-10-26 00:00:00 | 961.22 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest1 | 2024-01-11 00:00:00 | 1144.35 | 2024-01-18 00:00:00 | 1101.47 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest1 | 2024-01-29 00:00:00 | 1132.60 | 2024-02-06 00:00:00 | 1195.06 | PARTIAL | 0.50 | 5.51% |
| BUY | retest1 | 2024-01-29 00:00:00 | 1132.60 | 2024-02-13 00:00:00 | 1160.65 | STOP_HIT | 0.50 | 2.48% |
| BUY | retest1 | 2024-02-16 00:00:00 | 1241.45 | 2024-02-28 00:00:00 | 1175.79 | STOP_HIT | 1.00 | -5.29% |
| BUY | retest1 | 2024-03-22 00:00:00 | 1196.05 | 2024-04-04 00:00:00 | 1273.60 | PARTIAL | 0.50 | 6.48% |
| BUY | retest1 | 2024-03-22 00:00:00 | 1196.05 | 2024-05-07 00:00:00 | 1293.30 | TARGET_HIT | 0.50 | 8.13% |
