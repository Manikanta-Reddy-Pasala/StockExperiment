# Mazagoan Dock Shipbuilders Ltd. (MAZDOCK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 2576.10
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -0.33% / 0.00%
- **Sum % (uncompounded):** -1.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.33% | -2.0% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.33% | -2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.33% | -2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 00:00:00 | 967.85 | 514.22 | 894.90 | Stage2 pullback-breakout RSI=66 vol=2.6x ATR=46.59 |
| Stop hit — per-position SL triggered | 2023-08-30 00:00:00 | 931.98 | 553.85 | 915.97 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-11-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 00:00:00 | 1019.93 | 764.28 | 997.31 | Stage2 pullback-breakout RSI=55 vol=2.1x ATR=31.94 |
| Stop hit — per-position SL triggered | 2023-12-12 00:00:00 | 1036.85 | 791.34 | 1018.09 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 00:00:00 | 1143.93 | 814.49 | 1046.27 | Stage2 pullback-breakout RSI=69 vol=5.6x ATR=41.70 |
| Stop hit — per-position SL triggered | 2024-01-09 00:00:00 | 1115.20 | 844.93 | 1099.73 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 1118.35 | 935.10 | 993.13 | Stage2 pullback-breakout RSI=67 vol=7.2x ATR=45.54 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 1050.04 | 946.44 | 1047.74 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2024-04-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 00:00:00 | 1129.33 | 953.89 | 1068.44 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=49.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 00:00:00 | 1227.55 | 958.60 | 1091.07 | T1 booked 50% @ 1227.55 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 1129.33 | 971.21 | 1127.70 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-16 00:00:00 | 967.85 | 2023-08-30 00:00:00 | 931.98 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest1 | 2023-11-24 00:00:00 | 1019.93 | 2023-12-12 00:00:00 | 1036.85 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest1 | 2023-12-26 00:00:00 | 1143.93 | 2024-01-09 00:00:00 | 1115.20 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest1 | 2024-04-03 00:00:00 | 1118.35 | 2024-04-15 00:00:00 | 1050.04 | STOP_HIT | 1.00 | -6.11% |
| BUY | retest1 | 2024-04-23 00:00:00 | 1129.33 | 2024-04-25 00:00:00 | 1227.55 | PARTIAL | 0.50 | 8.70% |
| BUY | retest1 | 2024-04-23 00:00:00 | 1129.33 | 2024-05-06 00:00:00 | 1129.33 | STOP_HIT | 0.50 | 0.00% |
