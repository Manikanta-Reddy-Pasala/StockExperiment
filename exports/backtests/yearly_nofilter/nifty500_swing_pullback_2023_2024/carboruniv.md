# Carborundum Universal Ltd. (CARBORUNIV)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1006.20
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 2
- **Avg / median % per leg:** 1.92% / 1.90%
- **Sum % (uncompounded):** 9.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 0 | 3 | 2 | 1.92% | 9.6% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 1.92% | 9.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 3 | 2 | 1.92% | 9.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 00:00:00 | 1225.70 | 1044.79 | 1197.96 | Stage2 pullback-breakout RSI=61 vol=1.5x ATR=31.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 00:00:00 | 1288.12 | 1048.83 | 1207.51 | T1 booked 50% @ 1288.12 |
| Stop hit — per-position SL triggered | 2023-08-07 00:00:00 | 1225.70 | 1056.01 | 1214.85 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-11-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 00:00:00 | 1170.05 | 1094.69 | 1101.04 | Stage2 pullback-breakout RSI=67 vol=4.7x ATR=35.77 |
| Stop hit — per-position SL triggered | 2023-12-11 00:00:00 | 1192.30 | 1103.88 | 1157.93 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2024-01-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 00:00:00 | 1177.20 | 1114.07 | 1143.82 | Stage2 pullback-breakout RSI=58 vol=4.7x ATR=35.63 |
| Stop hit — per-position SL triggered | 2024-01-30 00:00:00 | 1123.76 | 1114.63 | 1140.71 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2024-04-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 00:00:00 | 1341.10 | 1139.10 | 1253.46 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=48.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 00:00:00 | 1437.26 | 1144.82 | 1285.18 | T1 booked 50% @ 1437.26 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-28 00:00:00 | 1225.70 | 2023-08-01 00:00:00 | 1288.12 | PARTIAL | 0.50 | 5.09% |
| BUY | retest1 | 2023-07-28 00:00:00 | 1225.70 | 2023-08-07 00:00:00 | 1225.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-22 00:00:00 | 1170.05 | 2023-12-11 00:00:00 | 1192.30 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest1 | 2024-01-24 00:00:00 | 1177.20 | 2024-01-30 00:00:00 | 1123.76 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest1 | 2024-04-24 00:00:00 | 1341.10 | 2024-04-26 00:00:00 | 1437.26 | PARTIAL | 0.50 | 7.17% |
