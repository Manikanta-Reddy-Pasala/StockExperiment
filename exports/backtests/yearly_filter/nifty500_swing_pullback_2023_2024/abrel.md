# Aditya Birla Real Estate Ltd. (ABREL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (911 bars)
- **Last close:** 1408.90
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 6
- **Target hits / Stop hits / Partials:** 2 / 6 / 3
- **Avg / median % per leg:** 2.60% / 0.00%
- **Sum % (uncompounded):** 28.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 2.60% | 28.5% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 2.60% | 28.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 5 | 45.5% | 2 | 6 | 3 | 2.60% | 28.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 00:00:00 | 861.50 | 758.79 | 805.13 | Stage2 pullback-breakout RSI=69 vol=5.6x ATR=23.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 00:00:00 | 908.21 | 762.71 | 827.94 | T1 booked 50% @ 908.21 |
| Target hit | 2023-08-18 00:00:00 | 1000.10 | 822.63 | 1000.93 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 00:00:00 | 1072.05 | 841.11 | 1014.92 | Stage2 pullback-breakout RSI=65 vol=1.9x ATR=32.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 00:00:00 | 1137.13 | 845.90 | 1027.56 | T1 booked 50% @ 1137.13 |
| Stop hit — per-position SL triggered | 2023-09-07 00:00:00 | 1072.05 | 850.59 | 1037.50 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-10-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 00:00:00 | 1146.95 | 893.59 | 1083.39 | Stage2 pullback-breakout RSI=61 vol=3.1x ATR=47.77 |
| Stop hit — per-position SL triggered | 2023-10-20 00:00:00 | 1075.29 | 914.69 | 1127.48 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-01-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 00:00:00 | 1280.00 | 1028.55 | 1240.61 | Stage2 pullback-breakout RSI=59 vol=2.8x ATR=42.03 |
| Stop hit — per-position SL triggered | 2024-01-04 00:00:00 | 1216.96 | 1036.68 | 1258.06 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-02-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 00:00:00 | 1547.25 | 1110.49 | 1414.62 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=67.36 |
| Stop hit — per-position SL triggered | 2024-02-07 00:00:00 | 1446.21 | 1121.28 | 1430.17 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2024-02-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 00:00:00 | 1491.60 | 1158.97 | 1434.60 | Stage2 pullback-breakout RSI=59 vol=2.1x ATR=62.76 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 1397.46 | 1181.68 | 1445.31 | SL hit (bars_held=8) |

### Cycle 7 — BUY (started 2024-03-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 00:00:00 | 1530.25 | 1185.15 | 1453.40 | Stage2 pullback-breakout RSI=60 vol=4.1x ATR=64.30 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 1433.80 | 1190.40 | 1452.75 | SL hit (bars_held=2) |

### Cycle 8 — BUY (started 2024-03-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 00:00:00 | 1505.95 | 1213.35 | 1444.62 | Stage2 pullback-breakout RSI=57 vol=1.8x ATR=66.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 00:00:00 | 1639.70 | 1217.50 | 1462.26 | T1 booked 50% @ 1639.70 |
| Target hit | 2024-05-07 00:00:00 | 1843.30 | 1351.38 | 1855.89 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-30 00:00:00 | 861.50 | 2023-07-05 00:00:00 | 908.21 | PARTIAL | 0.50 | 5.42% |
| BUY | retest1 | 2023-06-30 00:00:00 | 861.50 | 2023-08-18 00:00:00 | 1000.10 | TARGET_HIT | 0.50 | 16.09% |
| BUY | retest1 | 2023-09-01 00:00:00 | 1072.05 | 2023-09-05 00:00:00 | 1137.13 | PARTIAL | 0.50 | 6.07% |
| BUY | retest1 | 2023-09-01 00:00:00 | 1072.05 | 2023-09-07 00:00:00 | 1072.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-10 00:00:00 | 1146.95 | 2023-10-20 00:00:00 | 1075.29 | STOP_HIT | 1.00 | -6.25% |
| BUY | retest1 | 2024-01-01 00:00:00 | 1280.00 | 2024-01-04 00:00:00 | 1216.96 | STOP_HIT | 1.00 | -4.93% |
| BUY | retest1 | 2024-02-02 00:00:00 | 1547.25 | 2024-02-07 00:00:00 | 1446.21 | STOP_HIT | 1.00 | -6.53% |
| BUY | retest1 | 2024-02-26 00:00:00 | 1491.60 | 2024-03-06 00:00:00 | 1397.46 | STOP_HIT | 1.00 | -6.31% |
| BUY | retest1 | 2024-03-07 00:00:00 | 1530.25 | 2024-03-12 00:00:00 | 1433.80 | STOP_HIT | 1.00 | -6.30% |
| BUY | retest1 | 2024-03-27 00:00:00 | 1505.95 | 2024-03-28 00:00:00 | 1639.70 | PARTIAL | 0.50 | 8.88% |
| BUY | retest1 | 2024-03-27 00:00:00 | 1505.95 | 2024-05-07 00:00:00 | 1843.30 | TARGET_HIT | 0.50 | 22.40% |
