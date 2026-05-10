# The Ramco Cements Ltd. (RAMCOCEM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 953.00
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 4
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 1.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.09% | -0.5% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.09% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.26% | 2.3% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.26% | 2.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 5 | 33.3% | 1 | 10 | 4 | 0.12% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:30:00 | 1139.40 | 1135.14 | 0.00 | ORB-long ORB[1131.10,1137.80] vol=6.5x ATR=3.31 |
| Stop hit — per-position SL triggered | 2026-02-13 11:20:00 | 1136.09 | 1135.79 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:00:00 | 1154.40 | 1156.63 | 0.00 | ORB-short ORB[1157.70,1168.00] vol=1.9x ATR=4.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:10:00 | 1147.50 | 1155.99 | 0.00 | T1 1.5R @ 1147.50 |
| Stop hit — per-position SL triggered | 2026-02-18 13:20:00 | 1154.40 | 1155.25 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 1125.70 | 1117.20 | 0.00 | ORB-long ORB[1107.50,1124.00] vol=1.8x ATR=4.58 |
| Stop hit — per-position SL triggered | 2026-02-20 10:40:00 | 1121.12 | 1117.27 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:05:00 | 1073.40 | 1078.40 | 0.00 | ORB-short ORB[1075.90,1090.90] vol=1.8x ATR=4.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:10:00 | 1066.84 | 1076.06 | 0.00 | T1 1.5R @ 1066.84 |
| Target hit | 2026-03-05 13:05:00 | 1053.10 | 1052.83 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:15:00 | 1087.00 | 1091.34 | 0.00 | ORB-short ORB[1087.30,1094.20] vol=2.4x ATR=4.48 |
| Stop hit — per-position SL triggered | 2026-03-06 11:30:00 | 1091.48 | 1091.91 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:55:00 | 1019.70 | 1022.67 | 0.00 | ORB-short ORB[1031.00,1039.40] vol=3.8x ATR=6.52 |
| Stop hit — per-position SL triggered | 2026-03-10 10:40:00 | 1026.22 | 1022.60 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 10:55:00 | 950.85 | 937.60 | 0.00 | ORB-long ORB[927.00,939.05] vol=1.7x ATR=4.94 |
| Stop hit — per-position SL triggered | 2026-04-01 11:25:00 | 945.91 | 938.70 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:45:00 | 978.60 | 969.87 | 0.00 | ORB-long ORB[959.00,972.90] vol=3.6x ATR=6.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:55:00 | 987.70 | 975.23 | 0.00 | T1 1.5R @ 987.70 |
| Stop hit — per-position SL triggered | 2026-04-08 10:00:00 | 978.60 | 975.26 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:55:00 | 1007.75 | 1003.80 | 0.00 | ORB-long ORB[996.45,1005.00] vol=2.4x ATR=2.48 |
| Stop hit — per-position SL triggered | 2026-04-17 11:00:00 | 1005.27 | 1004.86 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 958.00 | 960.92 | 0.00 | ORB-short ORB[959.65,968.55] vol=1.5x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-04-29 11:35:00 | 960.44 | 959.35 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 916.00 | 922.30 | 0.00 | ORB-short ORB[918.95,927.00] vol=2.2x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:10:00 | 911.17 | 917.42 | 0.00 | T1 1.5R @ 911.17 |
| Stop hit — per-position SL triggered | 2026-05-05 12:50:00 | 916.00 | 913.46 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-13 10:30:00 | 1139.40 | 2026-02-13 11:20:00 | 1136.09 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-18 10:00:00 | 1154.40 | 2026-02-18 11:10:00 | 1147.50 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-18 10:00:00 | 1154.40 | 2026-02-18 13:20:00 | 1154.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:35:00 | 1125.70 | 2026-02-20 10:40:00 | 1121.12 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-05 10:05:00 | 1073.40 | 2026-03-05 10:10:00 | 1066.84 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-03-05 10:05:00 | 1073.40 | 2026-03-05 13:05:00 | 1053.10 | TARGET_HIT | 0.50 | 1.89% |
| SELL | retest1 | 2026-03-06 11:15:00 | 1087.00 | 2026-03-06 11:30:00 | 1091.48 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-10 09:55:00 | 1019.70 | 2026-03-10 10:40:00 | 1026.22 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2026-04-01 10:55:00 | 950.85 | 2026-04-01 11:25:00 | 945.91 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-04-08 09:45:00 | 978.60 | 2026-04-08 09:55:00 | 987.70 | PARTIAL | 0.50 | 0.93% |
| BUY | retest1 | 2026-04-08 09:45:00 | 978.60 | 2026-04-08 10:00:00 | 978.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:55:00 | 1007.75 | 2026-04-17 11:00:00 | 1005.27 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-29 10:55:00 | 958.00 | 2026-04-29 11:35:00 | 960.44 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-05-05 10:10:00 | 916.00 | 2026-05-05 11:10:00 | 911.17 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-05-05 10:10:00 | 916.00 | 2026-05-05 12:50:00 | 916.00 | STOP_HIT | 0.50 | 0.00% |
