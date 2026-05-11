# Newgen Software Technologies Ltd. (NEWGEN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:25:00 (36496 bars)
- **Last close:** 508.95
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
| ENTRY1 | 25 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 21
- **Target hits / Stop hits / Partials:** 4 / 21 / 11
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 6.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 4 | 30.8% | 1 | 9 | 3 | -0.04% | -0.5% |
| BUY @ 2nd Alert (retest1) | 13 | 4 | 30.8% | 1 | 9 | 3 | -0.04% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 11 | 47.8% | 3 | 12 | 8 | 0.31% | 7.1% |
| SELL @ 2nd Alert (retest1) | 23 | 11 | 47.8% | 3 | 12 | 8 | 0.31% | 7.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 36 | 15 | 41.7% | 4 | 21 | 11 | 0.18% | 6.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:45:00 | 978.00 | 985.29 | 0.00 | ORB-short ORB[978.50,992.70] vol=3.0x ATR=2.98 |
| Stop hit — per-position SL triggered | 2024-05-17 10:50:00 | 980.98 | 985.11 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:40:00 | 924.20 | 930.87 | 0.00 | ORB-short ORB[927.90,939.95] vol=1.9x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:30:00 | 918.66 | 925.69 | 0.00 | T1 1.5R @ 918.66 |
| Target hit | 2024-05-23 15:20:00 | 917.75 | 920.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 885.50 | 887.82 | 0.00 | ORB-short ORB[886.00,893.40] vol=2.8x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:35:00 | 880.97 | 885.05 | 0.00 | T1 1.5R @ 880.97 |
| Target hit | 2024-05-28 10:30:00 | 883.30 | 880.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2024-05-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:20:00 | 858.00 | 860.82 | 0.00 | ORB-short ORB[862.00,870.45] vol=2.1x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:20:00 | 853.97 | 858.45 | 0.00 | T1 1.5R @ 853.97 |
| Stop hit — per-position SL triggered | 2024-05-30 12:05:00 | 858.00 | 857.74 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:20:00 | 844.00 | 853.61 | 0.00 | ORB-short ORB[855.15,863.90] vol=2.0x ATR=2.97 |
| Stop hit — per-position SL triggered | 2024-05-31 10:45:00 | 846.97 | 851.42 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:20:00 | 903.15 | 889.55 | 0.00 | ORB-long ORB[875.65,889.00] vol=2.7x ATR=4.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 10:45:00 | 910.12 | 894.01 | 0.00 | T1 1.5R @ 910.12 |
| Stop hit — per-position SL triggered | 2024-06-06 11:10:00 | 903.15 | 895.13 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:30:00 | 1006.40 | 1014.73 | 0.00 | ORB-short ORB[1013.95,1024.30] vol=2.1x ATR=4.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:45:00 | 998.94 | 1010.55 | 0.00 | T1 1.5R @ 998.94 |
| Stop hit — per-position SL triggered | 2024-06-26 11:15:00 | 1006.40 | 1006.71 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 993.50 | 984.77 | 0.00 | ORB-long ORB[972.50,983.55] vol=5.4x ATR=4.79 |
| Stop hit — per-position SL triggered | 2024-07-01 10:35:00 | 988.71 | 990.13 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:05:00 | 942.85 | 953.18 | 0.00 | ORB-short ORB[950.50,959.00] vol=1.8x ATR=3.50 |
| Stop hit — per-position SL triggered | 2024-07-09 10:10:00 | 946.35 | 952.29 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 940.00 | 948.69 | 0.00 | ORB-short ORB[945.00,958.40] vol=2.3x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 934.47 | 946.34 | 0.00 | T1 1.5R @ 934.47 |
| Stop hit — per-position SL triggered | 2024-07-10 10:45:00 | 940.00 | 941.72 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 09:50:00 | 1063.35 | 1071.27 | 0.00 | ORB-short ORB[1067.45,1078.00] vol=1.7x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 10:00:00 | 1055.21 | 1068.19 | 0.00 | T1 1.5R @ 1055.21 |
| Target hit | 2024-07-30 15:20:00 | 1026.70 | 1042.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2024-08-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 11:10:00 | 1064.15 | 1057.44 | 0.00 | ORB-long ORB[1046.05,1059.90] vol=3.6x ATR=3.84 |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 1060.31 | 1057.60 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:35:00 | 1063.40 | 1071.86 | 0.00 | ORB-short ORB[1069.85,1079.40] vol=1.8x ATR=3.97 |
| Stop hit — per-position SL triggered | 2024-08-23 09:50:00 | 1067.37 | 1068.82 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-09-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:20:00 | 1079.75 | 1075.27 | 0.00 | ORB-long ORB[1051.05,1062.75] vol=15.8x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 10:30:00 | 1087.05 | 1075.49 | 0.00 | T1 1.5R @ 1087.05 |
| Stop hit — per-position SL triggered | 2024-09-05 12:20:00 | 1079.75 | 1077.04 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 11:15:00 | 1069.75 | 1076.06 | 0.00 | ORB-short ORB[1077.40,1088.85] vol=1.8x ATR=2.81 |
| Stop hit — per-position SL triggered | 2024-09-10 11:25:00 | 1072.56 | 1075.06 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 1109.35 | 1100.62 | 0.00 | ORB-long ORB[1090.00,1104.00] vol=2.3x ATR=5.14 |
| Stop hit — per-position SL triggered | 2024-09-13 09:35:00 | 1104.21 | 1101.54 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-09-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:25:00 | 1292.00 | 1295.53 | 0.00 | ORB-short ORB[1303.35,1316.40] vol=4.2x ATR=6.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:30:00 | 1281.78 | 1294.87 | 0.00 | T1 1.5R @ 1281.78 |
| Stop hit — per-position SL triggered | 2024-09-25 11:05:00 | 1292.00 | 1293.46 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-10-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:00:00 | 1246.20 | 1227.47 | 0.00 | ORB-long ORB[1214.70,1230.95] vol=2.9x ATR=6.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:20:00 | 1255.36 | 1238.15 | 0.00 | T1 1.5R @ 1255.36 |
| Target hit | 2024-10-31 11:45:00 | 1253.00 | 1253.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2024-12-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:05:00 | 1377.30 | 1370.17 | 0.00 | ORB-long ORB[1356.00,1375.00] vol=1.7x ATR=6.84 |
| Stop hit — per-position SL triggered | 2024-12-09 10:30:00 | 1370.46 | 1370.54 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-12-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:00:00 | 1423.85 | 1404.91 | 0.00 | ORB-long ORB[1390.55,1407.90] vol=2.0x ATR=7.02 |
| Stop hit — per-position SL triggered | 2024-12-16 10:05:00 | 1416.83 | 1406.68 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 11:15:00 | 1695.15 | 1698.64 | 0.00 | ORB-short ORB[1695.20,1711.95] vol=1.5x ATR=5.00 |
| Stop hit — per-position SL triggered | 2025-01-03 11:35:00 | 1700.15 | 1698.69 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 09:35:00 | 926.15 | 930.54 | 0.00 | ORB-short ORB[927.15,939.00] vol=2.5x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:10:00 | 921.17 | 927.96 | 0.00 | T1 1.5R @ 921.17 |
| Stop hit — per-position SL triggered | 2025-03-19 11:35:00 | 926.15 | 926.06 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:15:00 | 925.45 | 928.24 | 0.00 | ORB-short ORB[926.00,936.00] vol=1.7x ATR=3.20 |
| Stop hit — per-position SL triggered | 2025-04-17 11:20:00 | 928.65 | 927.41 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:45:00 | 955.05 | 946.50 | 0.00 | ORB-long ORB[936.00,949.00] vol=2.3x ATR=4.50 |
| Stop hit — per-position SL triggered | 2025-04-22 11:20:00 | 950.55 | 950.07 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:30:00 | 1103.20 | 1092.32 | 0.00 | ORB-long ORB[1081.35,1095.40] vol=2.6x ATR=5.06 |
| Stop hit — per-position SL triggered | 2025-05-08 09:40:00 | 1098.14 | 1096.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-17 10:45:00 | 978.00 | 2024-05-17 10:50:00 | 980.98 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-23 09:40:00 | 924.20 | 2024-05-23 10:30:00 | 918.66 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-05-23 09:40:00 | 924.20 | 2024-05-23 15:20:00 | 917.75 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2024-05-28 09:30:00 | 885.50 | 2024-05-28 09:35:00 | 880.97 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-28 09:30:00 | 885.50 | 2024-05-28 10:30:00 | 883.30 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2024-05-30 10:20:00 | 858.00 | 2024-05-30 11:20:00 | 853.97 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-30 10:20:00 | 858.00 | 2024-05-30 12:05:00 | 858.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-31 10:20:00 | 844.00 | 2024-05-31 10:45:00 | 846.97 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-06 10:20:00 | 903.15 | 2024-06-06 10:45:00 | 910.12 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-06-06 10:20:00 | 903.15 | 2024-06-06 11:10:00 | 903.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-26 09:30:00 | 1006.40 | 2024-06-26 09:45:00 | 998.94 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2024-06-26 09:30:00 | 1006.40 | 2024-06-26 11:15:00 | 1006.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-01 09:30:00 | 993.50 | 2024-07-01 10:35:00 | 988.71 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-07-09 10:05:00 | 942.85 | 2024-07-09 10:10:00 | 946.35 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-10 10:10:00 | 940.00 | 2024-07-10 10:20:00 | 934.47 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-07-10 10:10:00 | 940.00 | 2024-07-10 10:45:00 | 940.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-30 09:50:00 | 1063.35 | 2024-07-30 10:00:00 | 1055.21 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-07-30 09:50:00 | 1063.35 | 2024-07-30 15:20:00 | 1026.70 | TARGET_HIT | 0.50 | 3.45% |
| BUY | retest1 | 2024-08-13 11:10:00 | 1064.15 | 2024-08-13 11:15:00 | 1060.31 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-23 09:35:00 | 1063.40 | 2024-08-23 09:50:00 | 1067.37 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-05 10:20:00 | 1079.75 | 2024-09-05 10:30:00 | 1087.05 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-09-05 10:20:00 | 1079.75 | 2024-09-05 12:20:00 | 1079.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-10 11:15:00 | 1069.75 | 2024-09-10 11:25:00 | 1072.56 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-13 09:30:00 | 1109.35 | 2024-09-13 09:35:00 | 1104.21 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-09-25 10:25:00 | 1292.00 | 2024-09-25 10:30:00 | 1281.78 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2024-09-25 10:25:00 | 1292.00 | 2024-09-25 11:05:00 | 1292.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 10:00:00 | 1246.20 | 2024-10-31 10:20:00 | 1255.36 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-10-31 10:00:00 | 1246.20 | 2024-10-31 11:45:00 | 1253.00 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2024-12-09 10:05:00 | 1377.30 | 2024-12-09 10:30:00 | 1370.46 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-12-16 10:00:00 | 1423.85 | 2024-12-16 10:05:00 | 1416.83 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-01-03 11:15:00 | 1695.15 | 2025-01-03 11:35:00 | 1700.15 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-03-19 09:35:00 | 926.15 | 2025-03-19 10:10:00 | 921.17 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-03-19 09:35:00 | 926.15 | 2025-03-19 11:35:00 | 926.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-17 10:15:00 | 925.45 | 2025-04-17 11:20:00 | 928.65 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-22 09:45:00 | 955.05 | 2025-04-22 11:20:00 | 950.55 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-05-08 09:30:00 | 1103.20 | 2025-05-08 09:40:00 | 1098.14 | STOP_HIT | 1.00 | -0.46% |
