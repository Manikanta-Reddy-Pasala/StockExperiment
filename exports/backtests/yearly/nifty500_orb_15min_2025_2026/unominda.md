# UNO Minda Ltd. (UNOMINDA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1179.90
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
| ENTRY1 | 99 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 20 |
| STOP_HIT | 79 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 138 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 79
- **Target hits / Stop hits / Partials:** 20 / 79 / 39
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 20.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 27 | 42.2% | 9 | 37 | 18 | 0.15% | 9.6% |
| BUY @ 2nd Alert (retest1) | 64 | 27 | 42.2% | 9 | 37 | 18 | 0.15% | 9.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 74 | 32 | 43.2% | 11 | 42 | 21 | 0.14% | 10.4% |
| SELL @ 2nd Alert (retest1) | 74 | 32 | 43.2% | 11 | 42 | 21 | 0.14% | 10.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 138 | 59 | 42.8% | 20 | 79 | 39 | 0.15% | 20.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 983.85 | 989.53 | 0.00 | ORB-short ORB[987.00,996.70] vol=1.6x ATR=3.28 |
| Stop hit — per-position SL triggered | 2025-05-19 09:55:00 | 987.13 | 987.77 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:45:00 | 1004.40 | 1011.91 | 0.00 | ORB-short ORB[1009.40,1023.45] vol=1.7x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 10:15:00 | 998.27 | 1006.47 | 0.00 | T1 1.5R @ 998.27 |
| Stop hit — per-position SL triggered | 2025-05-27 11:50:00 | 1004.40 | 1003.04 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 09:30:00 | 1003.70 | 1010.10 | 0.00 | ORB-short ORB[1005.10,1019.60] vol=2.7x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 09:55:00 | 998.78 | 1005.35 | 0.00 | T1 1.5R @ 998.78 |
| Stop hit — per-position SL triggered | 2025-05-28 11:00:00 | 1003.70 | 1004.70 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 10:50:00 | 1012.45 | 1010.15 | 0.00 | ORB-long ORB[1002.70,1010.70] vol=6.6x ATR=3.09 |
| Stop hit — per-position SL triggered | 2025-05-29 10:55:00 | 1009.36 | 1009.94 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:40:00 | 1017.45 | 1015.32 | 0.00 | ORB-long ORB[1007.00,1017.00] vol=2.1x ATR=3.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 09:45:00 | 1022.81 | 1020.05 | 0.00 | T1 1.5R @ 1022.81 |
| Target hit | 2025-05-30 09:55:00 | 1019.85 | 1020.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2025-06-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:55:00 | 1027.00 | 1024.98 | 0.00 | ORB-long ORB[1005.50,1019.70] vol=3.1x ATR=3.75 |
| Stop hit — per-position SL triggered | 2025-06-02 11:10:00 | 1023.25 | 1025.15 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:35:00 | 1030.50 | 1025.76 | 0.00 | ORB-long ORB[1018.10,1029.00] vol=2.0x ATR=3.74 |
| Stop hit — per-position SL triggered | 2025-06-03 09:55:00 | 1026.76 | 1027.45 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:30:00 | 1095.70 | 1082.84 | 0.00 | ORB-long ORB[1072.20,1084.40] vol=1.5x ATR=4.95 |
| Stop hit — per-position SL triggered | 2025-06-06 11:05:00 | 1090.75 | 1087.22 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:05:00 | 1090.50 | 1097.51 | 0.00 | ORB-short ORB[1099.30,1107.90] vol=4.1x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:45:00 | 1086.21 | 1095.41 | 0.00 | T1 1.5R @ 1086.21 |
| Target hit | 2025-06-12 15:20:00 | 1072.00 | 1082.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2025-06-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:35:00 | 1051.70 | 1043.87 | 0.00 | ORB-long ORB[1036.00,1045.00] vol=2.1x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:40:00 | 1059.02 | 1047.26 | 0.00 | T1 1.5R @ 1059.02 |
| Stop hit — per-position SL triggered | 2025-06-20 09:45:00 | 1051.70 | 1049.05 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-23 10:50:00 | 1044.90 | 1051.79 | 0.00 | ORB-short ORB[1050.10,1063.50] vol=2.4x ATR=3.96 |
| Stop hit — per-position SL triggered | 2025-06-23 11:10:00 | 1048.86 | 1050.67 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:05:00 | 1064.50 | 1067.32 | 0.00 | ORB-short ORB[1068.50,1079.80] vol=2.4x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 11:15:00 | 1060.33 | 1066.93 | 0.00 | T1 1.5R @ 1060.33 |
| Stop hit — per-position SL triggered | 2025-06-26 11:25:00 | 1064.50 | 1066.78 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:50:00 | 1086.10 | 1094.85 | 0.00 | ORB-short ORB[1096.10,1109.90] vol=2.3x ATR=2.90 |
| Stop hit — per-position SL triggered | 2025-07-01 11:05:00 | 1089.00 | 1094.25 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 11:00:00 | 1098.00 | 1093.23 | 0.00 | ORB-long ORB[1082.30,1096.00] vol=1.5x ATR=3.15 |
| Stop hit — per-position SL triggered | 2025-07-02 12:20:00 | 1094.85 | 1095.25 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:10:00 | 1108.90 | 1105.69 | 0.00 | ORB-long ORB[1101.30,1108.00] vol=3.9x ATR=3.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:15:00 | 1113.80 | 1106.64 | 0.00 | T1 1.5R @ 1113.80 |
| Target hit | 2025-07-03 13:20:00 | 1112.60 | 1112.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2025-07-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:45:00 | 1079.00 | 1083.63 | 0.00 | ORB-short ORB[1082.20,1097.90] vol=2.5x ATR=3.22 |
| Stop hit — per-position SL triggered | 2025-07-11 09:50:00 | 1082.22 | 1083.58 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:20:00 | 1075.60 | 1071.63 | 0.00 | ORB-long ORB[1061.70,1071.80] vol=2.9x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 12:25:00 | 1080.53 | 1076.85 | 0.00 | T1 1.5R @ 1080.53 |
| Stop hit — per-position SL triggered | 2025-07-14 12:30:00 | 1075.60 | 1076.93 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:10:00 | 1133.20 | 1125.82 | 0.00 | ORB-long ORB[1121.00,1130.60] vol=2.0x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 10:15:00 | 1138.19 | 1129.57 | 0.00 | T1 1.5R @ 1138.19 |
| Stop hit — per-position SL triggered | 2025-07-17 10:50:00 | 1133.20 | 1133.34 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 1098.50 | 1106.34 | 0.00 | ORB-short ORB[1104.30,1117.60] vol=1.6x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-07-18 10:35:00 | 1101.48 | 1105.25 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:30:00 | 1071.40 | 1075.87 | 0.00 | ORB-short ORB[1074.40,1083.70] vol=2.2x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-07-22 10:35:00 | 1073.82 | 1075.84 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 10:25:00 | 1086.00 | 1079.46 | 0.00 | ORB-long ORB[1071.40,1079.50] vol=2.6x ATR=3.12 |
| Stop hit — per-position SL triggered | 2025-07-24 11:20:00 | 1082.88 | 1082.12 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:30:00 | 1053.90 | 1064.73 | 0.00 | ORB-short ORB[1061.30,1073.30] vol=1.6x ATR=4.63 |
| Stop hit — per-position SL triggered | 2025-07-30 09:35:00 | 1058.53 | 1062.91 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 10:15:00 | 1044.90 | 1051.29 | 0.00 | ORB-short ORB[1050.00,1058.30] vol=2.3x ATR=3.19 |
| Stop hit — per-position SL triggered | 2025-07-31 10:40:00 | 1048.09 | 1049.71 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:05:00 | 1082.00 | 1092.70 | 0.00 | ORB-short ORB[1095.60,1104.00] vol=1.7x ATR=3.87 |
| Stop hit — per-position SL triggered | 2025-08-06 11:10:00 | 1085.87 | 1092.57 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:45:00 | 1070.40 | 1076.24 | 0.00 | ORB-short ORB[1073.70,1082.40] vol=1.7x ATR=3.59 |
| Stop hit — per-position SL triggered | 2025-08-11 09:50:00 | 1073.99 | 1075.62 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:55:00 | 1076.00 | 1082.40 | 0.00 | ORB-short ORB[1086.20,1094.80] vol=7.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 11:15:00 | 1072.15 | 1079.71 | 0.00 | T1 1.5R @ 1072.15 |
| Stop hit — per-position SL triggered | 2025-08-12 12:05:00 | 1076.00 | 1078.59 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:15:00 | 1084.60 | 1076.78 | 0.00 | ORB-long ORB[1070.00,1080.40] vol=2.3x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 10:25:00 | 1089.66 | 1081.92 | 0.00 | T1 1.5R @ 1089.66 |
| Target hit | 2025-08-13 15:20:00 | 1114.00 | 1103.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2025-08-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:50:00 | 1224.50 | 1219.00 | 0.00 | ORB-long ORB[1212.90,1223.60] vol=1.5x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 10:15:00 | 1230.12 | 1221.48 | 0.00 | T1 1.5R @ 1230.12 |
| Stop hit — per-position SL triggered | 2025-08-19 10:45:00 | 1224.50 | 1222.33 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 11:15:00 | 1246.80 | 1239.77 | 0.00 | ORB-long ORB[1234.20,1243.50] vol=4.1x ATR=2.62 |
| Stop hit — per-position SL triggered | 2025-08-21 11:30:00 | 1244.18 | 1241.72 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:50:00 | 1241.30 | 1232.86 | 0.00 | ORB-long ORB[1227.10,1237.90] vol=1.9x ATR=3.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 11:15:00 | 1246.65 | 1235.86 | 0.00 | T1 1.5R @ 1246.65 |
| Target hit | 2025-08-22 15:20:00 | 1265.50 | 1256.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2025-08-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 10:35:00 | 1292.00 | 1283.28 | 0.00 | ORB-long ORB[1276.70,1287.00] vol=1.5x ATR=3.80 |
| Stop hit — per-position SL triggered | 2025-08-26 11:05:00 | 1288.20 | 1286.11 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 09:40:00 | 1288.40 | 1282.79 | 0.00 | ORB-long ORB[1272.00,1287.80] vol=1.6x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:50:00 | 1294.78 | 1284.93 | 0.00 | T1 1.5R @ 1294.78 |
| Stop hit — per-position SL triggered | 2025-08-28 10:15:00 | 1288.40 | 1290.34 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:05:00 | 1308.90 | 1303.17 | 0.00 | ORB-long ORB[1285.90,1304.40] vol=1.6x ATR=3.88 |
| Stop hit — per-position SL triggered | 2025-09-01 10:20:00 | 1305.02 | 1304.13 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 09:45:00 | 1296.90 | 1300.64 | 0.00 | ORB-short ORB[1298.80,1310.00] vol=3.2x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 10:35:00 | 1291.06 | 1298.78 | 0.00 | T1 1.5R @ 1291.06 |
| Target hit | 2025-09-03 15:20:00 | 1282.10 | 1289.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2025-09-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 10:55:00 | 1294.00 | 1303.50 | 0.00 | ORB-short ORB[1295.00,1312.50] vol=3.7x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 11:05:00 | 1287.24 | 1301.24 | 0.00 | T1 1.5R @ 1287.24 |
| Target hit | 2025-09-04 15:20:00 | 1281.70 | 1293.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2025-09-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 10:05:00 | 1302.70 | 1295.16 | 0.00 | ORB-long ORB[1279.80,1298.70] vol=1.8x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:20:00 | 1308.86 | 1297.98 | 0.00 | T1 1.5R @ 1308.86 |
| Stop hit — per-position SL triggered | 2025-09-05 10:45:00 | 1302.70 | 1301.96 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:35:00 | 1317.00 | 1313.47 | 0.00 | ORB-long ORB[1302.40,1316.70] vol=2.2x ATR=4.05 |
| Stop hit — per-position SL triggered | 2025-09-08 09:40:00 | 1312.95 | 1313.31 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 09:35:00 | 1289.90 | 1294.32 | 0.00 | ORB-short ORB[1292.20,1310.00] vol=3.7x ATR=3.76 |
| Stop hit — per-position SL triggered | 2025-09-09 09:40:00 | 1293.66 | 1294.30 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 11:05:00 | 1278.50 | 1270.89 | 0.00 | ORB-long ORB[1263.70,1272.20] vol=3.5x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-09-11 11:35:00 | 1275.70 | 1272.22 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:40:00 | 1291.30 | 1286.96 | 0.00 | ORB-long ORB[1278.00,1289.60] vol=1.6x ATR=3.04 |
| Stop hit — per-position SL triggered | 2025-09-12 09:50:00 | 1288.26 | 1287.67 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 09:45:00 | 1284.30 | 1290.87 | 0.00 | ORB-short ORB[1285.60,1298.40] vol=1.6x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-09-15 10:20:00 | 1287.96 | 1289.36 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:15:00 | 1315.00 | 1306.35 | 0.00 | ORB-long ORB[1300.50,1312.00] vol=1.7x ATR=3.76 |
| Stop hit — per-position SL triggered | 2025-09-16 10:30:00 | 1311.24 | 1307.22 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-09-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 11:05:00 | 1311.00 | 1316.01 | 0.00 | ORB-short ORB[1313.00,1327.00] vol=2.6x ATR=2.48 |
| Stop hit — per-position SL triggered | 2025-09-17 12:05:00 | 1313.48 | 1314.98 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:15:00 | 1312.30 | 1314.31 | 0.00 | ORB-short ORB[1312.60,1318.60] vol=1.7x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-09-19 11:25:00 | 1314.95 | 1314.34 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:15:00 | 1308.70 | 1319.56 | 0.00 | ORB-short ORB[1314.50,1324.80] vol=2.1x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 11:35:00 | 1304.62 | 1316.65 | 0.00 | T1 1.5R @ 1304.62 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1308.70 | 1313.58 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:45:00 | 1310.50 | 1305.59 | 0.00 | ORB-long ORB[1295.00,1310.40] vol=1.6x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 10:55:00 | 1316.92 | 1306.53 | 0.00 | T1 1.5R @ 1316.92 |
| Target hit | 2025-10-01 15:20:00 | 1323.60 | 1316.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2025-10-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:40:00 | 1310.90 | 1318.32 | 0.00 | ORB-short ORB[1313.10,1331.80] vol=2.0x ATR=3.40 |
| Stop hit — per-position SL triggered | 2025-10-03 11:00:00 | 1314.30 | 1317.42 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 11:10:00 | 1327.80 | 1315.96 | 0.00 | ORB-long ORB[1307.60,1317.50] vol=1.9x ATR=2.90 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 1324.90 | 1317.38 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:00:00 | 1348.70 | 1360.13 | 0.00 | ORB-short ORB[1361.80,1374.90] vol=3.0x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-10-08 11:40:00 | 1352.36 | 1357.77 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-10-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:20:00 | 1202.10 | 1196.34 | 0.00 | ORB-long ORB[1191.90,1201.50] vol=1.7x ATR=3.12 |
| Stop hit — per-position SL triggered | 2025-10-29 11:10:00 | 1198.98 | 1198.01 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 11:15:00 | 1221.10 | 1214.08 | 0.00 | ORB-long ORB[1207.70,1217.90] vol=4.4x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 11:20:00 | 1225.12 | 1216.42 | 0.00 | T1 1.5R @ 1225.12 |
| Stop hit — per-position SL triggered | 2025-10-30 12:40:00 | 1221.10 | 1219.44 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:35:00 | 1248.30 | 1232.17 | 0.00 | ORB-long ORB[1220.90,1237.00] vol=2.0x ATR=5.77 |
| Stop hit — per-position SL triggered | 2025-11-07 10:45:00 | 1242.53 | 1235.01 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 11:15:00 | 1307.80 | 1319.31 | 0.00 | ORB-short ORB[1317.20,1334.00] vol=3.9x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 11:30:00 | 1303.60 | 1316.85 | 0.00 | T1 1.5R @ 1303.60 |
| Stop hit — per-position SL triggered | 2025-11-13 11:35:00 | 1307.80 | 1316.59 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:05:00 | 1286.70 | 1290.88 | 0.00 | ORB-short ORB[1291.00,1309.00] vol=1.8x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 11:15:00 | 1282.28 | 1287.19 | 0.00 | T1 1.5R @ 1282.28 |
| Target hit | 2025-11-18 13:50:00 | 1282.30 | 1281.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 55 — SELL (started 2025-11-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 11:10:00 | 1297.30 | 1301.57 | 0.00 | ORB-short ORB[1300.00,1317.00] vol=1.7x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 12:05:00 | 1293.02 | 1300.86 | 0.00 | T1 1.5R @ 1293.02 |
| Stop hit — per-position SL triggered | 2025-11-20 13:25:00 | 1297.30 | 1298.93 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:15:00 | 1293.20 | 1297.67 | 0.00 | ORB-short ORB[1294.80,1302.80] vol=2.5x ATR=2.61 |
| Stop hit — per-position SL triggered | 2025-11-21 11:45:00 | 1295.81 | 1296.93 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 1292.30 | 1287.94 | 0.00 | ORB-long ORB[1274.90,1289.50] vol=1.8x ATR=4.34 |
| Stop hit — per-position SL triggered | 2025-11-26 09:35:00 | 1287.96 | 1288.15 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:50:00 | 1270.50 | 1281.42 | 0.00 | ORB-short ORB[1285.00,1300.00] vol=2.3x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:20:00 | 1263.81 | 1277.05 | 0.00 | T1 1.5R @ 1263.81 |
| Stop hit — per-position SL triggered | 2025-12-03 11:45:00 | 1270.50 | 1275.58 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:45:00 | 1276.20 | 1273.14 | 0.00 | ORB-long ORB[1260.10,1271.00] vol=4.7x ATR=3.03 |
| Stop hit — per-position SL triggered | 2025-12-04 10:55:00 | 1273.17 | 1273.20 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:00:00 | 1254.30 | 1258.75 | 0.00 | ORB-short ORB[1259.50,1272.40] vol=1.7x ATR=5.03 |
| Stop hit — per-position SL triggered | 2025-12-08 10:10:00 | 1259.33 | 1256.51 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 11:10:00 | 1237.80 | 1242.40 | 0.00 | ORB-short ORB[1240.00,1252.00] vol=2.5x ATR=4.12 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 1241.92 | 1242.99 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:05:00 | 1235.00 | 1229.26 | 0.00 | ORB-long ORB[1217.60,1229.70] vol=1.9x ATR=3.20 |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 1231.80 | 1232.38 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:10:00 | 1235.70 | 1244.28 | 0.00 | ORB-short ORB[1239.70,1250.80] vol=1.7x ATR=2.82 |
| Stop hit — per-position SL triggered | 2025-12-15 11:20:00 | 1238.52 | 1243.65 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:40:00 | 1249.40 | 1257.75 | 0.00 | ORB-short ORB[1251.00,1267.80] vol=3.0x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 10:55:00 | 1244.35 | 1257.10 | 0.00 | T1 1.5R @ 1244.35 |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 1249.40 | 1256.34 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:45:00 | 1241.10 | 1237.77 | 0.00 | ORB-long ORB[1222.30,1239.00] vol=1.9x ATR=2.78 |
| Stop hit — per-position SL triggered | 2025-12-19 10:50:00 | 1238.32 | 1237.93 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 09:30:00 | 1281.00 | 1274.19 | 0.00 | ORB-long ORB[1262.40,1274.90] vol=1.8x ATR=3.63 |
| Stop hit — per-position SL triggered | 2025-12-22 09:40:00 | 1277.37 | 1275.45 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 1304.40 | 1299.49 | 0.00 | ORB-long ORB[1288.70,1303.40] vol=12.3x ATR=2.92 |
| Stop hit — per-position SL triggered | 2025-12-24 11:20:00 | 1301.48 | 1300.36 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 09:30:00 | 1283.80 | 1288.04 | 0.00 | ORB-short ORB[1285.20,1296.00] vol=1.9x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 10:10:00 | 1278.32 | 1285.26 | 0.00 | T1 1.5R @ 1278.32 |
| Target hit | 2025-12-26 15:20:00 | 1271.90 | 1274.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 11:15:00 | 1270.10 | 1263.20 | 0.00 | ORB-long ORB[1256.10,1263.30] vol=2.4x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-12-30 12:25:00 | 1267.69 | 1265.47 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 1282.30 | 1270.91 | 0.00 | ORB-long ORB[1268.00,1280.00] vol=1.7x ATR=3.94 |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 1278.36 | 1271.93 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:05:00 | 1288.50 | 1287.80 | 0.00 | ORB-long ORB[1275.90,1284.00] vol=2.2x ATR=2.69 |
| Stop hit — per-position SL triggered | 2026-01-01 12:05:00 | 1285.81 | 1287.88 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:45:00 | 1297.40 | 1293.47 | 0.00 | ORB-long ORB[1287.80,1296.00] vol=2.3x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:50:00 | 1302.26 | 1297.14 | 0.00 | T1 1.5R @ 1302.26 |
| Target hit | 2026-01-02 12:45:00 | 1308.10 | 1308.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — BUY (started 2026-01-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:50:00 | 1345.10 | 1332.30 | 0.00 | ORB-long ORB[1324.50,1336.50] vol=2.8x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:00:00 | 1351.18 | 1337.16 | 0.00 | T1 1.5R @ 1351.18 |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 1345.10 | 1339.05 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:55:00 | 1320.70 | 1326.09 | 0.00 | ORB-short ORB[1333.90,1343.10] vol=1.9x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:10:00 | 1315.15 | 1324.96 | 0.00 | T1 1.5R @ 1315.15 |
| Stop hit — per-position SL triggered | 2026-01-06 13:15:00 | 1320.70 | 1320.70 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-01-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:35:00 | 1328.50 | 1320.49 | 0.00 | ORB-long ORB[1311.90,1321.00] vol=3.1x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 10:50:00 | 1333.16 | 1322.07 | 0.00 | T1 1.5R @ 1333.16 |
| Stop hit — per-position SL triggered | 2026-01-07 11:00:00 | 1328.50 | 1322.98 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:15:00 | 1301.20 | 1308.80 | 0.00 | ORB-short ORB[1308.20,1318.30] vol=2.2x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:45:00 | 1294.41 | 1302.98 | 0.00 | T1 1.5R @ 1294.41 |
| Target hit | 2026-01-08 15:20:00 | 1268.20 | 1288.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — SELL (started 2026-01-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:45:00 | 1257.80 | 1259.68 | 0.00 | ORB-short ORB[1260.70,1276.60] vol=1.8x ATR=4.92 |
| Stop hit — per-position SL triggered | 2026-01-09 09:50:00 | 1262.72 | 1260.18 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-01-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:45:00 | 1202.40 | 1207.21 | 0.00 | ORB-short ORB[1205.60,1222.60] vol=1.7x ATR=5.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:15:00 | 1194.56 | 1202.57 | 0.00 | T1 1.5R @ 1194.56 |
| Target hit | 2026-01-13 14:45:00 | 1199.40 | 1197.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 79 — BUY (started 2026-01-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:35:00 | 1160.80 | 1147.88 | 0.00 | ORB-long ORB[1134.40,1149.00] vol=1.6x ATR=5.35 |
| Stop hit — per-position SL triggered | 2026-01-22 10:25:00 | 1155.45 | 1154.03 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:15:00 | 1129.40 | 1137.62 | 0.00 | ORB-short ORB[1145.00,1154.90] vol=2.8x ATR=4.51 |
| Stop hit — per-position SL triggered | 2026-01-28 10:40:00 | 1133.91 | 1135.46 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-01-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:55:00 | 1131.30 | 1135.26 | 0.00 | ORB-short ORB[1136.70,1149.90] vol=2.3x ATR=3.90 |
| Stop hit — per-position SL triggered | 2026-01-29 13:10:00 | 1135.20 | 1132.70 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-01-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:20:00 | 1149.80 | 1146.59 | 0.00 | ORB-long ORB[1133.00,1142.00] vol=1.7x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 10:30:00 | 1155.14 | 1147.83 | 0.00 | T1 1.5R @ 1155.14 |
| Target hit | 2026-01-30 11:55:00 | 1159.10 | 1160.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 83 — BUY (started 2026-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:50:00 | 1220.10 | 1217.19 | 0.00 | ORB-long ORB[1200.20,1212.00] vol=2.1x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 12:15:00 | 1226.23 | 1220.11 | 0.00 | T1 1.5R @ 1226.23 |
| Target hit | 2026-02-10 15:20:00 | 1231.00 | 1223.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 1222.10 | 1230.65 | 0.00 | ORB-short ORB[1228.20,1241.00] vol=1.9x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-02-11 09:40:00 | 1225.12 | 1230.26 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-02-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 11:05:00 | 1246.10 | 1237.11 | 0.00 | ORB-long ORB[1231.50,1245.80] vol=1.9x ATR=3.86 |
| Stop hit — per-position SL triggered | 2026-02-12 11:25:00 | 1242.24 | 1239.58 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-02-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:05:00 | 1224.90 | 1228.74 | 0.00 | ORB-short ORB[1228.50,1246.80] vol=2.1x ATR=4.23 |
| Stop hit — per-position SL triggered | 2026-02-13 10:10:00 | 1229.13 | 1228.55 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 1232.20 | 1239.60 | 0.00 | ORB-short ORB[1233.40,1243.90] vol=1.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:30:00 | 1227.46 | 1238.58 | 0.00 | T1 1.5R @ 1227.46 |
| Target hit | 2026-02-16 15:20:00 | 1214.10 | 1222.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 1203.90 | 1206.09 | 0.00 | ORB-short ORB[1204.00,1215.70] vol=2.6x ATR=3.80 |
| Stop hit — per-position SL triggered | 2026-02-19 10:50:00 | 1207.70 | 1205.79 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 1198.90 | 1189.87 | 0.00 | ORB-long ORB[1181.90,1197.60] vol=5.0x ATR=5.27 |
| Stop hit — per-position SL triggered | 2026-02-20 10:10:00 | 1193.63 | 1190.43 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 1184.70 | 1196.71 | 0.00 | ORB-short ORB[1192.00,1207.60] vol=1.7x ATR=3.36 |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 1188.06 | 1192.54 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 1176.60 | 1179.17 | 0.00 | ORB-short ORB[1179.50,1194.40] vol=2.1x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-02-24 11:40:00 | 1179.53 | 1178.92 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2026-03-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:25:00 | 1120.10 | 1129.91 | 0.00 | ORB-short ORB[1130.10,1144.00] vol=2.2x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:50:00 | 1112.82 | 1128.27 | 0.00 | T1 1.5R @ 1112.82 |
| Target hit | 2026-03-05 14:45:00 | 1115.50 | 1114.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 93 — SELL (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 1032.40 | 1042.91 | 0.00 | ORB-short ORB[1045.20,1057.60] vol=1.9x ATR=4.75 |
| Stop hit — per-position SL triggered | 2026-03-13 11:00:00 | 1037.15 | 1041.10 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2026-04-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:25:00 | 1048.70 | 1052.86 | 0.00 | ORB-short ORB[1050.30,1061.00] vol=2.6x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-04-01 10:30:00 | 1053.26 | 1052.97 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2026-04-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:10:00 | 1100.20 | 1106.16 | 0.00 | ORB-short ORB[1103.40,1117.80] vol=1.6x ATR=3.07 |
| Stop hit — per-position SL triggered | 2026-04-16 11:30:00 | 1103.27 | 1105.74 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 1116.70 | 1125.29 | 0.00 | ORB-short ORB[1120.70,1137.20] vol=1.6x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:30:00 | 1110.33 | 1120.15 | 0.00 | T1 1.5R @ 1110.33 |
| Target hit | 2026-04-24 14:20:00 | 1107.70 | 1107.50 | 0.00 | Trail-exit close>VWAP |

### Cycle 97 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 1084.40 | 1097.53 | 0.00 | ORB-short ORB[1097.40,1111.50] vol=1.7x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:20:00 | 1078.58 | 1089.34 | 0.00 | T1 1.5R @ 1078.58 |
| Target hit | 2026-05-05 13:25:00 | 1081.40 | 1079.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 98 — SELL (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 1089.20 | 1093.02 | 0.00 | ORB-short ORB[1090.80,1099.70] vol=2.0x ATR=3.85 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 1093.05 | 1093.17 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2026-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 1169.20 | 1156.12 | 0.00 | ORB-long ORB[1141.40,1154.90] vol=1.8x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:50:00 | 1176.32 | 1158.76 | 0.00 | T1 1.5R @ 1176.32 |
| Target hit | 2026-05-08 15:20:00 | 1177.70 | 1172.44 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-19 09:30:00 | 983.85 | 2025-05-19 09:55:00 | 987.13 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-05-27 09:45:00 | 1004.40 | 2025-05-27 10:15:00 | 998.27 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-05-27 09:45:00 | 1004.40 | 2025-05-27 11:50:00 | 1004.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-28 09:30:00 | 1003.70 | 2025-05-28 09:55:00 | 998.78 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-05-28 09:30:00 | 1003.70 | 2025-05-28 11:00:00 | 1003.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-29 10:50:00 | 1012.45 | 2025-05-29 10:55:00 | 1009.36 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-05-30 09:40:00 | 1017.45 | 2025-05-30 09:45:00 | 1022.81 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-05-30 09:40:00 | 1017.45 | 2025-05-30 09:55:00 | 1019.85 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-06-02 10:55:00 | 1027.00 | 2025-06-02 11:10:00 | 1023.25 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-03 09:35:00 | 1030.50 | 2025-06-03 09:55:00 | 1026.76 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-06-06 10:30:00 | 1095.70 | 2025-06-06 11:05:00 | 1090.75 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-06-12 11:05:00 | 1090.50 | 2025-06-12 11:45:00 | 1086.21 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-12 11:05:00 | 1090.50 | 2025-06-12 15:20:00 | 1072.00 | TARGET_HIT | 0.50 | 1.70% |
| BUY | retest1 | 2025-06-20 09:35:00 | 1051.70 | 2025-06-20 09:40:00 | 1059.02 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-06-20 09:35:00 | 1051.70 | 2025-06-20 09:45:00 | 1051.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-23 10:50:00 | 1044.90 | 2025-06-23 11:10:00 | 1048.86 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-06-26 11:05:00 | 1064.50 | 2025-06-26 11:15:00 | 1060.33 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-26 11:05:00 | 1064.50 | 2025-06-26 11:25:00 | 1064.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 10:50:00 | 1086.10 | 2025-07-01 11:05:00 | 1089.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-02 11:00:00 | 1098.00 | 2025-07-02 12:20:00 | 1094.85 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-03 10:10:00 | 1108.90 | 2025-07-03 10:15:00 | 1113.80 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-07-03 10:10:00 | 1108.90 | 2025-07-03 13:20:00 | 1112.60 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-11 09:45:00 | 1079.00 | 2025-07-11 09:50:00 | 1082.22 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-14 10:20:00 | 1075.60 | 2025-07-14 12:25:00 | 1080.53 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-07-14 10:20:00 | 1075.60 | 2025-07-14 12:30:00 | 1075.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-17 10:10:00 | 1133.20 | 2025-07-17 10:15:00 | 1138.19 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-07-17 10:10:00 | 1133.20 | 2025-07-17 10:50:00 | 1133.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:15:00 | 1098.50 | 2025-07-18 10:35:00 | 1101.48 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-22 10:30:00 | 1071.40 | 2025-07-22 10:35:00 | 1073.82 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-24 10:25:00 | 1086.00 | 2025-07-24 11:20:00 | 1082.88 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-30 09:30:00 | 1053.90 | 2025-07-30 09:35:00 | 1058.53 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-07-31 10:15:00 | 1044.90 | 2025-07-31 10:40:00 | 1048.09 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-06 11:05:00 | 1082.00 | 2025-08-06 11:10:00 | 1085.87 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-08-11 09:45:00 | 1070.40 | 2025-08-11 09:50:00 | 1073.99 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-08-12 10:55:00 | 1076.00 | 2025-08-12 11:15:00 | 1072.15 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-12 10:55:00 | 1076.00 | 2025-08-12 12:05:00 | 1076.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-13 10:15:00 | 1084.60 | 2025-08-13 10:25:00 | 1089.66 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-13 10:15:00 | 1084.60 | 2025-08-13 15:20:00 | 1114.00 | TARGET_HIT | 0.50 | 2.71% |
| BUY | retest1 | 2025-08-19 09:50:00 | 1224.50 | 2025-08-19 10:15:00 | 1230.12 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-08-19 09:50:00 | 1224.50 | 2025-08-19 10:45:00 | 1224.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-21 11:15:00 | 1246.80 | 2025-08-21 11:30:00 | 1244.18 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-08-22 10:50:00 | 1241.30 | 2025-08-22 11:15:00 | 1246.65 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-08-22 10:50:00 | 1241.30 | 2025-08-22 15:20:00 | 1265.50 | TARGET_HIT | 0.50 | 1.95% |
| BUY | retest1 | 2025-08-26 10:35:00 | 1292.00 | 2025-08-26 11:05:00 | 1288.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-28 09:40:00 | 1288.40 | 2025-08-28 09:50:00 | 1294.78 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-08-28 09:40:00 | 1288.40 | 2025-08-28 10:15:00 | 1288.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 10:05:00 | 1308.90 | 2025-09-01 10:20:00 | 1305.02 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-03 09:45:00 | 1296.90 | 2025-09-03 10:35:00 | 1291.06 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-09-03 09:45:00 | 1296.90 | 2025-09-03 15:20:00 | 1282.10 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-09-04 10:55:00 | 1294.00 | 2025-09-04 11:05:00 | 1287.24 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-09-04 10:55:00 | 1294.00 | 2025-09-04 15:20:00 | 1281.70 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2025-09-05 10:05:00 | 1302.70 | 2025-09-05 10:20:00 | 1308.86 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-09-05 10:05:00 | 1302.70 | 2025-09-05 10:45:00 | 1302.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-08 09:35:00 | 1317.00 | 2025-09-08 09:40:00 | 1312.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-09-09 09:35:00 | 1289.90 | 2025-09-09 09:40:00 | 1293.66 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-11 11:05:00 | 1278.50 | 2025-09-11 11:35:00 | 1275.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-12 09:40:00 | 1291.30 | 2025-09-12 09:50:00 | 1288.26 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-15 09:45:00 | 1284.30 | 2025-09-15 10:20:00 | 1287.96 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-16 10:15:00 | 1315.00 | 2025-09-16 10:30:00 | 1311.24 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-17 11:05:00 | 1311.00 | 2025-09-17 12:05:00 | 1313.48 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-09-19 11:15:00 | 1312.30 | 2025-09-19 11:25:00 | 1314.95 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-24 11:15:00 | 1308.70 | 2025-09-24 11:35:00 | 1304.62 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-09-24 11:15:00 | 1308.70 | 2025-09-24 13:15:00 | 1308.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-01 10:45:00 | 1310.50 | 2025-10-01 10:55:00 | 1316.92 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-10-01 10:45:00 | 1310.50 | 2025-10-01 15:20:00 | 1323.60 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2025-10-03 10:40:00 | 1310.90 | 2025-10-03 11:00:00 | 1314.30 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-06 11:10:00 | 1327.80 | 2025-10-06 11:15:00 | 1324.90 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-08 11:00:00 | 1348.70 | 2025-10-08 11:40:00 | 1352.36 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-29 10:20:00 | 1202.10 | 2025-10-29 11:10:00 | 1198.98 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-30 11:15:00 | 1221.10 | 2025-10-30 11:20:00 | 1225.12 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-30 11:15:00 | 1221.10 | 2025-10-30 12:40:00 | 1221.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-07 10:35:00 | 1248.30 | 2025-11-07 10:45:00 | 1242.53 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-11-13 11:15:00 | 1307.80 | 2025-11-13 11:30:00 | 1303.60 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-11-13 11:15:00 | 1307.80 | 2025-11-13 11:35:00 | 1307.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-18 10:05:00 | 1286.70 | 2025-11-18 11:15:00 | 1282.28 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-18 10:05:00 | 1286.70 | 2025-11-18 13:50:00 | 1282.30 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-20 11:10:00 | 1297.30 | 2025-11-20 12:05:00 | 1293.02 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-20 11:10:00 | 1297.30 | 2025-11-20 13:25:00 | 1297.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 11:15:00 | 1293.20 | 2025-11-21 11:45:00 | 1295.81 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-26 09:30:00 | 1292.30 | 2025-11-26 09:35:00 | 1287.96 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-03 10:50:00 | 1270.50 | 2025-12-03 11:20:00 | 1263.81 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-12-03 10:50:00 | 1270.50 | 2025-12-03 11:45:00 | 1270.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-04 10:45:00 | 1276.20 | 2025-12-04 10:55:00 | 1273.17 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-08 10:00:00 | 1254.30 | 2025-12-08 10:10:00 | 1259.33 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-12-09 11:10:00 | 1237.80 | 2025-12-09 11:15:00 | 1241.92 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-11 11:05:00 | 1235.00 | 2025-12-11 14:15:00 | 1231.80 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-15 11:10:00 | 1235.70 | 2025-12-15 11:20:00 | 1238.52 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-17 10:40:00 | 1249.40 | 2025-12-17 10:55:00 | 1244.35 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-12-17 10:40:00 | 1249.40 | 2025-12-17 11:15:00 | 1249.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 10:45:00 | 1241.10 | 2025-12-19 10:50:00 | 1238.32 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-22 09:30:00 | 1281.00 | 2025-12-22 09:40:00 | 1277.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-24 10:55:00 | 1304.40 | 2025-12-24 11:20:00 | 1301.48 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-26 09:30:00 | 1283.80 | 2025-12-26 10:10:00 | 1278.32 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-12-26 09:30:00 | 1283.80 | 2025-12-26 15:20:00 | 1271.90 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2025-12-30 11:15:00 | 1270.10 | 2025-12-30 12:25:00 | 1267.69 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-31 11:00:00 | 1282.30 | 2025-12-31 11:15:00 | 1278.36 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-01-01 11:05:00 | 1288.50 | 2026-01-01 12:05:00 | 1285.81 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-02 10:45:00 | 1297.40 | 2026-01-02 10:50:00 | 1302.26 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-01-02 10:45:00 | 1297.40 | 2026-01-02 12:45:00 | 1308.10 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2026-01-05 10:50:00 | 1345.10 | 2026-01-05 11:00:00 | 1351.18 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-01-05 10:50:00 | 1345.10 | 2026-01-05 11:15:00 | 1345.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-06 10:55:00 | 1320.70 | 2026-01-06 11:10:00 | 1315.15 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-01-06 10:55:00 | 1320.70 | 2026-01-06 13:15:00 | 1320.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-07 10:35:00 | 1328.50 | 2026-01-07 10:50:00 | 1333.16 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-01-07 10:35:00 | 1328.50 | 2026-01-07 11:00:00 | 1328.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 10:15:00 | 1301.20 | 2026-01-08 10:45:00 | 1294.41 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-01-08 10:15:00 | 1301.20 | 2026-01-08 15:20:00 | 1268.20 | TARGET_HIT | 0.50 | 2.54% |
| SELL | retest1 | 2026-01-09 09:45:00 | 1257.80 | 2026-01-09 09:50:00 | 1262.72 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-01-13 09:45:00 | 1202.40 | 2026-01-13 11:15:00 | 1194.56 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-01-13 09:45:00 | 1202.40 | 2026-01-13 14:45:00 | 1199.40 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2026-01-22 09:35:00 | 1160.80 | 2026-01-22 10:25:00 | 1155.45 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-01-28 10:15:00 | 1129.40 | 2026-01-28 10:40:00 | 1133.91 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-01-29 10:55:00 | 1131.30 | 2026-01-29 13:10:00 | 1135.20 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-01-30 10:20:00 | 1149.80 | 2026-01-30 10:30:00 | 1155.14 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-01-30 10:20:00 | 1149.80 | 2026-01-30 11:55:00 | 1159.10 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2026-02-10 10:50:00 | 1220.10 | 2026-02-10 12:15:00 | 1226.23 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-10 10:50:00 | 1220.10 | 2026-02-10 15:20:00 | 1231.00 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2026-02-11 09:35:00 | 1222.10 | 2026-02-11 09:40:00 | 1225.12 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-12 11:05:00 | 1246.10 | 2026-02-12 11:25:00 | 1242.24 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-13 10:05:00 | 1224.90 | 2026-02-13 10:10:00 | 1229.13 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-16 11:15:00 | 1232.20 | 2026-02-16 11:30:00 | 1227.46 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-16 11:15:00 | 1232.20 | 2026-02-16 15:20:00 | 1214.10 | TARGET_HIT | 0.50 | 1.47% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1203.90 | 2026-02-19 10:50:00 | 1207.70 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-20 09:45:00 | 1198.90 | 2026-02-20 10:10:00 | 1193.63 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-23 10:50:00 | 1184.70 | 2026-02-23 11:15:00 | 1188.06 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-24 11:10:00 | 1176.60 | 2026-02-24 11:40:00 | 1179.53 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-05 10:25:00 | 1120.10 | 2026-03-05 10:50:00 | 1112.82 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-05 10:25:00 | 1120.10 | 2026-03-05 14:45:00 | 1115.50 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-13 10:40:00 | 1032.40 | 2026-03-13 11:00:00 | 1037.15 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-01 10:25:00 | 1048.70 | 2026-04-01 10:30:00 | 1053.26 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-16 11:10:00 | 1100.20 | 2026-04-16 11:30:00 | 1103.27 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1116.70 | 2026-04-24 10:30:00 | 1110.33 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1116.70 | 2026-04-24 14:20:00 | 1107.70 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2026-05-05 10:10:00 | 1084.40 | 2026-05-05 10:20:00 | 1078.58 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-05-05 10:10:00 | 1084.40 | 2026-05-05 13:25:00 | 1081.40 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2026-05-06 09:50:00 | 1089.20 | 2026-05-06 10:05:00 | 1093.05 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-08 10:45:00 | 1169.20 | 2026-05-08 10:50:00 | 1176.32 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-05-08 10:45:00 | 1169.20 | 2026-05-08 15:20:00 | 1177.70 | TARGET_HIT | 0.50 | 0.73% |
