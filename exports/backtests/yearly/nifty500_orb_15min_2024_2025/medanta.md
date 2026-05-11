# Global Health Ltd. (MEDANTA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-06-04 15:25:00 (19758 bars)
- **Last close:** 1197.00
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
| ENTRY1 | 51 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 9 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 42
- **Target hits / Stop hits / Partials:** 9 / 42 / 23
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 9.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 11 | 31.4% | 2 | 24 | 9 | -0.03% | -1.1% |
| BUY @ 2nd Alert (retest1) | 35 | 11 | 31.4% | 2 | 24 | 9 | -0.03% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 21 | 53.8% | 7 | 18 | 14 | 0.28% | 11.0% |
| SELL @ 2nd Alert (retest1) | 39 | 21 | 53.8% | 7 | 18 | 14 | 0.28% | 11.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 74 | 32 | 43.2% | 9 | 42 | 23 | 0.13% | 9.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 1398.65 | 1402.59 | 0.00 | ORB-short ORB[1400.00,1417.70] vol=1.6x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:50:00 | 1391.33 | 1398.42 | 0.00 | T1 1.5R @ 1391.33 |
| Target hit | 2024-05-16 12:30:00 | 1396.80 | 1395.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2024-05-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:45:00 | 1239.00 | 1233.67 | 0.00 | ORB-long ORB[1220.20,1232.00] vol=18.4x ATR=5.57 |
| Stop hit — per-position SL triggered | 2024-05-28 09:55:00 | 1233.43 | 1233.75 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:30:00 | 1300.00 | 1306.53 | 0.00 | ORB-short ORB[1306.90,1315.70] vol=3.6x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:35:00 | 1293.36 | 1302.39 | 0.00 | T1 1.5R @ 1293.36 |
| Stop hit — per-position SL triggered | 2024-06-25 09:45:00 | 1300.00 | 1300.47 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:15:00 | 1304.50 | 1293.47 | 0.00 | ORB-long ORB[1285.70,1297.95] vol=1.9x ATR=5.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 13:35:00 | 1312.06 | 1298.64 | 0.00 | T1 1.5R @ 1312.06 |
| Target hit | 2024-06-26 15:20:00 | 1304.55 | 1301.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 09:40:00 | 1300.85 | 1305.52 | 0.00 | ORB-short ORB[1302.80,1313.00] vol=4.2x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 09:50:00 | 1293.47 | 1305.30 | 0.00 | T1 1.5R @ 1293.47 |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 1300.85 | 1297.98 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-07-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:30:00 | 1249.05 | 1257.29 | 0.00 | ORB-short ORB[1256.30,1268.35] vol=5.6x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:35:00 | 1243.68 | 1256.81 | 0.00 | T1 1.5R @ 1243.68 |
| Stop hit — per-position SL triggered | 2024-07-09 11:40:00 | 1249.05 | 1255.34 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 1242.00 | 1253.65 | 0.00 | ORB-short ORB[1257.60,1265.95] vol=1.6x ATR=3.95 |
| Stop hit — per-position SL triggered | 2024-07-10 10:45:00 | 1245.95 | 1253.18 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-07-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:05:00 | 1259.95 | 1259.99 | 0.00 | ORB-short ORB[1260.00,1270.00] vol=11.6x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:30:00 | 1254.85 | 1259.86 | 0.00 | T1 1.5R @ 1254.85 |
| Stop hit — per-position SL triggered | 2024-07-11 11:25:00 | 1259.95 | 1259.67 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:45:00 | 1237.80 | 1246.61 | 0.00 | ORB-short ORB[1247.00,1256.75] vol=1.7x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 09:55:00 | 1231.47 | 1243.24 | 0.00 | T1 1.5R @ 1231.47 |
| Target hit | 2024-07-12 15:20:00 | 1218.60 | 1223.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-07-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-22 11:00:00 | 1199.55 | 1204.46 | 0.00 | ORB-short ORB[1200.10,1211.00] vol=3.5x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 11:55:00 | 1194.99 | 1202.90 | 0.00 | T1 1.5R @ 1194.99 |
| Target hit | 2024-07-22 15:20:00 | 1195.90 | 1198.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 1259.20 | 1257.14 | 0.00 | ORB-long ORB[1246.20,1257.80] vol=15.2x ATR=4.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:35:00 | 1266.66 | 1257.66 | 0.00 | T1 1.5R @ 1266.66 |
| Stop hit — per-position SL triggered | 2024-07-31 10:30:00 | 1259.20 | 1259.30 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-08-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:45:00 | 1066.90 | 1069.76 | 0.00 | ORB-short ORB[1067.30,1079.80] vol=1.6x ATR=2.33 |
| Stop hit — per-position SL triggered | 2024-08-23 10:20:00 | 1069.23 | 1068.95 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1061.20 | 1071.58 | 0.00 | ORB-short ORB[1070.60,1083.95] vol=2.2x ATR=4.41 |
| Stop hit — per-position SL triggered | 2024-08-28 10:10:00 | 1065.61 | 1067.03 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 11:15:00 | 1120.00 | 1127.66 | 0.00 | ORB-short ORB[1122.85,1138.00] vol=1.8x ATR=2.98 |
| Stop hit — per-position SL triggered | 2024-09-18 11:40:00 | 1122.98 | 1125.89 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-09-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:40:00 | 1087.35 | 1088.68 | 0.00 | ORB-short ORB[1093.20,1103.90] vol=13.3x ATR=3.54 |
| Stop hit — per-position SL triggered | 2024-09-24 10:15:00 | 1090.89 | 1088.60 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 10:20:00 | 1053.85 | 1058.26 | 0.00 | ORB-short ORB[1058.45,1068.00] vol=1.7x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:30:00 | 1049.37 | 1056.19 | 0.00 | T1 1.5R @ 1049.37 |
| Target hit | 2024-09-27 15:20:00 | 1025.55 | 1038.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-10-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:35:00 | 1062.25 | 1053.26 | 0.00 | ORB-long ORB[1044.95,1059.60] vol=1.7x ATR=4.71 |
| Stop hit — per-position SL triggered | 2024-10-24 11:10:00 | 1057.54 | 1055.25 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-10-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:40:00 | 1065.80 | 1074.32 | 0.00 | ORB-short ORB[1071.65,1082.70] vol=2.4x ATR=4.60 |
| Stop hit — per-position SL triggered | 2024-10-29 09:45:00 | 1070.40 | 1073.84 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:45:00 | 1098.50 | 1090.12 | 0.00 | ORB-long ORB[1078.05,1091.70] vol=2.0x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:00:00 | 1104.50 | 1094.66 | 0.00 | T1 1.5R @ 1104.50 |
| Stop hit — per-position SL triggered | 2024-10-31 10:10:00 | 1098.50 | 1095.38 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:15:00 | 1077.00 | 1070.62 | 0.00 | ORB-long ORB[1059.00,1072.80] vol=4.2x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 10:35:00 | 1084.06 | 1071.89 | 0.00 | T1 1.5R @ 1084.06 |
| Stop hit — per-position SL triggered | 2024-11-11 12:00:00 | 1077.00 | 1073.85 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 1035.50 | 1043.83 | 0.00 | ORB-short ORB[1042.00,1054.80] vol=2.1x ATR=6.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 1025.31 | 1040.72 | 0.00 | T1 1.5R @ 1025.31 |
| Target hit | 2024-11-13 10:30:00 | 1033.40 | 1032.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 22 — BUY (started 2024-11-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:55:00 | 1089.90 | 1081.01 | 0.00 | ORB-long ORB[1073.25,1088.00] vol=2.2x ATR=4.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 13:50:00 | 1097.07 | 1085.68 | 0.00 | T1 1.5R @ 1097.07 |
| Target hit | 2024-11-19 15:20:00 | 1093.80 | 1089.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2024-11-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:05:00 | 1083.70 | 1078.14 | 0.00 | ORB-long ORB[1071.05,1083.00] vol=1.6x ATR=3.53 |
| Stop hit — per-position SL triggered | 2024-11-29 10:10:00 | 1080.17 | 1078.22 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:15:00 | 1169.60 | 1161.84 | 0.00 | ORB-long ORB[1154.85,1165.00] vol=4.1x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:35:00 | 1177.70 | 1166.25 | 0.00 | T1 1.5R @ 1177.70 |
| Stop hit — per-position SL triggered | 2024-12-06 11:25:00 | 1169.60 | 1170.51 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-12-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 09:55:00 | 1172.35 | 1185.43 | 0.00 | ORB-short ORB[1179.35,1192.45] vol=1.7x ATR=5.27 |
| Stop hit — per-position SL triggered | 2024-12-10 10:05:00 | 1177.62 | 1182.54 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-12-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:10:00 | 1158.00 | 1145.31 | 0.00 | ORB-long ORB[1136.90,1151.15] vol=2.6x ATR=4.54 |
| Stop hit — per-position SL triggered | 2024-12-11 10:15:00 | 1153.46 | 1147.73 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-12-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:10:00 | 1115.30 | 1119.03 | 0.00 | ORB-short ORB[1127.15,1141.00] vol=7.8x ATR=5.90 |
| Stop hit — per-position SL triggered | 2024-12-12 10:25:00 | 1121.20 | 1118.90 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-12-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 10:40:00 | 1126.85 | 1118.14 | 0.00 | ORB-long ORB[1108.00,1118.55] vol=3.8x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 10:50:00 | 1133.22 | 1120.82 | 0.00 | T1 1.5R @ 1133.22 |
| Stop hit — per-position SL triggered | 2024-12-18 12:20:00 | 1126.85 | 1129.07 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-12-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:45:00 | 1099.65 | 1093.53 | 0.00 | ORB-long ORB[1085.00,1097.75] vol=1.6x ATR=4.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 11:20:00 | 1106.39 | 1098.91 | 0.00 | T1 1.5R @ 1106.39 |
| Stop hit — per-position SL triggered | 2024-12-24 12:15:00 | 1099.65 | 1099.33 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:50:00 | 1091.95 | 1096.62 | 0.00 | ORB-short ORB[1093.00,1103.00] vol=1.9x ATR=4.23 |
| Stop hit — per-position SL triggered | 2024-12-26 09:55:00 | 1096.18 | 1096.44 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:20:00 | 1119.50 | 1114.07 | 0.00 | ORB-long ORB[1105.00,1117.70] vol=6.6x ATR=3.48 |
| Stop hit — per-position SL triggered | 2024-12-27 10:25:00 | 1116.02 | 1115.92 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-01-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:55:00 | 1101.70 | 1092.56 | 0.00 | ORB-long ORB[1078.50,1093.90] vol=1.8x ATR=3.80 |
| Stop hit — per-position SL triggered | 2025-01-01 15:20:00 | 1099.25 | 1098.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:50:00 | 1082.05 | 1086.58 | 0.00 | ORB-short ORB[1082.50,1094.80] vol=2.0x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:00:00 | 1076.84 | 1085.42 | 0.00 | T1 1.5R @ 1076.84 |
| Target hit | 2025-01-06 15:20:00 | 1070.10 | 1074.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-01-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 09:45:00 | 1061.50 | 1071.13 | 0.00 | ORB-short ORB[1067.40,1081.00] vol=1.7x ATR=5.22 |
| Stop hit — per-position SL triggered | 2025-01-07 10:20:00 | 1066.72 | 1067.31 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-01-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 10:00:00 | 1058.35 | 1063.49 | 0.00 | ORB-short ORB[1060.00,1073.95] vol=1.9x ATR=5.36 |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 1063.71 | 1062.84 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:10:00 | 1031.75 | 1029.59 | 0.00 | ORB-long ORB[1018.75,1029.55] vol=1.6x ATR=3.21 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 1028.54 | 1029.55 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-01-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 10:55:00 | 1048.55 | 1051.92 | 0.00 | ORB-short ORB[1049.05,1059.90] vol=3.1x ATR=5.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:10:00 | 1041.00 | 1051.67 | 0.00 | T1 1.5R @ 1041.00 |
| Stop hit — per-position SL triggered | 2025-01-31 11:15:00 | 1048.55 | 1049.78 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-02-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 11:10:00 | 1170.40 | 1160.56 | 0.00 | ORB-long ORB[1152.20,1168.00] vol=2.0x ATR=5.45 |
| Stop hit — per-position SL triggered | 2025-02-07 11:15:00 | 1164.95 | 1161.09 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-02-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 10:00:00 | 1153.85 | 1142.16 | 0.00 | ORB-long ORB[1114.50,1129.70] vol=1.8x ATR=7.22 |
| Stop hit — per-position SL triggered | 2025-02-19 10:55:00 | 1146.63 | 1148.94 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-02-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 10:55:00 | 1207.70 | 1196.78 | 0.00 | ORB-long ORB[1189.90,1207.35] vol=2.9x ATR=6.64 |
| Stop hit — per-position SL triggered | 2025-02-25 11:05:00 | 1201.06 | 1197.31 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:45:00 | 1224.00 | 1214.62 | 0.00 | ORB-long ORB[1202.10,1219.00] vol=3.6x ATR=6.04 |
| Stop hit — per-position SL triggered | 2025-03-05 09:50:00 | 1217.96 | 1216.00 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:00:00 | 1192.75 | 1179.02 | 0.00 | ORB-long ORB[1163.00,1173.70] vol=6.1x ATR=4.31 |
| Stop hit — per-position SL triggered | 2025-03-13 10:05:00 | 1188.44 | 1180.62 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:30:00 | 1247.20 | 1242.19 | 0.00 | ORB-long ORB[1231.65,1245.00] vol=2.0x ATR=4.52 |
| Stop hit — per-position SL triggered | 2025-03-20 09:35:00 | 1242.68 | 1242.73 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-04-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:20:00 | 1256.25 | 1243.06 | 0.00 | ORB-long ORB[1235.95,1251.45] vol=2.4x ATR=5.61 |
| Stop hit — per-position SL triggered | 2025-04-02 10:35:00 | 1250.64 | 1244.22 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 09:50:00 | 1214.00 | 1227.50 | 0.00 | ORB-short ORB[1223.00,1240.00] vol=2.8x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 10:25:00 | 1204.62 | 1219.99 | 0.00 | T1 1.5R @ 1204.62 |
| Stop hit — per-position SL triggered | 2025-04-04 10:35:00 | 1214.00 | 1219.35 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-04-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 09:35:00 | 1239.45 | 1227.31 | 0.00 | ORB-long ORB[1217.35,1235.00] vol=2.0x ATR=7.57 |
| Stop hit — per-position SL triggered | 2025-04-08 09:40:00 | 1231.88 | 1230.27 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-04-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-11 09:30:00 | 1238.55 | 1246.18 | 0.00 | ORB-short ORB[1240.65,1257.75] vol=3.2x ATR=6.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-11 09:55:00 | 1228.90 | 1243.75 | 0.00 | T1 1.5R @ 1228.90 |
| Stop hit — per-position SL triggered | 2025-04-11 10:20:00 | 1238.55 | 1239.33 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-04-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:35:00 | 1284.20 | 1287.51 | 0.00 | ORB-short ORB[1287.00,1298.70] vol=3.2x ATR=4.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:30:00 | 1277.20 | 1286.36 | 0.00 | T1 1.5R @ 1277.20 |
| Target hit | 2025-04-17 15:20:00 | 1266.90 | 1274.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2025-04-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 11:00:00 | 1267.60 | 1261.94 | 0.00 | ORB-long ORB[1254.00,1264.90] vol=3.7x ATR=4.25 |
| Stop hit — per-position SL triggered | 2025-04-22 11:05:00 | 1263.35 | 1262.19 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:35:00 | 1189.70 | 1183.96 | 0.00 | ORB-long ORB[1171.50,1189.00] vol=1.8x ATR=5.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 09:50:00 | 1197.57 | 1185.51 | 0.00 | T1 1.5R @ 1197.57 |
| Stop hit — per-position SL triggered | 2025-04-30 10:00:00 | 1189.70 | 1186.42 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:00:00 | 1217.90 | 1213.90 | 0.00 | ORB-long ORB[1195.00,1211.60] vol=3.9x ATR=3.77 |
| Stop hit — per-position SL triggered | 2025-05-05 11:40:00 | 1214.13 | 1214.27 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:30:00 | 1398.65 | 2024-05-16 10:50:00 | 1391.33 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-05-16 09:30:00 | 1398.65 | 2024-05-16 12:30:00 | 1396.80 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-05-28 09:45:00 | 1239.00 | 2024-05-28 09:55:00 | 1233.43 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-06-25 09:30:00 | 1300.00 | 2024-06-25 09:35:00 | 1293.36 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-06-25 09:30:00 | 1300.00 | 2024-06-25 09:45:00 | 1300.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 11:15:00 | 1304.50 | 2024-06-26 13:35:00 | 1312.06 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-06-26 11:15:00 | 1304.50 | 2024-06-26 15:20:00 | 1304.55 | TARGET_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 09:40:00 | 1300.85 | 2024-06-27 09:50:00 | 1293.47 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-06-27 09:40:00 | 1300.85 | 2024-06-27 11:15:00 | 1300.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-09 10:30:00 | 1249.05 | 2024-07-09 10:35:00 | 1243.68 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-07-09 10:30:00 | 1249.05 | 2024-07-09 11:40:00 | 1249.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 10:35:00 | 1242.00 | 2024-07-10 10:45:00 | 1245.95 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-11 10:05:00 | 1259.95 | 2024-07-11 10:30:00 | 1254.85 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-11 10:05:00 | 1259.95 | 2024-07-11 11:25:00 | 1259.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 09:45:00 | 1237.80 | 2024-07-12 09:55:00 | 1231.47 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-12 09:45:00 | 1237.80 | 2024-07-12 15:20:00 | 1218.60 | TARGET_HIT | 0.50 | 1.55% |
| SELL | retest1 | 2024-07-22 11:00:00 | 1199.55 | 2024-07-22 11:55:00 | 1194.99 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-07-22 11:00:00 | 1199.55 | 2024-07-22 15:20:00 | 1195.90 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2024-07-31 09:30:00 | 1259.20 | 2024-07-31 09:35:00 | 1266.66 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-07-31 09:30:00 | 1259.20 | 2024-07-31 10:30:00 | 1259.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-23 09:45:00 | 1066.90 | 2024-08-23 10:20:00 | 1069.23 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1061.20 | 2024-08-28 10:10:00 | 1065.61 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-09-18 11:15:00 | 1120.00 | 2024-09-18 11:40:00 | 1122.98 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-24 09:40:00 | 1087.35 | 2024-09-24 10:15:00 | 1090.89 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-27 10:20:00 | 1053.85 | 2024-09-27 10:30:00 | 1049.37 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-27 10:20:00 | 1053.85 | 2024-09-27 15:20:00 | 1025.55 | TARGET_HIT | 0.50 | 2.69% |
| BUY | retest1 | 2024-10-24 10:35:00 | 1062.25 | 2024-10-24 11:10:00 | 1057.54 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-10-29 09:40:00 | 1065.80 | 2024-10-29 09:45:00 | 1070.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-10-31 09:45:00 | 1098.50 | 2024-10-31 10:00:00 | 1104.50 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-10-31 09:45:00 | 1098.50 | 2024-10-31 10:10:00 | 1098.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 10:15:00 | 1077.00 | 2024-11-11 10:35:00 | 1084.06 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-11-11 10:15:00 | 1077.00 | 2024-11-11 12:00:00 | 1077.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 09:30:00 | 1035.50 | 2024-11-13 09:40:00 | 1025.31 | PARTIAL | 0.50 | 0.98% |
| SELL | retest1 | 2024-11-13 09:30:00 | 1035.50 | 2024-11-13 10:30:00 | 1033.40 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-11-19 10:55:00 | 1089.90 | 2024-11-19 13:50:00 | 1097.07 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-11-19 10:55:00 | 1089.90 | 2024-11-19 15:20:00 | 1093.80 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2024-11-29 10:05:00 | 1083.70 | 2024-11-29 10:10:00 | 1080.17 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-06 10:15:00 | 1169.60 | 2024-12-06 10:35:00 | 1177.70 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-12-06 10:15:00 | 1169.60 | 2024-12-06 11:25:00 | 1169.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-10 09:55:00 | 1172.35 | 2024-12-10 10:05:00 | 1177.62 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-12-11 10:10:00 | 1158.00 | 2024-12-11 10:15:00 | 1153.46 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-12 10:10:00 | 1115.30 | 2024-12-12 10:25:00 | 1121.20 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-12-18 10:40:00 | 1126.85 | 2024-12-18 10:50:00 | 1133.22 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-12-18 10:40:00 | 1126.85 | 2024-12-18 12:20:00 | 1126.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 09:45:00 | 1099.65 | 2024-12-24 11:20:00 | 1106.39 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-12-24 09:45:00 | 1099.65 | 2024-12-24 12:15:00 | 1099.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 09:50:00 | 1091.95 | 2024-12-26 09:55:00 | 1096.18 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-27 10:20:00 | 1119.50 | 2024-12-27 10:25:00 | 1116.02 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-01 10:55:00 | 1101.70 | 2025-01-01 15:20:00 | 1099.25 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1082.05 | 2025-01-06 11:00:00 | 1076.84 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1082.05 | 2025-01-06 15:20:00 | 1070.10 | TARGET_HIT | 0.50 | 1.10% |
| SELL | retest1 | 2025-01-07 09:45:00 | 1061.50 | 2025-01-07 10:20:00 | 1066.72 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-01-10 10:00:00 | 1058.35 | 2025-01-10 10:15:00 | 1063.71 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-01-29 11:10:00 | 1031.75 | 2025-01-29 11:20:00 | 1028.54 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-31 10:55:00 | 1048.55 | 2025-01-31 11:10:00 | 1041.00 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2025-01-31 10:55:00 | 1048.55 | 2025-01-31 11:15:00 | 1048.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 11:10:00 | 1170.40 | 2025-02-07 11:15:00 | 1164.95 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-02-19 10:00:00 | 1153.85 | 2025-02-19 10:55:00 | 1146.63 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2025-02-25 10:55:00 | 1207.70 | 2025-02-25 11:05:00 | 1201.06 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-03-05 09:45:00 | 1224.00 | 2025-03-05 09:50:00 | 1217.96 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-03-13 10:00:00 | 1192.75 | 2025-03-13 10:05:00 | 1188.44 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-20 09:30:00 | 1247.20 | 2025-03-20 09:35:00 | 1242.68 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-02 10:20:00 | 1256.25 | 2025-04-02 10:35:00 | 1250.64 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-04-04 09:50:00 | 1214.00 | 2025-04-04 10:25:00 | 1204.62 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2025-04-04 09:50:00 | 1214.00 | 2025-04-04 10:35:00 | 1214.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-08 09:35:00 | 1239.45 | 2025-04-08 09:40:00 | 1231.88 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-04-11 09:30:00 | 1238.55 | 2025-04-11 09:55:00 | 1228.90 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2025-04-11 09:30:00 | 1238.55 | 2025-04-11 10:20:00 | 1238.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-17 10:35:00 | 1284.20 | 2025-04-17 11:30:00 | 1277.20 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-04-17 10:35:00 | 1284.20 | 2025-04-17 15:20:00 | 1266.90 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2025-04-22 11:00:00 | 1267.60 | 2025-04-22 11:05:00 | 1263.35 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-30 09:35:00 | 1189.70 | 2025-04-30 09:50:00 | 1197.57 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-04-30 09:35:00 | 1189.70 | 2025-04-30 10:00:00 | 1189.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 11:00:00 | 1217.90 | 2025-05-05 11:40:00 | 1214.13 | STOP_HIT | 1.00 | -0.31% |
