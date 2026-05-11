# Anupam Rasayan India Ltd. (ANURAS)

## Backtest Summary

- **Window:** 2025-07-10 09:15:00 → 2026-05-08 15:25:00 (15236 bars)
- **Last close:** 1369.00
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
| ENTRY1 | 62 |
| ENTRY2 | 0 |
| PARTIAL | 15 |
| TARGET_HIT | 6 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 56
- **Target hits / Stop hits / Partials:** 6 / 56 / 15
- **Avg / median % per leg:** -0.02% / -0.24%
- **Sum % (uncompounded):** -1.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 15 | 36.6% | 5 | 26 | 10 | 0.10% | 4.2% |
| BUY @ 2nd Alert (retest1) | 41 | 15 | 36.6% | 5 | 26 | 10 | 0.10% | 4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 36 | 6 | 16.7% | 1 | 30 | 5 | -0.16% | -5.7% |
| SELL @ 2nd Alert (retest1) | 36 | 6 | 16.7% | 1 | 30 | 5 | -0.16% | -5.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 77 | 21 | 27.3% | 6 | 56 | 15 | -0.02% | -1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 10:55:00 | 1158.30 | 1152.63 | 0.00 | ORB-long ORB[1143.30,1154.00] vol=1.5x ATR=4.45 |
| Stop hit — per-position SL triggered | 2025-07-10 12:05:00 | 1153.85 | 1155.48 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-07-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:10:00 | 1155.80 | 1150.95 | 0.00 | ORB-long ORB[1140.10,1154.00] vol=3.5x ATR=3.37 |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1152.43 | 1151.20 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:30:00 | 1146.00 | 1147.96 | 0.00 | ORB-short ORB[1146.10,1152.80] vol=1.9x ATR=2.76 |
| Stop hit — per-position SL triggered | 2025-07-18 09:45:00 | 1148.76 | 1147.38 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 11:15:00 | 1153.20 | 1150.54 | 0.00 | ORB-long ORB[1145.00,1152.50] vol=3.9x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 11:20:00 | 1155.85 | 1150.87 | 0.00 | T1 1.5R @ 1155.85 |
| Stop hit — per-position SL triggered | 2025-07-21 13:50:00 | 1153.20 | 1153.38 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-07-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:50:00 | 1131.60 | 1129.64 | 0.00 | ORB-long ORB[1121.10,1130.20] vol=2.5x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-07-29 11:05:00 | 1128.87 | 1129.75 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-08-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:05:00 | 1129.10 | 1135.45 | 0.00 | ORB-short ORB[1136.00,1147.20] vol=4.7x ATR=2.70 |
| Stop hit — per-position SL triggered | 2025-08-06 11:15:00 | 1131.80 | 1134.76 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:30:00 | 1150.00 | 1145.09 | 0.00 | ORB-long ORB[1135.10,1148.50] vol=1.8x ATR=3.78 |
| Stop hit — per-position SL triggered | 2025-08-13 09:35:00 | 1146.22 | 1145.49 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-08-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 10:05:00 | 1135.90 | 1141.62 | 0.00 | ORB-short ORB[1143.60,1152.40] vol=4.4x ATR=3.33 |
| Stop hit — per-position SL triggered | 2025-08-21 10:15:00 | 1139.23 | 1141.47 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-08-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:50:00 | 1118.80 | 1120.21 | 0.00 | ORB-short ORB[1120.10,1132.40] vol=1.8x ATR=4.28 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 1123.08 | 1119.42 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-09-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 11:10:00 | 1089.20 | 1095.93 | 0.00 | ORB-short ORB[1093.50,1101.20] vol=2.5x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 1091.61 | 1095.75 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-09-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:35:00 | 1085.20 | 1082.12 | 0.00 | ORB-long ORB[1069.40,1078.70] vol=3.8x ATR=4.33 |
| Stop hit — per-position SL triggered | 2025-09-22 13:55:00 | 1080.87 | 1083.40 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-09-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:35:00 | 1050.00 | 1057.91 | 0.00 | ORB-short ORB[1059.60,1073.70] vol=3.1x ATR=5.21 |
| Stop hit — per-position SL triggered | 2025-09-30 11:20:00 | 1055.21 | 1056.89 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:15:00 | 1078.80 | 1085.00 | 0.00 | ORB-short ORB[1080.50,1092.80] vol=1.6x ATR=5.30 |
| Stop hit — per-position SL triggered | 2025-10-06 12:15:00 | 1084.10 | 1082.63 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 1081.00 | 1077.06 | 0.00 | ORB-long ORB[1072.80,1079.80] vol=1.9x ATR=3.28 |
| Stop hit — per-position SL triggered | 2025-10-07 09:55:00 | 1077.72 | 1079.26 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-10-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 09:50:00 | 1090.10 | 1093.90 | 0.00 | ORB-short ORB[1093.00,1106.90] vol=3.1x ATR=4.08 |
| Stop hit — per-position SL triggered | 2025-10-10 12:00:00 | 1094.18 | 1090.66 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-10-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:25:00 | 1102.30 | 1096.76 | 0.00 | ORB-long ORB[1087.00,1100.50] vol=2.7x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:00:00 | 1107.94 | 1098.99 | 0.00 | T1 1.5R @ 1107.94 |
| Target hit | 2025-10-15 15:20:00 | 1127.20 | 1121.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-10-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:40:00 | 1124.80 | 1141.82 | 0.00 | ORB-short ORB[1125.00,1139.80] vol=2.0x ATR=7.35 |
| Stop hit — per-position SL triggered | 2025-10-16 11:00:00 | 1132.15 | 1140.27 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-10-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:10:00 | 1074.40 | 1075.14 | 0.00 | ORB-short ORB[1075.30,1087.90] vol=4.3x ATR=3.75 |
| Stop hit — per-position SL triggered | 2025-10-24 10:30:00 | 1078.15 | 1075.79 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-10-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:45:00 | 1097.60 | 1090.98 | 0.00 | ORB-long ORB[1081.60,1094.90] vol=2.5x ATR=4.25 |
| Stop hit — per-position SL triggered | 2025-10-27 12:00:00 | 1093.35 | 1092.75 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 11:00:00 | 1096.40 | 1094.50 | 0.00 | ORB-long ORB[1088.40,1095.90] vol=2.4x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-10-28 11:25:00 | 1094.13 | 1094.03 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-10-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 11:05:00 | 1086.20 | 1091.71 | 0.00 | ORB-short ORB[1089.80,1097.90] vol=2.4x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-10-31 11:45:00 | 1088.39 | 1090.26 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-11-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:05:00 | 1068.10 | 1073.55 | 0.00 | ORB-short ORB[1072.00,1087.00] vol=2.3x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:35:00 | 1064.50 | 1071.85 | 0.00 | T1 1.5R @ 1064.50 |
| Stop hit — per-position SL triggered | 2025-11-04 12:35:00 | 1068.10 | 1071.16 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-11-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:40:00 | 1067.30 | 1070.62 | 0.00 | ORB-short ORB[1070.80,1078.30] vol=1.7x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-11-14 11:30:00 | 1069.30 | 1070.34 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-11-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:45:00 | 1218.00 | 1226.96 | 0.00 | ORB-short ORB[1226.90,1244.00] vol=2.1x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:40:00 | 1212.23 | 1224.71 | 0.00 | T1 1.5R @ 1212.23 |
| Stop hit — per-position SL triggered | 2025-11-27 11:50:00 | 1218.00 | 1223.93 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-12-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 11:05:00 | 1217.00 | 1212.90 | 0.00 | ORB-long ORB[1205.10,1215.00] vol=2.6x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 12:20:00 | 1221.48 | 1215.82 | 0.00 | T1 1.5R @ 1221.48 |
| Target hit | 2025-12-03 15:20:00 | 1251.80 | 1235.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-12-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:45:00 | 1263.10 | 1254.55 | 0.00 | ORB-long ORB[1238.00,1256.00] vol=2.1x ATR=5.47 |
| Stop hit — per-position SL triggered | 2025-12-05 11:00:00 | 1257.63 | 1255.53 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-12-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 11:00:00 | 1276.50 | 1285.96 | 0.00 | ORB-short ORB[1288.10,1303.10] vol=5.1x ATR=4.67 |
| Stop hit — per-position SL triggered | 2025-12-11 11:10:00 | 1281.17 | 1284.21 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-12-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:55:00 | 1294.20 | 1285.83 | 0.00 | ORB-long ORB[1277.10,1287.00] vol=2.3x ATR=3.76 |
| Stop hit — per-position SL triggered | 2025-12-12 11:45:00 | 1290.44 | 1286.40 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:15:00 | 1291.50 | 1298.95 | 0.00 | ORB-short ORB[1295.50,1312.70] vol=3.7x ATR=3.92 |
| Stop hit — per-position SL triggered | 2025-12-15 11:55:00 | 1295.42 | 1297.17 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:30:00 | 1327.40 | 1322.23 | 0.00 | ORB-long ORB[1316.20,1324.60] vol=1.6x ATR=5.65 |
| Stop hit — per-position SL triggered | 2025-12-17 09:40:00 | 1321.75 | 1322.89 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-12-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:50:00 | 1329.30 | 1325.40 | 0.00 | ORB-long ORB[1318.30,1328.00] vol=6.2x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 11:45:00 | 1333.74 | 1328.48 | 0.00 | T1 1.5R @ 1333.74 |
| Stop hit — per-position SL triggered | 2025-12-18 12:05:00 | 1329.30 | 1329.02 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-12-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 10:45:00 | 1311.50 | 1314.20 | 0.00 | ORB-short ORB[1312.40,1329.20] vol=4.5x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 1315.04 | 1313.27 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-12-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:00:00 | 1318.10 | 1315.97 | 0.00 | ORB-long ORB[1303.30,1313.40] vol=3.0x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-12-30 10:40:00 | 1314.26 | 1316.60 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 1324.60 | 1318.67 | 0.00 | ORB-long ORB[1313.30,1319.90] vol=2.2x ATR=2.91 |
| Stop hit — per-position SL triggered | 2025-12-31 11:20:00 | 1321.69 | 1319.04 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2026-01-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:40:00 | 1314.80 | 1318.15 | 0.00 | ORB-short ORB[1318.10,1324.80] vol=2.8x ATR=2.18 |
| Stop hit — per-position SL triggered | 2026-01-01 10:45:00 | 1316.98 | 1317.58 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2026-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:50:00 | 1330.30 | 1323.22 | 0.00 | ORB-long ORB[1319.00,1326.50] vol=2.6x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:00:00 | 1334.09 | 1325.27 | 0.00 | T1 1.5R @ 1334.09 |
| Target hit | 2026-01-06 15:20:00 | 1352.00 | 1352.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1328.10 | 1332.45 | 0.00 | ORB-short ORB[1329.10,1339.20] vol=1.9x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:45:00 | 1321.52 | 1330.65 | 0.00 | T1 1.5R @ 1321.52 |
| Target hit | 2026-01-08 15:20:00 | 1320.20 | 1325.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2026-01-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 11:10:00 | 1330.60 | 1328.54 | 0.00 | ORB-long ORB[1315.00,1324.90] vol=8.8x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:40:00 | 1335.32 | 1330.58 | 0.00 | T1 1.5R @ 1335.32 |
| Target hit | 2026-01-13 13:40:00 | 1332.30 | 1332.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2026-01-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:40:00 | 1321.20 | 1330.99 | 0.00 | ORB-short ORB[1330.60,1339.90] vol=2.2x ATR=2.69 |
| Stop hit — per-position SL triggered | 2026-01-14 11:25:00 | 1323.89 | 1326.99 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-01-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:45:00 | 1228.30 | 1219.01 | 0.00 | ORB-long ORB[1210.10,1216.10] vol=3.6x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 10:50:00 | 1231.82 | 1219.82 | 0.00 | T1 1.5R @ 1231.82 |
| Stop hit — per-position SL triggered | 2026-01-30 11:00:00 | 1228.30 | 1222.22 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-02-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:50:00 | 1257.40 | 1240.94 | 0.00 | ORB-long ORB[1228.70,1245.30] vol=1.7x ATR=4.18 |
| Stop hit — per-position SL triggered | 2026-02-01 11:10:00 | 1253.22 | 1254.76 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 1338.80 | 1327.75 | 0.00 | ORB-long ORB[1315.30,1326.20] vol=2.0x ATR=6.04 |
| Stop hit — per-position SL triggered | 2026-02-09 10:45:00 | 1332.76 | 1330.27 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 1337.30 | 1330.13 | 0.00 | ORB-long ORB[1315.70,1333.90] vol=3.5x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 1344.38 | 1333.11 | 0.00 | T1 1.5R @ 1344.38 |
| Stop hit — per-position SL triggered | 2026-02-10 09:55:00 | 1337.30 | 1334.01 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2026-02-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:20:00 | 1333.40 | 1323.97 | 0.00 | ORB-long ORB[1311.10,1329.00] vol=1.7x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:30:00 | 1339.82 | 1325.77 | 0.00 | T1 1.5R @ 1339.82 |
| Stop hit — per-position SL triggered | 2026-02-11 10:40:00 | 1333.40 | 1328.39 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-02-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:10:00 | 1367.60 | 1356.03 | 0.00 | ORB-long ORB[1344.10,1362.80] vol=2.1x ATR=4.89 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1362.71 | 1356.78 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:15:00 | 1340.70 | 1351.57 | 0.00 | ORB-short ORB[1348.70,1368.20] vol=2.1x ATR=6.31 |
| Stop hit — per-position SL triggered | 2026-02-13 11:10:00 | 1347.01 | 1349.71 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 1282.90 | 1276.88 | 0.00 | ORB-long ORB[1262.00,1279.00] vol=2.1x ATR=3.30 |
| Stop hit — per-position SL triggered | 2026-02-18 11:05:00 | 1279.60 | 1277.01 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 1305.70 | 1315.73 | 0.00 | ORB-short ORB[1311.50,1329.10] vol=1.9x ATR=5.27 |
| Stop hit — per-position SL triggered | 2026-02-20 10:50:00 | 1310.97 | 1315.02 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-02-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:10:00 | 1257.00 | 1258.53 | 0.00 | ORB-short ORB[1261.90,1269.50] vol=4.6x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-02-26 10:40:00 | 1260.24 | 1259.41 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 1239.70 | 1245.54 | 0.00 | ORB-short ORB[1246.00,1253.60] vol=1.8x ATR=2.96 |
| Stop hit — per-position SL triggered | 2026-02-27 10:50:00 | 1242.66 | 1245.07 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1225.50 | 1235.47 | 0.00 | ORB-short ORB[1237.30,1251.50] vol=2.1x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:55:00 | 1220.41 | 1232.77 | 0.00 | T1 1.5R @ 1220.41 |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 1225.50 | 1229.84 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-03-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:50:00 | 1214.00 | 1227.44 | 0.00 | ORB-short ORB[1219.70,1230.10] vol=2.3x ATR=5.04 |
| Stop hit — per-position SL triggered | 2026-03-16 11:30:00 | 1219.04 | 1222.61 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-03-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:55:00 | 1247.80 | 1241.82 | 0.00 | ORB-long ORB[1227.90,1246.20] vol=3.1x ATR=3.55 |
| Stop hit — per-position SL triggered | 2026-03-17 11:20:00 | 1244.25 | 1242.30 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-03-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:50:00 | 1223.10 | 1225.15 | 0.00 | ORB-short ORB[1227.10,1244.40] vol=1.6x ATR=3.66 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 1226.76 | 1225.21 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-04-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:40:00 | 1229.00 | 1233.49 | 0.00 | ORB-short ORB[1230.00,1243.00] vol=1.7x ATR=4.99 |
| Stop hit — per-position SL triggered | 2026-04-07 09:55:00 | 1233.99 | 1233.07 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 1297.10 | 1293.34 | 0.00 | ORB-long ORB[1281.40,1296.00] vol=2.2x ATR=4.10 |
| Stop hit — per-position SL triggered | 2026-04-15 09:50:00 | 1293.00 | 1293.36 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-04-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:25:00 | 1277.40 | 1279.17 | 0.00 | ORB-short ORB[1285.10,1294.90] vol=2.7x ATR=3.35 |
| Stop hit — per-position SL triggered | 2026-04-16 10:30:00 | 1280.75 | 1279.18 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1330.70 | 1321.53 | 0.00 | ORB-long ORB[1311.10,1321.90] vol=5.2x ATR=4.44 |
| Stop hit — per-position SL triggered | 2026-04-21 09:45:00 | 1326.26 | 1323.10 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-04-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:05:00 | 1343.10 | 1336.13 | 0.00 | ORB-long ORB[1315.10,1329.90] vol=3.7x ATR=4.44 |
| Target hit | 2026-04-22 15:20:00 | 1344.00 | 1341.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2026-04-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:05:00 | 1362.30 | 1357.20 | 0.00 | ORB-long ORB[1345.60,1359.00] vol=2.9x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:20:00 | 1367.60 | 1359.68 | 0.00 | T1 1.5R @ 1367.60 |
| Stop hit — per-position SL triggered | 2026-04-23 10:50:00 | 1362.30 | 1360.40 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-04-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:10:00 | 1343.70 | 1345.86 | 0.00 | ORB-short ORB[1344.90,1364.40] vol=3.1x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:05:00 | 1336.79 | 1344.89 | 0.00 | T1 1.5R @ 1336.79 |
| Stop hit — per-position SL triggered | 2026-04-27 13:00:00 | 1343.70 | 1338.60 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:50:00 | 1304.00 | 1309.56 | 0.00 | ORB-short ORB[1306.20,1320.00] vol=3.1x ATR=4.84 |
| Stop hit — per-position SL triggered | 2026-04-29 12:10:00 | 1308.84 | 1308.25 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-10 10:55:00 | 1158.30 | 2025-07-10 12:05:00 | 1153.85 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-07-15 10:10:00 | 1155.80 | 2025-07-15 10:15:00 | 1152.43 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-18 09:30:00 | 1146.00 | 2025-07-18 09:45:00 | 1148.76 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-21 11:15:00 | 1153.20 | 2025-07-21 11:20:00 | 1155.85 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-07-21 11:15:00 | 1153.20 | 2025-07-21 13:50:00 | 1153.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-29 10:50:00 | 1131.60 | 2025-07-29 11:05:00 | 1128.87 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-06 11:05:00 | 1129.10 | 2025-08-06 11:15:00 | 1131.80 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-13 09:30:00 | 1150.00 | 2025-08-13 09:35:00 | 1146.22 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-08-21 10:05:00 | 1135.90 | 2025-08-21 10:15:00 | 1139.23 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-29 09:50:00 | 1118.80 | 2025-08-29 10:15:00 | 1123.08 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-09-16 11:10:00 | 1089.20 | 2025-09-16 11:15:00 | 1091.61 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-22 09:35:00 | 1085.20 | 2025-09-22 13:55:00 | 1080.87 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-09-30 10:35:00 | 1050.00 | 2025-09-30 11:20:00 | 1055.21 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-10-06 10:15:00 | 1078.80 | 2025-10-06 12:15:00 | 1084.10 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-10-07 09:30:00 | 1081.00 | 2025-10-07 09:55:00 | 1077.72 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-10 09:50:00 | 1090.10 | 2025-10-10 12:00:00 | 1094.18 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-15 10:25:00 | 1102.30 | 2025-10-15 11:00:00 | 1107.94 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-10-15 10:25:00 | 1102.30 | 2025-10-15 15:20:00 | 1127.20 | TARGET_HIT | 0.50 | 2.26% |
| SELL | retest1 | 2025-10-16 10:40:00 | 1124.80 | 2025-10-16 11:00:00 | 1132.15 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest1 | 2025-10-24 10:10:00 | 1074.40 | 2025-10-24 10:30:00 | 1078.15 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-10-27 10:45:00 | 1097.60 | 2025-10-27 12:00:00 | 1093.35 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-10-28 11:00:00 | 1096.40 | 2025-10-28 11:25:00 | 1094.13 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-31 11:05:00 | 1086.20 | 2025-10-31 11:45:00 | 1088.39 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-11-04 11:05:00 | 1068.10 | 2025-11-04 11:35:00 | 1064.50 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-04 11:05:00 | 1068.10 | 2025-11-04 12:35:00 | 1068.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-14 10:40:00 | 1067.30 | 2025-11-14 11:30:00 | 1069.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-27 10:45:00 | 1218.00 | 2025-11-27 11:40:00 | 1212.23 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-11-27 10:45:00 | 1218.00 | 2025-11-27 11:50:00 | 1218.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-03 11:05:00 | 1217.00 | 2025-12-03 12:20:00 | 1221.48 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-03 11:05:00 | 1217.00 | 2025-12-03 15:20:00 | 1251.80 | TARGET_HIT | 0.50 | 2.86% |
| BUY | retest1 | 2025-12-05 10:45:00 | 1263.10 | 2025-12-05 11:00:00 | 1257.63 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-12-11 11:00:00 | 1276.50 | 2025-12-11 11:10:00 | 1281.17 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-12 10:55:00 | 1294.20 | 2025-12-12 11:45:00 | 1290.44 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-15 11:15:00 | 1291.50 | 2025-12-15 11:55:00 | 1295.42 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-17 09:30:00 | 1327.40 | 2025-12-17 09:40:00 | 1321.75 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-12-18 10:50:00 | 1329.30 | 2025-12-18 11:45:00 | 1333.74 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-12-18 10:50:00 | 1329.30 | 2025-12-18 12:05:00 | 1329.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-26 10:45:00 | 1311.50 | 2025-12-26 12:15:00 | 1315.04 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-30 10:00:00 | 1318.10 | 2025-12-30 10:40:00 | 1314.26 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-31 11:00:00 | 1324.60 | 2025-12-31 11:20:00 | 1321.69 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-01 10:40:00 | 1314.80 | 2026-01-01 10:45:00 | 1316.98 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-01-06 10:50:00 | 1330.30 | 2026-01-06 11:00:00 | 1334.09 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-01-06 10:50:00 | 1330.30 | 2026-01-06 15:20:00 | 1352.00 | TARGET_HIT | 0.50 | 1.63% |
| SELL | retest1 | 2026-01-08 11:15:00 | 1328.10 | 2026-01-08 12:45:00 | 1321.52 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-01-08 11:15:00 | 1328.10 | 2026-01-08 15:20:00 | 1320.20 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2026-01-13 11:10:00 | 1330.60 | 2026-01-13 11:40:00 | 1335.32 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-01-13 11:10:00 | 1330.60 | 2026-01-13 13:40:00 | 1332.30 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2026-01-14 10:40:00 | 1321.20 | 2026-01-14 11:25:00 | 1323.89 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-01-30 10:45:00 | 1228.30 | 2026-01-30 10:50:00 | 1231.82 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-01-30 10:45:00 | 1228.30 | 2026-01-30 11:00:00 | 1228.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 10:50:00 | 1257.40 | 2026-02-01 11:10:00 | 1253.22 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-09 10:35:00 | 1338.80 | 2026-02-09 10:45:00 | 1332.76 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-02-10 09:30:00 | 1337.30 | 2026-02-10 09:40:00 | 1344.38 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-10 09:30:00 | 1337.30 | 2026-02-10 09:55:00 | 1337.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:20:00 | 1333.40 | 2026-02-11 10:30:00 | 1339.82 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-11 10:20:00 | 1333.40 | 2026-02-11 10:40:00 | 1333.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 10:10:00 | 1367.60 | 2026-02-12 10:15:00 | 1362.71 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-13 10:15:00 | 1340.70 | 2026-02-13 11:10:00 | 1347.01 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-02-18 10:55:00 | 1282.90 | 2026-02-18 11:05:00 | 1279.60 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-20 10:35:00 | 1305.70 | 2026-02-20 10:50:00 | 1310.97 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-02-26 10:10:00 | 1257.00 | 2026-02-26 10:40:00 | 1260.24 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-27 10:20:00 | 1239.70 | 2026-02-27 10:50:00 | 1242.66 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1225.50 | 2026-03-06 10:55:00 | 1220.41 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1225.50 | 2026-03-06 11:15:00 | 1225.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:50:00 | 1214.00 | 2026-03-16 11:30:00 | 1219.04 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-17 10:55:00 | 1247.80 | 2026-03-17 11:20:00 | 1244.25 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-23 10:50:00 | 1223.10 | 2026-03-23 11:05:00 | 1226.76 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-07 09:40:00 | 1229.00 | 2026-04-07 09:55:00 | 1233.99 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1297.10 | 2026-04-15 09:50:00 | 1293.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-16 10:25:00 | 1277.40 | 2026-04-16 10:30:00 | 1280.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1330.70 | 2026-04-21 09:45:00 | 1326.26 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-22 11:05:00 | 1343.10 | 2026-04-22 15:20:00 | 1344.00 | TARGET_HIT | 1.00 | 0.07% |
| BUY | retest1 | 2026-04-23 10:05:00 | 1362.30 | 2026-04-23 10:20:00 | 1367.60 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-23 10:05:00 | 1362.30 | 2026-04-23 10:50:00 | 1362.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-27 10:10:00 | 1343.70 | 2026-04-27 11:05:00 | 1336.79 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-27 10:10:00 | 1343.70 | 2026-04-27 13:00:00 | 1343.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 09:50:00 | 1304.00 | 2026-04-29 12:10:00 | 1308.84 | STOP_HIT | 1.00 | -0.37% |
