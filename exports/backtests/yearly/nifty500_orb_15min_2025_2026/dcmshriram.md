# DCM Shriram Ltd. (DCMSHRIRAM)

## Backtest Summary

- **Window:** 2025-06-10 09:15:00 → 2026-05-08 15:25:00 (16888 bars)
- **Last close:** 1237.00
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
| ENTRY1 | 73 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 13 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 103 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 60
- **Target hits / Stop hits / Partials:** 13 / 60 / 30
- **Avg / median % per leg:** 0.38% / 0.00%
- **Sum % (uncompounded):** 39.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 20 | 40.8% | 7 | 29 | 13 | 0.56% | 27.6% |
| BUY @ 2nd Alert (retest1) | 49 | 20 | 40.8% | 7 | 29 | 13 | 0.56% | 27.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 54 | 23 | 42.6% | 6 | 31 | 17 | 0.22% | 12.0% |
| SELL @ 2nd Alert (retest1) | 54 | 23 | 42.6% | 6 | 31 | 17 | 0.22% | 12.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 103 | 43 | 41.7% | 13 | 60 | 30 | 0.38% | 39.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 11:00:00 | 1093.80 | 1086.87 | 0.00 | ORB-long ORB[1074.60,1086.90] vol=2.0x ATR=4.72 |
| Stop hit — per-position SL triggered | 2025-06-10 15:20:00 | 1093.50 | 1089.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2025-06-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 11:00:00 | 1150.00 | 1146.10 | 0.00 | ORB-long ORB[1134.30,1149.40] vol=1.6x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 14:25:00 | 1156.15 | 1149.60 | 0.00 | T1 1.5R @ 1156.15 |
| Target hit | 2025-06-17 15:20:00 | 1155.00 | 1151.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-06-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:45:00 | 1173.80 | 1167.92 | 0.00 | ORB-long ORB[1157.10,1172.60] vol=3.0x ATR=3.52 |
| Stop hit — per-position SL triggered | 2025-06-20 09:50:00 | 1170.28 | 1168.06 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:15:00 | 1197.10 | 1204.87 | 0.00 | ORB-short ORB[1201.00,1216.10] vol=2.0x ATR=3.49 |
| Stop hit — per-position SL triggered | 2025-06-26 11:30:00 | 1200.59 | 1204.03 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-07-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:10:00 | 1209.20 | 1217.05 | 0.00 | ORB-short ORB[1213.20,1227.90] vol=1.6x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:35:00 | 1202.96 | 1215.51 | 0.00 | T1 1.5R @ 1202.96 |
| Stop hit — per-position SL triggered | 2025-07-02 10:50:00 | 1209.20 | 1214.67 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-07-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:30:00 | 1253.00 | 1237.78 | 0.00 | ORB-long ORB[1224.30,1235.00] vol=3.0x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:35:00 | 1260.07 | 1241.35 | 0.00 | T1 1.5R @ 1260.07 |
| Target hit | 2025-07-03 15:20:00 | 1446.00 | 1385.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2025-07-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:35:00 | 1425.10 | 1413.59 | 0.00 | ORB-long ORB[1397.60,1417.20] vol=2.8x ATR=7.66 |
| Stop hit — per-position SL triggered | 2025-07-16 09:45:00 | 1417.44 | 1416.90 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:40:00 | 1395.20 | 1403.12 | 0.00 | ORB-short ORB[1404.20,1414.60] vol=2.9x ATR=5.45 |
| Stop hit — per-position SL triggered | 2025-07-18 09:50:00 | 1400.65 | 1401.83 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:50:00 | 1379.20 | 1402.20 | 0.00 | ORB-short ORB[1403.10,1417.50] vol=2.4x ATR=5.67 |
| Stop hit — per-position SL triggered | 2025-07-25 10:55:00 | 1384.87 | 1400.16 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-08-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:10:00 | 1345.50 | 1358.97 | 0.00 | ORB-short ORB[1360.30,1377.50] vol=1.9x ATR=5.39 |
| Stop hit — per-position SL triggered | 2025-08-07 10:20:00 | 1350.89 | 1357.96 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-08-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:30:00 | 1312.00 | 1322.39 | 0.00 | ORB-short ORB[1325.00,1342.40] vol=2.9x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 10:45:00 | 1306.53 | 1316.55 | 0.00 | T1 1.5R @ 1306.53 |
| Target hit | 2025-08-14 15:20:00 | 1283.30 | 1296.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:45:00 | 1255.40 | 1258.59 | 0.00 | ORB-short ORB[1256.00,1274.10] vol=2.6x ATR=5.14 |
| Stop hit — per-position SL triggered | 2025-08-20 12:50:00 | 1260.54 | 1255.20 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-08-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 11:05:00 | 1275.20 | 1269.13 | 0.00 | ORB-long ORB[1264.70,1271.30] vol=2.4x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 11:10:00 | 1280.75 | 1270.73 | 0.00 | T1 1.5R @ 1280.75 |
| Target hit | 2025-08-21 12:15:00 | 1281.80 | 1283.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 1260.70 | 1270.77 | 0.00 | ORB-short ORB[1269.00,1281.40] vol=3.5x ATR=5.12 |
| Stop hit — per-position SL triggered | 2025-08-22 09:45:00 | 1265.82 | 1270.13 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:35:00 | 1246.00 | 1237.83 | 0.00 | ORB-long ORB[1230.30,1239.80] vol=3.6x ATR=5.27 |
| Stop hit — per-position SL triggered | 2025-09-11 09:45:00 | 1240.73 | 1238.10 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-09-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:35:00 | 1229.30 | 1228.47 | 0.00 | ORB-long ORB[1220.00,1228.20] vol=2.3x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 10:20:00 | 1234.15 | 1229.44 | 0.00 | T1 1.5R @ 1234.15 |
| Stop hit — per-position SL triggered | 2025-09-15 10:45:00 | 1229.30 | 1229.65 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-09-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:05:00 | 1234.90 | 1230.57 | 0.00 | ORB-long ORB[1219.30,1232.50] vol=2.6x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 10:40:00 | 1239.99 | 1233.83 | 0.00 | T1 1.5R @ 1239.99 |
| Target hit | 2025-09-17 15:20:00 | 1347.70 | 1361.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-09-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:20:00 | 1162.10 | 1153.32 | 0.00 | ORB-long ORB[1143.00,1158.30] vol=2.0x ATR=4.17 |
| Stop hit — per-position SL triggered | 2025-09-29 10:30:00 | 1157.93 | 1154.85 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-10-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:55:00 | 1202.70 | 1193.12 | 0.00 | ORB-long ORB[1186.00,1197.60] vol=6.7x ATR=4.56 |
| Stop hit — per-position SL triggered | 2025-10-06 12:15:00 | 1198.14 | 1195.89 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:15:00 | 1182.20 | 1189.28 | 0.00 | ORB-short ORB[1188.10,1194.50] vol=3.0x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:20:00 | 1177.59 | 1188.42 | 0.00 | T1 1.5R @ 1177.59 |
| Stop hit — per-position SL triggered | 2025-10-08 11:25:00 | 1182.20 | 1188.36 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-10-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:20:00 | 1162.30 | 1167.55 | 0.00 | ORB-short ORB[1166.40,1179.00] vol=1.9x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:40:00 | 1157.99 | 1166.17 | 0.00 | T1 1.5R @ 1157.99 |
| Stop hit — per-position SL triggered | 2025-10-14 10:50:00 | 1162.30 | 1165.21 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-10-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:20:00 | 1227.00 | 1219.52 | 0.00 | ORB-long ORB[1210.10,1226.00] vol=1.6x ATR=5.18 |
| Stop hit — per-position SL triggered | 2025-10-17 10:25:00 | 1221.82 | 1220.08 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-10-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:00:00 | 1243.90 | 1234.36 | 0.00 | ORB-long ORB[1222.10,1232.50] vol=3.4x ATR=4.48 |
| Stop hit — per-position SL triggered | 2025-10-24 11:30:00 | 1239.42 | 1240.35 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-10-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:40:00 | 1232.40 | 1225.74 | 0.00 | ORB-long ORB[1213.60,1231.10] vol=3.1x ATR=5.84 |
| Stop hit — per-position SL triggered | 2025-10-27 10:05:00 | 1226.56 | 1226.67 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-10-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:50:00 | 1305.00 | 1312.45 | 0.00 | ORB-short ORB[1311.50,1324.00] vol=3.8x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:55:00 | 1299.61 | 1310.75 | 0.00 | T1 1.5R @ 1299.61 |
| Target hit | 2025-10-31 15:20:00 | 1282.40 | 1299.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-11-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 11:10:00 | 1272.40 | 1283.52 | 0.00 | ORB-short ORB[1281.00,1290.50] vol=4.4x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 13:25:00 | 1267.12 | 1279.99 | 0.00 | T1 1.5R @ 1267.12 |
| Stop hit — per-position SL triggered | 2025-11-03 14:40:00 | 1272.40 | 1278.78 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:35:00 | 1219.00 | 1230.13 | 0.00 | ORB-short ORB[1228.00,1240.40] vol=3.1x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:40:00 | 1212.12 | 1222.68 | 0.00 | T1 1.5R @ 1212.12 |
| Stop hit — per-position SL triggered | 2025-11-06 09:45:00 | 1219.00 | 1222.48 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-11-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:40:00 | 1255.60 | 1241.50 | 0.00 | ORB-long ORB[1230.00,1244.60] vol=1.6x ATR=5.37 |
| Stop hit — per-position SL triggered | 2025-11-10 12:25:00 | 1250.23 | 1246.92 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-11-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:50:00 | 1228.00 | 1232.57 | 0.00 | ORB-short ORB[1233.50,1249.90] vol=1.5x ATR=3.77 |
| Stop hit — per-position SL triggered | 2025-11-11 09:55:00 | 1231.77 | 1235.25 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:45:00 | 1243.50 | 1233.88 | 0.00 | ORB-long ORB[1230.00,1241.20] vol=2.4x ATR=3.91 |
| Stop hit — per-position SL triggered | 2025-11-12 10:10:00 | 1239.59 | 1236.41 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-11-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 10:20:00 | 1219.60 | 1230.98 | 0.00 | ORB-short ORB[1228.80,1242.80] vol=6.3x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 10:25:00 | 1212.58 | 1222.74 | 0.00 | T1 1.5R @ 1212.58 |
| Stop hit — per-position SL triggered | 2025-11-17 10:30:00 | 1219.60 | 1222.16 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-11-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 11:05:00 | 1179.50 | 1185.53 | 0.00 | ORB-short ORB[1185.80,1201.00] vol=2.3x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 11:10:00 | 1175.02 | 1185.42 | 0.00 | T1 1.5R @ 1175.02 |
| Stop hit — per-position SL triggered | 2025-11-19 11:40:00 | 1179.50 | 1184.34 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-11-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:40:00 | 1171.20 | 1176.16 | 0.00 | ORB-short ORB[1175.10,1186.50] vol=1.9x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-11-20 10:05:00 | 1175.04 | 1175.49 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:30:00 | 1180.70 | 1175.68 | 0.00 | ORB-long ORB[1169.30,1180.10] vol=2.9x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:05:00 | 1187.49 | 1178.23 | 0.00 | T1 1.5R @ 1187.49 |
| Target hit | 2025-11-21 10:45:00 | 1204.10 | 1207.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 11:15:00 | 1200.00 | 1190.31 | 0.00 | ORB-long ORB[1187.40,1198.80] vol=5.4x ATR=4.39 |
| Stop hit — per-position SL triggered | 2025-11-25 11:25:00 | 1195.61 | 1191.14 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:40:00 | 1203.00 | 1192.47 | 0.00 | ORB-long ORB[1180.30,1193.40] vol=1.8x ATR=3.92 |
| Stop hit — per-position SL triggered | 2025-11-26 11:00:00 | 1199.08 | 1193.73 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-12-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 10:20:00 | 1221.30 | 1208.07 | 0.00 | ORB-long ORB[1197.90,1208.20] vol=4.3x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 10:30:00 | 1228.36 | 1212.92 | 0.00 | T1 1.5R @ 1228.36 |
| Stop hit — per-position SL triggered | 2025-12-02 12:10:00 | 1221.30 | 1221.34 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-12-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 10:05:00 | 1226.00 | 1216.55 | 0.00 | ORB-long ORB[1207.90,1221.80] vol=1.6x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:10:00 | 1232.34 | 1219.98 | 0.00 | T1 1.5R @ 1232.34 |
| Target hit | 2025-12-08 10:55:00 | 1227.20 | 1230.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2025-12-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:10:00 | 1204.50 | 1194.96 | 0.00 | ORB-long ORB[1186.90,1200.20] vol=1.6x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 11:05:00 | 1212.60 | 1197.68 | 0.00 | T1 1.5R @ 1212.60 |
| Target hit | 2025-12-09 15:20:00 | 1223.80 | 1207.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2025-12-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:20:00 | 1233.10 | 1228.01 | 0.00 | ORB-long ORB[1218.80,1230.20] vol=1.7x ATR=4.59 |
| Stop hit — per-position SL triggered | 2025-12-10 10:45:00 | 1228.51 | 1229.28 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:40:00 | 1238.90 | 1234.99 | 0.00 | ORB-long ORB[1226.00,1238.40] vol=2.7x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-12-17 09:45:00 | 1234.71 | 1236.69 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-12-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:30:00 | 1211.50 | 1218.94 | 0.00 | ORB-short ORB[1217.00,1233.00] vol=1.6x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:40:00 | 1205.23 | 1215.79 | 0.00 | T1 1.5R @ 1205.23 |
| Stop hit — per-position SL triggered | 2025-12-18 09:50:00 | 1211.50 | 1214.76 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-12-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:00:00 | 1236.10 | 1228.94 | 0.00 | ORB-long ORB[1220.00,1232.90] vol=1.7x ATR=3.93 |
| Stop hit — per-position SL triggered | 2025-12-22 10:05:00 | 1232.17 | 1229.22 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:35:00 | 1268.70 | 1265.06 | 0.00 | ORB-long ORB[1255.30,1265.20] vol=3.5x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:55:00 | 1276.15 | 1269.75 | 0.00 | T1 1.5R @ 1276.15 |
| Stop hit — per-position SL triggered | 2025-12-26 10:05:00 | 1268.70 | 1269.78 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:50:00 | 1230.50 | 1233.88 | 0.00 | ORB-short ORB[1233.70,1250.50] vol=3.0x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-12-29 10:55:00 | 1233.64 | 1234.93 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:05:00 | 1252.10 | 1243.88 | 0.00 | ORB-long ORB[1234.10,1245.00] vol=1.5x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-12-30 13:10:00 | 1248.45 | 1249.75 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-01-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:35:00 | 1266.50 | 1259.28 | 0.00 | ORB-long ORB[1251.70,1261.00] vol=2.8x ATR=3.81 |
| Stop hit — per-position SL triggered | 2026-01-01 10:40:00 | 1262.69 | 1259.84 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:55:00 | 1241.30 | 1245.16 | 0.00 | ORB-short ORB[1248.30,1258.50] vol=4.8x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:55:00 | 1236.27 | 1244.15 | 0.00 | T1 1.5R @ 1236.27 |
| Target hit | 2026-01-06 15:20:00 | 1230.10 | 1236.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2026-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:05:00 | 1211.00 | 1213.32 | 0.00 | ORB-short ORB[1211.10,1228.50] vol=2.5x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 1205.76 | 1213.01 | 0.00 | T1 1.5R @ 1205.76 |
| Target hit | 2026-01-08 15:20:00 | 1175.80 | 1195.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2026-01-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 11:05:00 | 1115.60 | 1126.41 | 0.00 | ORB-short ORB[1125.00,1138.50] vol=4.3x ATR=4.92 |
| Stop hit — per-position SL triggered | 2026-01-20 11:10:00 | 1120.52 | 1125.32 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:15:00 | 1110.00 | 1119.76 | 0.00 | ORB-short ORB[1121.00,1130.80] vol=2.2x ATR=3.93 |
| Stop hit — per-position SL triggered | 2026-01-28 11:40:00 | 1113.93 | 1119.40 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-01-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:45:00 | 1131.00 | 1136.95 | 0.00 | ORB-short ORB[1142.40,1150.80] vol=2.4x ATR=4.03 |
| Stop hit — per-position SL triggered | 2026-01-29 11:25:00 | 1135.03 | 1136.08 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:55:00 | 1191.50 | 1179.42 | 0.00 | ORB-long ORB[1158.40,1170.80] vol=5.8x ATR=4.50 |
| Stop hit — per-position SL triggered | 2026-01-30 11:30:00 | 1187.00 | 1180.37 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-02-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-01 09:45:00 | 1156.80 | 1161.16 | 0.00 | ORB-short ORB[1163.30,1177.80] vol=2.7x ATR=4.23 |
| Stop hit — per-position SL triggered | 2026-02-01 09:50:00 | 1161.03 | 1161.08 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-02-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 09:50:00 | 1155.20 | 1160.08 | 0.00 | ORB-short ORB[1156.70,1169.30] vol=2.4x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 10:20:00 | 1148.70 | 1156.96 | 0.00 | T1 1.5R @ 1148.70 |
| Stop hit — per-position SL triggered | 2026-02-04 11:05:00 | 1155.20 | 1156.09 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:15:00 | 1155.60 | 1158.83 | 0.00 | ORB-short ORB[1160.00,1167.80] vol=4.7x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 10:20:00 | 1149.66 | 1155.89 | 0.00 | T1 1.5R @ 1149.66 |
| Stop hit — per-position SL triggered | 2026-02-05 13:05:00 | 1155.60 | 1152.30 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-02-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 09:35:00 | 1147.00 | 1149.53 | 0.00 | ORB-short ORB[1148.00,1160.00] vol=1.9x ATR=2.94 |
| Stop hit — per-position SL triggered | 2026-02-06 09:45:00 | 1149.94 | 1149.43 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-02-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:10:00 | 1148.60 | 1153.57 | 0.00 | ORB-short ORB[1150.50,1164.10] vol=3.9x ATR=3.36 |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 1151.96 | 1153.27 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:10:00 | 1105.70 | 1108.67 | 0.00 | ORB-short ORB[1106.00,1116.30] vol=2.4x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:25:00 | 1101.52 | 1108.13 | 0.00 | T1 1.5R @ 1101.52 |
| Target hit | 2026-02-19 15:20:00 | 1078.00 | 1094.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:15:00 | 1075.30 | 1082.41 | 0.00 | ORB-short ORB[1081.00,1091.60] vol=1.6x ATR=3.84 |
| Stop hit — per-position SL triggered | 2026-02-23 10:20:00 | 1079.14 | 1082.28 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 1064.00 | 1071.41 | 0.00 | ORB-short ORB[1077.00,1092.80] vol=3.7x ATR=5.04 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 1069.04 | 1071.33 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-02-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:05:00 | 1072.00 | 1078.56 | 0.00 | ORB-short ORB[1074.40,1087.80] vol=3.7x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:25:00 | 1066.01 | 1072.58 | 0.00 | T1 1.5R @ 1066.01 |
| Stop hit — per-position SL triggered | 2026-02-25 12:20:00 | 1072.00 | 1071.33 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:15:00 | 1014.60 | 1000.54 | 0.00 | ORB-long ORB[995.60,1005.00] vol=3.3x ATR=5.65 |
| Stop hit — per-position SL triggered | 2026-03-04 11:20:00 | 1008.95 | 1005.95 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:15:00 | 1008.70 | 1001.08 | 0.00 | ORB-long ORB[987.60,1002.00] vol=4.6x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:20:00 | 1014.77 | 1001.62 | 0.00 | T1 1.5R @ 1014.77 |
| Stop hit — per-position SL triggered | 2026-03-06 10:25:00 | 1008.70 | 1001.91 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:55:00 | 1010.00 | 1005.28 | 0.00 | ORB-long ORB[995.10,1003.90] vol=1.8x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:25:00 | 1015.86 | 1007.16 | 0.00 | T1 1.5R @ 1015.86 |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 1010.00 | 1009.85 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:10:00 | 1044.10 | 1055.55 | 0.00 | ORB-short ORB[1052.00,1063.00] vol=2.8x ATR=5.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:20:00 | 1036.43 | 1054.17 | 0.00 | T1 1.5R @ 1036.43 |
| Target hit | 2026-03-20 12:50:00 | 1037.80 | 1036.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 67 — BUY (started 2026-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:10:00 | 1114.00 | 1109.88 | 0.00 | ORB-long ORB[1094.80,1106.60] vol=2.8x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:30:00 | 1119.14 | 1110.54 | 0.00 | T1 1.5R @ 1119.14 |
| Stop hit — per-position SL triggered | 2026-03-27 11:50:00 | 1114.00 | 1111.93 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-04-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 10:20:00 | 1152.00 | 1158.82 | 0.00 | ORB-short ORB[1154.00,1167.40] vol=2.1x ATR=3.90 |
| Stop hit — per-position SL triggered | 2026-04-08 10:30:00 | 1155.90 | 1158.78 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 1155.70 | 1140.83 | 0.00 | ORB-long ORB[1126.00,1135.90] vol=3.8x ATR=5.78 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 1149.92 | 1146.40 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-04-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:00:00 | 1202.70 | 1197.53 | 0.00 | ORB-long ORB[1192.50,1200.00] vol=3.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-04-21 11:05:00 | 1199.56 | 1197.55 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 1197.70 | 1202.17 | 0.00 | ORB-short ORB[1199.10,1215.00] vol=2.1x ATR=3.83 |
| Stop hit — per-position SL triggered | 2026-04-23 09:40:00 | 1201.53 | 1202.18 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 1200.70 | 1206.43 | 0.00 | ORB-short ORB[1204.00,1214.40] vol=6.2x ATR=2.69 |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 1203.39 | 1206.24 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 1270.40 | 1264.44 | 0.00 | ORB-long ORB[1254.00,1269.90] vol=1.6x ATR=5.57 |
| Stop hit — per-position SL triggered | 2026-05-08 09:55:00 | 1264.83 | 1261.35 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-10 11:00:00 | 1093.80 | 2025-06-10 15:20:00 | 1093.50 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest1 | 2025-06-17 11:00:00 | 1150.00 | 2025-06-17 14:25:00 | 1156.15 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-06-17 11:00:00 | 1150.00 | 2025-06-17 15:20:00 | 1155.00 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-06-20 09:45:00 | 1173.80 | 2025-06-20 09:50:00 | 1170.28 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-26 11:15:00 | 1197.10 | 2025-06-26 11:30:00 | 1200.59 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-02 10:10:00 | 1209.20 | 2025-07-02 10:35:00 | 1202.96 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-07-02 10:10:00 | 1209.20 | 2025-07-02 10:50:00 | 1209.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-03 10:30:00 | 1253.00 | 2025-07-03 10:35:00 | 1260.07 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-07-03 10:30:00 | 1253.00 | 2025-07-03 15:20:00 | 1446.00 | TARGET_HIT | 0.50 | 15.40% |
| BUY | retest1 | 2025-07-16 09:35:00 | 1425.10 | 2025-07-16 09:45:00 | 1417.44 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-07-18 09:40:00 | 1395.20 | 2025-07-18 09:50:00 | 1400.65 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-07-25 10:50:00 | 1379.20 | 2025-07-25 10:55:00 | 1384.87 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-08-07 10:10:00 | 1345.50 | 2025-08-07 10:20:00 | 1350.89 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-08-14 10:30:00 | 1312.00 | 2025-08-14 10:45:00 | 1306.53 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-08-14 10:30:00 | 1312.00 | 2025-08-14 15:20:00 | 1283.30 | TARGET_HIT | 0.50 | 2.19% |
| SELL | retest1 | 2025-08-20 09:45:00 | 1255.40 | 2025-08-20 12:50:00 | 1260.54 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-08-21 11:05:00 | 1275.20 | 2025-08-21 11:10:00 | 1280.75 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-08-21 11:05:00 | 1275.20 | 2025-08-21 12:15:00 | 1281.80 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2025-08-22 09:30:00 | 1260.70 | 2025-08-22 09:45:00 | 1265.82 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-09-11 09:35:00 | 1246.00 | 2025-09-11 09:45:00 | 1240.73 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-09-15 09:35:00 | 1229.30 | 2025-09-15 10:20:00 | 1234.15 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-15 09:35:00 | 1229.30 | 2025-09-15 10:45:00 | 1229.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 10:05:00 | 1234.90 | 2025-09-17 10:40:00 | 1239.99 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-09-17 10:05:00 | 1234.90 | 2025-09-17 15:20:00 | 1347.70 | TARGET_HIT | 0.50 | 9.13% |
| BUY | retest1 | 2025-09-29 10:20:00 | 1162.10 | 2025-09-29 10:30:00 | 1157.93 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-06 10:55:00 | 1202.70 | 2025-10-06 12:15:00 | 1198.14 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-10-08 11:15:00 | 1182.20 | 2025-10-08 11:20:00 | 1177.59 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-10-08 11:15:00 | 1182.20 | 2025-10-08 11:25:00 | 1182.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 10:20:00 | 1162.30 | 2025-10-14 10:40:00 | 1157.99 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-14 10:20:00 | 1162.30 | 2025-10-14 10:50:00 | 1162.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 10:20:00 | 1227.00 | 2025-10-17 10:25:00 | 1221.82 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-10-24 10:00:00 | 1243.90 | 2025-10-24 11:30:00 | 1239.42 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-27 09:40:00 | 1232.40 | 2025-10-27 10:05:00 | 1226.56 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-10-31 10:50:00 | 1305.00 | 2025-10-31 10:55:00 | 1299.61 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-31 10:50:00 | 1305.00 | 2025-10-31 15:20:00 | 1282.40 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2025-11-03 11:10:00 | 1272.40 | 2025-11-03 13:25:00 | 1267.12 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-11-03 11:10:00 | 1272.40 | 2025-11-03 14:40:00 | 1272.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-06 09:35:00 | 1219.00 | 2025-11-06 09:40:00 | 1212.12 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-11-06 09:35:00 | 1219.00 | 2025-11-06 09:45:00 | 1219.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-10 10:40:00 | 1255.60 | 2025-11-10 12:25:00 | 1250.23 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-11-11 09:50:00 | 1228.00 | 2025-11-11 09:55:00 | 1231.77 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-12 09:45:00 | 1243.50 | 2025-11-12 10:10:00 | 1239.59 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-17 10:20:00 | 1219.60 | 2025-11-17 10:25:00 | 1212.58 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-11-17 10:20:00 | 1219.60 | 2025-11-17 10:30:00 | 1219.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-19 11:05:00 | 1179.50 | 2025-11-19 11:10:00 | 1175.02 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-11-19 11:05:00 | 1179.50 | 2025-11-19 11:40:00 | 1179.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-20 09:40:00 | 1171.20 | 2025-11-20 10:05:00 | 1175.04 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-21 09:30:00 | 1180.70 | 2025-11-21 10:05:00 | 1187.49 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-11-21 09:30:00 | 1180.70 | 2025-11-21 10:45:00 | 1204.10 | TARGET_HIT | 0.50 | 1.98% |
| BUY | retest1 | 2025-11-25 11:15:00 | 1200.00 | 2025-11-25 11:25:00 | 1195.61 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-11-26 10:40:00 | 1203.00 | 2025-11-26 11:00:00 | 1199.08 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-02 10:20:00 | 1221.30 | 2025-12-02 10:30:00 | 1228.36 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-12-02 10:20:00 | 1221.30 | 2025-12-02 12:10:00 | 1221.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-08 10:05:00 | 1226.00 | 2025-12-08 10:10:00 | 1232.34 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-12-08 10:05:00 | 1226.00 | 2025-12-08 10:55:00 | 1227.20 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2025-12-09 10:10:00 | 1204.50 | 2025-12-09 11:05:00 | 1212.60 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-12-09 10:10:00 | 1204.50 | 2025-12-09 15:20:00 | 1223.80 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2025-12-10 10:20:00 | 1233.10 | 2025-12-10 10:45:00 | 1228.51 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-17 09:40:00 | 1238.90 | 2025-12-17 09:45:00 | 1234.71 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-18 09:30:00 | 1211.50 | 2025-12-18 09:40:00 | 1205.23 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-12-18 09:30:00 | 1211.50 | 2025-12-18 09:50:00 | 1211.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-22 10:00:00 | 1236.10 | 2025-12-22 10:05:00 | 1232.17 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-12-26 09:35:00 | 1268.70 | 2025-12-26 09:55:00 | 1276.15 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-12-26 09:35:00 | 1268.70 | 2025-12-26 10:05:00 | 1268.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 10:50:00 | 1230.50 | 2025-12-29 10:55:00 | 1233.64 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-30 10:05:00 | 1252.10 | 2025-12-30 13:10:00 | 1248.45 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-01 10:35:00 | 1266.50 | 2026-01-01 10:40:00 | 1262.69 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-06 10:55:00 | 1241.30 | 2026-01-06 11:55:00 | 1236.27 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-06 10:55:00 | 1241.30 | 2026-01-06 15:20:00 | 1230.10 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2026-01-08 11:05:00 | 1211.00 | 2026-01-08 11:15:00 | 1205.76 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-01-08 11:05:00 | 1211.00 | 2026-01-08 15:20:00 | 1175.80 | TARGET_HIT | 0.50 | 2.91% |
| SELL | retest1 | 2026-01-20 11:05:00 | 1115.60 | 2026-01-20 11:10:00 | 1120.52 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-01-28 10:15:00 | 1110.00 | 2026-01-28 11:40:00 | 1113.93 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-29 10:45:00 | 1131.00 | 2026-01-29 11:25:00 | 1135.03 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-01-30 10:55:00 | 1191.50 | 2026-01-30 11:30:00 | 1187.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-01 09:45:00 | 1156.80 | 2026-02-01 09:50:00 | 1161.03 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-04 09:50:00 | 1155.20 | 2026-02-04 10:20:00 | 1148.70 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-04 09:50:00 | 1155.20 | 2026-02-04 11:05:00 | 1155.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-05 10:15:00 | 1155.60 | 2026-02-05 10:20:00 | 1149.66 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-05 10:15:00 | 1155.60 | 2026-02-05 13:05:00 | 1155.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-06 09:35:00 | 1147.00 | 2026-02-06 09:45:00 | 1149.94 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-11 11:10:00 | 1148.60 | 2026-02-11 11:15:00 | 1151.96 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-19 11:10:00 | 1105.70 | 2026-02-19 11:25:00 | 1101.52 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-19 11:10:00 | 1105.70 | 2026-02-19 15:20:00 | 1078.00 | TARGET_HIT | 0.50 | 2.51% |
| SELL | retest1 | 2026-02-23 10:15:00 | 1075.30 | 2026-02-23 10:20:00 | 1079.14 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-24 09:30:00 | 1064.00 | 2026-02-24 09:35:00 | 1069.04 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-02-25 10:05:00 | 1072.00 | 2026-02-25 11:25:00 | 1066.01 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-25 10:05:00 | 1072.00 | 2026-02-25 12:20:00 | 1072.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-04 10:15:00 | 1014.60 | 2026-03-04 11:20:00 | 1008.95 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2026-03-06 10:15:00 | 1008.70 | 2026-03-06 10:20:00 | 1014.77 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-06 10:15:00 | 1008.70 | 2026-03-06 10:25:00 | 1008.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 09:55:00 | 1010.00 | 2026-03-11 10:25:00 | 1015.86 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-11 09:55:00 | 1010.00 | 2026-03-11 11:15:00 | 1010.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 10:10:00 | 1044.10 | 2026-03-20 10:20:00 | 1036.43 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-03-20 10:10:00 | 1044.10 | 2026-03-20 12:50:00 | 1037.80 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-27 11:10:00 | 1114.00 | 2026-03-27 11:30:00 | 1119.14 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-27 11:10:00 | 1114.00 | 2026-03-27 11:50:00 | 1114.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-08 10:20:00 | 1152.00 | 2026-04-08 10:30:00 | 1155.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-15 09:35:00 | 1155.70 | 2026-04-15 09:40:00 | 1149.92 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-21 11:00:00 | 1202.70 | 2026-04-21 11:05:00 | 1199.56 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-23 09:35:00 | 1197.70 | 2026-04-23 09:40:00 | 1201.53 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-28 11:00:00 | 1200.70 | 2026-04-28 11:15:00 | 1203.39 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-05-08 09:50:00 | 1270.40 | 2026-05-08 09:55:00 | 1264.83 | STOP_HIT | 1.00 | -0.44% |
