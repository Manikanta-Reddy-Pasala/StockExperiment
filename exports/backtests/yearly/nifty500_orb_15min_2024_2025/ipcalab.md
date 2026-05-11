# Ipca Laboratories Ltd. (IPCALAB)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:25:00 (34846 bars)
- **Last close:** 1527.00
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
| ENTRY1 | 96 |
| ENTRY2 | 0 |
| PARTIAL | 40 |
| TARGET_HIT | 27 |
| STOP_HIT | 69 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 70
- **Target hits / Stop hits / Partials:** 27 / 69 / 40
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 29.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 34 | 44.2% | 14 | 43 | 20 | 0.17% | 13.3% |
| BUY @ 2nd Alert (retest1) | 77 | 34 | 44.2% | 14 | 43 | 20 | 0.17% | 13.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 59 | 32 | 54.2% | 13 | 26 | 20 | 0.28% | 16.7% |
| SELL @ 2nd Alert (retest1) | 59 | 32 | 54.2% | 13 | 26 | 20 | 0.28% | 16.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 136 | 66 | 48.5% | 27 | 69 | 40 | 0.22% | 30.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:40:00 | 1285.50 | 1292.59 | 0.00 | ORB-short ORB[1291.60,1305.05] vol=2.5x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 11:10:00 | 1280.65 | 1290.20 | 0.00 | T1 1.5R @ 1280.65 |
| Stop hit — per-position SL triggered | 2024-05-14 11:15:00 | 1285.50 | 1290.15 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:35:00 | 1272.50 | 1277.50 | 0.00 | ORB-short ORB[1273.60,1284.70] vol=2.4x ATR=3.88 |
| Stop hit — per-position SL triggered | 2024-05-16 10:00:00 | 1276.38 | 1277.15 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:50:00 | 1280.05 | 1290.43 | 0.00 | ORB-short ORB[1288.40,1299.95] vol=1.7x ATR=3.11 |
| Stop hit — per-position SL triggered | 2024-05-17 11:10:00 | 1283.16 | 1289.48 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:50:00 | 1312.00 | 1307.81 | 0.00 | ORB-long ORB[1305.75,1310.45] vol=1.5x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-18 11:30:00 | 1318.10 | 1311.28 | 0.00 | T1 1.5R @ 1318.10 |
| Stop hit — per-position SL triggered | 2024-05-18 11:50:00 | 1312.00 | 1314.00 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 10:00:00 | 1295.60 | 1300.36 | 0.00 | ORB-short ORB[1298.95,1314.85] vol=2.3x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 12:00:00 | 1287.45 | 1298.28 | 0.00 | T1 1.5R @ 1287.45 |
| Target hit | 2024-05-21 13:55:00 | 1293.75 | 1292.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2024-05-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:45:00 | 1299.65 | 1301.86 | 0.00 | ORB-short ORB[1302.10,1313.95] vol=2.3x ATR=3.73 |
| Stop hit — per-position SL triggered | 2024-05-22 10:55:00 | 1303.38 | 1301.89 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-05-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:50:00 | 1310.05 | 1306.99 | 0.00 | ORB-long ORB[1296.60,1306.95] vol=2.4x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-05-28 10:00:00 | 1306.25 | 1307.19 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:10:00 | 1177.50 | 1162.12 | 0.00 | ORB-long ORB[1153.60,1170.55] vol=2.9x ATR=4.17 |
| Stop hit — per-position SL triggered | 2024-06-07 11:50:00 | 1173.33 | 1166.88 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:30:00 | 1168.10 | 1170.56 | 0.00 | ORB-short ORB[1170.60,1180.00] vol=3.9x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-06-11 10:30:00 | 1171.35 | 1169.51 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:05:00 | 1201.25 | 1198.40 | 0.00 | ORB-long ORB[1185.00,1196.00] vol=5.1x ATR=4.56 |
| Stop hit — per-position SL triggered | 2024-06-13 11:00:00 | 1196.69 | 1200.32 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:45:00 | 1210.25 | 1202.59 | 0.00 | ORB-long ORB[1197.30,1206.10] vol=4.5x ATR=3.06 |
| Stop hit — per-position SL triggered | 2024-06-14 11:00:00 | 1207.19 | 1203.77 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:10:00 | 1171.45 | 1178.31 | 0.00 | ORB-short ORB[1175.00,1187.90] vol=2.0x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 11:45:00 | 1164.99 | 1173.91 | 0.00 | T1 1.5R @ 1164.99 |
| Target hit | 2024-06-18 15:20:00 | 1162.90 | 1165.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-06-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:55:00 | 1151.55 | 1159.01 | 0.00 | ORB-short ORB[1159.55,1171.40] vol=2.1x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 11:20:00 | 1147.62 | 1153.71 | 0.00 | T1 1.5R @ 1147.62 |
| Target hit | 2024-06-19 15:20:00 | 1134.00 | 1143.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-06-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 10:05:00 | 1140.05 | 1133.11 | 0.00 | ORB-long ORB[1124.05,1135.95] vol=4.1x ATR=4.36 |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 1135.69 | 1134.89 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-06-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:00:00 | 1119.55 | 1126.83 | 0.00 | ORB-short ORB[1125.30,1137.50] vol=2.3x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:05:00 | 1115.14 | 1124.44 | 0.00 | T1 1.5R @ 1115.14 |
| Target hit | 2024-06-25 15:15:00 | 1103.70 | 1103.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — SELL (started 2024-07-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 11:00:00 | 1132.30 | 1136.59 | 0.00 | ORB-short ORB[1136.00,1146.00] vol=7.2x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 11:10:00 | 1127.87 | 1136.10 | 0.00 | T1 1.5R @ 1127.87 |
| Target hit | 2024-07-02 14:35:00 | 1127.80 | 1124.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:15:00 | 1144.40 | 1139.17 | 0.00 | ORB-long ORB[1128.00,1135.00] vol=1.5x ATR=3.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 10:50:00 | 1149.07 | 1141.94 | 0.00 | T1 1.5R @ 1149.07 |
| Target hit | 2024-07-03 14:25:00 | 1157.15 | 1162.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2024-07-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:55:00 | 1165.30 | 1160.15 | 0.00 | ORB-long ORB[1149.25,1162.75] vol=3.0x ATR=3.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 11:05:00 | 1170.25 | 1162.05 | 0.00 | T1 1.5R @ 1170.25 |
| Target hit | 2024-07-04 12:20:00 | 1170.05 | 1172.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 1189.75 | 1183.93 | 0.00 | ORB-long ORB[1174.50,1187.30] vol=3.5x ATR=3.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 09:40:00 | 1195.47 | 1189.46 | 0.00 | T1 1.5R @ 1195.47 |
| Stop hit — per-position SL triggered | 2024-07-05 09:45:00 | 1189.75 | 1189.31 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:30:00 | 1194.50 | 1188.43 | 0.00 | ORB-long ORB[1180.90,1194.10] vol=1.7x ATR=2.99 |
| Stop hit — per-position SL triggered | 2024-07-09 11:00:00 | 1191.51 | 1191.22 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:45:00 | 1239.55 | 1230.30 | 0.00 | ORB-long ORB[1224.00,1233.00] vol=4.3x ATR=3.21 |
| Stop hit — per-position SL triggered | 2024-07-15 10:50:00 | 1236.34 | 1230.58 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 10:10:00 | 1231.35 | 1220.35 | 0.00 | ORB-long ORB[1213.55,1224.50] vol=1.6x ATR=4.06 |
| Stop hit — per-position SL triggered | 2024-07-18 10:35:00 | 1227.29 | 1225.00 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:45:00 | 1237.00 | 1229.80 | 0.00 | ORB-long ORB[1221.25,1236.70] vol=3.0x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 11:05:00 | 1242.90 | 1230.73 | 0.00 | T1 1.5R @ 1242.90 |
| Target hit | 2024-07-25 15:20:00 | 1260.80 | 1249.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-07-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:50:00 | 1272.00 | 1267.08 | 0.00 | ORB-long ORB[1260.00,1270.00] vol=2.3x ATR=5.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:30:00 | 1279.51 | 1271.14 | 0.00 | T1 1.5R @ 1279.51 |
| Target hit | 2024-07-26 15:20:00 | 1284.30 | 1281.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-07-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:20:00 | 1300.00 | 1294.46 | 0.00 | ORB-long ORB[1285.30,1299.95] vol=2.9x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 10:40:00 | 1305.25 | 1296.31 | 0.00 | T1 1.5R @ 1305.25 |
| Stop hit — per-position SL triggered | 2024-07-29 11:50:00 | 1300.00 | 1299.72 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 1312.90 | 1308.05 | 0.00 | ORB-long ORB[1292.10,1305.50] vol=6.8x ATR=4.92 |
| Stop hit — per-position SL triggered | 2024-07-31 09:35:00 | 1307.98 | 1307.13 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:20:00 | 1312.25 | 1309.09 | 0.00 | ORB-long ORB[1300.00,1312.00] vol=1.6x ATR=3.42 |
| Stop hit — per-position SL triggered | 2024-08-01 10:45:00 | 1308.83 | 1309.43 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 09:55:00 | 1314.45 | 1308.72 | 0.00 | ORB-long ORB[1287.35,1303.55] vol=5.7x ATR=4.86 |
| Stop hit — per-position SL triggered | 2024-08-06 10:05:00 | 1309.59 | 1308.79 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:40:00 | 1352.85 | 1346.64 | 0.00 | ORB-long ORB[1338.70,1351.65] vol=2.3x ATR=4.79 |
| Stop hit — per-position SL triggered | 2024-08-08 10:05:00 | 1348.06 | 1349.49 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:40:00 | 1364.95 | 1355.06 | 0.00 | ORB-long ORB[1338.05,1356.40] vol=3.1x ATR=5.07 |
| Stop hit — per-position SL triggered | 2024-08-19 09:50:00 | 1359.88 | 1355.92 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:50:00 | 1342.55 | 1352.97 | 0.00 | ORB-short ORB[1349.70,1362.00] vol=2.1x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-08-20 11:00:00 | 1346.51 | 1351.57 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:35:00 | 1406.90 | 1397.25 | 0.00 | ORB-long ORB[1381.10,1400.85] vol=1.8x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-08-21 09:40:00 | 1402.22 | 1398.32 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 1392.00 | 1395.33 | 0.00 | ORB-short ORB[1395.00,1407.75] vol=3.9x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:55:00 | 1385.96 | 1392.56 | 0.00 | T1 1.5R @ 1385.96 |
| Target hit | 2024-08-23 13:30:00 | 1387.70 | 1387.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2024-08-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:40:00 | 1405.15 | 1396.19 | 0.00 | ORB-long ORB[1382.05,1398.40] vol=1.8x ATR=4.36 |
| Stop hit — per-position SL triggered | 2024-08-26 10:25:00 | 1400.79 | 1401.47 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-08-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:50:00 | 1404.10 | 1398.84 | 0.00 | ORB-long ORB[1388.60,1401.10] vol=2.8x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-08-27 10:10:00 | 1400.31 | 1400.44 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1391.20 | 1398.53 | 0.00 | ORB-short ORB[1393.00,1410.00] vol=2.0x ATR=4.20 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 1395.40 | 1397.57 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-08-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:50:00 | 1386.65 | 1393.71 | 0.00 | ORB-short ORB[1391.00,1403.45] vol=1.6x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:10:00 | 1380.72 | 1391.35 | 0.00 | T1 1.5R @ 1380.72 |
| Target hit | 2024-08-29 15:20:00 | 1371.00 | 1376.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:30:00 | 1381.75 | 1376.71 | 0.00 | ORB-long ORB[1366.00,1378.25] vol=3.1x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 09:45:00 | 1388.76 | 1380.06 | 0.00 | T1 1.5R @ 1388.76 |
| Target hit | 2024-08-30 10:20:00 | 1385.00 | 1386.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2024-09-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:40:00 | 1395.65 | 1391.54 | 0.00 | ORB-long ORB[1382.50,1394.00] vol=1.8x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 09:45:00 | 1400.63 | 1395.11 | 0.00 | T1 1.5R @ 1400.63 |
| Target hit | 2024-09-03 09:55:00 | 1396.60 | 1397.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — BUY (started 2024-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:40:00 | 1425.90 | 1419.43 | 0.00 | ORB-long ORB[1406.05,1423.95] vol=1.6x ATR=5.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:50:00 | 1433.59 | 1424.28 | 0.00 | T1 1.5R @ 1433.59 |
| Target hit | 2024-09-05 15:20:00 | 1446.00 | 1441.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2024-09-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:30:00 | 1454.55 | 1450.90 | 0.00 | ORB-long ORB[1437.60,1451.00] vol=2.6x ATR=4.46 |
| Stop hit — per-position SL triggered | 2024-09-12 10:40:00 | 1450.09 | 1451.47 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-09-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:10:00 | 1432.50 | 1440.26 | 0.00 | ORB-short ORB[1440.00,1461.00] vol=5.0x ATR=3.84 |
| Stop hit — per-position SL triggered | 2024-09-18 10:40:00 | 1436.34 | 1435.93 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-09-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 10:30:00 | 1466.00 | 1459.26 | 0.00 | ORB-long ORB[1446.75,1464.35] vol=3.2x ATR=5.55 |
| Stop hit — per-position SL triggered | 2024-09-19 10:35:00 | 1460.45 | 1460.47 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-09-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:30:00 | 1445.65 | 1439.92 | 0.00 | ORB-long ORB[1432.05,1445.10] vol=2.0x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 10:35:00 | 1451.51 | 1441.24 | 0.00 | T1 1.5R @ 1451.51 |
| Stop hit — per-position SL triggered | 2024-09-23 10:40:00 | 1445.65 | 1441.64 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 10:15:00 | 1500.50 | 1492.27 | 0.00 | ORB-long ORB[1480.35,1492.35] vol=1.9x ATR=4.54 |
| Stop hit — per-position SL triggered | 2024-09-25 10:45:00 | 1495.96 | 1494.09 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-09-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 11:00:00 | 1483.55 | 1488.41 | 0.00 | ORB-short ORB[1486.60,1507.35] vol=7.9x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:20:00 | 1479.12 | 1487.57 | 0.00 | T1 1.5R @ 1479.12 |
| Stop hit — per-position SL triggered | 2024-09-26 11:55:00 | 1483.55 | 1485.90 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-10-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:35:00 | 1490.45 | 1473.33 | 0.00 | ORB-long ORB[1453.10,1475.00] vol=1.9x ATR=5.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 11:30:00 | 1498.79 | 1480.31 | 0.00 | T1 1.5R @ 1498.79 |
| Stop hit — per-position SL triggered | 2024-10-04 14:20:00 | 1490.45 | 1493.69 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:50:00 | 1471.40 | 1483.87 | 0.00 | ORB-short ORB[1491.40,1504.80] vol=2.8x ATR=5.41 |
| Stop hit — per-position SL triggered | 2024-10-07 10:55:00 | 1476.81 | 1483.27 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-10-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:05:00 | 1507.90 | 1495.59 | 0.00 | ORB-long ORB[1470.30,1489.50] vol=3.3x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 10:20:00 | 1517.21 | 1505.18 | 0.00 | T1 1.5R @ 1517.21 |
| Target hit | 2024-10-08 15:20:00 | 1542.15 | 1528.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2024-10-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:05:00 | 1567.00 | 1554.56 | 0.00 | ORB-long ORB[1545.00,1563.95] vol=2.8x ATR=5.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:30:00 | 1574.71 | 1557.51 | 0.00 | T1 1.5R @ 1574.71 |
| Target hit | 2024-10-09 15:20:00 | 1614.30 | 1589.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2024-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:00:00 | 1641.30 | 1646.49 | 0.00 | ORB-short ORB[1646.55,1662.35] vol=1.8x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:30:00 | 1633.02 | 1641.79 | 0.00 | T1 1.5R @ 1633.02 |
| Target hit | 2024-10-17 11:30:00 | 1646.00 | 1641.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — SELL (started 2024-10-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:55:00 | 1567.40 | 1577.81 | 0.00 | ORB-short ORB[1579.90,1596.90] vol=2.1x ATR=6.03 |
| Stop hit — per-position SL triggered | 2024-10-25 11:15:00 | 1573.43 | 1576.99 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-10-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 10:55:00 | 1604.45 | 1588.30 | 0.00 | ORB-long ORB[1583.05,1599.60] vol=1.9x ATR=5.81 |
| Stop hit — per-position SL triggered | 2024-10-28 11:00:00 | 1598.64 | 1589.53 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:15:00 | 1559.80 | 1574.70 | 0.00 | ORB-short ORB[1583.05,1605.50] vol=2.9x ATR=5.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:20:00 | 1551.80 | 1570.83 | 0.00 | T1 1.5R @ 1551.80 |
| Stop hit — per-position SL triggered | 2024-10-29 10:40:00 | 1559.80 | 1565.00 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-11-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:45:00 | 1586.05 | 1595.99 | 0.00 | ORB-short ORB[1590.10,1605.00] vol=4.1x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 11:20:00 | 1579.18 | 1592.82 | 0.00 | T1 1.5R @ 1579.18 |
| Stop hit — per-position SL triggered | 2024-11-06 14:20:00 | 1586.05 | 1587.69 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-11-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 09:50:00 | 1564.00 | 1575.81 | 0.00 | ORB-short ORB[1571.55,1592.45] vol=1.5x ATR=5.40 |
| Stop hit — per-position SL triggered | 2024-11-07 10:55:00 | 1569.40 | 1570.51 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-11-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 10:40:00 | 1571.95 | 1563.51 | 0.00 | ORB-long ORB[1548.85,1566.05] vol=2.8x ATR=4.55 |
| Stop hit — per-position SL triggered | 2024-11-08 11:25:00 | 1567.40 | 1566.95 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-11 11:15:00 | 1540.00 | 1548.63 | 0.00 | ORB-short ORB[1547.60,1566.10] vol=6.2x ATR=5.08 |
| Stop hit — per-position SL triggered | 2024-11-11 11:25:00 | 1545.08 | 1548.19 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 09:55:00 | 1578.00 | 1566.66 | 0.00 | ORB-long ORB[1555.05,1574.45] vol=2.1x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 11:10:00 | 1587.01 | 1571.62 | 0.00 | T1 1.5R @ 1587.01 |
| Stop hit — per-position SL triggered | 2024-11-21 11:55:00 | 1578.00 | 1573.98 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-11-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:35:00 | 1620.50 | 1608.14 | 0.00 | ORB-long ORB[1595.25,1609.75] vol=4.0x ATR=5.01 |
| Stop hit — per-position SL triggered | 2024-11-25 09:40:00 | 1615.49 | 1609.64 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-11-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 10:20:00 | 1569.50 | 1582.42 | 0.00 | ORB-short ORB[1594.65,1617.35] vol=2.2x ATR=5.57 |
| Stop hit — per-position SL triggered | 2024-11-26 10:25:00 | 1575.07 | 1582.05 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-11-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:35:00 | 1547.05 | 1559.34 | 0.00 | ORB-short ORB[1558.15,1573.90] vol=2.2x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 11:10:00 | 1540.60 | 1555.85 | 0.00 | T1 1.5R @ 1540.60 |
| Target hit | 2024-11-27 15:20:00 | 1528.75 | 1538.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2024-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:30:00 | 1508.90 | 1518.70 | 0.00 | ORB-short ORB[1511.65,1533.90] vol=1.7x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-11-28 10:35:00 | 1513.17 | 1518.14 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-11-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 11:00:00 | 1536.80 | 1527.53 | 0.00 | ORB-long ORB[1514.55,1531.30] vol=2.0x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-11-29 11:45:00 | 1532.41 | 1529.89 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-12-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 09:55:00 | 1516.80 | 1522.23 | 0.00 | ORB-short ORB[1520.05,1542.00] vol=1.7x ATR=4.47 |
| Stop hit — per-position SL triggered | 2024-12-03 10:10:00 | 1521.27 | 1520.84 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-12-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:00:00 | 1481.20 | 1490.36 | 0.00 | ORB-short ORB[1484.15,1504.95] vol=1.7x ATR=4.96 |
| Stop hit — per-position SL triggered | 2024-12-04 10:10:00 | 1486.16 | 1488.94 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:30:00 | 1548.70 | 1542.03 | 0.00 | ORB-long ORB[1528.30,1542.00] vol=3.4x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-12-11 09:40:00 | 1543.52 | 1544.21 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:30:00 | 1564.80 | 1561.23 | 0.00 | ORB-long ORB[1555.10,1564.00] vol=1.9x ATR=3.40 |
| Stop hit — per-position SL triggered | 2024-12-12 09:35:00 | 1561.40 | 1561.77 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-12-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:00:00 | 1580.00 | 1572.45 | 0.00 | ORB-long ORB[1561.15,1576.00] vol=4.0x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-12-17 10:05:00 | 1575.41 | 1573.42 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-12-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 09:40:00 | 1583.65 | 1572.78 | 0.00 | ORB-long ORB[1546.00,1566.35] vol=2.5x ATR=5.21 |
| Stop hit — per-position SL triggered | 2024-12-18 09:50:00 | 1578.44 | 1577.05 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-12-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 11:05:00 | 1608.45 | 1600.40 | 0.00 | ORB-long ORB[1588.70,1607.95] vol=2.5x ATR=6.18 |
| Stop hit — per-position SL triggered | 2024-12-20 12:50:00 | 1602.27 | 1604.20 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:50:00 | 1595.90 | 1584.29 | 0.00 | ORB-long ORB[1570.10,1586.95] vol=1.6x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 10:10:00 | 1602.62 | 1589.81 | 0.00 | T1 1.5R @ 1602.62 |
| Target hit | 2024-12-24 11:25:00 | 1609.05 | 1609.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — BUY (started 2024-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:40:00 | 1611.95 | 1602.27 | 0.00 | ORB-long ORB[1587.80,1600.45] vol=2.2x ATR=6.32 |
| Stop hit — per-position SL triggered | 2024-12-27 09:50:00 | 1605.63 | 1604.07 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-12-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 11:05:00 | 1675.45 | 1664.10 | 0.00 | ORB-long ORB[1651.20,1673.35] vol=1.6x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 11:55:00 | 1684.82 | 1671.20 | 0.00 | T1 1.5R @ 1684.82 |
| Target hit | 2024-12-31 15:20:00 | 1694.90 | 1684.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2025-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:45:00 | 1720.05 | 1707.72 | 0.00 | ORB-long ORB[1686.45,1699.10] vol=2.9x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:35:00 | 1727.34 | 1713.99 | 0.00 | T1 1.5R @ 1727.34 |
| Target hit | 2025-01-02 15:20:00 | 1743.70 | 1732.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — SELL (started 2025-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:50:00 | 1713.30 | 1720.39 | 0.00 | ORB-short ORB[1719.95,1736.00] vol=12.1x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:45:00 | 1704.99 | 1718.76 | 0.00 | T1 1.5R @ 1704.99 |
| Stop hit — per-position SL triggered | 2025-01-06 12:05:00 | 1713.30 | 1718.26 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:15:00 | 1710.00 | 1723.23 | 0.00 | ORB-short ORB[1732.05,1749.30] vol=2.6x ATR=5.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:20:00 | 1701.16 | 1718.34 | 0.00 | T1 1.5R @ 1701.16 |
| Target hit | 2025-01-08 14:20:00 | 1701.00 | 1699.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 78 — SELL (started 2025-01-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:55:00 | 1639.20 | 1653.30 | 0.00 | ORB-short ORB[1653.70,1672.40] vol=2.7x ATR=6.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:30:00 | 1629.04 | 1645.77 | 0.00 | T1 1.5R @ 1629.04 |
| Target hit | 2025-01-10 15:20:00 | 1631.45 | 1634.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — SELL (started 2025-01-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 10:25:00 | 1575.05 | 1586.02 | 0.00 | ORB-short ORB[1579.70,1600.35] vol=2.4x ATR=5.88 |
| Stop hit — per-position SL triggered | 2025-01-14 10:45:00 | 1580.93 | 1584.56 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-01-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:10:00 | 1581.75 | 1577.32 | 0.00 | ORB-long ORB[1553.00,1567.45] vol=1.7x ATR=5.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 11:15:00 | 1589.78 | 1580.44 | 0.00 | T1 1.5R @ 1589.78 |
| Target hit | 2025-01-23 13:30:00 | 1583.00 | 1583.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 81 — SELL (started 2025-01-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:55:00 | 1482.30 | 1495.35 | 0.00 | ORB-short ORB[1499.50,1513.10] vol=1.6x ATR=6.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:20:00 | 1473.16 | 1490.94 | 0.00 | T1 1.5R @ 1473.16 |
| Target hit | 2025-01-27 15:20:00 | 1443.65 | 1454.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — SELL (started 2025-01-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:35:00 | 1403.95 | 1414.51 | 0.00 | ORB-short ORB[1426.90,1443.90] vol=5.6x ATR=6.72 |
| Stop hit — per-position SL triggered | 2025-01-28 10:50:00 | 1410.67 | 1413.33 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-01-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:55:00 | 1402.00 | 1394.11 | 0.00 | ORB-long ORB[1376.45,1391.10] vol=2.0x ATR=5.04 |
| Stop hit — per-position SL triggered | 2025-01-29 11:15:00 | 1396.96 | 1394.90 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-01-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 10:55:00 | 1444.05 | 1454.77 | 0.00 | ORB-short ORB[1455.50,1473.60] vol=1.8x ATR=4.04 |
| Stop hit — per-position SL triggered | 2025-01-31 11:25:00 | 1448.09 | 1453.43 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-02-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:20:00 | 1479.85 | 1475.40 | 0.00 | ORB-long ORB[1452.45,1471.75] vol=1.8x ATR=5.74 |
| Stop hit — per-position SL triggered | 2025-02-05 10:45:00 | 1474.11 | 1475.69 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2025-02-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 10:05:00 | 1481.05 | 1497.99 | 0.00 | ORB-short ORB[1491.45,1511.40] vol=2.0x ATR=6.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 10:15:00 | 1471.80 | 1495.03 | 0.00 | T1 1.5R @ 1471.80 |
| Stop hit — per-position SL triggered | 2025-02-06 13:25:00 | 1481.05 | 1482.11 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2025-02-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 11:00:00 | 1408.05 | 1420.78 | 0.00 | ORB-short ORB[1409.55,1424.30] vol=4.1x ATR=5.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:15:00 | 1399.71 | 1418.80 | 0.00 | T1 1.5R @ 1399.71 |
| Target hit | 2025-02-27 15:20:00 | 1366.20 | 1387.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — BUY (started 2025-03-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:35:00 | 1362.25 | 1348.78 | 0.00 | ORB-long ORB[1335.55,1353.00] vol=1.9x ATR=5.85 |
| Stop hit — per-position SL triggered | 2025-03-05 09:40:00 | 1356.40 | 1349.62 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-03-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:05:00 | 1338.90 | 1328.98 | 0.00 | ORB-long ORB[1314.95,1334.45] vol=1.8x ATR=5.64 |
| Stop hit — per-position SL triggered | 2025-03-13 10:15:00 | 1333.26 | 1329.94 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2025-03-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:40:00 | 1322.95 | 1308.07 | 0.00 | ORB-long ORB[1296.55,1311.45] vol=2.3x ATR=6.20 |
| Stop hit — per-position SL triggered | 2025-03-17 09:50:00 | 1316.75 | 1311.24 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-03-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:10:00 | 1346.20 | 1330.54 | 0.00 | ORB-long ORB[1311.45,1330.20] vol=4.2x ATR=5.44 |
| Stop hit — per-position SL triggered | 2025-03-18 12:20:00 | 1340.76 | 1333.55 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2025-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 11:00:00 | 1388.85 | 1374.92 | 0.00 | ORB-long ORB[1357.15,1375.95] vol=4.2x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 11:35:00 | 1397.55 | 1379.86 | 0.00 | T1 1.5R @ 1397.55 |
| Target hit | 2025-03-20 15:20:00 | 1404.75 | 1397.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 93 — BUY (started 2025-03-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:55:00 | 1421.00 | 1412.52 | 0.00 | ORB-long ORB[1399.30,1420.40] vol=2.9x ATR=5.19 |
| Stop hit — per-position SL triggered | 2025-03-27 11:10:00 | 1415.81 | 1412.93 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2025-04-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:10:00 | 1416.90 | 1433.36 | 0.00 | ORB-short ORB[1444.10,1464.40] vol=2.5x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:30:00 | 1406.49 | 1430.62 | 0.00 | T1 1.5R @ 1406.49 |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 1416.90 | 1425.16 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2025-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:35:00 | 1387.50 | 1384.85 | 0.00 | ORB-long ORB[1372.90,1387.00] vol=3.7x ATR=5.82 |
| Stop hit — per-position SL triggered | 2025-05-05 09:40:00 | 1381.68 | 1384.64 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2025-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 09:40:00 | 1354.00 | 1363.15 | 0.00 | ORB-short ORB[1357.80,1377.90] vol=1.8x ATR=6.90 |
| Stop hit — per-position SL triggered | 2025-05-07 09:55:00 | 1360.90 | 1359.61 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:40:00 | 1285.50 | 2024-05-14 11:10:00 | 1280.65 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-05-14 10:40:00 | 1285.50 | 2024-05-14 11:15:00 | 1285.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 09:35:00 | 1272.50 | 2024-05-16 10:00:00 | 1276.38 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-17 10:50:00 | 1280.05 | 2024-05-17 11:10:00 | 1283.16 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-05-18 09:50:00 | 1312.00 | 2024-05-18 11:30:00 | 1318.10 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-05-18 09:50:00 | 1312.00 | 2024-05-18 11:50:00 | 1312.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-21 10:00:00 | 1295.60 | 2024-05-21 12:00:00 | 1287.45 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-05-21 10:00:00 | 1295.60 | 2024-05-21 13:55:00 | 1293.75 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2024-05-22 10:45:00 | 1299.65 | 2024-05-22 10:55:00 | 1303.38 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-28 09:50:00 | 1310.05 | 2024-05-28 10:00:00 | 1306.25 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-07 11:10:00 | 1177.50 | 2024-06-07 11:50:00 | 1173.33 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-06-11 09:30:00 | 1168.10 | 2024-06-11 10:30:00 | 1171.35 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-13 10:05:00 | 1201.25 | 2024-06-13 11:00:00 | 1196.69 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-14 10:45:00 | 1210.25 | 2024-06-14 11:00:00 | 1207.19 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-06-18 10:10:00 | 1171.45 | 2024-06-18 11:45:00 | 1164.99 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-06-18 10:10:00 | 1171.45 | 2024-06-18 15:20:00 | 1162.90 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2024-06-19 10:55:00 | 1151.55 | 2024-06-19 11:20:00 | 1147.62 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-06-19 10:55:00 | 1151.55 | 2024-06-19 15:20:00 | 1134.00 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2024-06-21 10:05:00 | 1140.05 | 2024-06-21 11:15:00 | 1135.69 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-06-25 10:00:00 | 1119.55 | 2024-06-25 10:05:00 | 1115.14 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-06-25 10:00:00 | 1119.55 | 2024-06-25 15:15:00 | 1103.70 | TARGET_HIT | 0.50 | 1.42% |
| SELL | retest1 | 2024-07-02 11:00:00 | 1132.30 | 2024-07-02 11:10:00 | 1127.87 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-02 11:00:00 | 1132.30 | 2024-07-02 14:35:00 | 1127.80 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-03 10:15:00 | 1144.40 | 2024-07-03 10:50:00 | 1149.07 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-03 10:15:00 | 1144.40 | 2024-07-03 14:25:00 | 1157.15 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2024-07-04 10:55:00 | 1165.30 | 2024-07-04 11:05:00 | 1170.25 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-04 10:55:00 | 1165.30 | 2024-07-04 12:20:00 | 1170.05 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-05 09:30:00 | 1189.75 | 2024-07-05 09:40:00 | 1195.47 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-05 09:30:00 | 1189.75 | 2024-07-05 09:45:00 | 1189.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-09 10:30:00 | 1194.50 | 2024-07-09 11:00:00 | 1191.51 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-15 10:45:00 | 1239.55 | 2024-07-15 10:50:00 | 1236.34 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-18 10:10:00 | 1231.35 | 2024-07-18 10:35:00 | 1227.29 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-25 10:45:00 | 1237.00 | 2024-07-25 11:05:00 | 1242.90 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-25 10:45:00 | 1237.00 | 2024-07-25 15:20:00 | 1260.80 | TARGET_HIT | 0.50 | 1.92% |
| BUY | retest1 | 2024-07-26 09:50:00 | 1272.00 | 2024-07-26 10:30:00 | 1279.51 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-07-26 09:50:00 | 1272.00 | 2024-07-26 15:20:00 | 1284.30 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2024-07-29 10:20:00 | 1300.00 | 2024-07-29 10:40:00 | 1305.25 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-29 10:20:00 | 1300.00 | 2024-07-29 11:50:00 | 1300.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 09:30:00 | 1312.90 | 2024-07-31 09:35:00 | 1307.98 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-01 10:20:00 | 1312.25 | 2024-08-01 10:45:00 | 1308.83 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-06 09:55:00 | 1314.45 | 2024-08-06 10:05:00 | 1309.59 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-08 09:40:00 | 1352.85 | 2024-08-08 10:05:00 | 1348.06 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-19 09:40:00 | 1364.95 | 2024-08-19 09:50:00 | 1359.88 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-20 10:50:00 | 1342.55 | 2024-08-20 11:00:00 | 1346.51 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-21 09:35:00 | 1406.90 | 2024-08-21 09:40:00 | 1402.22 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-23 09:30:00 | 1392.00 | 2024-08-23 10:55:00 | 1385.96 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-23 09:30:00 | 1392.00 | 2024-08-23 13:30:00 | 1387.70 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-08-26 09:40:00 | 1405.15 | 2024-08-26 10:25:00 | 1400.79 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-27 09:50:00 | 1404.10 | 2024-08-27 10:10:00 | 1400.31 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1391.20 | 2024-08-28 09:40:00 | 1395.40 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-29 09:50:00 | 1386.65 | 2024-08-29 10:10:00 | 1380.72 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-29 09:50:00 | 1386.65 | 2024-08-29 15:20:00 | 1371.00 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2024-08-30 09:30:00 | 1381.75 | 2024-08-30 09:45:00 | 1388.76 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-30 09:30:00 | 1381.75 | 2024-08-30 10:20:00 | 1385.00 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2024-09-03 09:40:00 | 1395.65 | 2024-09-03 09:45:00 | 1400.63 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-09-03 09:40:00 | 1395.65 | 2024-09-03 09:55:00 | 1396.60 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2024-09-05 09:40:00 | 1425.90 | 2024-09-05 09:50:00 | 1433.59 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-09-05 09:40:00 | 1425.90 | 2024-09-05 15:20:00 | 1446.00 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2024-09-12 10:30:00 | 1454.55 | 2024-09-12 10:40:00 | 1450.09 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-18 10:10:00 | 1432.50 | 2024-09-18 10:40:00 | 1436.34 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-19 10:30:00 | 1466.00 | 2024-09-19 10:35:00 | 1460.45 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-23 10:30:00 | 1445.65 | 2024-09-23 10:35:00 | 1451.51 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-23 10:30:00 | 1445.65 | 2024-09-23 10:40:00 | 1445.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-25 10:15:00 | 1500.50 | 2024-09-25 10:45:00 | 1495.96 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-26 11:00:00 | 1483.55 | 2024-09-26 11:20:00 | 1479.12 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-09-26 11:00:00 | 1483.55 | 2024-09-26 11:55:00 | 1483.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-04 10:35:00 | 1490.45 | 2024-10-04 11:30:00 | 1498.79 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-10-04 10:35:00 | 1490.45 | 2024-10-04 14:20:00 | 1490.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 10:50:00 | 1471.40 | 2024-10-07 10:55:00 | 1476.81 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-10-08 10:05:00 | 1507.90 | 2024-10-08 10:20:00 | 1517.21 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-10-08 10:05:00 | 1507.90 | 2024-10-08 15:20:00 | 1542.15 | TARGET_HIT | 0.50 | 2.27% |
| BUY | retest1 | 2024-10-09 11:05:00 | 1567.00 | 2024-10-09 11:30:00 | 1574.71 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-10-09 11:05:00 | 1567.00 | 2024-10-09 15:20:00 | 1614.30 | TARGET_HIT | 0.50 | 3.02% |
| SELL | retest1 | 2024-10-17 11:00:00 | 1641.30 | 2024-10-17 11:30:00 | 1633.02 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-17 11:00:00 | 1641.30 | 2024-10-17 11:30:00 | 1646.00 | TARGET_HIT | 0.50 | -0.29% |
| SELL | retest1 | 2024-10-25 10:55:00 | 1567.40 | 2024-10-25 11:15:00 | 1573.43 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-28 10:55:00 | 1604.45 | 2024-10-28 11:00:00 | 1598.64 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-29 10:15:00 | 1559.80 | 2024-10-29 10:20:00 | 1551.80 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-29 10:15:00 | 1559.80 | 2024-10-29 10:40:00 | 1559.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-06 10:45:00 | 1586.05 | 2024-11-06 11:20:00 | 1579.18 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-11-06 10:45:00 | 1586.05 | 2024-11-06 14:20:00 | 1586.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-07 09:50:00 | 1564.00 | 2024-11-07 10:55:00 | 1569.40 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-11-08 10:40:00 | 1571.95 | 2024-11-08 11:25:00 | 1567.40 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-11 11:15:00 | 1540.00 | 2024-11-11 11:25:00 | 1545.08 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-21 09:55:00 | 1578.00 | 2024-11-21 11:10:00 | 1587.01 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-11-21 09:55:00 | 1578.00 | 2024-11-21 11:55:00 | 1578.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 09:35:00 | 1620.50 | 2024-11-25 09:40:00 | 1615.49 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-11-26 10:20:00 | 1569.50 | 2024-11-26 10:25:00 | 1575.07 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-27 10:35:00 | 1547.05 | 2024-11-27 11:10:00 | 1540.60 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-11-27 10:35:00 | 1547.05 | 2024-11-27 15:20:00 | 1528.75 | TARGET_HIT | 0.50 | 1.18% |
| SELL | retest1 | 2024-11-28 10:30:00 | 1508.90 | 2024-11-28 10:35:00 | 1513.17 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-11-29 11:00:00 | 1536.80 | 2024-11-29 11:45:00 | 1532.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-03 09:55:00 | 1516.80 | 2024-12-03 10:10:00 | 1521.27 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-04 10:00:00 | 1481.20 | 2024-12-04 10:10:00 | 1486.16 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-11 09:30:00 | 1548.70 | 2024-12-11 09:40:00 | 1543.52 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-12 09:30:00 | 1564.80 | 2024-12-12 09:35:00 | 1561.40 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-12-17 10:00:00 | 1580.00 | 2024-12-17 10:05:00 | 1575.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-18 09:40:00 | 1583.65 | 2024-12-18 09:50:00 | 1578.44 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-20 11:05:00 | 1608.45 | 2024-12-20 12:50:00 | 1602.27 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-24 09:50:00 | 1595.90 | 2024-12-24 10:10:00 | 1602.62 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-12-24 09:50:00 | 1595.90 | 2024-12-24 11:25:00 | 1609.05 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2024-12-27 09:40:00 | 1611.95 | 2024-12-27 09:50:00 | 1605.63 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-31 11:05:00 | 1675.45 | 2024-12-31 11:55:00 | 1684.82 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-12-31 11:05:00 | 1675.45 | 2024-12-31 15:20:00 | 1694.90 | TARGET_HIT | 0.50 | 1.16% |
| BUY | retest1 | 2025-01-02 10:45:00 | 1720.05 | 2025-01-02 11:35:00 | 1727.34 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-01-02 10:45:00 | 1720.05 | 2025-01-02 15:20:00 | 1743.70 | TARGET_HIT | 0.50 | 1.37% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1713.30 | 2025-01-06 11:45:00 | 1704.99 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-01-06 10:50:00 | 1713.30 | 2025-01-06 12:05:00 | 1713.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-08 11:15:00 | 1710.00 | 2025-01-08 11:20:00 | 1701.16 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-08 11:15:00 | 1710.00 | 2025-01-08 14:20:00 | 1701.00 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2025-01-10 09:55:00 | 1639.20 | 2025-01-10 10:30:00 | 1629.04 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-01-10 09:55:00 | 1639.20 | 2025-01-10 15:20:00 | 1631.45 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-14 10:25:00 | 1575.05 | 2025-01-14 10:45:00 | 1580.93 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-23 10:10:00 | 1581.75 | 2025-01-23 11:15:00 | 1589.78 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-01-23 10:10:00 | 1581.75 | 2025-01-23 13:30:00 | 1583.00 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2025-01-27 09:55:00 | 1482.30 | 2025-01-27 10:20:00 | 1473.16 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-01-27 09:55:00 | 1482.30 | 2025-01-27 15:20:00 | 1443.65 | TARGET_HIT | 0.50 | 2.61% |
| SELL | retest1 | 2025-01-28 10:35:00 | 1403.95 | 2025-01-28 10:50:00 | 1410.67 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-01-29 10:55:00 | 1402.00 | 2025-01-29 11:15:00 | 1396.96 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-01-31 10:55:00 | 1444.05 | 2025-01-31 11:25:00 | 1448.09 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-02-05 10:20:00 | 1479.85 | 2025-02-05 10:45:00 | 1474.11 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-02-06 10:05:00 | 1481.05 | 2025-02-06 10:15:00 | 1471.80 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-02-06 10:05:00 | 1481.05 | 2025-02-06 13:25:00 | 1481.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-27 11:00:00 | 1408.05 | 2025-02-27 11:15:00 | 1399.71 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-02-27 11:00:00 | 1408.05 | 2025-02-27 15:20:00 | 1366.20 | TARGET_HIT | 0.50 | 2.97% |
| BUY | retest1 | 2025-03-05 09:35:00 | 1362.25 | 2025-03-05 09:40:00 | 1356.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-03-13 10:05:00 | 1338.90 | 2025-03-13 10:15:00 | 1333.26 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-17 09:40:00 | 1322.95 | 2025-03-17 09:50:00 | 1316.75 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-03-18 11:10:00 | 1346.20 | 2025-03-18 12:20:00 | 1340.76 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-20 11:00:00 | 1388.85 | 2025-03-20 11:35:00 | 1397.55 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-03-20 11:00:00 | 1388.85 | 2025-03-20 15:20:00 | 1404.75 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2025-03-27 10:55:00 | 1421.00 | 2025-03-27 11:10:00 | 1415.81 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-25 10:10:00 | 1416.90 | 2025-04-25 10:30:00 | 1406.49 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2025-04-25 10:10:00 | 1416.90 | 2025-04-25 11:15:00 | 1416.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 09:35:00 | 1387.50 | 2025-05-05 09:40:00 | 1381.68 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-05-07 09:40:00 | 1354.00 | 2025-05-07 09:55:00 | 1360.90 | STOP_HIT | 1.00 | -0.51% |
