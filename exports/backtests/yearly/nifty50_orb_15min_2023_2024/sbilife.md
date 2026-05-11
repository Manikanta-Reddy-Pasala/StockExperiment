# SBILIFE (SBILIFE)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-09-02 15:25:00 (24280 bars)
- **Last close:** 1887.45
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
| ENTRY1 | 85 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 18 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 120 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 67
- **Target hits / Stop hits / Partials:** 18 / 67 / 35
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 15.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 26 | 46.4% | 8 | 30 | 18 | 0.17% | 9.3% |
| BUY @ 2nd Alert (retest1) | 56 | 26 | 46.4% | 8 | 30 | 18 | 0.17% | 9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 64 | 27 | 42.2% | 10 | 37 | 17 | 0.10% | 6.6% |
| SELL @ 2nd Alert (retest1) | 64 | 27 | 42.2% | 10 | 37 | 17 | 0.10% | 6.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 120 | 53 | 44.2% | 18 | 67 | 35 | 0.13% | 15.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 10:55:00 | 1173.05 | 1175.38 | 0.00 | ORB-short ORB[1173.85,1184.95] vol=1.7x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 11:35:00 | 1169.57 | 1174.42 | 0.00 | T1 1.5R @ 1169.57 |
| Target hit | 2023-05-17 15:20:00 | 1161.65 | 1167.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2023-05-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 10:55:00 | 1157.70 | 1155.84 | 0.00 | ORB-long ORB[1151.85,1157.45] vol=11.8x ATR=2.12 |
| Stop hit — per-position SL triggered | 2023-05-22 11:00:00 | 1155.58 | 1155.90 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-23 09:45:00 | 1145.30 | 1150.04 | 0.00 | ORB-short ORB[1147.00,1156.30] vol=1.8x ATR=3.05 |
| Stop hit — per-position SL triggered | 2023-05-23 09:50:00 | 1148.35 | 1149.92 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:45:00 | 1178.10 | 1175.84 | 0.00 | ORB-long ORB[1171.50,1176.85] vol=2.2x ATR=1.56 |
| Stop hit — per-position SL triggered | 2023-05-25 09:55:00 | 1176.54 | 1175.99 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 11:05:00 | 1190.30 | 1184.45 | 0.00 | ORB-long ORB[1173.15,1186.80] vol=2.3x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-26 12:00:00 | 1193.74 | 1185.83 | 0.00 | T1 1.5R @ 1193.74 |
| Target hit | 2023-05-26 15:20:00 | 1195.00 | 1191.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2023-05-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:05:00 | 1231.05 | 1224.47 | 0.00 | ORB-long ORB[1211.95,1222.00] vol=4.1x ATR=2.51 |
| Stop hit — per-position SL triggered | 2023-05-31 10:30:00 | 1228.54 | 1226.50 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-01 11:15:00 | 1227.30 | 1230.53 | 0.00 | ORB-short ORB[1229.10,1243.00] vol=3.5x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-01 12:20:00 | 1224.22 | 1229.39 | 0.00 | T1 1.5R @ 1224.22 |
| Target hit | 2023-06-01 15:20:00 | 1208.45 | 1220.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2023-06-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:50:00 | 1249.00 | 1245.78 | 0.00 | ORB-long ORB[1236.05,1246.75] vol=1.6x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-16 10:05:00 | 1253.30 | 1248.83 | 0.00 | T1 1.5R @ 1253.30 |
| Target hit | 2023-06-16 15:20:00 | 1280.25 | 1276.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 1295.00 | 1292.44 | 0.00 | ORB-long ORB[1280.50,1294.95] vol=3.2x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 09:40:00 | 1300.58 | 1294.69 | 0.00 | T1 1.5R @ 1300.58 |
| Target hit | 2023-06-20 11:20:00 | 1296.90 | 1298.29 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2023-06-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-23 10:00:00 | 1274.00 | 1268.84 | 0.00 | ORB-long ORB[1265.40,1273.00] vol=1.5x ATR=3.41 |
| Stop hit — per-position SL triggered | 2023-06-23 10:05:00 | 1270.59 | 1269.04 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 09:50:00 | 1263.25 | 1265.64 | 0.00 | ORB-short ORB[1266.35,1275.65] vol=6.9x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-27 09:55:00 | 1259.88 | 1265.21 | 0.00 | T1 1.5R @ 1259.88 |
| Stop hit — per-position SL triggered | 2023-06-27 11:55:00 | 1263.25 | 1262.88 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-07-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 11:05:00 | 1299.75 | 1306.26 | 0.00 | ORB-short ORB[1304.05,1312.00] vol=2.2x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 11:20:00 | 1296.30 | 1305.07 | 0.00 | T1 1.5R @ 1296.30 |
| Stop hit — per-position SL triggered | 2023-07-03 11:35:00 | 1299.75 | 1304.32 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:55:00 | 1304.20 | 1299.20 | 0.00 | ORB-long ORB[1285.10,1295.05] vol=2.1x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:10:00 | 1309.42 | 1301.67 | 0.00 | T1 1.5R @ 1309.42 |
| Stop hit — per-position SL triggered | 2023-07-05 10:35:00 | 1304.20 | 1302.66 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 11:05:00 | 1311.95 | 1304.07 | 0.00 | ORB-long ORB[1292.55,1302.35] vol=1.9x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 11:15:00 | 1315.98 | 1306.07 | 0.00 | T1 1.5R @ 1315.98 |
| Stop hit — per-position SL triggered | 2023-07-07 11:20:00 | 1311.95 | 1306.30 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 10:50:00 | 1285.00 | 1287.52 | 0.00 | ORB-short ORB[1285.55,1302.00] vol=12.7x ATR=3.38 |
| Stop hit — per-position SL triggered | 2023-07-10 12:30:00 | 1288.38 | 1287.41 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 11:00:00 | 1305.90 | 1300.60 | 0.00 | ORB-long ORB[1291.00,1303.00] vol=1.6x ATR=2.38 |
| Stop hit — per-position SL triggered | 2023-07-12 11:05:00 | 1303.52 | 1300.79 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 10:50:00 | 1328.10 | 1318.51 | 0.00 | ORB-long ORB[1305.05,1313.65] vol=1.8x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 11:00:00 | 1331.84 | 1321.24 | 0.00 | T1 1.5R @ 1331.84 |
| Stop hit — per-position SL triggered | 2023-07-13 12:35:00 | 1328.10 | 1325.34 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 09:40:00 | 1309.95 | 1314.29 | 0.00 | ORB-short ORB[1310.20,1321.60] vol=2.0x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 11:05:00 | 1305.33 | 1310.58 | 0.00 | T1 1.5R @ 1305.33 |
| Stop hit — per-position SL triggered | 2023-07-20 12:05:00 | 1309.95 | 1309.44 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 10:15:00 | 1309.00 | 1300.70 | 0.00 | ORB-long ORB[1287.90,1298.25] vol=2.7x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 11:10:00 | 1314.09 | 1305.03 | 0.00 | T1 1.5R @ 1314.09 |
| Stop hit — per-position SL triggered | 2023-07-24 11:45:00 | 1309.00 | 1309.65 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-08-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:45:00 | 1264.95 | 1269.15 | 0.00 | ORB-short ORB[1275.00,1285.00] vol=6.6x ATR=2.72 |
| Stop hit — per-position SL triggered | 2023-08-01 11:05:00 | 1267.67 | 1268.94 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 10:55:00 | 1271.70 | 1263.18 | 0.00 | ORB-long ORB[1252.05,1263.75] vol=1.6x ATR=3.20 |
| Stop hit — per-position SL triggered | 2023-08-03 11:05:00 | 1268.50 | 1263.33 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:30:00 | 1320.20 | 1325.43 | 0.00 | ORB-short ORB[1325.05,1340.00] vol=2.1x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 09:50:00 | 1314.40 | 1321.62 | 0.00 | T1 1.5R @ 1314.40 |
| Stop hit — per-position SL triggered | 2023-08-11 09:55:00 | 1320.20 | 1321.48 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 09:30:00 | 1296.35 | 1289.95 | 0.00 | ORB-long ORB[1278.75,1293.65] vol=1.6x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 09:40:00 | 1301.51 | 1294.65 | 0.00 | T1 1.5R @ 1301.51 |
| Target hit | 2023-08-22 11:50:00 | 1302.60 | 1302.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — SELL (started 2023-08-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 11:10:00 | 1283.65 | 1288.91 | 0.00 | ORB-short ORB[1284.05,1295.25] vol=9.1x ATR=3.16 |
| Stop hit — per-position SL triggered | 2023-08-25 11:20:00 | 1286.81 | 1288.75 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-09-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 11:10:00 | 1323.30 | 1314.60 | 0.00 | ORB-long ORB[1293.20,1304.95] vol=1.7x ATR=3.00 |
| Stop hit — per-position SL triggered | 2023-09-01 11:35:00 | 1320.30 | 1315.29 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-09-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 10:50:00 | 1305.30 | 1314.91 | 0.00 | ORB-short ORB[1319.25,1328.50] vol=1.8x ATR=2.70 |
| Stop hit — per-position SL triggered | 2023-09-04 11:40:00 | 1308.00 | 1311.75 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 10:15:00 | 1318.95 | 1328.26 | 0.00 | ORB-short ORB[1328.35,1338.95] vol=1.5x ATR=3.33 |
| Stop hit — per-position SL triggered | 2023-09-05 10:30:00 | 1322.28 | 1327.00 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-09-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 10:30:00 | 1338.40 | 1342.46 | 0.00 | ORB-short ORB[1339.50,1349.45] vol=2.0x ATR=2.87 |
| Stop hit — per-position SL triggered | 2023-09-08 11:35:00 | 1341.27 | 1339.15 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-09-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 10:30:00 | 1349.25 | 1346.64 | 0.00 | ORB-long ORB[1337.00,1348.80] vol=1.6x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 11:20:00 | 1353.47 | 1349.07 | 0.00 | T1 1.5R @ 1353.47 |
| Stop hit — per-position SL triggered | 2023-09-11 11:35:00 | 1349.25 | 1349.27 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:45:00 | 1342.95 | 1345.58 | 0.00 | ORB-short ORB[1344.00,1352.80] vol=1.5x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:50:00 | 1338.53 | 1342.66 | 0.00 | T1 1.5R @ 1338.53 |
| Target hit | 2023-09-12 12:40:00 | 1338.15 | 1338.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — SELL (started 2023-09-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-13 10:20:00 | 1330.75 | 1337.98 | 0.00 | ORB-short ORB[1342.40,1351.05] vol=2.6x ATR=3.66 |
| Stop hit — per-position SL triggered | 2023-09-13 10:35:00 | 1334.41 | 1337.12 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:30:00 | 1351.10 | 1348.30 | 0.00 | ORB-long ORB[1342.25,1350.00] vol=2.2x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 09:45:00 | 1355.59 | 1351.63 | 0.00 | T1 1.5R @ 1355.59 |
| Target hit | 2023-09-14 10:40:00 | 1354.20 | 1354.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2023-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-18 09:30:00 | 1370.10 | 1380.58 | 0.00 | ORB-short ORB[1373.10,1388.25] vol=1.6x ATR=4.23 |
| Stop hit — per-position SL triggered | 2023-09-18 09:35:00 | 1374.33 | 1380.08 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 11:15:00 | 1284.45 | 1278.97 | 0.00 | ORB-long ORB[1274.20,1282.95] vol=1.7x ATR=2.20 |
| Stop hit — per-position SL triggered | 2023-10-09 11:20:00 | 1282.25 | 1279.37 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-10-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-11 10:40:00 | 1294.00 | 1296.09 | 0.00 | ORB-short ORB[1295.00,1310.00] vol=2.8x ATR=2.46 |
| Stop hit — per-position SL triggered | 2023-10-11 12:25:00 | 1296.46 | 1295.88 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:35:00 | 1332.15 | 1329.94 | 0.00 | ORB-long ORB[1325.00,1332.00] vol=1.7x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 09:50:00 | 1336.84 | 1331.68 | 0.00 | T1 1.5R @ 1336.84 |
| Stop hit — per-position SL triggered | 2023-10-17 10:00:00 | 1332.15 | 1332.47 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-10-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 11:10:00 | 1350.35 | 1347.39 | 0.00 | ORB-long ORB[1339.00,1347.00] vol=4.2x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:30:00 | 1354.07 | 1347.98 | 0.00 | T1 1.5R @ 1354.07 |
| Stop hit — per-position SL triggered | 2023-10-18 11:40:00 | 1350.35 | 1348.19 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-10-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:25:00 | 1353.05 | 1359.79 | 0.00 | ORB-short ORB[1355.00,1366.55] vol=1.6x ATR=3.03 |
| Stop hit — per-position SL triggered | 2023-10-23 10:35:00 | 1356.08 | 1359.42 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 10:10:00 | 1296.00 | 1305.60 | 0.00 | ORB-short ORB[1305.10,1315.80] vol=1.6x ATR=3.67 |
| Stop hit — per-position SL triggered | 2023-10-26 10:30:00 | 1299.67 | 1302.60 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-11-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 10:50:00 | 1355.95 | 1363.49 | 0.00 | ORB-short ORB[1366.00,1377.95] vol=4.1x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 12:15:00 | 1351.79 | 1361.40 | 0.00 | T1 1.5R @ 1351.79 |
| Target hit | 2023-11-01 15:20:00 | 1333.30 | 1351.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2023-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 09:35:00 | 1331.30 | 1333.50 | 0.00 | ORB-short ORB[1331.35,1339.15] vol=2.6x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 10:00:00 | 1326.96 | 1331.53 | 0.00 | T1 1.5R @ 1326.96 |
| Target hit | 2023-11-07 10:40:00 | 1330.65 | 1330.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — BUY (started 2023-11-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 09:45:00 | 1356.95 | 1351.67 | 0.00 | ORB-long ORB[1343.80,1354.70] vol=1.8x ATR=3.25 |
| Stop hit — per-position SL triggered | 2023-11-10 10:05:00 | 1353.70 | 1352.12 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-11-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 09:35:00 | 1338.25 | 1345.45 | 0.00 | ORB-short ORB[1345.35,1359.70] vol=2.1x ATR=3.60 |
| Stop hit — per-position SL triggered | 2023-11-13 09:45:00 | 1341.85 | 1343.94 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:50:00 | 1408.55 | 1396.68 | 0.00 | ORB-long ORB[1388.25,1398.00] vol=3.2x ATR=4.06 |
| Stop hit — per-position SL triggered | 2023-11-21 09:55:00 | 1404.49 | 1397.93 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-11-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 11:10:00 | 1418.30 | 1426.39 | 0.00 | ORB-short ORB[1423.30,1434.35] vol=2.0x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 11:35:00 | 1413.98 | 1425.55 | 0.00 | T1 1.5R @ 1413.98 |
| Target hit | 2023-11-23 15:20:00 | 1413.95 | 1418.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2023-12-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-04 10:10:00 | 1422.80 | 1425.81 | 0.00 | ORB-short ORB[1428.00,1440.00] vol=5.3x ATR=3.93 |
| Stop hit — per-position SL triggered | 2023-12-04 10:40:00 | 1426.73 | 1424.79 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-12-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 10:35:00 | 1463.00 | 1454.41 | 0.00 | ORB-long ORB[1443.00,1458.00] vol=1.6x ATR=5.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 11:20:00 | 1471.07 | 1459.38 | 0.00 | T1 1.5R @ 1471.07 |
| Stop hit — per-position SL triggered | 2023-12-05 11:25:00 | 1463.00 | 1459.41 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-12-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 10:40:00 | 1448.80 | 1453.38 | 0.00 | ORB-short ORB[1452.05,1462.95] vol=1.7x ATR=2.79 |
| Stop hit — per-position SL triggered | 2023-12-06 12:40:00 | 1451.59 | 1451.56 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 10:30:00 | 1466.90 | 1469.19 | 0.00 | ORB-short ORB[1474.75,1483.15] vol=2.8x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 12:15:00 | 1461.21 | 1465.37 | 0.00 | T1 1.5R @ 1461.21 |
| Target hit | 2023-12-13 13:30:00 | 1466.20 | 1464.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — SELL (started 2023-12-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 10:30:00 | 1458.65 | 1467.86 | 0.00 | ORB-short ORB[1461.20,1477.00] vol=3.7x ATR=4.91 |
| Stop hit — per-position SL triggered | 2023-12-14 10:45:00 | 1463.56 | 1467.08 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-12-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 10:35:00 | 1450.25 | 1459.88 | 0.00 | ORB-short ORB[1463.40,1485.00] vol=2.8x ATR=3.91 |
| Stop hit — per-position SL triggered | 2023-12-15 10:55:00 | 1454.16 | 1458.40 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-12-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 10:55:00 | 1458.55 | 1455.07 | 0.00 | ORB-long ORB[1443.00,1457.10] vol=11.2x ATR=3.38 |
| Stop hit — per-position SL triggered | 2023-12-18 11:25:00 | 1455.17 | 1455.43 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-12-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-26 10:40:00 | 1392.60 | 1400.68 | 0.00 | ORB-short ORB[1397.15,1408.00] vol=2.4x ATR=3.68 |
| Stop hit — per-position SL triggered | 2023-12-26 11:05:00 | 1396.28 | 1398.50 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 10:50:00 | 1421.50 | 1413.81 | 0.00 | ORB-long ORB[1399.10,1410.30] vol=1.9x ATR=2.86 |
| Stop hit — per-position SL triggered | 2023-12-27 11:00:00 | 1418.64 | 1414.69 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-12-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:40:00 | 1439.00 | 1433.37 | 0.00 | ORB-long ORB[1425.55,1437.40] vol=1.7x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 09:55:00 | 1444.90 | 1436.56 | 0.00 | T1 1.5R @ 1444.90 |
| Stop hit — per-position SL triggered | 2023-12-28 10:00:00 | 1439.00 | 1436.64 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 1424.70 | 1436.47 | 0.00 | ORB-short ORB[1429.05,1449.70] vol=4.4x ATR=5.02 |
| Stop hit — per-position SL triggered | 2024-01-02 10:05:00 | 1429.72 | 1435.69 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 10:55:00 | 1440.05 | 1446.38 | 0.00 | ORB-short ORB[1446.60,1460.10] vol=3.2x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 11:05:00 | 1435.87 | 1444.79 | 0.00 | T1 1.5R @ 1435.87 |
| Target hit | 2024-01-08 15:20:00 | 1421.85 | 1433.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2024-01-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 11:05:00 | 1421.95 | 1424.70 | 0.00 | ORB-short ORB[1422.00,1434.80] vol=1.9x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-01-12 11:30:00 | 1424.22 | 1424.58 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-01-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:50:00 | 1427.30 | 1430.48 | 0.00 | ORB-short ORB[1433.45,1444.55] vol=5.5x ATR=3.75 |
| Stop hit — per-position SL triggered | 2024-01-15 10:00:00 | 1431.05 | 1430.40 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-02-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 10:00:00 | 1486.00 | 1481.17 | 0.00 | ORB-long ORB[1473.50,1485.30] vol=1.8x ATR=4.51 |
| Stop hit — per-position SL triggered | 2024-02-08 10:05:00 | 1481.49 | 1481.25 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 09:35:00 | 1446.05 | 1440.27 | 0.00 | ORB-long ORB[1428.20,1444.25] vol=3.5x ATR=6.29 |
| Stop hit — per-position SL triggered | 2024-02-13 10:00:00 | 1439.76 | 1441.48 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-02-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 10:40:00 | 1479.15 | 1467.36 | 0.00 | ORB-long ORB[1451.60,1463.90] vol=1.8x ATR=3.71 |
| Stop hit — per-position SL triggered | 2024-02-16 10:50:00 | 1475.44 | 1469.87 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-02-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 10:05:00 | 1495.80 | 1484.57 | 0.00 | ORB-long ORB[1473.35,1488.85] vol=2.5x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-02-20 10:10:00 | 1491.07 | 1486.47 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-02-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 10:00:00 | 1481.05 | 1490.96 | 0.00 | ORB-short ORB[1491.15,1512.60] vol=1.8x ATR=6.30 |
| Stop hit — per-position SL triggered | 2024-02-22 10:10:00 | 1487.35 | 1489.92 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-02-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 11:10:00 | 1514.40 | 1507.94 | 0.00 | ORB-long ORB[1501.25,1510.85] vol=3.1x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-23 11:20:00 | 1519.27 | 1513.01 | 0.00 | T1 1.5R @ 1519.27 |
| Target hit | 2024-02-23 15:20:00 | 1530.10 | 1526.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2024-02-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 11:15:00 | 1545.90 | 1543.03 | 0.00 | ORB-long ORB[1525.40,1540.40] vol=1.6x ATR=5.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 13:55:00 | 1553.42 | 1545.52 | 0.00 | T1 1.5R @ 1553.42 |
| Stop hit — per-position SL triggered | 2024-02-29 14:10:00 | 1545.90 | 1545.67 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-03-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:40:00 | 1529.70 | 1536.58 | 0.00 | ORB-short ORB[1531.30,1554.00] vol=2.1x ATR=4.12 |
| Stop hit — per-position SL triggered | 2024-03-04 10:15:00 | 1533.82 | 1533.37 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 10:15:00 | 1500.20 | 1508.18 | 0.00 | ORB-short ORB[1510.20,1521.30] vol=3.6x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 11:15:00 | 1493.77 | 1503.72 | 0.00 | T1 1.5R @ 1493.77 |
| Stop hit — per-position SL triggered | 2024-03-07 13:45:00 | 1500.20 | 1499.51 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 09:30:00 | 1548.05 | 1538.80 | 0.00 | ORB-long ORB[1530.00,1539.05] vol=2.6x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-03-12 09:35:00 | 1543.66 | 1542.20 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-03-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 10:20:00 | 1512.30 | 1517.83 | 0.00 | ORB-short ORB[1514.90,1532.60] vol=2.7x ATR=5.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:05:00 | 1504.40 | 1516.76 | 0.00 | T1 1.5R @ 1504.40 |
| Target hit | 2024-03-13 12:55:00 | 1506.05 | 1505.01 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — SELL (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 11:15:00 | 1484.60 | 1491.73 | 0.00 | ORB-short ORB[1486.25,1503.45] vol=2.8x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-18 11:35:00 | 1479.00 | 1490.54 | 0.00 | T1 1.5R @ 1479.00 |
| Stop hit — per-position SL triggered | 2024-03-18 12:00:00 | 1484.60 | 1489.49 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-03-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 11:05:00 | 1464.00 | 1474.50 | 0.00 | ORB-short ORB[1472.50,1488.95] vol=1.6x ATR=4.48 |
| Stop hit — per-position SL triggered | 2024-03-19 11:20:00 | 1468.48 | 1472.82 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-03-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:45:00 | 1453.90 | 1457.63 | 0.00 | ORB-short ORB[1458.05,1466.50] vol=2.5x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-03-20 10:55:00 | 1457.46 | 1457.52 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-03-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 11:15:00 | 1479.05 | 1468.53 | 0.00 | ORB-long ORB[1467.35,1478.80] vol=4.3x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 11:20:00 | 1484.51 | 1470.95 | 0.00 | T1 1.5R @ 1484.51 |
| Target hit | 2024-03-22 15:20:00 | 1502.40 | 1488.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2024-03-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 10:30:00 | 1493.85 | 1492.16 | 0.00 | ORB-long ORB[1480.55,1493.10] vol=4.7x ATR=3.53 |
| Stop hit — per-position SL triggered | 2024-03-27 11:00:00 | 1490.32 | 1492.41 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:50:00 | 1448.40 | 1454.43 | 0.00 | ORB-short ORB[1456.80,1468.80] vol=4.0x ATR=4.10 |
| Stop hit — per-position SL triggered | 2024-04-04 09:55:00 | 1452.50 | 1454.12 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-04-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 10:10:00 | 1501.00 | 1495.59 | 0.00 | ORB-long ORB[1487.00,1495.70] vol=1.6x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 10:30:00 | 1506.21 | 1498.47 | 0.00 | T1 1.5R @ 1506.21 |
| Target hit | 2024-04-08 15:20:00 | 1520.35 | 1513.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — SELL (started 2024-04-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 10:35:00 | 1487.25 | 1500.41 | 0.00 | ORB-short ORB[1505.00,1516.70] vol=2.2x ATR=4.64 |
| Stop hit — per-position SL triggered | 2024-04-10 10:45:00 | 1491.89 | 1496.87 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-04-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 10:20:00 | 1507.05 | 1502.68 | 0.00 | ORB-long ORB[1491.35,1504.75] vol=1.5x ATR=4.22 |
| Stop hit — per-position SL triggered | 2024-04-12 10:50:00 | 1502.83 | 1503.55 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-04-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:45:00 | 1460.00 | 1467.64 | 0.00 | ORB-short ORB[1464.30,1476.60] vol=2.4x ATR=5.11 |
| Stop hit — per-position SL triggered | 2024-04-18 09:50:00 | 1465.11 | 1467.27 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-04-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:10:00 | 1444.00 | 1445.85 | 0.00 | ORB-short ORB[1444.95,1464.00] vol=1.7x ATR=4.35 |
| Stop hit — per-position SL triggered | 2024-04-25 10:30:00 | 1448.35 | 1445.61 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-04-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-26 09:40:00 | 1440.85 | 1447.48 | 0.00 | ORB-short ORB[1447.85,1466.85] vol=1.7x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 12:00:00 | 1431.76 | 1442.35 | 0.00 | T1 1.5R @ 1431.76 |
| Target hit | 2024-04-26 15:20:00 | 1414.00 | 1428.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — BUY (started 2024-04-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 10:50:00 | 1442.85 | 1431.42 | 0.00 | ORB-long ORB[1423.20,1434.20] vol=1.9x ATR=4.01 |
| Stop hit — per-position SL triggered | 2024-04-30 10:55:00 | 1438.84 | 1431.88 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 10:45:00 | 1440.95 | 1445.50 | 0.00 | ORB-short ORB[1446.60,1457.45] vol=1.7x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 13:30:00 | 1434.77 | 1443.31 | 0.00 | T1 1.5R @ 1434.77 |
| Stop hit — per-position SL triggered | 2024-05-06 14:10:00 | 1440.95 | 1442.64 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-05-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 09:50:00 | 1438.10 | 1438.45 | 0.00 | ORB-short ORB[1438.55,1449.00] vol=8.0x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-05-07 10:40:00 | 1442.49 | 1437.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-17 10:55:00 | 1173.05 | 2023-05-17 11:35:00 | 1169.57 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-05-17 10:55:00 | 1173.05 | 2023-05-17 15:20:00 | 1161.65 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2023-05-22 10:55:00 | 1157.70 | 2023-05-22 11:00:00 | 1155.58 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-05-23 09:45:00 | 1145.30 | 2023-05-23 09:50:00 | 1148.35 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-05-25 09:45:00 | 1178.10 | 2023-05-25 09:55:00 | 1176.54 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-05-26 11:05:00 | 1190.30 | 2023-05-26 12:00:00 | 1193.74 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-05-26 11:05:00 | 1190.30 | 2023-05-26 15:20:00 | 1195.00 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2023-05-31 10:05:00 | 1231.05 | 2023-05-31 10:30:00 | 1228.54 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-01 11:15:00 | 1227.30 | 2023-06-01 12:20:00 | 1224.22 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-06-01 11:15:00 | 1227.30 | 2023-06-01 15:20:00 | 1208.45 | TARGET_HIT | 0.50 | 1.54% |
| BUY | retest1 | 2023-06-16 09:50:00 | 1249.00 | 2023-06-16 10:05:00 | 1253.30 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-06-16 09:50:00 | 1249.00 | 2023-06-16 15:20:00 | 1280.25 | TARGET_HIT | 0.50 | 2.50% |
| BUY | retest1 | 2023-06-20 09:30:00 | 1295.00 | 2023-06-20 09:40:00 | 1300.58 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-06-20 09:30:00 | 1295.00 | 2023-06-20 11:20:00 | 1296.90 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2023-06-23 10:00:00 | 1274.00 | 2023-06-23 10:05:00 | 1270.59 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-06-27 09:50:00 | 1263.25 | 2023-06-27 09:55:00 | 1259.88 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-06-27 09:50:00 | 1263.25 | 2023-06-27 11:55:00 | 1263.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-03 11:05:00 | 1299.75 | 2023-07-03 11:20:00 | 1296.30 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-07-03 11:05:00 | 1299.75 | 2023-07-03 11:35:00 | 1299.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-05 09:55:00 | 1304.20 | 2023-07-05 10:10:00 | 1309.42 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-07-05 09:55:00 | 1304.20 | 2023-07-05 10:35:00 | 1304.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-07 11:05:00 | 1311.95 | 2023-07-07 11:15:00 | 1315.98 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-07-07 11:05:00 | 1311.95 | 2023-07-07 11:20:00 | 1311.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-10 10:50:00 | 1285.00 | 2023-07-10 12:30:00 | 1288.38 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-12 11:00:00 | 1305.90 | 2023-07-12 11:05:00 | 1303.52 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-07-13 10:50:00 | 1328.10 | 2023-07-13 11:00:00 | 1331.84 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-07-13 10:50:00 | 1328.10 | 2023-07-13 12:35:00 | 1328.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-20 09:40:00 | 1309.95 | 2023-07-20 11:05:00 | 1305.33 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-07-20 09:40:00 | 1309.95 | 2023-07-20 12:05:00 | 1309.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-24 10:15:00 | 1309.00 | 2023-07-24 11:10:00 | 1314.09 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-07-24 10:15:00 | 1309.00 | 2023-07-24 11:45:00 | 1309.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-01 10:45:00 | 1264.95 | 2023-08-01 11:05:00 | 1267.67 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-08-03 10:55:00 | 1271.70 | 2023-08-03 11:05:00 | 1268.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-08-11 09:30:00 | 1320.20 | 2023-08-11 09:50:00 | 1314.40 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-08-11 09:30:00 | 1320.20 | 2023-08-11 09:55:00 | 1320.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-22 09:30:00 | 1296.35 | 2023-08-22 09:40:00 | 1301.51 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-08-22 09:30:00 | 1296.35 | 2023-08-22 11:50:00 | 1302.60 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2023-08-25 11:10:00 | 1283.65 | 2023-08-25 11:20:00 | 1286.81 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-01 11:10:00 | 1323.30 | 2023-09-01 11:35:00 | 1320.30 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-09-04 10:50:00 | 1305.30 | 2023-09-04 11:40:00 | 1308.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-09-05 10:15:00 | 1318.95 | 2023-09-05 10:30:00 | 1322.28 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-09-08 10:30:00 | 1338.40 | 2023-09-08 11:35:00 | 1341.27 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-09-11 10:30:00 | 1349.25 | 2023-09-11 11:20:00 | 1353.47 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-09-11 10:30:00 | 1349.25 | 2023-09-11 11:35:00 | 1349.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-12 09:45:00 | 1342.95 | 2023-09-12 09:50:00 | 1338.53 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-09-12 09:45:00 | 1342.95 | 2023-09-12 12:40:00 | 1338.15 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2023-09-13 10:20:00 | 1330.75 | 2023-09-13 10:35:00 | 1334.41 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-09-14 09:30:00 | 1351.10 | 2023-09-14 09:45:00 | 1355.59 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-09-14 09:30:00 | 1351.10 | 2023-09-14 10:40:00 | 1354.20 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2023-09-18 09:30:00 | 1370.10 | 2023-09-18 09:35:00 | 1374.33 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-10-09 11:15:00 | 1284.45 | 2023-10-09 11:20:00 | 1282.25 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-10-11 10:40:00 | 1294.00 | 2023-10-11 12:25:00 | 1296.46 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-17 09:35:00 | 1332.15 | 2023-10-17 09:50:00 | 1336.84 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-10-17 09:35:00 | 1332.15 | 2023-10-17 10:00:00 | 1332.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-18 11:10:00 | 1350.35 | 2023-10-18 11:30:00 | 1354.07 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-10-18 11:10:00 | 1350.35 | 2023-10-18 11:40:00 | 1350.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-23 10:25:00 | 1353.05 | 2023-10-23 10:35:00 | 1356.08 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-10-26 10:10:00 | 1296.00 | 2023-10-26 10:30:00 | 1299.67 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-11-01 10:50:00 | 1355.95 | 2023-11-01 12:15:00 | 1351.79 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-11-01 10:50:00 | 1355.95 | 2023-11-01 15:20:00 | 1333.30 | TARGET_HIT | 0.50 | 1.67% |
| SELL | retest1 | 2023-11-07 09:35:00 | 1331.30 | 2023-11-07 10:00:00 | 1326.96 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-11-07 09:35:00 | 1331.30 | 2023-11-07 10:40:00 | 1330.65 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2023-11-10 09:45:00 | 1356.95 | 2023-11-10 10:05:00 | 1353.70 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-11-13 09:35:00 | 1338.25 | 2023-11-13 09:45:00 | 1341.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-11-21 09:50:00 | 1408.55 | 2023-11-21 09:55:00 | 1404.49 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-23 11:10:00 | 1418.30 | 2023-11-23 11:35:00 | 1413.98 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-11-23 11:10:00 | 1418.30 | 2023-11-23 15:20:00 | 1413.95 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2023-12-04 10:10:00 | 1422.80 | 2023-12-04 10:40:00 | 1426.73 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-12-05 10:35:00 | 1463.00 | 2023-12-05 11:20:00 | 1471.07 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2023-12-05 10:35:00 | 1463.00 | 2023-12-05 11:25:00 | 1463.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-06 10:40:00 | 1448.80 | 2023-12-06 12:40:00 | 1451.59 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-12-13 10:30:00 | 1466.90 | 2023-12-13 12:15:00 | 1461.21 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-12-13 10:30:00 | 1466.90 | 2023-12-13 13:30:00 | 1466.20 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2023-12-14 10:30:00 | 1458.65 | 2023-12-14 10:45:00 | 1463.56 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-12-15 10:35:00 | 1450.25 | 2023-12-15 10:55:00 | 1454.16 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-12-18 10:55:00 | 1458.55 | 2023-12-18 11:25:00 | 1455.17 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-12-26 10:40:00 | 1392.60 | 2023-12-26 11:05:00 | 1396.28 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-27 10:50:00 | 1421.50 | 2023-12-27 11:00:00 | 1418.64 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-12-28 09:40:00 | 1439.00 | 2023-12-28 09:55:00 | 1444.90 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-12-28 09:40:00 | 1439.00 | 2023-12-28 10:00:00 | 1439.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 09:55:00 | 1424.70 | 2024-01-02 10:05:00 | 1429.72 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-01-08 10:55:00 | 1440.05 | 2024-01-08 11:05:00 | 1435.87 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-01-08 10:55:00 | 1440.05 | 2024-01-08 15:20:00 | 1421.85 | TARGET_HIT | 0.50 | 1.26% |
| SELL | retest1 | 2024-01-12 11:05:00 | 1421.95 | 2024-01-12 11:30:00 | 1424.22 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-01-15 09:50:00 | 1427.30 | 2024-01-15 10:00:00 | 1431.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-02-08 10:00:00 | 1486.00 | 2024-02-08 10:05:00 | 1481.49 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-02-13 09:35:00 | 1446.05 | 2024-02-13 10:00:00 | 1439.76 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-02-16 10:40:00 | 1479.15 | 2024-02-16 10:50:00 | 1475.44 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-02-20 10:05:00 | 1495.80 | 2024-02-20 10:10:00 | 1491.07 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-02-22 10:00:00 | 1481.05 | 2024-02-22 10:10:00 | 1487.35 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-02-23 11:10:00 | 1514.40 | 2024-02-23 11:20:00 | 1519.27 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-02-23 11:10:00 | 1514.40 | 2024-02-23 15:20:00 | 1530.10 | TARGET_HIT | 0.50 | 1.04% |
| BUY | retest1 | 2024-02-29 11:15:00 | 1545.90 | 2024-02-29 13:55:00 | 1553.42 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-02-29 11:15:00 | 1545.90 | 2024-02-29 14:10:00 | 1545.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-04 09:40:00 | 1529.70 | 2024-03-04 10:15:00 | 1533.82 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-03-07 10:15:00 | 1500.20 | 2024-03-07 11:15:00 | 1493.77 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-03-07 10:15:00 | 1500.20 | 2024-03-07 13:45:00 | 1500.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-12 09:30:00 | 1548.05 | 2024-03-12 09:35:00 | 1543.66 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-03-13 10:20:00 | 1512.30 | 2024-03-13 11:05:00 | 1504.40 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-03-13 10:20:00 | 1512.30 | 2024-03-13 12:55:00 | 1506.05 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2024-03-18 11:15:00 | 1484.60 | 2024-03-18 11:35:00 | 1479.00 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-03-18 11:15:00 | 1484.60 | 2024-03-18 12:00:00 | 1484.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-19 11:05:00 | 1464.00 | 2024-03-19 11:20:00 | 1468.48 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-03-20 10:45:00 | 1453.90 | 2024-03-20 10:55:00 | 1457.46 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-03-22 11:15:00 | 1479.05 | 2024-03-22 11:20:00 | 1484.51 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-03-22 11:15:00 | 1479.05 | 2024-03-22 15:20:00 | 1502.40 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2024-03-27 10:30:00 | 1493.85 | 2024-03-27 11:00:00 | 1490.32 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-04 09:50:00 | 1448.40 | 2024-04-04 09:55:00 | 1452.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-04-08 10:10:00 | 1501.00 | 2024-04-08 10:30:00 | 1506.21 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-04-08 10:10:00 | 1501.00 | 2024-04-08 15:20:00 | 1520.35 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2024-04-10 10:35:00 | 1487.25 | 2024-04-10 10:45:00 | 1491.89 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-04-12 10:20:00 | 1507.05 | 2024-04-12 10:50:00 | 1502.83 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-04-18 09:45:00 | 1460.00 | 2024-04-18 09:50:00 | 1465.11 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-04-25 10:10:00 | 1444.00 | 2024-04-25 10:30:00 | 1448.35 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-04-26 09:40:00 | 1440.85 | 2024-04-26 12:00:00 | 1431.76 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-04-26 09:40:00 | 1440.85 | 2024-04-26 15:20:00 | 1414.00 | TARGET_HIT | 0.50 | 1.86% |
| BUY | retest1 | 2024-04-30 10:50:00 | 1442.85 | 2024-04-30 10:55:00 | 1438.84 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-05-06 10:45:00 | 1440.95 | 2024-05-06 13:30:00 | 1434.77 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-06 10:45:00 | 1440.95 | 2024-05-06 14:10:00 | 1440.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-07 09:50:00 | 1438.10 | 2024-05-07 10:40:00 | 1442.49 | STOP_HIT | 1.00 | -0.30% |
