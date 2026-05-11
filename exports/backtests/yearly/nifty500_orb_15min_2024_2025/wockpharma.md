# Wockhardt Ltd. (WOCKPHARMA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1611.50
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
| ENTRY1 | 32 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 28
- **Target hits / Stop hits / Partials:** 4 / 28 / 11
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 7.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 5 | 25.0% | 2 | 15 | 3 | -0.07% | -1.5% |
| BUY @ 2nd Alert (retest1) | 20 | 5 | 25.0% | 2 | 15 | 3 | -0.07% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 10 | 43.5% | 2 | 13 | 8 | 0.37% | 8.5% |
| SELL @ 2nd Alert (retest1) | 23 | 10 | 43.5% | 2 | 13 | 8 | 0.37% | 8.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 43 | 15 | 34.9% | 4 | 28 | 11 | 0.16% | 7.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:40:00 | 548.85 | 545.88 | 0.00 | ORB-long ORB[541.00,544.30] vol=1.5x ATR=2.72 |
| Stop hit — per-position SL triggered | 2024-05-18 09:45:00 | 546.13 | 545.93 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:15:00 | 552.40 | 548.44 | 0.00 | ORB-long ORB[542.00,549.00] vol=5.2x ATR=3.21 |
| Stop hit — per-position SL triggered | 2024-05-23 10:20:00 | 549.19 | 548.57 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:35:00 | 585.70 | 578.93 | 0.00 | ORB-long ORB[572.00,580.00] vol=2.6x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 10:40:00 | 589.61 | 580.60 | 0.00 | T1 1.5R @ 589.61 |
| Stop hit — per-position SL triggered | 2024-06-11 10:50:00 | 585.70 | 581.83 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:40:00 | 597.00 | 591.04 | 0.00 | ORB-long ORB[586.35,591.90] vol=6.5x ATR=2.63 |
| Stop hit — per-position SL triggered | 2024-06-12 10:45:00 | 594.37 | 591.33 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:40:00 | 586.85 | 591.41 | 0.00 | ORB-short ORB[591.75,599.00] vol=1.9x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:50:00 | 582.08 | 589.83 | 0.00 | T1 1.5R @ 582.08 |
| Stop hit — per-position SL triggered | 2024-06-13 09:55:00 | 586.85 | 589.66 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-20 10:20:00 | 572.15 | 576.63 | 0.00 | ORB-short ORB[575.00,581.15] vol=2.4x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 13:35:00 | 567.38 | 573.33 | 0.00 | T1 1.5R @ 567.38 |
| Stop hit — per-position SL triggered | 2024-06-20 14:15:00 | 572.15 | 573.02 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 11:15:00 | 995.00 | 982.04 | 0.00 | ORB-long ORB[971.00,985.20] vol=7.1x ATR=4.61 |
| Stop hit — per-position SL triggered | 2024-08-23 11:25:00 | 990.39 | 984.80 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-09-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:10:00 | 1074.00 | 1081.24 | 0.00 | ORB-short ORB[1079.00,1089.90] vol=1.6x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:30:00 | 1067.13 | 1080.08 | 0.00 | T1 1.5R @ 1067.13 |
| Stop hit — per-position SL triggered | 2024-09-06 11:25:00 | 1074.00 | 1077.14 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-09-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:10:00 | 1004.85 | 1007.11 | 0.00 | ORB-short ORB[1006.05,1015.00] vol=1.6x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:25:00 | 997.44 | 1005.89 | 0.00 | T1 1.5R @ 997.44 |
| Stop hit — per-position SL triggered | 2024-09-24 10:35:00 | 1004.85 | 1005.33 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-10-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 09:35:00 | 970.00 | 975.66 | 0.00 | ORB-short ORB[975.00,987.00] vol=2.3x ATR=9.75 |
| Stop hit — per-position SL triggered | 2024-10-01 09:40:00 | 979.75 | 975.76 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 09:35:00 | 1261.20 | 1273.15 | 0.00 | ORB-short ORB[1266.30,1282.80] vol=1.6x ATR=6.84 |
| Stop hit — per-position SL triggered | 2024-11-06 09:40:00 | 1268.04 | 1272.49 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-11-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 09:30:00 | 1354.95 | 1361.58 | 0.00 | ORB-short ORB[1355.00,1373.70] vol=1.8x ATR=6.75 |
| Stop hit — per-position SL triggered | 2024-11-29 09:45:00 | 1361.70 | 1359.30 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-12-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:25:00 | 1441.05 | 1430.24 | 0.00 | ORB-long ORB[1420.00,1439.90] vol=1.9x ATR=8.18 |
| Stop hit — per-position SL triggered | 2024-12-03 10:45:00 | 1432.87 | 1431.15 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-12-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:35:00 | 1414.15 | 1408.21 | 0.00 | ORB-long ORB[1395.05,1412.00] vol=2.1x ATR=8.40 |
| Stop hit — per-position SL triggered | 2024-12-10 09:40:00 | 1405.75 | 1408.22 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-12-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:30:00 | 1416.95 | 1406.49 | 0.00 | ORB-long ORB[1395.15,1412.65] vol=1.8x ATR=5.79 |
| Stop hit — per-position SL triggered | 2024-12-11 10:40:00 | 1411.16 | 1406.85 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 1420.00 | 1428.89 | 0.00 | ORB-short ORB[1424.90,1443.00] vol=2.1x ATR=6.51 |
| Stop hit — per-position SL triggered | 2024-12-12 10:05:00 | 1426.51 | 1426.73 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-12-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:40:00 | 1363.00 | 1372.83 | 0.00 | ORB-short ORB[1375.10,1388.00] vol=2.7x ATR=6.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:10:00 | 1352.98 | 1367.18 | 0.00 | T1 1.5R @ 1352.98 |
| Stop hit — per-position SL triggered | 2024-12-13 10:50:00 | 1363.00 | 1361.84 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 1448.00 | 1431.79 | 0.00 | ORB-long ORB[1414.40,1429.50] vol=1.6x ATR=7.09 |
| Stop hit — per-position SL triggered | 2025-01-01 11:10:00 | 1440.91 | 1434.00 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:50:00 | 1446.40 | 1455.41 | 0.00 | ORB-short ORB[1455.75,1466.05] vol=1.7x ATR=6.40 |
| Stop hit — per-position SL triggered | 2025-01-02 10:00:00 | 1452.80 | 1454.82 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-01-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:10:00 | 1458.25 | 1466.40 | 0.00 | ORB-short ORB[1462.55,1484.10] vol=1.6x ATR=8.64 |
| Stop hit — per-position SL triggered | 2025-01-07 10:15:00 | 1466.89 | 1466.30 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-01-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:10:00 | 1448.35 | 1452.78 | 0.00 | ORB-short ORB[1451.05,1466.20] vol=1.8x ATR=6.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:00:00 | 1438.14 | 1450.75 | 0.00 | T1 1.5R @ 1438.14 |
| Stop hit — per-position SL triggered | 2025-01-09 11:20:00 | 1448.35 | 1450.10 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-01-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:35:00 | 1375.90 | 1364.70 | 0.00 | ORB-long ORB[1350.05,1367.95] vol=2.7x ATR=6.82 |
| Stop hit — per-position SL triggered | 2025-01-16 10:00:00 | 1369.08 | 1371.72 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-01-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:45:00 | 1429.20 | 1417.67 | 0.00 | ORB-long ORB[1398.05,1418.40] vol=2.5x ATR=6.85 |
| Stop hit — per-position SL triggered | 2025-01-20 09:50:00 | 1422.35 | 1418.34 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-01-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:45:00 | 1372.00 | 1382.32 | 0.00 | ORB-short ORB[1383.05,1392.85] vol=1.6x ATR=5.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:55:00 | 1363.64 | 1378.94 | 0.00 | T1 1.5R @ 1363.64 |
| Target hit | 2025-01-21 15:20:00 | 1341.00 | 1347.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-01-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 11:10:00 | 1304.00 | 1303.60 | 0.00 | ORB-long ORB[1284.55,1298.00] vol=2.5x ATR=6.73 |
| Stop hit — per-position SL triggered | 2025-01-23 11:30:00 | 1297.27 | 1303.47 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-01-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:50:00 | 1337.75 | 1302.99 | 0.00 | ORB-long ORB[1276.20,1294.80] vol=2.9x ATR=10.44 |
| Stop hit — per-position SL triggered | 2025-01-29 11:00:00 | 1327.31 | 1306.35 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 1373.75 | 1388.54 | 0.00 | ORB-short ORB[1386.20,1404.95] vol=2.1x ATR=8.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 10:05:00 | 1360.32 | 1379.79 | 0.00 | T1 1.5R @ 1360.32 |
| Target hit | 2025-02-21 15:20:00 | 1315.90 | 1341.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 09:35:00 | 1308.15 | 1315.11 | 0.00 | ORB-short ORB[1313.50,1326.80] vol=4.4x ATR=5.94 |
| Stop hit — per-position SL triggered | 2025-03-13 09:45:00 | 1314.09 | 1313.49 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:40:00 | 1312.00 | 1301.50 | 0.00 | ORB-long ORB[1288.05,1300.00] vol=2.4x ATR=5.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:55:00 | 1319.99 | 1306.58 | 0.00 | T1 1.5R @ 1319.99 |
| Target hit | 2025-03-18 10:55:00 | 1312.10 | 1314.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — BUY (started 2025-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:30:00 | 1497.65 | 1485.89 | 0.00 | ORB-long ORB[1470.00,1490.00] vol=3.0x ATR=9.48 |
| Stop hit — per-position SL triggered | 2025-03-21 09:35:00 | 1488.17 | 1485.66 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:00:00 | 1420.60 | 1405.23 | 0.00 | ORB-long ORB[1388.90,1402.00] vol=2.1x ATR=6.56 |
| Stop hit — per-position SL triggered | 2025-04-22 10:25:00 | 1414.04 | 1410.24 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-04-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:55:00 | 1434.30 | 1422.13 | 0.00 | ORB-long ORB[1407.70,1429.00] vol=2.2x ATR=6.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 10:05:00 | 1443.93 | 1435.47 | 0.00 | T1 1.5R @ 1443.93 |
| Target hit | 2025-04-24 13:00:00 | 1491.40 | 1491.60 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-18 09:40:00 | 548.85 | 2024-05-18 09:45:00 | 546.13 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-05-23 10:15:00 | 552.40 | 2024-05-23 10:20:00 | 549.19 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-06-11 10:35:00 | 585.70 | 2024-06-11 10:40:00 | 589.61 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-06-11 10:35:00 | 585.70 | 2024-06-11 10:50:00 | 585.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 10:40:00 | 597.00 | 2024-06-12 10:45:00 | 594.37 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-06-13 09:40:00 | 586.85 | 2024-06-13 09:50:00 | 582.08 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2024-06-13 09:40:00 | 586.85 | 2024-06-13 09:55:00 | 586.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-20 10:20:00 | 572.15 | 2024-06-20 13:35:00 | 567.38 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2024-06-20 10:20:00 | 572.15 | 2024-06-20 14:15:00 | 572.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-23 11:15:00 | 995.00 | 2024-08-23 11:25:00 | 990.39 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-09-06 10:10:00 | 1074.00 | 2024-09-06 10:30:00 | 1067.13 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-09-06 10:10:00 | 1074.00 | 2024-09-06 11:25:00 | 1074.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-24 10:10:00 | 1004.85 | 2024-09-24 10:25:00 | 997.44 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2024-09-24 10:10:00 | 1004.85 | 2024-09-24 10:35:00 | 1004.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-01 09:35:00 | 970.00 | 2024-10-01 09:40:00 | 979.75 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest1 | 2024-11-06 09:35:00 | 1261.20 | 2024-11-06 09:40:00 | 1268.04 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-11-29 09:30:00 | 1354.95 | 2024-11-29 09:45:00 | 1361.70 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-12-03 10:25:00 | 1441.05 | 2024-12-03 10:45:00 | 1432.87 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-12-10 09:35:00 | 1414.15 | 2024-12-10 09:40:00 | 1405.75 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-12-11 10:30:00 | 1416.95 | 2024-12-11 10:40:00 | 1411.16 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-12 09:40:00 | 1420.00 | 2024-12-12 10:05:00 | 1426.51 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-12-13 09:40:00 | 1363.00 | 2024-12-13 10:10:00 | 1352.98 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2024-12-13 09:40:00 | 1363.00 | 2024-12-13 10:50:00 | 1363.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 10:50:00 | 1448.00 | 2025-01-01 11:10:00 | 1440.91 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-01-02 09:50:00 | 1446.40 | 2025-01-02 10:00:00 | 1452.80 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-01-07 10:10:00 | 1458.25 | 2025-01-07 10:15:00 | 1466.89 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2025-01-09 10:10:00 | 1448.35 | 2025-01-09 11:00:00 | 1438.14 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2025-01-09 10:10:00 | 1448.35 | 2025-01-09 11:20:00 | 1448.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-16 09:35:00 | 1375.90 | 2025-01-16 10:00:00 | 1369.08 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-01-20 09:45:00 | 1429.20 | 2025-01-20 09:50:00 | 1422.35 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-01-21 09:45:00 | 1372.00 | 2025-01-21 09:55:00 | 1363.64 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-21 09:45:00 | 1372.00 | 2025-01-21 15:20:00 | 1341.00 | TARGET_HIT | 0.50 | 2.26% |
| BUY | retest1 | 2025-01-23 11:10:00 | 1304.00 | 2025-01-23 11:30:00 | 1297.27 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2025-01-29 10:50:00 | 1337.75 | 2025-01-29 11:00:00 | 1327.31 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest1 | 2025-02-21 09:40:00 | 1373.75 | 2025-02-21 10:05:00 | 1360.32 | PARTIAL | 0.50 | 0.98% |
| SELL | retest1 | 2025-02-21 09:40:00 | 1373.75 | 2025-02-21 15:20:00 | 1315.90 | TARGET_HIT | 0.50 | 4.21% |
| SELL | retest1 | 2025-03-13 09:35:00 | 1308.15 | 2025-03-13 09:45:00 | 1314.09 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-03-18 09:40:00 | 1312.00 | 2025-03-18 09:55:00 | 1319.99 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-03-18 09:40:00 | 1312.00 | 2025-03-18 10:55:00 | 1312.10 | TARGET_HIT | 0.50 | 0.01% |
| BUY | retest1 | 2025-03-21 09:30:00 | 1497.65 | 2025-03-21 09:35:00 | 1488.17 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2025-04-22 10:00:00 | 1420.60 | 2025-04-22 10:25:00 | 1414.04 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-04-24 09:55:00 | 1434.30 | 2025-04-24 10:05:00 | 1443.93 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-04-24 09:55:00 | 1434.30 | 2025-04-24 13:00:00 | 1491.40 | TARGET_HIT | 0.50 | 3.98% |
