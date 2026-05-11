# Signatureglobal (India) Ltd. (SIGNATURE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 903.00
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
| ENTRY1 | 86 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 19 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 122 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 67
- **Target hits / Stop hits / Partials:** 19 / 67 / 36
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 24.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 12 | 26.7% | 3 | 33 | 9 | -0.06% | -2.9% |
| BUY @ 2nd Alert (retest1) | 45 | 12 | 26.7% | 3 | 33 | 9 | -0.06% | -2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 77 | 43 | 55.8% | 16 | 34 | 27 | 0.35% | 27.0% |
| SELL @ 2nd Alert (retest1) | 77 | 43 | 55.8% | 16 | 34 | 27 | 0.35% | 27.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 122 | 55 | 45.1% | 19 | 67 | 36 | 0.20% | 24.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:55:00 | 1263.90 | 1266.19 | 0.00 | ORB-short ORB[1264.00,1271.10] vol=2.0x ATR=3.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 11:25:00 | 1258.18 | 1266.05 | 0.00 | T1 1.5R @ 1258.18 |
| Target hit | 2024-05-14 15:20:00 | 1254.90 | 1255.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2024-05-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:55:00 | 1263.00 | 1265.51 | 0.00 | ORB-short ORB[1263.30,1282.00] vol=1.9x ATR=5.29 |
| Stop hit — per-position SL triggered | 2024-05-16 10:05:00 | 1268.29 | 1265.56 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:45:00 | 1296.95 | 1302.40 | 0.00 | ORB-short ORB[1302.80,1314.00] vol=2.7x ATR=5.08 |
| Stop hit — per-position SL triggered | 2024-05-23 09:50:00 | 1302.03 | 1301.47 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 11:15:00 | 1280.50 | 1283.01 | 0.00 | ORB-short ORB[1282.25,1293.55] vol=3.3x ATR=4.30 |
| Stop hit — per-position SL triggered | 2024-05-28 11:30:00 | 1284.80 | 1282.22 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:15:00 | 1270.30 | 1277.27 | 0.00 | ORB-short ORB[1278.50,1290.95] vol=2.4x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:00:00 | 1264.49 | 1272.56 | 0.00 | T1 1.5R @ 1264.49 |
| Target hit | 2024-05-30 13:30:00 | 1264.45 | 1264.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2024-06-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:30:00 | 1300.40 | 1294.15 | 0.00 | ORB-long ORB[1263.50,1283.00] vol=2.0x ATR=8.86 |
| Stop hit — per-position SL triggered | 2024-06-10 10:45:00 | 1291.54 | 1295.01 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:50:00 | 1284.05 | 1282.14 | 0.00 | ORB-long ORB[1273.75,1283.20] vol=1.8x ATR=6.26 |
| Stop hit — per-position SL triggered | 2024-06-11 10:05:00 | 1277.79 | 1281.97 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:55:00 | 1302.10 | 1298.91 | 0.00 | ORB-long ORB[1285.00,1298.00] vol=12.4x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-06-12 10:10:00 | 1297.51 | 1298.95 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 11:00:00 | 1318.05 | 1305.07 | 0.00 | ORB-long ORB[1286.15,1303.95] vol=1.6x ATR=5.22 |
| Stop hit — per-position SL triggered | 2024-06-13 11:20:00 | 1312.83 | 1306.68 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:55:00 | 1463.70 | 1435.43 | 0.00 | ORB-long ORB[1407.70,1424.00] vol=6.2x ATR=10.67 |
| Stop hit — per-position SL triggered | 2024-06-25 10:00:00 | 1453.03 | 1437.83 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:50:00 | 1398.85 | 1406.80 | 0.00 | ORB-short ORB[1400.25,1417.85] vol=1.8x ATR=6.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:05:00 | 1388.79 | 1403.53 | 0.00 | T1 1.5R @ 1388.79 |
| Target hit | 2024-06-26 13:30:00 | 1391.65 | 1385.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:15:00 | 1396.95 | 1407.15 | 0.00 | ORB-short ORB[1399.30,1417.40] vol=1.8x ATR=5.97 |
| Stop hit — per-position SL triggered | 2024-06-27 10:20:00 | 1402.92 | 1406.95 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:40:00 | 1416.50 | 1420.31 | 0.00 | ORB-short ORB[1417.55,1436.75] vol=4.0x ATR=4.34 |
| Stop hit — per-position SL triggered | 2024-07-02 09:50:00 | 1420.84 | 1420.37 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 1439.00 | 1436.45 | 0.00 | ORB-long ORB[1426.00,1437.85] vol=1.8x ATR=4.48 |
| Stop hit — per-position SL triggered | 2024-07-03 10:10:00 | 1434.52 | 1439.64 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 1536.00 | 1549.31 | 0.00 | ORB-short ORB[1542.05,1564.00] vol=1.7x ATR=5.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:10:00 | 1527.64 | 1546.60 | 0.00 | T1 1.5R @ 1527.64 |
| Target hit | 2024-07-10 15:20:00 | 1513.65 | 1522.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 1507.05 | 1511.28 | 0.00 | ORB-short ORB[1510.00,1519.30] vol=3.2x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 09:50:00 | 1502.01 | 1507.60 | 0.00 | T1 1.5R @ 1502.01 |
| Stop hit — per-position SL triggered | 2024-07-12 09:55:00 | 1507.05 | 1507.78 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 09:30:00 | 1503.05 | 1507.79 | 0.00 | ORB-short ORB[1505.55,1516.00] vol=1.9x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-07-16 09:35:00 | 1507.33 | 1504.94 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 09:45:00 | 1507.00 | 1498.13 | 0.00 | ORB-long ORB[1486.00,1505.80] vol=3.1x ATR=5.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 09:55:00 | 1515.75 | 1501.89 | 0.00 | T1 1.5R @ 1515.75 |
| Stop hit — per-position SL triggered | 2024-07-23 10:05:00 | 1507.00 | 1502.04 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:05:00 | 1493.75 | 1483.31 | 0.00 | ORB-long ORB[1466.80,1481.40] vol=1.6x ATR=7.30 |
| Stop hit — per-position SL triggered | 2024-07-24 10:10:00 | 1486.45 | 1483.79 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 09:40:00 | 1483.90 | 1480.54 | 0.00 | ORB-long ORB[1472.35,1481.25] vol=3.6x ATR=4.51 |
| Stop hit — per-position SL triggered | 2024-07-25 09:45:00 | 1479.39 | 1480.99 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:35:00 | 1503.85 | 1498.15 | 0.00 | ORB-long ORB[1477.00,1490.00] vol=6.3x ATR=5.79 |
| Stop hit — per-position SL triggered | 2024-07-26 09:40:00 | 1498.06 | 1498.21 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 10:15:00 | 1509.00 | 1513.67 | 0.00 | ORB-short ORB[1511.00,1528.00] vol=2.1x ATR=5.11 |
| Stop hit — per-position SL triggered | 2024-07-31 10:25:00 | 1514.11 | 1512.83 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:30:00 | 1481.00 | 1490.53 | 0.00 | ORB-short ORB[1486.10,1505.85] vol=1.9x ATR=6.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:50:00 | 1471.95 | 1486.87 | 0.00 | T1 1.5R @ 1471.95 |
| Target hit | 2024-08-01 15:20:00 | 1448.80 | 1463.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-08-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:35:00 | 1417.25 | 1412.83 | 0.00 | ORB-long ORB[1401.30,1416.95] vol=1.9x ATR=7.18 |
| Stop hit — per-position SL triggered | 2024-08-09 09:45:00 | 1410.07 | 1412.58 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:40:00 | 1516.25 | 1511.14 | 0.00 | ORB-long ORB[1501.00,1514.00] vol=3.8x ATR=6.89 |
| Stop hit — per-position SL triggered | 2024-08-16 09:50:00 | 1509.36 | 1511.17 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:10:00 | 1495.35 | 1502.02 | 0.00 | ORB-short ORB[1500.00,1514.90] vol=4.3x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:55:00 | 1488.21 | 1497.32 | 0.00 | T1 1.5R @ 1488.21 |
| Target hit | 2024-08-22 14:40:00 | 1490.80 | 1486.92 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — SELL (started 2024-09-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 09:35:00 | 1507.25 | 1515.19 | 0.00 | ORB-short ORB[1509.90,1524.00] vol=2.0x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 09:50:00 | 1499.48 | 1512.93 | 0.00 | T1 1.5R @ 1499.48 |
| Target hit | 2024-09-02 15:20:00 | 1490.35 | 1500.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2024-09-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 10:55:00 | 1432.30 | 1423.90 | 0.00 | ORB-long ORB[1407.95,1425.40] vol=1.8x ATR=4.87 |
| Stop hit — per-position SL triggered | 2024-09-04 11:20:00 | 1427.43 | 1425.13 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:50:00 | 1454.30 | 1447.89 | 0.00 | ORB-long ORB[1440.90,1448.90] vol=3.6x ATR=3.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 11:15:00 | 1459.19 | 1449.74 | 0.00 | T1 1.5R @ 1459.19 |
| Stop hit — per-position SL triggered | 2024-09-05 13:10:00 | 1454.30 | 1454.52 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 1447.30 | 1460.71 | 0.00 | ORB-short ORB[1464.50,1472.90] vol=2.8x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:55:00 | 1440.16 | 1456.38 | 0.00 | T1 1.5R @ 1440.16 |
| Target hit | 2024-09-06 11:25:00 | 1446.00 | 1444.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:15:00 | 1473.25 | 1467.06 | 0.00 | ORB-long ORB[1460.15,1468.90] vol=2.0x ATR=3.48 |
| Stop hit — per-position SL triggered | 2024-09-12 10:20:00 | 1469.77 | 1467.62 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:40:00 | 1499.00 | 1495.42 | 0.00 | ORB-long ORB[1482.15,1498.00] vol=2.2x ATR=4.80 |
| Stop hit — per-position SL triggered | 2024-09-17 10:00:00 | 1494.20 | 1496.12 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:45:00 | 1487.50 | 1491.03 | 0.00 | ORB-short ORB[1490.10,1498.85] vol=3.0x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:40:00 | 1479.65 | 1487.81 | 0.00 | T1 1.5R @ 1479.65 |
| Stop hit — per-position SL triggered | 2024-09-18 11:35:00 | 1487.50 | 1486.20 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 1502.05 | 1493.42 | 0.00 | ORB-long ORB[1476.60,1491.50] vol=2.8x ATR=6.44 |
| Stop hit — per-position SL triggered | 2024-09-19 09:40:00 | 1495.61 | 1498.62 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:50:00 | 1487.85 | 1481.29 | 0.00 | ORB-long ORB[1474.00,1486.60] vol=1.7x ATR=4.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 11:10:00 | 1495.11 | 1481.93 | 0.00 | T1 1.5R @ 1495.11 |
| Stop hit — per-position SL triggered | 2024-09-20 12:20:00 | 1487.85 | 1486.83 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 11:00:00 | 1579.80 | 1586.17 | 0.00 | ORB-short ORB[1581.55,1605.00] vol=1.7x ATR=6.84 |
| Stop hit — per-position SL triggered | 2024-09-24 11:15:00 | 1586.64 | 1585.67 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 11:05:00 | 1610.25 | 1601.46 | 0.00 | ORB-long ORB[1591.20,1608.80] vol=1.6x ATR=4.45 |
| Stop hit — per-position SL triggered | 2024-09-27 11:15:00 | 1605.80 | 1601.60 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 09:30:00 | 1590.00 | 1595.28 | 0.00 | ORB-short ORB[1591.00,1610.55] vol=1.8x ATR=6.55 |
| Stop hit — per-position SL triggered | 2024-09-30 09:35:00 | 1596.55 | 1595.99 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:05:00 | 1539.95 | 1520.45 | 0.00 | ORB-long ORB[1507.35,1528.25] vol=2.6x ATR=6.93 |
| Stop hit — per-position SL triggered | 2024-10-08 11:40:00 | 1533.02 | 1524.52 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:20:00 | 1493.40 | 1499.37 | 0.00 | ORB-short ORB[1494.65,1515.50] vol=2.6x ATR=5.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 11:05:00 | 1485.64 | 1494.56 | 0.00 | T1 1.5R @ 1485.64 |
| Stop hit — per-position SL triggered | 2024-10-11 12:20:00 | 1493.40 | 1491.81 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:50:00 | 1492.45 | 1502.44 | 0.00 | ORB-short ORB[1500.00,1510.80] vol=2.4x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-10-15 09:55:00 | 1496.15 | 1501.78 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:00:00 | 1506.95 | 1521.13 | 0.00 | ORB-short ORB[1522.75,1540.55] vol=1.6x ATR=7.86 |
| Stop hit — per-position SL triggered | 2024-10-16 15:20:00 | 1510.00 | 1511.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2024-10-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:10:00 | 1507.05 | 1515.27 | 0.00 | ORB-short ORB[1511.00,1524.00] vol=3.1x ATR=4.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:15:00 | 1499.65 | 1513.45 | 0.00 | T1 1.5R @ 1499.65 |
| Stop hit — per-position SL triggered | 2024-10-17 10:35:00 | 1507.05 | 1510.72 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 09:35:00 | 1466.90 | 1475.19 | 0.00 | ORB-short ORB[1470.60,1490.10] vol=2.8x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:10:00 | 1459.48 | 1471.21 | 0.00 | T1 1.5R @ 1459.48 |
| Target hit | 2024-10-22 15:20:00 | 1440.05 | 1445.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2024-10-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 09:35:00 | 1422.55 | 1430.01 | 0.00 | ORB-short ORB[1424.35,1439.50] vol=5.1x ATR=5.99 |
| Stop hit — per-position SL triggered | 2024-10-24 09:40:00 | 1428.54 | 1429.35 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:40:00 | 1384.30 | 1394.73 | 0.00 | ORB-short ORB[1392.40,1412.50] vol=2.9x ATR=5.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:50:00 | 1376.51 | 1391.34 | 0.00 | T1 1.5R @ 1376.51 |
| Target hit | 2024-10-25 11:15:00 | 1378.70 | 1377.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — SELL (started 2024-10-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:50:00 | 1351.50 | 1355.80 | 0.00 | ORB-short ORB[1351.55,1371.80] vol=4.0x ATR=5.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:25:00 | 1343.44 | 1352.21 | 0.00 | T1 1.5R @ 1343.44 |
| Stop hit — per-position SL triggered | 2024-10-29 11:00:00 | 1351.50 | 1351.32 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 09:30:00 | 1376.25 | 1377.69 | 0.00 | ORB-short ORB[1379.70,1386.25] vol=4.7x ATR=5.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 09:40:00 | 1367.78 | 1376.05 | 0.00 | T1 1.5R @ 1367.78 |
| Stop hit — per-position SL triggered | 2024-10-31 09:50:00 | 1376.25 | 1375.29 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 09:35:00 | 1356.30 | 1366.52 | 0.00 | ORB-short ORB[1360.55,1370.45] vol=1.5x ATR=7.25 |
| Stop hit — per-position SL triggered | 2024-11-06 10:05:00 | 1363.55 | 1360.19 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:35:00 | 1410.80 | 1403.30 | 0.00 | ORB-long ORB[1392.40,1408.80] vol=4.0x ATR=6.70 |
| Stop hit — per-position SL triggered | 2024-11-07 09:55:00 | 1404.10 | 1405.98 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-11-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 09:50:00 | 1372.25 | 1377.63 | 0.00 | ORB-short ORB[1375.00,1393.00] vol=4.7x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 10:30:00 | 1362.56 | 1372.63 | 0.00 | T1 1.5R @ 1362.56 |
| Target hit | 2024-11-08 15:20:00 | 1269.65 | 1302.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2024-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:45:00 | 1282.10 | 1286.89 | 0.00 | ORB-short ORB[1285.05,1303.15] vol=2.3x ATR=5.62 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 1287.72 | 1286.87 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-11-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:40:00 | 1292.85 | 1296.90 | 0.00 | ORB-short ORB[1296.25,1309.70] vol=2.7x ATR=6.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 14:30:00 | 1283.17 | 1294.94 | 0.00 | T1 1.5R @ 1283.17 |
| Stop hit — per-position SL triggered | 2024-11-18 15:05:00 | 1292.85 | 1294.70 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-11-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 09:40:00 | 1317.00 | 1321.85 | 0.00 | ORB-short ORB[1318.90,1337.40] vol=4.1x ATR=4.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 10:05:00 | 1309.91 | 1319.69 | 0.00 | T1 1.5R @ 1309.91 |
| Target hit | 2024-11-29 11:30:00 | 1313.75 | 1313.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 55 — BUY (started 2024-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 1394.00 | 1386.67 | 0.00 | ORB-long ORB[1367.00,1387.15] vol=5.4x ATR=4.54 |
| Stop hit — per-position SL triggered | 2024-12-05 09:35:00 | 1389.46 | 1387.38 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:50:00 | 1334.90 | 1340.89 | 0.00 | ORB-short ORB[1339.70,1356.75] vol=2.7x ATR=5.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:55:00 | 1326.35 | 1338.94 | 0.00 | T1 1.5R @ 1326.35 |
| Target hit | 2024-12-06 15:20:00 | 1301.00 | 1317.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2024-12-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:25:00 | 1260.10 | 1267.08 | 0.00 | ORB-short ORB[1265.00,1279.65] vol=2.0x ATR=5.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 11:25:00 | 1251.98 | 1262.85 | 0.00 | T1 1.5R @ 1251.98 |
| Target hit | 2024-12-10 14:10:00 | 1250.75 | 1250.12 | 0.00 | Trail-exit close>VWAP |

### Cycle 58 — BUY (started 2024-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:45:00 | 1236.00 | 1228.11 | 0.00 | ORB-long ORB[1215.00,1229.00] vol=1.7x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:50:00 | 1245.39 | 1230.39 | 0.00 | T1 1.5R @ 1245.39 |
| Stop hit — per-position SL triggered | 2024-12-12 10:30:00 | 1236.00 | 1238.33 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:40:00 | 1269.00 | 1266.09 | 0.00 | ORB-long ORB[1259.60,1268.95] vol=4.3x ATR=4.75 |
| Stop hit — per-position SL triggered | 2024-12-17 09:45:00 | 1264.25 | 1266.56 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:30:00 | 1285.45 | 1279.78 | 0.00 | ORB-long ORB[1271.00,1285.00] vol=1.6x ATR=6.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:50:00 | 1295.72 | 1283.87 | 0.00 | T1 1.5R @ 1295.72 |
| Stop hit — per-position SL triggered | 2024-12-19 10:00:00 | 1285.45 | 1286.47 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-12-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:45:00 | 1275.25 | 1283.23 | 0.00 | ORB-short ORB[1280.05,1295.00] vol=3.6x ATR=4.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:05:00 | 1267.86 | 1279.25 | 0.00 | T1 1.5R @ 1267.86 |
| Target hit | 2024-12-20 12:30:00 | 1269.15 | 1268.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 62 — SELL (started 2024-12-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 10:30:00 | 1263.00 | 1266.97 | 0.00 | ORB-short ORB[1264.60,1279.55] vol=2.1x ATR=8.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:55:00 | 1250.98 | 1261.91 | 0.00 | T1 1.5R @ 1250.98 |
| Target hit | 2024-12-23 15:05:00 | 1260.50 | 1255.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — BUY (started 2024-12-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:55:00 | 1270.60 | 1255.38 | 0.00 | ORB-long ORB[1243.45,1257.90] vol=2.1x ATR=6.94 |
| Stop hit — per-position SL triggered | 2024-12-24 10:25:00 | 1263.66 | 1260.91 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-12-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:35:00 | 1349.00 | 1336.62 | 0.00 | ORB-long ORB[1328.90,1345.00] vol=2.3x ATR=5.33 |
| Stop hit — per-position SL triggered | 2024-12-27 10:40:00 | 1343.67 | 1337.53 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-12-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 09:35:00 | 1316.00 | 1322.58 | 0.00 | ORB-short ORB[1322.20,1342.00] vol=1.8x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 09:55:00 | 1308.97 | 1316.98 | 0.00 | T1 1.5R @ 1308.97 |
| Stop hit — per-position SL triggered | 2024-12-31 10:00:00 | 1316.00 | 1316.87 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:00:00 | 1342.75 | 1348.67 | 0.00 | ORB-short ORB[1344.60,1360.00] vol=1.9x ATR=7.44 |
| Stop hit — per-position SL triggered | 2025-01-01 15:15:00 | 1350.19 | 1345.06 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:05:00 | 1362.05 | 1356.58 | 0.00 | ORB-long ORB[1347.75,1362.00] vol=2.2x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:40:00 | 1368.74 | 1360.41 | 0.00 | T1 1.5R @ 1368.74 |
| Target hit | 2025-01-02 15:20:00 | 1394.75 | 1383.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 1395.05 | 1398.87 | 0.00 | ORB-short ORB[1395.45,1409.45] vol=2.1x ATR=5.75 |
| Stop hit — per-position SL triggered | 2025-01-03 09:35:00 | 1400.80 | 1398.68 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 1126.15 | 1137.60 | 0.00 | ORB-short ORB[1133.00,1148.45] vol=1.8x ATR=5.20 |
| Stop hit — per-position SL triggered | 2025-01-15 09:40:00 | 1131.35 | 1136.92 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:05:00 | 1101.40 | 1109.53 | 0.00 | ORB-short ORB[1109.50,1116.65] vol=2.1x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:20:00 | 1095.73 | 1106.79 | 0.00 | T1 1.5R @ 1095.73 |
| Stop hit — per-position SL triggered | 2025-01-21 11:35:00 | 1101.40 | 1100.24 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:45:00 | 1191.00 | 1176.97 | 0.00 | ORB-long ORB[1170.00,1178.40] vol=2.1x ATR=6.40 |
| Stop hit — per-position SL triggered | 2025-01-30 09:55:00 | 1184.60 | 1179.12 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-01-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:40:00 | 1197.45 | 1190.03 | 0.00 | ORB-long ORB[1185.20,1194.00] vol=2.1x ATR=5.35 |
| Stop hit — per-position SL triggered | 2025-01-31 10:05:00 | 1192.10 | 1192.96 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-02-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 09:35:00 | 1202.50 | 1208.47 | 0.00 | ORB-short ORB[1206.55,1218.25] vol=5.8x ATR=4.61 |
| Stop hit — per-position SL triggered | 2025-02-01 09:50:00 | 1207.11 | 1206.97 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-02-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:45:00 | 1273.00 | 1264.07 | 0.00 | ORB-long ORB[1255.00,1268.00] vol=2.5x ATR=5.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 10:10:00 | 1281.07 | 1269.24 | 0.00 | T1 1.5R @ 1281.07 |
| Stop hit — per-position SL triggered | 2025-02-04 10:15:00 | 1273.00 | 1269.37 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-02-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:35:00 | 1301.55 | 1292.65 | 0.00 | ORB-long ORB[1280.85,1298.00] vol=3.6x ATR=5.19 |
| Stop hit — per-position SL triggered | 2025-02-05 09:55:00 | 1296.36 | 1296.13 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 1305.15 | 1315.74 | 0.00 | ORB-short ORB[1311.00,1325.00] vol=2.6x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 09:35:00 | 1297.74 | 1313.88 | 0.00 | T1 1.5R @ 1297.74 |
| Stop hit — per-position SL triggered | 2025-02-06 09:40:00 | 1305.15 | 1312.45 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:30:00 | 1257.60 | 1264.70 | 0.00 | ORB-short ORB[1261.05,1274.00] vol=2.3x ATR=4.31 |
| Stop hit — per-position SL triggered | 2025-02-10 09:40:00 | 1261.91 | 1263.19 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-03-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:35:00 | 1144.90 | 1147.75 | 0.00 | ORB-short ORB[1145.25,1156.45] vol=2.5x ATR=5.36 |
| Stop hit — per-position SL triggered | 2025-03-26 10:25:00 | 1150.26 | 1146.85 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2025-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 09:30:00 | 1128.00 | 1132.29 | 0.00 | ORB-short ORB[1129.15,1141.00] vol=3.4x ATR=5.01 |
| Stop hit — per-position SL triggered | 2025-03-27 09:35:00 | 1133.01 | 1132.03 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-04-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:35:00 | 1086.60 | 1079.78 | 0.00 | ORB-long ORB[1071.45,1082.50] vol=3.2x ATR=4.02 |
| Stop hit — per-position SL triggered | 2025-04-02 09:45:00 | 1082.58 | 1083.00 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:05:00 | 1136.60 | 1129.20 | 0.00 | ORB-long ORB[1119.10,1135.40] vol=1.5x ATR=3.26 |
| Stop hit — per-position SL triggered | 2025-04-17 12:50:00 | 1133.34 | 1132.61 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 1130.70 | 1126.02 | 0.00 | ORB-long ORB[1120.10,1130.00] vol=5.3x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:50:00 | 1136.93 | 1127.26 | 0.00 | T1 1.5R @ 1136.93 |
| Target hit | 2025-04-21 13:40:00 | 1136.00 | 1138.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 83 — BUY (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 1155.80 | 1150.52 | 0.00 | ORB-long ORB[1140.50,1152.00] vol=3.4x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:05:00 | 1161.97 | 1152.46 | 0.00 | T1 1.5R @ 1161.97 |
| Target hit | 2025-04-22 14:35:00 | 1160.20 | 1160.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 84 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 1159.60 | 1168.04 | 0.00 | ORB-short ORB[1165.00,1180.70] vol=2.6x ATR=4.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:55:00 | 1152.89 | 1164.10 | 0.00 | T1 1.5R @ 1152.89 |
| Stop hit — per-position SL triggered | 2025-04-25 10:10:00 | 1159.60 | 1163.65 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:35:00 | 1143.00 | 1149.64 | 0.00 | ORB-short ORB[1144.60,1156.40] vol=3.2x ATR=4.88 |
| Stop hit — per-position SL triggered | 2025-04-29 10:10:00 | 1147.88 | 1146.88 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2025-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 09:30:00 | 1129.50 | 1132.98 | 0.00 | ORB-short ORB[1130.50,1141.40] vol=4.2x ATR=5.15 |
| Stop hit — per-position SL triggered | 2025-05-05 09:35:00 | 1134.65 | 1132.70 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:55:00 | 1263.90 | 2024-05-14 11:25:00 | 1258.18 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-14 10:55:00 | 1263.90 | 2024-05-14 15:20:00 | 1254.90 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2024-05-16 09:55:00 | 1263.00 | 2024-05-16 10:05:00 | 1268.29 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-05-23 09:45:00 | 1296.95 | 2024-05-23 09:50:00 | 1302.03 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-28 11:15:00 | 1280.50 | 2024-05-28 11:30:00 | 1284.80 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-30 10:15:00 | 1270.30 | 2024-05-30 11:00:00 | 1264.49 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-05-30 10:15:00 | 1270.30 | 2024-05-30 13:30:00 | 1264.45 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2024-06-10 10:30:00 | 1300.40 | 2024-06-10 10:45:00 | 1291.54 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2024-06-11 09:50:00 | 1284.05 | 2024-06-11 10:05:00 | 1277.79 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-06-12 09:55:00 | 1302.10 | 2024-06-12 10:10:00 | 1297.51 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-13 11:00:00 | 1318.05 | 2024-06-13 11:20:00 | 1312.83 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-25 09:55:00 | 1463.70 | 2024-06-25 10:00:00 | 1453.03 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest1 | 2024-06-26 09:50:00 | 1398.85 | 2024-06-26 10:05:00 | 1388.79 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-06-26 09:50:00 | 1398.85 | 2024-06-26 13:30:00 | 1391.65 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2024-06-27 10:15:00 | 1396.95 | 2024-06-27 10:20:00 | 1402.92 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-07-02 09:40:00 | 1416.50 | 2024-07-02 09:50:00 | 1420.84 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-03 09:30:00 | 1439.00 | 2024-07-03 10:10:00 | 1434.52 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-10 10:05:00 | 1536.00 | 2024-07-10 10:10:00 | 1527.64 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-07-10 10:05:00 | 1536.00 | 2024-07-10 15:20:00 | 1513.65 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2024-07-12 09:30:00 | 1507.05 | 2024-07-12 09:50:00 | 1502.01 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-07-12 09:30:00 | 1507.05 | 2024-07-12 09:55:00 | 1507.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-16 09:30:00 | 1503.05 | 2024-07-16 09:35:00 | 1507.33 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-23 09:45:00 | 1507.00 | 2024-07-23 09:55:00 | 1515.75 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-07-23 09:45:00 | 1507.00 | 2024-07-23 10:05:00 | 1507.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 10:05:00 | 1493.75 | 2024-07-24 10:10:00 | 1486.45 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-07-25 09:40:00 | 1483.90 | 2024-07-25 09:45:00 | 1479.39 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-26 09:35:00 | 1503.85 | 2024-07-26 09:40:00 | 1498.06 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-07-31 10:15:00 | 1509.00 | 2024-07-31 10:25:00 | 1514.11 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-01 10:30:00 | 1481.00 | 2024-08-01 10:50:00 | 1471.95 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-08-01 10:30:00 | 1481.00 | 2024-08-01 15:20:00 | 1448.80 | TARGET_HIT | 0.50 | 2.17% |
| BUY | retest1 | 2024-08-09 09:35:00 | 1417.25 | 2024-08-09 09:45:00 | 1410.07 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-08-16 09:40:00 | 1516.25 | 2024-08-16 09:50:00 | 1509.36 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-08-22 10:10:00 | 1495.35 | 2024-08-22 10:55:00 | 1488.21 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-08-22 10:10:00 | 1495.35 | 2024-08-22 14:40:00 | 1490.80 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2024-09-02 09:35:00 | 1507.25 | 2024-09-02 09:50:00 | 1499.48 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-09-02 09:35:00 | 1507.25 | 2024-09-02 15:20:00 | 1490.35 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2024-09-04 10:55:00 | 1432.30 | 2024-09-04 11:20:00 | 1427.43 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-05 10:50:00 | 1454.30 | 2024-09-05 11:15:00 | 1459.19 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-05 10:50:00 | 1454.30 | 2024-09-05 13:10:00 | 1454.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 09:45:00 | 1447.30 | 2024-09-06 09:55:00 | 1440.16 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-09-06 09:45:00 | 1447.30 | 2024-09-06 11:25:00 | 1446.00 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-09-12 10:15:00 | 1473.25 | 2024-09-12 10:20:00 | 1469.77 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-17 09:40:00 | 1499.00 | 2024-09-17 10:00:00 | 1494.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-18 09:45:00 | 1487.50 | 2024-09-18 10:40:00 | 1479.65 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-09-18 09:45:00 | 1487.50 | 2024-09-18 11:35:00 | 1487.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-19 09:30:00 | 1502.05 | 2024-09-19 09:40:00 | 1495.61 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-09-20 10:50:00 | 1487.85 | 2024-09-20 11:10:00 | 1495.11 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-09-20 10:50:00 | 1487.85 | 2024-09-20 12:20:00 | 1487.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-24 11:00:00 | 1579.80 | 2024-09-24 11:15:00 | 1586.64 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-09-27 11:05:00 | 1610.25 | 2024-09-27 11:15:00 | 1605.80 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-30 09:30:00 | 1590.00 | 2024-09-30 09:35:00 | 1596.55 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-10-08 11:05:00 | 1539.95 | 2024-10-08 11:40:00 | 1533.02 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-10-11 10:20:00 | 1493.40 | 2024-10-11 11:05:00 | 1485.64 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-11 10:20:00 | 1493.40 | 2024-10-11 12:20:00 | 1493.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-15 09:50:00 | 1492.45 | 2024-10-15 09:55:00 | 1496.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-16 10:00:00 | 1506.95 | 2024-10-16 15:20:00 | 1510.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-10-17 10:10:00 | 1507.05 | 2024-10-17 10:15:00 | 1499.65 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-17 10:10:00 | 1507.05 | 2024-10-17 10:35:00 | 1507.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 09:35:00 | 1466.90 | 2024-10-22 10:10:00 | 1459.48 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-22 09:35:00 | 1466.90 | 2024-10-22 15:20:00 | 1440.05 | TARGET_HIT | 0.50 | 1.83% |
| SELL | retest1 | 2024-10-24 09:35:00 | 1422.55 | 2024-10-24 09:40:00 | 1428.54 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-25 09:40:00 | 1384.30 | 2024-10-25 09:50:00 | 1376.51 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-10-25 09:40:00 | 1384.30 | 2024-10-25 11:15:00 | 1378.70 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-29 09:50:00 | 1351.50 | 2024-10-29 10:25:00 | 1343.44 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-10-29 09:50:00 | 1351.50 | 2024-10-29 11:00:00 | 1351.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-31 09:30:00 | 1376.25 | 2024-10-31 09:40:00 | 1367.78 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-10-31 09:30:00 | 1376.25 | 2024-10-31 09:50:00 | 1376.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-06 09:35:00 | 1356.30 | 2024-11-06 10:05:00 | 1363.55 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-11-07 09:35:00 | 1410.80 | 2024-11-07 09:55:00 | 1404.10 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-11-08 09:50:00 | 1372.25 | 2024-11-08 10:30:00 | 1362.56 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-11-08 09:50:00 | 1372.25 | 2024-11-08 15:20:00 | 1269.65 | TARGET_HIT | 0.50 | 7.48% |
| SELL | retest1 | 2024-11-13 09:45:00 | 1282.10 | 2024-11-13 09:50:00 | 1287.72 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-11-18 09:40:00 | 1292.85 | 2024-11-18 14:30:00 | 1283.17 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-11-18 09:40:00 | 1292.85 | 2024-11-18 15:05:00 | 1292.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-29 09:40:00 | 1317.00 | 2024-11-29 10:05:00 | 1309.91 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-11-29 09:40:00 | 1317.00 | 2024-11-29 11:30:00 | 1313.75 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2024-12-05 09:30:00 | 1394.00 | 2024-12-05 09:35:00 | 1389.46 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-06 09:50:00 | 1334.90 | 2024-12-06 09:55:00 | 1326.35 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-12-06 09:50:00 | 1334.90 | 2024-12-06 15:20:00 | 1301.00 | TARGET_HIT | 0.50 | 2.54% |
| SELL | retest1 | 2024-12-10 10:25:00 | 1260.10 | 2024-12-10 11:25:00 | 1251.98 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-12-10 10:25:00 | 1260.10 | 2024-12-10 14:10:00 | 1250.75 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2024-12-12 09:45:00 | 1236.00 | 2024-12-12 09:50:00 | 1245.39 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-12-12 09:45:00 | 1236.00 | 2024-12-12 10:30:00 | 1236.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-17 09:40:00 | 1269.00 | 2024-12-17 09:45:00 | 1264.25 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-19 09:30:00 | 1285.45 | 2024-12-19 09:50:00 | 1295.72 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2024-12-19 09:30:00 | 1285.45 | 2024-12-19 10:00:00 | 1285.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 09:45:00 | 1275.25 | 2024-12-20 10:05:00 | 1267.86 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-12-20 09:45:00 | 1275.25 | 2024-12-20 12:30:00 | 1269.15 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-23 10:30:00 | 1263.00 | 2024-12-23 11:55:00 | 1250.98 | PARTIAL | 0.50 | 0.95% |
| SELL | retest1 | 2024-12-23 10:30:00 | 1263.00 | 2024-12-23 15:05:00 | 1260.50 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-12-24 09:55:00 | 1270.60 | 2024-12-24 10:25:00 | 1263.66 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-12-27 10:35:00 | 1349.00 | 2024-12-27 10:40:00 | 1343.67 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-31 09:35:00 | 1316.00 | 2024-12-31 09:55:00 | 1308.97 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-12-31 09:35:00 | 1316.00 | 2024-12-31 10:00:00 | 1316.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-01 10:00:00 | 1342.75 | 2025-01-01 15:15:00 | 1350.19 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-01-02 10:05:00 | 1362.05 | 2025-01-02 10:40:00 | 1368.74 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-01-02 10:05:00 | 1362.05 | 2025-01-02 15:20:00 | 1394.75 | TARGET_HIT | 0.50 | 2.40% |
| SELL | retest1 | 2025-01-03 09:30:00 | 1395.05 | 2025-01-03 09:35:00 | 1400.80 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-15 09:30:00 | 1126.15 | 2025-01-15 09:40:00 | 1131.35 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-01-21 10:05:00 | 1101.40 | 2025-01-21 10:20:00 | 1095.73 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-21 10:05:00 | 1101.40 | 2025-01-21 11:35:00 | 1101.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-30 09:45:00 | 1191.00 | 2025-01-30 09:55:00 | 1184.60 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-01-31 09:40:00 | 1197.45 | 2025-01-31 10:05:00 | 1192.10 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-02-01 09:35:00 | 1202.50 | 2025-02-01 09:50:00 | 1207.11 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-02-04 09:45:00 | 1273.00 | 2025-02-04 10:10:00 | 1281.07 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-02-04 09:45:00 | 1273.00 | 2025-02-04 10:15:00 | 1273.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-05 09:35:00 | 1301.55 | 2025-02-05 09:55:00 | 1296.36 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-02-06 09:30:00 | 1305.15 | 2025-02-06 09:35:00 | 1297.74 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-02-06 09:30:00 | 1305.15 | 2025-02-06 09:40:00 | 1305.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-10 09:30:00 | 1257.60 | 2025-02-10 09:40:00 | 1261.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-03-26 09:35:00 | 1144.90 | 2025-03-26 10:25:00 | 1150.26 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-03-27 09:30:00 | 1128.00 | 2025-03-27 09:35:00 | 1133.01 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-04-02 09:35:00 | 1086.60 | 2025-04-02 09:45:00 | 1082.58 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-17 11:05:00 | 1136.60 | 2025-04-17 12:50:00 | 1133.34 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-21 09:30:00 | 1130.70 | 2025-04-21 09:50:00 | 1136.93 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-04-21 09:30:00 | 1130.70 | 2025-04-21 13:40:00 | 1136.00 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2025-04-22 09:30:00 | 1155.80 | 2025-04-22 10:05:00 | 1161.97 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-04-22 09:30:00 | 1155.80 | 2025-04-22 14:35:00 | 1160.20 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-04-25 09:35:00 | 1159.60 | 2025-04-25 09:55:00 | 1152.89 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-04-25 09:35:00 | 1159.60 | 2025-04-25 10:10:00 | 1159.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-29 09:35:00 | 1143.00 | 2025-04-29 10:10:00 | 1147.88 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-05-05 09:30:00 | 1129.50 | 2025-05-05 09:35:00 | 1134.65 | STOP_HIT | 1.00 | -0.46% |
