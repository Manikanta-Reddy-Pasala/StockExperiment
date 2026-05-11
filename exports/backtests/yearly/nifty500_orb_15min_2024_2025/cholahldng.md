# Cholamandalam Financial Holdings Ltd. (CHOLAHLDNG)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1785.00
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
| ENTRY1 | 33 |
| ENTRY2 | 0 |
| PARTIAL | 14 |
| TARGET_HIT | 5 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 28
- **Target hits / Stop hits / Partials:** 5 / 28 / 14
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 9.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 10 | 41.7% | 3 | 14 | 7 | 0.27% | 6.5% |
| BUY @ 2nd Alert (retest1) | 24 | 10 | 41.7% | 3 | 14 | 7 | 0.27% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 9 | 39.1% | 2 | 14 | 7 | 0.13% | 3.0% |
| SELL @ 2nd Alert (retest1) | 23 | 9 | 39.1% | 2 | 14 | 7 | 0.13% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 47 | 19 | 40.4% | 5 | 28 | 14 | 0.20% | 9.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 11:05:00 | 1084.35 | 1074.04 | 0.00 | ORB-long ORB[1068.00,1078.00] vol=2.3x ATR=2.92 |
| Stop hit — per-position SL triggered | 2024-05-15 11:25:00 | 1081.43 | 1074.29 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:45:00 | 1102.20 | 1110.50 | 0.00 | ORB-short ORB[1109.55,1118.70] vol=1.5x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 10:00:00 | 1096.13 | 1105.55 | 0.00 | T1 1.5R @ 1096.13 |
| Stop hit — per-position SL triggered | 2024-05-24 10:10:00 | 1102.20 | 1104.64 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-03 09:35:00 | 1099.15 | 1104.25 | 0.00 | ORB-short ORB[1105.90,1116.40] vol=1.9x ATR=6.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-03 10:20:00 | 1089.52 | 1102.10 | 0.00 | T1 1.5R @ 1089.52 |
| Stop hit — per-position SL triggered | 2024-06-03 10:55:00 | 1099.15 | 1097.51 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 10:00:00 | 1228.50 | 1234.23 | 0.00 | ORB-short ORB[1234.05,1247.65] vol=2.7x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-06-11 10:45:00 | 1233.68 | 1232.44 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:50:00 | 1285.75 | 1284.05 | 0.00 | ORB-long ORB[1274.50,1285.50] vol=11.3x ATR=8.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:55:00 | 1299.14 | 1284.64 | 0.00 | T1 1.5R @ 1299.14 |
| Stop hit — per-position SL triggered | 2024-06-14 10:25:00 | 1285.75 | 1285.21 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:25:00 | 1302.30 | 1295.19 | 0.00 | ORB-long ORB[1290.00,1299.35] vol=2.2x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-06-20 10:30:00 | 1298.34 | 1295.69 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:15:00 | 1281.25 | 1287.70 | 0.00 | ORB-short ORB[1287.05,1297.95] vol=2.4x ATR=4.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:40:00 | 1274.59 | 1286.18 | 0.00 | T1 1.5R @ 1274.59 |
| Stop hit — per-position SL triggered | 2024-06-21 11:40:00 | 1281.25 | 1282.24 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-24 10:30:00 | 1280.40 | 1286.62 | 0.00 | ORB-short ORB[1281.15,1295.00] vol=1.8x ATR=6.18 |
| Stop hit — per-position SL triggered | 2024-06-24 10:50:00 | 1286.58 | 1285.75 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:55:00 | 1308.55 | 1297.12 | 0.00 | ORB-long ORB[1288.05,1305.00] vol=2.7x ATR=3.81 |
| Stop hit — per-position SL triggered | 2024-06-26 11:00:00 | 1304.74 | 1297.82 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:45:00 | 1305.00 | 1296.79 | 0.00 | ORB-long ORB[1290.00,1299.45] vol=2.1x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:00:00 | 1311.68 | 1301.54 | 0.00 | T1 1.5R @ 1311.68 |
| Stop hit — per-position SL triggered | 2024-06-27 10:15:00 | 1305.00 | 1302.59 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 11:15:00 | 1489.00 | 1478.10 | 0.00 | ORB-long ORB[1469.00,1483.15] vol=4.5x ATR=5.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 11:20:00 | 1497.72 | 1479.97 | 0.00 | T1 1.5R @ 1497.72 |
| Stop hit — per-position SL triggered | 2024-07-15 11:30:00 | 1489.00 | 1485.34 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:50:00 | 1469.25 | 1493.55 | 0.00 | ORB-short ORB[1495.00,1512.05] vol=1.6x ATR=6.21 |
| Stop hit — per-position SL triggered | 2024-07-19 11:10:00 | 1475.46 | 1484.27 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 11:05:00 | 1467.20 | 1479.49 | 0.00 | ORB-short ORB[1470.00,1489.95] vol=1.7x ATR=5.76 |
| Stop hit — per-position SL triggered | 2024-08-06 11:15:00 | 1472.96 | 1478.69 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-08-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:45:00 | 1500.00 | 1482.46 | 0.00 | ORB-long ORB[1456.90,1474.85] vol=2.3x ATR=4.88 |
| Stop hit — per-position SL triggered | 2024-08-08 10:50:00 | 1495.12 | 1483.68 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:10:00 | 1579.85 | 1589.18 | 0.00 | ORB-short ORB[1587.00,1605.00] vol=3.8x ATR=5.99 |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 1585.84 | 1586.66 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-09-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:05:00 | 1817.00 | 1810.47 | 0.00 | ORB-long ORB[1783.60,1803.10] vol=8.6x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 11:50:00 | 1826.39 | 1814.41 | 0.00 | T1 1.5R @ 1826.39 |
| Target hit | 2024-09-12 15:20:00 | 1859.60 | 1834.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-10-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 09:55:00 | 1955.95 | 1955.11 | 0.00 | ORB-long ORB[1928.55,1952.45] vol=11.8x ATR=9.30 |
| Stop hit — per-position SL triggered | 2024-10-23 10:10:00 | 1946.65 | 1955.11 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-11-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:45:00 | 1750.25 | 1753.87 | 0.00 | ORB-short ORB[1763.95,1780.85] vol=13.9x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 11:40:00 | 1740.94 | 1753.15 | 0.00 | T1 1.5R @ 1740.94 |
| Target hit | 2024-11-06 15:20:00 | 1716.85 | 1737.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-11-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:05:00 | 1696.75 | 1708.96 | 0.00 | ORB-short ORB[1709.05,1734.50] vol=1.9x ATR=7.46 |
| Target hit | 2024-11-07 15:20:00 | 1693.95 | 1699.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 1533.45 | 1550.55 | 0.00 | ORB-short ORB[1560.05,1577.50] vol=1.6x ATR=5.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 11:20:00 | 1524.81 | 1546.96 | 0.00 | T1 1.5R @ 1524.81 |
| Stop hit — per-position SL triggered | 2024-12-05 12:15:00 | 1533.45 | 1539.37 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-12-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:10:00 | 1514.70 | 1519.59 | 0.00 | ORB-short ORB[1515.15,1535.50] vol=2.8x ATR=5.02 |
| Stop hit — per-position SL triggered | 2024-12-12 10:35:00 | 1519.72 | 1518.99 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:00:00 | 1386.70 | 1390.20 | 0.00 | ORB-short ORB[1391.25,1410.30] vol=1.8x ATR=5.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:55:00 | 1378.70 | 1388.59 | 0.00 | T1 1.5R @ 1378.70 |
| Stop hit — per-position SL triggered | 2024-12-26 14:00:00 | 1386.70 | 1386.55 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-12-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:30:00 | 1419.15 | 1408.92 | 0.00 | ORB-long ORB[1395.20,1413.00] vol=3.9x ATR=5.47 |
| Stop hit — per-position SL triggered | 2024-12-27 10:35:00 | 1413.68 | 1409.41 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-12-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:55:00 | 1400.95 | 1406.84 | 0.00 | ORB-short ORB[1404.10,1421.00] vol=2.0x ATR=4.78 |
| Stop hit — per-position SL triggered | 2024-12-30 10:05:00 | 1405.73 | 1405.93 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 11:15:00 | 1394.00 | 1397.66 | 0.00 | ORB-short ORB[1395.00,1408.95] vol=2.1x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:30:00 | 1388.06 | 1395.46 | 0.00 | T1 1.5R @ 1388.06 |
| Stop hit — per-position SL triggered | 2025-01-01 12:40:00 | 1394.00 | 1393.62 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-01-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:55:00 | 1551.10 | 1547.27 | 0.00 | ORB-long ORB[1526.10,1547.40] vol=1.9x ATR=6.71 |
| Stop hit — per-position SL triggered | 2025-01-07 11:20:00 | 1544.39 | 1547.51 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:30:00 | 1558.95 | 1547.24 | 0.00 | ORB-long ORB[1538.05,1555.00] vol=1.8x ATR=9.04 |
| Stop hit — per-position SL triggered | 2025-01-09 09:35:00 | 1549.91 | 1549.88 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-01-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 11:05:00 | 1450.80 | 1436.78 | 0.00 | ORB-long ORB[1432.75,1444.45] vol=2.4x ATR=5.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 11:30:00 | 1459.44 | 1440.56 | 0.00 | T1 1.5R @ 1459.44 |
| Stop hit — per-position SL triggered | 2025-01-20 13:35:00 | 1450.80 | 1444.34 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-01-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:50:00 | 1480.50 | 1463.47 | 0.00 | ORB-long ORB[1429.70,1445.00] vol=1.5x ATR=9.45 |
| Stop hit — per-position SL triggered | 2025-01-21 09:55:00 | 1471.05 | 1465.84 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-01-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:55:00 | 1455.95 | 1449.88 | 0.00 | ORB-long ORB[1418.70,1436.85] vol=16.1x ATR=5.33 |
| Stop hit — per-position SL triggered | 2025-01-23 11:00:00 | 1450.62 | 1449.94 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 1505.50 | 1494.33 | 0.00 | ORB-long ORB[1479.50,1500.00] vol=1.7x ATR=8.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:05:00 | 1517.89 | 1506.15 | 0.00 | T1 1.5R @ 1517.89 |
| Target hit | 2025-01-30 12:00:00 | 1535.75 | 1537.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — SELL (started 2025-02-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:50:00 | 1476.20 | 1478.05 | 0.00 | ORB-short ORB[1483.25,1499.75] vol=11.2x ATR=4.99 |
| Stop hit — per-position SL triggered | 2025-02-06 10:10:00 | 1481.19 | 1477.84 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-03-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 09:55:00 | 1749.10 | 1735.33 | 0.00 | ORB-long ORB[1721.30,1745.80] vol=2.7x ATR=9.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 10:15:00 | 1764.07 | 1746.10 | 0.00 | T1 1.5R @ 1764.07 |
| Target hit | 2025-03-28 13:20:00 | 1770.80 | 1771.82 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 11:05:00 | 1084.35 | 2024-05-15 11:25:00 | 1081.43 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-05-24 09:45:00 | 1102.20 | 2024-05-24 10:00:00 | 1096.13 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-05-24 09:45:00 | 1102.20 | 2024-05-24 10:10:00 | 1102.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-03 09:35:00 | 1099.15 | 2024-06-03 10:20:00 | 1089.52 | PARTIAL | 0.50 | 0.88% |
| SELL | retest1 | 2024-06-03 09:35:00 | 1099.15 | 2024-06-03 10:55:00 | 1099.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-11 10:00:00 | 1228.50 | 2024-06-11 10:45:00 | 1233.68 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-06-14 09:50:00 | 1285.75 | 2024-06-14 09:55:00 | 1299.14 | PARTIAL | 0.50 | 1.04% |
| BUY | retest1 | 2024-06-14 09:50:00 | 1285.75 | 2024-06-14 10:25:00 | 1285.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 10:25:00 | 1302.30 | 2024-06-20 10:30:00 | 1298.34 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-21 10:15:00 | 1281.25 | 2024-06-21 10:40:00 | 1274.59 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-06-21 10:15:00 | 1281.25 | 2024-06-21 11:40:00 | 1281.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-24 10:30:00 | 1280.40 | 2024-06-24 10:50:00 | 1286.58 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-06-26 10:55:00 | 1308.55 | 2024-06-26 11:00:00 | 1304.74 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-27 09:45:00 | 1305.00 | 2024-06-27 10:00:00 | 1311.68 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-06-27 09:45:00 | 1305.00 | 2024-06-27 10:15:00 | 1305.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-15 11:15:00 | 1489.00 | 2024-07-15 11:20:00 | 1497.72 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-07-15 11:15:00 | 1489.00 | 2024-07-15 11:30:00 | 1489.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-19 10:50:00 | 1469.25 | 2024-07-19 11:10:00 | 1475.46 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-08-06 11:05:00 | 1467.20 | 2024-08-06 11:15:00 | 1472.96 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-08-08 10:45:00 | 1500.00 | 2024-08-08 10:50:00 | 1495.12 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-20 10:10:00 | 1579.85 | 2024-08-20 11:15:00 | 1585.84 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-12 10:05:00 | 1817.00 | 2024-09-12 11:50:00 | 1826.39 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-12 10:05:00 | 1817.00 | 2024-09-12 15:20:00 | 1859.60 | TARGET_HIT | 0.50 | 2.34% |
| BUY | retest1 | 2024-10-23 09:55:00 | 1955.95 | 2024-10-23 10:10:00 | 1946.65 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-11-06 10:45:00 | 1750.25 | 2024-11-06 11:40:00 | 1740.94 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-11-06 10:45:00 | 1750.25 | 2024-11-06 15:20:00 | 1716.85 | TARGET_HIT | 0.50 | 1.91% |
| SELL | retest1 | 2024-11-07 10:05:00 | 1696.75 | 2024-11-07 15:20:00 | 1693.95 | TARGET_HIT | 1.00 | 0.17% |
| SELL | retest1 | 2024-12-05 10:55:00 | 1533.45 | 2024-12-05 11:20:00 | 1524.81 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-12-05 10:55:00 | 1533.45 | 2024-12-05 12:15:00 | 1533.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 10:10:00 | 1514.70 | 2024-12-12 10:35:00 | 1519.72 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-26 11:00:00 | 1386.70 | 2024-12-26 11:55:00 | 1378.70 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-12-26 11:00:00 | 1386.70 | 2024-12-26 14:00:00 | 1386.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 10:30:00 | 1419.15 | 2024-12-27 10:35:00 | 1413.68 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-30 09:55:00 | 1400.95 | 2024-12-30 10:05:00 | 1405.73 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-01 11:15:00 | 1394.00 | 2025-01-01 11:30:00 | 1388.06 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-01 11:15:00 | 1394.00 | 2025-01-01 12:40:00 | 1394.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-07 10:55:00 | 1551.10 | 2025-01-07 11:20:00 | 1544.39 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-01-09 09:30:00 | 1558.95 | 2025-01-09 09:35:00 | 1549.91 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2025-01-20 11:05:00 | 1450.80 | 2025-01-20 11:30:00 | 1459.44 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-01-20 11:05:00 | 1450.80 | 2025-01-20 13:35:00 | 1450.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-21 09:50:00 | 1480.50 | 2025-01-21 09:55:00 | 1471.05 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2025-01-23 10:55:00 | 1455.95 | 2025-01-23 11:00:00 | 1450.62 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-30 09:30:00 | 1505.50 | 2025-01-30 10:05:00 | 1517.89 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2025-01-30 09:30:00 | 1505.50 | 2025-01-30 12:00:00 | 1535.75 | TARGET_HIT | 0.50 | 2.01% |
| SELL | retest1 | 2025-02-06 09:50:00 | 1476.20 | 2025-02-06 10:10:00 | 1481.19 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-28 09:55:00 | 1749.10 | 2025-03-28 10:15:00 | 1764.07 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2025-03-28 09:55:00 | 1749.10 | 2025-03-28 13:20:00 | 1770.80 | TARGET_HIT | 0.50 | 1.24% |
