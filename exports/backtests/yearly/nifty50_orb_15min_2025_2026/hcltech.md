# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15238 bars)
- **Last close:** 1198.00
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
| PARTIAL | 28 |
| TARGET_HIT | 15 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 101 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 58
- **Target hits / Stop hits / Partials:** 15 / 58 / 28
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 10.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 28 | 53.8% | 10 | 24 | 18 | 0.21% | 11.0% |
| BUY @ 2nd Alert (retest1) | 52 | 28 | 53.8% | 10 | 24 | 18 | 0.21% | 11.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 15 | 30.6% | 5 | 34 | 10 | -0.00% | -0.1% |
| SELL @ 2nd Alert (retest1) | 49 | 15 | 30.6% | 5 | 34 | 10 | -0.00% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 101 | 43 | 42.6% | 15 | 58 | 28 | 0.11% | 10.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 10:45:00 | 1622.90 | 1617.55 | 0.00 | ORB-long ORB[1594.00,1611.90] vol=2.2x ATR=7.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 11:30:00 | 1634.50 | 1620.53 | 0.00 | T1 1.5R @ 1634.50 |
| Target hit | 2025-05-12 15:20:00 | 1669.50 | 1650.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2025-05-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:10:00 | 1647.00 | 1641.97 | 0.00 | ORB-long ORB[1628.40,1645.30] vol=2.0x ATR=3.57 |
| Stop hit — per-position SL triggered | 2025-05-21 12:00:00 | 1643.43 | 1642.92 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 11:10:00 | 1650.00 | 1654.30 | 0.00 | ORB-short ORB[1652.80,1666.80] vol=1.6x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-05-28 12:30:00 | 1653.50 | 1653.02 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:00:00 | 1656.00 | 1664.63 | 0.00 | ORB-short ORB[1664.00,1675.50] vol=3.0x ATR=3.70 |
| Stop hit — per-position SL triggered | 2025-05-29 11:05:00 | 1659.70 | 1663.99 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:45:00 | 1655.80 | 1652.27 | 0.00 | ORB-long ORB[1639.00,1655.00] vol=2.0x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-06-09 10:50:00 | 1652.87 | 1652.37 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-07-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 09:50:00 | 1659.40 | 1668.04 | 0.00 | ORB-short ORB[1662.00,1679.90] vol=2.7x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 10:20:00 | 1652.26 | 1664.15 | 0.00 | T1 1.5R @ 1652.26 |
| Stop hit — per-position SL triggered | 2025-07-10 11:00:00 | 1659.40 | 1661.32 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-07-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 09:40:00 | 1531.20 | 1535.55 | 0.00 | ORB-short ORB[1532.50,1550.70] vol=2.0x ATR=3.24 |
| Stop hit — per-position SL triggered | 2025-07-21 09:55:00 | 1534.44 | 1533.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-07-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:05:00 | 1477.40 | 1472.42 | 0.00 | ORB-long ORB[1467.00,1475.00] vol=2.4x ATR=3.42 |
| Stop hit — per-position SL triggered | 2025-07-29 10:10:00 | 1473.98 | 1472.54 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:45:00 | 1470.60 | 1475.17 | 0.00 | ORB-short ORB[1472.20,1484.30] vol=1.8x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-07-30 11:10:00 | 1473.39 | 1474.03 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 10:50:00 | 1469.60 | 1464.53 | 0.00 | ORB-long ORB[1461.00,1469.10] vol=1.8x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:20:00 | 1473.35 | 1465.88 | 0.00 | T1 1.5R @ 1473.35 |
| Target hit | 2025-07-31 15:00:00 | 1470.10 | 1471.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2025-08-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 10:55:00 | 1455.10 | 1458.19 | 0.00 | ORB-short ORB[1455.20,1466.90] vol=1.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 1457.97 | 1457.30 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-08-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:35:00 | 1465.00 | 1473.46 | 0.00 | ORB-short ORB[1474.20,1483.70] vol=1.7x ATR=3.52 |
| Stop hit — per-position SL triggered | 2025-08-06 09:50:00 | 1468.52 | 1471.72 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-08-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:45:00 | 1465.60 | 1467.00 | 0.00 | ORB-short ORB[1468.30,1477.00] vol=4.0x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-08-08 10:50:00 | 1468.47 | 1467.32 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-08-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 10:50:00 | 1482.90 | 1475.26 | 0.00 | ORB-long ORB[1465.60,1475.70] vol=3.8x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 12:10:00 | 1486.97 | 1479.56 | 0.00 | T1 1.5R @ 1486.97 |
| Target hit | 2025-08-11 15:20:00 | 1486.40 | 1484.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2025-08-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 09:30:00 | 1502.10 | 1497.22 | 0.00 | ORB-long ORB[1489.00,1498.80] vol=2.4x ATR=5.38 |
| Stop hit — per-position SL triggered | 2025-08-12 15:20:00 | 1498.90 | 1502.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-08-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:45:00 | 1507.80 | 1502.01 | 0.00 | ORB-long ORB[1495.70,1507.00] vol=1.9x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 11:05:00 | 1511.91 | 1504.09 | 0.00 | T1 1.5R @ 1511.91 |
| Stop hit — per-position SL triggered | 2025-08-14 11:40:00 | 1507.80 | 1507.14 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:15:00 | 1473.40 | 1474.29 | 0.00 | ORB-short ORB[1474.00,1482.00] vol=1.8x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-08-19 11:20:00 | 1475.77 | 1474.45 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:55:00 | 1470.20 | 1473.99 | 0.00 | ORB-short ORB[1480.00,1491.50] vol=2.0x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-08-22 11:10:00 | 1472.54 | 1473.63 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:40:00 | 1499.10 | 1490.08 | 0.00 | ORB-long ORB[1476.20,1490.90] vol=1.6x ATR=3.75 |
| Stop hit — per-position SL triggered | 2025-08-25 09:55:00 | 1495.35 | 1491.57 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:50:00 | 1492.90 | 1499.65 | 0.00 | ORB-short ORB[1494.80,1503.00] vol=1.6x ATR=2.78 |
| Stop hit — per-position SL triggered | 2025-08-26 10:55:00 | 1495.68 | 1499.15 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:40:00 | 1458.60 | 1451.53 | 0.00 | ORB-long ORB[1442.00,1456.00] vol=4.4x ATR=3.68 |
| Stop hit — per-position SL triggered | 2025-08-29 14:00:00 | 1454.92 | 1456.47 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-09-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 11:00:00 | 1454.10 | 1461.36 | 0.00 | ORB-short ORB[1461.30,1475.00] vol=2.6x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-09-03 11:05:00 | 1456.32 | 1461.14 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 10:50:00 | 1453.60 | 1459.09 | 0.00 | ORB-short ORB[1457.80,1465.90] vol=6.3x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-09-04 10:55:00 | 1455.55 | 1458.68 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:10:00 | 1419.00 | 1435.72 | 0.00 | ORB-short ORB[1444.70,1452.30] vol=5.0x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:15:00 | 1412.60 | 1427.11 | 0.00 | T1 1.5R @ 1412.60 |
| Stop hit — per-position SL triggered | 2025-09-05 10:20:00 | 1419.00 | 1425.55 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:55:00 | 1465.50 | 1456.46 | 0.00 | ORB-long ORB[1432.10,1454.00] vol=2.0x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 12:20:00 | 1470.03 | 1460.15 | 0.00 | T1 1.5R @ 1470.03 |
| Stop hit — per-position SL triggered | 2025-09-10 13:40:00 | 1465.50 | 1462.20 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:55:00 | 1463.50 | 1468.89 | 0.00 | ORB-short ORB[1468.90,1480.00] vol=3.6x ATR=2.40 |
| Stop hit — per-position SL triggered | 2025-09-12 11:05:00 | 1465.90 | 1468.80 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 11:00:00 | 1454.60 | 1457.44 | 0.00 | ORB-short ORB[1458.20,1467.20] vol=3.2x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-09-15 11:30:00 | 1456.26 | 1456.33 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:45:00 | 1470.10 | 1467.71 | 0.00 | ORB-long ORB[1461.10,1466.60] vol=1.6x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-09-16 10:50:00 | 1468.17 | 1467.87 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:15:00 | 1477.00 | 1482.57 | 0.00 | ORB-short ORB[1482.10,1493.80] vol=2.7x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 12:00:00 | 1474.49 | 1481.79 | 0.00 | T1 1.5R @ 1474.49 |
| Target hit | 2025-09-19 15:20:00 | 1469.10 | 1470.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2025-09-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 11:05:00 | 1436.40 | 1435.23 | 0.00 | ORB-long ORB[1415.00,1434.40] vol=2.2x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 11:40:00 | 1441.47 | 1435.53 | 0.00 | T1 1.5R @ 1441.47 |
| Target hit | 2025-09-22 14:40:00 | 1437.50 | 1437.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2025-09-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:05:00 | 1422.20 | 1431.26 | 0.00 | ORB-short ORB[1433.10,1444.20] vol=1.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-09-23 12:05:00 | 1424.31 | 1428.43 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 11:15:00 | 1404.50 | 1408.20 | 0.00 | ORB-short ORB[1407.10,1426.50] vol=2.5x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:35:00 | 1401.44 | 1407.42 | 0.00 | T1 1.5R @ 1401.44 |
| Target hit | 2025-09-26 15:20:00 | 1395.10 | 1400.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2025-09-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 10:40:00 | 1398.00 | 1394.87 | 0.00 | ORB-long ORB[1390.10,1397.00] vol=1.5x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:10:00 | 1402.04 | 1396.33 | 0.00 | T1 1.5R @ 1402.04 |
| Target hit | 2025-09-30 13:25:00 | 1398.70 | 1398.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2025-10-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 11:10:00 | 1414.60 | 1406.92 | 0.00 | ORB-long ORB[1390.80,1407.60] vol=1.6x ATR=3.51 |
| Stop hit — per-position SL triggered | 2025-10-06 11:25:00 | 1411.09 | 1407.28 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 10:50:00 | 1430.10 | 1425.56 | 0.00 | ORB-long ORB[1417.70,1426.90] vol=4.0x ATR=3.03 |
| Stop hit — per-position SL triggered | 2025-10-07 11:40:00 | 1427.07 | 1426.46 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 10:45:00 | 1493.60 | 1497.90 | 0.00 | ORB-short ORB[1496.50,1509.40] vol=1.6x ATR=3.88 |
| Stop hit — per-position SL triggered | 2025-10-15 11:50:00 | 1497.48 | 1497.38 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:45:00 | 1512.00 | 1506.73 | 0.00 | ORB-long ORB[1498.00,1509.40] vol=7.3x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 12:30:00 | 1516.52 | 1509.24 | 0.00 | T1 1.5R @ 1516.52 |
| Target hit | 2025-10-16 15:20:00 | 1514.00 | 1512.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2025-10-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:45:00 | 1526.30 | 1521.30 | 0.00 | ORB-long ORB[1505.50,1526.00] vol=2.0x ATR=5.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:20:00 | 1534.25 | 1524.90 | 0.00 | T1 1.5R @ 1534.25 |
| Target hit | 2025-10-23 14:25:00 | 1532.50 | 1533.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2025-10-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:40:00 | 1545.00 | 1537.67 | 0.00 | ORB-long ORB[1526.80,1538.50] vol=2.2x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 11:55:00 | 1549.46 | 1541.40 | 0.00 | T1 1.5R @ 1549.46 |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 1545.00 | 1541.73 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 11:15:00 | 1518.50 | 1532.26 | 0.00 | ORB-short ORB[1533.50,1541.90] vol=1.6x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 12:05:00 | 1514.36 | 1529.02 | 0.00 | T1 1.5R @ 1514.36 |
| Stop hit — per-position SL triggered | 2025-10-28 12:20:00 | 1518.50 | 1528.32 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:45:00 | 1543.50 | 1536.32 | 0.00 | ORB-long ORB[1522.00,1536.00] vol=1.6x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 11:05:00 | 1548.80 | 1540.22 | 0.00 | T1 1.5R @ 1548.80 |
| Target hit | 2025-10-29 15:20:00 | 1557.90 | 1549.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2025-10-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:45:00 | 1540.70 | 1546.44 | 0.00 | ORB-short ORB[1547.20,1559.90] vol=2.0x ATR=2.57 |
| Stop hit — per-position SL triggered | 2025-10-30 11:00:00 | 1543.27 | 1545.67 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 11:15:00 | 1548.60 | 1541.31 | 0.00 | ORB-long ORB[1530.20,1541.40] vol=6.2x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-11-03 11:25:00 | 1545.55 | 1541.72 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-11-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:45:00 | 1538.20 | 1531.40 | 0.00 | ORB-long ORB[1511.40,1524.50] vol=1.6x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:20:00 | 1543.42 | 1532.99 | 0.00 | T1 1.5R @ 1543.42 |
| Stop hit — per-position SL triggered | 2025-11-10 11:55:00 | 1538.20 | 1535.32 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:55:00 | 1591.30 | 1583.88 | 0.00 | ORB-long ORB[1575.00,1588.00] vol=2.8x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 12:25:00 | 1596.40 | 1586.17 | 0.00 | T1 1.5R @ 1596.40 |
| Stop hit — per-position SL triggered | 2025-11-12 15:20:00 | 1591.10 | 1591.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2025-11-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 11:00:00 | 1599.80 | 1589.65 | 0.00 | ORB-long ORB[1581.40,1599.50] vol=1.9x ATR=2.99 |
| Stop hit — per-position SL triggered | 2025-11-13 11:40:00 | 1596.81 | 1593.30 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 09:50:00 | 1612.10 | 1604.77 | 0.00 | ORB-long ORB[1588.60,1606.40] vol=1.8x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:00:00 | 1617.12 | 1606.96 | 0.00 | T1 1.5R @ 1617.12 |
| Target hit | 2025-11-19 15:20:00 | 1662.50 | 1649.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-11-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:50:00 | 1624.50 | 1623.63 | 0.00 | ORB-long ORB[1614.00,1622.40] vol=1.7x ATR=3.23 |
| Stop hit — per-position SL triggered | 2025-11-27 12:35:00 | 1621.27 | 1624.09 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 11:00:00 | 1630.80 | 1636.96 | 0.00 | ORB-short ORB[1640.90,1652.90] vol=2.5x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-12-02 11:10:00 | 1633.66 | 1635.90 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:05:00 | 1662.20 | 1656.65 | 0.00 | ORB-long ORB[1645.10,1659.90] vol=2.4x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 10:15:00 | 1668.51 | 1658.16 | 0.00 | T1 1.5R @ 1668.51 |
| Stop hit — per-position SL triggered | 2025-12-04 11:05:00 | 1662.20 | 1659.50 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:25:00 | 1665.80 | 1664.44 | 0.00 | ORB-long ORB[1651.00,1663.00] vol=2.8x ATR=3.40 |
| Stop hit — per-position SL triggered | 2025-12-05 10:30:00 | 1662.40 | 1664.56 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 10:55:00 | 1660.60 | 1664.17 | 0.00 | ORB-short ORB[1665.90,1677.40] vol=2.4x ATR=3.79 |
| Stop hit — per-position SL triggered | 2025-12-11 11:45:00 | 1664.39 | 1662.64 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:40:00 | 1664.10 | 1668.16 | 0.00 | ORB-short ORB[1669.90,1677.90] vol=2.2x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-12-12 11:20:00 | 1667.06 | 1667.50 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:55:00 | 1683.20 | 1671.55 | 0.00 | ORB-long ORB[1660.00,1673.20] vol=1.5x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-12-15 11:00:00 | 1680.39 | 1672.38 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:55:00 | 1655.20 | 1665.03 | 0.00 | ORB-short ORB[1661.30,1682.00] vol=2.1x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:55:00 | 1650.37 | 1659.41 | 0.00 | T1 1.5R @ 1650.37 |
| Target hit | 2025-12-16 15:20:00 | 1652.30 | 1653.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-12-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 11:10:00 | 1656.80 | 1662.57 | 0.00 | ORB-short ORB[1657.00,1673.90] vol=1.6x ATR=2.90 |
| Stop hit — per-position SL triggered | 2025-12-18 11:25:00 | 1659.70 | 1662.04 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 09:55:00 | 1651.20 | 1661.98 | 0.00 | ORB-short ORB[1662.00,1681.00] vol=1.7x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 10:10:00 | 1645.37 | 1658.84 | 0.00 | T1 1.5R @ 1645.37 |
| Stop hit — per-position SL triggered | 2025-12-19 10:25:00 | 1651.20 | 1656.67 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:55:00 | 1665.60 | 1658.83 | 0.00 | ORB-long ORB[1644.00,1657.90] vol=1.7x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 13:00:00 | 1670.06 | 1661.71 | 0.00 | T1 1.5R @ 1670.06 |
| Stop hit — per-position SL triggered | 2025-12-22 13:35:00 | 1665.60 | 1663.03 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:00:00 | 1662.00 | 1666.95 | 0.00 | ORB-short ORB[1668.00,1676.40] vol=1.8x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:20:00 | 1658.79 | 1665.74 | 0.00 | T1 1.5R @ 1658.79 |
| Stop hit — per-position SL triggered | 2025-12-26 13:45:00 | 1662.00 | 1661.79 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:55:00 | 1646.30 | 1651.49 | 0.00 | ORB-short ORB[1651.20,1660.80] vol=6.4x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-12-29 11:10:00 | 1649.03 | 1650.56 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-01-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:45:00 | 1633.80 | 1627.45 | 0.00 | ORB-long ORB[1620.50,1631.50] vol=4.2x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:20:00 | 1638.12 | 1629.00 | 0.00 | T1 1.5R @ 1638.12 |
| Target hit | 2026-01-01 15:05:00 | 1636.60 | 1637.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — SELL (started 2026-02-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 09:35:00 | 1598.00 | 1601.15 | 0.00 | ORB-short ORB[1600.00,1619.90] vol=4.3x ATR=4.81 |
| Stop hit — per-position SL triggered | 2026-02-09 10:00:00 | 1602.81 | 1601.11 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 1598.00 | 1594.31 | 0.00 | ORB-long ORB[1582.70,1595.00] vol=2.1x ATR=3.74 |
| Stop hit — per-position SL triggered | 2026-02-10 09:55:00 | 1594.26 | 1594.48 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 1563.40 | 1568.56 | 0.00 | ORB-short ORB[1564.40,1578.10] vol=1.5x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:45:00 | 1558.18 | 1564.77 | 0.00 | T1 1.5R @ 1558.18 |
| Target hit | 2026-02-11 14:10:00 | 1558.30 | 1558.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — SELL (started 2026-02-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:05:00 | 1449.30 | 1465.17 | 0.00 | ORB-short ORB[1461.90,1483.20] vol=2.2x ATR=6.18 |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 1455.48 | 1464.11 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 1417.50 | 1430.35 | 0.00 | ORB-short ORB[1422.00,1441.60] vol=1.6x ATR=4.21 |
| Stop hit — per-position SL triggered | 2026-02-23 10:55:00 | 1421.71 | 1429.38 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-03-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 10:55:00 | 1361.90 | 1346.39 | 0.00 | ORB-long ORB[1328.00,1342.00] vol=2.0x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 11:15:00 | 1368.70 | 1351.70 | 0.00 | T1 1.5R @ 1368.70 |
| Stop hit — per-position SL triggered | 2026-03-09 12:05:00 | 1361.90 | 1355.40 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:15:00 | 1348.10 | 1352.35 | 0.00 | ORB-short ORB[1348.50,1364.20] vol=1.9x ATR=3.11 |
| Stop hit — per-position SL triggered | 2026-03-10 11:30:00 | 1351.21 | 1352.23 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 1354.50 | 1357.92 | 0.00 | ORB-short ORB[1361.20,1376.00] vol=5.3x ATR=3.60 |
| Stop hit — per-position SL triggered | 2026-03-11 11:00:00 | 1358.10 | 1357.78 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 11:15:00 | 1433.30 | 1434.40 | 0.00 | ORB-short ORB[1436.80,1451.90] vol=2.9x ATR=3.00 |
| Stop hit — per-position SL triggered | 2026-04-20 11:20:00 | 1436.30 | 1434.48 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 1209.40 | 1219.50 | 0.00 | ORB-short ORB[1220.40,1232.70] vol=1.6x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 12:25:00 | 1205.33 | 1214.07 | 0.00 | T1 1.5R @ 1205.33 |
| Target hit | 2026-04-28 15:20:00 | 1195.90 | 1202.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 1206.20 | 1201.92 | 0.00 | ORB-long ORB[1193.00,1203.40] vol=2.1x ATR=3.00 |
| Stop hit — per-position SL triggered | 2026-04-29 11:40:00 | 1203.20 | 1203.09 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 1197.40 | 1201.72 | 0.00 | ORB-short ORB[1203.60,1212.00] vol=3.3x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1199.36 | 1201.42 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 10:45:00 | 1622.90 | 2025-05-12 11:30:00 | 1634.50 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-05-12 10:45:00 | 1622.90 | 2025-05-12 15:20:00 | 1669.50 | TARGET_HIT | 0.50 | 2.87% |
| BUY | retest1 | 2025-05-21 11:10:00 | 1647.00 | 2025-05-21 12:00:00 | 1643.43 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-28 11:10:00 | 1650.00 | 2025-05-28 12:30:00 | 1653.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-05-29 11:00:00 | 1656.00 | 2025-05-29 11:05:00 | 1659.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-09 10:45:00 | 1655.80 | 2025-06-09 10:50:00 | 1652.87 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-10 09:50:00 | 1659.40 | 2025-07-10 10:20:00 | 1652.26 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-07-10 09:50:00 | 1659.40 | 2025-07-10 11:00:00 | 1659.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-21 09:40:00 | 1531.20 | 2025-07-21 09:55:00 | 1534.44 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-29 10:05:00 | 1477.40 | 2025-07-29 10:10:00 | 1473.98 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-30 10:45:00 | 1470.60 | 2025-07-30 11:10:00 | 1473.39 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-31 10:50:00 | 1469.60 | 2025-07-31 11:20:00 | 1473.35 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-07-31 10:50:00 | 1469.60 | 2025-07-31 15:00:00 | 1470.10 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2025-08-01 10:55:00 | 1455.10 | 2025-08-01 11:15:00 | 1457.97 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-06 09:35:00 | 1465.00 | 2025-08-06 09:50:00 | 1468.52 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-08 10:45:00 | 1465.60 | 2025-08-08 10:50:00 | 1468.47 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-08-11 10:50:00 | 1482.90 | 2025-08-11 12:10:00 | 1486.97 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-08-11 10:50:00 | 1482.90 | 2025-08-11 15:20:00 | 1486.40 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-08-12 09:30:00 | 1502.10 | 2025-08-12 15:20:00 | 1498.90 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-08-14 10:45:00 | 1507.80 | 2025-08-14 11:05:00 | 1511.91 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-08-14 10:45:00 | 1507.80 | 2025-08-14 11:40:00 | 1507.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-19 11:15:00 | 1473.40 | 2025-08-19 11:20:00 | 1475.77 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-08-22 10:55:00 | 1470.20 | 2025-08-22 11:10:00 | 1472.54 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-08-25 09:40:00 | 1499.10 | 2025-08-25 09:55:00 | 1495.35 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-26 10:50:00 | 1492.90 | 2025-08-26 10:55:00 | 1495.68 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-08-29 10:40:00 | 1458.60 | 2025-08-29 14:00:00 | 1454.92 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-03 11:00:00 | 1454.10 | 2025-09-03 11:05:00 | 1456.32 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-09-04 10:50:00 | 1453.60 | 2025-09-04 10:55:00 | 1455.55 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-09-05 10:10:00 | 1419.00 | 2025-09-05 10:15:00 | 1412.60 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-09-05 10:10:00 | 1419.00 | 2025-09-05 10:20:00 | 1419.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-10 10:55:00 | 1465.50 | 2025-09-10 12:20:00 | 1470.03 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-09-10 10:55:00 | 1465.50 | 2025-09-10 13:40:00 | 1465.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-12 10:55:00 | 1463.50 | 2025-09-12 11:05:00 | 1465.90 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-09-15 11:00:00 | 1454.60 | 2025-09-15 11:30:00 | 1456.26 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2025-09-16 10:45:00 | 1470.10 | 2025-09-16 10:50:00 | 1468.17 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-09-19 11:15:00 | 1477.00 | 2025-09-19 12:00:00 | 1474.49 | PARTIAL | 0.50 | 0.17% |
| SELL | retest1 | 2025-09-19 11:15:00 | 1477.00 | 2025-09-19 15:20:00 | 1469.10 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2025-09-22 11:05:00 | 1436.40 | 2025-09-22 11:40:00 | 1441.47 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-09-22 11:05:00 | 1436.40 | 2025-09-22 14:40:00 | 1437.50 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2025-09-23 11:05:00 | 1422.20 | 2025-09-23 12:05:00 | 1424.31 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-09-26 11:15:00 | 1404.50 | 2025-09-26 11:35:00 | 1401.44 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-09-26 11:15:00 | 1404.50 | 2025-09-26 15:20:00 | 1395.10 | TARGET_HIT | 0.50 | 0.67% |
| BUY | retest1 | 2025-09-30 10:40:00 | 1398.00 | 2025-09-30 11:10:00 | 1402.04 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-09-30 10:40:00 | 1398.00 | 2025-09-30 13:25:00 | 1398.70 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2025-10-06 11:10:00 | 1414.60 | 2025-10-06 11:25:00 | 1411.09 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-07 10:50:00 | 1430.10 | 2025-10-07 11:40:00 | 1427.07 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-15 10:45:00 | 1493.60 | 2025-10-15 11:50:00 | 1497.48 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-16 10:45:00 | 1512.00 | 2025-10-16 12:30:00 | 1516.52 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-10-16 10:45:00 | 1512.00 | 2025-10-16 15:20:00 | 1514.00 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2025-10-23 09:45:00 | 1526.30 | 2025-10-23 10:20:00 | 1534.25 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-10-23 09:45:00 | 1526.30 | 2025-10-23 14:25:00 | 1532.50 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2025-10-27 10:40:00 | 1545.00 | 2025-10-27 11:55:00 | 1549.46 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-10-27 10:40:00 | 1545.00 | 2025-10-27 12:15:00 | 1545.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-28 11:15:00 | 1518.50 | 2025-10-28 12:05:00 | 1514.36 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-10-28 11:15:00 | 1518.50 | 2025-10-28 12:20:00 | 1518.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-29 10:45:00 | 1543.50 | 2025-10-29 11:05:00 | 1548.80 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-10-29 10:45:00 | 1543.50 | 2025-10-29 15:20:00 | 1557.90 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2025-10-30 10:45:00 | 1540.70 | 2025-10-30 11:00:00 | 1543.27 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-03 11:15:00 | 1548.60 | 2025-11-03 11:25:00 | 1545.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-10 10:45:00 | 1538.20 | 2025-11-10 11:20:00 | 1543.42 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-11-10 10:45:00 | 1538.20 | 2025-11-10 11:55:00 | 1538.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 10:55:00 | 1591.30 | 2025-11-12 12:25:00 | 1596.40 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-11-12 10:55:00 | 1591.30 | 2025-11-12 15:20:00 | 1591.10 | STOP_HIT | 0.50 | -0.01% |
| BUY | retest1 | 2025-11-13 11:00:00 | 1599.80 | 2025-11-13 11:40:00 | 1596.81 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-19 09:50:00 | 1612.10 | 2025-11-19 10:00:00 | 1617.12 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-11-19 09:50:00 | 1612.10 | 2025-11-19 15:20:00 | 1662.50 | TARGET_HIT | 0.50 | 3.13% |
| BUY | retest1 | 2025-11-27 10:50:00 | 1624.50 | 2025-11-27 12:35:00 | 1621.27 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-02 11:00:00 | 1630.80 | 2025-12-02 11:10:00 | 1633.66 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-12-04 10:05:00 | 1662.20 | 2025-12-04 10:15:00 | 1668.51 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-12-04 10:05:00 | 1662.20 | 2025-12-04 11:05:00 | 1662.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 10:25:00 | 1665.80 | 2025-12-05 10:30:00 | 1662.40 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-11 10:55:00 | 1660.60 | 2025-12-11 11:45:00 | 1664.39 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-12 10:40:00 | 1664.10 | 2025-12-12 11:20:00 | 1667.06 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-12-15 10:55:00 | 1683.20 | 2025-12-15 11:00:00 | 1680.39 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-16 10:55:00 | 1655.20 | 2025-12-16 11:55:00 | 1650.37 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-12-16 10:55:00 | 1655.20 | 2025-12-16 15:20:00 | 1652.30 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2025-12-18 11:10:00 | 1656.80 | 2025-12-18 11:25:00 | 1659.70 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-19 09:55:00 | 1651.20 | 2025-12-19 10:10:00 | 1645.37 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-19 09:55:00 | 1651.20 | 2025-12-19 10:25:00 | 1651.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-22 10:55:00 | 1665.60 | 2025-12-22 13:00:00 | 1670.06 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-12-22 10:55:00 | 1665.60 | 2025-12-22 13:35:00 | 1665.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-26 11:00:00 | 1662.00 | 2025-12-26 11:20:00 | 1658.79 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2025-12-26 11:00:00 | 1662.00 | 2025-12-26 13:45:00 | 1662.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 10:55:00 | 1646.30 | 2025-12-29 11:10:00 | 1649.03 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-01-01 10:45:00 | 1633.80 | 2026-01-01 11:20:00 | 1638.12 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-01-01 10:45:00 | 1633.80 | 2026-01-01 15:05:00 | 1636.60 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2026-02-09 09:35:00 | 1598.00 | 2026-02-09 10:00:00 | 1602.81 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-10 09:45:00 | 1598.00 | 2026-02-10 09:55:00 | 1594.26 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-11 09:30:00 | 1563.40 | 2026-02-11 09:45:00 | 1558.18 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-11 09:30:00 | 1563.40 | 2026-02-11 14:10:00 | 1558.30 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-18 10:05:00 | 1449.30 | 2026-02-18 10:15:00 | 1455.48 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-23 10:40:00 | 1417.50 | 2026-02-23 10:55:00 | 1421.71 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-09 10:55:00 | 1361.90 | 2026-03-09 11:15:00 | 1368.70 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-03-09 10:55:00 | 1361.90 | 2026-03-09 12:05:00 | 1361.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 11:15:00 | 1348.10 | 2026-03-10 11:30:00 | 1351.21 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-11 10:40:00 | 1354.50 | 2026-03-11 11:00:00 | 1358.10 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-20 11:15:00 | 1433.30 | 2026-04-20 11:20:00 | 1436.30 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1209.40 | 2026-04-28 12:25:00 | 1205.33 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1209.40 | 2026-04-28 15:20:00 | 1195.90 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2026-04-29 10:55:00 | 1206.20 | 2026-04-29 11:40:00 | 1203.20 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-05-06 11:10:00 | 1197.40 | 2026-05-06 11:15:00 | 1199.36 | STOP_HIT | 1.00 | -0.16% |
