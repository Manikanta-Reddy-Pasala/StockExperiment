# Bajaj Finserv Ltd. (BAJAJFINSV)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-01-23 15:25:00 (50181 bars)
- **Last close:** 1955.00
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
| ENTRY1 | 107 |
| ENTRY2 | 0 |
| PARTIAL | 41 |
| TARGET_HIT | 19 |
| STOP_HIT | 88 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 148 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 88
- **Target hits / Stop hits / Partials:** 19 / 88 / 41
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 16.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 28 | 40.6% | 9 | 41 | 19 | 0.08% | 5.2% |
| BUY @ 2nd Alert (retest1) | 69 | 28 | 40.6% | 9 | 41 | 19 | 0.08% | 5.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 79 | 32 | 40.5% | 10 | 47 | 22 | 0.14% | 11.2% |
| SELL @ 2nd Alert (retest1) | 79 | 32 | 40.5% | 10 | 47 | 22 | 0.14% | 11.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 148 | 60 | 40.5% | 19 | 88 | 41 | 0.11% | 16.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 10:25:00 | 1423.95 | 1416.85 | 0.00 | ORB-long ORB[1415.45,1423.00] vol=2.7x ATR=5.48 |
| Stop hit — per-position SL triggered | 2023-05-12 10:55:00 | 1418.47 | 1418.73 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:30:00 | 1432.95 | 1428.83 | 0.00 | ORB-long ORB[1421.00,1432.10] vol=3.2x ATR=4.24 |
| Stop hit — per-position SL triggered | 2023-05-16 09:35:00 | 1428.71 | 1429.15 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 11:00:00 | 1412.50 | 1421.71 | 0.00 | ORB-short ORB[1423.00,1430.40] vol=2.8x ATR=2.97 |
| Stop hit — per-position SL triggered | 2023-05-17 11:05:00 | 1415.47 | 1421.16 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:35:00 | 1408.95 | 1415.56 | 0.00 | ORB-short ORB[1413.00,1422.70] vol=1.9x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 09:45:00 | 1403.46 | 1412.07 | 0.00 | T1 1.5R @ 1403.46 |
| Target hit | 2023-05-19 10:45:00 | 1406.80 | 1406.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2023-05-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 10:30:00 | 1434.05 | 1427.79 | 0.00 | ORB-long ORB[1415.50,1432.50] vol=2.6x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-23 10:35:00 | 1439.09 | 1429.64 | 0.00 | T1 1.5R @ 1439.09 |
| Target hit | 2023-05-23 14:40:00 | 1435.50 | 1437.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2023-05-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 09:30:00 | 1418.20 | 1423.00 | 0.00 | ORB-short ORB[1420.00,1428.70] vol=1.7x ATR=3.42 |
| Stop hit — per-position SL triggered | 2023-05-25 09:40:00 | 1421.62 | 1422.46 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 11:15:00 | 1465.25 | 1458.02 | 0.00 | ORB-long ORB[1447.55,1463.45] vol=3.4x ATR=3.78 |
| Stop hit — per-position SL triggered | 2023-05-29 11:35:00 | 1461.47 | 1458.38 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 10:00:00 | 1449.95 | 1454.46 | 0.00 | ORB-short ORB[1453.35,1460.00] vol=2.2x ATR=3.21 |
| Stop hit — per-position SL triggered | 2023-06-02 10:25:00 | 1453.16 | 1453.90 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:40:00 | 1485.30 | 1479.78 | 0.00 | ORB-long ORB[1470.00,1482.95] vol=3.2x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 12:05:00 | 1489.81 | 1482.53 | 0.00 | T1 1.5R @ 1489.81 |
| Stop hit — per-position SL triggered | 2023-06-13 13:20:00 | 1485.30 | 1484.19 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 11:00:00 | 1474.45 | 1474.78 | 0.00 | ORB-short ORB[1477.70,1491.90] vol=1.9x ATR=2.51 |
| Stop hit — per-position SL triggered | 2023-06-14 13:10:00 | 1476.96 | 1474.51 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:30:00 | 1485.05 | 1480.33 | 0.00 | ORB-long ORB[1474.70,1481.55] vol=1.7x ATR=3.18 |
| Stop hit — per-position SL triggered | 2023-06-16 09:35:00 | 1481.87 | 1481.10 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 11:05:00 | 1504.75 | 1518.78 | 0.00 | ORB-short ORB[1524.00,1543.20] vol=1.6x ATR=3.77 |
| Stop hit — per-position SL triggered | 2023-06-20 12:10:00 | 1508.52 | 1515.66 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 09:35:00 | 1487.40 | 1493.21 | 0.00 | ORB-short ORB[1489.05,1507.65] vol=1.8x ATR=4.45 |
| Stop hit — per-position SL triggered | 2023-06-23 09:50:00 | 1491.85 | 1491.91 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:30:00 | 1534.25 | 1530.13 | 0.00 | ORB-long ORB[1519.00,1532.95] vol=2.0x ATR=3.60 |
| Stop hit — per-position SL triggered | 2023-06-30 09:45:00 | 1530.65 | 1530.64 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 10:10:00 | 1615.00 | 1607.93 | 0.00 | ORB-long ORB[1595.00,1607.95] vol=1.5x ATR=4.24 |
| Stop hit — per-position SL triggered | 2023-07-13 10:15:00 | 1610.76 | 1608.27 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-14 10:25:00 | 1611.50 | 1621.38 | 0.00 | ORB-short ORB[1621.55,1635.00] vol=2.1x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 10:45:00 | 1604.77 | 1619.70 | 0.00 | T1 1.5R @ 1604.77 |
| Stop hit — per-position SL triggered | 2023-07-14 11:10:00 | 1611.50 | 1618.46 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 10:55:00 | 1605.20 | 1612.39 | 0.00 | ORB-short ORB[1607.10,1621.80] vol=1.6x ATR=3.56 |
| Stop hit — per-position SL triggered | 2023-07-17 11:05:00 | 1608.76 | 1612.14 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 11:00:00 | 1622.95 | 1615.55 | 0.00 | ORB-long ORB[1610.00,1620.80] vol=1.8x ATR=3.55 |
| Stop hit — per-position SL triggered | 2023-07-18 11:05:00 | 1619.40 | 1615.65 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 10:40:00 | 1626.05 | 1622.51 | 0.00 | ORB-long ORB[1614.00,1625.00] vol=2.0x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 12:25:00 | 1631.88 | 1624.83 | 0.00 | T1 1.5R @ 1631.88 |
| Target hit | 2023-07-19 15:20:00 | 1644.80 | 1633.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2023-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 10:50:00 | 1649.10 | 1638.48 | 0.00 | ORB-long ORB[1622.00,1638.50] vol=5.2x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 14:40:00 | 1655.46 | 1646.47 | 0.00 | T1 1.5R @ 1655.46 |
| Target hit | 2023-07-24 15:20:00 | 1658.70 | 1648.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2023-07-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 10:50:00 | 1643.75 | 1651.42 | 0.00 | ORB-short ORB[1652.45,1664.90] vol=1.7x ATR=4.13 |
| Stop hit — per-position SL triggered | 2023-07-25 11:10:00 | 1647.88 | 1650.68 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:30:00 | 1657.15 | 1652.58 | 0.00 | ORB-long ORB[1645.85,1656.00] vol=2.2x ATR=3.90 |
| Stop hit — per-position SL triggered | 2023-07-26 11:05:00 | 1653.25 | 1653.62 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 10:35:00 | 1662.20 | 1649.45 | 0.00 | ORB-long ORB[1629.00,1648.70] vol=1.5x ATR=5.01 |
| Stop hit — per-position SL triggered | 2023-07-27 10:50:00 | 1657.19 | 1650.55 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 10:25:00 | 1593.00 | 1588.48 | 0.00 | ORB-long ORB[1565.80,1587.95] vol=3.9x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 11:30:00 | 1598.87 | 1589.53 | 0.00 | T1 1.5R @ 1598.87 |
| Stop hit — per-position SL triggered | 2023-07-31 11:40:00 | 1593.00 | 1589.64 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 11:10:00 | 1579.65 | 1586.86 | 0.00 | ORB-short ORB[1585.00,1599.95] vol=1.8x ATR=3.45 |
| Stop hit — per-position SL triggered | 2023-08-01 11:25:00 | 1583.10 | 1586.50 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:15:00 | 1551.10 | 1558.17 | 0.00 | ORB-short ORB[1555.20,1572.35] vol=1.5x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 11:00:00 | 1545.19 | 1555.58 | 0.00 | T1 1.5R @ 1545.19 |
| Stop hit — per-position SL triggered | 2023-08-02 12:05:00 | 1551.10 | 1552.82 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 10:30:00 | 1484.60 | 1493.44 | 0.00 | ORB-short ORB[1493.70,1507.35] vol=1.9x ATR=4.52 |
| Stop hit — per-position SL triggered | 2023-08-04 11:05:00 | 1489.12 | 1489.89 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 11:15:00 | 1508.20 | 1515.80 | 0.00 | ORB-short ORB[1516.05,1529.75] vol=2.4x ATR=3.10 |
| Stop hit — per-position SL triggered | 2023-08-09 11:30:00 | 1511.30 | 1515.24 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:20:00 | 1509.35 | 1515.55 | 0.00 | ORB-short ORB[1510.50,1522.60] vol=2.2x ATR=3.67 |
| Stop hit — per-position SL triggered | 2023-08-10 11:05:00 | 1513.02 | 1513.41 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-14 09:40:00 | 1477.85 | 1483.45 | 0.00 | ORB-short ORB[1478.00,1497.95] vol=2.0x ATR=4.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-14 11:10:00 | 1470.46 | 1478.13 | 0.00 | T1 1.5R @ 1470.46 |
| Stop hit — per-position SL triggered | 2023-08-14 11:30:00 | 1477.85 | 1477.41 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 11:10:00 | 1462.85 | 1465.14 | 0.00 | ORB-short ORB[1465.50,1478.30] vol=2.2x ATR=3.11 |
| Stop hit — per-position SL triggered | 2023-08-18 11:35:00 | 1465.96 | 1464.69 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 09:55:00 | 1462.65 | 1456.58 | 0.00 | ORB-long ORB[1452.20,1461.00] vol=2.0x ATR=3.73 |
| Stop hit — per-position SL triggered | 2023-08-21 10:00:00 | 1458.92 | 1456.75 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 09:40:00 | 1487.55 | 1484.88 | 0.00 | ORB-long ORB[1480.60,1487.25] vol=1.9x ATR=3.26 |
| Stop hit — per-position SL triggered | 2023-08-22 09:50:00 | 1484.29 | 1485.18 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-08-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-28 11:00:00 | 1493.95 | 1500.22 | 0.00 | ORB-short ORB[1496.10,1514.00] vol=1.9x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 11:55:00 | 1488.94 | 1498.18 | 0.00 | T1 1.5R @ 1488.94 |
| Stop hit — per-position SL triggered | 2023-08-28 12:25:00 | 1493.95 | 1497.07 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 09:45:00 | 1510.90 | 1507.03 | 0.00 | ORB-long ORB[1502.60,1510.00] vol=2.6x ATR=3.17 |
| Stop hit — per-position SL triggered | 2023-08-29 10:00:00 | 1507.73 | 1507.96 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 11:15:00 | 1518.65 | 1510.64 | 0.00 | ORB-long ORB[1505.00,1513.90] vol=4.7x ATR=2.43 |
| Stop hit — per-position SL triggered | 2023-08-30 11:30:00 | 1516.22 | 1511.42 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 11:15:00 | 1502.95 | 1497.58 | 0.00 | ORB-long ORB[1487.35,1495.00] vol=2.5x ATR=3.45 |
| Stop hit — per-position SL triggered | 2023-09-01 11:25:00 | 1499.50 | 1497.80 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 10:15:00 | 1508.90 | 1514.19 | 0.00 | ORB-short ORB[1510.10,1519.80] vol=1.7x ATR=3.19 |
| Stop hit — per-position SL triggered | 2023-09-05 10:30:00 | 1512.09 | 1513.45 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-09-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:50:00 | 1531.75 | 1545.81 | 0.00 | ORB-short ORB[1545.75,1558.90] vol=1.6x ATR=4.14 |
| Stop hit — per-position SL triggered | 2023-09-12 09:55:00 | 1535.89 | 1544.32 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 10:50:00 | 1561.50 | 1552.11 | 0.00 | ORB-long ORB[1546.35,1556.00] vol=2.4x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 11:30:00 | 1566.64 | 1555.93 | 0.00 | T1 1.5R @ 1566.64 |
| Stop hit — per-position SL triggered | 2023-09-15 11:45:00 | 1561.50 | 1556.48 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 10:05:00 | 1524.70 | 1539.04 | 0.00 | ORB-short ORB[1531.45,1550.00] vol=1.8x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 10:20:00 | 1516.96 | 1533.94 | 0.00 | T1 1.5R @ 1516.96 |
| Stop hit — per-position SL triggered | 2023-09-21 11:05:00 | 1524.70 | 1525.89 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 09:30:00 | 1563.50 | 1572.33 | 0.00 | ORB-short ORB[1567.00,1583.00] vol=1.7x ATR=3.90 |
| Stop hit — per-position SL triggered | 2023-09-26 09:35:00 | 1567.40 | 1571.66 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 09:30:00 | 1561.65 | 1565.44 | 0.00 | ORB-short ORB[1563.00,1574.95] vol=2.6x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 09:50:00 | 1556.86 | 1563.57 | 0.00 | T1 1.5R @ 1556.86 |
| Stop hit — per-position SL triggered | 2023-09-28 10:20:00 | 1561.65 | 1562.48 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 10:55:00 | 1543.25 | 1536.43 | 0.00 | ORB-long ORB[1527.00,1537.30] vol=1.5x ATR=3.55 |
| Stop hit — per-position SL triggered | 2023-10-03 11:00:00 | 1539.70 | 1536.69 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 10:40:00 | 1650.00 | 1644.29 | 0.00 | ORB-long ORB[1637.20,1645.35] vol=2.2x ATR=3.26 |
| Stop hit — per-position SL triggered | 2023-10-11 10:50:00 | 1646.74 | 1644.61 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-10-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-27 10:45:00 | 1560.25 | 1563.25 | 0.00 | ORB-short ORB[1562.00,1576.25] vol=1.6x ATR=4.87 |
| Stop hit — per-position SL triggered | 2023-10-27 11:20:00 | 1565.12 | 1563.21 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-11-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 10:10:00 | 1592.05 | 1588.13 | 0.00 | ORB-long ORB[1580.00,1588.60] vol=2.1x ATR=4.06 |
| Stop hit — per-position SL triggered | 2023-11-03 10:30:00 | 1587.99 | 1588.64 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-11-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 10:55:00 | 1583.60 | 1578.32 | 0.00 | ORB-long ORB[1572.65,1581.20] vol=2.3x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-10 12:05:00 | 1588.44 | 1581.71 | 0.00 | T1 1.5R @ 1588.44 |
| Target hit | 2023-11-10 15:20:00 | 1595.15 | 1589.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:15:00 | 1572.15 | 1584.25 | 0.00 | ORB-short ORB[1586.50,1597.75] vol=3.8x ATR=4.43 |
| Stop hit — per-position SL triggered | 2023-11-13 10:25:00 | 1576.58 | 1582.50 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 10:50:00 | 1607.65 | 1599.10 | 0.00 | ORB-long ORB[1592.60,1603.00] vol=2.6x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 13:10:00 | 1613.44 | 1603.01 | 0.00 | T1 1.5R @ 1613.44 |
| Stop hit — per-position SL triggered | 2023-11-15 13:35:00 | 1607.65 | 1604.60 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 10:55:00 | 1599.80 | 1607.14 | 0.00 | ORB-short ORB[1608.10,1616.55] vol=2.4x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 11:05:00 | 1594.66 | 1604.79 | 0.00 | T1 1.5R @ 1594.66 |
| Stop hit — per-position SL triggered | 2023-11-20 12:00:00 | 1599.80 | 1600.30 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 11:15:00 | 1618.80 | 1612.15 | 0.00 | ORB-long ORB[1602.50,1612.35] vol=3.8x ATR=2.87 |
| Stop hit — per-position SL triggered | 2023-11-22 11:55:00 | 1615.93 | 1613.89 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:30:00 | 1616.65 | 1622.10 | 0.00 | ORB-short ORB[1620.00,1628.95] vol=1.7x ATR=3.30 |
| Stop hit — per-position SL triggered | 2023-11-24 10:05:00 | 1619.95 | 1619.93 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 09:45:00 | 1630.50 | 1625.71 | 0.00 | ORB-long ORB[1617.50,1626.75] vol=1.7x ATR=3.08 |
| Stop hit — per-position SL triggered | 2023-11-28 10:10:00 | 1627.42 | 1627.17 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-29 10:35:00 | 1651.45 | 1657.71 | 0.00 | ORB-short ORB[1654.15,1665.00] vol=1.6x ATR=3.04 |
| Stop hit — per-position SL triggered | 2023-11-29 10:45:00 | 1654.49 | 1657.53 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 11:15:00 | 1697.95 | 1686.24 | 0.00 | ORB-long ORB[1673.25,1696.75] vol=1.7x ATR=5.82 |
| Stop hit — per-position SL triggered | 2023-12-05 11:25:00 | 1692.13 | 1686.80 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-12-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 11:05:00 | 1706.00 | 1702.97 | 0.00 | ORB-long ORB[1697.10,1704.00] vol=2.1x ATR=3.65 |
| Stop hit — per-position SL triggered | 2023-12-06 11:40:00 | 1702.35 | 1703.71 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-12-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 10:00:00 | 1707.95 | 1697.86 | 0.00 | ORB-long ORB[1690.00,1702.00] vol=2.8x ATR=4.12 |
| Stop hit — per-position SL triggered | 2023-12-07 10:10:00 | 1703.83 | 1698.92 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-12-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 09:30:00 | 1699.65 | 1706.96 | 0.00 | ORB-short ORB[1705.00,1712.25] vol=1.5x ATR=3.87 |
| Stop hit — per-position SL triggered | 2023-12-08 09:40:00 | 1703.52 | 1705.48 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 10:15:00 | 1718.00 | 1713.61 | 0.00 | ORB-long ORB[1702.80,1711.10] vol=1.6x ATR=3.24 |
| Stop hit — per-position SL triggered | 2023-12-12 10:30:00 | 1714.76 | 1714.06 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-12-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 10:50:00 | 1668.90 | 1685.75 | 0.00 | ORB-short ORB[1700.00,1709.50] vol=1.7x ATR=3.85 |
| Stop hit — per-position SL triggered | 2023-12-13 11:00:00 | 1672.75 | 1684.04 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 09:45:00 | 1719.20 | 1710.03 | 0.00 | ORB-long ORB[1698.00,1715.95] vol=2.2x ATR=4.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 09:55:00 | 1726.68 | 1714.86 | 0.00 | T1 1.5R @ 1726.68 |
| Target hit | 2023-12-14 10:30:00 | 1719.70 | 1719.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — SELL (started 2023-12-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 10:45:00 | 1722.00 | 1729.24 | 0.00 | ORB-short ORB[1725.70,1741.00] vol=1.5x ATR=4.31 |
| Stop hit — per-position SL triggered | 2023-12-15 11:45:00 | 1726.31 | 1727.53 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:35:00 | 1710.80 | 1717.56 | 0.00 | ORB-short ORB[1717.30,1726.55] vol=1.9x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 09:40:00 | 1705.38 | 1713.47 | 0.00 | T1 1.5R @ 1705.38 |
| Target hit | 2023-12-19 10:30:00 | 1709.50 | 1706.70 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — BUY (started 2023-12-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 10:30:00 | 1729.90 | 1727.69 | 0.00 | ORB-long ORB[1713.40,1722.10] vol=2.0x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 10:35:00 | 1734.53 | 1728.60 | 0.00 | T1 1.5R @ 1734.53 |
| Target hit | 2023-12-20 11:05:00 | 1730.50 | 1731.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — BUY (started 2023-12-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:40:00 | 1687.65 | 1677.32 | 0.00 | ORB-long ORB[1668.00,1683.00] vol=1.8x ATR=4.34 |
| Stop hit — per-position SL triggered | 2023-12-22 12:15:00 | 1683.31 | 1680.96 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2023-12-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-26 10:40:00 | 1669.00 | 1676.60 | 0.00 | ORB-short ORB[1672.05,1684.00] vol=1.8x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 11:00:00 | 1662.63 | 1672.71 | 0.00 | T1 1.5R @ 1662.63 |
| Target hit | 2023-12-26 15:20:00 | 1647.20 | 1656.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2023-12-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:35:00 | 1689.90 | 1682.76 | 0.00 | ORB-long ORB[1674.05,1684.95] vol=2.1x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 09:40:00 | 1695.87 | 1686.96 | 0.00 | T1 1.5R @ 1695.87 |
| Stop hit — per-position SL triggered | 2023-12-28 09:55:00 | 1689.90 | 1688.11 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 11:15:00 | 1677.70 | 1683.59 | 0.00 | ORB-short ORB[1680.00,1687.00] vol=2.2x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 11:45:00 | 1674.07 | 1681.59 | 0.00 | T1 1.5R @ 1674.07 |
| Stop hit — per-position SL triggered | 2024-01-01 11:50:00 | 1677.70 | 1681.43 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:35:00 | 1684.50 | 1680.73 | 0.00 | ORB-long ORB[1672.00,1684.25] vol=1.7x ATR=3.82 |
| Stop hit — per-position SL triggered | 2024-01-02 09:50:00 | 1680.68 | 1681.95 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:30:00 | 1702.50 | 1707.52 | 0.00 | ORB-short ORB[1705.00,1715.00] vol=1.9x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 09:50:00 | 1696.61 | 1704.13 | 0.00 | T1 1.5R @ 1696.61 |
| Stop hit — per-position SL triggered | 2024-01-09 10:00:00 | 1702.50 | 1703.84 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 09:35:00 | 1698.20 | 1691.71 | 0.00 | ORB-long ORB[1678.95,1696.35] vol=2.8x ATR=4.55 |
| Stop hit — per-position SL triggered | 2024-01-11 10:00:00 | 1693.65 | 1694.46 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 09:50:00 | 1655.70 | 1662.13 | 0.00 | ORB-short ORB[1660.00,1674.95] vol=1.7x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-01-12 10:10:00 | 1659.72 | 1660.24 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 10:30:00 | 1642.00 | 1650.51 | 0.00 | ORB-short ORB[1646.80,1664.15] vol=1.6x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 10:50:00 | 1635.59 | 1646.99 | 0.00 | T1 1.5R @ 1635.59 |
| Target hit | 2024-01-15 15:20:00 | 1633.85 | 1639.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2024-01-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 10:35:00 | 1598.45 | 1608.79 | 0.00 | ORB-short ORB[1603.30,1622.00] vol=3.1x ATR=4.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 11:45:00 | 1591.66 | 1602.61 | 0.00 | T1 1.5R @ 1591.66 |
| Target hit | 2024-01-17 15:20:00 | 1587.50 | 1593.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — BUY (started 2024-01-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 09:35:00 | 1603.10 | 1598.59 | 0.00 | ORB-long ORB[1590.00,1601.85] vol=2.1x ATR=4.46 |
| Stop hit — per-position SL triggered | 2024-01-19 09:45:00 | 1598.64 | 1599.05 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-01-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:55:00 | 1617.75 | 1623.28 | 0.00 | ORB-short ORB[1618.35,1632.10] vol=2.0x ATR=3.21 |
| Stop hit — per-position SL triggered | 2024-01-25 11:05:00 | 1620.96 | 1622.92 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 11:15:00 | 1622.65 | 1631.32 | 0.00 | ORB-short ORB[1624.85,1636.50] vol=1.6x ATR=4.49 |
| Stop hit — per-position SL triggered | 2024-01-29 12:40:00 | 1627.14 | 1627.81 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-01-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 10:40:00 | 1609.50 | 1601.25 | 0.00 | ORB-long ORB[1585.00,1606.00] vol=1.6x ATR=5.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 12:15:00 | 1617.15 | 1606.59 | 0.00 | T1 1.5R @ 1617.15 |
| Target hit | 2024-01-31 15:20:00 | 1621.75 | 1616.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2024-02-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 10:35:00 | 1651.70 | 1641.58 | 0.00 | ORB-long ORB[1626.70,1641.90] vol=2.4x ATR=4.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:15:00 | 1658.60 | 1644.83 | 0.00 | T1 1.5R @ 1658.60 |
| Stop hit — per-position SL triggered | 2024-02-02 12:25:00 | 1651.70 | 1649.17 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 10:15:00 | 1641.80 | 1649.75 | 0.00 | ORB-short ORB[1642.80,1655.00] vol=1.5x ATR=4.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 11:20:00 | 1634.83 | 1644.27 | 0.00 | T1 1.5R @ 1634.83 |
| Target hit | 2024-02-05 15:20:00 | 1614.00 | 1624.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:15:00 | 1586.05 | 1593.13 | 0.00 | ORB-short ORB[1597.20,1607.95] vol=2.4x ATR=4.72 |
| Stop hit — per-position SL triggered | 2024-02-08 10:30:00 | 1590.77 | 1591.80 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 11:00:00 | 1567.95 | 1573.35 | 0.00 | ORB-short ORB[1570.25,1587.30] vol=1.7x ATR=4.69 |
| Stop hit — per-position SL triggered | 2024-02-09 11:40:00 | 1572.64 | 1571.95 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-02-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 10:25:00 | 1559.15 | 1566.26 | 0.00 | ORB-short ORB[1565.00,1579.30] vol=3.1x ATR=4.12 |
| Stop hit — per-position SL triggered | 2024-02-12 10:30:00 | 1563.27 | 1565.80 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 09:35:00 | 1570.20 | 1565.39 | 0.00 | ORB-long ORB[1552.20,1569.75] vol=4.3x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 10:05:00 | 1578.48 | 1568.90 | 0.00 | T1 1.5R @ 1578.48 |
| Target hit | 2024-02-13 11:10:00 | 1575.00 | 1575.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 86 — BUY (started 2024-02-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 11:00:00 | 1569.75 | 1566.17 | 0.00 | ORB-long ORB[1555.00,1568.30] vol=2.6x ATR=3.90 |
| Stop hit — per-position SL triggered | 2024-02-14 11:35:00 | 1565.85 | 1567.31 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-02-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 10:20:00 | 1577.55 | 1582.22 | 0.00 | ORB-short ORB[1579.70,1590.00] vol=2.3x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 10:30:00 | 1570.93 | 1580.24 | 0.00 | T1 1.5R @ 1570.93 |
| Target hit | 2024-02-15 15:20:00 | 1570.10 | 1572.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — BUY (started 2024-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 09:35:00 | 1589.10 | 1583.52 | 0.00 | ORB-long ORB[1572.00,1585.50] vol=1.7x ATR=3.62 |
| Stop hit — per-position SL triggered | 2024-02-19 09:45:00 | 1585.48 | 1584.17 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 11:15:00 | 1603.35 | 1610.89 | 0.00 | ORB-short ORB[1604.00,1618.60] vol=2.1x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 11:25:00 | 1599.23 | 1609.69 | 0.00 | T1 1.5R @ 1599.23 |
| Target hit | 2024-02-21 15:20:00 | 1585.00 | 1598.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 90 — SELL (started 2024-02-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 10:05:00 | 1601.15 | 1609.96 | 0.00 | ORB-short ORB[1609.55,1623.45] vol=2.6x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-02-27 10:10:00 | 1605.43 | 1609.51 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-02-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 10:45:00 | 1573.10 | 1585.65 | 0.00 | ORB-short ORB[1578.85,1591.20] vol=3.4x ATR=4.94 |
| Stop hit — per-position SL triggered | 2024-02-29 11:05:00 | 1578.04 | 1582.11 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-03-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:40:00 | 1604.00 | 1610.70 | 0.00 | ORB-short ORB[1606.20,1621.95] vol=1.8x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-03-04 09:50:00 | 1607.70 | 1610.14 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-03-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 10:20:00 | 1613.35 | 1622.40 | 0.00 | ORB-short ORB[1623.05,1636.60] vol=1.7x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 10:50:00 | 1607.42 | 1620.59 | 0.00 | T1 1.5R @ 1607.42 |
| Target hit | 2024-03-05 15:20:00 | 1551.40 | 1577.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 94 — BUY (started 2024-03-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 09:45:00 | 1601.60 | 1594.24 | 0.00 | ORB-long ORB[1583.00,1598.90] vol=1.6x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 11:45:00 | 1608.19 | 1599.88 | 0.00 | T1 1.5R @ 1608.19 |
| Stop hit — per-position SL triggered | 2024-03-11 15:10:00 | 1601.60 | 1604.04 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 09:30:00 | 1601.20 | 1591.52 | 0.00 | ORB-long ORB[1582.25,1600.00] vol=2.5x ATR=5.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:35:00 | 1610.13 | 1598.27 | 0.00 | T1 1.5R @ 1610.13 |
| Stop hit — per-position SL triggered | 2024-03-12 09:50:00 | 1601.20 | 1599.28 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-03-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 10:05:00 | 1586.35 | 1571.08 | 0.00 | ORB-long ORB[1547.50,1569.90] vol=2.9x ATR=5.78 |
| Stop hit — per-position SL triggered | 2024-03-15 10:25:00 | 1580.57 | 1575.12 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-03-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:45:00 | 1553.55 | 1560.66 | 0.00 | ORB-short ORB[1556.05,1574.05] vol=2.7x ATR=4.45 |
| Stop hit — per-position SL triggered | 2024-03-19 09:50:00 | 1558.00 | 1560.23 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2024-03-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:30:00 | 1614.60 | 1603.66 | 0.00 | ORB-long ORB[1588.00,1608.20] vol=2.1x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 09:40:00 | 1621.09 | 1610.79 | 0.00 | T1 1.5R @ 1621.09 |
| Target hit | 2024-03-28 15:05:00 | 1644.40 | 1644.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 99 — SELL (started 2024-04-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-02 09:45:00 | 1634.05 | 1640.20 | 0.00 | ORB-short ORB[1636.85,1654.00] vol=2.7x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 10:05:00 | 1626.55 | 1636.71 | 0.00 | T1 1.5R @ 1626.55 |
| Stop hit — per-position SL triggered | 2024-04-02 10:30:00 | 1634.05 | 1634.94 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 1630.35 | 1638.34 | 0.00 | ORB-short ORB[1640.35,1650.30] vol=2.0x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-04-04 10:55:00 | 1635.08 | 1638.11 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2024-04-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:40:00 | 1695.85 | 1701.24 | 0.00 | ORB-short ORB[1697.00,1706.15] vol=2.6x ATR=3.95 |
| Stop hit — per-position SL triggered | 2024-04-12 09:45:00 | 1699.80 | 1700.25 | 0.00 | SL hit |

### Cycle 102 — SELL (started 2024-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-16 09:40:00 | 1631.80 | 1639.24 | 0.00 | ORB-short ORB[1637.05,1648.70] vol=1.6x ATR=5.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 10:05:00 | 1623.84 | 1635.12 | 0.00 | T1 1.5R @ 1623.84 |
| Stop hit — per-position SL triggered | 2024-04-16 10:10:00 | 1631.80 | 1634.46 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2024-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 09:30:00 | 1646.00 | 1639.21 | 0.00 | ORB-long ORB[1625.00,1639.20] vol=5.4x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-04-23 09:40:00 | 1641.73 | 1642.01 | 0.00 | SL hit |

### Cycle 104 — BUY (started 2024-04-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:40:00 | 1636.70 | 1630.26 | 0.00 | ORB-long ORB[1622.30,1629.15] vol=3.3x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 11:10:00 | 1642.03 | 1631.94 | 0.00 | T1 1.5R @ 1642.03 |
| Stop hit — per-position SL triggered | 2024-04-24 12:50:00 | 1636.70 | 1636.42 | 0.00 | SL hit |

### Cycle 105 — BUY (started 2024-04-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 10:45:00 | 1611.95 | 1604.11 | 0.00 | ORB-long ORB[1591.30,1607.25] vol=2.2x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 10:50:00 | 1617.45 | 1606.06 | 0.00 | T1 1.5R @ 1617.45 |
| Stop hit — per-position SL triggered | 2024-04-30 11:00:00 | 1611.95 | 1607.07 | 0.00 | SL hit |

### Cycle 106 — SELL (started 2024-05-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 09:50:00 | 1605.70 | 1609.46 | 0.00 | ORB-short ORB[1606.00,1621.00] vol=3.8x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:00:00 | 1598.86 | 1608.11 | 0.00 | T1 1.5R @ 1598.86 |
| Stop hit — per-position SL triggered | 2024-05-07 10:55:00 | 1605.70 | 1605.90 | 0.00 | SL hit |

### Cycle 107 — SELL (started 2024-05-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 11:00:00 | 1587.05 | 1593.49 | 0.00 | ORB-short ORB[1590.00,1613.10] vol=1.9x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 11:05:00 | 1581.55 | 1592.20 | 0.00 | T1 1.5R @ 1581.55 |
| Target hit | 2024-05-09 15:20:00 | 1561.65 | 1577.91 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 10:25:00 | 1423.95 | 2023-05-12 10:55:00 | 1418.47 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-05-16 09:30:00 | 1432.95 | 2023-05-16 09:35:00 | 1428.71 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-05-17 11:00:00 | 1412.50 | 2023-05-17 11:05:00 | 1415.47 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-05-19 09:35:00 | 1408.95 | 2023-05-19 09:45:00 | 1403.46 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-05-19 09:35:00 | 1408.95 | 2023-05-19 10:45:00 | 1406.80 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2023-05-23 10:30:00 | 1434.05 | 2023-05-23 10:35:00 | 1439.09 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-05-23 10:30:00 | 1434.05 | 2023-05-23 14:40:00 | 1435.50 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2023-05-25 09:30:00 | 1418.20 | 2023-05-25 09:40:00 | 1421.62 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-29 11:15:00 | 1465.25 | 2023-05-29 11:35:00 | 1461.47 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-06-02 10:00:00 | 1449.95 | 2023-06-02 10:25:00 | 1453.16 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-13 10:40:00 | 1485.30 | 2023-06-13 12:05:00 | 1489.81 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-06-13 10:40:00 | 1485.30 | 2023-06-13 13:20:00 | 1485.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-14 11:00:00 | 1474.45 | 2023-06-14 13:10:00 | 1476.96 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-06-16 09:30:00 | 1485.05 | 2023-06-16 09:35:00 | 1481.87 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-06-20 11:05:00 | 1504.75 | 2023-06-20 12:10:00 | 1508.52 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-06-23 09:35:00 | 1487.40 | 2023-06-23 09:50:00 | 1491.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-06-30 09:30:00 | 1534.25 | 2023-06-30 09:45:00 | 1530.65 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-13 10:10:00 | 1615.00 | 2023-07-13 10:15:00 | 1610.76 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-07-14 10:25:00 | 1611.50 | 2023-07-14 10:45:00 | 1604.77 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-07-14 10:25:00 | 1611.50 | 2023-07-14 11:10:00 | 1611.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-17 10:55:00 | 1605.20 | 2023-07-17 11:05:00 | 1608.76 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-18 11:00:00 | 1622.95 | 2023-07-18 11:05:00 | 1619.40 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-19 10:40:00 | 1626.05 | 2023-07-19 12:25:00 | 1631.88 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-07-19 10:40:00 | 1626.05 | 2023-07-19 15:20:00 | 1644.80 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2023-07-24 10:50:00 | 1649.10 | 2023-07-24 14:40:00 | 1655.46 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-07-24 10:50:00 | 1649.10 | 2023-07-24 15:20:00 | 1658.70 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2023-07-25 10:50:00 | 1643.75 | 2023-07-25 11:10:00 | 1647.88 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-07-26 10:30:00 | 1657.15 | 2023-07-26 11:05:00 | 1653.25 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-27 10:35:00 | 1662.20 | 2023-07-27 10:50:00 | 1657.19 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-07-31 10:25:00 | 1593.00 | 2023-07-31 11:30:00 | 1598.87 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-07-31 10:25:00 | 1593.00 | 2023-07-31 11:40:00 | 1593.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-01 11:10:00 | 1579.65 | 2023-08-01 11:25:00 | 1583.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-02 10:15:00 | 1551.10 | 2023-08-02 11:00:00 | 1545.19 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-08-02 10:15:00 | 1551.10 | 2023-08-02 12:05:00 | 1551.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-04 10:30:00 | 1484.60 | 2023-08-04 11:05:00 | 1489.12 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-08-09 11:15:00 | 1508.20 | 2023-08-09 11:30:00 | 1511.30 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-08-10 10:20:00 | 1509.35 | 2023-08-10 11:05:00 | 1513.02 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-14 09:40:00 | 1477.85 | 2023-08-14 11:10:00 | 1470.46 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-08-14 09:40:00 | 1477.85 | 2023-08-14 11:30:00 | 1477.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-18 11:10:00 | 1462.85 | 2023-08-18 11:35:00 | 1465.96 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-21 09:55:00 | 1462.65 | 2023-08-21 10:00:00 | 1458.92 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-08-22 09:40:00 | 1487.55 | 2023-08-22 09:50:00 | 1484.29 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-28 11:00:00 | 1493.95 | 2023-08-28 11:55:00 | 1488.94 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-08-28 11:00:00 | 1493.95 | 2023-08-28 12:25:00 | 1493.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-29 09:45:00 | 1510.90 | 2023-08-29 10:00:00 | 1507.73 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-30 11:15:00 | 1518.65 | 2023-08-30 11:30:00 | 1516.22 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-09-01 11:15:00 | 1502.95 | 2023-09-01 11:25:00 | 1499.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-09-05 10:15:00 | 1508.90 | 2023-09-05 10:30:00 | 1512.09 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-09-12 09:50:00 | 1531.75 | 2023-09-12 09:55:00 | 1535.89 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-09-15 10:50:00 | 1561.50 | 2023-09-15 11:30:00 | 1566.64 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-09-15 10:50:00 | 1561.50 | 2023-09-15 11:45:00 | 1561.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-21 10:05:00 | 1524.70 | 2023-09-21 10:20:00 | 1516.96 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-09-21 10:05:00 | 1524.70 | 2023-09-21 11:05:00 | 1524.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-26 09:30:00 | 1563.50 | 2023-09-26 09:35:00 | 1567.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-09-28 09:30:00 | 1561.65 | 2023-09-28 09:50:00 | 1556.86 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-09-28 09:30:00 | 1561.65 | 2023-09-28 10:20:00 | 1561.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-03 10:55:00 | 1543.25 | 2023-10-03 11:00:00 | 1539.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-10-11 10:40:00 | 1650.00 | 2023-10-11 10:50:00 | 1646.74 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-10-27 10:45:00 | 1560.25 | 2023-10-27 11:20:00 | 1565.12 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-11-03 10:10:00 | 1592.05 | 2023-11-03 10:30:00 | 1587.99 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-11-10 10:55:00 | 1583.60 | 2023-11-10 12:05:00 | 1588.44 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-11-10 10:55:00 | 1583.60 | 2023-11-10 15:20:00 | 1595.15 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2023-11-13 10:15:00 | 1572.15 | 2023-11-13 10:25:00 | 1576.58 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-11-15 10:50:00 | 1607.65 | 2023-11-15 13:10:00 | 1613.44 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-11-15 10:50:00 | 1607.65 | 2023-11-15 13:35:00 | 1607.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-20 10:55:00 | 1599.80 | 2023-11-20 11:05:00 | 1594.66 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-11-20 10:55:00 | 1599.80 | 2023-11-20 12:00:00 | 1599.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-22 11:15:00 | 1618.80 | 2023-11-22 11:55:00 | 1615.93 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-11-24 09:30:00 | 1616.65 | 2023-11-24 10:05:00 | 1619.95 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-11-28 09:45:00 | 1630.50 | 2023-11-28 10:10:00 | 1627.42 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-11-29 10:35:00 | 1651.45 | 2023-11-29 10:45:00 | 1654.49 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-12-05 11:15:00 | 1697.95 | 2023-12-05 11:25:00 | 1692.13 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-12-06 11:05:00 | 1706.00 | 2023-12-06 11:40:00 | 1702.35 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-07 10:00:00 | 1707.95 | 2023-12-07 10:10:00 | 1703.83 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-12-08 09:30:00 | 1699.65 | 2023-12-08 09:40:00 | 1703.52 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-12-12 10:15:00 | 1718.00 | 2023-12-12 10:30:00 | 1714.76 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-12-13 10:50:00 | 1668.90 | 2023-12-13 11:00:00 | 1672.75 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-12-14 09:45:00 | 1719.20 | 2023-12-14 09:55:00 | 1726.68 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-12-14 09:45:00 | 1719.20 | 2023-12-14 10:30:00 | 1719.70 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2023-12-15 10:45:00 | 1722.00 | 2023-12-15 11:45:00 | 1726.31 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-12-19 09:35:00 | 1710.80 | 2023-12-19 09:40:00 | 1705.38 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-12-19 09:35:00 | 1710.80 | 2023-12-19 10:30:00 | 1709.50 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2023-12-20 10:30:00 | 1729.90 | 2023-12-20 10:35:00 | 1734.53 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-12-20 10:30:00 | 1729.90 | 2023-12-20 11:05:00 | 1730.50 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2023-12-22 10:40:00 | 1687.65 | 2023-12-22 12:15:00 | 1683.31 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-12-26 10:40:00 | 1669.00 | 2023-12-26 11:00:00 | 1662.63 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-12-26 10:40:00 | 1669.00 | 2023-12-26 15:20:00 | 1647.20 | TARGET_HIT | 0.50 | 1.31% |
| BUY | retest1 | 2023-12-28 09:35:00 | 1689.90 | 2023-12-28 09:40:00 | 1695.87 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-12-28 09:35:00 | 1689.90 | 2023-12-28 09:55:00 | 1689.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-01 11:15:00 | 1677.70 | 2024-01-01 11:45:00 | 1674.07 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2024-01-01 11:15:00 | 1677.70 | 2024-01-01 11:50:00 | 1677.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-02 09:35:00 | 1684.50 | 2024-01-02 09:50:00 | 1680.68 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-01-09 09:30:00 | 1702.50 | 2024-01-09 09:50:00 | 1696.61 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-01-09 09:30:00 | 1702.50 | 2024-01-09 10:00:00 | 1702.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-11 09:35:00 | 1698.20 | 2024-01-11 10:00:00 | 1693.65 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-12 09:50:00 | 1655.70 | 2024-01-12 10:10:00 | 1659.72 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-15 10:30:00 | 1642.00 | 2024-01-15 10:50:00 | 1635.59 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-01-15 10:30:00 | 1642.00 | 2024-01-15 15:20:00 | 1633.85 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2024-01-17 10:35:00 | 1598.45 | 2024-01-17 11:45:00 | 1591.66 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-01-17 10:35:00 | 1598.45 | 2024-01-17 15:20:00 | 1587.50 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2024-01-19 09:35:00 | 1603.10 | 2024-01-19 09:45:00 | 1598.64 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-25 10:55:00 | 1617.75 | 2024-01-25 11:05:00 | 1620.96 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-01-29 11:15:00 | 1622.65 | 2024-01-29 12:40:00 | 1627.14 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-01-31 10:40:00 | 1609.50 | 2024-01-31 12:15:00 | 1617.15 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-01-31 10:40:00 | 1609.50 | 2024-01-31 15:20:00 | 1621.75 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2024-02-02 10:35:00 | 1651.70 | 2024-02-02 11:15:00 | 1658.60 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-02-02 10:35:00 | 1651.70 | 2024-02-02 12:25:00 | 1651.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-05 10:15:00 | 1641.80 | 2024-02-05 11:20:00 | 1634.83 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-02-05 10:15:00 | 1641.80 | 2024-02-05 15:20:00 | 1614.00 | TARGET_HIT | 0.50 | 1.69% |
| SELL | retest1 | 2024-02-08 10:15:00 | 1586.05 | 2024-02-08 10:30:00 | 1590.77 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-02-09 11:00:00 | 1567.95 | 2024-02-09 11:40:00 | 1572.64 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-02-12 10:25:00 | 1559.15 | 2024-02-12 10:30:00 | 1563.27 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-02-13 09:35:00 | 1570.20 | 2024-02-13 10:05:00 | 1578.48 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-02-13 09:35:00 | 1570.20 | 2024-02-13 11:10:00 | 1575.00 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-02-14 11:00:00 | 1569.75 | 2024-02-14 11:35:00 | 1565.85 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-15 10:20:00 | 1577.55 | 2024-02-15 10:30:00 | 1570.93 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-02-15 10:20:00 | 1577.55 | 2024-02-15 15:20:00 | 1570.10 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2024-02-19 09:35:00 | 1589.10 | 2024-02-19 09:45:00 | 1585.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-02-21 11:15:00 | 1603.35 | 2024-02-21 11:25:00 | 1599.23 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-02-21 11:15:00 | 1603.35 | 2024-02-21 15:20:00 | 1585.00 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2024-02-27 10:05:00 | 1601.15 | 2024-02-27 10:10:00 | 1605.43 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-02-29 10:45:00 | 1573.10 | 2024-02-29 11:05:00 | 1578.04 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-03-04 09:40:00 | 1604.00 | 2024-03-04 09:50:00 | 1607.70 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-03-05 10:20:00 | 1613.35 | 2024-03-05 10:50:00 | 1607.42 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-03-05 10:20:00 | 1613.35 | 2024-03-05 15:20:00 | 1551.40 | TARGET_HIT | 0.50 | 3.84% |
| BUY | retest1 | 2024-03-11 09:45:00 | 1601.60 | 2024-03-11 11:45:00 | 1608.19 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-03-11 09:45:00 | 1601.60 | 2024-03-11 15:10:00 | 1601.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-12 09:30:00 | 1601.20 | 2024-03-12 09:35:00 | 1610.13 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-03-12 09:30:00 | 1601.20 | 2024-03-12 09:50:00 | 1601.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-15 10:05:00 | 1586.35 | 2024-03-15 10:25:00 | 1580.57 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-03-19 09:45:00 | 1553.55 | 2024-03-19 09:50:00 | 1558.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-03-28 09:30:00 | 1614.60 | 2024-03-28 09:40:00 | 1621.09 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-03-28 09:30:00 | 1614.60 | 2024-03-28 15:05:00 | 1644.40 | TARGET_HIT | 0.50 | 1.85% |
| SELL | retest1 | 2024-04-02 09:45:00 | 1634.05 | 2024-04-02 10:05:00 | 1626.55 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-04-02 09:45:00 | 1634.05 | 2024-04-02 10:30:00 | 1634.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-04 10:50:00 | 1630.35 | 2024-04-04 10:55:00 | 1635.08 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-04-12 09:40:00 | 1695.85 | 2024-04-12 09:45:00 | 1699.80 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-04-16 09:40:00 | 1631.80 | 2024-04-16 10:05:00 | 1623.84 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-04-16 09:40:00 | 1631.80 | 2024-04-16 10:10:00 | 1631.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-23 09:30:00 | 1646.00 | 2024-04-23 09:40:00 | 1641.73 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-04-24 10:40:00 | 1636.70 | 2024-04-24 11:10:00 | 1642.03 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-04-24 10:40:00 | 1636.70 | 2024-04-24 12:50:00 | 1636.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-30 10:45:00 | 1611.95 | 2024-04-30 10:50:00 | 1617.45 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-04-30 10:45:00 | 1611.95 | 2024-04-30 11:00:00 | 1611.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-07 09:50:00 | 1605.70 | 2024-05-07 10:00:00 | 1598.86 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-07 09:50:00 | 1605.70 | 2024-05-07 10:55:00 | 1605.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-09 11:00:00 | 1587.05 | 2024-05-09 11:05:00 | 1581.55 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-05-09 11:00:00 | 1587.05 | 2024-05-09 15:20:00 | 1561.65 | TARGET_HIT | 0.50 | 1.60% |
