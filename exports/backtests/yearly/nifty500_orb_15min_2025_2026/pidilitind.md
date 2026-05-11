# Pidilite Industries Ltd. (PIDILITIND)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1472.00
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
| ENTRY1 | 94 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 12 |
| STOP_HIT | 82 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 82
- **Target hits / Stop hits / Partials:** 12 / 82 / 33
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 5.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 16 | 28.1% | 4 | 41 | 12 | 0.04% | 2.4% |
| BUY @ 2nd Alert (retest1) | 57 | 16 | 28.1% | 4 | 41 | 12 | 0.04% | 2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 70 | 29 | 41.4% | 8 | 41 | 21 | 0.05% | 3.3% |
| SELL @ 2nd Alert (retest1) | 70 | 29 | 41.4% | 8 | 41 | 21 | 0.05% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 127 | 45 | 35.4% | 12 | 82 | 33 | 0.05% | 5.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 10:15:00 | 1559.30 | 1552.98 | 0.00 | ORB-long ORB[1533.25,1556.00] vol=1.8x ATR=4.27 |
| Stop hit — per-position SL triggered | 2025-05-13 10:25:00 | 1555.03 | 1553.40 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 11:10:00 | 1536.75 | 1544.57 | 0.00 | ORB-short ORB[1542.70,1552.50] vol=1.5x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-19 11:40:00 | 1532.88 | 1542.37 | 0.00 | T1 1.5R @ 1532.88 |
| Stop hit — per-position SL triggered | 2025-05-19 12:20:00 | 1536.75 | 1540.88 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-21 11:15:00 | 1493.90 | 1500.85 | 0.00 | ORB-short ORB[1499.05,1510.45] vol=2.5x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 12:15:00 | 1487.99 | 1498.65 | 0.00 | T1 1.5R @ 1487.99 |
| Stop hit — per-position SL triggered | 2025-05-21 12:30:00 | 1493.90 | 1498.22 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:55:00 | 1507.80 | 1514.43 | 0.00 | ORB-short ORB[1509.95,1526.40] vol=1.6x ATR=3.83 |
| Stop hit — per-position SL triggered | 2025-05-27 10:35:00 | 1511.63 | 1512.68 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 11:15:00 | 1501.80 | 1507.61 | 0.00 | ORB-short ORB[1505.90,1515.50] vol=2.6x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-05-28 12:35:00 | 1504.48 | 1506.35 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 09:35:00 | 1509.90 | 1517.76 | 0.00 | ORB-short ORB[1515.00,1529.70] vol=2.5x ATR=3.48 |
| Stop hit — per-position SL triggered | 2025-06-09 09:45:00 | 1513.38 | 1515.96 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-20 09:35:00 | 1476.25 | 1481.41 | 0.00 | ORB-short ORB[1480.00,1490.80] vol=3.2x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-06-20 09:45:00 | 1479.41 | 1480.86 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 09:30:00 | 1496.40 | 1504.04 | 0.00 | ORB-short ORB[1497.65,1512.45] vol=1.7x ATR=4.95 |
| Stop hit — per-position SL triggered | 2025-06-24 09:40:00 | 1501.35 | 1503.53 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:05:00 | 1509.00 | 1506.57 | 0.00 | ORB-long ORB[1497.50,1507.00] vol=2.0x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:20:00 | 1513.83 | 1507.79 | 0.00 | T1 1.5R @ 1513.83 |
| Stop hit — per-position SL triggered | 2025-06-25 10:45:00 | 1509.00 | 1509.27 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 10:55:00 | 1526.00 | 1519.56 | 0.00 | ORB-long ORB[1507.80,1521.00] vol=2.6x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-07-02 11:00:00 | 1523.14 | 1519.86 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:50:00 | 1556.70 | 1552.85 | 0.00 | ORB-long ORB[1543.55,1556.00] vol=3.4x ATR=3.97 |
| Target hit | 2025-07-03 15:20:00 | 1557.50 | 1557.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 10:50:00 | 1542.85 | 1551.48 | 0.00 | ORB-short ORB[1551.25,1562.45] vol=1.7x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:20:00 | 1539.10 | 1550.06 | 0.00 | T1 1.5R @ 1539.10 |
| Stop hit — per-position SL triggered | 2025-07-04 13:10:00 | 1542.85 | 1544.16 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:30:00 | 1537.55 | 1540.08 | 0.00 | ORB-short ORB[1540.00,1547.75] vol=2.9x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:05:00 | 1533.05 | 1539.29 | 0.00 | T1 1.5R @ 1533.05 |
| Target hit | 2025-07-07 13:45:00 | 1526.25 | 1525.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2025-07-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:45:00 | 1529.25 | 1534.43 | 0.00 | ORB-short ORB[1532.85,1541.00] vol=2.8x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:05:00 | 1524.22 | 1533.14 | 0.00 | T1 1.5R @ 1524.22 |
| Stop hit — per-position SL triggered | 2025-07-08 12:50:00 | 1529.25 | 1529.67 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:40:00 | 1536.35 | 1531.37 | 0.00 | ORB-long ORB[1523.55,1532.65] vol=1.9x ATR=3.18 |
| Stop hit — per-position SL triggered | 2025-07-09 10:45:00 | 1533.17 | 1534.14 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 11:00:00 | 1532.55 | 1525.65 | 0.00 | ORB-long ORB[1523.00,1528.65] vol=1.8x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:30:00 | 1536.51 | 1528.11 | 0.00 | T1 1.5R @ 1536.51 |
| Stop hit — per-position SL triggered | 2025-07-10 12:05:00 | 1532.55 | 1531.18 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:45:00 | 1518.45 | 1527.09 | 0.00 | ORB-short ORB[1525.90,1534.95] vol=1.7x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:00:00 | 1514.68 | 1525.68 | 0.00 | T1 1.5R @ 1514.68 |
| Target hit | 2025-07-11 15:20:00 | 1498.95 | 1509.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-07-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 10:55:00 | 1480.55 | 1487.96 | 0.00 | ORB-short ORB[1482.65,1500.35] vol=2.1x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 11:30:00 | 1475.95 | 1486.06 | 0.00 | T1 1.5R @ 1475.95 |
| Stop hit — per-position SL triggered | 2025-07-14 14:40:00 | 1480.55 | 1480.12 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:45:00 | 1500.20 | 1499.64 | 0.00 | ORB-long ORB[1492.00,1499.00] vol=1.8x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 1497.97 | 1499.71 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:05:00 | 1484.55 | 1486.25 | 0.00 | ORB-short ORB[1491.60,1497.75] vol=9.7x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:20:00 | 1481.23 | 1486.05 | 0.00 | T1 1.5R @ 1481.23 |
| Stop hit — per-position SL triggered | 2025-07-18 10:50:00 | 1484.55 | 1485.61 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:55:00 | 1487.40 | 1486.86 | 0.00 | ORB-long ORB[1479.05,1486.00] vol=1.5x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-07-21 11:40:00 | 1485.05 | 1487.50 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:45:00 | 1477.50 | 1480.96 | 0.00 | ORB-short ORB[1482.35,1492.25] vol=2.1x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 11:35:00 | 1472.37 | 1479.48 | 0.00 | T1 1.5R @ 1472.37 |
| Target hit | 2025-07-22 15:20:00 | 1467.20 | 1471.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-07-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 11:00:00 | 1430.15 | 1437.94 | 0.00 | ORB-short ORB[1436.90,1449.00] vol=1.7x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 1432.80 | 1437.36 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 10:30:00 | 1440.70 | 1435.63 | 0.00 | ORB-long ORB[1425.50,1439.90] vol=1.6x ATR=3.62 |
| Stop hit — per-position SL triggered | 2025-08-01 11:05:00 | 1437.08 | 1436.16 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 11:05:00 | 1488.15 | 1484.14 | 0.00 | ORB-long ORB[1476.55,1488.00] vol=10.7x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-08-05 11:35:00 | 1485.30 | 1484.81 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 11:15:00 | 1538.15 | 1542.11 | 0.00 | ORB-short ORB[1540.00,1552.45] vol=2.2x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:30:00 | 1535.62 | 1541.33 | 0.00 | T1 1.5R @ 1535.62 |
| Target hit | 2025-08-13 15:20:00 | 1527.60 | 1535.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:45:00 | 1552.80 | 1548.68 | 0.00 | ORB-long ORB[1538.00,1551.50] vol=2.2x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-08-19 11:25:00 | 1550.24 | 1549.17 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:35:00 | 1559.10 | 1550.42 | 0.00 | ORB-long ORB[1544.95,1553.85] vol=3.2x ATR=2.70 |
| Stop hit — per-position SL triggered | 2025-08-25 10:45:00 | 1556.40 | 1551.10 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:30:00 | 1559.20 | 1548.18 | 0.00 | ORB-long ORB[1528.05,1542.30] vol=1.8x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 10:50:00 | 1565.05 | 1554.12 | 0.00 | T1 1.5R @ 1565.05 |
| Stop hit — per-position SL triggered | 2025-09-01 11:45:00 | 1559.20 | 1557.13 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:30:00 | 1571.45 | 1565.39 | 0.00 | ORB-long ORB[1558.55,1565.00] vol=2.7x ATR=2.97 |
| Stop hit — per-position SL triggered | 2025-09-02 09:35:00 | 1568.48 | 1566.44 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 10:05:00 | 1572.60 | 1560.57 | 0.00 | ORB-long ORB[1550.00,1563.50] vol=1.7x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-09-04 10:10:00 | 1569.21 | 1561.83 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 10:50:00 | 1538.75 | 1549.61 | 0.00 | ORB-short ORB[1552.50,1562.50] vol=1.7x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-09-10 10:55:00 | 1541.40 | 1549.37 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 09:55:00 | 1531.00 | 1533.89 | 0.00 | ORB-short ORB[1531.60,1539.10] vol=1.8x ATR=2.50 |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 1533.50 | 1532.15 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:50:00 | 1551.00 | 1545.65 | 0.00 | ORB-long ORB[1541.50,1549.00] vol=2.0x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-09-12 11:05:00 | 1548.68 | 1545.93 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 11:05:00 | 1543.50 | 1538.19 | 0.00 | ORB-long ORB[1534.00,1541.30] vol=1.8x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-09-16 12:55:00 | 1541.76 | 1539.04 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 11:00:00 | 1538.00 | 1535.71 | 0.00 | ORB-long ORB[1526.00,1536.95] vol=3.0x ATR=2.97 |
| Stop hit — per-position SL triggered | 2025-09-18 11:20:00 | 1535.03 | 1536.02 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 10:30:00 | 1527.00 | 1533.01 | 0.00 | ORB-short ORB[1529.05,1539.00] vol=1.6x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 11:45:00 | 1522.72 | 1530.34 | 0.00 | T1 1.5R @ 1522.72 |
| Target hit | 2025-09-19 15:15:00 | 1526.70 | 1525.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — BUY (started 2025-09-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 11:00:00 | 1534.75 | 1530.11 | 0.00 | ORB-long ORB[1525.40,1533.50] vol=2.2x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1532.54 | 1530.51 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 11:15:00 | 1498.50 | 1487.34 | 0.00 | ORB-long ORB[1475.00,1491.90] vol=2.2x ATR=3.42 |
| Stop hit — per-position SL triggered | 2025-09-24 11:35:00 | 1495.08 | 1487.90 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:30:00 | 1504.60 | 1508.58 | 0.00 | ORB-short ORB[1509.80,1518.50] vol=2.8x ATR=3.83 |
| Stop hit — per-position SL triggered | 2025-09-25 10:40:00 | 1508.43 | 1507.87 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:00:00 | 1456.30 | 1457.65 | 0.00 | ORB-short ORB[1458.10,1472.90] vol=2.3x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 12:00:00 | 1451.68 | 1456.76 | 0.00 | T1 1.5R @ 1451.68 |
| Stop hit — per-position SL triggered | 2025-10-01 12:05:00 | 1456.30 | 1456.78 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:55:00 | 1476.90 | 1468.91 | 0.00 | ORB-long ORB[1462.00,1470.70] vol=2.5x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:45:00 | 1482.74 | 1469.85 | 0.00 | T1 1.5R @ 1482.74 |
| Target hit | 2025-10-08 15:20:00 | 1492.60 | 1482.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-10-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 11:05:00 | 1505.00 | 1495.05 | 0.00 | ORB-long ORB[1486.60,1499.00] vol=5.1x ATR=3.55 |
| Stop hit — per-position SL triggered | 2025-10-09 11:35:00 | 1501.45 | 1496.27 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 10:45:00 | 1514.90 | 1508.77 | 0.00 | ORB-long ORB[1497.70,1509.80] vol=1.6x ATR=3.49 |
| Stop hit — per-position SL triggered | 2025-10-13 10:55:00 | 1511.41 | 1509.06 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:55:00 | 1514.70 | 1511.43 | 0.00 | ORB-long ORB[1500.00,1513.50] vol=1.7x ATR=3.21 |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 1511.49 | 1511.99 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:40:00 | 1533.70 | 1527.08 | 0.00 | ORB-long ORB[1515.30,1529.00] vol=3.6x ATR=4.06 |
| Stop hit — per-position SL triggered | 2025-10-17 09:45:00 | 1529.64 | 1527.58 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:20:00 | 1512.10 | 1512.48 | 0.00 | ORB-short ORB[1512.60,1519.80] vol=1.6x ATR=3.12 |
| Stop hit — per-position SL triggered | 2025-10-24 10:35:00 | 1515.22 | 1512.56 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 11:15:00 | 1517.60 | 1512.00 | 0.00 | ORB-long ORB[1505.00,1514.50] vol=1.6x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 11:40:00 | 1521.92 | 1513.17 | 0.00 | T1 1.5R @ 1521.92 |
| Stop hit — per-position SL triggered | 2025-10-27 11:50:00 | 1517.60 | 1513.63 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:30:00 | 1495.00 | 1502.25 | 0.00 | ORB-short ORB[1501.00,1512.00] vol=1.5x ATR=3.27 |
| Stop hit — per-position SL triggered | 2025-10-30 10:40:00 | 1498.27 | 1502.07 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:15:00 | 1452.20 | 1458.66 | 0.00 | ORB-short ORB[1459.40,1465.00] vol=3.0x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-11-04 10:40:00 | 1454.71 | 1457.75 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:40:00 | 1464.50 | 1462.12 | 0.00 | ORB-long ORB[1454.80,1463.20] vol=1.6x ATR=3.49 |
| Stop hit — per-position SL triggered | 2025-11-10 11:05:00 | 1461.01 | 1463.20 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:35:00 | 1482.90 | 1476.06 | 0.00 | ORB-long ORB[1469.70,1478.00] vol=1.7x ATR=3.26 |
| Stop hit — per-position SL triggered | 2025-11-12 09:50:00 | 1479.64 | 1477.08 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 09:30:00 | 1492.00 | 1485.55 | 0.00 | ORB-long ORB[1472.00,1489.00] vol=1.8x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-11-18 09:35:00 | 1488.46 | 1486.09 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:55:00 | 1482.70 | 1491.06 | 0.00 | ORB-short ORB[1489.20,1497.20] vol=1.6x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 11:15:00 | 1478.31 | 1489.84 | 0.00 | T1 1.5R @ 1478.31 |
| Stop hit — per-position SL triggered | 2025-11-19 12:15:00 | 1482.70 | 1486.71 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:45:00 | 1491.50 | 1485.45 | 0.00 | ORB-long ORB[1480.00,1489.70] vol=2.4x ATR=2.63 |
| Stop hit — per-position SL triggered | 2025-11-20 11:40:00 | 1488.87 | 1486.78 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 11:00:00 | 1466.00 | 1469.90 | 0.00 | ORB-short ORB[1466.80,1476.00] vol=1.8x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:40:00 | 1462.75 | 1468.07 | 0.00 | T1 1.5R @ 1462.75 |
| Target hit | 2025-11-24 14:45:00 | 1464.00 | 1463.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — BUY (started 2025-11-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 11:05:00 | 1469.70 | 1463.83 | 0.00 | ORB-long ORB[1459.30,1469.60] vol=2.2x ATR=3.44 |
| Stop hit — per-position SL triggered | 2025-11-25 11:10:00 | 1466.26 | 1463.95 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-11-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 09:40:00 | 1463.80 | 1468.49 | 0.00 | ORB-short ORB[1468.00,1478.00] vol=1.6x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-11-28 09:50:00 | 1466.73 | 1468.04 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:15:00 | 1465.40 | 1468.70 | 0.00 | ORB-short ORB[1468.60,1474.00] vol=2.1x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-12-01 11:20:00 | 1467.50 | 1468.65 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:35:00 | 1468.40 | 1463.54 | 0.00 | ORB-long ORB[1451.00,1462.90] vol=2.0x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-12-11 10:55:00 | 1465.24 | 1463.80 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 1454.50 | 1458.18 | 0.00 | ORB-short ORB[1455.80,1463.90] vol=1.9x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:20:00 | 1451.28 | 1456.72 | 0.00 | T1 1.5R @ 1451.28 |
| Stop hit — per-position SL triggered | 2025-12-24 11:30:00 | 1454.50 | 1456.29 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:50:00 | 1449.60 | 1454.39 | 0.00 | ORB-short ORB[1451.00,1461.00] vol=3.7x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:10:00 | 1445.97 | 1451.95 | 0.00 | T1 1.5R @ 1445.97 |
| Stop hit — per-position SL triggered | 2025-12-29 11:40:00 | 1449.60 | 1449.91 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:50:00 | 1441.70 | 1446.43 | 0.00 | ORB-short ORB[1447.00,1457.10] vol=3.9x ATR=2.84 |
| Stop hit — per-position SL triggered | 2025-12-30 11:05:00 | 1444.54 | 1445.37 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:55:00 | 1463.00 | 1459.45 | 0.00 | ORB-long ORB[1450.00,1461.00] vol=1.5x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 11:00:00 | 1466.68 | 1460.92 | 0.00 | T1 1.5R @ 1466.68 |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 1463.00 | 1462.05 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:35:00 | 1475.50 | 1472.14 | 0.00 | ORB-long ORB[1467.20,1473.90] vol=1.9x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 09:45:00 | 1479.44 | 1474.50 | 0.00 | T1 1.5R @ 1479.44 |
| Stop hit — per-position SL triggered | 2026-01-02 10:20:00 | 1475.50 | 1475.59 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:20:00 | 1490.60 | 1480.45 | 0.00 | ORB-long ORB[1474.60,1480.70] vol=2.7x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 10:25:00 | 1494.91 | 1483.45 | 0.00 | T1 1.5R @ 1494.91 |
| Stop hit — per-position SL triggered | 2026-01-05 10:30:00 | 1490.60 | 1484.73 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:15:00 | 1493.00 | 1498.83 | 0.00 | ORB-short ORB[1497.60,1508.00] vol=3.0x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-01-06 12:25:00 | 1495.84 | 1497.78 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:05:00 | 1493.60 | 1498.38 | 0.00 | ORB-short ORB[1502.00,1514.80] vol=2.1x ATR=2.35 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 1495.95 | 1497.44 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 11:15:00 | 1497.70 | 1494.92 | 0.00 | ORB-long ORB[1482.10,1496.40] vol=2.8x ATR=2.75 |
| Stop hit — per-position SL triggered | 2026-01-14 12:25:00 | 1494.95 | 1496.07 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 11:00:00 | 1488.10 | 1495.71 | 0.00 | ORB-short ORB[1491.00,1504.40] vol=1.8x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 11:05:00 | 1483.09 | 1495.17 | 0.00 | T1 1.5R @ 1483.09 |
| Stop hit — per-position SL triggered | 2026-01-16 11:20:00 | 1488.10 | 1494.43 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:00:00 | 1432.50 | 1435.63 | 0.00 | ORB-short ORB[1436.50,1443.50] vol=1.5x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 10:15:00 | 1427.57 | 1434.26 | 0.00 | T1 1.5R @ 1427.57 |
| Target hit | 2026-01-28 11:25:00 | 1431.30 | 1430.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — SELL (started 2026-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 11:10:00 | 1419.40 | 1432.58 | 0.00 | ORB-short ORB[1442.80,1458.30] vol=2.7x ATR=4.16 |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 1423.56 | 1431.80 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 09:30:00 | 1481.10 | 1476.25 | 0.00 | ORB-long ORB[1461.60,1476.10] vol=3.5x ATR=4.20 |
| Stop hit — per-position SL triggered | 2026-02-05 09:35:00 | 1476.90 | 1476.87 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 1477.90 | 1482.30 | 0.00 | ORB-short ORB[1481.60,1494.00] vol=2.4x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:10:00 | 1474.27 | 1481.61 | 0.00 | T1 1.5R @ 1474.27 |
| Target hit | 2026-02-10 14:25:00 | 1475.90 | 1475.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 75 — BUY (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1494.60 | 1486.91 | 0.00 | ORB-long ORB[1475.00,1491.60] vol=1.9x ATR=3.32 |
| Stop hit — per-position SL triggered | 2026-02-11 11:05:00 | 1491.28 | 1487.16 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-02-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:50:00 | 1474.30 | 1482.56 | 0.00 | ORB-short ORB[1480.30,1489.60] vol=1.7x ATR=3.32 |
| Stop hit — per-position SL triggered | 2026-02-13 11:25:00 | 1477.62 | 1481.50 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 1491.40 | 1486.83 | 0.00 | ORB-long ORB[1477.40,1490.00] vol=3.6x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-02-17 11:10:00 | 1488.23 | 1487.89 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 1480.60 | 1477.45 | 0.00 | ORB-long ORB[1470.00,1479.30] vol=1.6x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:40:00 | 1484.50 | 1479.03 | 0.00 | T1 1.5R @ 1484.50 |
| Stop hit — per-position SL triggered | 2026-02-24 12:40:00 | 1480.60 | 1480.16 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 1485.00 | 1487.02 | 0.00 | ORB-short ORB[1488.10,1495.50] vol=4.6x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 1487.34 | 1487.00 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 1498.90 | 1497.09 | 0.00 | ORB-long ORB[1487.10,1497.80] vol=8.0x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:05:00 | 1503.42 | 1499.01 | 0.00 | T1 1.5R @ 1503.42 |
| Stop hit — per-position SL triggered | 2026-02-26 11:40:00 | 1498.90 | 1501.08 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 1505.90 | 1506.76 | 0.00 | ORB-short ORB[1507.10,1518.90] vol=1.7x ATR=3.08 |
| Stop hit — per-position SL triggered | 2026-02-27 11:05:00 | 1508.98 | 1506.69 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 1426.20 | 1432.15 | 0.00 | ORB-short ORB[1432.60,1446.90] vol=3.4x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 1429.22 | 1431.52 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 1425.00 | 1430.49 | 0.00 | ORB-short ORB[1427.70,1438.60] vol=1.8x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:15:00 | 1420.88 | 1430.02 | 0.00 | T1 1.5R @ 1420.88 |
| Stop hit — per-position SL triggered | 2026-03-11 11:20:00 | 1425.00 | 1428.97 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:00:00 | 1362.60 | 1360.90 | 0.00 | ORB-long ORB[1345.60,1361.30] vol=1.7x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:40:00 | 1369.20 | 1361.78 | 0.00 | T1 1.5R @ 1369.20 |
| Target hit | 2026-03-18 15:20:00 | 1382.80 | 1377.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — SELL (started 2026-03-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 11:10:00 | 1282.00 | 1297.17 | 0.00 | ORB-short ORB[1296.10,1311.00] vol=1.7x ATR=3.99 |
| Stop hit — per-position SL triggered | 2026-03-30 12:00:00 | 1285.99 | 1292.62 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 1345.20 | 1350.00 | 0.00 | ORB-short ORB[1346.90,1365.00] vol=9.0x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:25:00 | 1339.76 | 1349.15 | 0.00 | T1 1.5R @ 1339.76 |
| Stop hit — per-position SL triggered | 2026-04-10 12:00:00 | 1345.20 | 1347.18 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-04-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:50:00 | 1341.30 | 1334.95 | 0.00 | ORB-long ORB[1314.30,1334.20] vol=5.5x ATR=5.13 |
| Stop hit — per-position SL triggered | 2026-04-13 11:05:00 | 1336.17 | 1336.27 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-04-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:55:00 | 1355.80 | 1343.21 | 0.00 | ORB-long ORB[1326.00,1343.90] vol=2.0x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:00:00 | 1361.61 | 1345.76 | 0.00 | T1 1.5R @ 1361.61 |
| Target hit | 2026-04-17 15:20:00 | 1393.10 | 1379.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1409.20 | 1395.12 | 0.00 | ORB-long ORB[1380.00,1394.60] vol=2.2x ATR=4.19 |
| Stop hit — per-position SL triggered | 2026-04-21 11:30:00 | 1405.01 | 1404.58 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:50:00 | 1393.60 | 1398.36 | 0.00 | ORB-short ORB[1395.00,1408.90] vol=2.4x ATR=3.62 |
| Stop hit — per-position SL triggered | 2026-04-24 10:45:00 | 1397.22 | 1397.98 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-04-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:05:00 | 1359.40 | 1368.30 | 0.00 | ORB-short ORB[1368.80,1381.10] vol=1.7x ATR=3.26 |
| Stop hit — per-position SL triggered | 2026-04-30 11:40:00 | 1362.66 | 1365.76 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 1373.80 | 1377.33 | 0.00 | ORB-short ORB[1376.70,1388.60] vol=2.8x ATR=4.23 |
| Stop hit — per-position SL triggered | 2026-05-04 10:35:00 | 1378.03 | 1377.54 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 1350.50 | 1357.84 | 0.00 | ORB-short ORB[1357.80,1369.00] vol=2.9x ATR=2.99 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 1353.49 | 1355.76 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 1476.90 | 1497.84 | 0.00 | ORB-short ORB[1492.70,1515.00] vol=2.4x ATR=8.29 |
| Stop hit — per-position SL triggered | 2026-05-08 10:15:00 | 1485.19 | 1493.15 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 10:15:00 | 1559.30 | 2025-05-13 10:25:00 | 1555.03 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-19 11:10:00 | 1536.75 | 2025-05-19 11:40:00 | 1532.88 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-05-19 11:10:00 | 1536.75 | 2025-05-19 12:20:00 | 1536.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-21 11:15:00 | 1493.90 | 2025-05-21 12:15:00 | 1487.99 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-05-21 11:15:00 | 1493.90 | 2025-05-21 12:30:00 | 1493.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-27 09:55:00 | 1507.80 | 2025-05-27 10:35:00 | 1511.63 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-28 11:15:00 | 1501.80 | 2025-05-28 12:35:00 | 1504.48 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-06-09 09:35:00 | 1509.90 | 2025-06-09 09:45:00 | 1513.38 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-20 09:35:00 | 1476.25 | 2025-06-20 09:45:00 | 1479.41 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-24 09:30:00 | 1496.40 | 2025-06-24 09:40:00 | 1501.35 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-25 10:05:00 | 1509.00 | 2025-06-25 10:20:00 | 1513.83 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-06-25 10:05:00 | 1509.00 | 2025-06-25 10:45:00 | 1509.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-02 10:55:00 | 1526.00 | 2025-07-02 11:00:00 | 1523.14 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-03 09:50:00 | 1556.70 | 2025-07-03 15:20:00 | 1557.50 | TARGET_HIT | 1.00 | 0.05% |
| SELL | retest1 | 2025-07-04 10:50:00 | 1542.85 | 2025-07-04 11:20:00 | 1539.10 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-07-04 10:50:00 | 1542.85 | 2025-07-04 13:10:00 | 1542.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-07 10:30:00 | 1537.55 | 2025-07-07 11:05:00 | 1533.05 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-07-07 10:30:00 | 1537.55 | 2025-07-07 13:45:00 | 1526.25 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2025-07-08 10:45:00 | 1529.25 | 2025-07-08 11:05:00 | 1524.22 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-08 10:45:00 | 1529.25 | 2025-07-08 12:50:00 | 1529.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-09 09:40:00 | 1536.35 | 2025-07-09 10:45:00 | 1533.17 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-10 11:00:00 | 1532.55 | 2025-07-10 11:30:00 | 1536.51 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-07-10 11:00:00 | 1532.55 | 2025-07-10 12:05:00 | 1532.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 10:45:00 | 1518.45 | 2025-07-11 11:00:00 | 1514.68 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-11 10:45:00 | 1518.45 | 2025-07-11 15:20:00 | 1498.95 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2025-07-14 10:55:00 | 1480.55 | 2025-07-14 11:30:00 | 1475.95 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-14 10:55:00 | 1480.55 | 2025-07-14 14:40:00 | 1480.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-17 10:45:00 | 1500.20 | 2025-07-17 11:15:00 | 1497.97 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-07-18 10:05:00 | 1484.55 | 2025-07-18 10:20:00 | 1481.23 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-07-18 10:05:00 | 1484.55 | 2025-07-18 10:50:00 | 1484.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 10:55:00 | 1487.40 | 2025-07-21 11:40:00 | 1485.05 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-22 10:45:00 | 1477.50 | 2025-07-22 11:35:00 | 1472.37 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-22 10:45:00 | 1477.50 | 2025-07-22 15:20:00 | 1467.20 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2025-07-29 11:00:00 | 1430.15 | 2025-07-29 11:15:00 | 1432.80 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-08-01 10:30:00 | 1440.70 | 2025-08-01 11:05:00 | 1437.08 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-05 11:05:00 | 1488.15 | 2025-08-05 11:35:00 | 1485.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-13 11:15:00 | 1538.15 | 2025-08-13 11:30:00 | 1535.62 | PARTIAL | 0.50 | 0.16% |
| SELL | retest1 | 2025-08-13 11:15:00 | 1538.15 | 2025-08-13 15:20:00 | 1527.60 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2025-08-19 10:45:00 | 1552.80 | 2025-08-19 11:25:00 | 1550.24 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-08-25 10:35:00 | 1559.10 | 2025-08-25 10:45:00 | 1556.40 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-01 10:30:00 | 1559.20 | 2025-09-01 10:50:00 | 1565.05 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-09-01 10:30:00 | 1559.20 | 2025-09-01 11:45:00 | 1559.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-02 09:30:00 | 1571.45 | 2025-09-02 09:35:00 | 1568.48 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-04 10:05:00 | 1572.60 | 2025-09-04 10:10:00 | 1569.21 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-10 10:50:00 | 1538.75 | 2025-09-10 10:55:00 | 1541.40 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-11 09:55:00 | 1531.00 | 2025-09-11 10:15:00 | 1533.50 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-12 10:50:00 | 1551.00 | 2025-09-12 11:05:00 | 1548.68 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-09-16 11:05:00 | 1543.50 | 2025-09-16 12:55:00 | 1541.76 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2025-09-18 11:00:00 | 1538.00 | 2025-09-18 11:20:00 | 1535.03 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-09-19 10:30:00 | 1527.00 | 2025-09-19 11:45:00 | 1522.72 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-09-19 10:30:00 | 1527.00 | 2025-09-19 15:15:00 | 1526.70 | TARGET_HIT | 0.50 | 0.02% |
| BUY | retest1 | 2025-09-22 11:00:00 | 1534.75 | 2025-09-22 11:15:00 | 1532.54 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-09-24 11:15:00 | 1498.50 | 2025-09-24 11:35:00 | 1495.08 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-25 10:30:00 | 1504.60 | 2025-09-25 10:40:00 | 1508.43 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-01 11:00:00 | 1456.30 | 2025-10-01 12:00:00 | 1451.68 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-10-01 11:00:00 | 1456.30 | 2025-10-01 12:05:00 | 1456.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-08 10:55:00 | 1476.90 | 2025-10-08 11:45:00 | 1482.74 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-08 10:55:00 | 1476.90 | 2025-10-08 15:20:00 | 1492.60 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2025-10-09 11:05:00 | 1505.00 | 2025-10-09 11:35:00 | 1501.45 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-13 10:45:00 | 1514.90 | 2025-10-13 10:55:00 | 1511.41 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-16 09:55:00 | 1514.70 | 2025-10-16 10:15:00 | 1511.49 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-17 09:40:00 | 1533.70 | 2025-10-17 09:45:00 | 1529.64 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-24 10:20:00 | 1512.10 | 2025-10-24 10:35:00 | 1515.22 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-27 11:15:00 | 1517.60 | 2025-10-27 11:40:00 | 1521.92 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-10-27 11:15:00 | 1517.60 | 2025-10-27 11:50:00 | 1517.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 10:30:00 | 1495.00 | 2025-10-30 10:40:00 | 1498.27 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-04 10:15:00 | 1452.20 | 2025-11-04 10:40:00 | 1454.71 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-10 09:40:00 | 1464.50 | 2025-11-10 11:05:00 | 1461.01 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-12 09:35:00 | 1482.90 | 2025-11-12 09:50:00 | 1479.64 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-18 09:30:00 | 1492.00 | 2025-11-18 09:35:00 | 1488.46 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-19 10:55:00 | 1482.70 | 2025-11-19 11:15:00 | 1478.31 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-11-19 10:55:00 | 1482.70 | 2025-11-19 12:15:00 | 1482.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-20 10:45:00 | 1491.50 | 2025-11-20 11:40:00 | 1488.87 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-11-24 11:00:00 | 1466.00 | 2025-11-24 11:40:00 | 1462.75 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-11-24 11:00:00 | 1466.00 | 2025-11-24 14:45:00 | 1464.00 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2025-11-25 11:05:00 | 1469.70 | 2025-11-25 11:10:00 | 1466.26 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-28 09:40:00 | 1463.80 | 2025-11-28 09:50:00 | 1466.73 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-01 11:15:00 | 1465.40 | 2025-12-01 11:20:00 | 1467.50 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-12-11 10:35:00 | 1468.40 | 2025-12-11 10:55:00 | 1465.24 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-24 10:55:00 | 1454.50 | 2025-12-24 11:20:00 | 1451.28 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-12-24 10:55:00 | 1454.50 | 2025-12-24 11:30:00 | 1454.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 10:50:00 | 1449.60 | 2025-12-29 11:10:00 | 1445.97 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-29 10:50:00 | 1449.60 | 2025-12-29 11:40:00 | 1449.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 10:50:00 | 1441.70 | 2025-12-30 11:05:00 | 1444.54 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-31 10:55:00 | 1463.00 | 2025-12-31 11:00:00 | 1466.68 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-12-31 10:55:00 | 1463.00 | 2025-12-31 11:15:00 | 1463.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 09:35:00 | 1475.50 | 2026-01-02 09:45:00 | 1479.44 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-01-02 09:35:00 | 1475.50 | 2026-01-02 10:20:00 | 1475.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-05 10:20:00 | 1490.60 | 2026-01-05 10:25:00 | 1494.91 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-01-05 10:20:00 | 1490.60 | 2026-01-05 10:30:00 | 1490.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-06 11:15:00 | 1493.00 | 2026-01-06 12:25:00 | 1495.84 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-01-08 11:05:00 | 1493.60 | 2026-01-08 11:35:00 | 1495.95 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-01-14 11:15:00 | 1497.70 | 2026-01-14 12:25:00 | 1494.95 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-01-16 11:00:00 | 1488.10 | 2026-01-16 11:05:00 | 1483.09 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-01-16 11:00:00 | 1488.10 | 2026-01-16 11:20:00 | 1488.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-28 10:00:00 | 1432.50 | 2026-01-28 10:15:00 | 1427.57 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-01-28 10:00:00 | 1432.50 | 2026-01-28 11:25:00 | 1431.30 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2026-01-29 11:10:00 | 1419.40 | 2026-01-29 11:15:00 | 1423.56 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-05 09:30:00 | 1481.10 | 2026-02-05 09:35:00 | 1476.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-10 10:55:00 | 1477.90 | 2026-02-10 11:10:00 | 1474.27 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-02-10 10:55:00 | 1477.90 | 2026-02-10 14:25:00 | 1475.90 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2026-02-11 11:00:00 | 1494.60 | 2026-02-11 11:05:00 | 1491.28 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-13 10:50:00 | 1474.30 | 2026-02-13 11:25:00 | 1477.62 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-17 10:45:00 | 1491.40 | 2026-02-17 11:10:00 | 1488.23 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-24 11:10:00 | 1480.60 | 2026-02-24 11:40:00 | 1484.50 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-02-24 11:10:00 | 1480.60 | 2026-02-24 12:40:00 | 1480.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 11:00:00 | 1485.00 | 2026-02-25 11:05:00 | 1487.34 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-02-26 10:45:00 | 1498.90 | 2026-02-26 11:05:00 | 1503.42 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-26 10:45:00 | 1498.90 | 2026-02-26 11:40:00 | 1498.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:40:00 | 1505.90 | 2026-02-27 11:05:00 | 1508.98 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-03-05 10:50:00 | 1426.20 | 2026-03-05 11:15:00 | 1429.22 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-11 11:10:00 | 1425.00 | 2026-03-11 11:15:00 | 1420.88 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-03-11 11:10:00 | 1425.00 | 2026-03-11 11:20:00 | 1425.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 10:00:00 | 1362.60 | 2026-03-18 10:40:00 | 1369.20 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-18 10:00:00 | 1362.60 | 2026-03-18 15:20:00 | 1382.80 | TARGET_HIT | 0.50 | 1.48% |
| SELL | retest1 | 2026-03-30 11:10:00 | 1282.00 | 2026-03-30 12:00:00 | 1285.99 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-10 11:00:00 | 1345.20 | 2026-04-10 11:25:00 | 1339.76 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-04-10 11:00:00 | 1345.20 | 2026-04-10 12:00:00 | 1345.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 10:50:00 | 1341.30 | 2026-04-13 11:05:00 | 1336.17 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-17 09:55:00 | 1355.80 | 2026-04-17 10:00:00 | 1361.61 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-04-17 09:55:00 | 1355.80 | 2026-04-17 15:20:00 | 1393.10 | TARGET_HIT | 0.50 | 2.75% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1409.20 | 2026-04-21 11:30:00 | 1405.01 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-24 09:50:00 | 1393.60 | 2026-04-24 10:45:00 | 1397.22 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-30 11:05:00 | 1359.40 | 2026-04-30 11:40:00 | 1362.66 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-04 10:30:00 | 1373.80 | 2026-05-04 10:35:00 | 1378.03 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-05 11:00:00 | 1350.50 | 2026-05-05 11:15:00 | 1353.49 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-08 09:35:00 | 1476.90 | 2026-05-08 10:15:00 | 1485.19 | STOP_HIT | 1.00 | -0.56% |
