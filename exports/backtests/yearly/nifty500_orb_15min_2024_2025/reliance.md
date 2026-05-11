# Reliance Industries Ltd. (RELIANCE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1436.00
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
| ENTRY1 | 74 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 11 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 106 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 63
- **Target hits / Stop hits / Partials:** 11 / 63 / 32
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 10.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 25 | 41.7% | 7 | 35 | 18 | 0.12% | 7.2% |
| BUY @ 2nd Alert (retest1) | 60 | 25 | 41.7% | 7 | 35 | 18 | 0.12% | 7.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 46 | 18 | 39.1% | 4 | 28 | 14 | 0.08% | 3.6% |
| SELL @ 2nd Alert (retest1) | 46 | 18 | 39.1% | 4 | 28 | 14 | 0.08% | 3.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 106 | 43 | 40.6% | 11 | 63 | 32 | 0.10% | 10.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:20:00 | 1412.80 | 1418.98 | 0.00 | ORB-short ORB[1415.63,1423.48] vol=1.6x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:45:00 | 1408.28 | 1417.73 | 0.00 | T1 1.5R @ 1408.28 |
| Stop hit — per-position SL triggered | 2024-05-16 12:05:00 | 1412.80 | 1415.88 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:35:00 | 1470.60 | 1465.59 | 0.00 | ORB-long ORB[1455.00,1468.50] vol=1.8x ATR=3.21 |
| Stop hit — per-position SL triggered | 2024-05-23 10:00:00 | 1467.39 | 1466.98 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 10:25:00 | 1477.48 | 1472.50 | 0.00 | ORB-long ORB[1468.00,1474.95] vol=1.6x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-05-28 10:30:00 | 1474.73 | 1472.68 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:00:00 | 1426.45 | 1432.04 | 0.00 | ORB-short ORB[1430.53,1440.40] vol=1.6x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-05-30 10:35:00 | 1428.96 | 1430.86 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:30:00 | 1462.20 | 1467.60 | 0.00 | ORB-short ORB[1464.10,1473.68] vol=1.7x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 09:35:00 | 1456.91 | 1466.24 | 0.00 | T1 1.5R @ 1456.91 |
| Stop hit — per-position SL triggered | 2024-06-11 09:40:00 | 1462.20 | 1465.24 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:55:00 | 1468.68 | 1459.56 | 0.00 | ORB-long ORB[1450.10,1461.98] vol=1.9x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:35:00 | 1474.91 | 1463.42 | 0.00 | T1 1.5R @ 1474.91 |
| Target hit | 2024-06-20 15:20:00 | 1474.78 | 1472.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 1459.08 | 1464.05 | 0.00 | ORB-short ORB[1463.50,1474.70] vol=1.9x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 12:35:00 | 1454.79 | 1461.16 | 0.00 | T1 1.5R @ 1454.79 |
| Target hit | 2024-06-21 15:00:00 | 1455.73 | 1454.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2024-06-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:05:00 | 1476.48 | 1464.35 | 0.00 | ORB-long ORB[1445.13,1459.38] vol=1.5x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:45:00 | 1481.00 | 1467.38 | 0.00 | T1 1.5R @ 1481.00 |
| Target hit | 2024-06-26 15:20:00 | 1516.05 | 1491.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:45:00 | 1532.50 | 1519.14 | 0.00 | ORB-long ORB[1506.00,1517.00] vol=1.6x ATR=5.32 |
| Stop hit — per-position SL triggered | 2024-06-27 09:50:00 | 1527.18 | 1520.80 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:35:00 | 1549.88 | 1561.40 | 0.00 | ORB-short ORB[1563.50,1575.00] vol=1.6x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:40:00 | 1544.42 | 1557.90 | 0.00 | T1 1.5R @ 1544.42 |
| Stop hit — per-position SL triggered | 2024-07-03 13:15:00 | 1549.88 | 1554.26 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:55:00 | 1566.68 | 1560.83 | 0.00 | ORB-long ORB[1555.88,1563.33] vol=1.7x ATR=3.10 |
| Stop hit — per-position SL triggered | 2024-07-04 10:05:00 | 1563.58 | 1561.29 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:10:00 | 1573.00 | 1559.59 | 0.00 | ORB-long ORB[1548.00,1559.50] vol=1.9x ATR=4.20 |
| Stop hit — per-position SL triggered | 2024-07-05 10:20:00 | 1568.80 | 1560.91 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 11:15:00 | 1599.65 | 1589.85 | 0.00 | ORB-long ORB[1585.78,1595.48] vol=2.3x ATR=3.60 |
| Stop hit — per-position SL triggered | 2024-07-08 11:40:00 | 1596.05 | 1591.58 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 1577.10 | 1583.91 | 0.00 | ORB-short ORB[1585.00,1597.50] vol=2.1x ATR=3.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 1572.44 | 1582.29 | 0.00 | T1 1.5R @ 1572.44 |
| Stop hit — per-position SL triggered | 2024-07-10 11:25:00 | 1577.10 | 1575.89 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:00:00 | 1578.00 | 1579.06 | 0.00 | ORB-short ORB[1581.30,1591.00] vol=1.5x ATR=3.58 |
| Stop hit — per-position SL triggered | 2024-07-11 11:10:00 | 1581.58 | 1579.14 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:50:00 | 1590.75 | 1581.85 | 0.00 | ORB-long ORB[1578.35,1589.33] vol=3.4x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:55:00 | 1596.81 | 1584.06 | 0.00 | T1 1.5R @ 1596.81 |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 1590.75 | 1587.42 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 10:55:00 | 1586.65 | 1596.58 | 0.00 | ORB-short ORB[1598.53,1605.85] vol=1.9x ATR=3.55 |
| Stop hit — per-position SL triggered | 2024-07-15 11:10:00 | 1590.20 | 1595.70 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 09:45:00 | 1589.00 | 1594.07 | 0.00 | ORB-short ORB[1593.00,1600.00] vol=2.1x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:05:00 | 1585.00 | 1591.64 | 0.00 | T1 1.5R @ 1585.00 |
| Stop hit — per-position SL triggered | 2024-07-16 10:15:00 | 1589.00 | 1591.23 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 11:10:00 | 1559.80 | 1570.82 | 0.00 | ORB-short ORB[1562.70,1579.50] vol=1.6x ATR=3.40 |
| Stop hit — per-position SL triggered | 2024-07-18 11:30:00 | 1563.20 | 1569.66 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 10:45:00 | 1482.45 | 1492.49 | 0.00 | ORB-short ORB[1494.25,1505.75] vol=1.7x ATR=2.98 |
| Stop hit — per-position SL triggered | 2024-07-23 10:55:00 | 1485.43 | 1491.01 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:35:00 | 1496.00 | 1482.65 | 0.00 | ORB-long ORB[1476.40,1486.38] vol=2.0x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 10:40:00 | 1502.85 | 1485.25 | 0.00 | T1 1.5R @ 1502.85 |
| Stop hit — per-position SL triggered | 2024-07-24 12:05:00 | 1496.00 | 1492.69 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 11:00:00 | 1518.80 | 1516.66 | 0.00 | ORB-long ORB[1511.78,1517.95] vol=1.6x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 11:35:00 | 1522.89 | 1517.82 | 0.00 | T1 1.5R @ 1522.89 |
| Stop hit — per-position SL triggered | 2024-07-29 12:15:00 | 1518.80 | 1518.52 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:00:00 | 1520.28 | 1516.10 | 0.00 | ORB-long ORB[1512.50,1519.98] vol=1.6x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 12:25:00 | 1523.65 | 1517.58 | 0.00 | T1 1.5R @ 1523.65 |
| Stop hit — per-position SL triggered | 2024-07-30 12:50:00 | 1520.28 | 1517.99 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 11:10:00 | 1515.18 | 1511.73 | 0.00 | ORB-long ORB[1504.30,1513.93] vol=2.0x ATR=1.94 |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 1513.24 | 1511.76 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 11:00:00 | 1470.48 | 1465.27 | 0.00 | ORB-long ORB[1456.00,1469.45] vol=1.6x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 14:05:00 | 1475.04 | 1468.13 | 0.00 | T1 1.5R @ 1475.04 |
| Target hit | 2024-08-09 15:20:00 | 1474.35 | 1469.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:15:00 | 1472.00 | 1465.82 | 0.00 | ORB-long ORB[1462.50,1471.08] vol=2.2x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-08-12 11:25:00 | 1469.53 | 1466.37 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:35:00 | 1479.50 | 1474.16 | 0.00 | ORB-long ORB[1467.88,1477.63] vol=1.9x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-08-16 09:40:00 | 1476.25 | 1474.48 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:40:00 | 1494.00 | 1489.07 | 0.00 | ORB-long ORB[1480.50,1491.90] vol=1.9x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-08-19 09:50:00 | 1490.80 | 1489.52 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:40:00 | 1494.00 | 1501.75 | 0.00 | ORB-short ORB[1501.75,1507.48] vol=1.6x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-08-28 10:45:00 | 1496.27 | 1501.37 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:35:00 | 1479.15 | 1474.76 | 0.00 | ORB-long ORB[1468.75,1475.98] vol=2.0x ATR=2.11 |
| Stop hit — per-position SL triggered | 2024-09-18 11:20:00 | 1477.04 | 1477.49 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 11:05:00 | 1457.85 | 1466.73 | 0.00 | ORB-short ORB[1466.25,1472.40] vol=1.9x ATR=2.22 |
| Stop hit — per-position SL triggered | 2024-09-19 12:00:00 | 1460.07 | 1464.38 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:35:00 | 1482.40 | 1478.24 | 0.00 | ORB-long ORB[1471.38,1479.98] vol=3.0x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 11:10:00 | 1486.97 | 1480.50 | 0.00 | T1 1.5R @ 1486.97 |
| Stop hit — per-position SL triggered | 2024-09-20 13:30:00 | 1482.40 | 1484.77 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 11:05:00 | 1499.58 | 1495.33 | 0.00 | ORB-long ORB[1488.05,1494.80] vol=1.8x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:45:00 | 1502.28 | 1496.15 | 0.00 | T1 1.5R @ 1502.28 |
| Stop hit — per-position SL triggered | 2024-09-26 12:10:00 | 1499.58 | 1496.51 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 11:15:00 | 1435.80 | 1443.02 | 0.00 | ORB-short ORB[1436.58,1449.40] vol=2.2x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 12:00:00 | 1432.60 | 1441.84 | 0.00 | T1 1.5R @ 1432.60 |
| Target hit | 2024-10-03 15:20:00 | 1407.75 | 1425.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 1379.08 | 1386.53 | 0.00 | ORB-short ORB[1386.00,1396.50] vol=2.9x ATR=3.58 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 1382.66 | 1386.28 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:00:00 | 1370.05 | 1375.82 | 0.00 | ORB-short ORB[1371.50,1380.08] vol=1.6x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-10-14 11:10:00 | 1372.19 | 1375.61 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 11:00:00 | 1347.13 | 1343.11 | 0.00 | ORB-long ORB[1336.15,1346.18] vol=2.5x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 11:45:00 | 1351.91 | 1344.26 | 0.00 | T1 1.5R @ 1351.91 |
| Stop hit — per-position SL triggered | 2024-10-23 12:15:00 | 1347.13 | 1344.69 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 10:30:00 | 1332.35 | 1336.73 | 0.00 | ORB-short ORB[1335.28,1343.00] vol=1.7x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 10:50:00 | 1327.80 | 1335.66 | 0.00 | T1 1.5R @ 1327.80 |
| Stop hit — per-position SL triggered | 2024-10-24 12:30:00 | 1332.35 | 1332.81 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:15:00 | 1333.85 | 1335.87 | 0.00 | ORB-short ORB[1334.05,1343.00] vol=1.9x ATR=1.93 |
| Stop hit — per-position SL triggered | 2024-10-31 11:40:00 | 1335.78 | 1335.77 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 10:55:00 | 1286.20 | 1278.95 | 0.00 | ORB-long ORB[1275.00,1282.70] vol=2.2x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-11-12 11:15:00 | 1282.75 | 1279.69 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-11-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:55:00 | 1296.00 | 1289.81 | 0.00 | ORB-long ORB[1287.20,1294.95] vol=2.0x ATR=3.16 |
| Stop hit — per-position SL triggered | 2024-11-26 11:05:00 | 1292.84 | 1290.08 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-11-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:40:00 | 1285.85 | 1290.46 | 0.00 | ORB-short ORB[1286.55,1293.20] vol=2.3x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:30:00 | 1281.81 | 1289.20 | 0.00 | T1 1.5R @ 1281.81 |
| Stop hit — per-position SL triggered | 2024-11-28 11:45:00 | 1285.85 | 1288.90 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:50:00 | 1295.90 | 1283.90 | 0.00 | ORB-long ORB[1275.25,1281.95] vol=1.6x ATR=3.76 |
| Stop hit — per-position SL triggered | 2024-11-29 11:50:00 | 1292.14 | 1287.75 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:55:00 | 1317.90 | 1312.76 | 0.00 | ORB-long ORB[1309.00,1317.00] vol=1.7x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:35:00 | 1322.05 | 1314.74 | 0.00 | T1 1.5R @ 1322.05 |
| Target hit | 2024-12-03 15:20:00 | 1323.60 | 1320.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2024-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:55:00 | 1313.40 | 1314.92 | 0.00 | ORB-short ORB[1315.45,1328.40] vol=13.3x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:55:00 | 1309.59 | 1314.27 | 0.00 | T1 1.5R @ 1309.59 |
| Target hit | 2024-12-04 12:25:00 | 1312.65 | 1312.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — SELL (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 11:15:00 | 1283.30 | 1284.99 | 0.00 | ORB-short ORB[1285.00,1290.00] vol=1.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-12-11 11:25:00 | 1285.31 | 1284.96 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:00:00 | 1268.55 | 1272.43 | 0.00 | ORB-short ORB[1270.00,1277.50] vol=1.5x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 12:00:00 | 1265.83 | 1271.20 | 0.00 | T1 1.5R @ 1265.83 |
| Stop hit — per-position SL triggered | 2024-12-12 13:10:00 | 1268.55 | 1268.52 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:25:00 | 1243.15 | 1248.73 | 0.00 | ORB-short ORB[1253.00,1264.90] vol=5.2x ATR=2.91 |
| Stop hit — per-position SL triggered | 2024-12-13 10:55:00 | 1246.06 | 1247.63 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:45:00 | 1251.90 | 1256.08 | 0.00 | ORB-short ORB[1253.45,1263.90] vol=1.8x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:40:00 | 1248.32 | 1252.48 | 0.00 | T1 1.5R @ 1248.32 |
| Stop hit — per-position SL triggered | 2024-12-17 10:45:00 | 1251.90 | 1252.47 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:35:00 | 1232.95 | 1227.62 | 0.00 | ORB-long ORB[1221.35,1227.50] vol=1.6x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-12-24 10:40:00 | 1230.76 | 1228.08 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 10:50:00 | 1214.75 | 1209.96 | 0.00 | ORB-long ORB[1206.60,1212.40] vol=1.7x ATR=2.56 |
| Stop hit — per-position SL triggered | 2024-12-31 11:35:00 | 1212.19 | 1211.19 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 1234.15 | 1247.80 | 0.00 | ORB-short ORB[1248.00,1260.85] vol=2.5x ATR=3.40 |
| Stop hit — per-position SL triggered | 2025-01-06 11:20:00 | 1237.55 | 1246.68 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 1251.70 | 1262.56 | 0.00 | ORB-short ORB[1259.60,1269.75] vol=1.7x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-01-09 11:05:00 | 1254.75 | 1261.12 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-01-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-13 10:00:00 | 1236.75 | 1232.81 | 0.00 | ORB-long ORB[1226.40,1235.00] vol=1.8x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:05:00 | 1241.44 | 1236.41 | 0.00 | T1 1.5R @ 1241.44 |
| Stop hit — per-position SL triggered | 2025-01-13 14:30:00 | 1236.75 | 1237.77 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-01-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 10:55:00 | 1234.75 | 1241.18 | 0.00 | ORB-short ORB[1244.10,1253.35] vol=1.6x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 11:40:00 | 1229.91 | 1239.19 | 0.00 | T1 1.5R @ 1229.91 |
| Stop hit — per-position SL triggered | 2025-01-14 14:05:00 | 1234.75 | 1236.34 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:55:00 | 1261.00 | 1267.01 | 0.00 | ORB-short ORB[1266.00,1273.00] vol=2.2x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 10:10:00 | 1257.41 | 1265.56 | 0.00 | T1 1.5R @ 1257.41 |
| Target hit | 2025-01-24 15:20:00 | 1246.55 | 1252.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2025-01-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:20:00 | 1245.35 | 1238.45 | 0.00 | ORB-long ORB[1232.40,1239.00] vol=1.5x ATR=2.45 |
| Stop hit — per-position SL triggered | 2025-01-30 10:30:00 | 1242.90 | 1239.30 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:55:00 | 1258.00 | 1255.97 | 0.00 | ORB-long ORB[1249.00,1257.85] vol=1.7x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:15:00 | 1261.56 | 1256.48 | 0.00 | T1 1.5R @ 1261.56 |
| Target hit | 2025-01-31 14:05:00 | 1258.85 | 1259.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — SELL (started 2025-02-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:45:00 | 1195.00 | 1213.05 | 0.00 | ORB-short ORB[1210.20,1226.90] vol=1.9x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-02-12 09:55:00 | 1199.79 | 1210.77 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:35:00 | 1235.90 | 1231.07 | 0.00 | ORB-long ORB[1222.00,1234.25] vol=1.7x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-02-20 09:45:00 | 1233.59 | 1231.63 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 11:10:00 | 1249.00 | 1233.78 | 0.00 | ORB-long ORB[1212.00,1222.45] vol=1.7x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 11:45:00 | 1253.99 | 1237.62 | 0.00 | T1 1.5R @ 1253.99 |
| Stop hit — per-position SL triggered | 2025-03-07 11:50:00 | 1249.00 | 1237.95 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:40:00 | 1255.80 | 1250.46 | 0.00 | ORB-long ORB[1244.85,1253.35] vol=1.6x ATR=2.97 |
| Stop hit — per-position SL triggered | 2025-03-10 10:25:00 | 1252.83 | 1252.45 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-18 11:15:00 | 1236.20 | 1239.47 | 0.00 | ORB-short ORB[1240.30,1248.35] vol=11.3x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-03-18 11:25:00 | 1238.19 | 1239.40 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-03-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:40:00 | 1248.35 | 1245.04 | 0.00 | ORB-long ORB[1238.80,1247.50] vol=1.7x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 09:45:00 | 1251.59 | 1245.81 | 0.00 | T1 1.5R @ 1251.59 |
| Stop hit — per-position SL triggered | 2025-03-19 09:55:00 | 1248.35 | 1246.31 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:15:00 | 1257.80 | 1256.10 | 0.00 | ORB-long ORB[1250.05,1255.95] vol=1.7x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-03-20 10:45:00 | 1255.34 | 1256.39 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:50:00 | 1278.95 | 1276.25 | 0.00 | ORB-long ORB[1270.10,1277.80] vol=1.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-03-21 11:10:00 | 1276.92 | 1276.50 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 11:00:00 | 1298.20 | 1290.41 | 0.00 | ORB-long ORB[1285.35,1294.65] vol=1.5x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 11:25:00 | 1301.98 | 1291.95 | 0.00 | T1 1.5R @ 1301.98 |
| Stop hit — per-position SL triggered | 2025-03-24 12:30:00 | 1298.20 | 1294.49 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-03-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 11:05:00 | 1292.80 | 1286.68 | 0.00 | ORB-long ORB[1280.00,1289.10] vol=1.6x ATR=2.50 |
| Stop hit — per-position SL triggered | 2025-03-28 11:15:00 | 1290.30 | 1287.22 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 10:55:00 | 1260.00 | 1268.35 | 0.00 | ORB-short ORB[1263.65,1275.00] vol=1.7x ATR=3.24 |
| Stop hit — per-position SL triggered | 2025-04-01 11:25:00 | 1263.24 | 1267.03 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-04-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 11:05:00 | 1248.20 | 1243.13 | 0.00 | ORB-long ORB[1233.05,1245.60] vol=1.8x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-04-03 11:15:00 | 1245.88 | 1243.54 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-04-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 10:35:00 | 1165.55 | 1174.75 | 0.00 | ORB-short ORB[1172.00,1186.60] vol=2.0x ATR=4.09 |
| Stop hit — per-position SL triggered | 2025-04-08 10:50:00 | 1169.64 | 1173.93 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:15:00 | 1246.30 | 1234.55 | 0.00 | ORB-long ORB[1227.60,1242.90] vol=1.6x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:35:00 | 1249.79 | 1236.47 | 0.00 | T1 1.5R @ 1249.79 |
| Target hit | 2025-04-17 15:20:00 | 1274.50 | 1260.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:45:00 | 1290.10 | 1281.25 | 0.00 | ORB-long ORB[1267.00,1276.00] vol=1.6x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:15:00 | 1294.31 | 1283.47 | 0.00 | T1 1.5R @ 1294.31 |
| Target hit | 2025-04-21 15:20:00 | 1295.70 | 1291.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2025-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:45:00 | 1302.80 | 1295.63 | 0.00 | ORB-long ORB[1291.10,1298.40] vol=1.5x ATR=2.99 |
| Stop hit — per-position SL triggered | 2025-04-23 10:00:00 | 1299.81 | 1298.04 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 10:20:00 | 1412.80 | 2024-05-16 10:45:00 | 1408.28 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-16 10:20:00 | 1412.80 | 2024-05-16 12:05:00 | 1412.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-23 09:35:00 | 1470.60 | 2024-05-23 10:00:00 | 1467.39 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-05-28 10:25:00 | 1477.48 | 2024-05-28 10:30:00 | 1474.73 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-05-30 10:00:00 | 1426.45 | 2024-05-30 10:35:00 | 1428.96 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-06-11 09:30:00 | 1462.20 | 2024-06-11 09:35:00 | 1456.91 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-06-11 09:30:00 | 1462.20 | 2024-06-11 09:40:00 | 1462.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 09:55:00 | 1468.68 | 2024-06-20 10:35:00 | 1474.91 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-06-20 09:55:00 | 1468.68 | 2024-06-20 15:20:00 | 1474.78 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2024-06-21 10:45:00 | 1459.08 | 2024-06-21 12:35:00 | 1454.79 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-06-21 10:45:00 | 1459.08 | 2024-06-21 15:00:00 | 1455.73 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-06-26 11:05:00 | 1476.48 | 2024-06-26 11:45:00 | 1481.00 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-06-26 11:05:00 | 1476.48 | 2024-06-26 15:20:00 | 1516.05 | TARGET_HIT | 0.50 | 2.68% |
| BUY | retest1 | 2024-06-27 09:45:00 | 1532.50 | 2024-06-27 09:50:00 | 1527.18 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-03 10:35:00 | 1549.88 | 2024-07-03 11:40:00 | 1544.42 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-07-03 10:35:00 | 1549.88 | 2024-07-03 13:15:00 | 1549.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 09:55:00 | 1566.68 | 2024-07-04 10:05:00 | 1563.58 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-05 10:10:00 | 1573.00 | 2024-07-05 10:20:00 | 1568.80 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-08 11:15:00 | 1599.65 | 2024-07-08 11:40:00 | 1596.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-10 10:05:00 | 1577.10 | 2024-07-10 10:20:00 | 1572.44 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-07-10 10:05:00 | 1577.10 | 2024-07-10 11:25:00 | 1577.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 11:00:00 | 1578.00 | 2024-07-11 11:10:00 | 1581.58 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-12 10:50:00 | 1590.75 | 2024-07-12 10:55:00 | 1596.81 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-07-12 10:50:00 | 1590.75 | 2024-07-12 11:15:00 | 1590.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-15 10:55:00 | 1586.65 | 2024-07-15 11:10:00 | 1590.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-16 09:45:00 | 1589.00 | 2024-07-16 10:05:00 | 1585.00 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-07-16 09:45:00 | 1589.00 | 2024-07-16 10:15:00 | 1589.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-18 11:10:00 | 1559.80 | 2024-07-18 11:30:00 | 1563.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-23 10:45:00 | 1482.45 | 2024-07-23 10:55:00 | 1485.43 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-24 10:35:00 | 1496.00 | 2024-07-24 10:40:00 | 1502.85 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-24 10:35:00 | 1496.00 | 2024-07-24 12:05:00 | 1496.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-29 11:00:00 | 1518.80 | 2024-07-29 11:35:00 | 1522.89 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-07-29 11:00:00 | 1518.80 | 2024-07-29 12:15:00 | 1518.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-30 11:00:00 | 1520.28 | 2024-07-30 12:25:00 | 1523.65 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2024-07-30 11:00:00 | 1520.28 | 2024-07-30 12:50:00 | 1520.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-01 11:10:00 | 1515.18 | 2024-08-01 11:15:00 | 1513.24 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2024-08-09 11:00:00 | 1470.48 | 2024-08-09 14:05:00 | 1475.04 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-08-09 11:00:00 | 1470.48 | 2024-08-09 15:20:00 | 1474.35 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-08-12 11:15:00 | 1472.00 | 2024-08-12 11:25:00 | 1469.53 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-08-16 09:35:00 | 1479.50 | 2024-08-16 09:40:00 | 1476.25 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-19 09:40:00 | 1494.00 | 2024-08-19 09:50:00 | 1490.80 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-28 10:40:00 | 1494.00 | 2024-08-28 10:45:00 | 1496.27 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-09-18 09:35:00 | 1479.15 | 2024-09-18 11:20:00 | 1477.04 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2024-09-19 11:05:00 | 1457.85 | 2024-09-19 12:00:00 | 1460.07 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-09-20 10:35:00 | 1482.40 | 2024-09-20 11:10:00 | 1486.97 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-09-20 10:35:00 | 1482.40 | 2024-09-20 13:30:00 | 1482.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 11:05:00 | 1499.58 | 2024-09-26 11:45:00 | 1502.28 | PARTIAL | 0.50 | 0.18% |
| BUY | retest1 | 2024-09-26 11:05:00 | 1499.58 | 2024-09-26 12:10:00 | 1499.58 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-03 11:15:00 | 1435.80 | 2024-10-03 12:00:00 | 1432.60 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2024-10-03 11:15:00 | 1435.80 | 2024-10-03 15:20:00 | 1407.75 | TARGET_HIT | 0.50 | 1.95% |
| SELL | retest1 | 2024-10-07 11:05:00 | 1379.08 | 2024-10-07 11:15:00 | 1382.66 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-14 11:00:00 | 1370.05 | 2024-10-14 11:10:00 | 1372.19 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-10-23 11:00:00 | 1347.13 | 2024-10-23 11:45:00 | 1351.91 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-10-23 11:00:00 | 1347.13 | 2024-10-23 12:15:00 | 1347.13 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-24 10:30:00 | 1332.35 | 2024-10-24 10:50:00 | 1327.80 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-10-24 10:30:00 | 1332.35 | 2024-10-24 12:30:00 | 1332.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-31 11:15:00 | 1333.85 | 2024-10-31 11:40:00 | 1335.78 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-11-12 10:55:00 | 1286.20 | 2024-11-12 11:15:00 | 1282.75 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-26 10:55:00 | 1296.00 | 2024-11-26 11:05:00 | 1292.84 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-11-28 10:40:00 | 1285.85 | 2024-11-28 11:30:00 | 1281.81 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-11-28 10:40:00 | 1285.85 | 2024-11-28 11:45:00 | 1285.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 10:50:00 | 1295.90 | 2024-11-29 11:50:00 | 1292.14 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-03 09:55:00 | 1317.90 | 2024-12-03 10:35:00 | 1322.05 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-12-03 09:55:00 | 1317.90 | 2024-12-03 15:20:00 | 1323.60 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-04 10:55:00 | 1313.40 | 2024-12-04 11:55:00 | 1309.59 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-12-04 10:55:00 | 1313.40 | 2024-12-04 12:25:00 | 1312.65 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2024-12-11 11:15:00 | 1283.30 | 2024-12-11 11:25:00 | 1285.31 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-12-12 11:00:00 | 1268.55 | 2024-12-12 12:00:00 | 1265.83 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2024-12-12 11:00:00 | 1268.55 | 2024-12-12 13:10:00 | 1268.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 10:25:00 | 1243.15 | 2024-12-13 10:55:00 | 1246.06 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-17 09:45:00 | 1251.90 | 2024-12-17 10:40:00 | 1248.32 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-12-17 09:45:00 | 1251.90 | 2024-12-17 10:45:00 | 1251.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 10:35:00 | 1232.95 | 2024-12-24 10:40:00 | 1230.76 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-12-31 10:50:00 | 1214.75 | 2024-12-31 11:35:00 | 1212.19 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-06 11:10:00 | 1234.15 | 2025-01-06 11:20:00 | 1237.55 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-09 10:45:00 | 1251.70 | 2025-01-09 11:05:00 | 1254.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-13 10:00:00 | 1236.75 | 2025-01-13 13:05:00 | 1241.44 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-01-13 10:00:00 | 1236.75 | 2025-01-13 14:30:00 | 1236.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-14 10:55:00 | 1234.75 | 2025-01-14 11:40:00 | 1229.91 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-14 10:55:00 | 1234.75 | 2025-01-14 14:05:00 | 1234.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 09:55:00 | 1261.00 | 2025-01-24 10:10:00 | 1257.41 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-01-24 09:55:00 | 1261.00 | 2025-01-24 15:20:00 | 1246.55 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2025-01-30 10:20:00 | 1245.35 | 2025-01-30 10:30:00 | 1242.90 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-01-31 10:55:00 | 1258.00 | 2025-01-31 11:15:00 | 1261.56 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-01-31 10:55:00 | 1258.00 | 2025-01-31 14:05:00 | 1258.85 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2025-02-12 09:45:00 | 1195.00 | 2025-02-12 09:55:00 | 1199.79 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-02-20 09:35:00 | 1235.90 | 2025-02-20 09:45:00 | 1233.59 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-03-07 11:10:00 | 1249.00 | 2025-03-07 11:45:00 | 1253.99 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-03-07 11:10:00 | 1249.00 | 2025-03-07 11:50:00 | 1249.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-10 09:40:00 | 1255.80 | 2025-03-10 10:25:00 | 1252.83 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-03-18 11:15:00 | 1236.20 | 2025-03-18 11:25:00 | 1238.19 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-03-19 09:40:00 | 1248.35 | 2025-03-19 09:45:00 | 1251.59 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-03-19 09:40:00 | 1248.35 | 2025-03-19 09:55:00 | 1248.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-20 10:15:00 | 1257.80 | 2025-03-20 10:45:00 | 1255.34 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-03-21 10:50:00 | 1278.95 | 2025-03-21 11:10:00 | 1276.92 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-03-24 11:00:00 | 1298.20 | 2025-03-24 11:25:00 | 1301.98 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-03-24 11:00:00 | 1298.20 | 2025-03-24 12:30:00 | 1298.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-28 11:05:00 | 1292.80 | 2025-03-28 11:15:00 | 1290.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-04-01 10:55:00 | 1260.00 | 2025-04-01 11:25:00 | 1263.24 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-04-03 11:05:00 | 1248.20 | 2025-04-03 11:15:00 | 1245.88 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-04-08 10:35:00 | 1165.55 | 2025-04-08 10:50:00 | 1169.64 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-17 11:15:00 | 1246.30 | 2025-04-17 11:35:00 | 1249.79 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-04-17 11:15:00 | 1246.30 | 2025-04-17 15:20:00 | 1274.50 | TARGET_HIT | 0.50 | 2.26% |
| BUY | retest1 | 2025-04-21 10:45:00 | 1290.10 | 2025-04-21 11:15:00 | 1294.31 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-04-21 10:45:00 | 1290.10 | 2025-04-21 15:20:00 | 1295.70 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-04-23 09:45:00 | 1302.80 | 2025-04-23 10:00:00 | 1299.81 | STOP_HIT | 1.00 | -0.23% |
