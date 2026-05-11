# Inventurus Knowledge Solutions Ltd. (IKS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1686.00
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
| ENTRY1 | 70 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 9 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 96 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 61
- **Target hits / Stop hits / Partials:** 9 / 61 / 26
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 6.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 25 | 43.9% | 7 | 32 | 18 | 0.11% | 6.1% |
| BUY @ 2nd Alert (retest1) | 57 | 25 | 43.9% | 7 | 32 | 18 | 0.11% | 6.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 10 | 25.6% | 2 | 29 | 8 | 0.01% | 0.4% |
| SELL @ 2nd Alert (retest1) | 39 | 10 | 25.6% | 2 | 29 | 8 | 0.01% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 96 | 35 | 36.5% | 9 | 61 | 26 | 0.07% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:30:00 | 1589.90 | 1583.85 | 0.00 | ORB-long ORB[1561.00,1579.00] vol=2.0x ATR=6.08 |
| Stop hit — per-position SL triggered | 2025-05-23 10:55:00 | 1583.82 | 1584.62 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 10:35:00 | 1606.10 | 1595.28 | 0.00 | ORB-long ORB[1580.90,1598.30] vol=2.2x ATR=5.06 |
| Stop hit — per-position SL triggered | 2025-05-27 10:45:00 | 1601.04 | 1595.63 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 10:10:00 | 1532.30 | 1519.24 | 0.00 | ORB-long ORB[1506.00,1520.90] vol=1.7x ATR=5.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:15:00 | 1540.71 | 1522.60 | 0.00 | T1 1.5R @ 1540.71 |
| Stop hit — per-position SL triggered | 2025-05-30 10:45:00 | 1532.30 | 1527.98 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:05:00 | 1549.90 | 1533.88 | 0.00 | ORB-long ORB[1514.10,1529.60] vol=2.5x ATR=6.83 |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 1543.07 | 1535.49 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 1544.30 | 1536.60 | 0.00 | ORB-long ORB[1519.90,1542.60] vol=2.1x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 09:35:00 | 1551.90 | 1540.93 | 0.00 | T1 1.5R @ 1551.90 |
| Stop hit — per-position SL triggered | 2025-06-03 09:40:00 | 1544.30 | 1541.76 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 1639.90 | 1628.12 | 0.00 | ORB-long ORB[1611.20,1625.00] vol=6.7x ATR=6.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 09:45:00 | 1649.04 | 1633.20 | 0.00 | T1 1.5R @ 1649.04 |
| Target hit | 2025-06-09 10:55:00 | 1649.40 | 1650.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2025-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 09:35:00 | 1592.50 | 1597.81 | 0.00 | ORB-short ORB[1594.20,1614.40] vol=14.4x ATR=6.12 |
| Stop hit — per-position SL triggered | 2025-06-25 09:40:00 | 1598.62 | 1597.71 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:35:00 | 1613.50 | 1598.71 | 0.00 | ORB-long ORB[1586.60,1601.00] vol=1.7x ATR=5.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 09:45:00 | 1622.11 | 1603.70 | 0.00 | T1 1.5R @ 1622.11 |
| Stop hit — per-position SL triggered | 2025-06-26 09:50:00 | 1613.50 | 1605.28 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-07-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:35:00 | 1635.10 | 1625.19 | 0.00 | ORB-long ORB[1602.60,1625.00] vol=2.6x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:50:00 | 1644.19 | 1629.35 | 0.00 | T1 1.5R @ 1644.19 |
| Stop hit — per-position SL triggered | 2025-07-09 11:35:00 | 1635.10 | 1632.70 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 09:35:00 | 1605.10 | 1601.75 | 0.00 | ORB-long ORB[1583.00,1604.00] vol=1.7x ATR=4.69 |
| Stop hit — per-position SL triggered | 2025-07-15 09:40:00 | 1600.41 | 1601.58 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:45:00 | 1604.20 | 1609.50 | 0.00 | ORB-short ORB[1606.20,1625.00] vol=3.7x ATR=4.60 |
| Stop hit — per-position SL triggered | 2025-07-18 09:50:00 | 1608.80 | 1610.73 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:35:00 | 1589.40 | 1601.06 | 0.00 | ORB-short ORB[1596.60,1615.20] vol=1.6x ATR=4.83 |
| Stop hit — per-position SL triggered | 2025-07-25 11:05:00 | 1594.23 | 1599.37 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:20:00 | 1605.20 | 1600.64 | 0.00 | ORB-long ORB[1582.60,1602.80] vol=4.0x ATR=5.14 |
| Stop hit — per-position SL triggered | 2025-07-28 10:45:00 | 1600.06 | 1600.84 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:00:00 | 1590.50 | 1582.20 | 0.00 | ORB-long ORB[1569.60,1588.50] vol=2.9x ATR=6.25 |
| Stop hit — per-position SL triggered | 2025-07-29 10:10:00 | 1584.25 | 1582.73 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:50:00 | 1579.80 | 1587.95 | 0.00 | ORB-short ORB[1582.00,1604.90] vol=3.5x ATR=5.36 |
| Stop hit — per-position SL triggered | 2025-07-30 12:00:00 | 1585.16 | 1586.75 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:50:00 | 1604.30 | 1612.20 | 0.00 | ORB-short ORB[1605.60,1626.10] vol=3.4x ATR=4.32 |
| Stop hit — per-position SL triggered | 2025-08-08 11:00:00 | 1608.62 | 1612.06 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:05:00 | 1575.70 | 1584.44 | 0.00 | ORB-short ORB[1581.10,1602.20] vol=4.5x ATR=4.33 |
| Stop hit — per-position SL triggered | 2025-08-12 12:05:00 | 1580.03 | 1583.73 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:45:00 | 1542.10 | 1554.32 | 0.00 | ORB-short ORB[1551.10,1569.50] vol=1.8x ATR=4.59 |
| Stop hit — per-position SL triggered | 2025-08-13 10:50:00 | 1546.69 | 1554.22 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:40:00 | 1573.20 | 1583.60 | 0.00 | ORB-short ORB[1583.80,1599.00] vol=2.7x ATR=5.81 |
| Stop hit — per-position SL triggered | 2025-08-22 09:45:00 | 1579.01 | 1583.07 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-09-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 10:50:00 | 1506.00 | 1513.85 | 0.00 | ORB-short ORB[1510.80,1520.90] vol=2.9x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-09-01 11:20:00 | 1510.21 | 1512.26 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-09-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:45:00 | 1559.00 | 1569.47 | 0.00 | ORB-short ORB[1561.60,1580.50] vol=2.1x ATR=4.46 |
| Stop hit — per-position SL triggered | 2025-09-05 11:30:00 | 1563.46 | 1568.58 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 10:30:00 | 1580.60 | 1572.99 | 0.00 | ORB-long ORB[1563.10,1579.00] vol=5.8x ATR=4.85 |
| Stop hit — per-position SL triggered | 2025-09-09 10:55:00 | 1575.75 | 1575.79 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 11:10:00 | 1557.10 | 1566.18 | 0.00 | ORB-short ORB[1560.50,1575.00] vol=2.0x ATR=3.42 |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 1560.52 | 1565.95 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:35:00 | 1523.10 | 1529.68 | 0.00 | ORB-short ORB[1526.50,1542.30] vol=1.9x ATR=3.93 |
| Stop hit — per-position SL triggered | 2025-09-19 10:05:00 | 1527.03 | 1526.51 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 11:05:00 | 1507.40 | 1498.02 | 0.00 | ORB-long ORB[1485.00,1503.80] vol=1.7x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 11:20:00 | 1513.51 | 1499.66 | 0.00 | T1 1.5R @ 1513.51 |
| Target hit | 2025-09-23 15:20:00 | 1514.30 | 1510.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-09-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:55:00 | 1516.40 | 1509.85 | 0.00 | ORB-long ORB[1496.50,1516.10] vol=5.2x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 11:20:00 | 1523.23 | 1512.08 | 0.00 | T1 1.5R @ 1523.23 |
| Stop hit — per-position SL triggered | 2025-09-24 11:55:00 | 1516.40 | 1512.47 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 09:30:00 | 1478.00 | 1471.70 | 0.00 | ORB-long ORB[1454.90,1473.10] vol=1.6x ATR=6.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 10:25:00 | 1487.50 | 1482.96 | 0.00 | T1 1.5R @ 1487.50 |
| Target hit | 2025-10-01 11:35:00 | 1492.10 | 1494.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — SELL (started 2025-10-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 09:45:00 | 1532.50 | 1537.35 | 0.00 | ORB-short ORB[1544.50,1554.60] vol=9.5x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:50:00 | 1526.18 | 1536.77 | 0.00 | T1 1.5R @ 1526.18 |
| Stop hit — per-position SL triggered | 2025-10-07 09:55:00 | 1532.50 | 1536.64 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:35:00 | 1530.40 | 1525.46 | 0.00 | ORB-long ORB[1515.90,1529.30] vol=1.6x ATR=5.82 |
| Stop hit — per-position SL triggered | 2025-10-09 09:45:00 | 1524.58 | 1526.11 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 10:55:00 | 1497.00 | 1501.79 | 0.00 | ORB-short ORB[1498.60,1508.50] vol=4.9x ATR=3.18 |
| Stop hit — per-position SL triggered | 2025-10-15 11:30:00 | 1500.18 | 1503.97 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 10:35:00 | 1527.10 | 1536.19 | 0.00 | ORB-short ORB[1531.20,1549.00] vol=1.6x ATR=4.91 |
| Stop hit — per-position SL triggered | 2025-10-23 11:20:00 | 1532.01 | 1530.58 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:05:00 | 1512.70 | 1522.27 | 0.00 | ORB-short ORB[1516.70,1526.50] vol=2.0x ATR=2.61 |
| Stop hit — per-position SL triggered | 2025-10-24 11:45:00 | 1515.31 | 1519.76 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:15:00 | 1535.30 | 1531.77 | 0.00 | ORB-long ORB[1525.00,1532.70] vol=2.0x ATR=4.76 |
| Stop hit — per-position SL triggered | 2025-10-27 10:50:00 | 1530.54 | 1532.11 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 10:30:00 | 1546.00 | 1537.62 | 0.00 | ORB-long ORB[1525.20,1541.20] vol=1.6x ATR=3.98 |
| Stop hit — per-position SL triggered | 2025-10-28 10:35:00 | 1542.02 | 1537.69 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:40:00 | 1569.70 | 1565.36 | 0.00 | ORB-long ORB[1551.60,1568.20] vol=3.4x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-10-29 11:05:00 | 1565.46 | 1565.80 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:45:00 | 1652.50 | 1650.44 | 0.00 | ORB-long ORB[1632.90,1650.00] vol=3.5x ATR=7.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:55:00 | 1663.85 | 1651.77 | 0.00 | T1 1.5R @ 1663.85 |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 1652.50 | 1652.30 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 10:40:00 | 1667.50 | 1678.85 | 0.00 | ORB-short ORB[1671.70,1694.00] vol=2.5x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 13:10:00 | 1657.92 | 1670.75 | 0.00 | T1 1.5R @ 1657.92 |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 1667.50 | 1670.62 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-11-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:05:00 | 1659.80 | 1665.64 | 0.00 | ORB-short ORB[1660.00,1675.00] vol=2.9x ATR=5.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:10:00 | 1651.61 | 1663.63 | 0.00 | T1 1.5R @ 1651.61 |
| Stop hit — per-position SL triggered | 2025-11-11 10:20:00 | 1659.80 | 1663.53 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 09:35:00 | 1617.10 | 1628.29 | 0.00 | ORB-short ORB[1629.40,1653.00] vol=2.8x ATR=5.59 |
| Stop hit — per-position SL triggered | 2025-11-13 09:50:00 | 1622.69 | 1621.14 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:55:00 | 1585.00 | 1588.56 | 0.00 | ORB-short ORB[1586.30,1599.40] vol=2.9x ATR=4.17 |
| Stop hit — per-position SL triggered | 2025-11-21 10:10:00 | 1589.17 | 1588.04 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:50:00 | 1704.20 | 1687.05 | 0.00 | ORB-long ORB[1673.30,1687.70] vol=4.8x ATR=5.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:55:00 | 1712.73 | 1697.65 | 0.00 | T1 1.5R @ 1712.73 |
| Stop hit — per-position SL triggered | 2025-11-27 11:05:00 | 1704.20 | 1700.30 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 11:15:00 | 1701.30 | 1697.17 | 0.00 | ORB-long ORB[1681.10,1698.90] vol=7.0x ATR=4.23 |
| Stop hit — per-position SL triggered | 2025-11-28 11:20:00 | 1697.07 | 1697.16 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-12-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 10:55:00 | 1701.60 | 1686.84 | 0.00 | ORB-long ORB[1679.90,1697.40] vol=2.7x ATR=5.81 |
| Stop hit — per-position SL triggered | 2025-12-01 11:50:00 | 1695.79 | 1690.74 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-12-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:20:00 | 1662.00 | 1665.79 | 0.00 | ORB-short ORB[1662.30,1683.00] vol=3.5x ATR=5.04 |
| Stop hit — per-position SL triggered | 2025-12-03 10:25:00 | 1667.04 | 1666.57 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:05:00 | 1681.40 | 1672.31 | 0.00 | ORB-long ORB[1653.00,1676.50] vol=1.6x ATR=5.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 10:55:00 | 1690.38 | 1678.23 | 0.00 | T1 1.5R @ 1690.38 |
| Stop hit — per-position SL triggered | 2025-12-04 11:55:00 | 1681.40 | 1679.11 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:35:00 | 1620.40 | 1631.43 | 0.00 | ORB-short ORB[1625.60,1645.10] vol=1.8x ATR=5.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:35:00 | 1612.17 | 1628.13 | 0.00 | T1 1.5R @ 1612.17 |
| Target hit | 2025-12-08 15:20:00 | 1592.80 | 1611.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2025-12-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:35:00 | 1645.60 | 1635.15 | 0.00 | ORB-long ORB[1631.10,1645.00] vol=1.8x ATR=5.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 10:55:00 | 1653.98 | 1640.56 | 0.00 | T1 1.5R @ 1653.98 |
| Target hit | 2025-12-15 15:20:00 | 1664.30 | 1652.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-12-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:45:00 | 1684.40 | 1675.13 | 0.00 | ORB-long ORB[1655.00,1678.00] vol=3.3x ATR=5.99 |
| Stop hit — per-position SL triggered | 2025-12-16 09:55:00 | 1678.41 | 1676.39 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 10:35:00 | 1654.60 | 1657.33 | 0.00 | ORB-short ORB[1666.50,1684.00] vol=2.0x ATR=4.96 |
| Stop hit — per-position SL triggered | 2025-12-18 12:05:00 | 1659.56 | 1656.31 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:10:00 | 1767.60 | 1759.24 | 0.00 | ORB-long ORB[1741.40,1760.00] vol=2.3x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 10:20:00 | 1773.86 | 1762.57 | 0.00 | T1 1.5R @ 1773.86 |
| Stop hit — per-position SL triggered | 2025-12-24 10:25:00 | 1767.60 | 1763.46 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:05:00 | 1728.40 | 1733.55 | 0.00 | ORB-short ORB[1735.50,1749.80] vol=3.0x ATR=4.26 |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 1732.66 | 1733.43 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1685.30 | 1678.56 | 0.00 | ORB-long ORB[1671.00,1681.50] vol=2.1x ATR=4.53 |
| Stop hit — per-position SL triggered | 2025-12-31 10:25:00 | 1680.77 | 1679.28 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:50:00 | 1671.60 | 1665.79 | 0.00 | ORB-long ORB[1655.50,1666.20] vol=2.5x ATR=4.09 |
| Stop hit — per-position SL triggered | 2026-01-01 11:20:00 | 1667.51 | 1666.92 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-01-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 10:10:00 | 1673.10 | 1680.05 | 0.00 | ORB-short ORB[1676.00,1691.10] vol=2.0x ATR=4.78 |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1677.88 | 1678.83 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-01-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:45:00 | 1708.60 | 1678.79 | 0.00 | ORB-long ORB[1665.00,1689.00] vol=1.7x ATR=7.09 |
| Stop hit — per-position SL triggered | 2026-01-05 09:50:00 | 1701.51 | 1680.57 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 1669.00 | 1683.50 | 0.00 | ORB-short ORB[1685.00,1704.00] vol=5.9x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 1664.17 | 1679.90 | 0.00 | T1 1.5R @ 1664.17 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1669.00 | 1674.19 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-01-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 09:40:00 | 1682.10 | 1686.70 | 0.00 | ORB-short ORB[1685.00,1701.60] vol=2.0x ATR=3.53 |
| Stop hit — per-position SL triggered | 2026-01-19 09:50:00 | 1685.63 | 1683.79 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-20 09:35:00 | 1693.20 | 1686.78 | 0.00 | ORB-long ORB[1672.00,1687.70] vol=3.6x ATR=6.16 |
| Stop hit — per-position SL triggered | 2026-01-20 09:50:00 | 1687.04 | 1687.74 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-01-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 10:45:00 | 1691.00 | 1690.24 | 0.00 | ORB-long ORB[1672.70,1690.00] vol=1.7x ATR=6.36 |
| Stop hit — per-position SL triggered | 2026-01-22 10:50:00 | 1684.64 | 1688.04 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:50:00 | 1551.60 | 1563.24 | 0.00 | ORB-short ORB[1565.20,1585.00] vol=2.1x ATR=5.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:15:00 | 1542.76 | 1554.58 | 0.00 | T1 1.5R @ 1542.76 |
| Target hit | 2026-01-29 12:50:00 | 1531.80 | 1530.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 61 — BUY (started 2026-02-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 09:35:00 | 1686.30 | 1666.42 | 0.00 | ORB-long ORB[1647.00,1670.00] vol=3.0x ATR=12.33 |
| Stop hit — per-position SL triggered | 2026-02-03 10:05:00 | 1673.97 | 1678.00 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 1714.00 | 1702.09 | 0.00 | ORB-long ORB[1685.20,1705.00] vol=5.9x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:15:00 | 1720.40 | 1705.84 | 0.00 | T1 1.5R @ 1720.40 |
| Stop hit — per-position SL triggered | 2026-02-09 11:50:00 | 1714.00 | 1711.63 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 1712.30 | 1708.63 | 0.00 | ORB-long ORB[1692.70,1709.00] vol=3.3x ATR=6.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:35:00 | 1722.42 | 1712.31 | 0.00 | T1 1.5R @ 1722.42 |
| Stop hit — per-position SL triggered | 2026-02-10 10:50:00 | 1712.30 | 1719.51 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 1642.50 | 1644.40 | 0.00 | ORB-short ORB[1647.70,1659.90] vol=5.1x ATR=6.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:45:00 | 1632.34 | 1643.17 | 0.00 | T1 1.5R @ 1632.34 |
| Stop hit — per-position SL triggered | 2026-02-18 12:50:00 | 1642.50 | 1641.60 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-03-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:00:00 | 1343.00 | 1331.38 | 0.00 | ORB-long ORB[1323.70,1340.00] vol=3.3x ATR=5.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:10:00 | 1351.92 | 1335.69 | 0.00 | T1 1.5R @ 1351.92 |
| Target hit | 2026-03-04 11:55:00 | 1344.80 | 1346.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — SELL (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 1504.80 | 1513.66 | 0.00 | ORB-short ORB[1512.40,1525.40] vol=2.0x ATR=5.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:35:00 | 1496.42 | 1505.68 | 0.00 | T1 1.5R @ 1496.42 |
| Stop hit — per-position SL triggered | 2026-04-17 11:40:00 | 1504.80 | 1503.17 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 1435.20 | 1428.81 | 0.00 | ORB-long ORB[1414.10,1434.50] vol=1.6x ATR=5.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:40:00 | 1443.92 | 1432.47 | 0.00 | T1 1.5R @ 1443.92 |
| Target hit | 2026-04-21 12:15:00 | 1437.40 | 1437.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — SELL (started 2026-04-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:30:00 | 1508.70 | 1519.60 | 0.00 | ORB-short ORB[1511.70,1530.90] vol=2.0x ATR=5.04 |
| Stop hit — per-position SL triggered | 2026-04-29 10:40:00 | 1513.74 | 1519.06 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 1687.70 | 1680.10 | 0.00 | ORB-long ORB[1660.00,1683.90] vol=2.3x ATR=7.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 12:35:00 | 1699.26 | 1685.08 | 0.00 | T1 1.5R @ 1699.26 |
| Target hit | 2026-05-05 13:10:00 | 1693.10 | 1693.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 70 — BUY (started 2026-05-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:55:00 | 1686.70 | 1682.16 | 0.00 | ORB-long ORB[1668.60,1685.70] vol=2.4x ATR=5.10 |
| Stop hit — per-position SL triggered | 2026-05-07 11:25:00 | 1681.60 | 1682.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-23 10:30:00 | 1589.90 | 2025-05-23 10:55:00 | 1583.82 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-27 10:35:00 | 1606.10 | 2025-05-27 10:45:00 | 1601.04 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-30 10:10:00 | 1532.30 | 2025-05-30 10:15:00 | 1540.71 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-05-30 10:10:00 | 1532.30 | 2025-05-30 10:45:00 | 1532.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-02 10:05:00 | 1549.90 | 2025-06-02 10:15:00 | 1543.07 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-06-03 09:30:00 | 1544.30 | 2025-06-03 09:35:00 | 1551.90 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-06-03 09:30:00 | 1544.30 | 2025-06-03 09:40:00 | 1544.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-09 09:30:00 | 1639.90 | 2025-06-09 09:45:00 | 1649.04 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-06-09 09:30:00 | 1639.90 | 2025-06-09 10:55:00 | 1649.40 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-06-25 09:35:00 | 1592.50 | 2025-06-25 09:40:00 | 1598.62 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-26 09:35:00 | 1613.50 | 2025-06-26 09:45:00 | 1622.11 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-06-26 09:35:00 | 1613.50 | 2025-06-26 09:50:00 | 1613.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-09 10:35:00 | 1635.10 | 2025-07-09 10:50:00 | 1644.19 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-07-09 10:35:00 | 1635.10 | 2025-07-09 11:35:00 | 1635.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 09:35:00 | 1605.10 | 2025-07-15 09:40:00 | 1600.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-18 09:45:00 | 1604.20 | 2025-07-18 09:50:00 | 1608.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-25 10:35:00 | 1589.40 | 2025-07-25 11:05:00 | 1594.23 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-28 10:20:00 | 1605.20 | 2025-07-28 10:45:00 | 1600.06 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-29 10:00:00 | 1590.50 | 2025-07-29 10:10:00 | 1584.25 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-07-30 10:50:00 | 1579.80 | 2025-07-30 12:00:00 | 1585.16 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-08-08 10:50:00 | 1604.30 | 2025-08-08 11:00:00 | 1608.62 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-12 11:05:00 | 1575.70 | 2025-08-12 12:05:00 | 1580.03 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-13 10:45:00 | 1542.10 | 2025-08-13 10:50:00 | 1546.69 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-22 09:40:00 | 1573.20 | 2025-08-22 09:45:00 | 1579.01 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-09-01 10:50:00 | 1506.00 | 2025-09-01 11:20:00 | 1510.21 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-09-05 10:45:00 | 1559.00 | 2025-09-05 11:30:00 | 1563.46 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-09 10:30:00 | 1580.60 | 2025-09-09 10:55:00 | 1575.75 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-09-16 11:10:00 | 1557.10 | 2025-09-16 11:15:00 | 1560.52 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-19 09:35:00 | 1523.10 | 2025-09-19 10:05:00 | 1527.03 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-23 11:05:00 | 1507.40 | 2025-09-23 11:20:00 | 1513.51 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-09-23 11:05:00 | 1507.40 | 2025-09-23 15:20:00 | 1514.30 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2025-09-24 10:55:00 | 1516.40 | 2025-09-24 11:20:00 | 1523.23 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-24 10:55:00 | 1516.40 | 2025-09-24 11:55:00 | 1516.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-01 09:30:00 | 1478.00 | 2025-10-01 10:25:00 | 1487.50 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-10-01 09:30:00 | 1478.00 | 2025-10-01 11:35:00 | 1492.10 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2025-10-07 09:45:00 | 1532.50 | 2025-10-07 09:50:00 | 1526.18 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-07 09:45:00 | 1532.50 | 2025-10-07 09:55:00 | 1532.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-09 09:35:00 | 1530.40 | 2025-10-09 09:45:00 | 1524.58 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-10-15 10:55:00 | 1497.00 | 2025-10-15 11:30:00 | 1500.18 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-23 10:35:00 | 1527.10 | 2025-10-23 11:20:00 | 1532.01 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-24 11:05:00 | 1512.70 | 2025-10-24 11:45:00 | 1515.31 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-10-27 10:15:00 | 1535.30 | 2025-10-27 10:50:00 | 1530.54 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-28 10:30:00 | 1546.00 | 2025-10-28 10:35:00 | 1542.02 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-29 10:40:00 | 1569.70 | 2025-10-29 11:05:00 | 1565.46 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-04 09:45:00 | 1652.50 | 2025-11-04 09:55:00 | 1663.85 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-11-04 09:45:00 | 1652.50 | 2025-11-04 11:15:00 | 1652.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-10 10:40:00 | 1667.50 | 2025-11-10 13:10:00 | 1657.92 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-11-10 10:40:00 | 1667.50 | 2025-11-10 13:15:00 | 1667.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 10:05:00 | 1659.80 | 2025-11-11 10:10:00 | 1651.61 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-11-11 10:05:00 | 1659.80 | 2025-11-11 10:20:00 | 1659.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-13 09:35:00 | 1617.10 | 2025-11-13 09:50:00 | 1622.69 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-21 09:55:00 | 1585.00 | 2025-11-21 10:10:00 | 1589.17 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-27 10:50:00 | 1704.20 | 2025-11-27 10:55:00 | 1712.73 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-11-27 10:50:00 | 1704.20 | 2025-11-27 11:05:00 | 1704.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-28 11:15:00 | 1701.30 | 2025-11-28 11:20:00 | 1697.07 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-01 10:55:00 | 1701.60 | 2025-12-01 11:50:00 | 1695.79 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-03 10:20:00 | 1662.00 | 2025-12-03 10:25:00 | 1667.04 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-04 10:05:00 | 1681.40 | 2025-12-04 10:55:00 | 1690.38 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-12-04 10:05:00 | 1681.40 | 2025-12-04 11:55:00 | 1681.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 10:35:00 | 1620.40 | 2025-12-08 11:35:00 | 1612.17 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-08 10:35:00 | 1620.40 | 2025-12-08 15:20:00 | 1592.80 | TARGET_HIT | 0.50 | 1.70% |
| BUY | retest1 | 2025-12-15 10:35:00 | 1645.60 | 2025-12-15 10:55:00 | 1653.98 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-12-15 10:35:00 | 1645.60 | 2025-12-15 15:20:00 | 1664.30 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2025-12-16 09:45:00 | 1684.40 | 2025-12-16 09:55:00 | 1678.41 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-12-18 10:35:00 | 1654.60 | 2025-12-18 12:05:00 | 1659.56 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-24 10:10:00 | 1767.60 | 2025-12-24 10:20:00 | 1773.86 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-24 10:10:00 | 1767.60 | 2025-12-24 10:25:00 | 1767.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-26 11:05:00 | 1728.40 | 2025-12-26 11:15:00 | 1732.66 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-31 10:15:00 | 1685.30 | 2025-12-31 10:25:00 | 1680.77 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-01 10:50:00 | 1671.60 | 2026-01-01 11:20:00 | 1667.51 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-02 10:10:00 | 1673.10 | 2026-01-02 10:15:00 | 1677.88 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-05 09:45:00 | 1708.60 | 2026-01-05 09:50:00 | 1701.51 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-01-08 11:00:00 | 1669.00 | 2026-01-08 11:15:00 | 1664.17 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-01-08 11:00:00 | 1669.00 | 2026-01-08 12:15:00 | 1669.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-19 09:40:00 | 1682.10 | 2026-01-19 09:50:00 | 1685.63 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-20 09:35:00 | 1693.20 | 2026-01-20 09:50:00 | 1687.04 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-01-22 10:45:00 | 1691.00 | 2026-01-22 10:50:00 | 1684.64 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-01-29 09:50:00 | 1551.60 | 2026-01-29 10:15:00 | 1542.76 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-01-29 09:50:00 | 1551.60 | 2026-01-29 12:50:00 | 1531.80 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2026-02-03 09:35:00 | 1686.30 | 2026-02-03 10:05:00 | 1673.97 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2026-02-09 11:00:00 | 1714.00 | 2026-02-09 11:15:00 | 1720.40 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-09 11:00:00 | 1714.00 | 2026-02-09 11:50:00 | 1714.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 09:30:00 | 1712.30 | 2026-02-10 09:35:00 | 1722.42 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-02-10 09:30:00 | 1712.30 | 2026-02-10 10:50:00 | 1712.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 09:55:00 | 1642.50 | 2026-02-18 11:45:00 | 1632.34 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-02-18 09:55:00 | 1642.50 | 2026-02-18 12:50:00 | 1642.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-04 10:00:00 | 1343.00 | 2026-03-04 10:10:00 | 1351.92 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-03-04 10:00:00 | 1343.00 | 2026-03-04 11:55:00 | 1344.80 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2026-04-17 09:35:00 | 1504.80 | 2026-04-17 10:35:00 | 1496.42 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-17 09:35:00 | 1504.80 | 2026-04-17 11:40:00 | 1504.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 10:00:00 | 1435.20 | 2026-04-21 10:40:00 | 1443.92 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-21 10:00:00 | 1435.20 | 2026-04-21 12:15:00 | 1437.40 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-04-29 10:30:00 | 1508.70 | 2026-04-29 10:40:00 | 1513.74 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-05 11:05:00 | 1687.70 | 2026-05-05 12:35:00 | 1699.26 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-05-05 11:05:00 | 1687.70 | 2026-05-05 13:10:00 | 1693.10 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-05-07 10:55:00 | 1686.70 | 2026-05-07 11:25:00 | 1681.60 | STOP_HIT | 1.00 | -0.30% |
