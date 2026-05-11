# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1845.00
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
| ENTRY1 | 76 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 18 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 105 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 58
- **Target hits / Stop hits / Partials:** 18 / 58 / 29
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 16.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 26 | 55.3% | 11 | 21 | 15 | 0.25% | 11.7% |
| BUY @ 2nd Alert (retest1) | 47 | 26 | 55.3% | 11 | 21 | 15 | 0.25% | 11.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 58 | 21 | 36.2% | 7 | 37 | 14 | 0.08% | 4.4% |
| SELL @ 2nd Alert (retest1) | 58 | 21 | 36.2% | 7 | 37 | 14 | 0.08% | 4.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 105 | 47 | 44.8% | 18 | 58 | 29 | 0.15% | 16.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:25:00 | 1533.40 | 1525.05 | 0.00 | ORB-long ORB[1516.05,1531.65] vol=1.6x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 10:45:00 | 1539.35 | 1530.14 | 0.00 | T1 1.5R @ 1539.35 |
| Target hit | 2024-05-14 12:20:00 | 1535.00 | 1535.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2024-05-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:50:00 | 1527.75 | 1531.86 | 0.00 | ORB-short ORB[1538.30,1550.25] vol=2.0x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-05-15 11:00:00 | 1531.44 | 1531.57 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:30:00 | 1474.80 | 1486.79 | 0.00 | ORB-short ORB[1482.55,1497.50] vol=1.7x ATR=4.72 |
| Stop hit — per-position SL triggered | 2024-05-27 10:40:00 | 1479.52 | 1486.17 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:20:00 | 1456.15 | 1466.61 | 0.00 | ORB-short ORB[1463.00,1473.40] vol=2.1x ATR=4.65 |
| Stop hit — per-position SL triggered | 2024-05-28 10:35:00 | 1460.80 | 1465.26 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 11:15:00 | 1474.05 | 1463.70 | 0.00 | ORB-long ORB[1432.00,1454.05] vol=1.7x ATR=7.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 13:25:00 | 1485.25 | 1470.70 | 0.00 | T1 1.5R @ 1485.25 |
| Target hit | 2024-06-05 15:20:00 | 1486.30 | 1475.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2024-06-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 11:00:00 | 1513.80 | 1516.82 | 0.00 | ORB-short ORB[1515.00,1525.00] vol=2.8x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 12:00:00 | 1509.84 | 1515.73 | 0.00 | T1 1.5R @ 1509.84 |
| Target hit | 2024-06-11 15:20:00 | 1499.60 | 1507.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 1499.00 | 1506.54 | 0.00 | ORB-short ORB[1506.15,1514.60] vol=1.8x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-06-13 10:15:00 | 1502.56 | 1503.45 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 11:10:00 | 1512.80 | 1516.74 | 0.00 | ORB-short ORB[1515.00,1522.85] vol=1.7x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-06-18 11:55:00 | 1515.33 | 1515.79 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:25:00 | 1510.00 | 1520.93 | 0.00 | ORB-short ORB[1519.90,1525.00] vol=1.5x ATR=2.95 |
| Stop hit — per-position SL triggered | 2024-06-19 10:40:00 | 1512.95 | 1518.01 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-20 11:00:00 | 1474.55 | 1478.78 | 0.00 | ORB-short ORB[1475.00,1487.95] vol=2.2x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-06-20 11:35:00 | 1477.33 | 1478.26 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:10:00 | 1498.40 | 1491.19 | 0.00 | ORB-long ORB[1475.05,1488.05] vol=2.8x ATR=4.46 |
| Stop hit — per-position SL triggered | 2024-06-24 14:20:00 | 1493.94 | 1493.09 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:50:00 | 1506.75 | 1501.98 | 0.00 | ORB-long ORB[1495.25,1505.90] vol=5.4x ATR=2.97 |
| Stop hit — per-position SL triggered | 2024-06-26 11:25:00 | 1503.78 | 1502.50 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:40:00 | 1543.90 | 1536.45 | 0.00 | ORB-long ORB[1526.00,1540.95] vol=2.4x ATR=3.54 |
| Stop hit — per-position SL triggered | 2024-07-04 10:45:00 | 1540.36 | 1536.98 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 1561.10 | 1564.97 | 0.00 | ORB-short ORB[1568.40,1574.80] vol=1.9x ATR=3.54 |
| Stop hit — per-position SL triggered | 2024-07-08 11:55:00 | 1564.64 | 1564.57 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 11:00:00 | 1575.80 | 1579.01 | 0.00 | ORB-short ORB[1580.20,1592.80] vol=1.5x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-07-10 11:20:00 | 1579.41 | 1578.88 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:10:00 | 1573.50 | 1581.73 | 0.00 | ORB-short ORB[1581.05,1602.50] vol=2.0x ATR=3.27 |
| Stop hit — per-position SL triggered | 2024-07-11 11:15:00 | 1576.77 | 1581.49 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 11:10:00 | 1590.95 | 1586.25 | 0.00 | ORB-long ORB[1577.30,1587.95] vol=1.5x ATR=2.94 |
| Stop hit — per-position SL triggered | 2024-07-15 11:45:00 | 1588.01 | 1586.96 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 09:35:00 | 1590.40 | 1584.20 | 0.00 | ORB-long ORB[1572.00,1587.85] vol=1.9x ATR=3.13 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 1587.27 | 1584.68 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:00:00 | 1610.25 | 1601.15 | 0.00 | ORB-long ORB[1592.00,1607.95] vol=2.4x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 12:35:00 | 1617.14 | 1611.25 | 0.00 | T1 1.5R @ 1617.14 |
| Target hit | 2024-07-24 15:20:00 | 1620.95 | 1615.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-07-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:50:00 | 1616.00 | 1612.64 | 0.00 | ORB-long ORB[1604.95,1615.00] vol=2.2x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 11:40:00 | 1622.51 | 1614.16 | 0.00 | T1 1.5R @ 1622.51 |
| Target hit | 2024-07-25 15:20:00 | 1658.00 | 1643.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-07-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 09:50:00 | 1699.50 | 1701.71 | 0.00 | ORB-short ORB[1702.10,1715.60] vol=2.2x ATR=3.83 |
| Stop hit — per-position SL triggered | 2024-07-31 10:00:00 | 1703.33 | 1701.75 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 09:35:00 | 1701.95 | 1707.90 | 0.00 | ORB-short ORB[1702.05,1725.75] vol=1.8x ATR=5.38 |
| Stop hit — per-position SL triggered | 2024-08-01 09:40:00 | 1707.33 | 1707.54 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:45:00 | 1732.10 | 1728.02 | 0.00 | ORB-long ORB[1709.40,1731.70] vol=2.1x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-08-07 10:50:00 | 1728.31 | 1728.12 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 11:05:00 | 1759.90 | 1763.26 | 0.00 | ORB-short ORB[1760.20,1769.50] vol=1.6x ATR=2.15 |
| Stop hit — per-position SL triggered | 2024-08-22 11:50:00 | 1762.05 | 1762.16 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 10:30:00 | 1764.00 | 1762.99 | 0.00 | ORB-long ORB[1751.90,1761.95] vol=14.5x ATR=2.92 |
| Stop hit — per-position SL triggered | 2024-08-23 10:35:00 | 1761.08 | 1762.99 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 11:10:00 | 1789.95 | 1783.28 | 0.00 | ORB-long ORB[1767.15,1781.60] vol=1.7x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-08-27 15:20:00 | 1788.50 | 1787.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 11:15:00 | 1815.35 | 1803.17 | 0.00 | ORB-long ORB[1791.30,1808.20] vol=1.6x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 12:00:00 | 1821.28 | 1806.96 | 0.00 | T1 1.5R @ 1821.28 |
| Stop hit — per-position SL triggered | 2024-08-30 15:10:00 | 1815.35 | 1817.91 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 1835.30 | 1829.10 | 0.00 | ORB-long ORB[1817.20,1833.65] vol=2.2x ATR=4.18 |
| Stop hit — per-position SL triggered | 2024-09-03 09:50:00 | 1831.12 | 1829.98 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:30:00 | 1857.25 | 1849.45 | 0.00 | ORB-long ORB[1831.10,1851.85] vol=2.1x ATR=4.05 |
| Stop hit — per-position SL triggered | 2024-09-11 10:40:00 | 1853.20 | 1849.82 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 11:10:00 | 1863.00 | 1860.52 | 0.00 | ORB-long ORB[1850.90,1862.50] vol=1.7x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 11:30:00 | 1866.85 | 1861.59 | 0.00 | T1 1.5R @ 1866.85 |
| Target hit | 2024-09-16 12:30:00 | 1863.95 | 1864.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2024-09-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 11:10:00 | 1858.30 | 1860.99 | 0.00 | ORB-short ORB[1858.55,1868.00] vol=1.6x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-09-18 12:10:00 | 1860.98 | 1860.14 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 11:15:00 | 1850.10 | 1849.62 | 0.00 | ORB-long ORB[1835.00,1850.00] vol=1.9x ATR=3.39 |
| Stop hit — per-position SL triggered | 2024-09-19 12:05:00 | 1846.71 | 1849.83 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:35:00 | 1846.85 | 1848.25 | 0.00 | ORB-short ORB[1848.40,1856.30] vol=3.3x ATR=3.98 |
| Stop hit — per-position SL triggered | 2024-09-20 10:40:00 | 1850.83 | 1848.55 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:15:00 | 1870.70 | 1865.95 | 0.00 | ORB-long ORB[1856.00,1869.45] vol=2.4x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-09-24 10:25:00 | 1867.50 | 1866.65 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:40:00 | 1934.65 | 1947.14 | 0.00 | ORB-short ORB[1946.90,1960.35] vol=2.1x ATR=5.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 10:45:00 | 1925.92 | 1945.65 | 0.00 | T1 1.5R @ 1925.92 |
| Stop hit — per-position SL triggered | 2024-09-30 12:00:00 | 1934.65 | 1940.77 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:45:00 | 1920.90 | 1915.83 | 0.00 | ORB-long ORB[1903.65,1918.00] vol=1.7x ATR=5.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 11:50:00 | 1928.60 | 1918.36 | 0.00 | T1 1.5R @ 1928.60 |
| Target hit | 2024-10-04 12:45:00 | 1928.50 | 1932.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 1891.65 | 1902.02 | 0.00 | ORB-short ORB[1902.20,1917.95] vol=2.7x ATR=4.56 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 1896.21 | 1901.80 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:20:00 | 1916.05 | 1913.47 | 0.00 | ORB-long ORB[1901.55,1914.60] vol=4.3x ATR=4.90 |
| Stop hit — per-position SL triggered | 2024-10-08 12:05:00 | 1911.15 | 1915.71 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:00:00 | 1919.00 | 1923.97 | 0.00 | ORB-short ORB[1924.40,1937.95] vol=2.4x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 13:00:00 | 1912.49 | 1920.92 | 0.00 | T1 1.5R @ 1912.49 |
| Target hit | 2024-10-10 15:20:00 | 1886.75 | 1902.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:15:00 | 1901.25 | 1911.32 | 0.00 | ORB-short ORB[1910.25,1920.40] vol=1.6x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-10-15 10:55:00 | 1905.25 | 1907.76 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 10:55:00 | 1893.45 | 1902.84 | 0.00 | ORB-short ORB[1904.40,1920.55] vol=2.7x ATR=4.24 |
| Stop hit — per-position SL triggered | 2024-10-21 11:20:00 | 1897.69 | 1901.27 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 09:45:00 | 1854.95 | 1864.26 | 0.00 | ORB-short ORB[1856.80,1872.35] vol=2.1x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:05:00 | 1845.63 | 1861.88 | 0.00 | T1 1.5R @ 1845.63 |
| Stop hit — per-position SL triggered | 2024-10-31 10:25:00 | 1854.95 | 1861.04 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:50:00 | 1755.95 | 1746.57 | 0.00 | ORB-long ORB[1730.00,1752.95] vol=2.1x ATR=4.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:20:00 | 1762.16 | 1751.56 | 0.00 | T1 1.5R @ 1762.16 |
| Target hit | 2024-11-19 15:20:00 | 1774.40 | 1773.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2024-11-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 09:50:00 | 1786.95 | 1773.65 | 0.00 | ORB-long ORB[1758.05,1779.90] vol=1.6x ATR=6.23 |
| Stop hit — per-position SL triggered | 2024-11-21 10:05:00 | 1780.72 | 1775.51 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 11:15:00 | 1773.30 | 1784.43 | 0.00 | ORB-short ORB[1793.20,1806.75] vol=1.7x ATR=3.68 |
| Stop hit — per-position SL triggered | 2024-11-26 11:25:00 | 1776.98 | 1783.96 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 1732.10 | 1742.54 | 0.00 | ORB-short ORB[1741.00,1760.70] vol=2.5x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:45:00 | 1726.04 | 1739.97 | 0.00 | T1 1.5R @ 1726.04 |
| Stop hit — per-position SL triggered | 2024-11-28 11:15:00 | 1732.10 | 1736.79 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:45:00 | 1790.05 | 1797.49 | 0.00 | ORB-short ORB[1791.45,1814.85] vol=2.0x ATR=3.14 |
| Stop hit — per-position SL triggered | 2024-12-04 10:50:00 | 1793.19 | 1797.00 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 1781.30 | 1795.95 | 0.00 | ORB-short ORB[1794.65,1808.70] vol=1.9x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-12-05 11:00:00 | 1785.89 | 1795.35 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 11:05:00 | 1809.90 | 1814.48 | 0.00 | ORB-short ORB[1810.05,1823.95] vol=2.1x ATR=3.66 |
| Stop hit — per-position SL triggered | 2024-12-06 12:00:00 | 1813.56 | 1813.48 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 11:10:00 | 1803.00 | 1808.41 | 0.00 | ORB-short ORB[1807.10,1823.00] vol=1.9x ATR=2.73 |
| Stop hit — per-position SL triggered | 2024-12-10 11:15:00 | 1805.73 | 1808.13 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:10:00 | 1807.70 | 1813.52 | 0.00 | ORB-short ORB[1812.05,1820.00] vol=1.9x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:45:00 | 1803.73 | 1812.01 | 0.00 | T1 1.5R @ 1803.73 |
| Stop hit — per-position SL triggered | 2024-12-12 13:35:00 | 1807.70 | 1807.14 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:05:00 | 1779.60 | 1790.27 | 0.00 | ORB-short ORB[1796.00,1811.90] vol=2.3x ATR=4.13 |
| Stop hit — per-position SL triggered | 2024-12-13 11:10:00 | 1783.73 | 1789.81 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:10:00 | 1795.10 | 1801.79 | 0.00 | ORB-short ORB[1800.70,1822.50] vol=2.0x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-12-16 11:35:00 | 1798.29 | 1800.90 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 11:05:00 | 1829.70 | 1822.33 | 0.00 | ORB-long ORB[1803.15,1823.25] vol=3.8x ATR=2.93 |
| Stop hit — per-position SL triggered | 2024-12-24 11:25:00 | 1826.77 | 1823.48 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:00:00 | 1869.95 | 1861.33 | 0.00 | ORB-long ORB[1856.00,1866.65] vol=1.5x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 10:20:00 | 1877.02 | 1865.74 | 0.00 | T1 1.5R @ 1877.02 |
| Target hit | 2024-12-30 13:20:00 | 1874.75 | 1875.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — SELL (started 2025-01-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 11:10:00 | 1886.85 | 1889.58 | 0.00 | ORB-short ORB[1888.05,1902.95] vol=3.7x ATR=5.15 |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 1892.00 | 1889.62 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:35:00 | 1877.30 | 1878.72 | 0.00 | ORB-short ORB[1879.85,1896.00] vol=10.2x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:50:00 | 1871.53 | 1878.30 | 0.00 | T1 1.5R @ 1871.53 |
| Stop hit — per-position SL triggered | 2025-01-02 09:55:00 | 1877.30 | 1878.12 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 11:00:00 | 1862.70 | 1866.51 | 0.00 | ORB-short ORB[1866.65,1884.00] vol=2.2x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 11:45:00 | 1857.16 | 1864.43 | 0.00 | T1 1.5R @ 1857.16 |
| Target hit | 2025-01-03 15:20:00 | 1846.20 | 1856.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2025-01-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:45:00 | 1818.00 | 1808.31 | 0.00 | ORB-long ORB[1786.05,1801.65] vol=3.1x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:00:00 | 1824.63 | 1811.42 | 0.00 | T1 1.5R @ 1824.63 |
| Target hit | 2025-01-23 15:20:00 | 1837.10 | 1826.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — SELL (started 2025-01-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:30:00 | 1809.40 | 1815.33 | 0.00 | ORB-short ORB[1814.00,1831.60] vol=1.6x ATR=5.38 |
| Stop hit — per-position SL triggered | 2025-01-24 09:35:00 | 1814.78 | 1814.70 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:45:00 | 1799.40 | 1807.87 | 0.00 | ORB-short ORB[1810.55,1824.25] vol=1.9x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 12:10:00 | 1792.47 | 1802.70 | 0.00 | T1 1.5R @ 1792.47 |
| Target hit | 2025-01-27 15:20:00 | 1784.85 | 1792.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-01-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:20:00 | 1712.75 | 1707.52 | 0.00 | ORB-long ORB[1696.00,1712.15] vol=2.2x ATR=5.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:50:00 | 1721.28 | 1709.11 | 0.00 | T1 1.5R @ 1721.28 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 1712.75 | 1711.08 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-02-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:55:00 | 1711.55 | 1723.14 | 0.00 | ORB-short ORB[1730.00,1748.95] vol=1.9x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:00:00 | 1704.80 | 1722.08 | 0.00 | T1 1.5R @ 1704.80 |
| Target hit | 2025-02-14 14:20:00 | 1705.00 | 1704.93 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — SELL (started 2025-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:40:00 | 1712.00 | 1720.80 | 0.00 | ORB-short ORB[1714.05,1729.00] vol=2.5x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 11:20:00 | 1705.07 | 1718.89 | 0.00 | T1 1.5R @ 1705.07 |
| Target hit | 2025-02-18 15:20:00 | 1699.45 | 1708.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2025-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 10:35:00 | 1691.90 | 1675.35 | 0.00 | ORB-long ORB[1650.00,1673.85] vol=2.0x ATR=7.11 |
| Stop hit — per-position SL triggered | 2025-02-19 14:00:00 | 1684.79 | 1681.74 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-02-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:25:00 | 1653.45 | 1660.21 | 0.00 | ORB-short ORB[1661.15,1684.40] vol=5.4x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 10:30:00 | 1647.81 | 1659.28 | 0.00 | T1 1.5R @ 1647.81 |
| Stop hit — per-position SL triggered | 2025-02-21 10:35:00 | 1653.45 | 1659.16 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 11:15:00 | 1575.00 | 1567.22 | 0.00 | ORB-long ORB[1553.05,1567.90] vol=3.5x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 11:35:00 | 1579.65 | 1568.47 | 0.00 | T1 1.5R @ 1579.65 |
| Stop hit — per-position SL triggered | 2025-03-05 12:45:00 | 1575.00 | 1573.46 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 09:30:00 | 1633.85 | 1623.11 | 0.00 | ORB-long ORB[1605.35,1628.80] vol=2.1x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 09:40:00 | 1640.79 | 1628.67 | 0.00 | T1 1.5R @ 1640.79 |
| Target hit | 2025-03-11 15:20:00 | 1653.65 | 1650.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2025-03-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 10:40:00 | 1750.00 | 1757.32 | 0.00 | ORB-short ORB[1758.65,1770.80] vol=2.2x ATR=4.04 |
| Stop hit — per-position SL triggered | 2025-03-26 10:45:00 | 1754.04 | 1756.53 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-03-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 10:50:00 | 1725.35 | 1733.96 | 0.00 | ORB-short ORB[1734.10,1752.30] vol=1.5x ATR=4.78 |
| Stop hit — per-position SL triggered | 2025-03-27 12:20:00 | 1730.13 | 1728.70 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-04-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 10:25:00 | 1704.30 | 1720.59 | 0.00 | ORB-short ORB[1710.05,1733.95] vol=1.7x ATR=5.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 12:10:00 | 1695.75 | 1713.38 | 0.00 | T1 1.5R @ 1695.75 |
| Target hit | 2025-04-01 15:20:00 | 1695.70 | 1703.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2025-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 10:00:00 | 1762.10 | 1754.24 | 0.00 | ORB-long ORB[1742.10,1756.30] vol=1.6x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:10:00 | 1768.70 | 1757.16 | 0.00 | T1 1.5R @ 1768.70 |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 1762.10 | 1757.49 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 11:10:00 | 1796.70 | 1788.13 | 0.00 | ORB-long ORB[1773.90,1795.00] vol=1.8x ATR=4.11 |
| Stop hit — per-position SL triggered | 2025-04-24 11:15:00 | 1792.59 | 1788.21 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-04-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:55:00 | 1780.00 | 1792.62 | 0.00 | ORB-short ORB[1801.00,1815.50] vol=2.5x ATR=5.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 11:45:00 | 1771.35 | 1787.59 | 0.00 | T1 1.5R @ 1771.35 |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 1780.00 | 1786.45 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-04-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 10:25:00 | 1821.80 | 1803.86 | 0.00 | ORB-long ORB[1783.40,1808.00] vol=1.6x ATR=5.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 10:45:00 | 1829.74 | 1811.29 | 0.00 | T1 1.5R @ 1829.74 |
| Target hit | 2025-04-28 15:20:00 | 1843.50 | 1829.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — SELL (started 2025-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 11:05:00 | 1794.40 | 1801.49 | 0.00 | ORB-short ORB[1801.10,1826.40] vol=1.8x ATR=4.47 |
| Stop hit — per-position SL triggered | 2025-05-07 11:25:00 | 1798.87 | 1800.66 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 10:25:00 | 1533.40 | 2024-05-14 10:45:00 | 1539.35 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-05-14 10:25:00 | 1533.40 | 2024-05-14 12:20:00 | 1535.00 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-05-15 10:50:00 | 1527.75 | 2024-05-15 11:00:00 | 1531.44 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-27 10:30:00 | 1474.80 | 2024-05-27 10:40:00 | 1479.52 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-28 10:20:00 | 1456.15 | 2024-05-28 10:35:00 | 1460.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-05 11:15:00 | 1474.05 | 2024-06-05 13:25:00 | 1485.25 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-06-05 11:15:00 | 1474.05 | 2024-06-05 15:20:00 | 1486.30 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2024-06-11 11:00:00 | 1513.80 | 2024-06-11 12:00:00 | 1509.84 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-06-11 11:00:00 | 1513.80 | 2024-06-11 15:20:00 | 1499.60 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-06-13 09:35:00 | 1499.00 | 2024-06-13 10:15:00 | 1502.56 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-06-18 11:10:00 | 1512.80 | 2024-06-18 11:55:00 | 1515.33 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-06-19 10:25:00 | 1510.00 | 2024-06-19 10:40:00 | 1512.95 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-06-20 11:00:00 | 1474.55 | 2024-06-20 11:35:00 | 1477.33 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-06-24 11:10:00 | 1498.40 | 2024-06-24 14:20:00 | 1493.94 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-26 10:50:00 | 1506.75 | 2024-06-26 11:25:00 | 1503.78 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-04 10:40:00 | 1543.90 | 2024-07-04 10:45:00 | 1540.36 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-08 11:10:00 | 1561.10 | 2024-07-08 11:55:00 | 1564.64 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-10 11:00:00 | 1575.80 | 2024-07-10 11:20:00 | 1579.41 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-11 11:10:00 | 1573.50 | 2024-07-11 11:15:00 | 1576.77 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-15 11:10:00 | 1590.95 | 2024-07-15 11:45:00 | 1588.01 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-07-18 09:35:00 | 1590.40 | 2024-07-18 09:40:00 | 1587.27 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-24 10:00:00 | 1610.25 | 2024-07-24 12:35:00 | 1617.14 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-24 10:00:00 | 1610.25 | 2024-07-24 15:20:00 | 1620.95 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2024-07-25 10:50:00 | 1616.00 | 2024-07-25 11:40:00 | 1622.51 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-25 10:50:00 | 1616.00 | 2024-07-25 15:20:00 | 1658.00 | TARGET_HIT | 0.50 | 2.60% |
| SELL | retest1 | 2024-07-31 09:50:00 | 1699.50 | 2024-07-31 10:00:00 | 1703.33 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-01 09:35:00 | 1701.95 | 2024-08-01 09:40:00 | 1707.33 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-07 10:45:00 | 1732.10 | 2024-08-07 10:50:00 | 1728.31 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-08-22 11:05:00 | 1759.90 | 2024-08-22 11:50:00 | 1762.05 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2024-08-23 10:30:00 | 1764.00 | 2024-08-23 10:35:00 | 1761.08 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-08-27 11:10:00 | 1789.95 | 2024-08-27 15:20:00 | 1788.50 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest1 | 2024-08-30 11:15:00 | 1815.35 | 2024-08-30 12:00:00 | 1821.28 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-08-30 11:15:00 | 1815.35 | 2024-08-30 15:10:00 | 1815.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 09:35:00 | 1835.30 | 2024-09-03 09:50:00 | 1831.12 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-11 10:30:00 | 1857.25 | 2024-09-11 10:40:00 | 1853.20 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-16 11:10:00 | 1863.00 | 2024-09-16 11:30:00 | 1866.85 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2024-09-16 11:10:00 | 1863.00 | 2024-09-16 12:30:00 | 1863.95 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2024-09-18 11:10:00 | 1858.30 | 2024-09-18 12:10:00 | 1860.98 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-09-19 11:15:00 | 1850.10 | 2024-09-19 12:05:00 | 1846.71 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-20 10:35:00 | 1846.85 | 2024-09-20 10:40:00 | 1850.83 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-24 10:15:00 | 1870.70 | 2024-09-24 10:25:00 | 1867.50 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-09-30 10:40:00 | 1934.65 | 2024-09-30 10:45:00 | 1925.92 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-30 10:40:00 | 1934.65 | 2024-09-30 12:00:00 | 1934.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-04 10:45:00 | 1920.90 | 2024-10-04 11:50:00 | 1928.60 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-10-04 10:45:00 | 1920.90 | 2024-10-04 12:45:00 | 1928.50 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-07 11:05:00 | 1891.65 | 2024-10-07 11:15:00 | 1896.21 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-10-08 10:20:00 | 1916.05 | 2024-10-08 12:05:00 | 1911.15 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-10 11:00:00 | 1919.00 | 2024-10-10 13:00:00 | 1912.49 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-10-10 11:00:00 | 1919.00 | 2024-10-10 15:20:00 | 1886.75 | TARGET_HIT | 0.50 | 1.68% |
| SELL | retest1 | 2024-10-15 10:15:00 | 1901.25 | 2024-10-15 10:55:00 | 1905.25 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-10-21 10:55:00 | 1893.45 | 2024-10-21 11:20:00 | 1897.69 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-31 09:45:00 | 1854.95 | 2024-10-31 10:05:00 | 1845.63 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-31 09:45:00 | 1854.95 | 2024-10-31 10:25:00 | 1854.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 10:50:00 | 1755.95 | 2024-11-19 11:20:00 | 1762.16 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-11-19 10:50:00 | 1755.95 | 2024-11-19 15:20:00 | 1774.40 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2024-11-21 09:50:00 | 1786.95 | 2024-11-21 10:05:00 | 1780.72 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-26 11:15:00 | 1773.30 | 2024-11-26 11:25:00 | 1776.98 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-11-28 10:35:00 | 1732.10 | 2024-11-28 10:45:00 | 1726.04 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-11-28 10:35:00 | 1732.10 | 2024-11-28 11:15:00 | 1732.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-04 10:45:00 | 1790.05 | 2024-12-04 10:50:00 | 1793.19 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-12-05 10:55:00 | 1781.30 | 2024-12-05 11:00:00 | 1785.89 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-06 11:05:00 | 1809.90 | 2024-12-06 12:00:00 | 1813.56 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-10 11:10:00 | 1803.00 | 2024-12-10 11:15:00 | 1805.73 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-12-12 11:10:00 | 1807.70 | 2024-12-12 11:45:00 | 1803.73 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2024-12-12 11:10:00 | 1807.70 | 2024-12-12 13:35:00 | 1807.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 11:05:00 | 1779.60 | 2024-12-13 11:10:00 | 1783.73 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-16 11:10:00 | 1795.10 | 2024-12-16 11:35:00 | 1798.29 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-12-24 11:05:00 | 1829.70 | 2024-12-24 11:25:00 | 1826.77 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-12-30 10:00:00 | 1869.95 | 2024-12-30 10:20:00 | 1877.02 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-30 10:00:00 | 1869.95 | 2024-12-30 13:20:00 | 1874.75 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-01-01 11:10:00 | 1886.85 | 2025-01-01 11:15:00 | 1892.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-02 09:35:00 | 1877.30 | 2025-01-02 09:50:00 | 1871.53 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-01-02 09:35:00 | 1877.30 | 2025-01-02 09:55:00 | 1877.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 11:00:00 | 1862.70 | 2025-01-03 11:45:00 | 1857.16 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-01-03 11:00:00 | 1862.70 | 2025-01-03 15:20:00 | 1846.20 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2025-01-23 09:45:00 | 1818.00 | 2025-01-23 10:00:00 | 1824.63 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-01-23 09:45:00 | 1818.00 | 2025-01-23 15:20:00 | 1837.10 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2025-01-24 09:30:00 | 1809.40 | 2025-01-24 09:35:00 | 1814.78 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-27 10:45:00 | 1799.40 | 2025-01-27 12:10:00 | 1792.47 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-27 10:45:00 | 1799.40 | 2025-01-27 15:20:00 | 1784.85 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2025-01-29 10:20:00 | 1712.75 | 2025-01-29 10:50:00 | 1721.28 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-29 10:20:00 | 1712.75 | 2025-01-29 11:20:00 | 1712.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-14 10:55:00 | 1711.55 | 2025-02-14 11:00:00 | 1704.80 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-02-14 10:55:00 | 1711.55 | 2025-02-14 14:20:00 | 1705.00 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-02-18 10:40:00 | 1712.00 | 2025-02-18 11:20:00 | 1705.07 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-02-18 10:40:00 | 1712.00 | 2025-02-18 15:20:00 | 1699.45 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2025-02-19 10:35:00 | 1691.90 | 2025-02-19 14:00:00 | 1684.79 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-02-21 10:25:00 | 1653.45 | 2025-02-21 10:30:00 | 1647.81 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-02-21 10:25:00 | 1653.45 | 2025-02-21 10:35:00 | 1653.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-05 11:15:00 | 1575.00 | 2025-03-05 11:35:00 | 1579.65 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-03-05 11:15:00 | 1575.00 | 2025-03-05 12:45:00 | 1575.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-11 09:30:00 | 1633.85 | 2025-03-11 09:40:00 | 1640.79 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-03-11 09:30:00 | 1633.85 | 2025-03-11 15:20:00 | 1653.65 | TARGET_HIT | 0.50 | 1.21% |
| SELL | retest1 | 2025-03-26 10:40:00 | 1750.00 | 2025-03-26 10:45:00 | 1754.04 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-03-27 10:50:00 | 1725.35 | 2025-03-27 12:20:00 | 1730.13 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-04-01 10:25:00 | 1704.30 | 2025-04-01 12:10:00 | 1695.75 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-04-01 10:25:00 | 1704.30 | 2025-04-01 15:20:00 | 1695.70 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-23 10:00:00 | 1762.10 | 2025-04-23 10:10:00 | 1768.70 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-04-23 10:00:00 | 1762.10 | 2025-04-23 10:15:00 | 1762.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-24 11:10:00 | 1796.70 | 2025-04-24 11:15:00 | 1792.59 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-04-25 10:55:00 | 1780.00 | 2025-04-25 11:45:00 | 1771.35 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-04-25 10:55:00 | 1780.00 | 2025-04-25 12:15:00 | 1780.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-28 10:25:00 | 1821.80 | 2025-04-28 10:45:00 | 1829.74 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-04-28 10:25:00 | 1821.80 | 2025-04-28 15:20:00 | 1843.50 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2025-05-07 11:05:00 | 1794.40 | 2025-05-07 11:25:00 | 1798.87 | STOP_HIT | 1.00 | -0.25% |
