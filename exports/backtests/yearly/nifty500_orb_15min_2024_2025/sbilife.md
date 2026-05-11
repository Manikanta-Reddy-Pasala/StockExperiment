# SBI Life Insurance Company Ltd. (SBILIFE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1871.10
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
| ENTRY1 | 82 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 14 |
| STOP_HIT | 68 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 114 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 68
- **Target hits / Stop hits / Partials:** 14 / 68 / 32
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 11.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 24 | 38.1% | 9 | 39 | 15 | 0.12% | 7.2% |
| BUY @ 2nd Alert (retest1) | 63 | 24 | 38.1% | 9 | 39 | 15 | 0.12% | 7.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 51 | 22 | 43.1% | 5 | 29 | 17 | 0.09% | 4.7% |
| SELL @ 2nd Alert (retest1) | 51 | 22 | 43.1% | 5 | 29 | 17 | 0.09% | 4.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 114 | 46 | 40.4% | 14 | 68 | 32 | 0.10% | 11.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 11:00:00 | 1440.95 | 1436.38 | 0.00 | ORB-long ORB[1432.00,1440.55] vol=2.9x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-05-15 11:25:00 | 1437.39 | 1436.78 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:40:00 | 1447.10 | 1439.57 | 0.00 | ORB-long ORB[1424.05,1440.85] vol=2.1x ATR=3.04 |
| Stop hit — per-position SL triggered | 2024-05-23 09:45:00 | 1444.06 | 1440.14 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 11:00:00 | 1446.70 | 1440.24 | 0.00 | ORB-long ORB[1437.95,1445.95] vol=2.7x ATR=2.93 |
| Stop hit — per-position SL triggered | 2024-05-24 11:40:00 | 1443.77 | 1442.09 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:30:00 | 1432.00 | 1437.26 | 0.00 | ORB-short ORB[1433.45,1448.00] vol=3.0x ATR=3.65 |
| Stop hit — per-position SL triggered | 2024-05-27 09:35:00 | 1435.65 | 1437.21 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 10:50:00 | 1431.05 | 1420.47 | 0.00 | ORB-long ORB[1405.90,1418.05] vol=1.6x ATR=4.32 |
| Stop hit — per-position SL triggered | 2024-05-28 11:20:00 | 1426.73 | 1422.21 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:40:00 | 1378.15 | 1381.99 | 0.00 | ORB-short ORB[1382.15,1393.80] vol=2.7x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:45:00 | 1372.83 | 1381.49 | 0.00 | T1 1.5R @ 1372.83 |
| Stop hit — per-position SL triggered | 2024-05-31 11:05:00 | 1378.15 | 1381.01 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 11:05:00 | 1422.50 | 1429.92 | 0.00 | ORB-short ORB[1425.00,1436.85] vol=2.8x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-06-10 12:00:00 | 1426.52 | 1428.55 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:35:00 | 1437.60 | 1433.72 | 0.00 | ORB-long ORB[1425.80,1434.45] vol=2.7x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:50:00 | 1443.29 | 1435.04 | 0.00 | T1 1.5R @ 1443.29 |
| Target hit | 2024-06-12 15:20:00 | 1452.30 | 1447.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2024-06-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:25:00 | 1459.85 | 1468.80 | 0.00 | ORB-short ORB[1460.85,1480.00] vol=2.5x ATR=4.85 |
| Stop hit — per-position SL triggered | 2024-06-18 13:05:00 | 1464.70 | 1463.03 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:35:00 | 1456.15 | 1466.73 | 0.00 | ORB-short ORB[1467.10,1478.40] vol=1.8x ATR=3.50 |
| Stop hit — per-position SL triggered | 2024-06-19 11:25:00 | 1459.65 | 1464.45 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:00:00 | 1444.25 | 1448.50 | 0.00 | ORB-short ORB[1445.00,1457.40] vol=1.6x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-06-25 10:10:00 | 1447.44 | 1448.02 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:00:00 | 1474.70 | 1463.44 | 0.00 | ORB-long ORB[1457.60,1467.00] vol=2.4x ATR=3.23 |
| Stop hit — per-position SL triggered | 2024-06-26 11:10:00 | 1471.47 | 1463.98 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 1487.65 | 1495.72 | 0.00 | ORB-short ORB[1489.15,1510.35] vol=1.9x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 09:40:00 | 1482.16 | 1492.46 | 0.00 | T1 1.5R @ 1482.16 |
| Target hit | 2024-07-02 10:40:00 | 1484.10 | 1482.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — BUY (started 2024-07-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:45:00 | 1512.05 | 1506.80 | 0.00 | ORB-long ORB[1492.25,1509.85] vol=3.0x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:05:00 | 1518.13 | 1510.03 | 0.00 | T1 1.5R @ 1518.13 |
| Stop hit — per-position SL triggered | 2024-07-04 10:10:00 | 1512.05 | 1510.18 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:55:00 | 1569.45 | 1566.17 | 0.00 | ORB-long ORB[1557.90,1566.95] vol=2.0x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 11:10:00 | 1574.24 | 1567.32 | 0.00 | T1 1.5R @ 1574.24 |
| Stop hit — per-position SL triggered | 2024-07-15 11:25:00 | 1569.45 | 1567.60 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 10:00:00 | 1647.45 | 1632.00 | 0.00 | ORB-long ORB[1615.05,1627.25] vol=2.4x ATR=5.88 |
| Stop hit — per-position SL triggered | 2024-07-18 10:10:00 | 1641.57 | 1633.89 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-22 10:55:00 | 1632.20 | 1638.03 | 0.00 | ORB-short ORB[1635.65,1647.70] vol=1.6x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 11:45:00 | 1626.95 | 1635.25 | 0.00 | T1 1.5R @ 1626.95 |
| Target hit | 2024-07-22 15:20:00 | 1620.50 | 1629.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:30:00 | 1616.00 | 1625.55 | 0.00 | ORB-short ORB[1619.30,1638.00] vol=1.9x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 09:40:00 | 1608.79 | 1621.86 | 0.00 | T1 1.5R @ 1608.79 |
| Stop hit — per-position SL triggered | 2024-07-23 09:45:00 | 1616.00 | 1621.46 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 11:00:00 | 1747.10 | 1735.52 | 0.00 | ORB-long ORB[1720.00,1732.60] vol=1.9x ATR=4.23 |
| Stop hit — per-position SL triggered | 2024-07-31 11:05:00 | 1742.87 | 1735.77 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 10:25:00 | 1680.40 | 1684.86 | 0.00 | ORB-short ORB[1683.40,1698.90] vol=2.6x ATR=5.48 |
| Stop hit — per-position SL triggered | 2024-08-07 10:30:00 | 1685.88 | 1684.93 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 11:00:00 | 1727.25 | 1719.92 | 0.00 | ORB-long ORB[1708.00,1722.90] vol=2.1x ATR=4.94 |
| Stop hit — per-position SL triggered | 2024-08-09 15:20:00 | 1723.15 | 1725.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2024-08-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 11:00:00 | 1686.90 | 1692.26 | 0.00 | ORB-short ORB[1699.40,1721.30] vol=1.6x ATR=5.16 |
| Stop hit — per-position SL triggered | 2024-08-13 11:25:00 | 1692.06 | 1691.50 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 11:05:00 | 1696.40 | 1687.33 | 0.00 | ORB-long ORB[1673.00,1696.20] vol=2.7x ATR=4.45 |
| Stop hit — per-position SL triggered | 2024-08-14 11:15:00 | 1691.95 | 1687.99 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 09:40:00 | 1680.65 | 1691.24 | 0.00 | ORB-short ORB[1692.70,1702.10] vol=1.6x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 10:05:00 | 1673.44 | 1687.25 | 0.00 | T1 1.5R @ 1673.44 |
| Stop hit — per-position SL triggered | 2024-08-16 11:05:00 | 1680.65 | 1680.22 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 11:00:00 | 1666.95 | 1681.97 | 0.00 | ORB-short ORB[1683.65,1696.80] vol=2.4x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-08-19 11:10:00 | 1670.59 | 1681.25 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 1698.35 | 1683.69 | 0.00 | ORB-long ORB[1672.75,1689.20] vol=2.1x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:00:00 | 1705.15 | 1694.25 | 0.00 | T1 1.5R @ 1705.15 |
| Target hit | 2024-08-20 15:20:00 | 1764.85 | 1741.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-08-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:10:00 | 1782.05 | 1777.50 | 0.00 | ORB-long ORB[1751.20,1767.35] vol=2.2x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 12:30:00 | 1791.47 | 1780.54 | 0.00 | T1 1.5R @ 1791.47 |
| Target hit | 2024-08-21 15:20:00 | 1796.85 | 1789.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2024-08-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:55:00 | 1807.50 | 1800.17 | 0.00 | ORB-long ORB[1792.30,1804.95] vol=1.6x ATR=4.24 |
| Stop hit — per-position SL triggered | 2024-08-22 11:10:00 | 1803.26 | 1802.68 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:55:00 | 1804.45 | 1794.31 | 0.00 | ORB-long ORB[1787.40,1801.15] vol=2.2x ATR=4.89 |
| Stop hit — per-position SL triggered | 2024-08-27 10:25:00 | 1799.56 | 1799.90 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:55:00 | 1851.95 | 1844.97 | 0.00 | ORB-long ORB[1835.50,1848.60] vol=1.6x ATR=4.18 |
| Stop hit — per-position SL triggered | 2024-08-29 10:20:00 | 1847.77 | 1846.14 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:45:00 | 1859.75 | 1849.68 | 0.00 | ORB-long ORB[1837.00,1849.00] vol=1.6x ATR=5.10 |
| Stop hit — per-position SL triggered | 2024-08-30 11:30:00 | 1854.65 | 1854.54 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:45:00 | 1870.30 | 1869.25 | 0.00 | ORB-long ORB[1849.05,1864.30] vol=3.1x ATR=5.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 11:00:00 | 1879.08 | 1870.16 | 0.00 | T1 1.5R @ 1879.08 |
| Stop hit — per-position SL triggered | 2024-09-02 11:30:00 | 1870.30 | 1870.30 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:55:00 | 1900.00 | 1895.97 | 0.00 | ORB-long ORB[1881.75,1893.05] vol=2.7x ATR=5.54 |
| Stop hit — per-position SL triggered | 2024-09-03 13:00:00 | 1894.46 | 1898.52 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:35:00 | 1893.15 | 1901.52 | 0.00 | ORB-short ORB[1897.05,1913.80] vol=1.7x ATR=4.88 |
| Stop hit — per-position SL triggered | 2024-09-06 10:45:00 | 1898.03 | 1900.71 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 10:45:00 | 1930.70 | 1916.64 | 0.00 | ORB-long ORB[1895.05,1923.50] vol=1.7x ATR=6.55 |
| Stop hit — per-position SL triggered | 2024-09-09 11:00:00 | 1924.15 | 1917.99 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 10:15:00 | 1855.05 | 1858.03 | 0.00 | ORB-short ORB[1860.15,1871.75] vol=1.6x ATR=5.05 |
| Stop hit — per-position SL triggered | 2024-09-13 10:20:00 | 1860.10 | 1858.01 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:50:00 | 1838.35 | 1831.40 | 0.00 | ORB-long ORB[1815.05,1831.50] vol=2.8x ATR=5.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:40:00 | 1846.14 | 1835.21 | 0.00 | T1 1.5R @ 1846.14 |
| Stop hit — per-position SL triggered | 2024-09-18 13:00:00 | 1838.35 | 1840.09 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:45:00 | 1858.85 | 1854.57 | 0.00 | ORB-long ORB[1846.10,1855.80] vol=2.2x ATR=5.32 |
| Stop hit — per-position SL triggered | 2024-09-20 10:25:00 | 1853.53 | 1855.67 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:10:00 | 1900.55 | 1895.00 | 0.00 | ORB-long ORB[1880.00,1895.90] vol=1.8x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 13:20:00 | 1906.83 | 1897.78 | 0.00 | T1 1.5R @ 1906.83 |
| Target hit | 2024-09-23 15:20:00 | 1921.05 | 1908.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2024-09-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 11:05:00 | 1911.90 | 1897.86 | 0.00 | ORB-long ORB[1887.40,1900.85] vol=1.9x ATR=4.34 |
| Stop hit — per-position SL triggered | 2024-09-27 11:35:00 | 1907.56 | 1902.10 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:55:00 | 1869.50 | 1879.85 | 0.00 | ORB-short ORB[1874.10,1893.60] vol=1.7x ATR=5.37 |
| Stop hit — per-position SL triggered | 2024-09-30 11:20:00 | 1874.87 | 1877.10 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 09:35:00 | 1827.95 | 1822.62 | 0.00 | ORB-long ORB[1803.10,1824.50] vol=2.4x ATR=7.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:45:00 | 1838.78 | 1827.94 | 0.00 | T1 1.5R @ 1838.78 |
| Target hit | 2024-10-04 10:00:00 | 1829.00 | 1832.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — SELL (started 2024-10-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:40:00 | 1793.60 | 1799.28 | 0.00 | ORB-short ORB[1794.25,1819.85] vol=1.7x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 11:00:00 | 1784.89 | 1797.61 | 0.00 | T1 1.5R @ 1784.89 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 1793.60 | 1797.02 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-08 09:30:00 | 1769.80 | 1782.84 | 0.00 | ORB-short ORB[1774.00,1799.05] vol=1.6x ATR=7.06 |
| Stop hit — per-position SL triggered | 2024-10-08 09:50:00 | 1776.86 | 1777.05 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:50:00 | 1724.35 | 1728.33 | 0.00 | ORB-short ORB[1729.30,1740.05] vol=11.3x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 11:20:00 | 1719.08 | 1727.43 | 0.00 | T1 1.5R @ 1719.08 |
| Stop hit — per-position SL triggered | 2024-10-11 11:30:00 | 1724.35 | 1727.34 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:55:00 | 1734.65 | 1740.40 | 0.00 | ORB-short ORB[1738.95,1753.05] vol=1.7x ATR=5.12 |
| Stop hit — per-position SL triggered | 2024-10-15 10:00:00 | 1739.77 | 1738.56 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:00:00 | 1711.60 | 1723.41 | 0.00 | ORB-short ORB[1717.10,1739.00] vol=1.9x ATR=5.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:10:00 | 1703.82 | 1719.03 | 0.00 | T1 1.5R @ 1703.82 |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 1711.60 | 1718.98 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 11:15:00 | 1716.70 | 1707.75 | 0.00 | ORB-long ORB[1693.45,1712.00] vol=2.4x ATR=4.94 |
| Stop hit — per-position SL triggered | 2024-10-23 11:55:00 | 1711.76 | 1708.60 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-10-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:50:00 | 1617.75 | 1635.66 | 0.00 | ORB-short ORB[1628.30,1650.95] vol=2.4x ATR=5.81 |
| Stop hit — per-position SL triggered | 2024-10-25 10:55:00 | 1623.56 | 1633.04 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-10-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 11:10:00 | 1625.70 | 1610.75 | 0.00 | ORB-long ORB[1595.90,1618.85] vol=1.6x ATR=6.90 |
| Stop hit — per-position SL triggered | 2024-10-29 11:20:00 | 1618.80 | 1611.67 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-30 11:10:00 | 1628.50 | 1636.35 | 0.00 | ORB-short ORB[1632.00,1649.35] vol=2.7x ATR=4.56 |
| Stop hit — per-position SL triggered | 2024-10-30 12:10:00 | 1633.06 | 1635.18 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-11-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 10:45:00 | 1492.25 | 1496.27 | 0.00 | ORB-short ORB[1500.60,1515.50] vol=4.3x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 11:35:00 | 1484.80 | 1495.18 | 0.00 | T1 1.5R @ 1484.80 |
| Target hit | 2024-11-21 15:20:00 | 1474.95 | 1484.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2024-12-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 09:50:00 | 1425.75 | 1428.68 | 0.00 | ORB-short ORB[1426.25,1438.45] vol=2.5x ATR=4.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:40:00 | 1419.45 | 1427.55 | 0.00 | T1 1.5R @ 1419.45 |
| Stop hit — per-position SL triggered | 2024-12-03 11:55:00 | 1425.75 | 1425.49 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:25:00 | 1432.40 | 1437.31 | 0.00 | ORB-short ORB[1441.85,1456.05] vol=2.7x ATR=3.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 10:55:00 | 1426.80 | 1433.37 | 0.00 | T1 1.5R @ 1426.80 |
| Stop hit — per-position SL triggered | 2024-12-05 12:10:00 | 1432.40 | 1430.21 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 11:15:00 | 1430.00 | 1437.34 | 0.00 | ORB-short ORB[1431.05,1442.55] vol=1.9x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-12-06 12:00:00 | 1433.35 | 1435.92 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 11:00:00 | 1470.50 | 1463.27 | 0.00 | ORB-long ORB[1449.05,1465.95] vol=2.4x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 11:40:00 | 1476.02 | 1467.39 | 0.00 | T1 1.5R @ 1476.02 |
| Target hit | 2024-12-09 13:35:00 | 1472.35 | 1472.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 57 — BUY (started 2024-12-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:35:00 | 1467.20 | 1461.61 | 0.00 | ORB-long ORB[1454.45,1463.90] vol=2.7x ATR=3.10 |
| Stop hit — per-position SL triggered | 2024-12-11 11:10:00 | 1464.10 | 1463.89 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:00:00 | 1421.00 | 1428.42 | 0.00 | ORB-short ORB[1422.05,1441.95] vol=1.8x ATR=5.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:30:00 | 1412.81 | 1426.26 | 0.00 | T1 1.5R @ 1412.81 |
| Stop hit — per-position SL triggered | 2024-12-13 11:55:00 | 1421.00 | 1423.25 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 11:15:00 | 1403.65 | 1410.52 | 0.00 | ORB-short ORB[1405.35,1419.00] vol=3.8x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 11:35:00 | 1399.31 | 1408.93 | 0.00 | T1 1.5R @ 1399.31 |
| Target hit | 2024-12-18 15:20:00 | 1398.45 | 1402.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2024-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:35:00 | 1401.00 | 1396.88 | 0.00 | ORB-long ORB[1385.20,1397.85] vol=2.0x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:00:00 | 1406.27 | 1398.63 | 0.00 | T1 1.5R @ 1406.27 |
| Target hit | 2024-12-26 12:00:00 | 1402.90 | 1402.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — BUY (started 2025-01-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:30:00 | 1459.00 | 1448.41 | 0.00 | ORB-long ORB[1434.35,1450.00] vol=3.2x ATR=5.05 |
| Stop hit — per-position SL triggered | 2025-01-07 10:40:00 | 1453.95 | 1448.87 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 1466.30 | 1461.34 | 0.00 | ORB-long ORB[1453.95,1462.00] vol=2.4x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-01-09 10:55:00 | 1462.09 | 1461.66 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-10 11:05:00 | 1475.75 | 1466.69 | 0.00 | ORB-long ORB[1460.05,1475.25] vol=2.4x ATR=3.79 |
| Stop hit — per-position SL triggered | 2025-01-10 11:40:00 | 1471.96 | 1468.96 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:00:00 | 1500.70 | 1482.02 | 0.00 | ORB-long ORB[1471.30,1487.00] vol=1.7x ATR=4.93 |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 1495.77 | 1484.03 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:50:00 | 1480.95 | 1481.75 | 0.00 | ORB-short ORB[1486.25,1504.85] vol=5.5x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 10:05:00 | 1474.15 | 1480.72 | 0.00 | T1 1.5R @ 1474.15 |
| Stop hit — per-position SL triggered | 2025-01-15 10:55:00 | 1480.95 | 1479.03 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:35:00 | 1519.55 | 1512.35 | 0.00 | ORB-long ORB[1500.00,1519.50] vol=1.6x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:50:00 | 1528.30 | 1516.83 | 0.00 | T1 1.5R @ 1528.30 |
| Stop hit — per-position SL triggered | 2025-01-17 13:30:00 | 1519.55 | 1526.40 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:20:00 | 1461.15 | 1455.55 | 0.00 | ORB-long ORB[1448.65,1459.00] vol=2.4x ATR=3.67 |
| Stop hit — per-position SL triggered | 2025-01-23 11:20:00 | 1457.48 | 1456.84 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:00:00 | 1445.50 | 1434.26 | 0.00 | ORB-long ORB[1410.50,1429.95] vol=5.2x ATR=3.55 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 1441.95 | 1436.71 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-01-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 11:05:00 | 1471.55 | 1465.96 | 0.00 | ORB-long ORB[1453.90,1471.40] vol=2.0x ATR=3.90 |
| Stop hit — per-position SL triggered | 2025-01-30 11:20:00 | 1467.65 | 1466.35 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:10:00 | 1466.65 | 1484.67 | 0.00 | ORB-short ORB[1470.55,1491.55] vol=1.7x ATR=5.21 |
| Stop hit — per-position SL triggered | 2025-02-01 11:15:00 | 1471.86 | 1483.03 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:55:00 | 1454.55 | 1463.19 | 0.00 | ORB-short ORB[1462.90,1474.55] vol=1.7x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 11:15:00 | 1449.26 | 1461.81 | 0.00 | T1 1.5R @ 1449.26 |
| Stop hit — per-position SL triggered | 2025-02-10 11:30:00 | 1454.55 | 1461.10 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-12 10:55:00 | 1437.80 | 1419.41 | 0.00 | ORB-long ORB[1411.65,1427.45] vol=3.7x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-02-12 11:00:00 | 1433.01 | 1420.18 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 10:15:00 | 1463.50 | 1459.68 | 0.00 | ORB-long ORB[1446.00,1457.50] vol=1.9x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 10:50:00 | 1470.03 | 1462.23 | 0.00 | T1 1.5R @ 1470.03 |
| Target hit | 2025-02-13 15:10:00 | 1470.00 | 1470.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — SELL (started 2025-02-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:50:00 | 1463.30 | 1467.49 | 0.00 | ORB-short ORB[1466.55,1477.15] vol=2.3x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:00:00 | 1457.87 | 1466.87 | 0.00 | T1 1.5R @ 1457.87 |
| Target hit | 2025-02-14 14:15:00 | 1460.50 | 1457.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 75 — BUY (started 2025-03-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 10:00:00 | 1410.70 | 1403.05 | 0.00 | ORB-long ORB[1390.70,1405.75] vol=2.1x ATR=5.04 |
| Stop hit — per-position SL triggered | 2025-03-04 13:05:00 | 1405.66 | 1407.23 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-03-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:00:00 | 1422.00 | 1412.56 | 0.00 | ORB-long ORB[1400.00,1412.45] vol=2.3x ATR=5.60 |
| Stop hit — per-position SL triggered | 2025-03-11 11:15:00 | 1416.40 | 1416.33 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:40:00 | 1469.20 | 1463.61 | 0.00 | ORB-long ORB[1455.00,1465.10] vol=1.8x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:15:00 | 1473.39 | 1465.28 | 0.00 | T1 1.5R @ 1473.39 |
| Stop hit — per-position SL triggered | 2025-03-19 10:45:00 | 1469.20 | 1466.55 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-03-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:10:00 | 1509.20 | 1505.13 | 0.00 | ORB-long ORB[1493.45,1507.75] vol=5.7x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:45:00 | 1514.59 | 1506.57 | 0.00 | T1 1.5R @ 1514.59 |
| Target hit | 2025-03-21 15:20:00 | 1548.70 | 1532.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:15:00 | 1549.35 | 1539.29 | 0.00 | ORB-long ORB[1529.65,1545.00] vol=1.9x ATR=3.75 |
| Stop hit — per-position SL triggered | 2025-03-27 11:20:00 | 1545.60 | 1539.40 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:15:00 | 1608.00 | 1610.96 | 0.00 | ORB-short ORB[1608.60,1625.00] vol=1.6x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:40:00 | 1602.75 | 1609.49 | 0.00 | T1 1.5R @ 1602.75 |
| Stop hit — per-position SL triggered | 2025-04-23 11:15:00 | 1608.00 | 1608.40 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 11:00:00 | 1626.50 | 1622.78 | 0.00 | ORB-long ORB[1603.00,1623.20] vol=2.4x ATR=4.95 |
| Stop hit — per-position SL triggered | 2025-04-24 11:20:00 | 1621.55 | 1622.84 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-04-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 10:05:00 | 1747.20 | 1740.61 | 0.00 | ORB-long ORB[1718.50,1737.40] vol=1.9x ATR=4.95 |
| Stop hit — per-position SL triggered | 2025-04-30 10:25:00 | 1742.25 | 1742.09 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 11:00:00 | 1440.95 | 2024-05-15 11:25:00 | 1437.39 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-23 09:40:00 | 1447.10 | 2024-05-23 09:45:00 | 1444.06 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-05-24 11:00:00 | 1446.70 | 2024-05-24 11:40:00 | 1443.77 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-05-27 09:30:00 | 1432.00 | 2024-05-27 09:35:00 | 1435.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-28 10:50:00 | 1431.05 | 2024-05-28 11:20:00 | 1426.73 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-31 10:40:00 | 1378.15 | 2024-05-31 10:45:00 | 1372.83 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-05-31 10:40:00 | 1378.15 | 2024-05-31 11:05:00 | 1378.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-10 11:05:00 | 1422.50 | 2024-06-10 12:00:00 | 1426.52 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-12 10:35:00 | 1437.60 | 2024-06-12 10:50:00 | 1443.29 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-06-12 10:35:00 | 1437.60 | 2024-06-12 15:20:00 | 1452.30 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2024-06-18 10:25:00 | 1459.85 | 2024-06-18 13:05:00 | 1464.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-19 10:35:00 | 1456.15 | 2024-06-19 11:25:00 | 1459.65 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-06-25 10:00:00 | 1444.25 | 2024-06-25 10:10:00 | 1447.44 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-06-26 11:00:00 | 1474.70 | 2024-06-26 11:10:00 | 1471.47 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-02 09:30:00 | 1487.65 | 2024-07-02 09:40:00 | 1482.16 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-07-02 09:30:00 | 1487.65 | 2024-07-02 10:40:00 | 1484.10 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2024-07-04 09:45:00 | 1512.05 | 2024-07-04 10:05:00 | 1518.13 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-04 09:45:00 | 1512.05 | 2024-07-04 10:10:00 | 1512.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-15 10:55:00 | 1569.45 | 2024-07-15 11:10:00 | 1574.24 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-07-15 10:55:00 | 1569.45 | 2024-07-15 11:25:00 | 1569.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-18 10:00:00 | 1647.45 | 2024-07-18 10:10:00 | 1641.57 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-22 10:55:00 | 1632.20 | 2024-07-22 11:45:00 | 1626.95 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-07-22 10:55:00 | 1632.20 | 2024-07-22 15:20:00 | 1620.50 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2024-07-23 09:30:00 | 1616.00 | 2024-07-23 09:40:00 | 1608.79 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-23 09:30:00 | 1616.00 | 2024-07-23 09:45:00 | 1616.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 11:00:00 | 1747.10 | 2024-07-31 11:05:00 | 1742.87 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-07 10:25:00 | 1680.40 | 2024-08-07 10:30:00 | 1685.88 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-09 11:00:00 | 1727.25 | 2024-08-09 15:20:00 | 1723.15 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-13 11:00:00 | 1686.90 | 2024-08-13 11:25:00 | 1692.06 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-14 11:05:00 | 1696.40 | 2024-08-14 11:15:00 | 1691.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-16 09:40:00 | 1680.65 | 2024-08-16 10:05:00 | 1673.44 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-16 09:40:00 | 1680.65 | 2024-08-16 11:05:00 | 1680.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-19 11:00:00 | 1666.95 | 2024-08-19 11:10:00 | 1670.59 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-20 09:30:00 | 1698.35 | 2024-08-20 10:00:00 | 1705.15 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-20 09:30:00 | 1698.35 | 2024-08-20 15:20:00 | 1764.85 | TARGET_HIT | 0.50 | 3.92% |
| BUY | retest1 | 2024-08-21 11:10:00 | 1782.05 | 2024-08-21 12:30:00 | 1791.47 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-08-21 11:10:00 | 1782.05 | 2024-08-21 15:20:00 | 1796.85 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2024-08-22 10:55:00 | 1807.50 | 2024-08-22 11:10:00 | 1803.26 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-27 09:55:00 | 1804.45 | 2024-08-27 10:25:00 | 1799.56 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-29 09:55:00 | 1851.95 | 2024-08-29 10:20:00 | 1847.77 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-30 10:45:00 | 1859.75 | 2024-08-30 11:30:00 | 1854.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-02 10:45:00 | 1870.30 | 2024-09-02 11:00:00 | 1879.08 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-09-02 10:45:00 | 1870.30 | 2024-09-02 11:30:00 | 1870.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 09:55:00 | 1900.00 | 2024-09-03 13:00:00 | 1894.46 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-06 10:35:00 | 1893.15 | 2024-09-06 10:45:00 | 1898.03 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-09 10:45:00 | 1930.70 | 2024-09-09 11:00:00 | 1924.15 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-13 10:15:00 | 1855.05 | 2024-09-13 10:20:00 | 1860.10 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-18 09:50:00 | 1838.35 | 2024-09-18 10:40:00 | 1846.14 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-09-18 09:50:00 | 1838.35 | 2024-09-18 13:00:00 | 1838.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-20 09:45:00 | 1858.85 | 2024-09-20 10:25:00 | 1853.53 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-23 11:10:00 | 1900.55 | 2024-09-23 13:20:00 | 1906.83 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-09-23 11:10:00 | 1900.55 | 2024-09-23 15:20:00 | 1921.05 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2024-09-27 11:05:00 | 1911.90 | 2024-09-27 11:35:00 | 1907.56 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-30 10:55:00 | 1869.50 | 2024-09-30 11:20:00 | 1874.87 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-04 09:35:00 | 1827.95 | 2024-10-04 09:45:00 | 1838.78 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-10-04 09:35:00 | 1827.95 | 2024-10-04 10:00:00 | 1829.00 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2024-10-07 10:40:00 | 1793.60 | 2024-10-07 11:00:00 | 1784.89 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-07 10:40:00 | 1793.60 | 2024-10-07 11:15:00 | 1793.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-08 09:30:00 | 1769.80 | 2024-10-08 09:50:00 | 1776.86 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-11 10:50:00 | 1724.35 | 2024-10-11 11:20:00 | 1719.08 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-10-11 10:50:00 | 1724.35 | 2024-10-11 11:30:00 | 1724.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-15 09:55:00 | 1734.65 | 2024-10-15 10:00:00 | 1739.77 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-17 10:00:00 | 1711.60 | 2024-10-17 10:10:00 | 1703.82 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-10-17 10:00:00 | 1711.60 | 2024-10-17 10:15:00 | 1711.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-23 11:15:00 | 1716.70 | 2024-10-23 11:55:00 | 1711.76 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-25 10:50:00 | 1617.75 | 2024-10-25 10:55:00 | 1623.56 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-29 11:10:00 | 1625.70 | 2024-10-29 11:20:00 | 1618.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-30 11:10:00 | 1628.50 | 2024-10-30 12:10:00 | 1633.06 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-21 10:45:00 | 1492.25 | 2024-11-21 11:35:00 | 1484.80 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-11-21 10:45:00 | 1492.25 | 2024-11-21 15:20:00 | 1474.95 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2024-12-03 09:50:00 | 1425.75 | 2024-12-03 10:40:00 | 1419.45 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-03 09:50:00 | 1425.75 | 2024-12-03 11:55:00 | 1425.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 10:25:00 | 1432.40 | 2024-12-05 10:55:00 | 1426.80 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-12-05 10:25:00 | 1432.40 | 2024-12-05 12:10:00 | 1432.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-06 11:15:00 | 1430.00 | 2024-12-06 12:00:00 | 1433.35 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-09 11:00:00 | 1470.50 | 2024-12-09 11:40:00 | 1476.02 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-09 11:00:00 | 1470.50 | 2024-12-09 13:35:00 | 1472.35 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-12-11 10:35:00 | 1467.20 | 2024-12-11 11:10:00 | 1464.10 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-12-13 10:00:00 | 1421.00 | 2024-12-13 10:30:00 | 1412.81 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-12-13 10:00:00 | 1421.00 | 2024-12-13 11:55:00 | 1421.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-18 11:15:00 | 1403.65 | 2024-12-18 11:35:00 | 1399.31 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-12-18 11:15:00 | 1403.65 | 2024-12-18 15:20:00 | 1398.45 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2024-12-26 09:35:00 | 1401.00 | 2024-12-26 10:00:00 | 1406.27 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-26 09:35:00 | 1401.00 | 2024-12-26 12:00:00 | 1402.90 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2025-01-07 10:30:00 | 1459.00 | 2025-01-07 10:40:00 | 1453.95 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-09 10:45:00 | 1466.30 | 2025-01-09 10:55:00 | 1462.09 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-10 11:05:00 | 1475.75 | 2025-01-10 11:40:00 | 1471.96 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-14 11:00:00 | 1500.70 | 2025-01-14 11:15:00 | 1495.77 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-15 09:50:00 | 1480.95 | 2025-01-15 10:05:00 | 1474.15 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-01-15 09:50:00 | 1480.95 | 2025-01-15 10:55:00 | 1480.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-17 09:35:00 | 1519.55 | 2025-01-17 09:50:00 | 1528.30 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-01-17 09:35:00 | 1519.55 | 2025-01-17 13:30:00 | 1519.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 10:20:00 | 1461.15 | 2025-01-23 11:20:00 | 1457.48 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-29 11:00:00 | 1445.50 | 2025-01-29 11:20:00 | 1441.95 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-30 11:05:00 | 1471.55 | 2025-01-30 11:20:00 | 1467.65 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-02-01 11:10:00 | 1466.65 | 2025-02-01 11:15:00 | 1471.86 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-02-10 10:55:00 | 1454.55 | 2025-02-10 11:15:00 | 1449.26 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-02-10 10:55:00 | 1454.55 | 2025-02-10 11:30:00 | 1454.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-12 10:55:00 | 1437.80 | 2025-02-12 11:00:00 | 1433.01 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-13 10:15:00 | 1463.50 | 2025-02-13 10:50:00 | 1470.03 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-02-13 10:15:00 | 1463.50 | 2025-02-13 15:10:00 | 1470.00 | TARGET_HIT | 0.50 | 0.44% |
| SELL | retest1 | 2025-02-14 10:50:00 | 1463.30 | 2025-02-14 11:00:00 | 1457.87 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-02-14 10:50:00 | 1463.30 | 2025-02-14 14:15:00 | 1460.50 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2025-03-04 10:00:00 | 1410.70 | 2025-03-04 13:05:00 | 1405.66 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-11 10:00:00 | 1422.00 | 2025-03-11 11:15:00 | 1416.40 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-19 09:40:00 | 1469.20 | 2025-03-19 10:15:00 | 1473.39 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-03-19 09:40:00 | 1469.20 | 2025-03-19 10:45:00 | 1469.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 10:10:00 | 1509.20 | 2025-03-21 10:45:00 | 1514.59 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-03-21 10:10:00 | 1509.20 | 2025-03-21 15:20:00 | 1548.70 | TARGET_HIT | 0.50 | 2.62% |
| BUY | retest1 | 2025-03-27 11:15:00 | 1549.35 | 2025-03-27 11:20:00 | 1545.60 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-04-23 10:15:00 | 1608.00 | 2025-04-23 10:40:00 | 1602.75 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-04-23 10:15:00 | 1608.00 | 2025-04-23 11:15:00 | 1608.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-24 11:00:00 | 1626.50 | 2025-04-24 11:20:00 | 1621.55 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-04-30 10:05:00 | 1747.20 | 2025-04-30 10:25:00 | 1742.25 | STOP_HIT | 1.00 | -0.28% |
