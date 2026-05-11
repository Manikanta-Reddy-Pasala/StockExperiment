# Colgate Palmolive (India) Ltd. (COLPAL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55354 bars)
- **Last close:** 2193.70
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
| ENTRY1 | 104 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 12 |
| STOP_HIT | 92 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 92
- **Target hits / Stop hits / Partials:** 12 / 92 / 33
- **Avg / median % per leg:** 0.02% / -0.19%
- **Sum % (uncompounded):** 2.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 24 | 31.2% | 7 | 53 | 17 | -0.00% | -0.3% |
| BUY @ 2nd Alert (retest1) | 77 | 24 | 31.2% | 7 | 53 | 17 | -0.00% | -0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 60 | 21 | 35.0% | 5 | 39 | 16 | 0.04% | 2.7% |
| SELL @ 2nd Alert (retest1) | 60 | 21 | 35.0% | 5 | 39 | 16 | 0.04% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 137 | 45 | 32.8% | 12 | 92 | 33 | 0.02% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 11:15:00 | 1635.45 | 1632.91 | 0.00 | ORB-long ORB[1616.65,1628.45] vol=3.3x ATR=5.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-12 11:55:00 | 1643.66 | 1634.15 | 0.00 | T1 1.5R @ 1643.66 |
| Stop hit — per-position SL triggered | 2023-05-12 12:00:00 | 1635.45 | 1634.31 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 09:35:00 | 1665.15 | 1652.25 | 0.00 | ORB-long ORB[1640.70,1659.85] vol=2.2x ATR=6.53 |
| Stop hit — per-position SL triggered | 2023-05-17 10:05:00 | 1658.62 | 1655.56 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 10:05:00 | 1657.50 | 1672.58 | 0.00 | ORB-short ORB[1665.00,1687.00] vol=2.1x ATR=5.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-18 10:35:00 | 1649.45 | 1667.45 | 0.00 | T1 1.5R @ 1649.45 |
| Stop hit — per-position SL triggered | 2023-05-18 10:45:00 | 1657.50 | 1666.74 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 10:50:00 | 1613.50 | 1602.36 | 0.00 | ORB-long ORB[1595.50,1608.75] vol=2.6x ATR=4.35 |
| Stop hit — per-position SL triggered | 2023-05-22 11:05:00 | 1609.15 | 1604.76 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-05-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-23 09:50:00 | 1601.15 | 1606.57 | 0.00 | ORB-short ORB[1602.15,1618.50] vol=1.6x ATR=3.83 |
| Stop hit — per-position SL triggered | 2023-05-23 09:55:00 | 1604.98 | 1606.38 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 10:20:00 | 1601.20 | 1597.16 | 0.00 | ORB-long ORB[1585.95,1600.50] vol=2.5x ATR=3.68 |
| Stop hit — per-position SL triggered | 2023-05-24 10:40:00 | 1597.52 | 1597.96 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-05-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 11:10:00 | 1572.20 | 1580.02 | 0.00 | ORB-short ORB[1576.70,1598.75] vol=4.4x ATR=3.33 |
| Stop hit — per-position SL triggered | 2023-05-25 13:15:00 | 1575.53 | 1576.21 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-05-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 11:05:00 | 1589.30 | 1596.52 | 0.00 | ORB-short ORB[1589.40,1600.00] vol=2.2x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 11:10:00 | 1584.68 | 1594.87 | 0.00 | T1 1.5R @ 1584.68 |
| Stop hit — per-position SL triggered | 2023-05-31 15:00:00 | 1589.30 | 1583.67 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 09:55:00 | 1607.00 | 1612.93 | 0.00 | ORB-short ORB[1608.00,1619.75] vol=1.7x ATR=3.74 |
| Stop hit — per-position SL triggered | 2023-06-02 10:45:00 | 1610.74 | 1610.58 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 09:45:00 | 1637.00 | 1632.05 | 0.00 | ORB-long ORB[1625.10,1635.15] vol=2.3x ATR=3.66 |
| Stop hit — per-position SL triggered | 2023-06-07 10:10:00 | 1633.34 | 1634.46 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 10:20:00 | 1607.05 | 1616.18 | 0.00 | ORB-short ORB[1616.05,1632.65] vol=2.1x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 12:45:00 | 1600.73 | 1610.49 | 0.00 | T1 1.5R @ 1600.73 |
| Target hit | 2023-06-09 15:20:00 | 1596.50 | 1604.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2023-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 09:30:00 | 1652.55 | 1643.53 | 0.00 | ORB-long ORB[1625.00,1647.00] vol=2.5x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 09:50:00 | 1658.90 | 1649.08 | 0.00 | T1 1.5R @ 1658.90 |
| Target hit | 2023-06-13 13:00:00 | 1654.25 | 1655.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2023-06-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:35:00 | 1658.10 | 1649.33 | 0.00 | ORB-long ORB[1640.15,1653.95] vol=1.6x ATR=5.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-16 13:20:00 | 1665.86 | 1657.49 | 0.00 | T1 1.5R @ 1665.86 |
| Stop hit — per-position SL triggered | 2023-06-16 14:40:00 | 1658.10 | 1658.77 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-06-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 10:30:00 | 1644.10 | 1649.82 | 0.00 | ORB-short ORB[1645.05,1659.90] vol=3.1x ATR=3.15 |
| Stop hit — per-position SL triggered | 2023-06-20 10:45:00 | 1647.25 | 1649.62 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-06-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 10:00:00 | 1676.95 | 1667.14 | 0.00 | ORB-long ORB[1647.05,1669.30] vol=1.7x ATR=4.73 |
| Stop hit — per-position SL triggered | 2023-06-21 10:20:00 | 1672.22 | 1672.05 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-06-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 10:00:00 | 1682.05 | 1675.13 | 0.00 | ORB-long ORB[1668.45,1682.00] vol=1.6x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-27 10:40:00 | 1687.74 | 1677.64 | 0.00 | T1 1.5R @ 1687.74 |
| Target hit | 2023-06-27 15:20:00 | 1691.05 | 1685.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2023-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 09:30:00 | 1683.00 | 1685.67 | 0.00 | ORB-short ORB[1683.20,1692.20] vol=1.5x ATR=3.51 |
| Stop hit — per-position SL triggered | 2023-07-03 09:45:00 | 1686.51 | 1685.34 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:35:00 | 1751.00 | 1738.38 | 0.00 | ORB-long ORB[1713.90,1734.90] vol=7.0x ATR=9.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 09:40:00 | 1765.33 | 1743.54 | 0.00 | T1 1.5R @ 1765.33 |
| Stop hit — per-position SL triggered | 2023-07-05 09:45:00 | 1751.00 | 1742.28 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 09:50:00 | 1760.50 | 1765.68 | 0.00 | ORB-short ORB[1760.85,1787.10] vol=1.7x ATR=6.86 |
| Stop hit — per-position SL triggered | 2023-07-10 10:45:00 | 1767.36 | 1763.82 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 11:15:00 | 1784.00 | 1794.45 | 0.00 | ORB-short ORB[1785.25,1808.80] vol=2.2x ATR=3.96 |
| Stop hit — per-position SL triggered | 2023-07-12 11:50:00 | 1787.96 | 1793.20 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 10:35:00 | 1827.85 | 1818.31 | 0.00 | ORB-long ORB[1811.60,1824.00] vol=2.4x ATR=4.78 |
| Stop hit — per-position SL triggered | 2023-07-14 10:45:00 | 1823.07 | 1819.38 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 09:30:00 | 1856.25 | 1850.38 | 0.00 | ORB-long ORB[1842.10,1856.20] vol=2.9x ATR=4.71 |
| Stop hit — per-position SL triggered | 2023-07-18 09:35:00 | 1851.54 | 1850.47 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 11:15:00 | 1835.05 | 1841.25 | 0.00 | ORB-short ORB[1836.05,1848.80] vol=2.8x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 11:20:00 | 1829.70 | 1839.97 | 0.00 | T1 1.5R @ 1829.70 |
| Stop hit — per-position SL triggered | 2023-07-19 11:40:00 | 1835.05 | 1839.33 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 11:15:00 | 1843.30 | 1834.91 | 0.00 | ORB-long ORB[1820.80,1835.65] vol=1.7x ATR=2.80 |
| Stop hit — per-position SL triggered | 2023-07-20 11:25:00 | 1840.50 | 1835.04 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 10:50:00 | 1845.00 | 1835.76 | 0.00 | ORB-long ORB[1821.25,1842.10] vol=9.6x ATR=3.97 |
| Stop hit — per-position SL triggered | 2023-07-21 10:55:00 | 1841.03 | 1835.87 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-07-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 10:20:00 | 1850.00 | 1841.13 | 0.00 | ORB-long ORB[1825.20,1844.80] vol=2.6x ATR=4.25 |
| Stop hit — per-position SL triggered | 2023-07-24 10:25:00 | 1845.75 | 1841.71 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:15:00 | 1990.00 | 1999.98 | 0.00 | ORB-short ORB[1990.55,2007.90] vol=1.5x ATR=5.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 10:35:00 | 1981.74 | 1995.25 | 0.00 | T1 1.5R @ 1981.74 |
| Stop hit — per-position SL triggered | 2023-08-02 11:00:00 | 1990.00 | 1993.06 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 11:10:00 | 2014.00 | 2007.78 | 0.00 | ORB-long ORB[1996.00,2013.90] vol=4.6x ATR=4.12 |
| Stop hit — per-position SL triggered | 2023-08-07 11:30:00 | 2009.88 | 2009.42 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 09:45:00 | 1978.45 | 1985.84 | 0.00 | ORB-short ORB[1980.20,2004.05] vol=1.9x ATR=4.68 |
| Stop hit — per-position SL triggered | 2023-08-08 09:50:00 | 1983.13 | 1983.42 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:50:00 | 1972.10 | 1979.83 | 0.00 | ORB-short ORB[1977.85,1994.35] vol=4.4x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-09 12:35:00 | 1966.12 | 1974.42 | 0.00 | T1 1.5R @ 1966.12 |
| Stop hit — per-position SL triggered | 2023-08-09 15:00:00 | 1972.10 | 1971.01 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 10:15:00 | 1978.00 | 1971.18 | 0.00 | ORB-long ORB[1958.00,1973.35] vol=1.9x ATR=4.56 |
| Stop hit — per-position SL triggered | 2023-08-10 10:20:00 | 1973.44 | 1971.62 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 09:35:00 | 1970.90 | 1959.82 | 0.00 | ORB-long ORB[1945.95,1967.25] vol=1.7x ATR=6.33 |
| Stop hit — per-position SL triggered | 2023-08-14 09:40:00 | 1964.57 | 1961.88 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-08-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 09:45:00 | 1997.60 | 1978.22 | 0.00 | ORB-long ORB[1950.00,1978.00] vol=1.8x ATR=7.78 |
| Stop hit — per-position SL triggered | 2023-08-23 10:00:00 | 1989.82 | 1984.12 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-25 09:55:00 | 1998.00 | 1988.42 | 0.00 | ORB-long ORB[1969.05,1987.95] vol=2.6x ATR=5.90 |
| Stop hit — per-position SL triggered | 2023-08-25 10:00:00 | 1992.10 | 1989.56 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-01 11:15:00 | 1928.00 | 1939.34 | 0.00 | ORB-short ORB[1932.60,1951.30] vol=4.3x ATR=3.64 |
| Stop hit — per-position SL triggered | 2023-09-01 11:20:00 | 1931.64 | 1937.62 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 11:05:00 | 1978.95 | 1965.67 | 0.00 | ORB-long ORB[1943.05,1967.75] vol=3.4x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 11:15:00 | 1984.71 | 1969.23 | 0.00 | T1 1.5R @ 1984.71 |
| Stop hit — per-position SL triggered | 2023-09-05 11:40:00 | 1978.95 | 1971.62 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 09:35:00 | 2021.90 | 2010.21 | 0.00 | ORB-long ORB[1994.00,2017.00] vol=2.3x ATR=6.17 |
| Stop hit — per-position SL triggered | 2023-09-06 09:40:00 | 2015.73 | 2010.79 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 11:00:00 | 2015.00 | 2021.89 | 0.00 | ORB-short ORB[2017.00,2031.80] vol=6.2x ATR=3.76 |
| Stop hit — per-position SL triggered | 2023-09-07 11:50:00 | 2018.76 | 2019.99 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:55:00 | 2034.30 | 2027.63 | 0.00 | ORB-long ORB[2013.05,2028.95] vol=1.8x ATR=4.78 |
| Stop hit — per-position SL triggered | 2023-09-08 10:10:00 | 2029.52 | 2029.53 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 10:25:00 | 2032.05 | 2025.10 | 0.00 | ORB-long ORB[2009.65,2032.00] vol=1.6x ATR=3.99 |
| Stop hit — per-position SL triggered | 2023-09-11 10:40:00 | 2028.06 | 2026.20 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 10:50:00 | 1993.00 | 1999.00 | 0.00 | ORB-short ORB[2008.70,2021.00] vol=1.6x ATR=4.54 |
| Stop hit — per-position SL triggered | 2023-09-12 10:55:00 | 1997.54 | 1998.82 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-09-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-13 11:10:00 | 1966.65 | 1982.81 | 0.00 | ORB-short ORB[1990.35,1999.00] vol=1.7x ATR=4.44 |
| Stop hit — per-position SL triggered | 2023-09-13 11:20:00 | 1971.09 | 1982.10 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 09:45:00 | 1962.80 | 1967.82 | 0.00 | ORB-short ORB[1964.00,1974.10] vol=2.5x ATR=4.61 |
| Stop hit — per-position SL triggered | 2023-09-14 09:50:00 | 1967.41 | 1967.62 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 10:50:00 | 2009.05 | 2002.67 | 0.00 | ORB-long ORB[1987.60,2006.30] vol=3.5x ATR=5.61 |
| Stop hit — per-position SL triggered | 2023-09-18 12:00:00 | 2003.44 | 2004.46 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-09-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 09:35:00 | 2015.90 | 2011.00 | 0.00 | ORB-long ORB[2000.20,2014.00] vol=1.6x ATR=4.71 |
| Stop hit — per-position SL triggered | 2023-09-21 09:45:00 | 2011.19 | 2011.43 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-09-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-22 10:55:00 | 2012.70 | 1997.80 | 0.00 | ORB-long ORB[1980.05,1997.00] vol=2.8x ATR=4.46 |
| Stop hit — per-position SL triggered | 2023-09-22 11:35:00 | 2008.24 | 2000.09 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-09-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 09:45:00 | 2054.45 | 2079.16 | 0.00 | ORB-short ORB[2070.45,2096.00] vol=1.7x ATR=10.73 |
| Stop hit — per-position SL triggered | 2023-09-27 09:50:00 | 2065.18 | 2078.10 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 09:30:00 | 1997.45 | 1991.16 | 0.00 | ORB-long ORB[1972.00,1997.25] vol=1.8x ATR=5.58 |
| Stop hit — per-position SL triggered | 2023-10-09 10:00:00 | 1991.87 | 1992.98 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-10-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 09:50:00 | 2029.00 | 2017.80 | 0.00 | ORB-long ORB[2002.05,2025.00] vol=3.1x ATR=5.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 10:00:00 | 2036.70 | 2021.95 | 0.00 | T1 1.5R @ 2036.70 |
| Stop hit — per-position SL triggered | 2023-10-10 10:05:00 | 2029.00 | 2022.40 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-10-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 10:55:00 | 2083.20 | 2056.26 | 0.00 | ORB-long ORB[2030.90,2054.35] vol=2.1x ATR=6.46 |
| Stop hit — per-position SL triggered | 2023-10-11 11:20:00 | 2076.74 | 2061.10 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-10-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 09:50:00 | 2076.15 | 2067.31 | 0.00 | ORB-long ORB[2043.00,2071.40] vol=2.2x ATR=5.27 |
| Stop hit — per-position SL triggered | 2023-10-13 10:05:00 | 2070.88 | 2069.84 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 11:15:00 | 2078.05 | 2069.89 | 0.00 | ORB-long ORB[2051.10,2074.60] vol=3.5x ATR=5.88 |
| Stop hit — per-position SL triggered | 2023-10-17 11:35:00 | 2072.17 | 2069.95 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-10-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 10:50:00 | 2072.35 | 2054.83 | 0.00 | ORB-long ORB[2035.55,2054.20] vol=1.9x ATR=5.52 |
| Stop hit — per-position SL triggered | 2023-10-19 10:55:00 | 2066.83 | 2055.51 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-10-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 11:10:00 | 2089.65 | 2099.50 | 0.00 | ORB-short ORB[2094.50,2110.50] vol=2.2x ATR=5.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 11:45:00 | 2081.97 | 2097.44 | 0.00 | T1 1.5R @ 2081.97 |
| Stop hit — per-position SL triggered | 2023-10-23 13:50:00 | 2089.65 | 2090.52 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-10-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:40:00 | 2042.40 | 2051.99 | 0.00 | ORB-short ORB[2042.45,2064.50] vol=2.1x ATR=7.52 |
| Stop hit — per-position SL triggered | 2023-10-26 09:45:00 | 2049.92 | 2051.64 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-11-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 10:45:00 | 2113.70 | 2109.02 | 0.00 | ORB-long ORB[2095.00,2113.65] vol=1.6x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 11:15:00 | 2120.90 | 2111.33 | 0.00 | T1 1.5R @ 2120.90 |
| Stop hit — per-position SL triggered | 2023-11-02 11:50:00 | 2113.70 | 2114.06 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-03 10:45:00 | 2110.55 | 2113.52 | 0.00 | ORB-short ORB[2115.10,2133.40] vol=1.6x ATR=4.79 |
| Stop hit — per-position SL triggered | 2023-11-03 11:25:00 | 2115.34 | 2113.07 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:45:00 | 2111.00 | 2099.48 | 0.00 | ORB-long ORB[2088.15,2100.95] vol=2.2x ATR=4.82 |
| Stop hit — per-position SL triggered | 2023-11-06 10:50:00 | 2106.18 | 2099.97 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-11-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 10:00:00 | 2114.00 | 2106.55 | 0.00 | ORB-long ORB[2095.25,2109.65] vol=1.5x ATR=4.92 |
| Stop hit — per-position SL triggered | 2023-11-13 10:25:00 | 2109.08 | 2107.98 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-11-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 09:40:00 | 2151.25 | 2142.73 | 0.00 | ORB-long ORB[2128.90,2142.20] vol=2.0x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 10:05:00 | 2158.28 | 2147.08 | 0.00 | T1 1.5R @ 2158.28 |
| Stop hit — per-position SL triggered | 2023-11-17 10:50:00 | 2151.25 | 2149.09 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 09:35:00 | 2320.00 | 2313.01 | 0.00 | ORB-long ORB[2290.20,2317.50] vol=2.3x ATR=6.62 |
| Stop hit — per-position SL triggered | 2023-12-05 09:50:00 | 2313.38 | 2317.02 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-12-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-07 09:45:00 | 2303.30 | 2315.58 | 0.00 | ORB-short ORB[2310.00,2337.55] vol=1.6x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 14:35:00 | 2294.07 | 2305.72 | 0.00 | T1 1.5R @ 2294.07 |
| Stop hit — per-position SL triggered | 2023-12-07 15:00:00 | 2303.30 | 2304.88 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2023-12-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 09:45:00 | 2301.55 | 2303.64 | 0.00 | ORB-short ORB[2302.30,2319.95] vol=1.5x ATR=5.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 09:55:00 | 2293.18 | 2302.06 | 0.00 | T1 1.5R @ 2293.18 |
| Target hit | 2023-12-08 12:30:00 | 2292.35 | 2292.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — BUY (started 2023-12-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 10:55:00 | 2320.65 | 2308.40 | 0.00 | ORB-long ORB[2288.05,2309.75] vol=1.7x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 12:05:00 | 2328.03 | 2312.43 | 0.00 | T1 1.5R @ 2328.03 |
| Stop hit — per-position SL triggered | 2023-12-12 12:45:00 | 2320.65 | 2314.24 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 10:20:00 | 2352.05 | 2340.72 | 0.00 | ORB-long ORB[2331.15,2349.00] vol=1.8x ATR=6.20 |
| Stop hit — per-position SL triggered | 2023-12-13 10:25:00 | 2345.85 | 2341.82 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 11:05:00 | 2398.35 | 2384.36 | 0.00 | ORB-long ORB[2357.20,2390.90] vol=2.0x ATR=5.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 12:55:00 | 2407.34 | 2388.89 | 0.00 | T1 1.5R @ 2407.34 |
| Target hit | 2023-12-19 15:20:00 | 2429.30 | 2401.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2023-12-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 10:40:00 | 2377.95 | 2372.74 | 0.00 | ORB-long ORB[2354.05,2368.40] vol=1.6x ATR=6.14 |
| Stop hit — per-position SL triggered | 2023-12-21 12:20:00 | 2371.81 | 2374.34 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-12-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:45:00 | 2437.25 | 2423.78 | 0.00 | ORB-long ORB[2397.40,2429.30] vol=2.8x ATR=6.48 |
| Stop hit — per-position SL triggered | 2023-12-26 10:50:00 | 2430.77 | 2424.00 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2023-12-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:50:00 | 2476.90 | 2470.43 | 0.00 | ORB-long ORB[2458.95,2471.90] vol=1.8x ATR=5.70 |
| Stop hit — per-position SL triggered | 2023-12-27 10:10:00 | 2471.20 | 2471.65 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2023-12-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 10:35:00 | 2484.95 | 2472.22 | 0.00 | ORB-long ORB[2457.10,2480.00] vol=2.0x ATR=6.05 |
| Stop hit — per-position SL triggered | 2023-12-28 10:50:00 | 2478.90 | 2477.30 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 11:15:00 | 2499.50 | 2510.54 | 0.00 | ORB-short ORB[2500.10,2520.00] vol=2.1x ATR=6.00 |
| Stop hit — per-position SL triggered | 2024-01-04 13:20:00 | 2505.50 | 2507.55 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 10:25:00 | 2432.85 | 2415.65 | 0.00 | ORB-long ORB[2397.10,2419.50] vol=2.7x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 10:30:00 | 2441.61 | 2417.27 | 0.00 | T1 1.5R @ 2441.61 |
| Target hit | 2024-01-11 14:10:00 | 2448.15 | 2453.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — SELL (started 2024-01-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:50:00 | 2457.00 | 2475.58 | 0.00 | ORB-short ORB[2480.35,2501.85] vol=2.0x ATR=8.73 |
| Stop hit — per-position SL triggered | 2024-01-18 10:00:00 | 2465.73 | 2472.62 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-01-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 10:00:00 | 2545.60 | 2526.87 | 0.00 | ORB-long ORB[2495.00,2509.20] vol=4.1x ATR=7.71 |
| Stop hit — per-position SL triggered | 2024-01-20 10:05:00 | 2537.89 | 2528.07 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-01-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:55:00 | 2478.40 | 2499.73 | 0.00 | ORB-short ORB[2498.10,2518.20] vol=2.1x ATR=7.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 11:00:00 | 2467.55 | 2496.86 | 0.00 | T1 1.5R @ 2467.55 |
| Stop hit — per-position SL triggered | 2024-01-25 11:20:00 | 2478.40 | 2494.64 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-01-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-30 10:45:00 | 2510.00 | 2522.01 | 0.00 | ORB-short ORB[2511.25,2533.00] vol=1.7x ATR=5.38 |
| Stop hit — per-position SL triggered | 2024-01-30 11:35:00 | 2515.38 | 2517.73 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-02-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 10:55:00 | 2538.90 | 2524.80 | 0.00 | ORB-long ORB[2495.25,2518.05] vol=1.6x ATR=5.72 |
| Stop hit — per-position SL triggered | 2024-02-02 12:10:00 | 2533.18 | 2527.94 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-02-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 11:05:00 | 2530.20 | 2539.19 | 0.00 | ORB-short ORB[2541.85,2570.00] vol=1.6x ATR=5.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 11:55:00 | 2522.14 | 2536.29 | 0.00 | T1 1.5R @ 2522.14 |
| Stop hit — per-position SL triggered | 2024-02-07 13:15:00 | 2530.20 | 2532.65 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-02-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:50:00 | 2529.00 | 2541.00 | 0.00 | ORB-short ORB[2543.25,2557.30] vol=2.2x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 11:00:00 | 2522.13 | 2538.30 | 0.00 | T1 1.5R @ 2522.13 |
| Target hit | 2024-02-08 15:20:00 | 2504.45 | 2514.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2024-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-12 09:30:00 | 2543.95 | 2532.24 | 0.00 | ORB-long ORB[2511.45,2539.05] vol=1.7x ATR=7.20 |
| Stop hit — per-position SL triggered | 2024-02-12 09:35:00 | 2536.75 | 2533.11 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-13 09:45:00 | 2517.90 | 2525.53 | 0.00 | ORB-short ORB[2518.00,2529.95] vol=3.6x ATR=7.86 |
| Stop hit — per-position SL triggered | 2024-02-13 10:15:00 | 2525.76 | 2523.12 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-02-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 09:30:00 | 2562.80 | 2554.85 | 0.00 | ORB-long ORB[2535.00,2558.00] vol=2.7x ATR=7.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-14 09:50:00 | 2574.35 | 2560.80 | 0.00 | T1 1.5R @ 2574.35 |
| Target hit | 2024-02-14 12:35:00 | 2570.05 | 2570.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 83 — SELL (started 2024-02-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 09:45:00 | 2591.65 | 2607.09 | 0.00 | ORB-short ORB[2596.60,2618.15] vol=1.8x ATR=8.27 |
| Stop hit — per-position SL triggered | 2024-02-15 10:15:00 | 2599.92 | 2602.63 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 09:35:00 | 2594.25 | 2587.35 | 0.00 | ORB-long ORB[2571.85,2585.10] vol=1.5x ATR=8.27 |
| Stop hit — per-position SL triggered | 2024-02-19 09:45:00 | 2585.98 | 2585.79 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 11:10:00 | 2525.00 | 2543.67 | 0.00 | ORB-short ORB[2545.70,2575.00] vol=1.6x ATR=5.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 11:35:00 | 2516.99 | 2539.87 | 0.00 | T1 1.5R @ 2516.99 |
| Stop hit — per-position SL triggered | 2024-02-20 11:55:00 | 2525.00 | 2538.69 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 09:40:00 | 2528.75 | 2533.64 | 0.00 | ORB-short ORB[2531.10,2542.20] vol=1.8x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 09:50:00 | 2519.57 | 2528.97 | 0.00 | T1 1.5R @ 2519.57 |
| Stop hit — per-position SL triggered | 2024-02-26 10:35:00 | 2528.75 | 2527.23 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-02-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:20:00 | 2520.70 | 2530.22 | 0.00 | ORB-short ORB[2525.85,2541.85] vol=1.5x ATR=4.91 |
| Stop hit — per-position SL triggered | 2024-02-28 10:30:00 | 2525.61 | 2528.51 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 11:15:00 | 2549.80 | 2537.30 | 0.00 | ORB-long ORB[2525.00,2548.00] vol=3.7x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-03-04 11:30:00 | 2545.51 | 2537.82 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-03-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 10:30:00 | 2544.60 | 2556.61 | 0.00 | ORB-short ORB[2558.15,2583.00] vol=1.6x ATR=6.30 |
| Stop hit — per-position SL triggered | 2024-03-05 10:35:00 | 2550.90 | 2556.41 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:55:00 | 2511.65 | 2528.32 | 0.00 | ORB-short ORB[2528.70,2548.50] vol=2.2x ATR=8.37 |
| Stop hit — per-position SL triggered | 2024-03-06 10:30:00 | 2520.02 | 2522.24 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 10:55:00 | 2619.00 | 2599.61 | 0.00 | ORB-long ORB[2578.70,2605.55] vol=1.6x ATR=6.80 |
| Stop hit — per-position SL triggered | 2024-03-11 11:05:00 | 2612.20 | 2601.87 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-03-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:05:00 | 2665.70 | 2692.51 | 0.00 | ORB-short ORB[2693.00,2729.95] vol=1.5x ATR=7.85 |
| Stop hit — per-position SL triggered | 2024-03-19 10:10:00 | 2673.55 | 2691.27 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2024-03-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 10:50:00 | 2653.00 | 2638.61 | 0.00 | ORB-long ORB[2612.00,2639.00] vol=2.0x ATR=8.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:55:00 | 2665.37 | 2642.02 | 0.00 | T1 1.5R @ 2665.37 |
| Stop hit — per-position SL triggered | 2024-03-20 11:10:00 | 2653.00 | 2643.83 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2024-03-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 11:00:00 | 2707.35 | 2690.18 | 0.00 | ORB-long ORB[2675.15,2699.15] vol=3.4x ATR=6.92 |
| Stop hit — per-position SL triggered | 2024-03-22 13:10:00 | 2700.43 | 2697.49 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-03-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 09:45:00 | 2732.80 | 2705.41 | 0.00 | ORB-long ORB[2679.00,2712.85] vol=1.9x ATR=9.27 |
| Stop hit — per-position SL triggered | 2024-03-26 10:05:00 | 2723.53 | 2716.45 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-03-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 10:10:00 | 2708.30 | 2725.10 | 0.00 | ORB-short ORB[2729.05,2754.00] vol=2.2x ATR=8.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-27 11:00:00 | 2696.27 | 2717.11 | 0.00 | T1 1.5R @ 2696.27 |
| Target hit | 2024-03-27 15:20:00 | 2656.00 | 2684.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 97 — SELL (started 2024-04-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-01 09:30:00 | 2700.00 | 2709.46 | 0.00 | ORB-short ORB[2703.30,2736.00] vol=3.3x ATR=11.84 |
| Stop hit — per-position SL triggered | 2024-04-01 09:55:00 | 2711.84 | 2706.70 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2024-04-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:30:00 | 2771.95 | 2784.91 | 0.00 | ORB-short ORB[2780.05,2815.90] vol=2.1x ATR=8.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 09:35:00 | 2759.04 | 2780.07 | 0.00 | T1 1.5R @ 2759.04 |
| Target hit | 2024-04-04 10:45:00 | 2757.85 | 2754.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 99 — SELL (started 2024-04-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:35:00 | 2681.35 | 2698.89 | 0.00 | ORB-short ORB[2701.80,2719.75] vol=2.3x ATR=7.78 |
| Stop hit — per-position SL triggered | 2024-04-18 09:40:00 | 2689.13 | 2697.94 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2024-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 11:10:00 | 2680.70 | 2673.03 | 0.00 | ORB-long ORB[2655.25,2679.95] vol=1.7x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 11:20:00 | 2688.84 | 2674.09 | 0.00 | T1 1.5R @ 2688.84 |
| Target hit | 2024-04-23 15:05:00 | 2688.50 | 2689.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 101 — BUY (started 2024-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:00:00 | 2727.00 | 2710.23 | 0.00 | ORB-long ORB[2675.05,2708.00] vol=1.7x ATR=7.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 10:45:00 | 2738.17 | 2720.19 | 0.00 | T1 1.5R @ 2738.17 |
| Target hit | 2024-04-24 15:20:00 | 2750.20 | 2737.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 102 — SELL (started 2024-05-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:40:00 | 2780.00 | 2813.35 | 0.00 | ORB-short ORB[2803.40,2823.35] vol=1.6x ATR=9.79 |
| Stop hit — per-position SL triggered | 2024-05-03 10:45:00 | 2789.79 | 2812.44 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2024-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 11:00:00 | 2837.10 | 2816.54 | 0.00 | ORB-long ORB[2795.05,2828.00] vol=3.2x ATR=9.04 |
| Stop hit — per-position SL triggered | 2024-05-06 11:20:00 | 2828.06 | 2821.29 | 0.00 | SL hit |

### Cycle 104 — BUY (started 2024-05-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 10:55:00 | 2811.05 | 2800.02 | 0.00 | ORB-long ORB[2767.00,2803.30] vol=2.1x ATR=9.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 11:45:00 | 2825.44 | 2803.89 | 0.00 | T1 1.5R @ 2825.44 |
| Stop hit — per-position SL triggered | 2024-05-10 12:20:00 | 2811.05 | 2806.11 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 11:15:00 | 1635.45 | 2023-05-12 11:55:00 | 1643.66 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-05-12 11:15:00 | 1635.45 | 2023-05-12 12:00:00 | 1635.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-17 09:35:00 | 1665.15 | 2023-05-17 10:05:00 | 1658.62 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-05-18 10:05:00 | 1657.50 | 2023-05-18 10:35:00 | 1649.45 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-05-18 10:05:00 | 1657.50 | 2023-05-18 10:45:00 | 1657.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-22 10:50:00 | 1613.50 | 2023-05-22 11:05:00 | 1609.15 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-05-23 09:50:00 | 1601.15 | 2023-05-23 09:55:00 | 1604.98 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-24 10:20:00 | 1601.20 | 2023-05-24 10:40:00 | 1597.52 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-05-25 11:10:00 | 1572.20 | 2023-05-25 13:15:00 | 1575.53 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-05-31 11:05:00 | 1589.30 | 2023-05-31 11:10:00 | 1584.68 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-05-31 11:05:00 | 1589.30 | 2023-05-31 15:00:00 | 1589.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-02 09:55:00 | 1607.00 | 2023-06-02 10:45:00 | 1610.74 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-07 09:45:00 | 1637.00 | 2023-06-07 10:10:00 | 1633.34 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-06-09 10:20:00 | 1607.05 | 2023-06-09 12:45:00 | 1600.73 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-06-09 10:20:00 | 1607.05 | 2023-06-09 15:20:00 | 1596.50 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2023-06-13 09:30:00 | 1652.55 | 2023-06-13 09:50:00 | 1658.90 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-06-13 09:30:00 | 1652.55 | 2023-06-13 13:00:00 | 1654.25 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2023-06-16 09:35:00 | 1658.10 | 2023-06-16 13:20:00 | 1665.86 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-06-16 09:35:00 | 1658.10 | 2023-06-16 14:40:00 | 1658.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-20 10:30:00 | 1644.10 | 2023-06-20 10:45:00 | 1647.25 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-06-21 10:00:00 | 1676.95 | 2023-06-21 10:20:00 | 1672.22 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-06-27 10:00:00 | 1682.05 | 2023-06-27 10:40:00 | 1687.74 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-06-27 10:00:00 | 1682.05 | 2023-06-27 15:20:00 | 1691.05 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2023-07-03 09:30:00 | 1683.00 | 2023-07-03 09:45:00 | 1686.51 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-05 09:35:00 | 1751.00 | 2023-07-05 09:40:00 | 1765.33 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2023-07-05 09:35:00 | 1751.00 | 2023-07-05 09:45:00 | 1751.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-10 09:50:00 | 1760.50 | 2023-07-10 10:45:00 | 1767.36 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-07-12 11:15:00 | 1784.00 | 2023-07-12 11:50:00 | 1787.96 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-14 10:35:00 | 1827.85 | 2023-07-14 10:45:00 | 1823.07 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-18 09:30:00 | 1856.25 | 2023-07-18 09:35:00 | 1851.54 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-19 11:15:00 | 1835.05 | 2023-07-19 11:20:00 | 1829.70 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-07-19 11:15:00 | 1835.05 | 2023-07-19 11:40:00 | 1835.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-20 11:15:00 | 1843.30 | 2023-07-20 11:25:00 | 1840.50 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-07-21 10:50:00 | 1845.00 | 2023-07-21 10:55:00 | 1841.03 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-24 10:20:00 | 1850.00 | 2023-07-24 10:25:00 | 1845.75 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-08-02 10:15:00 | 1990.00 | 2023-08-02 10:35:00 | 1981.74 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-08-02 10:15:00 | 1990.00 | 2023-08-02 11:00:00 | 1990.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-07 11:10:00 | 2014.00 | 2023-08-07 11:30:00 | 2009.88 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-08 09:45:00 | 1978.45 | 2023-08-08 09:50:00 | 1983.13 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-09 10:50:00 | 1972.10 | 2023-08-09 12:35:00 | 1966.12 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-08-09 10:50:00 | 1972.10 | 2023-08-09 15:00:00 | 1972.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-10 10:15:00 | 1978.00 | 2023-08-10 10:20:00 | 1973.44 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-08-14 09:35:00 | 1970.90 | 2023-08-14 09:40:00 | 1964.57 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-08-23 09:45:00 | 1997.60 | 2023-08-23 10:00:00 | 1989.82 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-08-25 09:55:00 | 1998.00 | 2023-08-25 10:00:00 | 1992.10 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-09-01 11:15:00 | 1928.00 | 2023-09-01 11:20:00 | 1931.64 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-09-05 11:05:00 | 1978.95 | 2023-09-05 11:15:00 | 1984.71 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-09-05 11:05:00 | 1978.95 | 2023-09-05 11:40:00 | 1978.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-06 09:35:00 | 2021.90 | 2023-09-06 09:40:00 | 2015.73 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-09-07 11:00:00 | 2015.00 | 2023-09-07 11:50:00 | 2018.76 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-09-08 09:55:00 | 2034.30 | 2023-09-08 10:10:00 | 2029.52 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-09-11 10:25:00 | 2032.05 | 2023-09-11 10:40:00 | 2028.06 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-09-12 10:50:00 | 1993.00 | 2023-09-12 10:55:00 | 1997.54 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-09-13 11:10:00 | 1966.65 | 2023-09-13 11:20:00 | 1971.09 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-09-14 09:45:00 | 1962.80 | 2023-09-14 09:50:00 | 1967.41 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-09-18 10:50:00 | 2009.05 | 2023-09-18 12:00:00 | 2003.44 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-09-21 09:35:00 | 2015.90 | 2023-09-21 09:45:00 | 2011.19 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-09-22 10:55:00 | 2012.70 | 2023-09-22 11:35:00 | 2008.24 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-09-27 09:45:00 | 2054.45 | 2023-09-27 09:50:00 | 2065.18 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2023-10-09 09:30:00 | 1997.45 | 2023-10-09 10:00:00 | 1991.87 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-10-10 09:50:00 | 2029.00 | 2023-10-10 10:00:00 | 2036.70 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-10-10 09:50:00 | 2029.00 | 2023-10-10 10:05:00 | 2029.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-11 10:55:00 | 2083.20 | 2023-10-11 11:20:00 | 2076.74 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-10-13 09:50:00 | 2076.15 | 2023-10-13 10:05:00 | 2070.88 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-10-17 11:15:00 | 2078.05 | 2023-10-17 11:35:00 | 2072.17 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-10-19 10:50:00 | 2072.35 | 2023-10-19 10:55:00 | 2066.83 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-10-23 11:10:00 | 2089.65 | 2023-10-23 11:45:00 | 2081.97 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-10-23 11:10:00 | 2089.65 | 2023-10-23 13:50:00 | 2089.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-26 09:40:00 | 2042.40 | 2023-10-26 09:45:00 | 2049.92 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-11-02 10:45:00 | 2113.70 | 2023-11-02 11:15:00 | 2120.90 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-11-02 10:45:00 | 2113.70 | 2023-11-02 11:50:00 | 2113.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-03 10:45:00 | 2110.55 | 2023-11-03 11:25:00 | 2115.34 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-06 10:45:00 | 2111.00 | 2023-11-06 10:50:00 | 2106.18 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-13 10:00:00 | 2114.00 | 2023-11-13 10:25:00 | 2109.08 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-17 09:40:00 | 2151.25 | 2023-11-17 10:05:00 | 2158.28 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-11-17 09:40:00 | 2151.25 | 2023-11-17 10:50:00 | 2151.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-05 09:35:00 | 2320.00 | 2023-12-05 09:50:00 | 2313.38 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-12-07 09:45:00 | 2303.30 | 2023-12-07 14:35:00 | 2294.07 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-12-07 09:45:00 | 2303.30 | 2023-12-07 15:00:00 | 2303.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-08 09:45:00 | 2301.55 | 2023-12-08 09:55:00 | 2293.18 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-12-08 09:45:00 | 2301.55 | 2023-12-08 12:30:00 | 2292.35 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2023-12-12 10:55:00 | 2320.65 | 2023-12-12 12:05:00 | 2328.03 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-12-12 10:55:00 | 2320.65 | 2023-12-12 12:45:00 | 2320.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-13 10:20:00 | 2352.05 | 2023-12-13 10:25:00 | 2345.85 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-19 11:05:00 | 2398.35 | 2023-12-19 12:55:00 | 2407.34 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-12-19 11:05:00 | 2398.35 | 2023-12-19 15:20:00 | 2429.30 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2023-12-21 10:40:00 | 2377.95 | 2023-12-21 12:20:00 | 2371.81 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-26 10:45:00 | 2437.25 | 2023-12-26 10:50:00 | 2430.77 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-12-27 09:50:00 | 2476.90 | 2023-12-27 10:10:00 | 2471.20 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-12-28 10:35:00 | 2484.95 | 2023-12-28 10:50:00 | 2478.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-04 11:15:00 | 2499.50 | 2024-01-04 13:20:00 | 2505.50 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-01-11 10:25:00 | 2432.85 | 2024-01-11 10:30:00 | 2441.61 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-01-11 10:25:00 | 2432.85 | 2024-01-11 14:10:00 | 2448.15 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2024-01-18 09:50:00 | 2457.00 | 2024-01-18 10:00:00 | 2465.73 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-01-20 10:00:00 | 2545.60 | 2024-01-20 10:05:00 | 2537.89 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-01-25 10:55:00 | 2478.40 | 2024-01-25 11:00:00 | 2467.55 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-01-25 10:55:00 | 2478.40 | 2024-01-25 11:20:00 | 2478.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-30 10:45:00 | 2510.00 | 2024-01-30 11:35:00 | 2515.38 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-02-02 10:55:00 | 2538.90 | 2024-02-02 12:10:00 | 2533.18 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-02-07 11:05:00 | 2530.20 | 2024-02-07 11:55:00 | 2522.14 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-02-07 11:05:00 | 2530.20 | 2024-02-07 13:15:00 | 2530.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-08 10:50:00 | 2529.00 | 2024-02-08 11:00:00 | 2522.13 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-02-08 10:50:00 | 2529.00 | 2024-02-08 15:20:00 | 2504.45 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2024-02-12 09:30:00 | 2543.95 | 2024-02-12 09:35:00 | 2536.75 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-02-13 09:45:00 | 2517.90 | 2024-02-13 10:15:00 | 2525.76 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-02-14 09:30:00 | 2562.80 | 2024-02-14 09:50:00 | 2574.35 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-02-14 09:30:00 | 2562.80 | 2024-02-14 12:35:00 | 2570.05 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2024-02-15 09:45:00 | 2591.65 | 2024-02-15 10:15:00 | 2599.92 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-02-19 09:35:00 | 2594.25 | 2024-02-19 09:45:00 | 2585.98 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-02-20 11:10:00 | 2525.00 | 2024-02-20 11:35:00 | 2516.99 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-02-20 11:10:00 | 2525.00 | 2024-02-20 11:55:00 | 2525.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-26 09:40:00 | 2528.75 | 2024-02-26 09:50:00 | 2519.57 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-02-26 09:40:00 | 2528.75 | 2024-02-26 10:35:00 | 2528.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-28 10:20:00 | 2520.70 | 2024-02-28 10:30:00 | 2525.61 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-03-04 11:15:00 | 2549.80 | 2024-03-04 11:30:00 | 2545.51 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-03-05 10:30:00 | 2544.60 | 2024-03-05 10:35:00 | 2550.90 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-03-06 09:55:00 | 2511.65 | 2024-03-06 10:30:00 | 2520.02 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-11 10:55:00 | 2619.00 | 2024-03-11 11:05:00 | 2612.20 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-03-19 10:05:00 | 2665.70 | 2024-03-19 10:10:00 | 2673.55 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-03-20 10:50:00 | 2653.00 | 2024-03-20 10:55:00 | 2665.37 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-03-20 10:50:00 | 2653.00 | 2024-03-20 11:10:00 | 2653.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-22 11:00:00 | 2707.35 | 2024-03-22 13:10:00 | 2700.43 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-03-26 09:45:00 | 2732.80 | 2024-03-26 10:05:00 | 2723.53 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-03-27 10:10:00 | 2708.30 | 2024-03-27 11:00:00 | 2696.27 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-03-27 10:10:00 | 2708.30 | 2024-03-27 15:20:00 | 2656.00 | TARGET_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2024-04-01 09:30:00 | 2700.00 | 2024-04-01 09:55:00 | 2711.84 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-04-04 09:30:00 | 2771.95 | 2024-04-04 09:35:00 | 2759.04 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-04-04 09:30:00 | 2771.95 | 2024-04-04 10:45:00 | 2757.85 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2024-04-18 09:35:00 | 2681.35 | 2024-04-18 09:40:00 | 2689.13 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-23 11:10:00 | 2680.70 | 2024-04-23 11:20:00 | 2688.84 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-04-23 11:10:00 | 2680.70 | 2024-04-23 15:05:00 | 2688.50 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2024-04-24 10:00:00 | 2727.00 | 2024-04-24 10:45:00 | 2738.17 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-04-24 10:00:00 | 2727.00 | 2024-04-24 15:20:00 | 2750.20 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2024-05-03 10:40:00 | 2780.00 | 2024-05-03 10:45:00 | 2789.79 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-05-06 11:00:00 | 2837.10 | 2024-05-06 11:20:00 | 2828.06 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-05-10 10:55:00 | 2811.05 | 2024-05-10 11:45:00 | 2825.44 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-05-10 10:55:00 | 2811.05 | 2024-05-10 12:20:00 | 2811.05 | STOP_HIT | 0.50 | 0.00% |
