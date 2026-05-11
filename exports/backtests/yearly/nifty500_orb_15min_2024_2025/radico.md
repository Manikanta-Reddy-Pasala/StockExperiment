# Radico Khaitan Ltd (RADICO)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-07-04 15:25:00 (21408 bars)
- **Last close:** 2622.00
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 7 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 56
- **Target hits / Stop hits / Partials:** 7 / 56 / 22
- **Avg / median % per leg:** 0.03% / 0.00%
- **Sum % (uncompounded):** 2.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 13 | 30.2% | 3 | 30 | 10 | 0.02% | 0.9% |
| BUY @ 2nd Alert (retest1) | 43 | 13 | 30.2% | 3 | 30 | 10 | 0.02% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 16 | 38.1% | 4 | 26 | 12 | 0.04% | 1.8% |
| SELL @ 2nd Alert (retest1) | 42 | 16 | 38.1% | 4 | 26 | 12 | 0.04% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 85 | 29 | 34.1% | 7 | 56 | 22 | 0.03% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:25:00 | 1606.65 | 1621.28 | 0.00 | ORB-short ORB[1616.50,1640.50] vol=1.5x ATR=11.79 |
| Stop hit — per-position SL triggered | 2024-05-13 11:05:00 | 1618.44 | 1619.95 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:25:00 | 1680.95 | 1639.29 | 0.00 | ORB-long ORB[1621.00,1639.45] vol=1.9x ATR=9.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:30:00 | 1695.42 | 1643.96 | 0.00 | T1 1.5R @ 1695.42 |
| Stop hit — per-position SL triggered | 2024-05-16 10:35:00 | 1680.95 | 1645.85 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 11:10:00 | 1703.05 | 1712.56 | 0.00 | ORB-short ORB[1706.85,1725.15] vol=2.2x ATR=4.53 |
| Stop hit — per-position SL triggered | 2024-05-17 11:20:00 | 1707.58 | 1712.09 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-18 09:45:00 | 1718.35 | 1720.58 | 0.00 | ORB-short ORB[1735.00,1748.80] vol=2.4x ATR=10.11 |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 1722.65 | 0.00 | 0.00 | EOD overnight gap close |

### Cycle 5 — SELL (started 2024-05-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:45:00 | 1703.60 | 1708.92 | 0.00 | ORB-short ORB[1705.00,1726.05] vol=2.0x ATR=8.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 11:55:00 | 1691.39 | 1703.52 | 0.00 | T1 1.5R @ 1691.39 |
| Target hit | 2024-05-21 15:20:00 | 1691.00 | 1699.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2024-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:45:00 | 1691.00 | 1694.74 | 0.00 | ORB-short ORB[1692.20,1702.75] vol=2.3x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 11:50:00 | 1684.32 | 1692.06 | 0.00 | T1 1.5R @ 1684.32 |
| Stop hit — per-position SL triggered | 2024-05-23 12:20:00 | 1691.00 | 1691.64 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:35:00 | 1666.00 | 1676.58 | 0.00 | ORB-short ORB[1675.05,1685.70] vol=1.5x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 10:40:00 | 1659.17 | 1673.84 | 0.00 | T1 1.5R @ 1659.17 |
| Target hit | 2024-05-24 12:50:00 | 1650.90 | 1649.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2024-05-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:40:00 | 1578.20 | 1584.63 | 0.00 | ORB-short ORB[1579.45,1592.90] vol=1.6x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:00:00 | 1567.78 | 1583.13 | 0.00 | T1 1.5R @ 1567.78 |
| Stop hit — per-position SL triggered | 2024-05-31 10:45:00 | 1578.20 | 1579.89 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-06 09:55:00 | 1702.35 | 1706.96 | 0.00 | ORB-short ORB[1704.40,1727.00] vol=1.5x ATR=8.13 |
| Stop hit — per-position SL triggered | 2024-06-06 10:45:00 | 1710.48 | 1702.39 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:00:00 | 1718.70 | 1711.00 | 0.00 | ORB-long ORB[1699.55,1714.15] vol=1.6x ATR=5.28 |
| Stop hit — per-position SL triggered | 2024-06-07 10:05:00 | 1713.42 | 1711.46 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 10:05:00 | 1693.80 | 1697.42 | 0.00 | ORB-short ORB[1697.45,1718.90] vol=1.5x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:00:00 | 1687.44 | 1694.42 | 0.00 | T1 1.5R @ 1687.44 |
| Stop hit — per-position SL triggered | 2024-06-11 14:40:00 | 1693.80 | 1691.89 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 09:55:00 | 1714.50 | 1720.37 | 0.00 | ORB-short ORB[1718.30,1734.85] vol=1.8x ATR=4.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 10:25:00 | 1707.49 | 1717.29 | 0.00 | T1 1.5R @ 1707.49 |
| Target hit | 2024-06-14 14:25:00 | 1711.55 | 1711.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2024-06-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:30:00 | 1799.55 | 1787.05 | 0.00 | ORB-long ORB[1768.55,1786.85] vol=2.8x ATR=7.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 11:15:00 | 1810.94 | 1804.04 | 0.00 | T1 1.5R @ 1810.94 |
| Target hit | 2024-06-24 14:15:00 | 1819.70 | 1819.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2024-06-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 09:55:00 | 1797.65 | 1807.31 | 0.00 | ORB-short ORB[1803.05,1821.00] vol=1.7x ATR=7.54 |
| Stop hit — per-position SL triggered | 2024-06-28 10:05:00 | 1805.19 | 1806.06 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 1768.15 | 1773.74 | 0.00 | ORB-short ORB[1770.20,1787.65] vol=1.9x ATR=6.25 |
| Stop hit — per-position SL triggered | 2024-07-02 10:05:00 | 1774.40 | 1769.79 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:25:00 | 1740.00 | 1742.73 | 0.00 | ORB-short ORB[1742.05,1761.95] vol=2.1x ATR=5.52 |
| Stop hit — per-position SL triggered | 2024-07-03 10:40:00 | 1745.52 | 1742.49 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:30:00 | 1663.00 | 1669.85 | 0.00 | ORB-short ORB[1670.00,1680.00] vol=3.0x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 11:10:00 | 1654.96 | 1668.31 | 0.00 | T1 1.5R @ 1654.96 |
| Stop hit — per-position SL triggered | 2024-07-11 12:15:00 | 1663.00 | 1666.94 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:35:00 | 1690.00 | 1682.87 | 0.00 | ORB-long ORB[1675.40,1685.00] vol=2.2x ATR=5.99 |
| Stop hit — per-position SL triggered | 2024-07-16 11:55:00 | 1684.01 | 1688.18 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 11:00:00 | 1694.65 | 1707.10 | 0.00 | ORB-short ORB[1698.70,1720.35] vol=1.8x ATR=6.01 |
| Stop hit — per-position SL triggered | 2024-07-18 11:05:00 | 1700.66 | 1706.91 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:30:00 | 1735.00 | 1722.26 | 0.00 | ORB-long ORB[1705.00,1722.95] vol=3.0x ATR=7.83 |
| Stop hit — per-position SL triggered | 2024-07-24 09:35:00 | 1727.17 | 1726.53 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 11:10:00 | 1763.80 | 1754.27 | 0.00 | ORB-long ORB[1734.45,1751.95] vol=4.3x ATR=6.02 |
| Stop hit — per-position SL triggered | 2024-07-29 11:20:00 | 1757.78 | 1754.82 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:35:00 | 1732.65 | 1729.54 | 0.00 | ORB-long ORB[1715.00,1732.50] vol=3.4x ATR=7.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 09:50:00 | 1743.81 | 1732.42 | 0.00 | T1 1.5R @ 1743.81 |
| Stop hit — per-position SL triggered | 2024-07-30 10:00:00 | 1732.65 | 1732.56 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:15:00 | 1748.00 | 1737.13 | 0.00 | ORB-long ORB[1722.75,1742.60] vol=2.2x ATR=6.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 10:20:00 | 1757.93 | 1740.91 | 0.00 | T1 1.5R @ 1757.93 |
| Target hit | 2024-07-31 11:55:00 | 1750.20 | 1752.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2024-08-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 10:25:00 | 1681.80 | 1679.68 | 0.00 | ORB-long ORB[1662.85,1679.95] vol=7.2x ATR=4.61 |
| Stop hit — per-position SL triggered | 2024-08-13 11:10:00 | 1677.19 | 1680.78 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:45:00 | 1740.80 | 1735.58 | 0.00 | ORB-long ORB[1723.60,1737.00] vol=3.6x ATR=4.85 |
| Stop hit — per-position SL triggered | 2024-08-22 09:55:00 | 1735.95 | 1736.15 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1810.00 | 1823.51 | 0.00 | ORB-short ORB[1822.35,1831.20] vol=2.7x ATR=6.96 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 1816.96 | 1819.23 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:30:00 | 1834.95 | 1824.34 | 0.00 | ORB-long ORB[1816.20,1832.90] vol=3.3x ATR=6.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 12:15:00 | 1844.25 | 1834.76 | 0.00 | T1 1.5R @ 1844.25 |
| Target hit | 2024-08-30 12:40:00 | 1875.00 | 1876.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — SELL (started 2024-09-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 11:05:00 | 1990.80 | 2005.17 | 0.00 | ORB-short ORB[1996.10,2022.80] vol=1.7x ATR=7.14 |
| Stop hit — per-position SL triggered | 2024-09-03 11:10:00 | 1997.94 | 2004.30 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:35:00 | 2147.00 | 2132.39 | 0.00 | ORB-long ORB[2116.80,2140.90] vol=1.7x ATR=7.05 |
| Stop hit — per-position SL triggered | 2024-09-23 10:50:00 | 2139.95 | 2134.16 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 09:30:00 | 2151.65 | 2143.68 | 0.00 | ORB-long ORB[2115.50,2146.40] vol=7.0x ATR=11.97 |
| Stop hit — per-position SL triggered | 2024-09-26 09:40:00 | 2139.68 | 2142.97 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 09:40:00 | 2040.00 | 2047.55 | 0.00 | ORB-short ORB[2040.50,2070.00] vol=2.6x ATR=8.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 10:00:00 | 2027.35 | 2043.09 | 0.00 | T1 1.5R @ 2027.35 |
| Stop hit — per-position SL triggered | 2024-10-03 10:10:00 | 2040.00 | 2042.27 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:25:00 | 2225.00 | 2262.61 | 0.00 | ORB-short ORB[2279.15,2308.45] vol=2.2x ATR=10.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 11:15:00 | 2208.89 | 2250.68 | 0.00 | T1 1.5R @ 2208.89 |
| Stop hit — per-position SL triggered | 2024-10-29 12:45:00 | 2225.00 | 2241.00 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:35:00 | 2383.35 | 2390.32 | 0.00 | ORB-short ORB[2385.10,2410.05] vol=1.5x ATR=8.13 |
| Stop hit — per-position SL triggered | 2024-11-06 10:40:00 | 2391.48 | 2390.56 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:35:00 | 2422.00 | 2401.85 | 0.00 | ORB-long ORB[2384.35,2413.20] vol=1.7x ATR=11.76 |
| Stop hit — per-position SL triggered | 2024-11-07 10:05:00 | 2410.24 | 2412.81 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-11-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:40:00 | 2233.05 | 2257.01 | 0.00 | ORB-short ORB[2260.70,2284.20] vol=2.9x ATR=11.28 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 2244.33 | 2253.83 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:55:00 | 2213.35 | 2232.66 | 0.00 | ORB-short ORB[2232.15,2259.35] vol=2.5x ATR=12.65 |
| Stop hit — per-position SL triggered | 2024-11-21 10:35:00 | 2226.00 | 2225.26 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:30:00 | 2389.90 | 2371.00 | 0.00 | ORB-long ORB[2341.90,2374.20] vol=3.4x ATR=12.29 |
| Stop hit — per-position SL triggered | 2024-11-26 11:40:00 | 2377.61 | 2374.38 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:50:00 | 2378.95 | 2360.40 | 0.00 | ORB-long ORB[2342.55,2363.70] vol=2.8x ATR=7.33 |
| Stop hit — per-position SL triggered | 2024-11-27 11:05:00 | 2371.62 | 2362.77 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:05:00 | 2421.30 | 2404.05 | 0.00 | ORB-long ORB[2380.05,2416.00] vol=4.0x ATR=9.58 |
| Stop hit — per-position SL triggered | 2024-11-28 10:10:00 | 2411.72 | 2404.23 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:35:00 | 2355.50 | 2376.58 | 0.00 | ORB-short ORB[2374.65,2404.20] vol=1.5x ATR=8.23 |
| Stop hit — per-position SL triggered | 2024-12-04 10:50:00 | 2363.73 | 2372.80 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:30:00 | 2350.25 | 2356.00 | 0.00 | ORB-short ORB[2354.20,2372.35] vol=1.7x ATR=6.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 12:20:00 | 2341.26 | 2352.53 | 0.00 | T1 1.5R @ 2341.26 |
| Stop hit — per-position SL triggered | 2024-12-06 13:50:00 | 2350.25 | 2350.10 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-12-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 11:00:00 | 2363.50 | 2339.38 | 0.00 | ORB-long ORB[2328.05,2357.20] vol=3.2x ATR=9.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 11:05:00 | 2377.21 | 2344.15 | 0.00 | T1 1.5R @ 2377.21 |
| Stop hit — per-position SL triggered | 2024-12-09 11:15:00 | 2363.50 | 2347.29 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:50:00 | 2427.20 | 2415.21 | 0.00 | ORB-long ORB[2376.80,2403.10] vol=8.7x ATR=7.78 |
| Stop hit — per-position SL triggered | 2024-12-10 10:55:00 | 2419.42 | 2416.83 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:45:00 | 2452.45 | 2440.45 | 0.00 | ORB-long ORB[2421.20,2444.95] vol=1.8x ATR=9.99 |
| Stop hit — per-position SL triggered | 2024-12-11 10:05:00 | 2442.46 | 2443.44 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:05:00 | 2509.60 | 2522.12 | 0.00 | ORB-short ORB[2525.25,2552.70] vol=1.7x ATR=9.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:50:00 | 2496.03 | 2519.54 | 0.00 | T1 1.5R @ 2496.03 |
| Stop hit — per-position SL triggered | 2024-12-26 13:10:00 | 2509.60 | 2515.87 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:50:00 | 2582.65 | 2560.01 | 0.00 | ORB-long ORB[2535.00,2570.00] vol=2.2x ATR=10.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 09:55:00 | 2598.92 | 2571.71 | 0.00 | T1 1.5R @ 2598.92 |
| Stop hit — per-position SL triggered | 2024-12-30 10:00:00 | 2582.65 | 2571.63 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-01-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:50:00 | 2627.75 | 2605.35 | 0.00 | ORB-long ORB[2594.00,2613.65] vol=1.6x ATR=7.18 |
| Stop hit — per-position SL triggered | 2025-01-03 10:55:00 | 2620.57 | 2605.75 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:40:00 | 2561.80 | 2542.87 | 0.00 | ORB-long ORB[2514.80,2547.00] vol=1.7x ATR=12.78 |
| Stop hit — per-position SL triggered | 2025-01-07 09:45:00 | 2549.02 | 2543.67 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-13 09:30:00 | 2242.55 | 2227.60 | 0.00 | ORB-long ORB[2211.30,2235.30] vol=2.1x ATR=14.28 |
| Stop hit — per-position SL triggered | 2025-01-13 09:35:00 | 2228.27 | 2227.91 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:30:00 | 2299.05 | 2321.26 | 0.00 | ORB-short ORB[2329.40,2358.10] vol=2.4x ATR=9.00 |
| Stop hit — per-position SL triggered | 2025-01-21 10:35:00 | 2308.05 | 2320.40 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 11:10:00 | 2236.50 | 2211.87 | 0.00 | ORB-long ORB[2176.55,2202.20] vol=2.8x ATR=9.55 |
| Stop hit — per-position SL triggered | 2025-01-23 14:50:00 | 2226.95 | 2230.21 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:10:00 | 2187.00 | 2151.59 | 0.00 | ORB-long ORB[2107.45,2138.65] vol=1.9x ATR=10.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 11:30:00 | 2202.06 | 2164.74 | 0.00 | T1 1.5R @ 2202.06 |
| Stop hit — per-position SL triggered | 2025-01-29 11:35:00 | 2187.00 | 2166.69 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:45:00 | 2076.90 | 2071.46 | 0.00 | ORB-long ORB[2030.25,2059.60] vol=4.0x ATR=7.61 |
| Stop hit — per-position SL triggered | 2025-03-05 11:35:00 | 2069.29 | 2071.60 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-03-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 10:40:00 | 2147.85 | 2129.18 | 0.00 | ORB-long ORB[2105.00,2127.85] vol=1.9x ATR=8.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 11:05:00 | 2160.43 | 2133.20 | 0.00 | T1 1.5R @ 2160.43 |
| Stop hit — per-position SL triggered | 2025-03-06 11:15:00 | 2147.85 | 2133.67 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:10:00 | 2200.00 | 2174.31 | 0.00 | ORB-long ORB[2151.60,2182.95] vol=2.2x ATR=9.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 10:25:00 | 2213.82 | 2184.89 | 0.00 | T1 1.5R @ 2213.82 |
| Stop hit — per-position SL triggered | 2025-03-13 10:35:00 | 2200.00 | 2191.33 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-03-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:30:00 | 2234.95 | 2222.50 | 0.00 | ORB-long ORB[2210.00,2231.95] vol=1.9x ATR=6.66 |
| Stop hit — per-position SL triggered | 2025-03-18 10:55:00 | 2228.29 | 2224.30 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:35:00 | 2261.30 | 2246.30 | 0.00 | ORB-long ORB[2235.30,2260.00] vol=3.4x ATR=6.87 |
| Stop hit — per-position SL triggered | 2025-03-19 10:40:00 | 2254.43 | 2247.72 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-03-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:55:00 | 2273.50 | 2281.38 | 0.00 | ORB-short ORB[2278.20,2305.85] vol=2.4x ATR=8.41 |
| Stop hit — per-position SL triggered | 2025-03-20 12:35:00 | 2281.91 | 2279.38 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-03-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-21 10:35:00 | 2259.85 | 2270.89 | 0.00 | ORB-short ORB[2279.00,2298.70] vol=1.8x ATR=7.58 |
| Stop hit — per-position SL triggered | 2025-03-21 14:55:00 | 2267.43 | 2266.34 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-03-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 11:05:00 | 2343.55 | 2321.53 | 0.00 | ORB-long ORB[2302.85,2332.00] vol=2.0x ATR=9.53 |
| Stop hit — per-position SL triggered | 2025-03-26 11:10:00 | 2334.02 | 2322.13 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:10:00 | 2379.00 | 2350.92 | 0.00 | ORB-long ORB[2322.00,2344.05] vol=3.8x ATR=7.84 |
| Stop hit — per-position SL triggered | 2025-03-27 11:25:00 | 2371.16 | 2355.30 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:55:00 | 2476.90 | 2493.55 | 0.00 | ORB-short ORB[2494.90,2521.20] vol=1.6x ATR=7.69 |
| Stop hit — per-position SL triggered | 2025-04-23 11:00:00 | 2484.59 | 2493.27 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 2425.70 | 2440.96 | 0.00 | ORB-short ORB[2432.30,2465.00] vol=2.2x ATR=8.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:40:00 | 2413.30 | 2435.98 | 0.00 | T1 1.5R @ 2413.30 |
| Target hit | 2025-04-25 12:10:00 | 2405.00 | 2398.20 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:25:00 | 1606.65 | 2024-05-13 11:05:00 | 1618.44 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2024-05-16 10:25:00 | 1680.95 | 2024-05-16 10:30:00 | 1695.42 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2024-05-16 10:25:00 | 1680.95 | 2024-05-16 10:35:00 | 1680.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-17 11:10:00 | 1703.05 | 2024-05-17 11:20:00 | 1707.58 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-05-18 09:45:00 | 1718.35 | 2024-05-21 09:15:00 | 1722.65 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-21 09:45:00 | 1703.60 | 2024-05-21 11:55:00 | 1691.39 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-05-21 09:45:00 | 1703.60 | 2024-05-21 15:20:00 | 1691.00 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2024-05-23 10:45:00 | 1691.00 | 2024-05-23 11:50:00 | 1684.32 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-05-23 10:45:00 | 1691.00 | 2024-05-23 12:20:00 | 1691.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-24 10:35:00 | 1666.00 | 2024-05-24 10:40:00 | 1659.17 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-24 10:35:00 | 1666.00 | 2024-05-24 12:50:00 | 1650.90 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2024-05-31 09:40:00 | 1578.20 | 2024-05-31 10:00:00 | 1567.78 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-05-31 09:40:00 | 1578.20 | 2024-05-31 10:45:00 | 1578.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-06 09:55:00 | 1702.35 | 2024-06-06 10:45:00 | 1710.48 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-06-07 10:00:00 | 1718.70 | 2024-06-07 10:05:00 | 1713.42 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-11 10:05:00 | 1693.80 | 2024-06-11 11:00:00 | 1687.44 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-06-11 10:05:00 | 1693.80 | 2024-06-11 14:40:00 | 1693.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-14 09:55:00 | 1714.50 | 2024-06-14 10:25:00 | 1707.49 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-06-14 09:55:00 | 1714.50 | 2024-06-14 14:25:00 | 1711.55 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2024-06-24 10:30:00 | 1799.55 | 2024-06-24 11:15:00 | 1810.94 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-06-24 10:30:00 | 1799.55 | 2024-06-24 14:15:00 | 1819.70 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2024-06-28 09:55:00 | 1797.65 | 2024-06-28 10:05:00 | 1805.19 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-02 09:30:00 | 1768.15 | 2024-07-02 10:05:00 | 1774.40 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-03 10:25:00 | 1740.00 | 2024-07-03 10:40:00 | 1745.52 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-11 10:30:00 | 1663.00 | 2024-07-11 11:10:00 | 1654.96 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-07-11 10:30:00 | 1663.00 | 2024-07-11 12:15:00 | 1663.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 09:35:00 | 1690.00 | 2024-07-16 11:55:00 | 1684.01 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-18 11:00:00 | 1694.65 | 2024-07-18 11:05:00 | 1700.66 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-24 09:30:00 | 1735.00 | 2024-07-24 09:35:00 | 1727.17 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-07-29 11:10:00 | 1763.80 | 2024-07-29 11:20:00 | 1757.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-30 09:35:00 | 1732.65 | 2024-07-30 09:50:00 | 1743.81 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-07-30 09:35:00 | 1732.65 | 2024-07-30 10:00:00 | 1732.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 10:15:00 | 1748.00 | 2024-07-31 10:20:00 | 1757.93 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-07-31 10:15:00 | 1748.00 | 2024-07-31 11:55:00 | 1750.20 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-08-13 10:25:00 | 1681.80 | 2024-08-13 11:10:00 | 1677.19 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-22 09:45:00 | 1740.80 | 2024-08-22 09:55:00 | 1735.95 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1810.00 | 2024-08-28 09:40:00 | 1816.96 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-30 10:30:00 | 1834.95 | 2024-08-30 12:15:00 | 1844.25 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-30 10:30:00 | 1834.95 | 2024-08-30 12:40:00 | 1875.00 | TARGET_HIT | 0.50 | 2.18% |
| SELL | retest1 | 2024-09-03 11:05:00 | 1990.80 | 2024-09-03 11:10:00 | 1997.94 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-23 10:35:00 | 2147.00 | 2024-09-23 10:50:00 | 2139.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-26 09:30:00 | 2151.65 | 2024-09-26 09:40:00 | 2139.68 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-10-03 09:40:00 | 2040.00 | 2024-10-03 10:00:00 | 2027.35 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-10-03 09:40:00 | 2040.00 | 2024-10-03 10:10:00 | 2040.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 10:25:00 | 2225.00 | 2024-10-29 11:15:00 | 2208.89 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-10-29 10:25:00 | 2225.00 | 2024-10-29 12:45:00 | 2225.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-06 10:35:00 | 2383.35 | 2024-11-06 10:40:00 | 2391.48 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-07 09:35:00 | 2422.00 | 2024-11-07 10:05:00 | 2410.24 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-11-13 09:40:00 | 2233.05 | 2024-11-13 09:50:00 | 2244.33 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-11-21 09:55:00 | 2213.35 | 2024-11-21 10:35:00 | 2226.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-11-26 10:30:00 | 2389.90 | 2024-11-26 11:40:00 | 2377.61 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-11-27 10:50:00 | 2378.95 | 2024-11-27 11:05:00 | 2371.62 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-28 10:05:00 | 2421.30 | 2024-11-28 10:10:00 | 2411.72 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-12-04 10:35:00 | 2355.50 | 2024-12-04 10:50:00 | 2363.73 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-06 10:30:00 | 2350.25 | 2024-12-06 12:20:00 | 2341.26 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-06 10:30:00 | 2350.25 | 2024-12-06 13:50:00 | 2350.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-09 11:00:00 | 2363.50 | 2024-12-09 11:05:00 | 2377.21 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-12-09 11:00:00 | 2363.50 | 2024-12-09 11:15:00 | 2363.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 10:50:00 | 2427.20 | 2024-12-10 10:55:00 | 2419.42 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-11 09:45:00 | 2452.45 | 2024-12-11 10:05:00 | 2442.46 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-26 11:05:00 | 2509.60 | 2024-12-26 11:50:00 | 2496.03 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-12-26 11:05:00 | 2509.60 | 2024-12-26 13:10:00 | 2509.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 09:50:00 | 2582.65 | 2024-12-30 09:55:00 | 2598.92 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-12-30 09:50:00 | 2582.65 | 2024-12-30 10:00:00 | 2582.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-03 10:50:00 | 2627.75 | 2025-01-03 10:55:00 | 2620.57 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-07 09:40:00 | 2561.80 | 2025-01-07 09:45:00 | 2549.02 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-01-13 09:30:00 | 2242.55 | 2025-01-13 09:35:00 | 2228.27 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2025-01-21 10:30:00 | 2299.05 | 2025-01-21 10:35:00 | 2308.05 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-01-23 11:10:00 | 2236.50 | 2025-01-23 14:50:00 | 2226.95 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-01-29 11:10:00 | 2187.00 | 2025-01-29 11:30:00 | 2202.06 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-01-29 11:10:00 | 2187.00 | 2025-01-29 11:35:00 | 2187.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-05 10:45:00 | 2076.90 | 2025-03-05 11:35:00 | 2069.29 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-03-06 10:40:00 | 2147.85 | 2025-03-06 11:05:00 | 2160.43 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-03-06 10:40:00 | 2147.85 | 2025-03-06 11:15:00 | 2147.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-13 10:10:00 | 2200.00 | 2025-03-13 10:25:00 | 2213.82 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-03-13 10:10:00 | 2200.00 | 2025-03-13 10:35:00 | 2200.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 10:30:00 | 2234.95 | 2025-03-18 10:55:00 | 2228.29 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-19 10:35:00 | 2261.30 | 2025-03-19 10:40:00 | 2254.43 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-20 10:55:00 | 2273.50 | 2025-03-20 12:35:00 | 2281.91 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-03-21 10:35:00 | 2259.85 | 2025-03-21 14:55:00 | 2267.43 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-26 11:05:00 | 2343.55 | 2025-03-26 11:10:00 | 2334.02 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-03-27 11:10:00 | 2379.00 | 2025-03-27 11:25:00 | 2371.16 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-04-23 10:55:00 | 2476.90 | 2025-04-23 11:00:00 | 2484.59 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-04-25 09:30:00 | 2425.70 | 2025-04-25 09:40:00 | 2413.30 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-04-25 09:30:00 | 2425.70 | 2025-04-25 12:10:00 | 2405.00 | TARGET_HIT | 0.50 | 0.85% |
