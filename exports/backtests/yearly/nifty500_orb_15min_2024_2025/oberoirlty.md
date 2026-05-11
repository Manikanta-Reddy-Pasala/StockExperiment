# Oberoi Realty Ltd. (OBEROIRLTY)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35371 bars)
- **Last close:** 1710.00
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
| ENTRY1 | 57 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 9 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 48
- **Target hits / Stop hits / Partials:** 9 / 48 / 19
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 8.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 13 | 37.1% | 5 | 22 | 8 | 0.12% | 4.2% |
| BUY @ 2nd Alert (retest1) | 35 | 13 | 37.1% | 5 | 22 | 8 | 0.12% | 4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 41 | 15 | 36.6% | 4 | 26 | 11 | 0.11% | 4.6% |
| SELL @ 2nd Alert (retest1) | 41 | 15 | 36.6% | 4 | 26 | 11 | 0.11% | 4.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 76 | 28 | 36.8% | 9 | 48 | 19 | 0.12% | 8.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:40:00 | 1806.00 | 1783.64 | 0.00 | ORB-long ORB[1775.00,1793.90] vol=2.6x ATR=7.76 |
| Stop hit — per-position SL triggered | 2024-05-24 10:45:00 | 1798.24 | 1784.64 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:15:00 | 1844.45 | 1832.33 | 0.00 | ORB-long ORB[1812.50,1834.60] vol=2.4x ATR=6.15 |
| Stop hit — per-position SL triggered | 2024-05-29 10:30:00 | 1838.30 | 1833.83 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 11:15:00 | 1919.00 | 1924.60 | 0.00 | ORB-short ORB[1922.15,1941.75] vol=1.7x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 12:05:00 | 1912.76 | 1922.66 | 0.00 | T1 1.5R @ 1912.76 |
| Stop hit — per-position SL triggered | 2024-06-12 13:25:00 | 1919.00 | 1919.83 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:05:00 | 1946.35 | 1936.71 | 0.00 | ORB-long ORB[1918.45,1943.25] vol=4.2x ATR=5.13 |
| Stop hit — per-position SL triggered | 2024-06-13 10:10:00 | 1941.22 | 1936.76 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:40:00 | 1887.50 | 1876.58 | 0.00 | ORB-long ORB[1862.05,1880.00] vol=1.7x ATR=6.21 |
| Stop hit — per-position SL triggered | 2024-06-20 09:50:00 | 1881.29 | 1878.20 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:55:00 | 1776.70 | 1761.88 | 0.00 | ORB-long ORB[1758.00,1772.90] vol=1.8x ATR=6.62 |
| Stop hit — per-position SL triggered | 2024-06-28 11:15:00 | 1770.08 | 1763.89 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-07-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:05:00 | 1804.60 | 1785.98 | 0.00 | ORB-long ORB[1770.00,1792.00] vol=2.2x ATR=7.28 |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 1797.32 | 1788.85 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-07-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:35:00 | 1789.15 | 1799.95 | 0.00 | ORB-short ORB[1799.10,1809.90] vol=1.5x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:45:00 | 1782.39 | 1797.92 | 0.00 | T1 1.5R @ 1782.39 |
| Stop hit — per-position SL triggered | 2024-07-04 11:40:00 | 1789.15 | 1794.32 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 11:15:00 | 1776.10 | 1783.33 | 0.00 | ORB-short ORB[1788.45,1810.00] vol=2.0x ATR=4.88 |
| Stop hit — per-position SL triggered | 2024-07-05 12:35:00 | 1780.98 | 1782.02 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:35:00 | 1763.75 | 1773.79 | 0.00 | ORB-short ORB[1773.30,1798.00] vol=2.6x ATR=5.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 09:40:00 | 1755.35 | 1770.24 | 0.00 | T1 1.5R @ 1755.35 |
| Target hit | 2024-07-08 11:10:00 | 1755.60 | 1751.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2024-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 09:30:00 | 1753.50 | 1743.28 | 0.00 | ORB-long ORB[1725.00,1748.30] vol=2.8x ATR=5.36 |
| Stop hit — per-position SL triggered | 2024-07-10 09:35:00 | 1748.14 | 1746.94 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:20:00 | 1743.65 | 1735.33 | 0.00 | ORB-long ORB[1715.65,1741.70] vol=2.0x ATR=6.23 |
| Stop hit — per-position SL triggered | 2024-07-16 10:30:00 | 1737.42 | 1735.87 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 1705.35 | 1717.06 | 0.00 | ORB-short ORB[1708.20,1731.45] vol=1.8x ATR=6.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:40:00 | 1694.88 | 1710.26 | 0.00 | T1 1.5R @ 1694.88 |
| Stop hit — per-position SL triggered | 2024-07-18 09:45:00 | 1705.35 | 1710.39 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 1765.90 | 1752.48 | 0.00 | ORB-long ORB[1733.05,1759.25] vol=1.9x ATR=7.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:35:00 | 1777.11 | 1758.87 | 0.00 | T1 1.5R @ 1777.11 |
| Target hit | 2024-07-26 13:55:00 | 1783.70 | 1783.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2024-07-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:40:00 | 1813.00 | 1796.35 | 0.00 | ORB-long ORB[1787.95,1802.95] vol=1.8x ATR=6.91 |
| Stop hit — per-position SL triggered | 2024-07-29 09:45:00 | 1806.09 | 1798.59 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 10:15:00 | 1803.40 | 1805.67 | 0.00 | ORB-short ORB[1805.15,1823.25] vol=1.6x ATR=5.45 |
| Stop hit — per-position SL triggered | 2024-07-30 11:05:00 | 1808.85 | 1805.01 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:25:00 | 1858.95 | 1841.27 | 0.00 | ORB-long ORB[1833.30,1846.75] vol=2.8x ATR=5.51 |
| Stop hit — per-position SL triggered | 2024-07-31 10:30:00 | 1853.44 | 1842.60 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:30:00 | 1799.00 | 1785.60 | 0.00 | ORB-long ORB[1769.60,1794.50] vol=1.7x ATR=7.83 |
| Stop hit — per-position SL triggered | 2024-08-09 09:55:00 | 1791.17 | 1794.47 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 11:00:00 | 1753.40 | 1761.90 | 0.00 | ORB-short ORB[1777.30,1793.85] vol=1.7x ATR=7.64 |
| Stop hit — per-position SL triggered | 2024-08-13 11:10:00 | 1761.04 | 1761.72 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 1742.30 | 1734.57 | 0.00 | ORB-long ORB[1724.00,1738.35] vol=2.5x ATR=6.57 |
| Stop hit — per-position SL triggered | 2024-08-16 09:50:00 | 1735.73 | 1738.76 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:55:00 | 1756.10 | 1761.62 | 0.00 | ORB-short ORB[1762.30,1780.00] vol=1.5x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-08-19 11:00:00 | 1760.78 | 1761.67 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 11:10:00 | 1744.75 | 1752.61 | 0.00 | ORB-short ORB[1748.60,1766.45] vol=1.7x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 11:30:00 | 1738.70 | 1751.15 | 0.00 | T1 1.5R @ 1738.70 |
| Stop hit — per-position SL triggered | 2024-08-21 13:40:00 | 1744.75 | 1744.33 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:50:00 | 1721.00 | 1717.09 | 0.00 | ORB-long ORB[1699.95,1719.95] vol=2.1x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 11:10:00 | 1726.58 | 1719.09 | 0.00 | T1 1.5R @ 1726.58 |
| Target hit | 2024-08-26 15:20:00 | 1737.15 | 1731.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1713.50 | 1725.43 | 0.00 | ORB-short ORB[1726.00,1738.90] vol=2.1x ATR=4.98 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 1718.48 | 1719.44 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 1697.10 | 1704.37 | 0.00 | ORB-short ORB[1703.30,1719.45] vol=4.1x ATR=4.16 |
| Stop hit — per-position SL triggered | 2024-08-29 09:50:00 | 1701.26 | 1701.59 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 11:00:00 | 1765.65 | 1772.19 | 0.00 | ORB-short ORB[1766.00,1788.95] vol=1.6x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 12:05:00 | 1758.27 | 1769.98 | 0.00 | T1 1.5R @ 1758.27 |
| Target hit | 2024-09-02 15:20:00 | 1763.40 | 1763.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 11:15:00 | 1779.90 | 1773.96 | 0.00 | ORB-long ORB[1752.20,1778.00] vol=1.9x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 11:45:00 | 1786.45 | 1777.73 | 0.00 | T1 1.5R @ 1786.45 |
| Stop hit — per-position SL triggered | 2024-09-03 11:55:00 | 1779.90 | 1778.39 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 11:05:00 | 1793.75 | 1788.78 | 0.00 | ORB-long ORB[1776.25,1791.90] vol=1.8x ATR=4.80 |
| Stop hit — per-position SL triggered | 2024-09-05 11:10:00 | 1788.95 | 1788.87 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:55:00 | 1739.70 | 1757.83 | 0.00 | ORB-short ORB[1775.15,1787.50] vol=2.9x ATR=6.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 11:00:00 | 1729.36 | 1754.90 | 0.00 | T1 1.5R @ 1729.36 |
| Stop hit — per-position SL triggered | 2024-09-06 11:25:00 | 1739.70 | 1750.77 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:50:00 | 1773.60 | 1766.29 | 0.00 | ORB-long ORB[1753.10,1768.60] vol=3.1x ATR=4.36 |
| Stop hit — per-position SL triggered | 2024-09-11 11:30:00 | 1769.24 | 1768.83 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 1841.40 | 1835.14 | 0.00 | ORB-long ORB[1817.55,1840.30] vol=2.3x ATR=6.61 |
| Stop hit — per-position SL triggered | 2024-09-16 09:35:00 | 1834.79 | 1835.37 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:10:00 | 1798.60 | 1815.60 | 0.00 | ORB-short ORB[1815.30,1830.40] vol=2.1x ATR=7.78 |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 1806.38 | 1813.56 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 10:35:00 | 1912.50 | 1901.92 | 0.00 | ORB-long ORB[1876.55,1898.65] vol=1.9x ATR=7.06 |
| Stop hit — per-position SL triggered | 2024-09-25 11:20:00 | 1905.44 | 1903.93 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 09:55:00 | 1916.05 | 1903.82 | 0.00 | ORB-long ORB[1890.05,1910.00] vol=2.5x ATR=7.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 10:05:00 | 1927.02 | 1909.02 | 0.00 | T1 1.5R @ 1927.02 |
| Stop hit — per-position SL triggered | 2024-09-30 10:10:00 | 1916.05 | 1909.34 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:05:00 | 1790.05 | 1797.55 | 0.00 | ORB-short ORB[1805.40,1819.30] vol=2.6x ATR=7.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:20:00 | 1779.34 | 1791.69 | 0.00 | T1 1.5R @ 1779.34 |
| Target hit | 2024-10-07 15:20:00 | 1749.15 | 1761.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2024-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:35:00 | 1951.60 | 1937.52 | 0.00 | ORB-long ORB[1925.00,1939.80] vol=2.0x ATR=7.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:50:00 | 1963.32 | 1944.39 | 0.00 | T1 1.5R @ 1963.32 |
| Target hit | 2024-10-14 15:20:00 | 1994.05 | 1989.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2024-10-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:30:00 | 1945.80 | 1925.91 | 0.00 | ORB-long ORB[1907.05,1927.75] vol=3.2x ATR=7.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:55:00 | 1956.96 | 1932.57 | 0.00 | T1 1.5R @ 1956.96 |
| Target hit | 2024-10-31 15:20:00 | 1973.25 | 1954.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-11-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 10:10:00 | 2035.50 | 2025.03 | 0.00 | ORB-long ORB[2010.00,2033.35] vol=2.4x ATR=8.12 |
| Stop hit — per-position SL triggered | 2024-11-06 10:25:00 | 2027.38 | 2025.98 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:30:00 | 2063.20 | 2039.00 | 0.00 | ORB-long ORB[2022.00,2051.60] vol=3.4x ATR=9.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 09:35:00 | 2077.66 | 2045.44 | 0.00 | T1 1.5R @ 2077.66 |
| Stop hit — per-position SL triggered | 2024-11-07 09:55:00 | 2063.20 | 2055.35 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:40:00 | 2011.65 | 2030.65 | 0.00 | ORB-short ORB[2020.20,2044.00] vol=1.5x ATR=8.62 |
| Stop hit — per-position SL triggered | 2024-11-12 10:20:00 | 2020.27 | 2022.96 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:30:00 | 1954.50 | 1973.93 | 0.00 | ORB-short ORB[1960.90,1989.30] vol=1.5x ATR=9.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 10:00:00 | 1940.36 | 1965.15 | 0.00 | T1 1.5R @ 1940.36 |
| Stop hit — per-position SL triggered | 2024-11-18 10:15:00 | 1954.50 | 1963.42 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 11:15:00 | 1901.20 | 1912.24 | 0.00 | ORB-short ORB[1915.45,1938.80] vol=1.6x ATR=7.67 |
| Stop hit — per-position SL triggered | 2024-11-22 11:55:00 | 1908.87 | 1911.38 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:45:00 | 2172.80 | 2160.19 | 0.00 | ORB-long ORB[2142.65,2167.85] vol=2.5x ATR=7.90 |
| Stop hit — per-position SL triggered | 2024-12-09 09:55:00 | 2164.90 | 2160.66 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 11:00:00 | 2123.60 | 2133.07 | 0.00 | ORB-short ORB[2131.10,2148.45] vol=1.9x ATR=5.30 |
| Stop hit — per-position SL triggered | 2024-12-10 11:05:00 | 2128.90 | 2132.72 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:30:00 | 2083.00 | 2105.02 | 0.00 | ORB-short ORB[2119.20,2139.50] vol=4.9x ATR=6.89 |
| Stop hit — per-position SL triggered | 2024-12-13 11:55:00 | 2089.89 | 2092.98 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:45:00 | 2290.20 | 2268.55 | 0.00 | ORB-long ORB[2241.25,2269.80] vol=2.7x ATR=10.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:05:00 | 2305.33 | 2276.42 | 0.00 | T1 1.5R @ 2305.33 |
| Target hit | 2024-12-17 15:20:00 | 2312.45 | 2310.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2024-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:30:00 | 2308.05 | 2320.87 | 0.00 | ORB-short ORB[2311.00,2343.65] vol=1.8x ATR=7.69 |
| Stop hit — per-position SL triggered | 2024-12-27 09:35:00 | 2315.74 | 2320.23 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 11:00:00 | 2302.85 | 2317.78 | 0.00 | ORB-short ORB[2315.80,2337.70] vol=1.7x ATR=6.39 |
| Stop hit — per-position SL triggered | 2024-12-30 11:05:00 | 2309.24 | 2315.66 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:10:00 | 2265.75 | 2278.19 | 0.00 | ORB-short ORB[2269.95,2296.80] vol=1.6x ATR=9.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 10:45:00 | 2251.44 | 2274.09 | 0.00 | T1 1.5R @ 2251.44 |
| Target hit | 2025-01-06 15:20:00 | 2207.15 | 2227.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2025-01-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:35:00 | 2249.85 | 2236.49 | 0.00 | ORB-long ORB[2211.00,2239.35] vol=1.8x ATR=10.92 |
| Stop hit — per-position SL triggered | 2025-01-07 09:50:00 | 2238.93 | 2239.91 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 11:05:00 | 2198.50 | 2200.38 | 0.00 | ORB-short ORB[2222.65,2252.00] vol=2.3x ATR=9.54 |
| Stop hit — per-position SL triggered | 2025-01-10 11:15:00 | 2208.04 | 2200.50 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:15:00 | 1820.15 | 1825.55 | 0.00 | ORB-short ORB[1822.20,1840.00] vol=1.9x ATR=5.80 |
| Stop hit — per-position SL triggered | 2025-02-06 11:40:00 | 1825.95 | 1825.40 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-03-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:50:00 | 1608.25 | 1616.32 | 0.00 | ORB-short ORB[1619.00,1636.45] vol=4.0x ATR=5.70 |
| Stop hit — per-position SL triggered | 2025-03-20 11:00:00 | 1613.95 | 1616.01 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-03-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 10:35:00 | 1609.60 | 1619.83 | 0.00 | ORB-short ORB[1612.10,1633.95] vol=2.3x ATR=5.18 |
| Stop hit — per-position SL triggered | 2025-03-27 10:55:00 | 1614.78 | 1617.43 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:00:00 | 1623.70 | 1637.13 | 0.00 | ORB-short ORB[1632.00,1656.00] vol=2.1x ATR=6.07 |
| Stop hit — per-position SL triggered | 2025-04-17 10:20:00 | 1629.77 | 1635.05 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-04-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:25:00 | 1695.50 | 1707.61 | 0.00 | ORB-short ORB[1709.00,1732.00] vol=2.1x ATR=6.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:40:00 | 1686.12 | 1705.46 | 0.00 | T1 1.5R @ 1686.12 |
| Stop hit — per-position SL triggered | 2025-04-23 11:20:00 | 1695.50 | 1702.08 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 09:45:00 | 1675.00 | 1689.03 | 0.00 | ORB-short ORB[1686.20,1708.00] vol=2.7x ATR=6.98 |
| Stop hit — per-position SL triggered | 2025-04-24 09:50:00 | 1681.98 | 1688.35 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-24 10:40:00 | 1806.00 | 2024-05-24 10:45:00 | 1798.24 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-05-29 10:15:00 | 1844.45 | 2024-05-29 10:30:00 | 1838.30 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-12 11:15:00 | 1919.00 | 2024-06-12 12:05:00 | 1912.76 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-06-12 11:15:00 | 1919.00 | 2024-06-12 13:25:00 | 1919.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-13 10:05:00 | 1946.35 | 2024-06-13 10:10:00 | 1941.22 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-20 09:40:00 | 1887.50 | 2024-06-20 09:50:00 | 1881.29 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-28 10:55:00 | 1776.70 | 2024-06-28 11:15:00 | 1770.08 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-02 10:05:00 | 1804.60 | 2024-07-02 10:15:00 | 1797.32 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-04 10:35:00 | 1789.15 | 2024-07-04 10:45:00 | 1782.39 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-07-04 10:35:00 | 1789.15 | 2024-07-04 11:40:00 | 1789.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-05 11:15:00 | 1776.10 | 2024-07-05 12:35:00 | 1780.98 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-08 09:35:00 | 1763.75 | 2024-07-08 09:40:00 | 1755.35 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-07-08 09:35:00 | 1763.75 | 2024-07-08 11:10:00 | 1755.60 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-10 09:30:00 | 1753.50 | 2024-07-10 09:35:00 | 1748.14 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-16 10:20:00 | 1743.65 | 2024-07-16 10:30:00 | 1737.42 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1705.35 | 2024-07-18 09:40:00 | 1694.88 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-07-18 09:30:00 | 1705.35 | 2024-07-18 09:45:00 | 1705.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 09:30:00 | 1765.90 | 2024-07-26 09:35:00 | 1777.11 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-07-26 09:30:00 | 1765.90 | 2024-07-26 13:55:00 | 1783.70 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2024-07-29 09:40:00 | 1813.00 | 2024-07-29 09:45:00 | 1806.09 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-07-30 10:15:00 | 1803.40 | 2024-07-30 11:05:00 | 1808.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-31 10:25:00 | 1858.95 | 2024-07-31 10:30:00 | 1853.44 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-09 09:30:00 | 1799.00 | 2024-08-09 09:55:00 | 1791.17 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-08-13 11:00:00 | 1753.40 | 2024-08-13 11:10:00 | 1761.04 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-16 09:30:00 | 1742.30 | 2024-08-16 09:50:00 | 1735.73 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-19 10:55:00 | 1756.10 | 2024-08-19 11:00:00 | 1760.78 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-21 11:10:00 | 1744.75 | 2024-08-21 11:30:00 | 1738.70 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-08-21 11:10:00 | 1744.75 | 2024-08-21 13:40:00 | 1744.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 10:50:00 | 1721.00 | 2024-08-26 11:10:00 | 1726.58 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-26 10:50:00 | 1721.00 | 2024-08-26 15:20:00 | 1737.15 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1713.50 | 2024-08-28 09:40:00 | 1718.48 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-29 09:30:00 | 1697.10 | 2024-08-29 09:50:00 | 1701.26 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-02 11:00:00 | 1765.65 | 2024-09-02 12:05:00 | 1758.27 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-02 11:00:00 | 1765.65 | 2024-09-02 15:20:00 | 1763.40 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-09-03 11:15:00 | 1779.90 | 2024-09-03 11:45:00 | 1786.45 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-03 11:15:00 | 1779.90 | 2024-09-03 11:55:00 | 1779.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-05 11:05:00 | 1793.75 | 2024-09-05 11:10:00 | 1788.95 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-06 10:55:00 | 1739.70 | 2024-09-06 11:00:00 | 1729.36 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-06 10:55:00 | 1739.70 | 2024-09-06 11:25:00 | 1739.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 10:50:00 | 1773.60 | 2024-09-11 11:30:00 | 1769.24 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-16 09:30:00 | 1841.40 | 2024-09-16 09:35:00 | 1834.79 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-19 10:10:00 | 1798.60 | 2024-09-19 10:15:00 | 1806.38 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-09-25 10:35:00 | 1912.50 | 2024-09-25 11:20:00 | 1905.44 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-30 09:55:00 | 1916.05 | 2024-09-30 10:05:00 | 1927.02 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-09-30 09:55:00 | 1916.05 | 2024-09-30 10:10:00 | 1916.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 10:05:00 | 1790.05 | 2024-10-07 10:20:00 | 1779.34 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-10-07 10:05:00 | 1790.05 | 2024-10-07 15:20:00 | 1749.15 | TARGET_HIT | 0.50 | 2.28% |
| BUY | retest1 | 2024-10-14 09:35:00 | 1951.60 | 2024-10-14 09:50:00 | 1963.32 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-10-14 09:35:00 | 1951.60 | 2024-10-14 15:20:00 | 1994.05 | TARGET_HIT | 0.50 | 2.18% |
| BUY | retest1 | 2024-10-31 10:30:00 | 1945.80 | 2024-10-31 10:55:00 | 1956.96 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-10-31 10:30:00 | 1945.80 | 2024-10-31 15:20:00 | 1973.25 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2024-11-06 10:10:00 | 2035.50 | 2024-11-06 10:25:00 | 2027.38 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-11-07 09:30:00 | 2063.20 | 2024-11-07 09:35:00 | 2077.66 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-11-07 09:30:00 | 2063.20 | 2024-11-07 09:55:00 | 2063.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-12 09:40:00 | 2011.65 | 2024-11-12 10:20:00 | 2020.27 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-11-18 09:30:00 | 1954.50 | 2024-11-18 10:00:00 | 1940.36 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-11-18 09:30:00 | 1954.50 | 2024-11-18 10:15:00 | 1954.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-22 11:15:00 | 1901.20 | 2024-11-22 11:55:00 | 1908.87 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-12-09 09:45:00 | 2172.80 | 2024-12-09 09:55:00 | 2164.90 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-10 11:00:00 | 2123.60 | 2024-12-10 11:05:00 | 2128.90 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-13 10:30:00 | 2083.00 | 2024-12-13 11:55:00 | 2089.89 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-17 09:45:00 | 2290.20 | 2024-12-17 10:05:00 | 2305.33 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-12-17 09:45:00 | 2290.20 | 2024-12-17 15:20:00 | 2312.45 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2024-12-27 09:30:00 | 2308.05 | 2024-12-27 09:35:00 | 2315.74 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-30 11:00:00 | 2302.85 | 2024-12-30 11:05:00 | 2309.24 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-06 10:10:00 | 2265.75 | 2025-01-06 10:45:00 | 2251.44 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-01-06 10:10:00 | 2265.75 | 2025-01-06 15:20:00 | 2207.15 | TARGET_HIT | 0.50 | 2.59% |
| BUY | retest1 | 2025-01-07 09:35:00 | 2249.85 | 2025-01-07 09:50:00 | 2238.93 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-01-10 11:05:00 | 2198.50 | 2025-01-10 11:15:00 | 2208.04 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-02-06 11:15:00 | 1820.15 | 2025-02-06 11:40:00 | 1825.95 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-20 10:50:00 | 1608.25 | 2025-03-20 11:00:00 | 1613.95 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-03-27 10:35:00 | 1609.60 | 2025-03-27 10:55:00 | 1614.78 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-17 10:00:00 | 1623.70 | 2025-04-17 10:20:00 | 1629.77 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-23 10:25:00 | 1695.50 | 2025-04-23 10:40:00 | 1686.12 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-04-23 10:25:00 | 1695.50 | 2025-04-23 11:20:00 | 1695.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-24 09:45:00 | 1675.00 | 2025-04-24 09:50:00 | 1681.98 | STOP_HIT | 1.00 | -0.42% |
