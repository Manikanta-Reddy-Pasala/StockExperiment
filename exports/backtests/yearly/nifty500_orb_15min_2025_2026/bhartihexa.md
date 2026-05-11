# Bharti Hexacom Ltd. (BHARTIHEXA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (16963 bars)
- **Last close:** 1499.90
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
| PARTIAL | 35 |
| TARGET_HIT | 16 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 105 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 54
- **Target hits / Stop hits / Partials:** 16 / 54 / 35
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 20.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 11 | 28.9% | 2 | 27 | 9 | -0.04% | -1.3% |
| BUY @ 2nd Alert (retest1) | 38 | 11 | 28.9% | 2 | 27 | 9 | -0.04% | -1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 67 | 40 | 59.7% | 14 | 27 | 26 | 0.33% | 22.0% |
| SELL @ 2nd Alert (retest1) | 67 | 40 | 59.7% | 14 | 27 | 26 | 0.33% | 22.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 105 | 51 | 48.6% | 16 | 54 | 35 | 0.20% | 20.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:45:00 | 1864.60 | 1843.93 | 0.00 | ORB-long ORB[1816.10,1838.00] vol=2.3x ATR=7.51 |
| Stop hit — per-position SL triggered | 2025-06-02 11:05:00 | 1857.09 | 1848.37 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 11:15:00 | 1809.90 | 1819.06 | 0.00 | ORB-short ORB[1810.40,1833.90] vol=2.3x ATR=5.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 11:35:00 | 1802.03 | 1816.94 | 0.00 | T1 1.5R @ 1802.03 |
| Target hit | 2025-06-09 15:20:00 | 1787.10 | 1804.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-06-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:10:00 | 1799.90 | 1791.26 | 0.00 | ORB-long ORB[1769.80,1794.90] vol=2.0x ATR=5.35 |
| Stop hit — per-position SL triggered | 2025-06-11 10:15:00 | 1794.55 | 1791.56 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 11:15:00 | 1803.50 | 1788.44 | 0.00 | ORB-long ORB[1771.00,1796.90] vol=2.5x ATR=5.50 |
| Stop hit — per-position SL triggered | 2025-06-16 11:20:00 | 1798.00 | 1788.78 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 11:15:00 | 1749.90 | 1766.90 | 0.00 | ORB-short ORB[1764.00,1785.20] vol=2.1x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 11:25:00 | 1743.02 | 1762.58 | 0.00 | T1 1.5R @ 1743.02 |
| Stop hit — per-position SL triggered | 2025-06-17 11:35:00 | 1749.90 | 1761.54 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:20:00 | 1746.00 | 1760.19 | 0.00 | ORB-short ORB[1746.60,1768.00] vol=1.6x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:30:00 | 1737.24 | 1757.43 | 0.00 | T1 1.5R @ 1737.24 |
| Stop hit — per-position SL triggered | 2025-06-19 10:35:00 | 1746.00 | 1757.07 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-07-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 11:05:00 | 1966.20 | 1953.49 | 0.00 | ORB-long ORB[1943.90,1958.50] vol=3.2x ATR=5.11 |
| Stop hit — per-position SL triggered | 2025-07-03 11:10:00 | 1961.09 | 1953.72 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 09:35:00 | 1865.00 | 1879.50 | 0.00 | ORB-short ORB[1872.10,1900.00] vol=1.7x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 10:40:00 | 1855.14 | 1868.42 | 0.00 | T1 1.5R @ 1855.14 |
| Target hit | 2025-07-07 15:20:00 | 1837.80 | 1851.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 1804.00 | 1817.00 | 0.00 | ORB-short ORB[1820.00,1845.70] vol=3.6x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:25:00 | 1797.62 | 1812.19 | 0.00 | T1 1.5R @ 1797.62 |
| Target hit | 2025-07-08 15:20:00 | 1784.00 | 1793.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2025-07-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:00:00 | 1811.70 | 1790.85 | 0.00 | ORB-long ORB[1771.20,1795.10] vol=2.1x ATR=7.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 10:45:00 | 1822.21 | 1800.72 | 0.00 | T1 1.5R @ 1822.21 |
| Target hit | 2025-07-21 15:20:00 | 1822.90 | 1817.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-07-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 11:00:00 | 1824.70 | 1815.94 | 0.00 | ORB-long ORB[1806.90,1823.50] vol=2.0x ATR=5.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:50:00 | 1833.07 | 1820.30 | 0.00 | T1 1.5R @ 1833.07 |
| Stop hit — per-position SL triggered | 2025-07-24 12:30:00 | 1824.70 | 1821.10 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:25:00 | 1800.90 | 1790.05 | 0.00 | ORB-long ORB[1766.20,1790.00] vol=2.0x ATR=7.64 |
| Stop hit — per-position SL triggered | 2025-07-28 10:40:00 | 1793.26 | 1790.67 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:50:00 | 1734.20 | 1744.33 | 0.00 | ORB-short ORB[1746.90,1761.50] vol=1.7x ATR=8.04 |
| Stop hit — per-position SL triggered | 2025-07-29 11:10:00 | 1742.24 | 1743.84 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 11:10:00 | 1774.10 | 1779.70 | 0.00 | ORB-short ORB[1775.40,1800.00] vol=1.9x ATR=5.23 |
| Stop hit — per-position SL triggered | 2025-07-30 11:25:00 | 1779.33 | 1779.47 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-08-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 09:35:00 | 1761.40 | 1767.88 | 0.00 | ORB-short ORB[1767.10,1788.90] vol=5.1x ATR=6.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:45:00 | 1751.76 | 1765.60 | 0.00 | T1 1.5R @ 1751.76 |
| Target hit | 2025-08-07 15:20:00 | 1741.20 | 1746.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-08-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 09:40:00 | 1745.10 | 1752.41 | 0.00 | ORB-short ORB[1747.10,1768.40] vol=1.6x ATR=6.02 |
| Stop hit — per-position SL triggered | 2025-08-12 09:45:00 | 1751.12 | 1752.21 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 11:00:00 | 1719.50 | 1732.62 | 0.00 | ORB-short ORB[1736.80,1756.90] vol=1.8x ATR=4.58 |
| Stop hit — per-position SL triggered | 2025-08-13 11:05:00 | 1724.08 | 1732.29 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 09:30:00 | 1694.10 | 1704.55 | 0.00 | ORB-short ORB[1701.00,1718.00] vol=3.0x ATR=6.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:50:00 | 1684.44 | 1695.02 | 0.00 | T1 1.5R @ 1684.44 |
| Stop hit — per-position SL triggered | 2025-08-18 10:25:00 | 1694.10 | 1692.92 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:00:00 | 1794.50 | 1784.66 | 0.00 | ORB-long ORB[1772.00,1791.40] vol=1.9x ATR=5.45 |
| Stop hit — per-position SL triggered | 2025-08-21 10:35:00 | 1789.05 | 1788.33 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:00:00 | 1782.00 | 1793.41 | 0.00 | ORB-short ORB[1791.00,1807.00] vol=2.1x ATR=5.48 |
| Stop hit — per-position SL triggered | 2025-08-22 10:20:00 | 1787.48 | 1791.52 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 11:10:00 | 1791.10 | 1780.27 | 0.00 | ORB-long ORB[1765.10,1785.70] vol=2.6x ATR=5.52 |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 1785.58 | 1780.64 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-09-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:45:00 | 1740.30 | 1754.25 | 0.00 | ORB-short ORB[1759.50,1779.80] vol=6.2x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:55:00 | 1733.06 | 1752.64 | 0.00 | T1 1.5R @ 1733.06 |
| Stop hit — per-position SL triggered | 2025-09-05 12:20:00 | 1740.30 | 1745.15 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 10:30:00 | 1749.10 | 1755.13 | 0.00 | ORB-short ORB[1760.00,1774.90] vol=2.1x ATR=5.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 11:20:00 | 1741.55 | 1752.28 | 0.00 | T1 1.5R @ 1741.55 |
| Target hit | 2025-09-10 13:05:00 | 1744.30 | 1743.88 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — SELL (started 2025-09-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:00:00 | 1708.00 | 1719.06 | 0.00 | ORB-short ORB[1715.60,1740.50] vol=1.9x ATR=4.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 12:15:00 | 1700.64 | 1709.43 | 0.00 | T1 1.5R @ 1700.64 |
| Target hit | 2025-09-12 15:20:00 | 1703.40 | 1704.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2025-09-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 10:45:00 | 1715.90 | 1725.95 | 0.00 | ORB-short ORB[1725.90,1746.90] vol=2.1x ATR=4.54 |
| Stop hit — per-position SL triggered | 2025-09-17 14:45:00 | 1720.44 | 1717.59 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 10:45:00 | 1709.00 | 1703.50 | 0.00 | ORB-long ORB[1685.30,1699.50] vol=11.6x ATR=5.47 |
| Stop hit — per-position SL triggered | 2025-09-23 10:55:00 | 1703.53 | 1703.63 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 11:15:00 | 1646.10 | 1656.50 | 0.00 | ORB-short ORB[1646.20,1666.80] vol=4.3x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 11:45:00 | 1639.66 | 1653.35 | 0.00 | T1 1.5R @ 1639.66 |
| Stop hit — per-position SL triggered | 2025-10-03 12:35:00 | 1646.10 | 1649.88 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:25:00 | 1749.90 | 1740.28 | 0.00 | ORB-long ORB[1735.40,1748.70] vol=3.3x ATR=6.44 |
| Stop hit — per-position SL triggered | 2025-10-15 10:45:00 | 1743.46 | 1741.95 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 11:05:00 | 1780.00 | 1769.19 | 0.00 | ORB-long ORB[1748.90,1767.70] vol=12.9x ATR=6.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:15:00 | 1789.29 | 1771.41 | 0.00 | T1 1.5R @ 1789.29 |
| Stop hit — per-position SL triggered | 2025-10-17 11:20:00 | 1780.00 | 1772.59 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 10:30:00 | 1797.40 | 1807.47 | 0.00 | ORB-short ORB[1806.70,1824.70] vol=2.0x ATR=5.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 11:40:00 | 1789.48 | 1803.44 | 0.00 | T1 1.5R @ 1789.48 |
| Target hit | 2025-10-23 15:20:00 | 1779.30 | 1792.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2025-10-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:40:00 | 1784.20 | 1782.49 | 0.00 | ORB-long ORB[1765.20,1781.90] vol=7.0x ATR=5.19 |
| Stop hit — per-position SL triggered | 2025-10-24 09:45:00 | 1779.01 | 1781.54 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:00:00 | 1832.30 | 1820.53 | 0.00 | ORB-long ORB[1798.30,1823.70] vol=1.9x ATR=7.58 |
| Stop hit — per-position SL triggered | 2025-10-27 10:15:00 | 1824.72 | 1821.19 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 11:00:00 | 1864.50 | 1875.87 | 0.00 | ORB-short ORB[1866.00,1885.90] vol=1.6x ATR=5.83 |
| Stop hit — per-position SL triggered | 2025-10-28 11:05:00 | 1870.33 | 1875.51 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 11:05:00 | 1837.40 | 1843.46 | 0.00 | ORB-short ORB[1846.00,1868.00] vol=2.7x ATR=5.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:35:00 | 1829.20 | 1841.58 | 0.00 | T1 1.5R @ 1829.20 |
| Target hit | 2025-11-06 15:20:00 | 1813.30 | 1824.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2025-11-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 11:05:00 | 1737.60 | 1743.04 | 0.00 | ORB-short ORB[1752.00,1762.00] vol=1.9x ATR=4.98 |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 1742.58 | 1742.78 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:10:00 | 1803.20 | 1795.23 | 0.00 | ORB-long ORB[1770.20,1796.00] vol=2.4x ATR=6.97 |
| Stop hit — per-position SL triggered | 2025-11-11 11:00:00 | 1796.23 | 1797.56 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-11-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:55:00 | 1794.10 | 1783.70 | 0.00 | ORB-long ORB[1774.40,1788.80] vol=5.0x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 11:45:00 | 1799.59 | 1786.43 | 0.00 | T1 1.5R @ 1799.59 |
| Stop hit — per-position SL triggered | 2025-11-17 12:05:00 | 1794.10 | 1787.66 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-11-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:55:00 | 1817.70 | 1833.87 | 0.00 | ORB-short ORB[1827.90,1851.90] vol=2.0x ATR=5.51 |
| Stop hit — per-position SL triggered | 2025-11-19 12:20:00 | 1823.21 | 1829.96 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-11-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 09:50:00 | 1841.50 | 1828.68 | 0.00 | ORB-long ORB[1817.00,1825.80] vol=2.1x ATR=5.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 10:40:00 | 1849.46 | 1838.45 | 0.00 | T1 1.5R @ 1849.46 |
| Target hit | 2025-11-20 13:45:00 | 1847.00 | 1847.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2025-11-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:55:00 | 1706.50 | 1719.16 | 0.00 | ORB-short ORB[1722.00,1742.30] vol=1.6x ATR=5.59 |
| Stop hit — per-position SL triggered | 2025-11-25 11:00:00 | 1712.09 | 1718.75 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 11:05:00 | 1776.40 | 1756.42 | 0.00 | ORB-long ORB[1746.00,1771.90] vol=1.6x ATR=6.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 11:40:00 | 1786.13 | 1762.12 | 0.00 | T1 1.5R @ 1786.13 |
| Stop hit — per-position SL triggered | 2025-11-26 12:30:00 | 1776.40 | 1768.14 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1772.30 | 1758.86 | 0.00 | ORB-long ORB[1745.00,1769.80] vol=1.9x ATR=6.74 |
| Stop hit — per-position SL triggered | 2025-11-28 12:45:00 | 1765.56 | 1769.59 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:50:00 | 1743.20 | 1753.85 | 0.00 | ORB-short ORB[1749.00,1761.00] vol=2.4x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:15:00 | 1736.61 | 1751.20 | 0.00 | T1 1.5R @ 1736.61 |
| Stop hit — per-position SL triggered | 2025-12-01 11:35:00 | 1743.20 | 1746.81 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-12-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:10:00 | 1724.50 | 1728.67 | 0.00 | ORB-short ORB[1728.20,1738.40] vol=2.2x ATR=4.00 |
| Stop hit — per-position SL triggered | 2025-12-03 11:35:00 | 1728.50 | 1728.36 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:30:00 | 1771.50 | 1760.42 | 0.00 | ORB-long ORB[1740.40,1764.90] vol=2.6x ATR=5.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 11:15:00 | 1779.51 | 1766.04 | 0.00 | T1 1.5R @ 1779.51 |
| Stop hit — per-position SL triggered | 2025-12-04 12:45:00 | 1771.50 | 1770.65 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:00:00 | 1732.60 | 1739.72 | 0.00 | ORB-short ORB[1740.20,1755.00] vol=3.4x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:35:00 | 1726.95 | 1737.63 | 0.00 | T1 1.5R @ 1726.95 |
| Stop hit — per-position SL triggered | 2025-12-08 12:30:00 | 1732.60 | 1735.09 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:05:00 | 1688.00 | 1692.25 | 0.00 | ORB-short ORB[1688.50,1702.20] vol=8.3x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:10:00 | 1681.20 | 1688.00 | 0.00 | T1 1.5R @ 1681.20 |
| Target hit | 2025-12-10 15:20:00 | 1669.90 | 1678.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:15:00 | 1682.00 | 1686.01 | 0.00 | ORB-short ORB[1682.50,1697.80] vol=1.7x ATR=4.88 |
| Stop hit — per-position SL triggered | 2025-12-12 15:00:00 | 1686.88 | 1682.30 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:00:00 | 1792.30 | 1776.00 | 0.00 | ORB-long ORB[1752.80,1772.40] vol=1.9x ATR=7.38 |
| Stop hit — per-position SL triggered | 2025-12-19 10:35:00 | 1784.92 | 1781.31 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 09:50:00 | 1798.70 | 1803.38 | 0.00 | ORB-short ORB[1802.10,1808.50] vol=1.7x ATR=4.05 |
| Stop hit — per-position SL triggered | 2025-12-24 10:30:00 | 1802.75 | 1801.07 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:55:00 | 1843.70 | 1836.93 | 0.00 | ORB-long ORB[1825.00,1842.00] vol=1.9x ATR=5.54 |
| Stop hit — per-position SL triggered | 2025-12-29 10:00:00 | 1838.16 | 1837.46 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:50:00 | 1781.10 | 1811.02 | 0.00 | ORB-short ORB[1824.70,1848.50] vol=1.9x ATR=7.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:00:00 | 1769.22 | 1802.46 | 0.00 | T1 1.5R @ 1769.22 |
| Stop hit — per-position SL triggered | 2025-12-30 11:50:00 | 1781.10 | 1787.36 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:15:00 | 1802.90 | 1799.20 | 0.00 | ORB-long ORB[1789.70,1802.60] vol=1.7x ATR=4.28 |
| Stop hit — per-position SL triggered | 2025-12-31 11:35:00 | 1798.62 | 1796.60 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-01-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 10:20:00 | 1791.10 | 1804.90 | 0.00 | ORB-short ORB[1801.90,1822.00] vol=1.6x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:30:00 | 1782.39 | 1797.48 | 0.00 | T1 1.5R @ 1782.39 |
| Target hit | 2026-01-02 12:50:00 | 1787.00 | 1783.42 | 0.00 | Trail-exit close>VWAP |

### Cycle 55 — SELL (started 2026-01-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:20:00 | 1680.10 | 1688.94 | 0.00 | ORB-short ORB[1684.10,1703.00] vol=1.9x ATR=4.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:15:00 | 1673.23 | 1683.76 | 0.00 | T1 1.5R @ 1673.23 |
| Stop hit — per-position SL triggered | 2026-01-13 11:45:00 | 1680.10 | 1679.34 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 11:00:00 | 1647.10 | 1659.27 | 0.00 | ORB-short ORB[1660.00,1672.80] vol=1.8x ATR=3.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:50:00 | 1641.52 | 1651.52 | 0.00 | T1 1.5R @ 1641.52 |
| Target hit | 2026-01-14 15:00:00 | 1641.90 | 1641.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — BUY (started 2026-01-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-20 11:00:00 | 1619.30 | 1604.35 | 0.00 | ORB-long ORB[1591.10,1614.00] vol=5.6x ATR=5.46 |
| Stop hit — per-position SL triggered | 2026-01-20 11:30:00 | 1613.84 | 1608.95 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-01-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-21 09:30:00 | 1607.60 | 1596.17 | 0.00 | ORB-long ORB[1583.80,1600.70] vol=1.5x ATR=6.83 |
| Stop hit — per-position SL triggered | 2026-01-21 09:35:00 | 1600.77 | 1596.83 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1610.90 | 1618.50 | 0.00 | ORB-short ORB[1615.00,1639.00] vol=2.0x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:45:00 | 1604.86 | 1616.92 | 0.00 | T1 1.5R @ 1604.86 |
| Stop hit — per-position SL triggered | 2026-01-22 12:40:00 | 1610.90 | 1613.20 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:05:00 | 1529.50 | 1532.28 | 0.00 | ORB-short ORB[1530.00,1545.00] vol=4.1x ATR=4.19 |
| Stop hit — per-position SL triggered | 2026-01-29 10:30:00 | 1533.69 | 1532.15 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:00:00 | 1562.50 | 1551.46 | 0.00 | ORB-long ORB[1541.00,1561.20] vol=1.5x ATR=5.20 |
| Stop hit — per-position SL triggered | 2026-02-01 11:10:00 | 1557.30 | 1552.92 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:50:00 | 1713.30 | 1724.85 | 0.00 | ORB-short ORB[1724.10,1735.00] vol=2.4x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:55:00 | 1706.78 | 1724.63 | 0.00 | T1 1.5R @ 1706.78 |
| Target hit | 2026-02-10 15:20:00 | 1678.60 | 1704.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2026-02-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:55:00 | 1678.20 | 1681.49 | 0.00 | ORB-short ORB[1681.00,1704.80] vol=2.5x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:45:00 | 1671.76 | 1679.85 | 0.00 | T1 1.5R @ 1671.76 |
| Target hit | 2026-02-24 15:20:00 | 1654.40 | 1667.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 1653.30 | 1656.74 | 0.00 | ORB-short ORB[1653.50,1664.70] vol=3.3x ATR=3.37 |
| Stop hit — per-position SL triggered | 2026-02-25 12:30:00 | 1656.67 | 1654.96 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 1586.70 | 1596.15 | 0.00 | ORB-short ORB[1590.00,1607.60] vol=1.6x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:10:00 | 1580.53 | 1593.27 | 0.00 | T1 1.5R @ 1580.53 |
| Stop hit — per-position SL triggered | 2026-03-10 11:20:00 | 1586.70 | 1592.34 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 1562.10 | 1573.22 | 0.00 | ORB-short ORB[1565.00,1578.90] vol=3.6x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:25:00 | 1554.43 | 1567.13 | 0.00 | T1 1.5R @ 1554.43 |
| Target hit | 2026-03-11 15:20:00 | 1533.40 | 1557.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2026-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:10:00 | 1526.50 | 1512.34 | 0.00 | ORB-long ORB[1501.60,1522.90] vol=2.0x ATR=4.98 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 1521.52 | 1512.49 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-03-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 10:20:00 | 1600.80 | 1583.51 | 0.00 | ORB-long ORB[1561.10,1584.30] vol=1.5x ATR=7.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 10:25:00 | 1612.03 | 1587.06 | 0.00 | T1 1.5R @ 1612.03 |
| Stop hit — per-position SL triggered | 2026-03-19 14:05:00 | 1600.80 | 1607.34 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:35:00 | 1595.00 | 1590.51 | 0.00 | ORB-long ORB[1580.00,1592.80] vol=2.3x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 14:00:00 | 1604.26 | 1595.18 | 0.00 | T1 1.5R @ 1604.26 |
| Stop hit — per-position SL triggered | 2026-03-25 14:20:00 | 1595.00 | 1595.26 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:15:00 | 1547.60 | 1561.82 | 0.00 | ORB-short ORB[1560.10,1582.50] vol=1.5x ATR=7.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 12:50:00 | 1535.65 | 1554.85 | 0.00 | T1 1.5R @ 1535.65 |
| Stop hit — per-position SL triggered | 2026-03-27 13:00:00 | 1547.60 | 1554.15 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-02 10:45:00 | 1864.60 | 2025-06-02 11:05:00 | 1857.09 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-06-09 11:15:00 | 1809.90 | 2025-06-09 11:35:00 | 1802.03 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-06-09 11:15:00 | 1809.90 | 2025-06-09 15:20:00 | 1787.10 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2025-06-11 10:10:00 | 1799.90 | 2025-06-11 10:15:00 | 1794.55 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-16 11:15:00 | 1803.50 | 2025-06-16 11:20:00 | 1798.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-17 11:15:00 | 1749.90 | 2025-06-17 11:25:00 | 1743.02 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-17 11:15:00 | 1749.90 | 2025-06-17 11:35:00 | 1749.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-19 10:20:00 | 1746.00 | 2025-06-19 10:30:00 | 1737.24 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-06-19 10:20:00 | 1746.00 | 2025-06-19 10:35:00 | 1746.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-03 11:05:00 | 1966.20 | 2025-07-03 11:10:00 | 1961.09 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-07 09:35:00 | 1865.00 | 2025-07-07 10:40:00 | 1855.14 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-07-07 09:35:00 | 1865.00 | 2025-07-07 15:20:00 | 1837.80 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2025-07-08 11:05:00 | 1804.00 | 2025-07-08 11:25:00 | 1797.62 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-08 11:05:00 | 1804.00 | 2025-07-08 15:20:00 | 1784.00 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2025-07-21 10:00:00 | 1811.70 | 2025-07-21 10:45:00 | 1822.21 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-07-21 10:00:00 | 1811.70 | 2025-07-21 15:20:00 | 1822.90 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2025-07-24 11:00:00 | 1824.70 | 2025-07-24 11:50:00 | 1833.07 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-07-24 11:00:00 | 1824.70 | 2025-07-24 12:30:00 | 1824.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-28 10:25:00 | 1800.90 | 2025-07-28 10:40:00 | 1793.26 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-07-29 10:50:00 | 1734.20 | 2025-07-29 11:10:00 | 1742.24 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-07-30 11:10:00 | 1774.10 | 2025-07-30 11:25:00 | 1779.33 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-07 09:35:00 | 1761.40 | 2025-08-07 09:45:00 | 1751.76 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-08-07 09:35:00 | 1761.40 | 2025-08-07 15:20:00 | 1741.20 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2025-08-12 09:40:00 | 1745.10 | 2025-08-12 09:45:00 | 1751.12 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-08-13 11:00:00 | 1719.50 | 2025-08-13 11:05:00 | 1724.08 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-18 09:30:00 | 1694.10 | 2025-08-18 09:50:00 | 1684.44 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-08-18 09:30:00 | 1694.10 | 2025-08-18 10:25:00 | 1694.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-21 10:00:00 | 1794.50 | 2025-08-21 10:35:00 | 1789.05 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-22 10:00:00 | 1782.00 | 2025-08-22 10:20:00 | 1787.48 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-03 11:10:00 | 1791.10 | 2025-09-03 11:15:00 | 1785.58 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-09-05 10:45:00 | 1740.30 | 2025-09-05 10:55:00 | 1733.06 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-09-05 10:45:00 | 1740.30 | 2025-09-05 12:20:00 | 1740.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-10 10:30:00 | 1749.10 | 2025-09-10 11:20:00 | 1741.55 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-09-10 10:30:00 | 1749.10 | 2025-09-10 13:05:00 | 1744.30 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-09-12 10:00:00 | 1708.00 | 2025-09-12 12:15:00 | 1700.64 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-09-12 10:00:00 | 1708.00 | 2025-09-12 15:20:00 | 1703.40 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-09-17 10:45:00 | 1715.90 | 2025-09-17 14:45:00 | 1720.44 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-23 10:45:00 | 1709.00 | 2025-09-23 10:55:00 | 1703.53 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-03 11:15:00 | 1646.10 | 2025-10-03 11:45:00 | 1639.66 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-10-03 11:15:00 | 1646.10 | 2025-10-03 12:35:00 | 1646.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 10:25:00 | 1749.90 | 2025-10-15 10:45:00 | 1743.46 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-17 11:05:00 | 1780.00 | 2025-10-17 11:15:00 | 1789.29 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-10-17 11:05:00 | 1780.00 | 2025-10-17 11:20:00 | 1780.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-23 10:30:00 | 1797.40 | 2025-10-23 11:40:00 | 1789.48 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-10-23 10:30:00 | 1797.40 | 2025-10-23 15:20:00 | 1779.30 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2025-10-24 09:40:00 | 1784.20 | 2025-10-24 09:45:00 | 1779.01 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-27 10:00:00 | 1832.30 | 2025-10-27 10:15:00 | 1824.72 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-10-28 11:00:00 | 1864.50 | 2025-10-28 11:05:00 | 1870.33 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-06 11:05:00 | 1837.40 | 2025-11-06 11:35:00 | 1829.20 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-11-06 11:05:00 | 1837.40 | 2025-11-06 15:20:00 | 1813.30 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2025-11-10 11:05:00 | 1737.60 | 2025-11-10 11:15:00 | 1742.58 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-11 10:10:00 | 1803.20 | 2025-11-11 11:00:00 | 1796.23 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-11-17 10:55:00 | 1794.10 | 2025-11-17 11:45:00 | 1799.59 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-11-17 10:55:00 | 1794.10 | 2025-11-17 12:05:00 | 1794.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-19 10:55:00 | 1817.70 | 2025-11-19 12:20:00 | 1823.21 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-20 09:50:00 | 1841.50 | 2025-11-20 10:40:00 | 1849.46 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-11-20 09:50:00 | 1841.50 | 2025-11-20 13:45:00 | 1847.00 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2025-11-25 10:55:00 | 1706.50 | 2025-11-25 11:00:00 | 1712.09 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-26 11:05:00 | 1776.40 | 2025-11-26 11:40:00 | 1786.13 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-11-26 11:05:00 | 1776.40 | 2025-11-26 12:30:00 | 1776.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-28 10:15:00 | 1772.30 | 2025-11-28 12:45:00 | 1765.56 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-12-01 10:50:00 | 1743.20 | 2025-12-01 11:15:00 | 1736.61 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-12-01 10:50:00 | 1743.20 | 2025-12-01 11:35:00 | 1743.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 11:10:00 | 1724.50 | 2025-12-03 11:35:00 | 1728.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-04 10:30:00 | 1771.50 | 2025-12-04 11:15:00 | 1779.51 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-04 10:30:00 | 1771.50 | 2025-12-04 12:45:00 | 1771.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 11:00:00 | 1732.60 | 2025-12-08 11:35:00 | 1726.95 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-08 11:00:00 | 1732.60 | 2025-12-08 12:30:00 | 1732.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-10 11:05:00 | 1688.00 | 2025-12-10 11:10:00 | 1681.20 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-12-10 11:05:00 | 1688.00 | 2025-12-10 15:20:00 | 1669.90 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2025-12-12 10:15:00 | 1682.00 | 2025-12-12 15:00:00 | 1686.88 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-19 10:00:00 | 1792.30 | 2025-12-19 10:35:00 | 1784.92 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-12-24 09:50:00 | 1798.70 | 2025-12-24 10:30:00 | 1802.75 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-29 09:55:00 | 1843.70 | 2025-12-29 10:00:00 | 1838.16 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-30 10:50:00 | 1781.10 | 2025-12-30 11:00:00 | 1769.22 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-12-30 10:50:00 | 1781.10 | 2025-12-30 11:50:00 | 1781.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 11:15:00 | 1802.90 | 2025-12-31 11:35:00 | 1798.62 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-02 10:20:00 | 1791.10 | 2026-01-02 10:30:00 | 1782.39 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-01-02 10:20:00 | 1791.10 | 2026-01-02 12:50:00 | 1787.00 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2026-01-13 10:20:00 | 1680.10 | 2026-01-13 11:15:00 | 1673.23 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-13 10:20:00 | 1680.10 | 2026-01-13 11:45:00 | 1680.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-14 11:00:00 | 1647.10 | 2026-01-14 11:50:00 | 1641.52 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-01-14 11:00:00 | 1647.10 | 2026-01-14 15:00:00 | 1641.90 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-01-20 11:00:00 | 1619.30 | 2026-01-20 11:30:00 | 1613.84 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-01-21 09:30:00 | 1607.60 | 2026-01-21 09:35:00 | 1600.77 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-01-22 11:15:00 | 1610.90 | 2026-01-22 11:45:00 | 1604.86 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-22 11:15:00 | 1610.90 | 2026-01-22 12:40:00 | 1610.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 10:05:00 | 1529.50 | 2026-01-29 10:30:00 | 1533.69 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-01 11:00:00 | 1562.50 | 2026-02-01 11:10:00 | 1557.30 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-10 10:50:00 | 1713.30 | 2026-02-10 10:55:00 | 1706.78 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-10 10:50:00 | 1713.30 | 2026-02-10 15:20:00 | 1678.60 | TARGET_HIT | 0.50 | 2.03% |
| SELL | retest1 | 2026-02-24 10:55:00 | 1678.20 | 2026-02-24 11:45:00 | 1671.76 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-24 10:55:00 | 1678.20 | 2026-02-24 15:20:00 | 1654.40 | TARGET_HIT | 0.50 | 1.42% |
| SELL | retest1 | 2026-02-25 11:00:00 | 1653.30 | 2026-02-25 12:30:00 | 1656.67 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-03-10 10:50:00 | 1586.70 | 2026-03-10 11:10:00 | 1580.53 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-10 10:50:00 | 1586.70 | 2026-03-10 11:20:00 | 1586.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:35:00 | 1562.10 | 2026-03-11 11:25:00 | 1554.43 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-11 10:35:00 | 1562.10 | 2026-03-11 15:20:00 | 1533.40 | TARGET_HIT | 0.50 | 1.84% |
| BUY | retest1 | 2026-03-12 11:10:00 | 1526.50 | 2026-03-12 11:15:00 | 1521.52 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-19 10:20:00 | 1600.80 | 2026-03-19 10:25:00 | 1612.03 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-19 10:20:00 | 1600.80 | 2026-03-19 14:05:00 | 1600.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 09:35:00 | 1595.00 | 2026-03-25 14:00:00 | 1604.26 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-25 09:35:00 | 1595.00 | 2026-03-25 14:20:00 | 1595.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 10:15:00 | 1547.60 | 2026-03-27 12:50:00 | 1535.65 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-03-27 10:15:00 | 1547.60 | 2026-03-27 13:00:00 | 1547.60 | STOP_HIT | 0.50 | 0.00% |
