# Mankind Pharma Ltd. (MANKIND)

## Backtest Summary

- **Window:** 2023-05-09 09:15:00 → 2026-05-08 15:15:00 (5191 bars)
- **Last close:** 2423.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 254 |
| ALERT1 | 163 |
| ALERT2 | 160 |
| ALERT2_SKIP | 116 |
| ALERT3 | 342 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 111 |
| PARTIAL | 6 |
| TARGET_HIT | 7 |
| STOP_HIT | 106 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 119 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 84
- **Target hits / Stop hits / Partials:** 7 / 106 / 6
- **Avg / median % per leg:** 0.17% / -0.95%
- **Sum % (uncompounded):** 20.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 12 | 21.8% | 7 | 48 | 0 | 0.48% | 26.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 55 | 12 | 21.8% | 7 | 48 | 0 | 0.48% | 26.2% |
| SELL (all) | 64 | 23 | 35.9% | 0 | 58 | 6 | -0.09% | -5.6% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.88% | -1.8% |
| SELL @ 3rd Alert (retest2) | 62 | 23 | 37.1% | 0 | 56 | 6 | -0.06% | -3.9% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.88% | -1.8% |
| retest2 (combined) | 117 | 35 | 29.9% | 7 | 104 | 6 | 0.19% | 22.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 15:15:00 | 1379.20 | 1386.93 | 1387.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 11:15:00 | 1363.00 | 1380.70 | 1384.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 09:15:00 | 1358.75 | 1357.73 | 1365.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 1358.75 | 1357.73 | 1365.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 1358.75 | 1357.73 | 1365.27 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 09:15:00 | 1366.00 | 1333.18 | 1330.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 15:15:00 | 1370.85 | 1357.31 | 1345.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 10:15:00 | 1352.25 | 1358.57 | 1348.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 10:15:00 | 1352.25 | 1358.57 | 1348.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 1352.25 | 1358.57 | 1348.31 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 12:15:00 | 1337.85 | 1347.84 | 1348.77 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 14:15:00 | 1360.00 | 1350.36 | 1349.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 09:15:00 | 1371.10 | 1354.45 | 1351.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 10:15:00 | 1438.50 | 1445.82 | 1419.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 12:15:00 | 1422.85 | 1437.42 | 1419.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 12:15:00 | 1422.85 | 1437.42 | 1419.76 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 1669.85 | 1690.48 | 1690.95 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 12:15:00 | 1724.00 | 1692.71 | 1689.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-23 13:15:00 | 1753.70 | 1704.91 | 1695.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 14:15:00 | 1696.00 | 1703.13 | 1695.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 14:15:00 | 1696.00 | 1703.13 | 1695.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 14:15:00 | 1696.00 | 1703.13 | 1695.44 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 11:15:00 | 1692.00 | 1696.53 | 1696.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-27 13:15:00 | 1681.85 | 1692.24 | 1694.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 09:15:00 | 1717.50 | 1693.28 | 1694.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 09:15:00 | 1717.50 | 1693.28 | 1694.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 1717.50 | 1693.28 | 1694.27 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 1702.80 | 1695.18 | 1695.04 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 15:15:00 | 1695.00 | 1701.00 | 1701.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 09:15:00 | 1690.10 | 1698.82 | 1700.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 1706.45 | 1687.89 | 1692.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 1706.45 | 1687.89 | 1692.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 1706.45 | 1687.89 | 1692.03 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 12:15:00 | 1702.15 | 1695.75 | 1695.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 1712.60 | 1700.99 | 1697.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 09:15:00 | 1720.00 | 1724.11 | 1715.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 14:15:00 | 1730.25 | 1732.14 | 1726.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 1730.25 | 1732.14 | 1726.83 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 10:15:00 | 1860.00 | 1897.58 | 1901.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 13:15:00 | 1834.90 | 1871.92 | 1887.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 12:15:00 | 1867.65 | 1839.69 | 1860.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 12:15:00 | 1867.65 | 1839.69 | 1860.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 12:15:00 | 1867.65 | 1839.69 | 1860.82 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-08-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 10:15:00 | 1890.00 | 1803.80 | 1792.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 1926.15 | 1872.13 | 1835.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 14:15:00 | 1888.00 | 1890.81 | 1860.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 09:15:00 | 1822.00 | 1874.20 | 1858.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 1822.00 | 1874.20 | 1858.08 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 12:15:00 | 1809.05 | 1846.84 | 1848.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 13:15:00 | 1805.25 | 1838.53 | 1844.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 09:15:00 | 1800.00 | 1793.50 | 1810.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 13:15:00 | 1805.00 | 1795.09 | 1805.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 13:15:00 | 1805.00 | 1795.09 | 1805.54 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 10:15:00 | 1841.95 | 1811.67 | 1810.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 12:15:00 | 1847.00 | 1823.19 | 1816.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 1824.90 | 1830.43 | 1822.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 1824.90 | 1830.43 | 1822.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 1824.90 | 1830.43 | 1822.58 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 15:15:00 | 1823.05 | 1824.94 | 1825.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 09:15:00 | 1806.00 | 1821.15 | 1823.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 15:15:00 | 1826.50 | 1818.83 | 1820.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 15:15:00 | 1826.50 | 1818.83 | 1820.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 15:15:00 | 1826.50 | 1818.83 | 1820.76 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 1831.45 | 1822.08 | 1821.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 11:15:00 | 1833.05 | 1824.27 | 1822.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 15:15:00 | 1828.00 | 1828.07 | 1825.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 15:15:00 | 1828.00 | 1828.07 | 1825.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 15:15:00 | 1828.00 | 1828.07 | 1825.49 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 11:15:00 | 1838.15 | 1843.32 | 1843.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 15:15:00 | 1832.40 | 1838.67 | 1841.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 10:15:00 | 1841.20 | 1838.40 | 1840.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 10:15:00 | 1841.20 | 1838.40 | 1840.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 10:15:00 | 1841.20 | 1838.40 | 1840.48 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 15:15:00 | 1800.55 | 1785.72 | 1784.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 10:15:00 | 1805.50 | 1792.51 | 1787.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 12:15:00 | 1780.05 | 1790.67 | 1787.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 12:15:00 | 1780.05 | 1790.67 | 1787.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 1780.05 | 1790.67 | 1787.93 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 14:15:00 | 1710.25 | 1772.69 | 1780.11 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 15:15:00 | 1773.25 | 1763.09 | 1761.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 12:15:00 | 1790.00 | 1772.49 | 1766.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 09:15:00 | 1780.00 | 1781.44 | 1773.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 09:15:00 | 1780.00 | 1781.44 | 1773.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 1780.00 | 1781.44 | 1773.69 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 11:15:00 | 1756.00 | 1770.79 | 1772.60 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 13:15:00 | 1769.80 | 1768.99 | 1768.98 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 14:15:00 | 1768.60 | 1768.91 | 1768.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 15:15:00 | 1768.00 | 1768.73 | 1768.86 | Break + close below crossover candle low |

### Cycle 24 — BUY (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-12 09:15:00 | 1775.95 | 1770.17 | 1769.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-12 11:15:00 | 1790.85 | 1776.68 | 1772.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 14:15:00 | 1778.80 | 1780.29 | 1775.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 14:15:00 | 1778.80 | 1780.29 | 1775.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 14:15:00 | 1778.80 | 1780.29 | 1775.66 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 10:15:00 | 1776.00 | 1791.62 | 1792.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 09:15:00 | 1741.00 | 1769.21 | 1779.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 10:15:00 | 1718.00 | 1713.85 | 1730.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 12:15:00 | 1732.00 | 1718.65 | 1729.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 1732.00 | 1718.65 | 1729.85 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 12:15:00 | 1741.50 | 1734.98 | 1734.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-22 14:15:00 | 1752.15 | 1737.83 | 1735.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 09:15:00 | 1734.00 | 1738.21 | 1736.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 1734.00 | 1738.21 | 1736.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 1734.00 | 1738.21 | 1736.43 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 12:15:00 | 1725.00 | 1734.83 | 1735.31 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 15:15:00 | 1750.50 | 1738.05 | 1736.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 09:15:00 | 1763.50 | 1743.14 | 1739.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 14:15:00 | 1749.05 | 1755.47 | 1747.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 14:15:00 | 1749.05 | 1755.47 | 1747.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 1749.05 | 1755.47 | 1747.81 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 12:15:00 | 1771.80 | 1789.36 | 1790.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 09:15:00 | 1755.05 | 1777.34 | 1784.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 11:15:00 | 1780.40 | 1774.29 | 1781.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 11:15:00 | 1780.40 | 1774.29 | 1781.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 11:15:00 | 1780.40 | 1774.29 | 1781.32 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 15:15:00 | 1796.05 | 1785.39 | 1784.87 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 1781.00 | 1784.51 | 1784.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 1761.95 | 1775.17 | 1779.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 1792.85 | 1778.71 | 1780.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 1792.85 | 1778.71 | 1780.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1792.85 | 1778.71 | 1780.80 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 11:15:00 | 1784.00 | 1782.37 | 1782.27 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 12:15:00 | 1776.05 | 1781.11 | 1781.70 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 12:15:00 | 1783.00 | 1780.55 | 1780.50 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 13:15:00 | 1773.55 | 1779.15 | 1779.87 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 10:15:00 | 1790.00 | 1780.67 | 1780.21 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 14:15:00 | 1776.55 | 1779.63 | 1779.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 15:15:00 | 1771.05 | 1777.92 | 1779.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 12:15:00 | 1781.30 | 1776.78 | 1778.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 12:15:00 | 1781.30 | 1776.78 | 1778.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 1781.30 | 1776.78 | 1778.03 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 14:15:00 | 1790.00 | 1781.06 | 1779.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 09:15:00 | 1823.90 | 1790.58 | 1784.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 09:15:00 | 1805.70 | 1808.86 | 1799.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 12:15:00 | 1809.95 | 1814.40 | 1808.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 1809.95 | 1814.40 | 1808.82 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 10:15:00 | 1798.10 | 1804.79 | 1805.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 1780.45 | 1797.38 | 1800.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 15:15:00 | 1735.90 | 1734.98 | 1754.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1735.80 | 1713.51 | 1728.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1735.80 | 1713.51 | 1728.93 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 1755.05 | 1736.28 | 1734.92 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 13:15:00 | 1728.50 | 1735.36 | 1735.46 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 14:15:00 | 1755.00 | 1739.29 | 1737.23 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 09:15:00 | 1723.00 | 1740.74 | 1740.83 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 10:15:00 | 1753.95 | 1740.69 | 1739.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 12:15:00 | 1780.25 | 1751.10 | 1744.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 15:15:00 | 1767.00 | 1780.72 | 1770.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 15:15:00 | 1767.00 | 1780.72 | 1770.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 1767.00 | 1780.72 | 1770.83 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 11:15:00 | 1743.00 | 1761.16 | 1763.27 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 09:15:00 | 1782.00 | 1762.67 | 1762.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 10:15:00 | 1804.00 | 1770.94 | 1766.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 11:15:00 | 1837.65 | 1840.85 | 1821.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 14:15:00 | 1823.35 | 1836.60 | 1824.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 1823.35 | 1836.60 | 1824.61 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 1937.05 | 1953.28 | 1954.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 15:15:00 | 1909.00 | 1920.75 | 1930.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 10:15:00 | 1924.00 | 1920.15 | 1928.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 12:15:00 | 1910.05 | 1919.23 | 1926.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 12:15:00 | 1910.05 | 1919.23 | 1926.44 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 10:15:00 | 1926.55 | 1909.59 | 1909.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 1960.10 | 1922.91 | 1916.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 14:15:00 | 1938.10 | 1939.89 | 1929.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 14:15:00 | 1938.10 | 1939.89 | 1929.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 14:15:00 | 1938.10 | 1939.89 | 1929.11 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 15:15:00 | 1958.00 | 1968.21 | 1969.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 09:15:00 | 1919.00 | 1958.37 | 1964.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 1863.25 | 1862.24 | 1880.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 13:15:00 | 1880.00 | 1867.73 | 1877.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 13:15:00 | 1880.00 | 1867.73 | 1877.63 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 15:15:00 | 1936.00 | 1884.87 | 1881.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 10:15:00 | 1942.65 | 1922.76 | 1906.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 11:15:00 | 1933.25 | 1939.40 | 1926.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 12:15:00 | 1925.80 | 1936.68 | 1926.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 1925.80 | 1936.68 | 1926.28 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1883.25 | 1917.73 | 1918.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 1875.00 | 1909.18 | 1914.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 12:15:00 | 1927.00 | 1904.91 | 1910.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 12:15:00 | 1927.00 | 1904.91 | 1910.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 12:15:00 | 1927.00 | 1904.91 | 1910.14 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 1937.40 | 1912.97 | 1912.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 1967.00 | 1932.14 | 1925.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 1968.35 | 1977.47 | 1956.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 09:15:00 | 1971.10 | 1980.31 | 1975.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 1971.10 | 1980.31 | 1975.03 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 11:15:00 | 2066.95 | 2082.31 | 2082.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 13:15:00 | 2056.00 | 2074.34 | 2078.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-08 15:15:00 | 2022.00 | 2021.33 | 2042.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 2013.00 | 2019.66 | 2040.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 2013.00 | 2019.66 | 2040.14 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 09:15:00 | 2098.00 | 2055.74 | 2050.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 12:15:00 | 2130.00 | 2077.44 | 2062.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 10:15:00 | 2192.95 | 2203.58 | 2163.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 14:15:00 | 2190.00 | 2200.81 | 2188.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 14:15:00 | 2190.00 | 2200.81 | 2188.69 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 2130.40 | 2188.30 | 2195.39 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 09:15:00 | 2192.15 | 2174.98 | 2174.20 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 2151.70 | 2172.64 | 2173.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 2141.15 | 2162.87 | 2168.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 09:15:00 | 2062.85 | 2048.77 | 2078.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 09:15:00 | 2062.85 | 2048.77 | 2078.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 2062.85 | 2048.77 | 2078.89 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 10:15:00 | 2064.35 | 2047.56 | 2046.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 10:15:00 | 2081.30 | 2057.96 | 2052.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 10:15:00 | 2083.90 | 2085.05 | 2071.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 2227.75 | 2229.36 | 2200.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 2227.75 | 2229.36 | 2200.88 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 11:15:00 | 2159.95 | 2192.91 | 2196.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 09:15:00 | 2156.95 | 2176.84 | 2186.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 12:15:00 | 2187.20 | 2177.83 | 2184.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 12:15:00 | 2187.20 | 2177.83 | 2184.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 2187.20 | 2177.83 | 2184.41 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 2209.95 | 2189.40 | 2188.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 2239.60 | 2207.44 | 2198.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 14:15:00 | 2208.45 | 2221.16 | 2210.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 14:15:00 | 2208.45 | 2221.16 | 2210.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 2208.45 | 2221.16 | 2210.12 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 2187.40 | 2209.45 | 2210.80 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 12:15:00 | 2211.45 | 2206.91 | 2206.47 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 13:15:00 | 2202.10 | 2205.95 | 2206.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 2186.95 | 2202.15 | 2204.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 10:15:00 | 2114.00 | 2108.06 | 2134.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 2071.60 | 2097.28 | 2117.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 2071.60 | 2097.28 | 2117.42 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 13:15:00 | 2123.50 | 2100.00 | 2099.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 14:15:00 | 2138.50 | 2107.70 | 2103.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-02 09:15:00 | 2139.55 | 2146.26 | 2131.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 09:15:00 | 2139.55 | 2146.26 | 2131.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 2139.55 | 2146.26 | 2131.89 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 13:15:00 | 2139.90 | 2155.89 | 2157.20 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 09:15:00 | 2163.00 | 2154.79 | 2154.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 10:15:00 | 2192.05 | 2162.24 | 2157.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 14:15:00 | 2160.25 | 2167.28 | 2162.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 14:15:00 | 2160.25 | 2167.28 | 2162.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 2160.25 | 2167.28 | 2162.28 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 12:15:00 | 2149.85 | 2160.01 | 2160.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 14:15:00 | 2141.60 | 2154.73 | 2157.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 2098.20 | 2083.75 | 2106.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 11:15:00 | 2101.00 | 2087.20 | 2106.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 2101.00 | 2087.20 | 2106.15 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-03-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 10:15:00 | 2154.00 | 2118.83 | 2115.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 11:15:00 | 2216.65 | 2138.39 | 2124.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 15:15:00 | 2161.00 | 2163.60 | 2143.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 15:15:00 | 2161.00 | 2163.60 | 2143.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 2161.00 | 2163.60 | 2143.25 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 13:15:00 | 2127.45 | 2148.97 | 2150.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 14:15:00 | 2105.00 | 2140.18 | 2146.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 14:15:00 | 2125.00 | 2110.85 | 2124.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 14:15:00 | 2125.00 | 2110.85 | 2124.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 2125.00 | 2110.85 | 2124.57 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 09:15:00 | 2181.65 | 2136.85 | 2132.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 14:15:00 | 2211.05 | 2166.92 | 2150.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 2154.90 | 2169.97 | 2154.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 2154.90 | 2169.97 | 2154.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 2154.90 | 2169.97 | 2154.80 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 11:15:00 | 2304.35 | 2336.15 | 2336.90 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 2366.95 | 2340.92 | 2338.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 09:15:00 | 2375.55 | 2362.46 | 2354.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 13:15:00 | 2369.65 | 2376.83 | 2364.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 14:15:00 | 2369.75 | 2375.42 | 2365.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 2369.75 | 2375.42 | 2365.34 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 13:15:00 | 2340.45 | 2359.20 | 2361.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 2335.00 | 2347.26 | 2354.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 12:15:00 | 2279.45 | 2278.06 | 2297.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 13:15:00 | 2297.95 | 2278.06 | 2297.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 2291.50 | 2280.75 | 2297.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:00:00 | 2291.50 | 2280.75 | 2297.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 2340.85 | 2292.86 | 2298.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:00:00 | 2340.85 | 2292.86 | 2298.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 2339.40 | 2302.17 | 2302.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:15:00 | 2334.35 | 2302.17 | 2302.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 2359.55 | 2313.64 | 2307.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 12:15:00 | 2376.15 | 2326.15 | 2313.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 10:15:00 | 2355.15 | 2360.22 | 2338.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-19 11:00:00 | 2355.15 | 2360.22 | 2338.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 15:15:00 | 2324.00 | 2347.01 | 2339.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 09:45:00 | 2356.85 | 2349.44 | 2341.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 12:00:00 | 2352.05 | 2351.36 | 2344.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 09:15:00 | 2369.35 | 2348.40 | 2344.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 15:15:00 | 2365.00 | 2389.62 | 2391.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-04-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 15:15:00 | 2365.00 | 2389.62 | 2391.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 12:15:00 | 2359.05 | 2374.73 | 2382.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 14:15:00 | 2380.05 | 2373.43 | 2380.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 14:15:00 | 2380.05 | 2373.43 | 2380.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 2380.05 | 2373.43 | 2380.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 15:00:00 | 2380.05 | 2373.43 | 2380.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 2375.00 | 2373.74 | 2380.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:15:00 | 2391.00 | 2373.74 | 2380.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 2398.60 | 2378.72 | 2381.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:45:00 | 2415.10 | 2378.72 | 2381.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 2391.50 | 2381.27 | 2382.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:30:00 | 2397.65 | 2381.27 | 2382.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 2371.85 | 2379.11 | 2381.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 10:30:00 | 2359.80 | 2371.62 | 2376.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:15:00 | 2241.81 | 2274.84 | 2300.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-09 09:15:00 | 2204.80 | 2196.78 | 2226.81 | SL hit (close>ema200) qty=0.50 sl=2196.78 alert=retest2 |

### Cycle 76 — BUY (started 2024-05-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 13:15:00 | 2251.00 | 2213.65 | 2213.05 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 12:15:00 | 2190.40 | 2211.62 | 2213.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 13:15:00 | 2177.25 | 2204.75 | 2210.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 09:15:00 | 2240.55 | 2204.38 | 2208.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 2240.55 | 2204.38 | 2208.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 2240.55 | 2204.38 | 2208.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:00:00 | 2240.55 | 2204.38 | 2208.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 2257.00 | 2214.91 | 2212.53 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 12:15:00 | 2188.50 | 2218.27 | 2220.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 10:15:00 | 2181.55 | 2197.96 | 2208.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 2116.85 | 2107.95 | 2132.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-18 11:30:00 | 2113.00 | 2108.66 | 2130.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 2131.75 | 2113.81 | 2129.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 2131.75 | 2113.81 | 2129.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 2140.85 | 2119.22 | 2130.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 2140.85 | 2119.22 | 2130.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 2121.95 | 2119.77 | 2129.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:30:00 | 2133.40 | 2119.77 | 2129.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 2092.70 | 2101.92 | 2116.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 10:15:00 | 2089.50 | 2101.92 | 2116.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 11:30:00 | 2091.55 | 2100.61 | 2108.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 12:00:00 | 2090.55 | 2100.61 | 2108.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 13:45:00 | 2088.05 | 2097.12 | 2105.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 2097.00 | 2098.38 | 2104.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 2106.35 | 2098.38 | 2104.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 2103.65 | 2099.44 | 2104.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 2103.65 | 2099.44 | 2104.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 2111.35 | 2101.82 | 2105.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 2111.35 | 2101.82 | 2105.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 2093.60 | 2100.17 | 2104.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 12:30:00 | 2091.95 | 2099.14 | 2103.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:30:00 | 2091.65 | 2095.92 | 2101.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:30:00 | 2075.80 | 2089.40 | 2096.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 11:15:00 | 2115.05 | 2097.10 | 2098.81 | SL hit (close>static) qty=1.00 sl=2113.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 2103.45 | 2100.53 | 2100.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 14:15:00 | 2115.70 | 2103.56 | 2101.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 14:15:00 | 2115.35 | 2118.94 | 2112.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 14:15:00 | 2115.35 | 2118.94 | 2112.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 2115.35 | 2118.94 | 2112.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 2115.35 | 2118.94 | 2112.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 2117.75 | 2118.70 | 2112.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 2108.85 | 2118.70 | 2112.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 2115.05 | 2117.97 | 2112.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 12:15:00 | 2137.25 | 2115.50 | 2112.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 10:30:00 | 2144.50 | 2120.78 | 2116.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 14:15:00 | 2087.70 | 2124.38 | 2121.72 | SL hit (close<static) qty=1.00 sl=2091.30 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 15:15:00 | 2097.00 | 2118.90 | 2119.48 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 2148.50 | 2122.32 | 2120.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 15:15:00 | 2170.00 | 2140.30 | 2131.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 09:15:00 | 2122.55 | 2136.75 | 2130.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 2122.55 | 2136.75 | 2130.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2122.55 | 2136.75 | 2130.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 2122.55 | 2136.75 | 2130.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 2137.85 | 2136.97 | 2130.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 11:15:00 | 2145.85 | 2136.97 | 2130.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 2065.30 | 2120.05 | 2126.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 2065.30 | 2120.05 | 2126.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 2013.40 | 2098.72 | 2115.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 2100.70 | 2092.47 | 2109.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 13:15:00 | 2100.70 | 2092.47 | 2109.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 2100.70 | 2092.47 | 2109.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 14:00:00 | 2100.70 | 2092.47 | 2109.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 2111.25 | 2096.22 | 2109.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 2111.25 | 2096.22 | 2109.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 2092.00 | 2095.38 | 2108.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 2146.00 | 2095.38 | 2108.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 2163.90 | 2109.08 | 2113.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:30:00 | 2145.30 | 2109.08 | 2113.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 2183.90 | 2124.05 | 2119.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 11:15:00 | 2190.00 | 2177.13 | 2170.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 12:15:00 | 2238.05 | 2239.77 | 2222.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 13:00:00 | 2238.05 | 2239.77 | 2222.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 2234.00 | 2244.99 | 2238.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 2236.40 | 2244.99 | 2238.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 2233.00 | 2242.59 | 2238.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 12:30:00 | 2241.30 | 2240.52 | 2237.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 2221.00 | 2236.62 | 2236.17 | SL hit (close<static) qty=1.00 sl=2225.05 alert=retest2 |

### Cycle 85 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 2212.60 | 2231.82 | 2234.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 15:15:00 | 2202.60 | 2225.97 | 2231.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 2217.00 | 2207.18 | 2215.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 2217.00 | 2207.18 | 2215.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 2217.00 | 2207.18 | 2215.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 2228.20 | 2207.18 | 2215.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 2204.75 | 2206.70 | 2214.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 2196.25 | 2207.54 | 2213.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 2163.85 | 2198.80 | 2208.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 2143.95 | 2135.64 | 2134.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 2143.95 | 2135.64 | 2134.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 2161.60 | 2140.83 | 2137.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 10:15:00 | 2141.10 | 2144.93 | 2140.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 10:15:00 | 2141.10 | 2144.93 | 2140.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 2141.10 | 2144.93 | 2140.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:00:00 | 2141.10 | 2144.93 | 2140.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 2141.95 | 2144.33 | 2140.48 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 15:15:00 | 2129.75 | 2137.01 | 2137.92 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 2140.00 | 2137.67 | 2137.60 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 15:15:00 | 2133.85 | 2136.91 | 2137.26 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 2166.95 | 2142.92 | 2139.96 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 2124.60 | 2144.84 | 2147.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 2100.00 | 2135.87 | 2142.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 2116.30 | 2099.97 | 2118.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 2116.30 | 2099.97 | 2118.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 2116.30 | 2099.97 | 2118.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:30:00 | 2142.15 | 2099.97 | 2118.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 2119.90 | 2103.96 | 2118.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:30:00 | 2127.50 | 2103.96 | 2118.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 2128.75 | 2108.92 | 2119.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:45:00 | 2128.05 | 2108.92 | 2119.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 2150.55 | 2120.15 | 2121.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:30:00 | 2154.05 | 2120.15 | 2121.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 10:15:00 | 2150.00 | 2126.12 | 2124.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 09:15:00 | 2175.80 | 2149.08 | 2137.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 12:15:00 | 2147.10 | 2149.11 | 2140.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 13:00:00 | 2147.10 | 2149.11 | 2140.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 2128.00 | 2144.89 | 2139.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:00:00 | 2128.00 | 2144.89 | 2139.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 2115.90 | 2139.09 | 2137.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 2115.90 | 2139.09 | 2137.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 15:15:00 | 2122.00 | 2135.67 | 2135.91 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 2146.20 | 2136.97 | 2136.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 14:15:00 | 2150.00 | 2142.91 | 2139.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 2142.55 | 2143.81 | 2140.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 2142.55 | 2143.81 | 2140.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2142.55 | 2143.81 | 2140.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:15:00 | 2166.70 | 2146.11 | 2142.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 13:00:00 | 2162.15 | 2149.32 | 2144.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 10:15:00 | 2128.55 | 2143.23 | 2143.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 10:15:00 | 2128.55 | 2143.23 | 2143.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 11:15:00 | 2119.00 | 2138.38 | 2141.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 09:15:00 | 2129.85 | 2127.51 | 2133.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 2129.85 | 2127.51 | 2133.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 2129.85 | 2127.51 | 2133.89 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 10:15:00 | 2144.00 | 2134.13 | 2133.59 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 14:15:00 | 2126.25 | 2133.07 | 2133.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 11:15:00 | 2111.70 | 2124.42 | 2128.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 2109.60 | 2097.47 | 2107.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 2109.60 | 2097.47 | 2107.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2109.60 | 2097.47 | 2107.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 2109.60 | 2097.47 | 2107.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 2100.35 | 2098.05 | 2106.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 2098.10 | 2104.62 | 2107.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:45:00 | 2099.95 | 2103.91 | 2106.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 10:30:00 | 2095.25 | 2105.54 | 2107.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 12:15:00 | 2136.00 | 2113.81 | 2110.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 12:15:00 | 2136.00 | 2113.81 | 2110.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 14:15:00 | 2149.60 | 2123.63 | 2115.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 09:15:00 | 2087.00 | 2119.24 | 2115.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 2087.00 | 2119.24 | 2115.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 2087.00 | 2119.24 | 2115.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 2087.00 | 2119.24 | 2115.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 10:15:00 | 2072.00 | 2109.79 | 2111.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-26 11:15:00 | 2064.70 | 2100.78 | 2107.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 09:15:00 | 2068.30 | 2066.08 | 2077.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 2068.30 | 2066.08 | 2077.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 2068.30 | 2066.08 | 2077.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 15:00:00 | 2058.50 | 2064.75 | 2072.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 1955.57 | 1995.37 | 2008.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-05 12:15:00 | 1993.55 | 1991.16 | 2002.71 | SL hit (close>ema200) qty=0.50 sl=1991.16 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 2027.60 | 2007.90 | 2006.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 12:15:00 | 2032.00 | 2012.72 | 2008.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 09:15:00 | 2033.50 | 2045.36 | 2034.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 09:15:00 | 2033.50 | 2045.36 | 2034.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 2033.50 | 2045.36 | 2034.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:00:00 | 2033.50 | 2045.36 | 2034.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 2065.55 | 2049.40 | 2036.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 11:30:00 | 2071.20 | 2054.56 | 2040.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 2092.00 | 2064.54 | 2050.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-16 11:15:00 | 2278.32 | 2227.67 | 2194.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 14:15:00 | 2393.95 | 2425.25 | 2427.97 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 2458.80 | 2430.60 | 2429.41 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 2389.80 | 2427.02 | 2428.77 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 2475.00 | 2426.99 | 2424.77 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 13:15:00 | 2401.00 | 2441.68 | 2443.98 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 2454.05 | 2443.73 | 2443.42 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 09:15:00 | 2421.05 | 2442.23 | 2443.52 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 2485.65 | 2444.60 | 2441.73 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 2444.35 | 2461.35 | 2462.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 11:15:00 | 2417.05 | 2449.67 | 2456.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 10:15:00 | 2420.40 | 2418.62 | 2434.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 10:45:00 | 2420.65 | 2418.62 | 2434.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 2432.95 | 2412.05 | 2425.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 2432.95 | 2412.05 | 2425.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 2442.00 | 2418.04 | 2426.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 2536.95 | 2418.04 | 2426.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 2535.05 | 2441.44 | 2436.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 2639.15 | 2526.12 | 2485.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 13:15:00 | 2686.60 | 2691.02 | 2648.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 14:00:00 | 2686.60 | 2691.02 | 2648.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 2661.00 | 2687.15 | 2657.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:30:00 | 2703.10 | 2686.81 | 2660.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 15:00:00 | 2685.00 | 2680.74 | 2665.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 10:15:00 | 2651.00 | 2670.37 | 2664.03 | SL hit (close<static) qty=1.00 sl=2652.80 alert=retest2 |

### Cycle 111 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 2616.00 | 2652.24 | 2656.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 13:15:00 | 2591.50 | 2640.09 | 2650.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 13:15:00 | 2568.70 | 2543.49 | 2570.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 13:15:00 | 2568.70 | 2543.49 | 2570.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 2568.70 | 2543.49 | 2570.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:00:00 | 2568.70 | 2543.49 | 2570.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 2573.65 | 2549.52 | 2570.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 2573.65 | 2549.52 | 2570.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 2585.95 | 2556.81 | 2572.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 2599.00 | 2556.81 | 2572.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 2548.50 | 2555.14 | 2569.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 10:45:00 | 2535.00 | 2552.65 | 2567.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:15:00 | 2532.95 | 2552.65 | 2567.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 15:15:00 | 2531.75 | 2542.07 | 2556.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 11:15:00 | 2593.25 | 2563.54 | 2562.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 11:15:00 | 2593.25 | 2563.54 | 2562.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 14:15:00 | 2617.15 | 2580.36 | 2571.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 2575.60 | 2582.06 | 2573.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 2575.60 | 2582.06 | 2573.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 2575.60 | 2582.06 | 2573.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:45:00 | 2576.00 | 2582.06 | 2573.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 2563.65 | 2578.38 | 2572.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 2563.65 | 2578.38 | 2572.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 2583.05 | 2579.31 | 2573.72 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 2561.60 | 2570.26 | 2570.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 15:15:00 | 2526.00 | 2561.40 | 2566.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2600.60 | 2569.24 | 2569.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 2600.60 | 2569.24 | 2569.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 2600.60 | 2569.24 | 2569.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 2601.05 | 2569.24 | 2569.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 10:15:00 | 2597.10 | 2574.81 | 2572.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 2646.55 | 2599.82 | 2586.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 2681.95 | 2686.25 | 2659.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 09:15:00 | 2668.60 | 2686.25 | 2659.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 2649.25 | 2678.85 | 2659.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:15:00 | 2647.00 | 2678.85 | 2659.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 2637.00 | 2670.48 | 2657.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 2637.00 | 2670.48 | 2657.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 2738.05 | 2682.18 | 2664.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:30:00 | 2758.05 | 2699.03 | 2673.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 15:00:00 | 2760.75 | 2740.37 | 2715.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 14:45:00 | 2744.90 | 2738.50 | 2726.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 12:15:00 | 2689.90 | 2716.56 | 2719.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 2689.90 | 2716.56 | 2719.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 2676.35 | 2703.15 | 2711.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 2682.40 | 2658.54 | 2672.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 2682.40 | 2658.54 | 2672.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 2682.40 | 2658.54 | 2672.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:00:00 | 2682.40 | 2658.54 | 2672.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 2669.25 | 2660.68 | 2672.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:00:00 | 2666.50 | 2664.04 | 2671.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:30:00 | 2657.40 | 2663.27 | 2670.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:30:00 | 2645.80 | 2649.86 | 2662.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 2533.17 | 2585.78 | 2618.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 2524.53 | 2585.78 | 2618.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 14:15:00 | 2513.51 | 2548.75 | 2585.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 13:15:00 | 2525.00 | 2521.64 | 2553.69 | SL hit (close>ema200) qty=0.50 sl=2521.64 alert=retest2 |

### Cycle 116 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 2532.90 | 2481.29 | 2476.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 2643.65 | 2513.76 | 2491.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 2664.90 | 2689.10 | 2659.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 09:45:00 | 2678.35 | 2689.10 | 2659.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 2663.70 | 2684.02 | 2659.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:00:00 | 2681.00 | 2683.41 | 2661.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 2675.80 | 2705.77 | 2709.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 2675.80 | 2705.77 | 2709.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 2661.80 | 2688.71 | 2699.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 2670.30 | 2658.72 | 2673.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 2670.30 | 2658.72 | 2673.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 2670.30 | 2658.72 | 2673.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 15:00:00 | 2620.00 | 2656.89 | 2668.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 2624.85 | 2606.85 | 2605.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 2624.85 | 2606.85 | 2605.47 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 12:15:00 | 2575.25 | 2602.16 | 2604.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 14:15:00 | 2566.70 | 2590.47 | 2598.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 14:15:00 | 2566.00 | 2552.90 | 2571.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-21 15:00:00 | 2566.00 | 2552.90 | 2571.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 2549.00 | 2553.27 | 2568.23 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2024-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 12:15:00 | 2589.65 | 2571.54 | 2569.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 2602.30 | 2579.85 | 2573.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 2593.15 | 2595.36 | 2584.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 2593.15 | 2595.36 | 2584.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 2576.05 | 2591.08 | 2585.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 2576.05 | 2591.08 | 2585.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 2578.10 | 2588.49 | 2584.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 2582.70 | 2588.49 | 2584.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 2580.95 | 2586.98 | 2584.21 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 2560.80 | 2579.35 | 2581.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 14:15:00 | 2555.05 | 2569.89 | 2575.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 13:15:00 | 2563.65 | 2558.12 | 2566.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 13:15:00 | 2563.65 | 2558.12 | 2566.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 2563.65 | 2558.12 | 2566.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:30:00 | 2564.50 | 2558.12 | 2566.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 2560.30 | 2558.56 | 2565.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:45:00 | 2569.95 | 2558.56 | 2565.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 2565.00 | 2559.85 | 2565.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 2555.10 | 2559.85 | 2565.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 2557.05 | 2559.29 | 2564.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 12:30:00 | 2541.95 | 2555.59 | 2561.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 10:15:00 | 2597.00 | 2568.69 | 2565.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 2597.00 | 2568.69 | 2565.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 2613.00 | 2577.55 | 2569.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 11:15:00 | 2576.95 | 2592.31 | 2583.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 11:15:00 | 2576.95 | 2592.31 | 2583.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 2576.95 | 2592.31 | 2583.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:45:00 | 2579.95 | 2592.31 | 2583.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 2574.25 | 2588.70 | 2582.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 13:00:00 | 2574.25 | 2588.70 | 2582.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 14:15:00 | 2559.40 | 2579.05 | 2579.34 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 13:15:00 | 2613.65 | 2584.62 | 2581.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 2621.00 | 2600.64 | 2592.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 10:15:00 | 2601.00 | 2604.00 | 2596.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 10:45:00 | 2600.00 | 2604.00 | 2596.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 2601.05 | 2603.41 | 2596.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 12:15:00 | 2608.00 | 2603.41 | 2596.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 13:15:00 | 2594.60 | 2601.16 | 2596.77 | SL hit (close<static) qty=1.00 sl=2596.25 alert=retest2 |

### Cycle 125 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 2875.00 | 2891.00 | 2891.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 09:15:00 | 2859.10 | 2884.62 | 2888.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 09:15:00 | 2889.60 | 2866.68 | 2874.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 2889.60 | 2866.68 | 2874.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 2889.60 | 2866.68 | 2874.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:30:00 | 2907.00 | 2866.68 | 2874.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 2888.75 | 2871.09 | 2876.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:30:00 | 2892.20 | 2871.09 | 2876.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 2881.15 | 2873.10 | 2876.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 2861.20 | 2871.72 | 2875.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 15:15:00 | 2890.00 | 2878.13 | 2877.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 15:15:00 | 2890.00 | 2878.13 | 2877.82 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 09:15:00 | 2871.20 | 2876.74 | 2877.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 10:15:00 | 2853.05 | 2872.00 | 2875.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 12:15:00 | 2871.85 | 2871.67 | 2874.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-01 13:15:00 | 2879.15 | 2871.67 | 2874.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 2870.60 | 2871.45 | 2873.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:30:00 | 2870.30 | 2871.45 | 2873.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 2887.70 | 2874.70 | 2875.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:30:00 | 2898.55 | 2874.70 | 2875.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 2883.80 | 2876.52 | 2876.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 2896.85 | 2880.59 | 2877.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 15:15:00 | 2898.75 | 2899.78 | 2890.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 15:15:00 | 2898.75 | 2899.78 | 2890.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 2898.75 | 2899.78 | 2890.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 2888.70 | 2899.78 | 2890.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 2878.05 | 2895.44 | 2889.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 2878.05 | 2895.44 | 2889.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 2877.60 | 2891.87 | 2888.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:45:00 | 2870.90 | 2891.87 | 2888.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 2878.00 | 2889.09 | 2887.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:30:00 | 2878.75 | 2889.09 | 2887.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 2917.75 | 2920.05 | 2905.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 2917.75 | 2920.05 | 2905.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 2889.90 | 2914.02 | 2904.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 2880.25 | 2914.02 | 2904.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 2889.60 | 2909.13 | 2903.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:45:00 | 2886.35 | 2909.13 | 2903.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 2884.75 | 2900.74 | 2900.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 2884.95 | 2900.74 | 2900.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 2886.00 | 2897.79 | 2898.83 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 11:15:00 | 2925.00 | 2904.06 | 2901.47 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 13:15:00 | 2894.55 | 2907.18 | 2907.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 14:15:00 | 2885.20 | 2902.78 | 2905.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 11:15:00 | 2639.10 | 2629.66 | 2690.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:30:00 | 2646.80 | 2629.66 | 2690.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 2671.00 | 2644.12 | 2678.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 2617.55 | 2644.12 | 2678.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 2600.00 | 2635.30 | 2671.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 14:30:00 | 2594.75 | 2611.19 | 2644.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:15:00 | 2594.00 | 2611.19 | 2644.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 09:45:00 | 2595.30 | 2600.70 | 2633.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 12:00:00 | 2555.85 | 2582.86 | 2619.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 2591.35 | 2573.89 | 2591.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:00:00 | 2591.35 | 2573.89 | 2591.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 2589.65 | 2577.04 | 2591.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-20 10:15:00 | 2685.40 | 2614.61 | 2605.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 2685.40 | 2614.61 | 2605.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 2700.95 | 2631.88 | 2614.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 2632.50 | 2667.96 | 2643.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 2632.50 | 2667.96 | 2643.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 2632.50 | 2667.96 | 2643.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 2632.50 | 2667.96 | 2643.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 2596.80 | 2653.73 | 2638.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 2596.80 | 2653.73 | 2638.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 2590.55 | 2623.68 | 2627.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 2565.40 | 2612.03 | 2621.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 2553.60 | 2535.09 | 2564.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 2553.60 | 2535.09 | 2564.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2553.60 | 2535.09 | 2564.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 2553.60 | 2535.09 | 2564.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2592.80 | 2546.63 | 2567.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 2592.80 | 2546.63 | 2567.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 2588.60 | 2555.03 | 2569.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 2591.55 | 2555.03 | 2569.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 2577.00 | 2559.42 | 2569.90 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 2650.00 | 2579.05 | 2577.10 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 2525.05 | 2574.80 | 2575.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 2410.75 | 2509.10 | 2540.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 2458.05 | 2453.27 | 2494.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 15:00:00 | 2458.05 | 2453.27 | 2494.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 2465.25 | 2447.20 | 2480.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 2465.25 | 2447.20 | 2480.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 2458.00 | 2450.50 | 2474.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:30:00 | 2477.75 | 2450.50 | 2474.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 2457.55 | 2451.91 | 2472.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 15:00:00 | 2457.55 | 2451.91 | 2472.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 2469.70 | 2457.55 | 2470.14 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 2488.65 | 2475.21 | 2474.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 11:15:00 | 2507.45 | 2484.78 | 2479.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 14:15:00 | 2477.45 | 2492.53 | 2485.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 14:15:00 | 2477.45 | 2492.53 | 2485.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 2477.45 | 2492.53 | 2485.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 15:00:00 | 2477.45 | 2492.53 | 2485.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 2480.00 | 2490.02 | 2484.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 2464.50 | 2490.02 | 2484.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 2458.05 | 2483.63 | 2482.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:00:00 | 2458.05 | 2483.63 | 2482.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 10:15:00 | 2466.85 | 2480.27 | 2480.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 11:15:00 | 2441.60 | 2472.54 | 2477.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 2506.45 | 2461.30 | 2467.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 2506.45 | 2461.30 | 2467.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 2506.45 | 2461.30 | 2467.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 2513.95 | 2461.30 | 2467.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 2494.45 | 2467.93 | 2470.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:30:00 | 2507.50 | 2467.93 | 2470.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 2498.00 | 2476.69 | 2473.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 10:15:00 | 2502.20 | 2486.35 | 2480.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 12:15:00 | 2544.65 | 2549.27 | 2525.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 13:00:00 | 2544.65 | 2549.27 | 2525.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 2523.20 | 2544.06 | 2525.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:00:00 | 2523.20 | 2544.06 | 2525.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 2548.10 | 2544.87 | 2527.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:45:00 | 2513.65 | 2544.87 | 2527.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 2506.00 | 2537.91 | 2527.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 2506.00 | 2537.91 | 2527.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 2518.35 | 2534.00 | 2526.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 2518.35 | 2534.00 | 2526.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 13:15:00 | 2498.05 | 2517.49 | 2519.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 14:15:00 | 2485.65 | 2511.12 | 2516.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 15:15:00 | 2484.00 | 2482.57 | 2494.79 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 09:15:00 | 2471.90 | 2482.57 | 2494.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 09:45:00 | 2473.75 | 2481.12 | 2493.02 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 2494.60 | 2483.82 | 2493.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 2494.60 | 2483.82 | 2493.16 | SL hit (close>ema400) qty=1.00 sl=2493.16 alert=retest1 |

### Cycle 140 — BUY (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 15:15:00 | 2516.05 | 2500.28 | 2498.32 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 2490.85 | 2496.91 | 2497.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 12:15:00 | 2478.20 | 2490.92 | 2494.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 2507.00 | 2492.37 | 2494.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 2507.00 | 2492.37 | 2494.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 2507.00 | 2492.37 | 2494.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 2507.00 | 2492.37 | 2494.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 2504.45 | 2494.79 | 2495.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 2468.65 | 2494.79 | 2495.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 2470.00 | 2489.83 | 2492.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:30:00 | 2459.30 | 2485.15 | 2490.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:15:00 | 2461.45 | 2485.15 | 2490.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 11:15:00 | 2515.90 | 2463.43 | 2458.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 2515.90 | 2463.43 | 2458.62 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 11:15:00 | 2402.55 | 2455.58 | 2461.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 2390.90 | 2433.59 | 2449.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 13:15:00 | 2357.00 | 2351.86 | 2376.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 14:00:00 | 2357.00 | 2351.86 | 2376.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 2384.20 | 2358.33 | 2377.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 2384.20 | 2358.33 | 2377.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 2387.95 | 2364.25 | 2378.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 2387.80 | 2364.25 | 2378.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 2386.05 | 2368.61 | 2378.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:30:00 | 2364.25 | 2371.18 | 2376.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 14:15:00 | 2345.25 | 2327.77 | 2326.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 2345.25 | 2327.77 | 2326.22 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 2299.95 | 2324.63 | 2325.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 2279.15 | 2303.55 | 2313.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 2347.30 | 2286.22 | 2296.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 2347.30 | 2286.22 | 2296.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 2347.30 | 2286.22 | 2296.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 2347.30 | 2286.22 | 2296.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 2265.70 | 2282.12 | 2293.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 2219.40 | 2282.12 | 2293.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 2337.55 | 2283.46 | 2280.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 2337.55 | 2283.46 | 2280.18 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 13:15:00 | 2279.15 | 2282.28 | 2282.63 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 2286.10 | 2283.05 | 2282.95 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 15:15:00 | 2279.20 | 2282.28 | 2282.61 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 2293.60 | 2284.54 | 2283.61 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 14:15:00 | 2277.45 | 2283.24 | 2283.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-06 15:15:00 | 2272.00 | 2280.99 | 2282.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 09:15:00 | 2290.30 | 2282.85 | 2283.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 2290.30 | 2282.85 | 2283.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 2290.30 | 2282.85 | 2283.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:00:00 | 2290.30 | 2282.85 | 2283.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 10:15:00 | 2308.80 | 2288.04 | 2285.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 11:15:00 | 2316.20 | 2293.67 | 2288.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 2301.05 | 2305.62 | 2297.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 2301.05 | 2305.62 | 2297.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 2301.05 | 2305.62 | 2297.58 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 2261.25 | 2288.39 | 2292.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 2250.00 | 2280.71 | 2288.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 2195.00 | 2176.92 | 2204.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 2195.00 | 2176.92 | 2204.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 2195.00 | 2176.92 | 2204.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 2149.95 | 2182.90 | 2193.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 10:15:00 | 2142.65 | 2176.82 | 2189.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 2144.25 | 2148.57 | 2166.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 10:00:00 | 2146.00 | 2148.06 | 2164.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 2164.50 | 2151.35 | 2164.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 10:30:00 | 2169.15 | 2151.35 | 2164.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 2160.65 | 2153.21 | 2164.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-20 09:15:00 | 2207.10 | 2168.13 | 2167.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 2207.10 | 2168.13 | 2167.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 14:15:00 | 2243.75 | 2207.31 | 2189.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 2361.80 | 2367.65 | 2330.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 09:30:00 | 2358.45 | 2367.65 | 2330.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 2407.00 | 2395.28 | 2374.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:30:00 | 2382.05 | 2395.28 | 2374.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 2424.00 | 2434.96 | 2414.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 2424.00 | 2434.96 | 2414.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 2415.00 | 2427.19 | 2415.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 2446.85 | 2427.19 | 2415.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 2449.30 | 2431.61 | 2418.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 15:15:00 | 2465.05 | 2434.61 | 2424.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 2466.00 | 2428.35 | 2427.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 11:15:00 | 2423.00 | 2426.87 | 2427.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 11:15:00 | 2423.00 | 2426.87 | 2427.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 2421.05 | 2424.89 | 2426.07 | Break + close below crossover candle low |

### Cycle 156 — BUY (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 13:15:00 | 2445.40 | 2426.92 | 2426.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 14:15:00 | 2453.65 | 2432.27 | 2428.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2417.40 | 2431.33 | 2429.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 2417.40 | 2431.33 | 2429.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 2417.40 | 2431.33 | 2429.16 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 2402.40 | 2424.03 | 2426.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 13:15:00 | 2386.30 | 2416.49 | 2422.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-09 15:15:00 | 2325.05 | 2324.27 | 2351.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-11 09:15:00 | 2368.05 | 2324.27 | 2351.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 2331.55 | 2325.73 | 2349.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:45:00 | 2316.45 | 2326.72 | 2348.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 2405.00 | 2353.13 | 2352.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 2405.00 | 2353.13 | 2352.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 10:15:00 | 2474.00 | 2424.72 | 2395.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 11:15:00 | 2577.90 | 2582.30 | 2555.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 11:45:00 | 2574.80 | 2582.30 | 2555.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 2571.00 | 2577.43 | 2557.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:45:00 | 2557.90 | 2577.43 | 2557.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 2555.00 | 2572.93 | 2560.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 2555.00 | 2572.93 | 2560.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 2545.00 | 2567.34 | 2558.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 2534.80 | 2567.34 | 2558.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 2536.80 | 2561.23 | 2556.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 2536.80 | 2561.23 | 2556.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 2536.80 | 2556.35 | 2555.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:15:00 | 2530.60 | 2556.35 | 2555.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 13:15:00 | 2538.00 | 2552.68 | 2553.55 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 2569.90 | 2554.81 | 2554.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 13:15:00 | 2589.00 | 2570.71 | 2563.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 2556.80 | 2575.21 | 2567.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 2556.80 | 2575.21 | 2567.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2556.80 | 2575.21 | 2567.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:45:00 | 2553.60 | 2575.21 | 2567.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 2567.00 | 2573.57 | 2567.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:45:00 | 2557.50 | 2573.57 | 2567.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 2571.30 | 2573.12 | 2568.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:30:00 | 2557.70 | 2573.12 | 2568.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 12:15:00 | 2555.00 | 2569.49 | 2566.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:00:00 | 2555.00 | 2569.49 | 2566.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 2560.00 | 2567.59 | 2566.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:30:00 | 2555.30 | 2567.59 | 2566.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 2535.50 | 2560.12 | 2563.01 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 2591.00 | 2566.30 | 2565.55 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 10:15:00 | 2560.10 | 2565.06 | 2565.06 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 2576.80 | 2565.35 | 2564.74 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 2547.30 | 2561.46 | 2563.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 13:15:00 | 2541.30 | 2557.43 | 2561.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 14:15:00 | 2559.10 | 2557.76 | 2560.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 15:00:00 | 2559.10 | 2557.76 | 2560.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 2577.00 | 2560.85 | 2561.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:15:00 | 2575.00 | 2560.85 | 2561.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 2576.50 | 2563.98 | 2563.06 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 12:15:00 | 2502.20 | 2552.11 | 2557.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 2488.70 | 2539.42 | 2551.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 12:15:00 | 2406.40 | 2406.10 | 2444.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 13:00:00 | 2406.40 | 2406.10 | 2444.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 2388.00 | 2379.99 | 2398.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:30:00 | 2411.70 | 2379.99 | 2398.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 2412.50 | 2386.49 | 2399.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 2412.50 | 2386.49 | 2399.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 2431.30 | 2395.45 | 2402.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 2431.30 | 2395.45 | 2402.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 2453.40 | 2410.49 | 2408.40 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 2394.90 | 2416.01 | 2417.33 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2490.00 | 2424.01 | 2419.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 2510.00 | 2441.21 | 2427.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 2518.30 | 2529.10 | 2500.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 2518.30 | 2529.10 | 2500.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2518.30 | 2529.10 | 2500.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:00:00 | 2557.90 | 2530.94 | 2511.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:30:00 | 2548.00 | 2539.34 | 2525.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:30:00 | 2547.50 | 2543.84 | 2532.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 13:00:00 | 2551.70 | 2545.41 | 2534.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 2563.00 | 2564.48 | 2554.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 2551.20 | 2564.48 | 2554.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2553.10 | 2562.21 | 2554.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-21 10:15:00 | 2546.20 | 2551.67 | 2552.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 2546.20 | 2551.67 | 2552.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 2523.80 | 2546.10 | 2549.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 2437.00 | 2433.24 | 2457.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 2437.00 | 2433.24 | 2457.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2437.00 | 2433.24 | 2457.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:15:00 | 2449.40 | 2433.24 | 2457.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 2438.30 | 2434.25 | 2456.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:45:00 | 2435.50 | 2434.94 | 2454.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 2429.80 | 2436.57 | 2453.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 2459.90 | 2450.42 | 2450.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 2459.90 | 2450.42 | 2450.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 2483.70 | 2457.07 | 2453.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 2454.10 | 2465.51 | 2460.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 2454.10 | 2465.51 | 2460.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 2454.10 | 2465.51 | 2460.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 2455.10 | 2465.51 | 2460.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 2447.90 | 2461.99 | 2458.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:45:00 | 2445.00 | 2461.99 | 2458.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 12:15:00 | 2443.00 | 2455.99 | 2456.63 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 2460.00 | 2457.11 | 2456.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 2523.00 | 2470.29 | 2462.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 2467.20 | 2488.28 | 2477.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 2467.20 | 2488.28 | 2477.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 2467.20 | 2488.28 | 2477.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 2467.20 | 2488.28 | 2477.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 2464.80 | 2483.58 | 2475.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 2462.40 | 2483.58 | 2475.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 2445.10 | 2468.48 | 2470.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 13:15:00 | 2431.90 | 2457.58 | 2464.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 2359.00 | 2355.61 | 2379.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 2350.40 | 2356.80 | 2368.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 2350.40 | 2356.80 | 2368.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:45:00 | 2349.00 | 2355.42 | 2366.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 13:15:00 | 2368.90 | 2363.29 | 2363.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 2368.90 | 2363.29 | 2363.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 2378.00 | 2366.23 | 2364.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 2372.60 | 2389.12 | 2379.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 2372.60 | 2389.12 | 2379.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 2372.60 | 2389.12 | 2379.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 2372.60 | 2389.12 | 2379.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 2381.40 | 2387.58 | 2379.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 2419.80 | 2384.86 | 2379.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 14:45:00 | 2395.50 | 2395.21 | 2389.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 2368.00 | 2388.29 | 2386.88 | SL hit (close<static) qty=1.00 sl=2372.20 alert=retest2 |

### Cycle 177 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 2359.60 | 2382.56 | 2384.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 2351.10 | 2373.28 | 2378.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 2369.50 | 2368.20 | 2373.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 13:15:00 | 2369.50 | 2368.20 | 2373.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 2369.50 | 2368.20 | 2373.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:45:00 | 2369.60 | 2368.20 | 2373.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 2353.50 | 2365.38 | 2371.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 2350.60 | 2361.82 | 2369.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 2351.10 | 2356.19 | 2364.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 2340.00 | 2355.10 | 2362.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 2347.10 | 2355.35 | 2360.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 2357.30 | 2354.45 | 2359.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 2357.30 | 2354.45 | 2359.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 2358.40 | 2355.24 | 2359.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 2358.40 | 2355.24 | 2359.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 2357.00 | 2355.60 | 2359.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 2338.10 | 2350.90 | 2356.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 2332.00 | 2309.94 | 2309.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 2332.00 | 2309.94 | 2309.12 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 2295.40 | 2308.25 | 2309.65 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 2334.10 | 2310.13 | 2309.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 2353.00 | 2318.70 | 2313.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 2329.20 | 2333.31 | 2323.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 2329.20 | 2333.31 | 2323.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 2316.00 | 2329.96 | 2323.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 2316.00 | 2329.96 | 2323.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 2312.00 | 2326.37 | 2322.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:15:00 | 2318.00 | 2326.37 | 2322.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 14:15:00 | 2316.20 | 2319.58 | 2319.94 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 15:15:00 | 2325.00 | 2320.66 | 2320.40 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 2293.00 | 2315.13 | 2317.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 2272.80 | 2306.66 | 2313.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 2316.00 | 2294.85 | 2302.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 2316.00 | 2294.85 | 2302.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2316.00 | 2294.85 | 2302.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 2316.00 | 2294.85 | 2302.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2345.90 | 2305.06 | 2306.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 2345.90 | 2305.06 | 2306.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 2363.40 | 2316.73 | 2311.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 12:15:00 | 2408.20 | 2335.02 | 2320.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 10:15:00 | 2359.10 | 2360.96 | 2341.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:45:00 | 2359.90 | 2360.96 | 2341.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 2411.50 | 2419.65 | 2408.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:45:00 | 2411.10 | 2419.65 | 2408.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 2478.50 | 2434.87 | 2420.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 2493.80 | 2434.87 | 2420.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 2603.50 | 2650.03 | 2652.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 14:15:00 | 2603.50 | 2650.03 | 2652.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 2578.70 | 2631.76 | 2643.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 15:15:00 | 2598.00 | 2593.77 | 2614.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 09:30:00 | 2593.70 | 2592.11 | 2611.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 2616.40 | 2596.62 | 2610.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 2616.40 | 2596.62 | 2610.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 2621.00 | 2601.49 | 2611.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:30:00 | 2623.00 | 2601.49 | 2611.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 2606.10 | 2602.42 | 2610.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:45:00 | 2611.40 | 2602.42 | 2610.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 2605.10 | 2602.95 | 2610.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:45:00 | 2612.20 | 2602.95 | 2610.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 2555.00 | 2548.27 | 2561.96 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 2575.00 | 2568.94 | 2568.77 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 2558.70 | 2566.89 | 2567.85 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 10:15:00 | 2574.10 | 2568.15 | 2568.13 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 2563.20 | 2567.30 | 2567.79 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 2595.70 | 2571.76 | 2569.66 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 2588.00 | 2593.30 | 2593.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 2568.70 | 2588.38 | 2591.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 2542.90 | 2541.76 | 2560.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 2538.10 | 2541.76 | 2560.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 2458.00 | 2443.09 | 2459.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 2458.00 | 2443.09 | 2459.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 2466.20 | 2447.71 | 2459.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 2467.50 | 2447.71 | 2459.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2449.00 | 2447.97 | 2458.76 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 2500.00 | 2469.73 | 2465.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 2606.00 | 2528.08 | 2507.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 2592.70 | 2597.71 | 2574.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:00:00 | 2592.70 | 2597.71 | 2574.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 2574.10 | 2589.87 | 2574.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 2574.10 | 2589.87 | 2574.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 2574.80 | 2586.85 | 2574.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 2567.50 | 2586.85 | 2574.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 2572.40 | 2583.96 | 2574.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 2572.40 | 2583.96 | 2574.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 2562.30 | 2579.63 | 2573.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 2562.30 | 2579.63 | 2573.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 2542.00 | 2567.54 | 2568.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 2500.20 | 2537.18 | 2551.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 2496.80 | 2489.06 | 2510.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 2496.80 | 2489.06 | 2510.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 2502.50 | 2491.75 | 2509.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 2501.80 | 2491.75 | 2509.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2494.30 | 2486.47 | 2499.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 2496.60 | 2486.47 | 2499.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2480.50 | 2485.27 | 2497.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 2474.70 | 2485.27 | 2497.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 12:15:00 | 2506.80 | 2489.90 | 2497.63 | SL hit (close>static) qty=1.00 sl=2499.70 alert=retest2 |

### Cycle 194 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 2520.90 | 2504.65 | 2503.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 2535.50 | 2510.82 | 2506.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 2565.20 | 2573.62 | 2559.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 2565.20 | 2573.62 | 2559.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 2567.50 | 2573.72 | 2563.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 2567.20 | 2573.72 | 2563.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 2557.80 | 2570.54 | 2562.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 2557.80 | 2570.54 | 2562.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 2554.10 | 2567.25 | 2561.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 2554.10 | 2567.25 | 2561.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 2549.80 | 2560.01 | 2559.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 2530.30 | 2560.01 | 2559.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 2552.80 | 2558.24 | 2558.87 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 2573.60 | 2561.71 | 2560.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 2579.00 | 2566.38 | 2562.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 09:15:00 | 2575.80 | 2583.76 | 2576.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 2575.80 | 2583.76 | 2576.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2575.80 | 2583.76 | 2576.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 2575.80 | 2583.76 | 2576.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 2574.10 | 2581.83 | 2576.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 2577.00 | 2581.83 | 2576.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 2572.20 | 2579.90 | 2576.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 2572.00 | 2579.90 | 2576.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 2581.40 | 2580.20 | 2576.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:00:00 | 2585.70 | 2581.30 | 2577.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:00:00 | 2588.20 | 2595.91 | 2590.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:00:00 | 2589.20 | 2594.57 | 2590.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 2564.70 | 2587.86 | 2587.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 2564.70 | 2587.86 | 2587.94 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 2597.20 | 2576.61 | 2575.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 11:15:00 | 2600.30 | 2581.35 | 2578.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 2618.10 | 2638.06 | 2621.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 2618.10 | 2638.06 | 2621.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 2618.10 | 2638.06 | 2621.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 2618.10 | 2638.06 | 2621.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 2600.50 | 2630.55 | 2619.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:45:00 | 2602.90 | 2630.55 | 2619.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 2614.00 | 2627.24 | 2618.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:30:00 | 2604.00 | 2627.24 | 2618.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 2602.60 | 2620.53 | 2617.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 2603.50 | 2620.53 | 2617.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 2595.00 | 2611.42 | 2613.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 2584.80 | 2606.09 | 2610.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 12:15:00 | 2538.60 | 2536.66 | 2554.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 13:00:00 | 2538.60 | 2536.66 | 2554.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 2443.50 | 2431.52 | 2443.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 2443.50 | 2431.52 | 2443.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 2456.00 | 2436.41 | 2444.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 2411.00 | 2436.41 | 2444.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:30:00 | 2425.10 | 2429.35 | 2438.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 15:15:00 | 2461.00 | 2439.94 | 2440.71 | SL hit (close>static) qty=1.00 sl=2460.20 alert=retest2 |

### Cycle 200 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 2453.20 | 2442.59 | 2441.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 13:15:00 | 2481.00 | 2458.40 | 2450.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 2455.90 | 2465.41 | 2456.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 2455.90 | 2465.41 | 2456.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 2455.90 | 2465.41 | 2456.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 2454.00 | 2465.41 | 2456.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 2454.70 | 2463.27 | 2456.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 2455.50 | 2463.27 | 2456.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 2461.00 | 2462.81 | 2457.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:30:00 | 2463.50 | 2462.99 | 2457.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 2450.10 | 2460.41 | 2457.05 | SL hit (close<static) qty=1.00 sl=2452.10 alert=retest2 |

### Cycle 201 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 2442.20 | 2465.86 | 2466.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 2428.80 | 2454.15 | 2461.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 2440.40 | 2432.61 | 2441.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 2440.40 | 2432.61 | 2441.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 2440.40 | 2432.61 | 2441.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 2440.40 | 2432.61 | 2441.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 2434.10 | 2432.91 | 2441.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 2431.80 | 2432.91 | 2441.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2436.10 | 2433.55 | 2440.67 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 2451.50 | 2442.66 | 2442.39 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 2437.00 | 2441.53 | 2441.90 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 2450.00 | 2443.22 | 2442.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 2456.00 | 2446.88 | 2444.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 2440.00 | 2447.57 | 2445.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 2440.00 | 2447.57 | 2445.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2440.00 | 2447.57 | 2445.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 2443.90 | 2447.57 | 2445.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 2460.00 | 2450.06 | 2446.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:45:00 | 2464.90 | 2453.25 | 2448.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 14:15:00 | 2441.30 | 2462.82 | 2463.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 2441.30 | 2462.82 | 2463.04 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 2470.10 | 2464.27 | 2463.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 2482.10 | 2469.24 | 2466.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 2455.10 | 2466.41 | 2465.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 12:15:00 | 2455.10 | 2466.41 | 2465.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 2455.10 | 2466.41 | 2465.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:00:00 | 2455.10 | 2466.41 | 2465.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 2446.90 | 2462.51 | 2463.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 2436.20 | 2453.56 | 2458.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 2413.50 | 2410.49 | 2420.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 2413.50 | 2410.49 | 2420.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 2413.50 | 2410.49 | 2420.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 2420.60 | 2410.49 | 2420.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 2426.50 | 2413.69 | 2420.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 2426.70 | 2413.69 | 2420.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 2435.50 | 2418.05 | 2422.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 2435.50 | 2418.05 | 2422.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 2444.30 | 2426.70 | 2425.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 2445.10 | 2432.75 | 2428.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 2426.50 | 2431.50 | 2428.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 2426.50 | 2431.50 | 2428.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2426.50 | 2431.50 | 2428.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 2426.50 | 2431.50 | 2428.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 2447.70 | 2434.74 | 2430.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 2450.60 | 2434.74 | 2430.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 12:15:00 | 2413.00 | 2428.40 | 2430.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 2413.00 | 2428.40 | 2430.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 2400.00 | 2422.72 | 2427.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 2396.90 | 2392.66 | 2406.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 2396.90 | 2392.66 | 2406.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2390.60 | 2395.18 | 2403.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 2372.00 | 2390.43 | 2400.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 2253.40 | 2318.27 | 2348.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2227.70 | 2226.50 | 2247.43 | SL hit (close>ema200) qty=0.50 sl=2226.50 alert=retest2 |

### Cycle 210 — BUY (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 15:15:00 | 2285.00 | 2252.45 | 2252.25 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 11:15:00 | 2246.60 | 2257.22 | 2258.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 2231.00 | 2247.11 | 2252.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 15:15:00 | 2227.00 | 2224.05 | 2231.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 09:15:00 | 2229.90 | 2224.05 | 2231.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2227.60 | 2224.76 | 2231.48 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 2243.00 | 2235.13 | 2234.75 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 2213.40 | 2231.97 | 2233.53 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 2240.60 | 2233.54 | 2233.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 14:15:00 | 2245.40 | 2235.91 | 2234.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 11:15:00 | 2227.70 | 2236.21 | 2235.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 11:15:00 | 2227.70 | 2236.21 | 2235.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 2227.70 | 2236.21 | 2235.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 2230.00 | 2236.21 | 2235.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 2222.50 | 2233.47 | 2234.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 2218.10 | 2228.40 | 2231.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 2230.00 | 2228.72 | 2231.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 15:15:00 | 2230.00 | 2228.72 | 2231.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 2230.00 | 2228.72 | 2231.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 2236.30 | 2230.24 | 2231.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 2238.90 | 2231.97 | 2232.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 2238.90 | 2231.97 | 2232.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 2234.80 | 2232.54 | 2232.80 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 12:15:00 | 2244.90 | 2235.01 | 2233.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 15:15:00 | 2250.00 | 2240.06 | 2236.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 11:15:00 | 2248.90 | 2250.70 | 2243.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 11:30:00 | 2247.20 | 2250.70 | 2243.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 2256.40 | 2251.84 | 2244.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 15:00:00 | 2264.10 | 2254.94 | 2247.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 2262.90 | 2258.50 | 2250.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 2239.50 | 2254.01 | 2249.53 | SL hit (close<static) qty=1.00 sl=2243.70 alert=retest2 |

### Cycle 217 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 2239.50 | 2247.44 | 2247.45 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 2249.30 | 2247.81 | 2247.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 11:15:00 | 2260.00 | 2250.25 | 2248.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 2244.60 | 2249.93 | 2248.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 2244.60 | 2249.93 | 2248.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 2244.60 | 2249.93 | 2248.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 2244.60 | 2249.93 | 2248.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 2250.90 | 2250.12 | 2249.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:15:00 | 2250.90 | 2250.12 | 2249.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 2250.90 | 2250.28 | 2249.26 | EMA400 retest candle locked (from upside) |

### Cycle 219 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 2242.00 | 2247.57 | 2248.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 2229.00 | 2243.85 | 2246.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 2237.80 | 2237.54 | 2241.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 2237.80 | 2237.54 | 2241.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 2237.80 | 2237.54 | 2241.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 2238.90 | 2237.54 | 2241.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 2211.30 | 2206.52 | 2215.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:45:00 | 2219.90 | 2206.52 | 2215.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 2209.90 | 2208.65 | 2213.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 2209.90 | 2208.65 | 2213.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 2195.40 | 2205.75 | 2211.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:00:00 | 2190.30 | 2202.67 | 2207.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:00:00 | 2190.50 | 2199.12 | 2204.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:45:00 | 2182.40 | 2193.88 | 2201.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 2154.40 | 2129.30 | 2126.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 2154.40 | 2129.30 | 2126.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 2169.90 | 2147.65 | 2137.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 2210.90 | 2216.37 | 2195.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 2210.90 | 2216.37 | 2195.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 2202.00 | 2212.52 | 2200.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 2202.00 | 2212.52 | 2200.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 2189.20 | 2207.86 | 2199.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 2189.20 | 2207.86 | 2199.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 2196.00 | 2205.48 | 2198.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 2202.00 | 2205.48 | 2198.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 2178.10 | 2200.01 | 2196.97 | SL hit (close<static) qty=1.00 sl=2188.10 alert=retest2 |

### Cycle 221 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 2176.50 | 2191.34 | 2193.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 2165.80 | 2181.30 | 2186.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 2177.00 | 2167.55 | 2175.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 2177.00 | 2167.55 | 2175.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 2177.00 | 2167.55 | 2175.68 | EMA400 retest candle locked (from downside) |

### Cycle 222 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 2184.50 | 2178.84 | 2178.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 2197.60 | 2182.59 | 2180.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 2172.10 | 2181.82 | 2180.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 2172.10 | 2181.82 | 2180.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 2172.10 | 2181.82 | 2180.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 2172.10 | 2181.82 | 2180.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 2172.60 | 2179.98 | 2179.69 | EMA400 retest candle locked (from upside) |

### Cycle 223 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 2171.70 | 2178.32 | 2178.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 15:15:00 | 2163.00 | 2172.54 | 2175.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 2176.30 | 2173.29 | 2175.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 2176.30 | 2173.29 | 2175.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 2176.30 | 2173.29 | 2175.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 2173.30 | 2173.29 | 2175.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 2181.60 | 2174.95 | 2176.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 2179.00 | 2174.95 | 2176.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 2199.00 | 2179.76 | 2178.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 2203.10 | 2188.93 | 2183.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 2191.30 | 2201.02 | 2193.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 2191.30 | 2201.02 | 2193.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 2191.30 | 2201.02 | 2193.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 2191.30 | 2201.02 | 2193.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 2197.70 | 2200.36 | 2193.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 2201.00 | 2200.36 | 2193.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 2194.00 | 2199.09 | 2193.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 2198.60 | 2199.09 | 2193.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2209.00 | 2201.07 | 2195.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:45:00 | 2223.90 | 2206.81 | 2198.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:45:00 | 2218.30 | 2211.53 | 2201.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 2226.20 | 2258.59 | 2255.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 2221.00 | 2251.07 | 2252.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 2221.00 | 2251.07 | 2252.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 2196.90 | 2235.27 | 2244.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 2212.80 | 2210.14 | 2224.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 2215.70 | 2210.14 | 2224.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 2223.50 | 2213.74 | 2223.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 2230.30 | 2213.74 | 2223.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 2215.00 | 2213.99 | 2222.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 2208.40 | 2213.99 | 2222.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2204.30 | 2212.05 | 2221.27 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 2244.50 | 2220.16 | 2219.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 2250.00 | 2232.34 | 2225.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 2227.20 | 2235.10 | 2229.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 2227.20 | 2235.10 | 2229.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2227.20 | 2235.10 | 2229.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 2225.80 | 2235.10 | 2229.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 2198.10 | 2227.70 | 2226.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 2198.10 | 2227.70 | 2226.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 227 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 11:15:00 | 2180.10 | 2218.18 | 2222.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 14:15:00 | 2175.40 | 2199.58 | 2211.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 14:15:00 | 2126.10 | 2123.37 | 2145.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-20 14:45:00 | 2131.10 | 2123.37 | 2145.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2148.30 | 2111.58 | 2123.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 2144.90 | 2111.58 | 2123.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2125.70 | 2114.40 | 2123.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 2125.70 | 2114.40 | 2123.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 2112.20 | 2113.06 | 2120.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:30:00 | 2122.90 | 2113.06 | 2120.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2149.00 | 2120.25 | 2123.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 2149.00 | 2120.25 | 2123.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 228 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 2150.00 | 2126.20 | 2125.80 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 2112.60 | 2125.05 | 2125.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 2105.00 | 2121.04 | 2123.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 2095.30 | 2094.28 | 2105.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 2095.30 | 2094.28 | 2105.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 2072.00 | 2089.30 | 2100.90 | EMA400 retest candle locked (from downside) |

### Cycle 230 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 2117.60 | 2104.52 | 2104.15 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 2095.80 | 2102.93 | 2103.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 2091.30 | 2100.29 | 2102.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 11:15:00 | 2103.00 | 2099.05 | 2101.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 11:15:00 | 2103.00 | 2099.05 | 2101.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 2103.00 | 2099.05 | 2101.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:45:00 | 2099.80 | 2099.05 | 2101.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 2109.60 | 2101.16 | 2101.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 2109.60 | 2101.16 | 2101.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 232 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 2120.90 | 2105.11 | 2103.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 2135.80 | 2115.35 | 2109.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 2109.90 | 2118.32 | 2112.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 13:15:00 | 2109.90 | 2118.32 | 2112.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 2109.90 | 2118.32 | 2112.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 2109.90 | 2118.32 | 2112.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 2097.60 | 2114.17 | 2111.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 2097.60 | 2114.17 | 2111.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 2100.00 | 2111.34 | 2110.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 2082.50 | 2111.34 | 2110.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 233 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 2089.20 | 2106.91 | 2108.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 2058.70 | 2097.27 | 2103.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 2080.00 | 2076.16 | 2088.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 2122.60 | 2076.16 | 2088.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2122.90 | 2085.51 | 2091.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:15:00 | 2140.00 | 2085.51 | 2091.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 234 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 2155.00 | 2099.40 | 2097.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 2171.90 | 2113.90 | 2104.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 2087.40 | 2133.29 | 2120.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 2087.40 | 2133.29 | 2120.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2087.40 | 2133.29 | 2120.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 2087.40 | 2133.29 | 2120.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 2082.30 | 2123.09 | 2117.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:30:00 | 2068.00 | 2123.09 | 2117.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 235 — SELL (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 12:15:00 | 2082.60 | 2109.22 | 2111.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2055.00 | 2082.20 | 2092.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2067.00 | 2065.14 | 2076.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 2067.00 | 2065.14 | 2076.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2067.00 | 2065.14 | 2076.51 | EMA400 retest candle locked (from downside) |

### Cycle 236 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 2083.30 | 2077.74 | 2077.50 | EMA200 above EMA400 |

### Cycle 237 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 2072.10 | 2077.48 | 2078.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 2052.70 | 2071.17 | 2074.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 2083.20 | 2071.15 | 2073.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 2083.20 | 2071.15 | 2073.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 2083.20 | 2071.15 | 2073.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 2083.20 | 2071.15 | 2073.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 2073.10 | 2071.54 | 2073.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:30:00 | 2065.00 | 2068.07 | 2071.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 2058.30 | 2068.07 | 2071.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 2083.00 | 2073.57 | 2073.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 238 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 2083.00 | 2073.57 | 2073.30 | EMA200 above EMA400 |

### Cycle 239 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 2062.40 | 2072.42 | 2073.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 14:15:00 | 2061.80 | 2070.30 | 2072.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 12:15:00 | 2064.00 | 2063.21 | 2067.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 12:15:00 | 2064.00 | 2063.21 | 2067.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 2064.00 | 2063.21 | 2067.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 2064.00 | 2063.21 | 2067.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2077.50 | 2066.70 | 2068.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 2077.50 | 2066.70 | 2068.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 2074.60 | 2068.28 | 2069.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 2085.00 | 2068.28 | 2069.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 240 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 2081.00 | 2070.83 | 2070.15 | EMA200 above EMA400 |

### Cycle 241 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 2062.50 | 2068.55 | 2069.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 2052.10 | 2065.26 | 2067.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 13:15:00 | 2039.70 | 2034.71 | 2043.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 13:15:00 | 2039.70 | 2034.71 | 2043.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 2039.70 | 2034.71 | 2043.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:45:00 | 2044.30 | 2034.71 | 2043.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 2048.40 | 2037.45 | 2043.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 2048.40 | 2037.45 | 2043.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 2045.00 | 2038.96 | 2043.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 2030.70 | 2038.96 | 2043.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 2057.90 | 2043.54 | 2045.08 | SL hit (close>static) qty=1.00 sl=2050.00 alert=retest2 |

### Cycle 242 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 2057.30 | 2047.03 | 2046.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 2060.00 | 2050.85 | 2048.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 2240.90 | 2245.56 | 2207.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 15:00:00 | 2240.90 | 2245.56 | 2207.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2215.10 | 2241.78 | 2212.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:30:00 | 2243.20 | 2236.32 | 2219.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 15:00:00 | 2245.60 | 2236.32 | 2219.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 15:15:00 | 2203.80 | 2214.88 | 2216.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 243 — SELL (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 15:15:00 | 2203.80 | 2214.88 | 2216.06 | EMA200 below EMA400 |

### Cycle 244 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 2230.30 | 2217.97 | 2217.35 | EMA200 above EMA400 |

### Cycle 245 — SELL (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 10:15:00 | 2204.50 | 2215.27 | 2216.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 2185.20 | 2207.17 | 2211.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 13:15:00 | 2157.20 | 2153.97 | 2175.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 14:00:00 | 2157.20 | 2153.97 | 2175.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2177.50 | 2161.82 | 2174.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 2193.30 | 2161.82 | 2174.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 2206.00 | 2170.65 | 2177.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 2206.00 | 2170.65 | 2177.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 246 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 2234.40 | 2183.40 | 2182.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 2245.00 | 2214.47 | 2199.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 2233.80 | 2235.08 | 2219.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:15:00 | 2190.40 | 2235.08 | 2219.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 2210.40 | 2230.14 | 2218.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 2226.00 | 2230.14 | 2218.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 2177.00 | 2210.08 | 2213.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 247 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 2177.00 | 2210.08 | 2213.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 2154.10 | 2198.88 | 2208.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 2118.60 | 2086.10 | 2108.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 2118.60 | 2086.10 | 2108.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 2118.60 | 2086.10 | 2108.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 2119.00 | 2086.10 | 2108.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 2120.00 | 2092.88 | 2109.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 2124.00 | 2092.88 | 2109.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 2116.90 | 2103.07 | 2110.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:45:00 | 2119.00 | 2103.07 | 2110.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1958.60 | 1943.73 | 1969.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1964.90 | 1943.73 | 1969.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1988.00 | 1956.32 | 1971.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 1988.00 | 1956.32 | 1971.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1978.50 | 1960.76 | 1971.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 2018.30 | 1960.76 | 1971.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 248 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2008.00 | 1979.40 | 1979.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 2032.20 | 1995.94 | 1986.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 1994.40 | 2024.96 | 2014.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 1994.40 | 2024.96 | 2014.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1994.40 | 2024.96 | 2014.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:15:00 | 1997.60 | 2024.96 | 2014.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 1999.00 | 2019.77 | 2012.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 11:45:00 | 2006.00 | 2017.21 | 2012.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:30:00 | 2004.00 | 2013.05 | 2011.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 14:00:00 | 2007.60 | 2013.05 | 2011.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 10:15:00 | 1977.60 | 2004.63 | 2007.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 249 — SELL (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 10:15:00 | 1977.60 | 2004.63 | 2007.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 1936.00 | 1988.94 | 1998.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 12:15:00 | 1979.60 | 1978.85 | 1990.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 13:00:00 | 1979.60 | 1978.85 | 1990.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 2002.80 | 1985.28 | 1991.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 2002.80 | 1985.28 | 1991.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 1988.80 | 1985.99 | 1991.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 1969.80 | 1985.99 | 1991.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1988.10 | 1986.41 | 1991.29 | EMA400 retest candle locked (from downside) |

### Cycle 250 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 2004.50 | 1994.13 | 1993.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 2022.60 | 1999.83 | 1996.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 13:15:00 | 2042.90 | 2050.15 | 2033.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 13:15:00 | 2042.90 | 2050.15 | 2033.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 2042.90 | 2050.15 | 2033.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 14:00:00 | 2042.90 | 2050.15 | 2033.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 2043.50 | 2048.82 | 2034.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 15:00:00 | 2043.50 | 2048.82 | 2034.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 2037.20 | 2045.88 | 2035.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 2037.20 | 2045.88 | 2035.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 2043.10 | 2045.33 | 2036.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:30:00 | 2033.80 | 2045.33 | 2036.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2076.40 | 2072.23 | 2060.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2084.70 | 2075.38 | 2063.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 2084.30 | 2078.61 | 2068.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:30:00 | 2080.30 | 2078.03 | 2068.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 2080.00 | 2078.03 | 2068.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 2096.80 | 2107.83 | 2096.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 2096.80 | 2107.83 | 2096.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 2101.50 | 2106.56 | 2096.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 2112.80 | 2105.74 | 2098.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-23 09:15:00 | 2293.17 | 2240.49 | 2205.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 251 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 2245.60 | 2265.13 | 2266.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 2235.00 | 2256.12 | 2261.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 2245.20 | 2244.36 | 2253.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 13:45:00 | 2250.00 | 2244.36 | 2253.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 2246.90 | 2244.87 | 2252.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 2257.40 | 2244.87 | 2252.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2278.60 | 2251.48 | 2254.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 2276.30 | 2251.48 | 2254.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 2274.00 | 2255.98 | 2256.22 | EMA400 retest candle locked (from downside) |

### Cycle 252 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 2265.40 | 2257.87 | 2257.05 | EMA200 above EMA400 |

### Cycle 253 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 2248.00 | 2255.89 | 2256.23 | EMA200 below EMA400 |

### Cycle 254 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 2267.40 | 2258.20 | 2257.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 13:15:00 | 2300.40 | 2271.12 | 2264.00 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-22 09:45:00 | 2356.85 | 2024-04-26 15:15:00 | 2365.00 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2024-04-22 12:00:00 | 2352.05 | 2024-04-26 15:15:00 | 2365.00 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2024-04-23 09:15:00 | 2369.35 | 2024-04-26 15:15:00 | 2365.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-05-02 10:30:00 | 2359.80 | 2024-05-07 11:15:00 | 2241.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 10:30:00 | 2359.80 | 2024-05-09 09:15:00 | 2204.80 | STOP_HIT | 0.50 | 6.57% |
| SELL | retest2 | 2024-05-22 10:15:00 | 2089.50 | 2024-05-27 11:15:00 | 2115.05 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-05-23 11:30:00 | 2091.55 | 2024-05-27 11:15:00 | 2115.05 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-05-23 12:00:00 | 2090.55 | 2024-05-27 11:15:00 | 2115.05 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-05-23 13:45:00 | 2088.05 | 2024-05-27 13:15:00 | 2103.45 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-05-24 12:30:00 | 2091.95 | 2024-05-27 13:15:00 | 2103.45 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-05-24 13:30:00 | 2091.65 | 2024-05-27 13:15:00 | 2103.45 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-05-27 09:30:00 | 2075.80 | 2024-05-27 13:15:00 | 2103.45 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-29 12:15:00 | 2137.25 | 2024-05-30 14:15:00 | 2087.70 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-05-30 10:30:00 | 2144.50 | 2024-05-30 14:15:00 | 2087.70 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2024-06-03 11:15:00 | 2145.85 | 2024-06-04 10:15:00 | 2065.30 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2024-06-19 12:30:00 | 2241.30 | 2024-06-19 13:15:00 | 2221.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-06-21 14:15:00 | 2196.25 | 2024-07-01 13:15:00 | 2143.95 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2024-06-21 15:00:00 | 2163.85 | 2024-07-01 13:15:00 | 2143.95 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2024-07-15 12:15:00 | 2166.70 | 2024-07-16 10:15:00 | 2128.55 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-07-15 13:00:00 | 2162.15 | 2024-07-16 10:15:00 | 2128.55 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-07-25 09:15:00 | 2098.10 | 2024-07-25 12:15:00 | 2136.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-07-25 09:45:00 | 2099.95 | 2024-07-25 12:15:00 | 2136.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-07-25 10:30:00 | 2095.25 | 2024-07-25 12:15:00 | 2136.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-07-30 15:00:00 | 2058.50 | 2024-08-05 09:15:00 | 1955.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 15:00:00 | 2058.50 | 2024-08-05 12:15:00 | 1993.55 | STOP_HIT | 0.50 | 3.16% |
| BUY | retest2 | 2024-08-08 11:30:00 | 2071.20 | 2024-08-16 11:15:00 | 2278.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-09 09:15:00 | 2092.00 | 2024-08-16 14:15:00 | 2301.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-26 10:30:00 | 2703.10 | 2024-09-27 10:15:00 | 2651.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-09-26 15:00:00 | 2685.00 | 2024-09-27 10:15:00 | 2651.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-10-03 10:45:00 | 2535.00 | 2024-10-04 11:15:00 | 2593.25 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-10-03 11:15:00 | 2532.95 | 2024-10-04 11:15:00 | 2593.25 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-10-03 15:15:00 | 2531.75 | 2024-10-04 11:15:00 | 2593.25 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-10-11 13:30:00 | 2758.05 | 2024-10-16 12:15:00 | 2689.90 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-10-14 15:00:00 | 2760.75 | 2024-10-16 12:15:00 | 2689.90 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-10-15 14:45:00 | 2744.90 | 2024-10-16 12:15:00 | 2689.90 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-10-21 13:00:00 | 2666.50 | 2024-10-23 09:15:00 | 2533.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 13:30:00 | 2657.40 | 2024-10-23 09:15:00 | 2524.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:30:00 | 2645.80 | 2024-10-23 14:15:00 | 2513.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 13:00:00 | 2666.50 | 2024-10-24 13:15:00 | 2525.00 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2024-10-21 13:30:00 | 2657.40 | 2024-10-24 13:15:00 | 2525.00 | STOP_HIT | 0.50 | 4.98% |
| SELL | retest2 | 2024-10-22 09:30:00 | 2645.80 | 2024-10-24 13:15:00 | 2525.00 | STOP_HIT | 0.50 | 4.57% |
| BUY | retest2 | 2024-11-05 12:00:00 | 2681.00 | 2024-11-08 11:15:00 | 2675.80 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-11-12 15:00:00 | 2620.00 | 2024-11-18 12:15:00 | 2624.85 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-11-29 12:30:00 | 2541.95 | 2024-12-02 10:15:00 | 2597.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-12-06 12:15:00 | 2608.00 | 2024-12-06 13:15:00 | 2594.60 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-12-09 10:15:00 | 2621.20 | 2024-12-19 13:15:00 | 2883.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-31 12:30:00 | 2861.20 | 2024-12-31 15:15:00 | 2890.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-01-15 14:30:00 | 2594.75 | 2025-01-20 10:15:00 | 2685.40 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2025-01-15 15:15:00 | 2594.00 | 2025-01-20 10:15:00 | 2685.40 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-01-16 09:45:00 | 2595.30 | 2025-01-20 10:15:00 | 2685.40 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2025-01-16 12:00:00 | 2555.85 | 2025-01-20 10:15:00 | 2685.40 | STOP_HIT | 1.00 | -5.07% |
| SELL | retest1 | 2025-02-07 09:15:00 | 2471.90 | 2025-02-07 10:15:00 | 2494.60 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest1 | 2025-02-07 09:45:00 | 2473.75 | 2025-02-07 10:15:00 | 2494.60 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-02-11 10:30:00 | 2459.30 | 2025-02-13 11:15:00 | 2515.90 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-02-11 11:15:00 | 2461.45 | 2025-02-13 11:15:00 | 2515.90 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-02-19 14:30:00 | 2364.25 | 2025-02-25 14:15:00 | 2345.25 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2025-03-03 09:15:00 | 2219.40 | 2025-03-04 09:15:00 | 2337.55 | STOP_HIT | 1.00 | -5.32% |
| SELL | retest2 | 2025-03-18 09:15:00 | 2149.95 | 2025-03-20 09:15:00 | 2207.10 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-03-18 10:15:00 | 2142.65 | 2025-03-20 09:15:00 | 2207.10 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-03-19 09:15:00 | 2144.25 | 2025-03-20 09:15:00 | 2207.10 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-03-19 10:00:00 | 2146.00 | 2025-03-20 09:15:00 | 2207.10 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-04-01 15:15:00 | 2465.05 | 2025-04-03 11:15:00 | 2423.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-04-03 09:15:00 | 2466.00 | 2025-04-03 11:15:00 | 2423.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-04-11 10:45:00 | 2316.45 | 2025-04-15 09:15:00 | 2405.00 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2025-05-14 15:00:00 | 2557.90 | 2025-05-21 10:15:00 | 2546.20 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-05-15 14:30:00 | 2548.00 | 2025-05-21 10:15:00 | 2546.20 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-05-16 11:30:00 | 2547.50 | 2025-05-21 10:15:00 | 2546.20 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-05-16 13:00:00 | 2551.70 | 2025-05-21 10:15:00 | 2546.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-05-26 11:45:00 | 2435.50 | 2025-05-28 09:15:00 | 2459.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-05-26 13:15:00 | 2429.80 | 2025-05-28 09:15:00 | 2459.90 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-06-06 10:45:00 | 2349.00 | 2025-06-10 13:15:00 | 2368.90 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-06-12 09:15:00 | 2419.80 | 2025-06-13 09:15:00 | 2368.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-06-12 14:45:00 | 2395.50 | 2025-06-13 09:15:00 | 2368.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-06-17 10:45:00 | 2350.60 | 2025-06-25 09:15:00 | 2332.00 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-06-17 14:15:00 | 2351.10 | 2025-06-25 09:15:00 | 2332.00 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-06-18 09:15:00 | 2340.00 | 2025-06-25 09:15:00 | 2332.00 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-06-18 11:30:00 | 2347.10 | 2025-06-25 09:15:00 | 2332.00 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-06-19 09:30:00 | 2338.10 | 2025-06-25 09:15:00 | 2332.00 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-07-09 10:15:00 | 2493.80 | 2025-07-23 14:15:00 | 2603.50 | STOP_HIT | 1.00 | 4.40% |
| SELL | retest2 | 2025-09-01 11:15:00 | 2474.70 | 2025-09-01 12:15:00 | 2506.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-10 14:00:00 | 2585.70 | 2025-09-12 12:15:00 | 2564.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-12 10:00:00 | 2588.20 | 2025-09-12 12:15:00 | 2564.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-12 11:00:00 | 2589.20 | 2025-09-12 12:15:00 | 2564.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-10-06 09:15:00 | 2411.00 | 2025-10-06 15:15:00 | 2461.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-10-06 11:30:00 | 2425.10 | 2025-10-06 15:15:00 | 2461.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-08 13:30:00 | 2463.50 | 2025-10-08 14:15:00 | 2450.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-10-09 09:15:00 | 2464.50 | 2025-10-10 14:15:00 | 2456.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-10-09 10:15:00 | 2466.60 | 2025-10-13 09:15:00 | 2442.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-09 11:00:00 | 2465.90 | 2025-10-13 09:15:00 | 2442.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-10 12:00:00 | 2500.00 | 2025-10-13 09:15:00 | 2442.20 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-10-17 11:45:00 | 2464.90 | 2025-10-21 14:15:00 | 2441.30 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-30 11:15:00 | 2450.60 | 2025-10-31 12:15:00 | 2413.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-11-04 11:30:00 | 2372.00 | 2025-11-07 09:15:00 | 2253.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:30:00 | 2372.00 | 2025-11-12 09:15:00 | 2227.70 | STOP_HIT | 0.50 | 6.08% |
| BUY | retest2 | 2025-11-26 15:00:00 | 2264.10 | 2025-11-27 11:15:00 | 2239.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-27 09:30:00 | 2262.90 | 2025-11-27 11:15:00 | 2239.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-08 10:00:00 | 2190.30 | 2025-12-19 09:15:00 | 2154.40 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2025-12-08 12:00:00 | 2190.50 | 2025-12-19 09:15:00 | 2154.40 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2025-12-08 12:45:00 | 2182.40 | 2025-12-19 09:15:00 | 2154.40 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2025-12-26 09:15:00 | 2202.00 | 2025-12-26 09:15:00 | 2178.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-01-06 11:45:00 | 2223.90 | 2026-01-09 10:15:00 | 2221.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-01-06 12:45:00 | 2218.30 | 2026-01-09 10:15:00 | 2221.00 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2026-01-09 09:45:00 | 2226.20 | 2026-01-09 10:15:00 | 2221.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2026-02-13 14:30:00 | 2065.00 | 2026-02-16 11:15:00 | 2083.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-02-13 15:00:00 | 2058.30 | 2026-02-16 11:15:00 | 2083.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-02-24 09:15:00 | 2030.70 | 2026-02-24 10:15:00 | 2057.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-02-24 12:30:00 | 2044.30 | 2026-02-24 13:15:00 | 2057.30 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-24 13:15:00 | 2041.40 | 2026-02-24 13:15:00 | 2057.30 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-03-02 14:30:00 | 2243.20 | 2026-03-04 15:15:00 | 2203.80 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-03-02 15:00:00 | 2245.60 | 2026-03-04 15:15:00 | 2203.80 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-03-12 10:15:00 | 2226.00 | 2026-03-13 09:15:00 | 2177.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-03-30 11:45:00 | 2006.00 | 2026-04-01 10:15:00 | 1977.60 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-03-30 13:30:00 | 2004.00 | 2026-04-01 10:15:00 | 1977.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-03-30 14:00:00 | 2007.60 | 2026-04-01 10:15:00 | 1977.60 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2084.70 | 2026-04-23 09:15:00 | 2293.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:45:00 | 2084.30 | 2026-04-23 09:15:00 | 2292.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 14:30:00 | 2080.30 | 2026-04-23 09:15:00 | 2288.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 15:15:00 | 2080.00 | 2026-04-23 09:15:00 | 2288.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-17 09:15:00 | 2112.80 | 2026-04-29 13:15:00 | 2245.60 | STOP_HIT | 1.00 | 6.29% |
