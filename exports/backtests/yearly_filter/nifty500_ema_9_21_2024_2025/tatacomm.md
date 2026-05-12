# Tata Communications Ltd. (TATACOMM)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 1582.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 159 |
| ALERT1 | 112 |
| ALERT2 | 110 |
| ALERT2_SKIP | 58 |
| ALERT3 | 285 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 132 |
| PARTIAL | 9 |
| TARGET_HIT | 1 |
| STOP_HIT | 136 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 146 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 108
- **Target hits / Stop hits / Partials:** 1 / 136 / 9
- **Avg / median % per leg:** -0.14% / -0.96%
- **Sum % (uncompounded):** -19.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 16 | 30.8% | 1 | 50 | 1 | -0.07% | -3.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.81% | 11.2% |
| BUY @ 3rd Alert (retest2) | 48 | 14 | 29.2% | 0 | 48 | 0 | -0.31% | -14.8% |
| SELL (all) | 94 | 22 | 23.4% | 0 | 86 | 8 | -0.17% | -16.2% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.55% | 7.7% |
| SELL @ 3rd Alert (retest2) | 91 | 20 | 22.0% | 0 | 84 | 7 | -0.26% | -23.9% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.70% | 18.9% |
| retest2 (combined) | 139 | 34 | 24.5% | 0 | 132 | 7 | -0.28% | -38.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 1746.40 | 1735.72 | 1735.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 1747.00 | 1741.03 | 1738.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 11:15:00 | 1791.55 | 1794.62 | 1781.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 12:00:00 | 1791.55 | 1794.62 | 1781.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1812.25 | 1815.62 | 1806.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 1800.00 | 1815.62 | 1806.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1821.30 | 1816.75 | 1807.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:00:00 | 1823.60 | 1818.12 | 1809.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 1830.25 | 1817.88 | 1811.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:15:00 | 1827.90 | 1830.10 | 1828.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 13:15:00 | 1825.15 | 1827.41 | 1827.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 1825.00 | 1826.93 | 1827.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 13:15:00 | 1825.00 | 1826.93 | 1827.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 1818.40 | 1825.22 | 1826.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 1799.65 | 1799.25 | 1806.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 13:30:00 | 1800.05 | 1799.25 | 1806.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 1778.40 | 1774.38 | 1782.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 1778.40 | 1774.38 | 1782.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1780.75 | 1775.65 | 1782.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 1780.75 | 1775.65 | 1782.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1770.00 | 1774.52 | 1781.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 1791.15 | 1774.52 | 1781.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1815.40 | 1782.70 | 1784.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:45:00 | 1816.00 | 1782.70 | 1784.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 1798.10 | 1785.78 | 1785.49 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 14:15:00 | 1772.95 | 1784.50 | 1785.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 09:15:00 | 1731.35 | 1771.71 | 1779.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1715.25 | 1697.20 | 1724.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 1715.25 | 1697.20 | 1724.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 1726.75 | 1705.80 | 1723.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 1726.75 | 1705.80 | 1723.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1741.50 | 1712.94 | 1725.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 1741.50 | 1712.94 | 1725.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1759.00 | 1731.83 | 1731.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 1780.55 | 1741.58 | 1736.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 1877.35 | 1877.59 | 1849.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 1877.35 | 1877.59 | 1849.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1871.95 | 1887.63 | 1874.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 1871.95 | 1887.63 | 1874.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 1877.00 | 1885.50 | 1875.10 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 15:15:00 | 1866.00 | 1873.49 | 1874.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 1851.90 | 1865.25 | 1869.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 1847.15 | 1845.46 | 1855.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 1847.15 | 1845.46 | 1855.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1847.15 | 1845.46 | 1855.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:45:00 | 1848.00 | 1845.46 | 1855.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1859.45 | 1848.26 | 1855.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 1859.45 | 1848.26 | 1855.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 1866.40 | 1851.89 | 1856.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:45:00 | 1861.40 | 1851.89 | 1856.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 1877.60 | 1857.03 | 1858.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:45:00 | 1874.75 | 1857.03 | 1858.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 14:15:00 | 1865.35 | 1860.05 | 1859.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 1869.70 | 1862.69 | 1860.95 | Break + close above crossover candle high |

### Cycle 8 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 1845.00 | 1859.73 | 1860.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 1841.50 | 1850.99 | 1855.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 12:15:00 | 1850.45 | 1849.77 | 1853.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 12:15:00 | 1850.45 | 1849.77 | 1853.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 1850.45 | 1849.77 | 1853.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:30:00 | 1850.15 | 1849.77 | 1853.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 1854.65 | 1850.75 | 1853.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:00:00 | 1854.65 | 1850.75 | 1853.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 1848.55 | 1850.31 | 1853.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 1848.55 | 1850.31 | 1853.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 1854.95 | 1851.24 | 1853.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 1858.60 | 1851.24 | 1853.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1860.90 | 1853.17 | 1854.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:30:00 | 1864.45 | 1853.17 | 1854.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 1852.70 | 1853.08 | 1853.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:30:00 | 1859.80 | 1853.08 | 1853.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 1845.85 | 1851.63 | 1853.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:30:00 | 1849.35 | 1851.63 | 1853.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 1842.50 | 1835.21 | 1840.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:30:00 | 1856.55 | 1835.21 | 1840.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 1841.00 | 1836.37 | 1840.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 1833.30 | 1836.37 | 1840.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 1856.90 | 1840.47 | 1842.14 | SL hit (close>static) qty=1.00 sl=1849.40 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 1854.95 | 1843.37 | 1843.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 1869.75 | 1848.64 | 1845.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 1841.90 | 1847.93 | 1845.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 1841.90 | 1847.93 | 1845.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1841.90 | 1847.93 | 1845.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 1841.90 | 1847.93 | 1845.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1856.40 | 1849.62 | 1846.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 1846.75 | 1849.62 | 1846.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 1850.00 | 1856.67 | 1851.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 1850.00 | 1856.67 | 1851.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 1848.05 | 1854.95 | 1851.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:45:00 | 1845.55 | 1854.95 | 1851.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 1851.15 | 1854.91 | 1852.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 1851.15 | 1854.91 | 1852.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1854.15 | 1854.76 | 1852.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 1846.50 | 1854.76 | 1852.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1859.00 | 1855.61 | 1852.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 10:15:00 | 1862.05 | 1855.61 | 1852.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 11:00:00 | 1862.85 | 1857.05 | 1853.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 09:15:00 | 1841.75 | 1852.41 | 1852.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 1841.75 | 1852.41 | 1852.80 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 1856.75 | 1853.27 | 1853.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 14:15:00 | 1890.10 | 1861.17 | 1856.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 12:15:00 | 1887.50 | 1887.68 | 1878.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 13:00:00 | 1887.50 | 1887.68 | 1878.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1876.80 | 1884.90 | 1878.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 1876.80 | 1884.90 | 1878.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 1877.00 | 1883.32 | 1878.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 1879.95 | 1883.32 | 1878.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:00:00 | 1882.15 | 1883.09 | 1879.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:45:00 | 1880.90 | 1882.07 | 1878.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:15:00 | 1885.75 | 1882.07 | 1878.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1889.60 | 1883.57 | 1879.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 1860.20 | 1880.33 | 1880.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 1860.20 | 1880.33 | 1880.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 1848.55 | 1873.98 | 1877.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 12:15:00 | 1860.20 | 1854.88 | 1863.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 12:15:00 | 1860.20 | 1854.88 | 1863.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 1860.20 | 1854.88 | 1863.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:45:00 | 1867.45 | 1854.88 | 1863.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 1857.55 | 1855.41 | 1863.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 14:15:00 | 1850.00 | 1855.41 | 1863.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 14:15:00 | 1852.05 | 1851.51 | 1856.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:30:00 | 1850.00 | 1852.77 | 1855.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:00:00 | 1850.75 | 1852.77 | 1855.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 1861.30 | 1854.47 | 1856.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 1861.30 | 1854.47 | 1856.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 1862.30 | 1856.04 | 1856.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:30:00 | 1865.60 | 1856.04 | 1856.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-11 13:15:00 | 1863.95 | 1857.62 | 1857.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 1863.95 | 1857.62 | 1857.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 10:15:00 | 1874.00 | 1864.67 | 1862.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 10:15:00 | 1861.70 | 1867.71 | 1865.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 10:15:00 | 1861.70 | 1867.71 | 1865.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1861.70 | 1867.71 | 1865.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:00:00 | 1861.70 | 1867.71 | 1865.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 1860.95 | 1866.36 | 1865.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 1860.95 | 1866.36 | 1865.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 1861.00 | 1865.29 | 1864.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:30:00 | 1860.80 | 1865.29 | 1864.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 1855.60 | 1863.35 | 1863.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 14:15:00 | 1848.45 | 1860.37 | 1862.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 1851.70 | 1850.81 | 1856.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 13:00:00 | 1851.70 | 1850.81 | 1856.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1876.90 | 1855.17 | 1857.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 1876.90 | 1855.17 | 1857.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 15:15:00 | 1874.00 | 1858.94 | 1858.72 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 1817.05 | 1850.56 | 1854.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 11:15:00 | 1814.00 | 1837.88 | 1848.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 1808.60 | 1806.08 | 1822.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 11:45:00 | 1808.85 | 1806.08 | 1822.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1799.80 | 1785.28 | 1795.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 1802.45 | 1785.28 | 1795.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1809.95 | 1790.21 | 1797.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 1809.95 | 1790.21 | 1797.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 1838.00 | 1808.54 | 1804.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 1875.85 | 1840.02 | 1827.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 1871.15 | 1871.23 | 1853.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 09:30:00 | 1871.45 | 1871.23 | 1853.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1944.70 | 1964.74 | 1950.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:30:00 | 1964.65 | 1966.80 | 1952.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 1901.00 | 1949.02 | 1950.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 1901.00 | 1949.02 | 1950.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 1869.40 | 1933.09 | 1943.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1901.40 | 1893.76 | 1914.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 1901.40 | 1893.76 | 1914.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 1882.00 | 1870.60 | 1881.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 1870.05 | 1870.60 | 1881.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:30:00 | 1872.80 | 1869.55 | 1879.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 1871.50 | 1870.67 | 1878.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 1894.10 | 1869.58 | 1874.63 | SL hit (close>static) qty=1.00 sl=1885.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 1890.50 | 1878.42 | 1878.06 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 1856.00 | 1878.13 | 1878.80 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 14:15:00 | 1886.65 | 1879.58 | 1878.80 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 1870.70 | 1878.06 | 1878.26 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 1882.95 | 1879.04 | 1878.69 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 1871.95 | 1877.62 | 1878.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 1861.70 | 1874.44 | 1876.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1860.80 | 1843.53 | 1852.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1860.80 | 1843.53 | 1852.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1860.80 | 1843.53 | 1852.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 1860.00 | 1843.53 | 1852.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 1853.50 | 1845.52 | 1852.73 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 1860.00 | 1856.47 | 1856.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 1879.90 | 1861.16 | 1858.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 1872.70 | 1873.04 | 1867.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 11:15:00 | 1872.70 | 1873.04 | 1867.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1872.70 | 1873.04 | 1867.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 1870.10 | 1873.04 | 1867.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1884.65 | 1878.01 | 1872.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:30:00 | 1894.85 | 1886.13 | 1879.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:00:00 | 1898.70 | 1886.13 | 1879.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 1918.10 | 1932.32 | 1933.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 1918.10 | 1932.32 | 1933.28 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1950.10 | 1934.47 | 1932.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 1968.75 | 1945.95 | 1938.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 09:15:00 | 1946.90 | 1953.67 | 1945.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 10:00:00 | 1946.90 | 1953.67 | 1945.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1953.05 | 1953.54 | 1946.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 2017.00 | 1957.04 | 1950.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 13:15:00 | 1950.50 | 1960.40 | 1961.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 13:15:00 | 1950.50 | 1960.40 | 1961.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 15:15:00 | 1944.70 | 1956.13 | 1959.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 10:15:00 | 1965.00 | 1957.09 | 1959.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 10:15:00 | 1965.00 | 1957.09 | 1959.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1965.00 | 1957.09 | 1959.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:00:00 | 1965.00 | 1957.09 | 1959.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 1975.00 | 1960.67 | 1960.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 12:15:00 | 1980.70 | 1964.68 | 1962.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 1951.10 | 1966.77 | 1964.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 1951.10 | 1966.77 | 1964.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1951.10 | 1966.77 | 1964.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 1951.10 | 1966.77 | 1964.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 1948.00 | 1963.01 | 1963.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 1940.45 | 1956.22 | 1959.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1955.35 | 1933.00 | 1941.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1955.35 | 1933.00 | 1941.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1955.35 | 1933.00 | 1941.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 1959.80 | 1933.00 | 1941.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 1988.00 | 1944.00 | 1945.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 1988.00 | 1944.00 | 1945.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 1997.90 | 1954.78 | 1950.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 2003.90 | 1986.68 | 1970.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 1980.55 | 1992.89 | 1979.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 14:15:00 | 1980.55 | 1992.89 | 1979.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1980.55 | 1992.89 | 1979.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 1980.55 | 1992.89 | 1979.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1982.00 | 1990.71 | 1979.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 1986.15 | 1990.71 | 1979.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:45:00 | 1985.10 | 1990.30 | 1980.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 15:15:00 | 2027.90 | 2033.95 | 2034.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 15:15:00 | 2027.90 | 2033.95 | 2034.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 2010.00 | 2029.16 | 2031.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1982.90 | 1980.20 | 1993.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 11:00:00 | 1982.90 | 1980.20 | 1993.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 1998.55 | 1975.94 | 1984.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 2005.00 | 1975.94 | 1984.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 2002.20 | 1981.19 | 1986.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:30:00 | 1995.90 | 1981.19 | 1986.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 2008.35 | 1991.47 | 1990.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 11:15:00 | 2029.70 | 2011.90 | 2001.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 11:15:00 | 2133.15 | 2134.96 | 2107.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 12:00:00 | 2133.15 | 2134.96 | 2107.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 2104.80 | 2125.49 | 2113.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 2105.25 | 2125.49 | 2113.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 2110.00 | 2122.39 | 2113.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 11:15:00 | 2118.15 | 2122.39 | 2113.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 12:15:00 | 2122.15 | 2140.96 | 2141.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 2122.15 | 2140.96 | 2141.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 13:15:00 | 2096.30 | 2132.03 | 2136.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 2013.05 | 2009.06 | 2050.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 2013.05 | 2009.06 | 2050.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1987.30 | 1967.54 | 1983.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:00:00 | 1987.30 | 1967.54 | 1983.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1985.00 | 1971.03 | 1983.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 11:15:00 | 1978.55 | 1971.03 | 1983.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 12:00:00 | 1978.20 | 1972.47 | 1982.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 12:45:00 | 1978.90 | 1972.97 | 1982.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:15:00 | 1879.95 | 1916.11 | 1928.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:15:00 | 1879.62 | 1908.68 | 1924.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:15:00 | 1879.29 | 1908.68 | 1924.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 13:15:00 | 1865.70 | 1861.50 | 1882.24 | SL hit (close>ema200) qty=0.50 sl=1861.50 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 1814.30 | 1791.51 | 1789.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 1819.20 | 1797.05 | 1792.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 1785.45 | 1796.96 | 1793.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 13:15:00 | 1785.45 | 1796.96 | 1793.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 1785.45 | 1796.96 | 1793.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 1785.45 | 1796.96 | 1793.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 1786.00 | 1794.77 | 1792.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:15:00 | 1792.10 | 1794.77 | 1792.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 1792.10 | 1794.24 | 1792.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 1774.65 | 1794.24 | 1792.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 1774.70 | 1790.33 | 1790.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 1761.10 | 1784.48 | 1788.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 1776.55 | 1772.80 | 1780.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 15:00:00 | 1776.55 | 1772.80 | 1780.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1791.95 | 1776.78 | 1780.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1791.95 | 1776.78 | 1780.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1761.30 | 1774.20 | 1779.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:15:00 | 1755.10 | 1774.20 | 1779.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 1782.35 | 1759.30 | 1757.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 1782.35 | 1759.30 | 1757.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 1789.90 | 1765.42 | 1760.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 15:15:00 | 1802.00 | 1804.52 | 1790.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 09:15:00 | 1808.85 | 1804.52 | 1790.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1805.15 | 1804.64 | 1792.11 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 1766.00 | 1786.32 | 1787.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 1754.50 | 1773.14 | 1778.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1776.15 | 1758.16 | 1764.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1776.15 | 1758.16 | 1764.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1776.15 | 1758.16 | 1764.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1776.15 | 1758.16 | 1764.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1749.75 | 1756.48 | 1763.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 1734.60 | 1754.56 | 1759.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 12:15:00 | 1760.95 | 1750.73 | 1750.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 1760.95 | 1750.73 | 1750.58 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 1733.30 | 1747.57 | 1749.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 1730.60 | 1744.18 | 1747.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 1719.25 | 1716.89 | 1726.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 12:00:00 | 1719.25 | 1716.89 | 1726.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1729.05 | 1719.32 | 1726.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 1729.05 | 1719.32 | 1726.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1732.90 | 1722.04 | 1727.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 1732.90 | 1722.04 | 1727.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1737.15 | 1725.06 | 1728.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:45:00 | 1744.35 | 1725.06 | 1728.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1780.80 | 1738.60 | 1733.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 1785.55 | 1747.99 | 1738.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 1778.40 | 1780.21 | 1764.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 1778.40 | 1780.21 | 1764.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 1768.35 | 1776.79 | 1766.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 1768.35 | 1776.79 | 1766.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 1765.95 | 1774.62 | 1766.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 1757.10 | 1774.62 | 1766.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1758.10 | 1771.32 | 1765.94 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 09:15:00 | 1756.60 | 1764.10 | 1764.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 1735.00 | 1758.28 | 1761.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1753.90 | 1747.36 | 1753.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 1753.90 | 1747.36 | 1753.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1753.90 | 1747.36 | 1753.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 1753.90 | 1747.36 | 1753.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 1745.30 | 1746.95 | 1752.42 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 1773.75 | 1757.11 | 1755.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 1796.00 | 1769.06 | 1761.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 1798.95 | 1802.55 | 1790.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 1798.95 | 1802.55 | 1790.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1798.95 | 1802.55 | 1790.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 1797.25 | 1802.55 | 1790.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1802.40 | 1803.06 | 1793.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:30:00 | 1802.05 | 1803.06 | 1793.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1790.00 | 1799.16 | 1794.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 1790.00 | 1799.16 | 1794.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 1808.25 | 1800.98 | 1795.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 14:00:00 | 1809.50 | 1803.61 | 1797.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 14:15:00 | 1786.90 | 1796.36 | 1797.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 14:15:00 | 1786.90 | 1796.36 | 1797.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 1781.85 | 1792.84 | 1795.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 15:15:00 | 1790.30 | 1789.15 | 1792.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 09:15:00 | 1789.85 | 1789.15 | 1792.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1788.70 | 1789.06 | 1791.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:15:00 | 1781.80 | 1790.27 | 1791.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 1808.05 | 1786.59 | 1788.50 | SL hit (close>static) qty=1.00 sl=1798.55 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 1814.00 | 1792.07 | 1790.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 13:15:00 | 1833.95 | 1807.34 | 1798.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 1797.65 | 1831.94 | 1822.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 1797.65 | 1831.94 | 1822.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1797.65 | 1831.94 | 1822.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 1797.65 | 1831.94 | 1822.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 1796.90 | 1824.93 | 1820.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 1793.20 | 1824.93 | 1820.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 1825.60 | 1824.40 | 1821.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:30:00 | 1820.10 | 1824.40 | 1821.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1830.20 | 1843.78 | 1838.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:45:00 | 1831.25 | 1843.78 | 1838.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 1812.05 | 1837.43 | 1835.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 1812.05 | 1837.43 | 1835.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 1812.50 | 1832.45 | 1833.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 1811.00 | 1828.16 | 1831.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 13:15:00 | 1722.10 | 1718.70 | 1733.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 14:00:00 | 1722.10 | 1718.70 | 1733.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 1722.25 | 1717.47 | 1725.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:00:00 | 1722.25 | 1717.47 | 1725.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 1726.45 | 1719.26 | 1726.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:30:00 | 1727.10 | 1719.26 | 1726.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1732.95 | 1722.00 | 1726.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1732.95 | 1722.00 | 1726.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1728.75 | 1723.35 | 1726.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1741.60 | 1723.35 | 1726.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1723.90 | 1725.28 | 1727.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 1717.85 | 1725.28 | 1727.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:00:00 | 1721.95 | 1718.56 | 1721.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:45:00 | 1722.85 | 1718.57 | 1721.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 1722.00 | 1710.98 | 1710.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 1722.00 | 1710.98 | 1710.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 1726.80 | 1714.14 | 1712.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 1716.70 | 1719.17 | 1715.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 1716.70 | 1719.17 | 1715.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1716.70 | 1719.17 | 1715.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:30:00 | 1719.60 | 1719.17 | 1715.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 1730.20 | 1721.38 | 1716.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:45:00 | 1733.40 | 1723.46 | 1718.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 12:15:00 | 1734.75 | 1723.46 | 1718.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 11:15:00 | 1714.40 | 1730.06 | 1730.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 1714.40 | 1730.06 | 1730.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 1689.65 | 1714.65 | 1722.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 10:15:00 | 1695.60 | 1694.56 | 1703.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 10:45:00 | 1697.90 | 1694.56 | 1703.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 1712.45 | 1698.14 | 1704.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:00:00 | 1712.45 | 1698.14 | 1704.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 1699.85 | 1698.48 | 1704.14 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 1742.40 | 1711.20 | 1708.83 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 1698.45 | 1709.50 | 1709.74 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 11:15:00 | 1719.75 | 1710.59 | 1710.13 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 1688.00 | 1711.49 | 1711.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 1659.45 | 1696.17 | 1704.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 1661.05 | 1659.79 | 1680.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 1703.10 | 1668.45 | 1682.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1703.10 | 1668.45 | 1682.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 1703.10 | 1668.45 | 1682.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1679.70 | 1670.70 | 1682.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 1674.00 | 1670.70 | 1682.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 13:30:00 | 1671.15 | 1672.62 | 1681.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 15:00:00 | 1674.70 | 1673.03 | 1680.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 1675.10 | 1674.06 | 1680.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1683.45 | 1675.93 | 1680.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:45:00 | 1686.65 | 1675.93 | 1680.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 1690.00 | 1678.75 | 1681.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 1695.85 | 1678.75 | 1681.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-15 12:15:00 | 1689.35 | 1683.73 | 1683.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 1689.35 | 1683.73 | 1683.40 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 13:15:00 | 1676.55 | 1682.30 | 1682.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 14:15:00 | 1673.50 | 1680.54 | 1681.94 | Break + close below crossover candle low |

### Cycle 55 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 1698.00 | 1683.94 | 1683.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 12:15:00 | 1708.90 | 1690.47 | 1686.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 12:15:00 | 1701.65 | 1703.96 | 1697.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 12:30:00 | 1707.95 | 1703.96 | 1697.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1697.00 | 1702.22 | 1697.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 15:00:00 | 1697.00 | 1702.22 | 1697.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 1696.60 | 1701.09 | 1697.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 1693.00 | 1701.09 | 1697.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1681.70 | 1697.21 | 1695.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 1681.70 | 1697.21 | 1695.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1701.95 | 1698.16 | 1696.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:15:00 | 1702.15 | 1698.16 | 1696.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:30:00 | 1712.90 | 1710.54 | 1706.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 1677.45 | 1700.33 | 1702.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 1677.45 | 1700.33 | 1702.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 1666.20 | 1693.50 | 1699.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1681.00 | 1674.83 | 1687.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 1681.00 | 1674.83 | 1687.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1657.30 | 1671.03 | 1683.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 10:45:00 | 1629.35 | 1662.82 | 1678.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:00:00 | 1633.35 | 1646.83 | 1657.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 1551.68 | 1587.90 | 1614.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 10:15:00 | 1573.10 | 1570.14 | 1588.99 | SL hit (close>ema200) qty=0.50 sl=1570.14 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 1590.30 | 1583.40 | 1583.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 1611.40 | 1592.91 | 1587.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 10:15:00 | 1614.90 | 1615.14 | 1602.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:00:00 | 1614.90 | 1615.14 | 1602.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1608.30 | 1613.77 | 1603.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1605.40 | 1613.77 | 1603.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1599.15 | 1610.85 | 1603.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1599.15 | 1610.85 | 1603.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1614.65 | 1611.61 | 1604.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1617.80 | 1611.61 | 1604.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 1619.55 | 1613.20 | 1605.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 1577.35 | 1605.14 | 1603.11 | SL hit (close<static) qty=1.00 sl=1595.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 1578.10 | 1599.73 | 1600.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 1568.00 | 1593.38 | 1597.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1574.35 | 1574.33 | 1585.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 1574.35 | 1574.33 | 1585.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1574.35 | 1574.33 | 1585.29 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 1616.55 | 1589.18 | 1586.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 12:15:00 | 1626.75 | 1600.83 | 1592.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 1604.00 | 1604.75 | 1597.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 10:15:00 | 1604.65 | 1604.75 | 1597.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1595.60 | 1602.00 | 1597.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 1595.60 | 1602.00 | 1597.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1588.00 | 1599.20 | 1596.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 1588.00 | 1599.20 | 1596.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 1585.45 | 1593.39 | 1594.25 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 10:15:00 | 1614.55 | 1597.27 | 1595.70 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 1577.85 | 1594.07 | 1595.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 1568.50 | 1588.96 | 1593.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 1530.10 | 1522.63 | 1542.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 1530.10 | 1522.63 | 1542.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1550.55 | 1528.13 | 1538.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 1549.15 | 1528.13 | 1538.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1550.55 | 1532.61 | 1539.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 1536.70 | 1535.72 | 1539.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 14:15:00 | 1502.75 | 1500.08 | 1499.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 1502.75 | 1500.08 | 1499.92 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 10:15:00 | 1497.85 | 1499.84 | 1499.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 11:15:00 | 1487.85 | 1497.44 | 1498.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 1344.70 | 1323.56 | 1335.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 1344.70 | 1323.56 | 1335.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1344.70 | 1323.56 | 1335.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 1344.70 | 1323.56 | 1335.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1341.60 | 1327.17 | 1336.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 1347.80 | 1327.17 | 1336.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 1383.65 | 1342.58 | 1341.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 1387.75 | 1351.62 | 1346.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 1390.85 | 1391.74 | 1377.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:45:00 | 1386.00 | 1391.74 | 1377.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 1383.75 | 1388.18 | 1380.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:00:00 | 1383.75 | 1388.18 | 1380.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1385.00 | 1386.92 | 1380.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 1379.15 | 1386.92 | 1380.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1391.30 | 1387.80 | 1381.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 1384.55 | 1387.80 | 1381.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1381.00 | 1386.44 | 1381.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 1381.00 | 1386.44 | 1381.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 1380.90 | 1385.33 | 1381.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:15:00 | 1378.50 | 1385.33 | 1381.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 1380.40 | 1384.34 | 1381.61 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1366.00 | 1377.88 | 1379.08 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 09:15:00 | 1430.20 | 1388.34 | 1383.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 10:15:00 | 1459.90 | 1402.65 | 1390.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 1503.35 | 1506.67 | 1484.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 15:00:00 | 1503.35 | 1506.67 | 1484.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1498.85 | 1503.69 | 1488.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 1488.75 | 1503.69 | 1488.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 1505.15 | 1502.90 | 1491.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:30:00 | 1492.80 | 1502.90 | 1491.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 1498.10 | 1504.52 | 1497.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:30:00 | 1491.90 | 1504.52 | 1497.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 1499.20 | 1503.46 | 1497.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:30:00 | 1494.70 | 1503.46 | 1497.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 1506.75 | 1504.12 | 1498.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 1570.65 | 1504.69 | 1499.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 13:15:00 | 1588.35 | 1595.04 | 1595.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 13:15:00 | 1588.35 | 1595.04 | 1595.73 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 14:15:00 | 1606.35 | 1593.86 | 1593.40 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 1584.55 | 1593.14 | 1593.64 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 1597.00 | 1594.05 | 1593.98 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 1590.00 | 1593.24 | 1593.61 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1597.10 | 1594.01 | 1593.93 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 1591.25 | 1593.46 | 1593.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 1590.50 | 1592.89 | 1593.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 1605.25 | 1589.30 | 1590.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 1605.25 | 1589.30 | 1590.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1605.25 | 1589.30 | 1590.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 1605.25 | 1589.30 | 1590.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1577.60 | 1586.96 | 1589.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:45:00 | 1575.20 | 1585.05 | 1588.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:30:00 | 1575.65 | 1582.47 | 1586.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 14:15:00 | 1574.65 | 1582.47 | 1586.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 1570.40 | 1581.38 | 1585.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1580.75 | 1581.26 | 1585.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-02 13:15:00 | 1591.75 | 1586.86 | 1586.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 1591.75 | 1586.86 | 1586.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 1595.60 | 1588.60 | 1587.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1595.35 | 1606.43 | 1598.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1595.35 | 1606.43 | 1598.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1595.35 | 1606.43 | 1598.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1595.35 | 1606.43 | 1598.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1617.25 | 1608.60 | 1600.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 1607.50 | 1608.60 | 1600.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 1602.35 | 1607.20 | 1601.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:45:00 | 1604.45 | 1607.20 | 1601.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 1597.05 | 1605.17 | 1600.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:00:00 | 1597.05 | 1605.17 | 1600.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 1598.95 | 1603.93 | 1600.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:30:00 | 1591.10 | 1603.93 | 1600.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1517.85 | 1586.24 | 1593.14 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 1587.45 | 1565.55 | 1563.04 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 09:15:00 | 1548.00 | 1562.79 | 1563.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-15 11:15:00 | 1544.80 | 1557.29 | 1560.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 13:15:00 | 1557.70 | 1556.56 | 1559.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 13:15:00 | 1557.70 | 1556.56 | 1559.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 13:15:00 | 1557.70 | 1556.56 | 1559.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 13:30:00 | 1558.50 | 1556.56 | 1559.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 14:15:00 | 1591.40 | 1563.53 | 1562.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 1598.00 | 1574.35 | 1567.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1584.90 | 1588.71 | 1580.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1584.90 | 1588.71 | 1580.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1584.90 | 1588.71 | 1580.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 1581.70 | 1588.71 | 1580.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 1582.20 | 1591.53 | 1585.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:30:00 | 1584.10 | 1591.53 | 1585.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 1565.60 | 1586.34 | 1583.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 1565.60 | 1586.34 | 1583.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 09:15:00 | 1538.90 | 1573.76 | 1577.88 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 1594.20 | 1578.72 | 1578.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 11:15:00 | 1607.60 | 1584.49 | 1580.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 10:15:00 | 1590.00 | 1594.21 | 1588.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 10:15:00 | 1590.00 | 1594.21 | 1588.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1590.00 | 1594.21 | 1588.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1586.40 | 1594.21 | 1588.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1592.90 | 1593.95 | 1588.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:45:00 | 1579.00 | 1593.95 | 1588.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 1592.20 | 1593.60 | 1589.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:45:00 | 1596.30 | 1593.60 | 1589.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 1582.50 | 1591.38 | 1588.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:00:00 | 1582.50 | 1591.38 | 1588.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 1583.40 | 1589.78 | 1588.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:45:00 | 1583.30 | 1589.78 | 1588.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1589.00 | 1589.07 | 1588.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:30:00 | 1569.40 | 1589.07 | 1588.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1605.30 | 1592.31 | 1589.65 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1549.60 | 1583.25 | 1587.14 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 1584.80 | 1580.99 | 1580.51 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 1576.60 | 1580.11 | 1580.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 12:15:00 | 1571.80 | 1578.45 | 1579.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 14:15:00 | 1578.50 | 1577.72 | 1578.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 14:15:00 | 1578.50 | 1577.72 | 1578.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 1578.50 | 1577.72 | 1578.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 1578.50 | 1577.72 | 1578.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 1578.10 | 1577.79 | 1578.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 1580.50 | 1577.79 | 1578.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 1592.10 | 1580.65 | 1579.99 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 1567.50 | 1581.13 | 1582.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 13:15:00 | 1561.70 | 1575.34 | 1579.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1575.50 | 1570.45 | 1575.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1575.50 | 1570.45 | 1575.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1575.50 | 1570.45 | 1575.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 1578.30 | 1570.45 | 1575.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1572.10 | 1570.78 | 1575.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:15:00 | 1567.60 | 1570.78 | 1575.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 1585.10 | 1573.65 | 1576.31 | SL hit (close>static) qty=1.00 sl=1576.60 alert=retest2 |

### Cycle 87 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 1588.10 | 1579.54 | 1578.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 1592.70 | 1582.17 | 1579.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 1576.60 | 1581.06 | 1579.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 1576.60 | 1581.06 | 1579.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1576.60 | 1581.06 | 1579.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 1576.80 | 1581.06 | 1579.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 1567.00 | 1578.25 | 1578.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 1560.00 | 1574.60 | 1576.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 1549.00 | 1545.43 | 1554.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:15:00 | 1547.50 | 1545.43 | 1554.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1549.90 | 1546.32 | 1554.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:45:00 | 1537.50 | 1544.36 | 1552.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:30:00 | 1537.20 | 1541.69 | 1550.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 11:15:00 | 1555.90 | 1533.27 | 1531.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1555.90 | 1533.27 | 1531.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1563.70 | 1539.36 | 1534.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 1559.20 | 1559.21 | 1548.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 11:30:00 | 1553.80 | 1559.21 | 1548.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1631.80 | 1639.68 | 1626.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1631.60 | 1639.68 | 1626.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1621.00 | 1635.95 | 1626.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 1621.00 | 1635.95 | 1626.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1630.30 | 1634.82 | 1626.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 12:15:00 | 1634.00 | 1634.82 | 1626.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:00:00 | 1635.10 | 1632.49 | 1627.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 1671.20 | 1683.74 | 1685.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 1671.20 | 1683.74 | 1685.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1659.40 | 1674.57 | 1679.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 1676.20 | 1673.22 | 1677.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 12:15:00 | 1676.20 | 1673.22 | 1677.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1676.20 | 1673.22 | 1677.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:30:00 | 1677.80 | 1673.22 | 1677.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1674.90 | 1673.55 | 1677.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 1660.10 | 1674.82 | 1677.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 1667.80 | 1672.28 | 1675.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:00:00 | 1666.00 | 1671.03 | 1674.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 13:15:00 | 1682.00 | 1666.22 | 1667.90 | SL hit (close>static) qty=1.00 sl=1678.60 alert=retest2 |

### Cycle 91 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 1682.10 | 1669.39 | 1669.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 15:15:00 | 1684.20 | 1672.36 | 1670.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 15:15:00 | 1711.00 | 1712.00 | 1702.63 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:15:00 | 1728.90 | 1712.00 | 1702.63 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1718.40 | 1732.00 | 1727.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 1718.40 | 1732.00 | 1727.32 | SL hit (close<ema400) qty=1.00 sl=1727.32 alert=retest1 |

### Cycle 92 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1713.00 | 1723.58 | 1724.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1693.90 | 1710.67 | 1716.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1704.00 | 1703.85 | 1709.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:30:00 | 1703.80 | 1703.85 | 1709.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1706.10 | 1704.30 | 1709.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1706.10 | 1704.30 | 1709.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1701.10 | 1703.66 | 1708.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 1701.10 | 1703.66 | 1708.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1697.60 | 1702.45 | 1707.49 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 1712.00 | 1707.97 | 1707.87 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1704.00 | 1708.80 | 1709.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1665.60 | 1696.87 | 1702.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 13:15:00 | 1651.20 | 1650.12 | 1666.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 13:45:00 | 1651.30 | 1650.12 | 1666.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1646.60 | 1649.97 | 1662.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 09:30:00 | 1637.90 | 1648.04 | 1655.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 11:15:00 | 1669.00 | 1654.13 | 1656.74 | SL hit (close>static) qty=1.00 sl=1668.50 alert=retest2 |

### Cycle 95 — BUY (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 12:15:00 | 1669.00 | 1656.98 | 1656.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 1672.10 | 1661.50 | 1658.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 1664.60 | 1665.38 | 1661.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 1664.60 | 1665.38 | 1661.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1667.90 | 1665.88 | 1661.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 1684.40 | 1667.63 | 1663.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1752.00 | 1764.20 | 1764.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 1752.00 | 1764.20 | 1764.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 1722.80 | 1746.60 | 1752.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 1713.60 | 1713.00 | 1724.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 1713.60 | 1713.00 | 1724.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1719.70 | 1713.78 | 1722.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 1713.60 | 1713.78 | 1722.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1725.10 | 1716.05 | 1723.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 1725.10 | 1716.05 | 1723.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1725.50 | 1717.94 | 1723.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 1725.20 | 1717.94 | 1723.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1719.50 | 1718.25 | 1723.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 1719.20 | 1718.25 | 1723.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1729.60 | 1720.74 | 1723.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1729.60 | 1720.74 | 1723.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1730.00 | 1722.59 | 1723.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 1734.00 | 1722.59 | 1723.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 1728.30 | 1725.11 | 1724.93 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 1721.50 | 1724.20 | 1724.55 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 1731.70 | 1725.59 | 1725.10 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 1722.20 | 1724.75 | 1725.04 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 14:15:00 | 1730.10 | 1725.82 | 1725.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 1791.10 | 1739.47 | 1731.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 13:15:00 | 1771.40 | 1773.34 | 1759.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 14:00:00 | 1771.40 | 1773.34 | 1759.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1761.00 | 1771.19 | 1762.31 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 1734.30 | 1755.39 | 1757.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 1729.90 | 1744.62 | 1751.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 1743.70 | 1736.78 | 1744.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 1743.70 | 1736.78 | 1744.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 1743.70 | 1736.78 | 1744.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 1737.90 | 1736.78 | 1744.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1728.80 | 1728.60 | 1735.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1715.00 | 1728.66 | 1734.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 15:15:00 | 1720.00 | 1728.58 | 1731.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 1738.90 | 1729.27 | 1731.28 | SL hit (close>static) qty=1.00 sl=1735.60 alert=retest2 |

### Cycle 103 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1748.90 | 1730.61 | 1728.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 1763.70 | 1743.80 | 1735.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1720.80 | 1743.90 | 1738.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1720.80 | 1743.90 | 1738.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1720.80 | 1743.90 | 1738.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 1720.80 | 1743.90 | 1738.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1737.50 | 1742.62 | 1738.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:30:00 | 1724.50 | 1742.62 | 1738.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1763.80 | 1746.86 | 1740.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:15:00 | 1775.40 | 1746.86 | 1740.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 1732.30 | 1745.22 | 1741.52 | SL hit (close<static) qty=1.00 sl=1737.20 alert=retest2 |

### Cycle 104 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 1711.60 | 1738.50 | 1738.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 1691.30 | 1724.69 | 1732.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 1679.50 | 1663.92 | 1684.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 15:00:00 | 1679.50 | 1663.92 | 1684.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1660.80 | 1665.39 | 1681.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1656.10 | 1671.56 | 1673.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 1655.00 | 1665.70 | 1670.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:45:00 | 1655.10 | 1661.87 | 1667.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:15:00 | 1655.00 | 1661.01 | 1667.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 1667.00 | 1661.27 | 1666.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 1652.40 | 1661.27 | 1666.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 10:30:00 | 1654.60 | 1658.51 | 1663.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:00:00 | 1652.00 | 1657.46 | 1662.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 1653.80 | 1658.73 | 1661.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1656.20 | 1658.22 | 1661.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:15:00 | 1651.50 | 1657.62 | 1660.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 1671.40 | 1648.03 | 1647.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 1671.40 | 1648.03 | 1647.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 1675.00 | 1653.42 | 1649.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 1677.00 | 1678.95 | 1667.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 11:45:00 | 1677.60 | 1678.95 | 1667.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 1657.40 | 1676.14 | 1669.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 1657.40 | 1676.14 | 1669.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 1686.00 | 1678.11 | 1670.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 1667.40 | 1678.11 | 1670.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1660.70 | 1674.63 | 1669.72 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 1650.00 | 1666.12 | 1666.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 15:15:00 | 1642.00 | 1652.14 | 1657.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 15:15:00 | 1608.00 | 1605.85 | 1618.28 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 09:15:00 | 1600.70 | 1605.85 | 1618.28 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1598.60 | 1604.40 | 1616.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 1598.60 | 1604.40 | 1616.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1570.60 | 1591.74 | 1603.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1558.00 | 1574.42 | 1588.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:00:00 | 1552.00 | 1568.75 | 1583.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:00:00 | 1555.60 | 1566.12 | 1580.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 15:15:00 | 1556.00 | 1562.29 | 1575.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:15:00 | 1520.66 | 1548.99 | 1565.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 1549.10 | 1547.49 | 1560.54 | SL hit (close>ema200) qty=0.50 sl=1547.49 alert=retest1 |

### Cycle 107 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1561.50 | 1558.66 | 1558.28 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 1551.70 | 1560.53 | 1560.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1549.70 | 1558.36 | 1559.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 1550.70 | 1550.45 | 1553.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 1547.00 | 1550.45 | 1553.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1550.80 | 1550.52 | 1553.57 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 1575.10 | 1556.93 | 1554.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 11:15:00 | 1584.50 | 1562.44 | 1557.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 1578.50 | 1588.32 | 1578.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 12:15:00 | 1578.50 | 1588.32 | 1578.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1578.50 | 1588.32 | 1578.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1578.50 | 1588.32 | 1578.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1585.20 | 1587.70 | 1578.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:30:00 | 1586.00 | 1587.70 | 1578.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1583.30 | 1586.82 | 1579.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 1589.70 | 1583.74 | 1581.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 1587.90 | 1583.91 | 1581.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1679.90 | 1686.94 | 1687.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1679.90 | 1686.94 | 1687.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 1671.50 | 1683.85 | 1685.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 1669.70 | 1658.98 | 1666.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 15:15:00 | 1669.70 | 1658.98 | 1666.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1669.70 | 1658.98 | 1666.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 1645.80 | 1658.08 | 1659.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 1633.60 | 1616.78 | 1616.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 1633.60 | 1616.78 | 1616.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 1643.90 | 1622.21 | 1618.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 1658.10 | 1658.44 | 1646.76 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1695.80 | 1658.44 | 1646.76 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 09:15:00 | 1780.59 | 1713.99 | 1690.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-10-10 10:15:00 | 1865.38 | 1743.19 | 1705.72 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 112 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 1873.30 | 1913.81 | 1917.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 1870.00 | 1896.12 | 1905.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 13:15:00 | 1883.00 | 1868.86 | 1881.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 13:15:00 | 1883.00 | 1868.86 | 1881.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1883.00 | 1868.86 | 1881.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:00:00 | 1883.00 | 1868.86 | 1881.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1896.00 | 1874.29 | 1882.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 1896.00 | 1874.29 | 1882.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 1890.00 | 1877.43 | 1883.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 1907.00 | 1877.43 | 1883.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 1926.80 | 1887.30 | 1887.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 1941.00 | 1903.91 | 1895.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 1905.30 | 1920.59 | 1908.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1905.30 | 1920.59 | 1908.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1905.30 | 1920.59 | 1908.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 1905.30 | 1920.59 | 1908.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1910.20 | 1918.51 | 1908.69 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1893.20 | 1905.48 | 1905.51 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 1910.00 | 1905.73 | 1905.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 1934.20 | 1912.24 | 1908.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 14:15:00 | 1912.90 | 1914.24 | 1910.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 14:15:00 | 1912.90 | 1914.24 | 1910.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 1912.90 | 1914.24 | 1910.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 1912.90 | 1914.24 | 1910.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 1916.00 | 1914.59 | 1910.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 1906.90 | 1914.59 | 1910.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1897.00 | 1911.07 | 1909.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 1898.10 | 1911.07 | 1909.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 1887.30 | 1906.32 | 1907.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1872.80 | 1895.49 | 1901.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 1879.20 | 1878.91 | 1889.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:45:00 | 1881.10 | 1878.91 | 1889.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1905.70 | 1884.32 | 1890.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 1905.70 | 1884.32 | 1890.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1905.00 | 1888.45 | 1891.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 1925.80 | 1888.45 | 1891.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 1919.70 | 1894.70 | 1894.28 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 1878.90 | 1894.16 | 1895.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 1869.00 | 1889.13 | 1893.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 1853.00 | 1852.22 | 1865.48 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1843.50 | 1852.22 | 1865.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1851.00 | 1848.34 | 1861.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1856.50 | 1848.34 | 1861.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 1844.70 | 1845.63 | 1856.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:45:00 | 1835.00 | 1844.89 | 1853.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 1833.00 | 1838.85 | 1848.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:30:00 | 1837.30 | 1835.56 | 1843.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 1834.10 | 1827.56 | 1835.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 1854.00 | 1832.85 | 1837.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-12 15:15:00 | 1854.00 | 1832.85 | 1837.43 | SL hit (close>ema400) qty=1.00 sl=1837.43 alert=retest1 |

### Cycle 119 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 1884.00 | 1847.79 | 1843.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 13:15:00 | 1896.70 | 1863.21 | 1851.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 14:15:00 | 1870.20 | 1885.19 | 1872.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 1870.20 | 1885.19 | 1872.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1870.20 | 1885.19 | 1872.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 1870.20 | 1885.19 | 1872.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1879.00 | 1883.95 | 1873.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 1896.90 | 1883.95 | 1873.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:45:00 | 1888.80 | 1897.77 | 1890.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 1889.10 | 1891.22 | 1889.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 1879.20 | 1887.60 | 1887.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 1879.20 | 1887.60 | 1887.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 11:15:00 | 1872.00 | 1884.48 | 1886.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1900.00 | 1881.05 | 1882.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1900.00 | 1881.05 | 1882.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1900.00 | 1881.05 | 1882.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 1906.20 | 1881.05 | 1882.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 1901.00 | 1885.04 | 1884.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 11:15:00 | 1907.20 | 1889.47 | 1886.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1903.30 | 1907.88 | 1898.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 09:45:00 | 1910.70 | 1907.88 | 1898.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1900.50 | 1906.40 | 1898.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 1898.70 | 1906.40 | 1898.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1906.50 | 1906.42 | 1899.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:30:00 | 1907.50 | 1906.24 | 1900.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 15:00:00 | 1919.60 | 1908.52 | 1902.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1897.10 | 1908.09 | 1903.23 | SL hit (close<static) qty=1.00 sl=1899.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 1887.80 | 1899.00 | 1899.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 1864.00 | 1889.15 | 1895.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 12:15:00 | 1837.00 | 1833.81 | 1852.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 13:00:00 | 1837.00 | 1833.81 | 1852.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 1847.80 | 1836.61 | 1851.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:45:00 | 1847.20 | 1836.61 | 1851.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 1854.90 | 1840.27 | 1852.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:45:00 | 1851.40 | 1840.27 | 1852.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 1850.00 | 1842.21 | 1851.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 1827.10 | 1839.91 | 1849.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 1867.30 | 1845.39 | 1851.56 | SL hit (close>static) qty=1.00 sl=1855.60 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 1846.00 | 1837.49 | 1836.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 09:15:00 | 1870.00 | 1844.00 | 1839.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 15:15:00 | 1860.00 | 1862.98 | 1853.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 15:15:00 | 1860.00 | 1862.98 | 1853.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 1860.00 | 1862.98 | 1853.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 1850.40 | 1862.98 | 1853.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1850.20 | 1860.43 | 1853.22 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 1835.30 | 1847.22 | 1848.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 13:15:00 | 1831.90 | 1838.07 | 1842.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 1838.50 | 1838.16 | 1842.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 1838.50 | 1838.16 | 1842.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1838.50 | 1838.16 | 1842.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 1838.50 | 1838.16 | 1842.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1823.10 | 1834.34 | 1840.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 13:00:00 | 1819.90 | 1831.08 | 1837.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 14:00:00 | 1814.90 | 1827.84 | 1835.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 13:45:00 | 1811.30 | 1809.84 | 1820.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1811.80 | 1797.30 | 1795.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 1811.80 | 1797.30 | 1795.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 11:15:00 | 1840.00 | 1808.14 | 1801.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 1823.70 | 1825.32 | 1816.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 13:00:00 | 1823.70 | 1825.32 | 1816.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1807.10 | 1821.81 | 1816.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 1803.90 | 1821.81 | 1816.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1798.60 | 1817.17 | 1815.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 1796.60 | 1817.17 | 1815.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1807.00 | 1815.13 | 1814.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 1810.00 | 1815.13 | 1814.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 1809.40 | 1813.99 | 1813.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 12:15:00 | 1800.90 | 1811.37 | 1812.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 12:15:00 | 1800.90 | 1811.37 | 1812.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 1789.10 | 1806.92 | 1810.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1817.30 | 1804.62 | 1808.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 1817.30 | 1804.62 | 1808.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1817.30 | 1804.62 | 1808.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 1817.30 | 1804.62 | 1808.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1813.00 | 1806.30 | 1808.65 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 1822.80 | 1811.01 | 1810.47 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 14:15:00 | 1798.80 | 1810.18 | 1811.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 09:15:00 | 1792.30 | 1804.50 | 1808.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 1795.20 | 1785.41 | 1791.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 1795.20 | 1785.41 | 1791.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1795.20 | 1785.41 | 1791.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 1801.50 | 1785.41 | 1791.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1785.20 | 1785.36 | 1791.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 1779.90 | 1785.36 | 1791.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1797.90 | 1788.18 | 1788.42 | SL hit (close>static) qty=1.00 sl=1797.30 alert=retest2 |

### Cycle 129 — BUY (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 13:15:00 | 1795.40 | 1789.63 | 1789.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 14:15:00 | 1799.90 | 1791.68 | 1790.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 15:15:00 | 1789.90 | 1791.32 | 1790.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 15:15:00 | 1789.90 | 1791.32 | 1790.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1789.90 | 1791.32 | 1790.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 1779.60 | 1791.32 | 1790.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1790.40 | 1791.14 | 1790.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 1802.20 | 1791.14 | 1790.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:45:00 | 1804.80 | 1811.82 | 1809.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 12:15:00 | 1800.00 | 1807.73 | 1808.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 1800.00 | 1807.73 | 1808.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 14:15:00 | 1795.20 | 1803.87 | 1806.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 10:15:00 | 1804.30 | 1801.91 | 1804.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 1804.30 | 1801.91 | 1804.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1804.30 | 1801.91 | 1804.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1796.00 | 1801.00 | 1802.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 1813.70 | 1804.04 | 1804.05 | SL hit (close>static) qty=1.00 sl=1808.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 1815.40 | 1806.31 | 1805.08 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 1801.00 | 1804.13 | 1804.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 1793.60 | 1802.03 | 1803.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 1797.40 | 1797.34 | 1800.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 13:15:00 | 1797.40 | 1797.34 | 1800.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1797.40 | 1797.34 | 1800.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:00:00 | 1797.40 | 1797.34 | 1800.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1799.90 | 1797.85 | 1800.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 1798.70 | 1797.85 | 1800.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1799.90 | 1798.26 | 1800.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 1795.30 | 1798.26 | 1800.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:15:00 | 1793.80 | 1798.71 | 1800.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:15:00 | 1705.53 | 1739.42 | 1759.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:15:00 | 1704.11 | 1739.42 | 1759.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 1736.90 | 1728.42 | 1746.38 | SL hit (close>ema200) qty=0.50 sl=1728.42 alert=retest2 |

### Cycle 133 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 1763.40 | 1745.86 | 1743.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 12:15:00 | 1766.00 | 1755.96 | 1750.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1753.70 | 1755.51 | 1750.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 1753.70 | 1755.51 | 1750.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1754.10 | 1755.23 | 1750.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 1751.90 | 1755.23 | 1750.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1742.20 | 1752.62 | 1750.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1737.00 | 1752.62 | 1750.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1733.00 | 1748.70 | 1748.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 1733.00 | 1748.70 | 1748.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1727.00 | 1744.36 | 1746.58 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 14:15:00 | 1762.70 | 1747.02 | 1746.74 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 1727.90 | 1744.40 | 1746.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 1717.60 | 1739.04 | 1743.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1738.90 | 1727.73 | 1734.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 1738.90 | 1727.73 | 1734.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1738.90 | 1727.73 | 1734.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 1738.90 | 1727.73 | 1734.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1713.30 | 1724.85 | 1732.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:30:00 | 1656.90 | 1701.70 | 1721.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 09:15:00 | 1574.06 | 1662.99 | 1699.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 12:15:00 | 1558.10 | 1557.16 | 1583.61 | SL hit (close>ema200) qty=0.50 sl=1557.16 alert=retest2 |

### Cycle 137 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 1574.10 | 1554.69 | 1553.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 11:15:00 | 1600.00 | 1574.33 | 1564.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 1562.20 | 1591.39 | 1578.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 1562.20 | 1591.39 | 1578.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1562.20 | 1591.39 | 1578.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 1562.20 | 1591.39 | 1578.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 1536.50 | 1580.41 | 1574.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 1536.50 | 1580.41 | 1574.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1522.90 | 1568.91 | 1570.06 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1602.20 | 1571.86 | 1569.80 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 1564.30 | 1575.36 | 1576.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 1553.90 | 1569.32 | 1573.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 1557.50 | 1551.55 | 1559.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 12:00:00 | 1557.50 | 1551.55 | 1559.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1550.70 | 1551.38 | 1558.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 1552.70 | 1551.38 | 1558.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1557.00 | 1551.77 | 1557.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:45:00 | 1556.90 | 1551.77 | 1557.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1540.00 | 1549.41 | 1555.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 1557.80 | 1549.41 | 1555.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1559.00 | 1551.33 | 1555.96 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 1577.90 | 1559.47 | 1559.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1593.00 | 1573.20 | 1566.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 1673.20 | 1676.22 | 1645.49 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 15:00:00 | 1706.10 | 1685.93 | 1662.17 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1671.80 | 1686.15 | 1666.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1663.60 | 1686.15 | 1666.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1672.00 | 1683.32 | 1667.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 1652.00 | 1672.92 | 1666.96 | SL hit (close<ema400) qty=1.00 sl=1666.96 alert=retest1 |

### Cycle 142 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 1636.10 | 1661.80 | 1662.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 12:15:00 | 1621.20 | 1643.83 | 1653.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 10:15:00 | 1641.60 | 1632.98 | 1643.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:00:00 | 1641.60 | 1632.98 | 1643.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1639.90 | 1634.36 | 1642.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 1642.30 | 1634.36 | 1642.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1643.20 | 1636.13 | 1642.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 1642.00 | 1636.13 | 1642.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1649.40 | 1638.79 | 1643.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 1649.40 | 1638.79 | 1643.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1651.50 | 1641.33 | 1644.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 1646.30 | 1643.34 | 1644.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 1641.90 | 1643.53 | 1644.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 1649.30 | 1645.37 | 1645.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 1649.30 | 1645.37 | 1645.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 14:15:00 | 1659.00 | 1648.09 | 1646.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 1649.60 | 1651.76 | 1648.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 10:15:00 | 1649.60 | 1651.76 | 1648.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1649.60 | 1651.76 | 1648.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1649.60 | 1651.76 | 1648.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1648.70 | 1651.15 | 1648.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 1649.60 | 1651.15 | 1648.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1657.50 | 1652.42 | 1649.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 1670.40 | 1653.77 | 1651.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 11:00:00 | 1663.50 | 1670.61 | 1664.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 12:00:00 | 1680.10 | 1672.51 | 1666.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 1636.00 | 1662.30 | 1664.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 1636.00 | 1662.30 | 1664.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 1621.00 | 1648.86 | 1657.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 1628.00 | 1610.61 | 1627.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 14:15:00 | 1628.00 | 1610.61 | 1627.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 1628.00 | 1610.61 | 1627.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 1628.00 | 1610.61 | 1627.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1648.00 | 1618.08 | 1629.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1624.80 | 1618.08 | 1629.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 13:15:00 | 1657.30 | 1637.37 | 1635.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 1657.30 | 1637.37 | 1635.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 15:15:00 | 1661.60 | 1645.60 | 1639.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1626.30 | 1641.74 | 1638.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1626.30 | 1641.74 | 1638.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1626.30 | 1641.74 | 1638.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 1625.90 | 1641.74 | 1638.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 1596.00 | 1632.59 | 1634.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 1590.90 | 1624.25 | 1630.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 1482.40 | 1469.53 | 1497.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:45:00 | 1481.00 | 1469.53 | 1497.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1491.10 | 1475.41 | 1495.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:45:00 | 1484.10 | 1475.41 | 1495.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1471.80 | 1474.69 | 1493.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:00:00 | 1456.50 | 1471.07 | 1488.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1416.00 | 1470.85 | 1486.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 09:45:00 | 1454.20 | 1470.23 | 1477.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 1494.30 | 1480.91 | 1479.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 1494.30 | 1480.91 | 1479.17 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1452.80 | 1477.60 | 1478.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1416.90 | 1452.10 | 1463.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1407.10 | 1398.21 | 1417.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1407.10 | 1398.21 | 1417.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1407.10 | 1398.21 | 1417.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1417.00 | 1398.21 | 1417.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1410.00 | 1400.57 | 1416.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 1439.10 | 1400.57 | 1416.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1417.10 | 1403.87 | 1416.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 1424.20 | 1403.87 | 1416.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1421.10 | 1407.32 | 1417.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 1424.80 | 1407.32 | 1417.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1404.60 | 1406.78 | 1416.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1402.10 | 1406.78 | 1416.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:45:00 | 1402.00 | 1407.42 | 1415.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 1424.60 | 1410.86 | 1416.45 | SL hit (close>static) qty=1.00 sl=1421.50 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 1437.00 | 1420.09 | 1419.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1472.30 | 1430.53 | 1424.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1458.90 | 1473.19 | 1454.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1458.90 | 1473.19 | 1454.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1458.90 | 1473.19 | 1454.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1462.40 | 1473.19 | 1454.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1446.90 | 1467.93 | 1453.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1460.10 | 1453.59 | 1451.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 11:15:00 | 1440.50 | 1450.62 | 1450.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 1440.50 | 1450.62 | 1450.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 1434.90 | 1445.35 | 1448.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 1453.00 | 1441.70 | 1445.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 15:15:00 | 1453.00 | 1441.70 | 1445.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1453.00 | 1441.70 | 1445.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1393.20 | 1441.70 | 1445.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 09:45:00 | 1406.00 | 1396.08 | 1413.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:45:00 | 1410.60 | 1400.84 | 1408.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 1447.60 | 1417.42 | 1414.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1447.60 | 1417.42 | 1414.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 1450.00 | 1423.93 | 1417.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1419.20 | 1432.15 | 1424.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1419.20 | 1432.15 | 1424.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1419.20 | 1432.15 | 1424.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1419.20 | 1432.15 | 1424.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1411.50 | 1428.02 | 1423.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 1412.20 | 1428.02 | 1423.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1397.20 | 1416.38 | 1418.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1373.30 | 1404.31 | 1412.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1387.90 | 1373.56 | 1389.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1387.90 | 1373.56 | 1389.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1387.90 | 1373.56 | 1389.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:15:00 | 1384.30 | 1373.56 | 1389.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1389.40 | 1376.73 | 1389.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1337.50 | 1389.58 | 1392.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 15:15:00 | 1381.00 | 1371.52 | 1378.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:45:00 | 1369.50 | 1372.13 | 1378.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 12:00:00 | 1382.20 | 1376.46 | 1379.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 1392.30 | 1382.48 | 1381.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 1392.30 | 1382.48 | 1381.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1436.50 | 1400.31 | 1391.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1465.00 | 1465.41 | 1442.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 1460.20 | 1465.41 | 1442.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1492.60 | 1494.49 | 1475.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 1494.20 | 1494.43 | 1477.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 1498.30 | 1529.80 | 1533.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 09:15:00 | 1498.30 | 1529.80 | 1533.02 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 1527.00 | 1519.17 | 1518.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 1578.70 | 1531.07 | 1524.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 1555.60 | 1566.25 | 1550.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 1555.60 | 1566.25 | 1550.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1555.60 | 1566.25 | 1550.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 1538.90 | 1566.25 | 1550.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 1533.20 | 1559.64 | 1548.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 1533.20 | 1559.64 | 1548.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 1523.10 | 1552.33 | 1546.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:30:00 | 1516.80 | 1552.33 | 1546.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 1522.50 | 1538.52 | 1540.49 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1576.90 | 1540.54 | 1538.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1596.60 | 1557.26 | 1546.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 1593.20 | 1597.55 | 1585.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:15:00 | 1577.70 | 1597.55 | 1585.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1575.00 | 1593.04 | 1584.20 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 10:15:00 | 1573.50 | 1581.14 | 1581.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 13:15:00 | 1554.90 | 1573.45 | 1577.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1588.30 | 1567.63 | 1570.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 1588.30 | 1567.63 | 1570.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1588.30 | 1567.63 | 1570.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 1588.30 | 1567.63 | 1570.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1576.20 | 1569.35 | 1570.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:30:00 | 1592.90 | 1569.35 | 1570.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 1589.90 | 1573.46 | 1572.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 1596.00 | 1585.86 | 1579.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 1586.00 | 1590.30 | 1584.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 1586.00 | 1590.30 | 1584.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1586.00 | 1590.30 | 1584.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:30:00 | 1578.60 | 1590.30 | 1584.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1592.90 | 1590.82 | 1585.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:15:00 | 1600.70 | 1593.49 | 1588.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 15:15:00 | 1582.60 | 1590.52 | 1587.67 | SL hit (close<static) qty=1.00 sl=1583.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 12:00:00 | 1823.60 | 2024-05-27 13:15:00 | 1825.00 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2024-05-23 09:15:00 | 1830.25 | 2024-05-27 13:15:00 | 1825.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-05-27 11:15:00 | 1827.90 | 2024-05-27 13:15:00 | 1825.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-05-27 13:15:00 | 1825.15 | 2024-05-27 13:15:00 | 1825.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-06-27 09:15:00 | 1833.30 | 2024-06-27 09:15:00 | 1856.90 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-07-01 10:15:00 | 1862.05 | 2024-07-02 09:15:00 | 1841.75 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-07-01 11:00:00 | 1862.85 | 2024-07-02 09:15:00 | 1841.75 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-07-05 09:15:00 | 1879.95 | 2024-07-08 10:15:00 | 1860.20 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-07-05 10:00:00 | 1882.15 | 2024-07-08 10:15:00 | 1860.20 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-07-05 10:45:00 | 1880.90 | 2024-07-08 10:15:00 | 1860.20 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-07-05 11:15:00 | 1885.75 | 2024-07-08 10:15:00 | 1860.20 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-07-09 14:15:00 | 1850.00 | 2024-07-11 13:15:00 | 1863.95 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-07-10 14:15:00 | 1852.05 | 2024-07-11 13:15:00 | 1863.95 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-07-11 10:30:00 | 1850.00 | 2024-07-11 13:15:00 | 1863.95 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-07-11 11:00:00 | 1850.75 | 2024-07-11 13:15:00 | 1863.95 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-08-02 10:30:00 | 1964.65 | 2024-08-05 09:15:00 | 1901.00 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2024-08-08 09:15:00 | 1870.05 | 2024-08-09 09:15:00 | 1894.10 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-08-08 10:30:00 | 1872.80 | 2024-08-09 09:15:00 | 1894.10 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-08-08 13:00:00 | 1871.50 | 2024-08-09 09:15:00 | 1894.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-08-22 09:30:00 | 1894.85 | 2024-08-29 10:15:00 | 1918.10 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2024-08-22 10:00:00 | 1898.70 | 2024-08-29 10:15:00 | 1918.10 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2024-09-03 09:15:00 | 2017.00 | 2024-09-04 13:15:00 | 1950.50 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-09-12 09:15:00 | 1986.15 | 2024-09-17 15:15:00 | 2027.90 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2024-09-12 09:45:00 | 1985.10 | 2024-09-17 15:15:00 | 2027.90 | STOP_HIT | 1.00 | 2.16% |
| BUY | retest2 | 2024-09-30 11:15:00 | 2118.15 | 2024-10-04 12:15:00 | 2122.15 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2024-10-11 11:15:00 | 1978.55 | 2024-10-17 09:15:00 | 1879.95 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2024-10-11 12:00:00 | 1978.20 | 2024-10-17 10:15:00 | 1879.62 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2024-10-11 12:45:00 | 1978.90 | 2024-10-17 10:15:00 | 1879.29 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2024-10-11 11:15:00 | 1978.55 | 2024-10-18 13:15:00 | 1865.70 | STOP_HIT | 0.50 | 5.70% |
| SELL | retest2 | 2024-10-11 12:00:00 | 1978.20 | 2024-10-18 13:15:00 | 1865.70 | STOP_HIT | 0.50 | 5.69% |
| SELL | retest2 | 2024-10-11 12:45:00 | 1978.90 | 2024-10-18 13:15:00 | 1865.70 | STOP_HIT | 0.50 | 5.72% |
| SELL | retest2 | 2024-11-04 10:15:00 | 1755.10 | 2024-11-06 11:15:00 | 1782.35 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-11-18 09:15:00 | 1734.60 | 2024-11-19 12:15:00 | 1760.95 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-12-05 14:00:00 | 1809.50 | 2024-12-06 14:15:00 | 1786.90 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-12-10 12:15:00 | 1781.80 | 2024-12-11 09:15:00 | 1808.05 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-12-27 11:15:00 | 1717.85 | 2025-01-01 11:15:00 | 1722.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-12-30 11:00:00 | 1721.95 | 2025-01-01 11:15:00 | 1722.00 | STOP_HIT | 1.00 | -0.00% |
| SELL | retest2 | 2024-12-30 11:45:00 | 1722.85 | 2025-01-01 11:15:00 | 1722.00 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-01-02 11:45:00 | 1733.40 | 2025-01-06 11:15:00 | 1714.40 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-01-02 12:15:00 | 1734.75 | 2025-01-06 11:15:00 | 1714.40 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-01-14 12:15:00 | 1674.00 | 2025-01-15 12:15:00 | 1689.35 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-01-14 13:30:00 | 1671.15 | 2025-01-15 12:15:00 | 1689.35 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-01-14 15:00:00 | 1674.70 | 2025-01-15 12:15:00 | 1689.35 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-01-15 09:15:00 | 1675.10 | 2025-01-15 12:15:00 | 1689.35 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-01-20 11:15:00 | 1702.15 | 2025-01-22 09:15:00 | 1677.45 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-01-21 11:30:00 | 1712.90 | 2025-01-22 09:15:00 | 1677.45 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-01-23 10:45:00 | 1629.35 | 2025-01-28 09:15:00 | 1551.68 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2025-01-23 10:45:00 | 1629.35 | 2025-01-29 10:15:00 | 1573.10 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2025-01-24 14:00:00 | 1633.35 | 2025-01-31 09:15:00 | 1590.30 | STOP_HIT | 1.00 | 2.64% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1617.80 | 2025-02-03 09:15:00 | 1577.35 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-02-01 15:00:00 | 1619.55 | 2025-02-03 09:15:00 | 1577.35 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-02-13 13:00:00 | 1536.70 | 2025-02-19 14:15:00 | 1502.75 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2025-03-19 09:15:00 | 1570.65 | 2025-03-25 13:15:00 | 1588.35 | STOP_HIT | 1.00 | 1.13% |
| SELL | retest2 | 2025-04-01 11:45:00 | 1575.20 | 2025-04-02 13:15:00 | 1591.75 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-04-01 13:30:00 | 1575.65 | 2025-04-02 13:15:00 | 1591.75 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-04-01 14:15:00 | 1574.65 | 2025-04-02 13:15:00 | 1591.75 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-04-02 09:15:00 | 1570.40 | 2025-04-02 13:15:00 | 1591.75 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-05-05 11:15:00 | 1567.60 | 2025-05-05 11:15:00 | 1585.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-05-08 11:45:00 | 1537.50 | 2025-05-12 11:15:00 | 1555.90 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-05-08 12:30:00 | 1537.20 | 2025-05-12 11:15:00 | 1555.90 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-05-20 12:15:00 | 1634.00 | 2025-05-30 10:15:00 | 1671.20 | STOP_HIT | 1.00 | 2.28% |
| BUY | retest2 | 2025-05-20 15:00:00 | 1635.10 | 2025-05-30 10:15:00 | 1671.20 | STOP_HIT | 1.00 | 2.21% |
| SELL | retest2 | 2025-06-02 15:15:00 | 1660.10 | 2025-06-04 13:15:00 | 1682.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-06-03 09:45:00 | 1667.80 | 2025-06-04 13:15:00 | 1682.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-03 11:00:00 | 1666.00 | 2025-06-04 13:15:00 | 1682.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest1 | 2025-06-09 09:15:00 | 1728.90 | 2025-06-11 09:15:00 | 1718.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-06-11 11:30:00 | 1728.50 | 2025-06-11 13:15:00 | 1713.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-24 09:30:00 | 1637.90 | 2025-06-24 11:15:00 | 1669.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1684.40 | 2025-07-08 09:15:00 | 1752.00 | STOP_HIT | 1.00 | 4.01% |
| SELL | retest2 | 2025-07-25 09:15:00 | 1715.00 | 2025-07-28 09:15:00 | 1738.90 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-07-25 15:15:00 | 1720.00 | 2025-07-28 09:15:00 | 1738.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-07-28 10:30:00 | 1720.90 | 2025-07-29 09:15:00 | 1739.40 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-29 10:30:00 | 1725.10 | 2025-07-30 09:15:00 | 1748.90 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-31 12:15:00 | 1775.40 | 2025-07-31 14:15:00 | 1732.30 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1656.10 | 2025-08-14 10:15:00 | 1671.40 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-08-08 10:30:00 | 1655.00 | 2025-08-14 10:15:00 | 1671.40 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-08-08 12:45:00 | 1655.10 | 2025-08-14 10:15:00 | 1671.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-08 14:15:00 | 1655.00 | 2025-08-14 10:15:00 | 1671.40 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-08-11 09:15:00 | 1652.40 | 2025-08-14 10:15:00 | 1671.40 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-08-11 10:30:00 | 1654.60 | 2025-08-14 10:15:00 | 1671.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-08-11 13:00:00 | 1652.00 | 2025-08-14 10:15:00 | 1671.40 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-08-12 09:15:00 | 1653.80 | 2025-08-14 10:15:00 | 1671.40 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-08-12 11:15:00 | 1651.50 | 2025-08-14 10:15:00 | 1671.40 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest1 | 2025-08-25 09:15:00 | 1600.70 | 2025-08-29 10:15:00 | 1520.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-08-25 09:15:00 | 1600.70 | 2025-08-29 13:15:00 | 1549.10 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1558.00 | 2025-09-03 10:15:00 | 1561.50 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-08-28 11:00:00 | 1552.00 | 2025-09-03 10:15:00 | 1561.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-28 12:00:00 | 1555.60 | 2025-09-03 10:15:00 | 1561.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-08-28 15:15:00 | 1556.00 | 2025-09-03 10:15:00 | 1561.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-09-12 09:15:00 | 1589.70 | 2025-09-22 09:15:00 | 1679.90 | STOP_HIT | 1.00 | 5.67% |
| BUY | retest2 | 2025-09-12 10:15:00 | 1587.90 | 2025-09-22 09:15:00 | 1679.90 | STOP_HIT | 1.00 | 5.79% |
| SELL | retest2 | 2025-09-25 15:15:00 | 1645.80 | 2025-10-06 10:15:00 | 1633.60 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest1 | 2025-10-08 09:15:00 | 1695.80 | 2025-10-10 09:15:00 | 1780.59 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-08 09:15:00 | 1695.80 | 2025-10-10 10:15:00 | 1865.38 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-10-14 09:15:00 | 1868.40 | 2025-10-20 11:15:00 | 1873.30 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest1 | 2025-11-10 09:15:00 | 1843.50 | 2025-11-12 15:15:00 | 1854.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-11-11 09:45:00 | 1835.00 | 2025-11-13 11:15:00 | 1884.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-11-11 12:45:00 | 1833.00 | 2025-11-13 11:15:00 | 1884.00 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-11-12 09:30:00 | 1837.30 | 2025-11-13 11:15:00 | 1884.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-11-12 15:00:00 | 1834.10 | 2025-11-13 11:15:00 | 1884.00 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-11-17 09:15:00 | 1896.90 | 2025-11-19 10:15:00 | 1879.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-11-18 12:45:00 | 1888.80 | 2025-11-19 10:15:00 | 1879.20 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-19 09:15:00 | 1889.10 | 2025-11-19 10:15:00 | 1879.20 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-11-21 12:30:00 | 1907.50 | 2025-11-24 09:15:00 | 1897.10 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-11-21 15:00:00 | 1919.60 | 2025-11-24 09:15:00 | 1897.10 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-11-27 09:30:00 | 1827.10 | 2025-11-27 10:15:00 | 1867.30 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1822.80 | 2025-12-02 15:15:00 | 1846.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-11-28 13:45:00 | 1820.10 | 2025-12-02 15:15:00 | 1846.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-12-01 12:00:00 | 1822.30 | 2025-12-02 15:15:00 | 1846.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-12-08 13:00:00 | 1819.90 | 2025-12-16 09:15:00 | 1811.80 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-12-08 14:00:00 | 1814.90 | 2025-12-16 09:15:00 | 1811.80 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-12-09 13:45:00 | 1811.30 | 2025-12-16 09:15:00 | 1811.80 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-12-18 11:15:00 | 1810.00 | 2025-12-18 12:15:00 | 1800.90 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-18 12:00:00 | 1809.40 | 2025-12-18 12:15:00 | 1800.90 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-12-26 11:15:00 | 1779.90 | 2025-12-29 12:15:00 | 1797.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-30 10:15:00 | 1802.20 | 2026-01-02 12:15:00 | 1800.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2026-01-02 10:45:00 | 1804.80 | 2026-01-02 12:15:00 | 1800.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-01-06 09:15:00 | 1796.00 | 2026-01-06 10:15:00 | 1813.70 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1795.30 | 2026-01-12 10:15:00 | 1705.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:15:00 | 1793.80 | 2026-01-12 10:15:00 | 1704.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1795.30 | 2026-01-12 14:15:00 | 1736.90 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2026-01-08 10:15:00 | 1793.80 | 2026-01-12 14:15:00 | 1736.90 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2026-01-21 14:30:00 | 1656.90 | 2026-01-22 09:15:00 | 1574.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-21 14:30:00 | 1656.90 | 2026-01-27 12:15:00 | 1558.10 | STOP_HIT | 0.50 | 5.96% |
| BUY | retest1 | 2026-02-12 15:00:00 | 1706.10 | 2026-02-13 14:15:00 | 1652.00 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2026-02-18 09:15:00 | 1646.30 | 2026-02-18 13:15:00 | 1649.30 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1641.90 | 2026-02-18 13:15:00 | 1649.30 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-02-20 09:15:00 | 1670.40 | 2026-02-24 10:15:00 | 1636.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-02-23 11:00:00 | 1663.50 | 2026-02-24 10:15:00 | 1636.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-02-23 12:00:00 | 1680.10 | 2026-02-24 10:15:00 | 1636.00 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2026-02-26 09:15:00 | 1624.80 | 2026-02-26 13:15:00 | 1657.30 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-03-06 15:00:00 | 1456.50 | 2026-03-11 10:15:00 | 1494.30 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1416.00 | 2026-03-11 10:15:00 | 1494.30 | STOP_HIT | 1.00 | -5.53% |
| SELL | retest2 | 2026-03-10 09:45:00 | 1454.20 | 2026-03-11 10:15:00 | 1494.30 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1402.10 | 2026-03-17 13:15:00 | 1424.60 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-03-17 12:45:00 | 1402.00 | 2026-03-17 13:15:00 | 1424.60 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1460.10 | 2026-03-20 11:15:00 | 1440.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1393.20 | 2026-03-25 11:15:00 | 1447.60 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2026-03-24 09:45:00 | 1406.00 | 2026-03-25 11:15:00 | 1447.60 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-03-24 14:45:00 | 1410.60 | 2026-03-25 11:15:00 | 1447.60 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1337.50 | 2026-04-06 13:15:00 | 1392.30 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2026-04-02 15:15:00 | 1381.00 | 2026-04-06 13:15:00 | 1392.30 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-04-06 09:45:00 | 1369.50 | 2026-04-06 13:15:00 | 1392.30 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-04-06 12:00:00 | 1382.20 | 2026-04-06 13:15:00 | 1392.30 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-04-13 11:00:00 | 1494.20 | 2026-04-20 09:15:00 | 1498.30 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2026-05-08 14:15:00 | 1600.70 | 2026-05-08 15:15:00 | 1582.60 | STOP_HIT | 1.00 | -1.13% |
