# ACC Ltd. (ACC)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 1393.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 191 |
| ALERT1 | 137 |
| ALERT2 | 131 |
| ALERT2_SKIP | 59 |
| ALERT3 | 378 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 183 |
| PARTIAL | 19 |
| TARGET_HIT | 4 |
| STOP_HIT | 180 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 203 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 75 / 128
- **Target hits / Stop hits / Partials:** 4 / 180 / 19
- **Avg / median % per leg:** 0.49% / -0.29%
- **Sum % (uncompounded):** 100.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 19 | 25.0% | 2 | 74 | 0 | -0.23% | -17.7% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.06% | 0.1% |
| BUY @ 3rd Alert (retest2) | 75 | 18 | 24.0% | 2 | 73 | 0 | -0.24% | -17.7% |
| SELL (all) | 127 | 56 | 44.1% | 2 | 106 | 19 | 0.93% | 117.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 127 | 56 | 44.1% | 2 | 106 | 19 | 0.93% | 117.8% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.06% | 0.1% |
| retest2 (combined) | 202 | 74 | 36.6% | 4 | 179 | 19 | 0.50% | 100.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 14:15:00 | 1783.25 | 1793.30 | 1793.80 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 10:15:00 | 1800.45 | 1794.53 | 1794.15 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 1760.30 | 1787.68 | 1791.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 12:15:00 | 1754.75 | 1781.10 | 1787.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 13:15:00 | 1728.00 | 1725.33 | 1742.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-19 13:30:00 | 1726.65 | 1725.33 | 1742.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 1729.90 | 1726.24 | 1741.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 14:45:00 | 1734.35 | 1726.24 | 1741.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 1779.45 | 1737.33 | 1743.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:00:00 | 1779.45 | 1737.33 | 1743.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 1816.15 | 1753.09 | 1750.29 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 15:15:00 | 1783.00 | 1794.28 | 1794.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 09:15:00 | 1771.95 | 1789.82 | 1792.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 14:15:00 | 1776.05 | 1774.56 | 1782.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-25 15:15:00 | 1783.00 | 1774.56 | 1782.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 1783.00 | 1776.25 | 1782.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:15:00 | 1780.00 | 1776.25 | 1782.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 1784.55 | 1777.91 | 1782.84 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 09:15:00 | 1792.60 | 1784.46 | 1783.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 11:15:00 | 1805.75 | 1790.97 | 1787.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 14:15:00 | 1793.85 | 1794.17 | 1789.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 15:15:00 | 1794.00 | 1794.17 | 1789.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 1811.95 | 1797.70 | 1792.17 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 14:15:00 | 1775.35 | 1794.64 | 1795.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 15:15:00 | 1772.00 | 1790.12 | 1793.01 | Break + close below crossover candle low |

### Cycle 8 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 1820.80 | 1796.25 | 1795.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 09:15:00 | 1853.15 | 1821.56 | 1813.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 10:15:00 | 1843.80 | 1844.89 | 1833.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-07 11:00:00 | 1843.80 | 1844.89 | 1833.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 1832.95 | 1848.79 | 1842.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 1832.95 | 1848.79 | 1842.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 1823.35 | 1843.70 | 1840.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:00:00 | 1823.35 | 1843.70 | 1840.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 1834.00 | 1839.45 | 1839.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:15:00 | 1830.05 | 1839.45 | 1839.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 1838.45 | 1839.25 | 1839.08 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 10:15:00 | 1831.80 | 1837.76 | 1838.42 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 12:15:00 | 1842.60 | 1839.57 | 1839.18 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 15:15:00 | 1833.05 | 1837.91 | 1838.49 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 10:15:00 | 1845.00 | 1840.05 | 1839.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 10:15:00 | 1852.00 | 1843.38 | 1841.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 11:15:00 | 1850.00 | 1850.13 | 1846.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-14 12:00:00 | 1850.00 | 1850.13 | 1846.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 13:15:00 | 1851.75 | 1850.52 | 1847.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 13:45:00 | 1850.00 | 1850.52 | 1847.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 13:15:00 | 1862.00 | 1857.95 | 1853.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 13:30:00 | 1860.00 | 1857.95 | 1853.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 1850.35 | 1856.43 | 1853.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:45:00 | 1851.30 | 1856.43 | 1853.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 1847.50 | 1854.64 | 1852.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 09:15:00 | 1860.50 | 1854.64 | 1852.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-19 10:15:00 | 1841.95 | 1850.69 | 1851.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 10:15:00 | 1841.95 | 1850.69 | 1851.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 11:15:00 | 1832.00 | 1846.95 | 1850.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 11:15:00 | 1836.50 | 1832.86 | 1839.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 11:15:00 | 1836.50 | 1832.86 | 1839.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 1836.50 | 1832.86 | 1839.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 11:45:00 | 1836.90 | 1832.86 | 1839.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 12:15:00 | 1838.25 | 1833.94 | 1839.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 13:00:00 | 1838.25 | 1833.94 | 1839.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 13:15:00 | 1837.10 | 1834.57 | 1839.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 13:45:00 | 1838.00 | 1834.57 | 1839.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 1839.95 | 1836.04 | 1838.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 1844.10 | 1836.04 | 1838.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 1843.00 | 1837.44 | 1839.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:30:00 | 1844.50 | 1837.44 | 1839.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 1834.40 | 1836.83 | 1838.88 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 13:15:00 | 1849.30 | 1841.60 | 1840.69 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 1832.35 | 1840.36 | 1840.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 1803.60 | 1831.73 | 1836.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 1785.70 | 1783.95 | 1799.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 14:00:00 | 1785.70 | 1783.95 | 1799.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 1788.30 | 1788.16 | 1797.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 13:30:00 | 1779.70 | 1785.91 | 1793.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 09:15:00 | 1816.90 | 1793.00 | 1795.17 | SL hit (close>static) qty=1.00 sl=1802.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 1815.50 | 1797.50 | 1797.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 15:15:00 | 1826.70 | 1820.74 | 1814.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 10:15:00 | 1820.00 | 1821.90 | 1816.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 10:30:00 | 1818.20 | 1821.90 | 1816.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 1810.40 | 1819.44 | 1817.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 10:00:00 | 1810.40 | 1819.44 | 1817.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 1811.00 | 1817.75 | 1816.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 13:00:00 | 1815.00 | 1816.45 | 1816.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 13:30:00 | 1818.15 | 1817.35 | 1816.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 12:45:00 | 1815.75 | 1817.36 | 1817.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-06 13:15:00 | 1815.90 | 1817.07 | 1817.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-07-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 13:15:00 | 1815.90 | 1817.07 | 1817.17 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 14:15:00 | 1820.45 | 1817.74 | 1817.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 15:15:00 | 1822.50 | 1818.69 | 1817.92 | Break + close above crossover candle high |

### Cycle 19 — SELL (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 09:15:00 | 1810.50 | 1817.06 | 1817.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 10:15:00 | 1802.45 | 1814.13 | 1815.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 10:15:00 | 1799.95 | 1797.10 | 1804.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 10:15:00 | 1799.95 | 1797.10 | 1804.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 1799.95 | 1797.10 | 1804.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 11:00:00 | 1799.95 | 1797.10 | 1804.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 1795.25 | 1795.30 | 1801.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 14:30:00 | 1796.20 | 1795.30 | 1801.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 15:15:00 | 1790.00 | 1794.24 | 1800.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 12:30:00 | 1783.05 | 1791.03 | 1796.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 13:30:00 | 1785.05 | 1785.23 | 1789.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 10:45:00 | 1785.85 | 1788.55 | 1790.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 13:15:00 | 1785.25 | 1789.04 | 1790.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 1800.30 | 1782.61 | 1783.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:30:00 | 1804.40 | 1782.61 | 1783.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-17 10:15:00 | 1811.00 | 1788.29 | 1785.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 1811.00 | 1788.29 | 1785.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 13:15:00 | 1817.95 | 1800.87 | 1792.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 09:15:00 | 1803.05 | 1804.67 | 1796.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 1803.05 | 1804.67 | 1796.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 1803.05 | 1804.67 | 1796.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:45:00 | 1801.35 | 1804.67 | 1796.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 1793.60 | 1802.06 | 1796.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 11:30:00 | 1789.95 | 1802.06 | 1796.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 1795.80 | 1800.81 | 1796.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 09:15:00 | 1802.80 | 1798.04 | 1796.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 09:15:00 | 1809.00 | 1799.42 | 1797.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 09:15:00 | 1792.00 | 1797.94 | 1797.39 | SL hit (close<static) qty=1.00 sl=1792.65 alert=retest2 |

### Cycle 21 — SELL (started 2023-07-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 10:15:00 | 1793.20 | 1796.99 | 1797.01 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 11:15:00 | 1800.35 | 1797.66 | 1797.31 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 09:15:00 | 1791.00 | 1797.40 | 1797.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 11:15:00 | 1783.50 | 1793.07 | 1795.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 1806.50 | 1786.48 | 1790.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 1806.50 | 1786.48 | 1790.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1806.50 | 1786.48 | 1790.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 10:00:00 | 1806.50 | 1786.48 | 1790.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 1811.45 | 1791.47 | 1792.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 11:00:00 | 1811.45 | 1791.47 | 1792.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 11:15:00 | 1803.15 | 1793.81 | 1793.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 09:15:00 | 1870.00 | 1814.66 | 1803.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 09:15:00 | 1901.75 | 1905.09 | 1877.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-27 09:30:00 | 1902.65 | 1905.09 | 1877.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 1989.80 | 2002.49 | 1988.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 1989.80 | 2002.49 | 1988.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 1981.15 | 1998.22 | 1987.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 1981.05 | 1998.22 | 1987.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 1980.50 | 1994.68 | 1986.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 12:30:00 | 1984.55 | 1994.68 | 1986.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 15:15:00 | 1968.30 | 1982.01 | 1982.42 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 09:15:00 | 1999.00 | 1985.41 | 1983.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 10:15:00 | 2043.90 | 1997.11 | 1989.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 09:15:00 | 2025.70 | 2033.17 | 2021.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 09:15:00 | 2025.70 | 2033.17 | 2021.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 2025.70 | 2033.17 | 2021.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 10:00:00 | 2025.70 | 2033.17 | 2021.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 2021.70 | 2030.88 | 2021.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:00:00 | 2021.70 | 2030.88 | 2021.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 2015.60 | 2027.82 | 2020.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:30:00 | 2013.90 | 2027.82 | 2020.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 12:15:00 | 2015.75 | 2025.41 | 2020.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 13:15:00 | 2026.10 | 2025.41 | 2020.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 2030.35 | 2026.04 | 2021.40 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 1986.20 | 2015.07 | 2017.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 11:15:00 | 1981.65 | 1993.92 | 2003.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 13:15:00 | 1900.10 | 1892.05 | 1906.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-17 14:00:00 | 1900.10 | 1892.05 | 1906.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 1900.00 | 1893.64 | 1905.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 14:30:00 | 1906.25 | 1893.64 | 1905.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 1888.00 | 1893.18 | 1903.60 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 13:15:00 | 1940.70 | 1913.91 | 1910.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 09:15:00 | 1949.70 | 1927.07 | 1918.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 09:15:00 | 1971.35 | 1972.60 | 1958.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 09:30:00 | 1971.30 | 1972.60 | 1958.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 1965.25 | 1972.58 | 1964.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:00:00 | 1965.25 | 1972.58 | 1964.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 1970.85 | 1972.23 | 1964.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 09:15:00 | 1985.50 | 1972.23 | 1964.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 1997.70 | 1977.32 | 1967.66 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 12:15:00 | 1964.25 | 1971.82 | 1972.63 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 12:15:00 | 1982.30 | 1971.04 | 1970.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 13:15:00 | 1994.25 | 1975.68 | 1973.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 14:15:00 | 1982.45 | 1984.89 | 1980.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 14:15:00 | 1982.45 | 1984.89 | 1980.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 14:15:00 | 1982.45 | 1984.89 | 1980.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 11:00:00 | 1991.00 | 1983.24 | 1980.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 09:15:00 | 1944.25 | 1984.58 | 1984.05 | SL hit (close<static) qty=1.00 sl=1979.80 alert=retest2 |

### Cycle 31 — SELL (started 2023-08-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 10:15:00 | 1959.15 | 1979.49 | 1981.79 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 14:15:00 | 2019.20 | 1980.91 | 1980.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 10:15:00 | 2037.55 | 2014.27 | 2001.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 09:15:00 | 2059.90 | 2070.33 | 2051.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-06 09:45:00 | 2062.35 | 2070.33 | 2051.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 2051.30 | 2065.33 | 2052.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:45:00 | 2044.55 | 2065.33 | 2052.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 2041.50 | 2060.56 | 2051.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:00:00 | 2041.50 | 2060.56 | 2051.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 2042.30 | 2056.91 | 2050.35 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 09:15:00 | 2028.15 | 2046.96 | 2047.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 12:15:00 | 2020.00 | 2033.61 | 2038.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 09:15:00 | 2045.10 | 2031.51 | 2035.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 2045.10 | 2031.51 | 2035.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 2045.10 | 2031.51 | 2035.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 10:00:00 | 2045.10 | 2031.51 | 2035.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 10:15:00 | 2051.00 | 2035.41 | 2036.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 10:45:00 | 2048.00 | 2035.41 | 2036.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 11:15:00 | 2062.45 | 2040.82 | 2039.29 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 2020.20 | 2039.02 | 2040.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 2007.70 | 2029.72 | 2035.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 2027.15 | 2025.06 | 2031.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 11:00:00 | 2027.15 | 2025.06 | 2031.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 2046.25 | 2029.30 | 2033.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:00:00 | 2046.25 | 2029.30 | 2033.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 2044.70 | 2032.38 | 2034.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:30:00 | 2043.90 | 2032.38 | 2034.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 13:15:00 | 2050.00 | 2035.90 | 2035.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-13 14:15:00 | 2061.35 | 2040.99 | 2037.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-14 11:15:00 | 2040.05 | 2047.11 | 2042.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 11:15:00 | 2040.05 | 2047.11 | 2042.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 2040.05 | 2047.11 | 2042.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 11:30:00 | 2039.90 | 2047.11 | 2042.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 2038.25 | 2045.34 | 2042.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 12:30:00 | 2040.00 | 2045.34 | 2042.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 13:15:00 | 2038.30 | 2043.93 | 2041.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 14:00:00 | 2038.30 | 2043.93 | 2041.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 15:15:00 | 2042.00 | 2042.58 | 2041.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:15:00 | 2041.85 | 2042.58 | 2041.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 2048.00 | 2043.66 | 2042.07 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 13:15:00 | 2037.05 | 2040.83 | 2041.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-15 14:15:00 | 2023.40 | 2037.35 | 2039.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 11:15:00 | 1995.05 | 1986.28 | 1998.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 11:15:00 | 1995.05 | 1986.28 | 1998.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 1995.05 | 1986.28 | 1998.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:45:00 | 1998.90 | 1986.28 | 1998.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 2000.35 | 1989.09 | 1998.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 13:00:00 | 2000.35 | 1989.09 | 1998.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 1996.15 | 1990.50 | 1998.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 13:45:00 | 1999.50 | 1990.50 | 1998.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 1994.90 | 1991.38 | 1998.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:30:00 | 1997.10 | 1991.38 | 1998.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 1974.05 | 1987.70 | 1995.45 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 14:15:00 | 2000.00 | 1991.43 | 1990.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 09:15:00 | 2020.70 | 1998.81 | 1994.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 15:15:00 | 2015.00 | 2018.08 | 2011.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 09:15:00 | 2018.10 | 2018.08 | 2011.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 2022.50 | 2018.96 | 2012.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:45:00 | 2019.00 | 2018.96 | 2012.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 2014.35 | 2018.04 | 2012.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:00:00 | 2014.35 | 2018.04 | 2012.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 2007.35 | 2015.90 | 2012.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 12:00:00 | 2007.35 | 2015.90 | 2012.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 2005.45 | 2013.81 | 2011.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 12:30:00 | 2003.20 | 2013.81 | 2011.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 1995.50 | 2008.52 | 2009.57 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 13:15:00 | 2017.75 | 2011.35 | 2010.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 10:15:00 | 2035.95 | 2017.79 | 2013.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 2003.65 | 2023.88 | 2019.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 2003.65 | 2023.88 | 2019.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 2003.65 | 2023.88 | 2019.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 10:00:00 | 2003.65 | 2023.88 | 2019.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 2030.20 | 2025.14 | 2020.88 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 1992.60 | 2015.51 | 2017.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 1984.20 | 2003.87 | 2007.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 1979.40 | 1978.69 | 1990.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 11:15:00 | 1986.00 | 1980.71 | 1989.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 1986.00 | 1980.71 | 1989.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:00:00 | 1986.00 | 1980.71 | 1989.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 1984.60 | 1981.49 | 1988.77 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 2028.75 | 1992.99 | 1992.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 10:15:00 | 2047.00 | 2003.79 | 1997.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 10:15:00 | 2030.60 | 2031.14 | 2017.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 10:30:00 | 2033.25 | 2031.14 | 2017.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 15:15:00 | 2024.05 | 2030.18 | 2022.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 09:15:00 | 2028.40 | 2030.18 | 2022.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 2018.80 | 2027.90 | 2022.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 09:45:00 | 2020.25 | 2027.90 | 2022.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 2016.70 | 2025.66 | 2021.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:45:00 | 2011.95 | 2025.66 | 2021.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 2022.00 | 2022.95 | 2021.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 13:45:00 | 2022.00 | 2022.95 | 2021.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 14:15:00 | 2008.50 | 2020.06 | 2020.09 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 10:15:00 | 2030.90 | 2021.57 | 2020.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 11:15:00 | 2034.15 | 2028.52 | 2025.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 2022.70 | 2031.83 | 2029.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 2022.70 | 2031.83 | 2029.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 2022.70 | 2031.83 | 2029.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 2022.70 | 2031.83 | 2029.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 2019.60 | 2029.38 | 2028.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:45:00 | 2013.50 | 2029.38 | 2028.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 2018.55 | 2027.22 | 2027.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 2009.90 | 2023.56 | 2025.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 14:15:00 | 2030.40 | 2016.88 | 2019.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 14:15:00 | 2030.40 | 2016.88 | 2019.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 2030.40 | 2016.88 | 2019.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:45:00 | 2037.25 | 2016.88 | 2019.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 2021.50 | 2017.81 | 2019.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:15:00 | 2020.75 | 2017.81 | 2019.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 12:15:00 | 1919.71 | 1956.16 | 1979.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 13:15:00 | 1898.20 | 1892.16 | 1911.45 | SL hit (close>ema200) qty=0.50 sl=1892.16 alert=retest2 |

### Cycle 46 — BUY (started 2023-11-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 10:15:00 | 1856.70 | 1839.96 | 1838.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 09:15:00 | 1870.55 | 1849.53 | 1845.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 12:15:00 | 1852.35 | 1853.03 | 1848.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 13:00:00 | 1852.35 | 1853.03 | 1848.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 15:15:00 | 1850.00 | 1852.65 | 1849.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:45:00 | 1844.50 | 1851.90 | 1849.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 1846.55 | 1850.83 | 1848.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:00:00 | 1846.55 | 1850.83 | 1848.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 1842.65 | 1849.19 | 1848.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 12:00:00 | 1842.65 | 1849.19 | 1848.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2023-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 12:15:00 | 1840.00 | 1847.35 | 1847.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 13:15:00 | 1835.95 | 1845.07 | 1846.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 10:15:00 | 1844.75 | 1840.23 | 1843.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 10:15:00 | 1844.75 | 1840.23 | 1843.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 1844.75 | 1840.23 | 1843.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 11:00:00 | 1844.75 | 1840.23 | 1843.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 11:15:00 | 1845.30 | 1841.25 | 1843.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 11:45:00 | 1846.30 | 1841.25 | 1843.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 12:15:00 | 1838.35 | 1840.67 | 1842.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 13:00:00 | 1838.35 | 1840.67 | 1842.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 1831.40 | 1838.58 | 1841.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 10:15:00 | 1827.60 | 1838.58 | 1841.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 09:15:00 | 1866.05 | 1830.59 | 1827.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 1866.05 | 1830.59 | 1827.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 10:15:00 | 1874.00 | 1839.27 | 1831.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 11:15:00 | 1864.10 | 1865.86 | 1853.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 11:30:00 | 1863.80 | 1865.86 | 1853.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 1853.60 | 1865.85 | 1858.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 1856.95 | 1865.85 | 1858.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 1868.50 | 1866.38 | 1859.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 14:00:00 | 1871.25 | 1865.98 | 1860.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 15:00:00 | 1876.35 | 1868.05 | 1862.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-05 09:15:00 | 2058.38 | 2011.34 | 1959.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2023-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 12:15:00 | 2207.55 | 2215.18 | 2215.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 14:15:00 | 2201.00 | 2211.35 | 2213.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 10:15:00 | 2114.50 | 2111.70 | 2135.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 11:00:00 | 2114.50 | 2111.70 | 2135.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 10:15:00 | 2109.20 | 2104.37 | 2118.99 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 10:15:00 | 2150.20 | 2127.53 | 2124.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 13:15:00 | 2163.50 | 2142.86 | 2133.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 09:15:00 | 2223.55 | 2228.27 | 2206.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 10:00:00 | 2223.55 | 2228.27 | 2206.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 2352.25 | 2364.32 | 2336.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 09:45:00 | 2348.25 | 2364.32 | 2336.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 12:15:00 | 2351.90 | 2363.51 | 2352.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 13:00:00 | 2351.90 | 2363.51 | 2352.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 13:15:00 | 2348.00 | 2360.41 | 2352.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:00:00 | 2348.00 | 2360.41 | 2352.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 2314.80 | 2351.29 | 2348.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 15:00:00 | 2314.80 | 2351.29 | 2348.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 15:15:00 | 2302.00 | 2341.43 | 2344.66 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 10:15:00 | 2332.55 | 2325.90 | 2325.63 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 14:15:00 | 2317.65 | 2327.89 | 2328.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 09:15:00 | 2315.20 | 2324.39 | 2326.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 14:15:00 | 2262.00 | 2255.56 | 2272.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 15:00:00 | 2262.00 | 2255.56 | 2272.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 2265.70 | 2259.26 | 2271.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 10:30:00 | 2258.40 | 2258.31 | 2269.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 09:45:00 | 2253.75 | 2260.37 | 2265.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 12:30:00 | 2260.55 | 2262.05 | 2264.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 14:15:00 | 2285.15 | 2268.18 | 2267.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 14:15:00 | 2285.15 | 2268.18 | 2267.33 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 2237.05 | 2264.56 | 2266.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 13:15:00 | 2230.05 | 2257.66 | 2263.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 2226.00 | 2224.33 | 2239.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 14:00:00 | 2226.00 | 2224.33 | 2239.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 2240.50 | 2227.57 | 2239.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 2240.50 | 2227.57 | 2239.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 2232.50 | 2228.55 | 2239.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 2280.00 | 2228.55 | 2239.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 2282.50 | 2239.34 | 2243.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:30:00 | 2308.40 | 2239.34 | 2243.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 10:15:00 | 2275.95 | 2246.66 | 2246.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 13:15:00 | 2350.00 | 2274.72 | 2259.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 2505.80 | 2515.53 | 2457.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 14:30:00 | 2522.75 | 2515.53 | 2457.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 2538.05 | 2534.74 | 2490.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 12:45:00 | 2496.15 | 2534.74 | 2490.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 2533.55 | 2540.65 | 2513.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 2533.55 | 2540.65 | 2513.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 2533.45 | 2534.06 | 2521.82 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 14:15:00 | 2498.00 | 2516.35 | 2516.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 2482.15 | 2506.89 | 2512.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 10:15:00 | 2507.10 | 2506.93 | 2511.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 10:15:00 | 2507.10 | 2506.93 | 2511.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 2507.10 | 2506.93 | 2511.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 11:00:00 | 2507.10 | 2506.93 | 2511.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 12:15:00 | 2509.45 | 2505.67 | 2510.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 13:00:00 | 2509.45 | 2505.67 | 2510.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 13:15:00 | 2516.90 | 2507.92 | 2510.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:00:00 | 2516.90 | 2507.92 | 2510.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 2490.40 | 2504.41 | 2508.89 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 14:15:00 | 2527.05 | 2510.77 | 2509.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 09:15:00 | 2548.00 | 2520.81 | 2514.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 10:15:00 | 2518.70 | 2520.39 | 2514.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 10:15:00 | 2518.70 | 2520.39 | 2514.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 2518.70 | 2520.39 | 2514.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 10:30:00 | 2514.00 | 2520.39 | 2514.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 2507.75 | 2517.86 | 2513.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 11:30:00 | 2506.60 | 2517.86 | 2513.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 2503.60 | 2515.01 | 2512.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 12:45:00 | 2509.25 | 2515.01 | 2512.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 13:15:00 | 2504.05 | 2512.82 | 2512.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 14:15:00 | 2509.90 | 2512.82 | 2512.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 14:15:00 | 2506.35 | 2511.52 | 2511.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 14:15:00 | 2506.35 | 2511.52 | 2511.60 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 15:15:00 | 2514.85 | 2512.19 | 2511.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 09:15:00 | 2526.30 | 2515.01 | 2513.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 10:15:00 | 2513.25 | 2514.66 | 2513.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 10:15:00 | 2513.25 | 2514.66 | 2513.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 2513.25 | 2514.66 | 2513.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 2513.25 | 2514.66 | 2513.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 2526.20 | 2516.97 | 2514.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 13:00:00 | 2532.00 | 2519.97 | 2515.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 13:15:00 | 2509.05 | 2517.79 | 2515.36 | SL hit (close<static) qty=1.00 sl=2509.85 alert=retest2 |

### Cycle 61 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 2660.00 | 2678.92 | 2680.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 2654.75 | 2674.09 | 2678.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 13:15:00 | 2661.05 | 2660.95 | 2669.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 14:00:00 | 2661.05 | 2660.95 | 2669.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 2678.95 | 2664.55 | 2670.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 2678.95 | 2664.55 | 2670.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 2680.00 | 2667.64 | 2671.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 2686.10 | 2667.64 | 2671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 10:15:00 | 2680.50 | 2672.64 | 2673.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:00:00 | 2680.50 | 2672.64 | 2673.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 11:15:00 | 2684.30 | 2674.98 | 2674.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 12:15:00 | 2702.20 | 2680.42 | 2676.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 14:15:00 | 2680.00 | 2680.99 | 2677.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 14:15:00 | 2680.00 | 2680.99 | 2677.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 2680.00 | 2680.99 | 2677.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 15:00:00 | 2680.00 | 2680.99 | 2677.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 2675.25 | 2679.84 | 2677.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:15:00 | 2698.60 | 2679.84 | 2677.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 2699.90 | 2683.85 | 2679.58 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 11:15:00 | 2672.45 | 2679.98 | 2680.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 2665.10 | 2675.89 | 2678.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 2619.00 | 2605.05 | 2630.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 11:00:00 | 2619.00 | 2605.05 | 2630.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 2614.80 | 2611.01 | 2627.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:30:00 | 2619.55 | 2611.01 | 2627.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 2628.00 | 2614.41 | 2627.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 14:45:00 | 2618.70 | 2614.41 | 2627.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 2662.00 | 2623.93 | 2630.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 2650.60 | 2623.93 | 2630.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 2644.60 | 2628.06 | 2631.83 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 2649.65 | 2636.71 | 2635.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 12:15:00 | 2658.50 | 2641.06 | 2637.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 12:15:00 | 2694.80 | 2695.65 | 2677.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 13:00:00 | 2694.80 | 2695.65 | 2677.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 2683.00 | 2694.77 | 2682.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:00:00 | 2683.00 | 2694.77 | 2682.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 2680.05 | 2691.83 | 2682.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 11:15:00 | 2675.25 | 2691.83 | 2682.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 2672.75 | 2688.01 | 2681.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:15:00 | 2666.00 | 2688.01 | 2681.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 2663.95 | 2679.95 | 2678.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:00:00 | 2663.95 | 2679.95 | 2678.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2024-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 14:15:00 | 2666.60 | 2677.28 | 2677.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 2633.45 | 2666.55 | 2672.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 2639.50 | 2637.93 | 2653.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 15:00:00 | 2639.50 | 2637.93 | 2653.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 2628.20 | 2638.23 | 2651.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 10:30:00 | 2622.65 | 2635.79 | 2648.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 12:00:00 | 2623.45 | 2633.32 | 2646.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-11 09:15:00 | 2667.70 | 2643.30 | 2646.69 | SL hit (close>static) qty=1.00 sl=2656.85 alert=retest2 |

### Cycle 66 — BUY (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 13:15:00 | 2440.25 | 2425.02 | 2423.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 13:15:00 | 2454.35 | 2437.16 | 2431.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 14:15:00 | 2449.95 | 2450.27 | 2442.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 14:45:00 | 2449.85 | 2450.27 | 2442.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 2452.30 | 2471.47 | 2459.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 2452.30 | 2471.47 | 2459.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 2458.60 | 2468.89 | 2459.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 2469.95 | 2468.89 | 2459.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-05 12:15:00 | 2577.55 | 2596.64 | 2596.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-04-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 12:15:00 | 2577.55 | 2596.64 | 2596.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 09:15:00 | 2552.90 | 2578.88 | 2585.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 15:15:00 | 2454.75 | 2449.34 | 2466.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 09:15:00 | 2462.15 | 2449.34 | 2466.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 2462.30 | 2451.93 | 2465.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:45:00 | 2446.25 | 2451.07 | 2464.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:30:00 | 2446.50 | 2451.19 | 2461.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 12:30:00 | 2441.35 | 2419.62 | 2420.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 13:15:00 | 2456.15 | 2426.93 | 2423.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 13:15:00 | 2456.15 | 2426.93 | 2423.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 2506.80 | 2450.74 | 2435.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 11:15:00 | 2526.25 | 2532.45 | 2499.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 11:45:00 | 2538.90 | 2532.45 | 2499.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 12:15:00 | 2524.85 | 2540.17 | 2523.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:00:00 | 2524.85 | 2540.17 | 2523.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 2529.65 | 2538.06 | 2524.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:30:00 | 2523.80 | 2538.06 | 2524.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 2533.25 | 2537.10 | 2525.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 14:30:00 | 2530.00 | 2537.10 | 2525.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 2542.40 | 2537.46 | 2527.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 12:30:00 | 2558.80 | 2540.58 | 2531.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 13:15:00 | 2528.10 | 2531.63 | 2532.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 13:15:00 | 2528.10 | 2531.63 | 2532.07 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 09:15:00 | 2542.35 | 2533.15 | 2532.59 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 2515.70 | 2529.74 | 2531.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 12:15:00 | 2495.15 | 2517.64 | 2524.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 09:15:00 | 2510.25 | 2505.02 | 2515.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 09:15:00 | 2510.25 | 2505.02 | 2515.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 2510.25 | 2505.02 | 2515.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:45:00 | 2513.55 | 2505.02 | 2515.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 2458.30 | 2495.68 | 2509.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 11:15:00 | 2449.90 | 2495.68 | 2509.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 12:15:00 | 2452.75 | 2401.09 | 2399.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 2452.75 | 2401.09 | 2399.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 2461.00 | 2413.08 | 2404.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 2466.90 | 2473.00 | 2457.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:30:00 | 2467.10 | 2473.00 | 2457.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 2506.90 | 2512.92 | 2496.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 2492.40 | 2512.92 | 2496.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 2514.50 | 2523.10 | 2511.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 2520.05 | 2523.10 | 2511.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 2508.65 | 2520.21 | 2511.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 2508.65 | 2520.21 | 2511.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 2532.15 | 2522.60 | 2513.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:15:00 | 2537.00 | 2522.60 | 2513.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:15:00 | 2537.30 | 2523.68 | 2514.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 11:00:00 | 2544.95 | 2535.69 | 2524.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 2573.45 | 2582.38 | 2583.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 2573.45 | 2582.38 | 2583.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 2535.00 | 2569.37 | 2576.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 2515.00 | 2513.19 | 2531.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 2515.00 | 2513.19 | 2531.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 2515.00 | 2513.19 | 2531.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 2536.05 | 2513.19 | 2531.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 2526.50 | 2514.81 | 2527.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:45:00 | 2532.75 | 2514.81 | 2527.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 2562.00 | 2524.25 | 2530.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 2562.00 | 2524.25 | 2530.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 2548.70 | 2529.14 | 2532.37 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 2633.00 | 2553.64 | 2543.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 15:15:00 | 2701.75 | 2644.23 | 2599.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 2553.65 | 2626.12 | 2595.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 2553.65 | 2626.12 | 2595.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 2553.65 | 2626.12 | 2595.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 2514.85 | 2626.12 | 2595.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 2414.55 | 2583.80 | 2579.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 2414.55 | 2583.80 | 2579.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 2280.40 | 2523.12 | 2552.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 15:15:00 | 2280.00 | 2393.31 | 2474.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 2385.85 | 2380.39 | 2453.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 2385.85 | 2380.39 | 2453.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 2483.00 | 2410.68 | 2436.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 2476.40 | 2410.68 | 2436.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 2493.90 | 2427.32 | 2441.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 2493.90 | 2427.32 | 2441.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 2461.00 | 2450.84 | 2450.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 2473.55 | 2455.39 | 2452.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 11:15:00 | 2655.80 | 2657.21 | 2636.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 11:30:00 | 2657.10 | 2657.21 | 2636.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 2611.05 | 2645.89 | 2639.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 2615.00 | 2645.89 | 2639.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 2621.40 | 2640.99 | 2637.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:15:00 | 2622.75 | 2640.99 | 2637.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 2629.35 | 2635.61 | 2635.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 2629.35 | 2635.61 | 2635.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 2618.55 | 2632.19 | 2634.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 13:15:00 | 2629.00 | 2627.28 | 2630.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 14:00:00 | 2629.00 | 2627.28 | 2630.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 2621.65 | 2625.20 | 2628.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 13:00:00 | 2603.35 | 2622.34 | 2626.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 2600.50 | 2619.82 | 2625.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 2587.85 | 2613.43 | 2621.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 2633.00 | 2595.97 | 2593.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 2633.00 | 2595.97 | 2593.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 2656.45 | 2609.65 | 2602.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 2603.80 | 2617.00 | 2609.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 2603.80 | 2617.00 | 2609.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 2603.80 | 2617.00 | 2609.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 2603.80 | 2617.00 | 2609.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 2640.25 | 2621.65 | 2611.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 2618.00 | 2621.65 | 2611.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 2619.75 | 2628.49 | 2619.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 2619.75 | 2628.49 | 2619.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 2619.80 | 2626.75 | 2619.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:45:00 | 2616.50 | 2626.75 | 2619.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 2620.00 | 2625.40 | 2619.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:45:00 | 2623.00 | 2625.40 | 2619.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 2616.20 | 2623.56 | 2618.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 2616.20 | 2623.56 | 2618.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 2615.05 | 2621.86 | 2618.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 2633.00 | 2621.86 | 2618.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 2721.40 | 2734.85 | 2735.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 2721.40 | 2734.85 | 2735.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 12:15:00 | 2691.65 | 2720.79 | 2728.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 09:15:00 | 2703.00 | 2697.98 | 2713.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-08 10:00:00 | 2703.00 | 2697.98 | 2713.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 2698.45 | 2673.59 | 2686.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 2698.45 | 2673.59 | 2686.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 2688.65 | 2676.60 | 2686.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 2673.00 | 2678.26 | 2686.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 2672.00 | 2657.47 | 2665.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:15:00 | 2675.15 | 2662.35 | 2666.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:00:00 | 2677.00 | 2665.28 | 2667.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 2668.95 | 2666.01 | 2667.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 2677.35 | 2669.24 | 2668.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 2677.35 | 2669.24 | 2668.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 12:15:00 | 2690.10 | 2674.98 | 2671.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 2695.55 | 2698.39 | 2688.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 15:00:00 | 2695.55 | 2698.39 | 2688.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 2708.50 | 2705.69 | 2696.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 13:15:00 | 2713.25 | 2705.69 | 2696.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:15:00 | 2710.40 | 2706.55 | 2697.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 10:15:00 | 2679.05 | 2701.42 | 2698.36 | SL hit (close<static) qty=1.00 sl=2695.30 alert=retest2 |

### Cycle 81 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 2688.40 | 2696.61 | 2696.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 15:15:00 | 2675.50 | 2692.39 | 2694.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 2662.45 | 2642.44 | 2660.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 2662.45 | 2642.44 | 2660.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 2662.45 | 2642.44 | 2660.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 2655.65 | 2642.44 | 2660.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 2652.90 | 2644.53 | 2659.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 2663.30 | 2644.53 | 2659.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 2652.90 | 2646.20 | 2659.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 2652.90 | 2646.20 | 2659.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 2647.65 | 2645.08 | 2656.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:00:00 | 2647.65 | 2645.08 | 2656.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 2648.10 | 2644.10 | 2652.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 2602.40 | 2657.39 | 2657.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 2634.75 | 2655.46 | 2656.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 10:15:00 | 2638.60 | 2652.03 | 2654.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:15:00 | 2635.05 | 2650.27 | 2653.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 2628.00 | 2605.00 | 2619.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 2628.00 | 2605.00 | 2619.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 2636.00 | 2611.20 | 2621.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:45:00 | 2638.10 | 2611.20 | 2621.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 2632.50 | 2615.46 | 2622.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:15:00 | 2633.00 | 2615.46 | 2622.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 2616.90 | 2619.70 | 2622.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 2640.75 | 2624.60 | 2624.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 2640.75 | 2624.60 | 2624.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 2659.40 | 2631.56 | 2627.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 2603.20 | 2634.85 | 2631.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 14:15:00 | 2603.20 | 2634.85 | 2631.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 2603.20 | 2634.85 | 2631.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:30:00 | 2596.00 | 2634.85 | 2631.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 2614.25 | 2630.73 | 2630.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:15:00 | 2601.30 | 2630.73 | 2630.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 2583.85 | 2621.35 | 2625.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 10:15:00 | 2520.45 | 2570.80 | 2589.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2422.00 | 2402.49 | 2441.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 2422.00 | 2402.49 | 2441.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 2397.50 | 2385.65 | 2398.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 2377.60 | 2385.65 | 2398.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 15:15:00 | 2339.80 | 2316.85 | 2313.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 2339.80 | 2316.85 | 2313.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 10:15:00 | 2341.50 | 2325.14 | 2318.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 2335.80 | 2340.86 | 2331.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:45:00 | 2337.55 | 2340.86 | 2331.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 2320.00 | 2336.69 | 2330.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:00:00 | 2320.00 | 2336.69 | 2330.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 2325.90 | 2334.53 | 2330.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 2339.40 | 2330.83 | 2329.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 12:15:00 | 2322.00 | 2328.52 | 2328.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 12:15:00 | 2322.00 | 2328.52 | 2328.71 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 10:15:00 | 2336.95 | 2329.18 | 2328.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 11:15:00 | 2358.55 | 2335.05 | 2331.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 2339.90 | 2343.49 | 2337.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 2339.90 | 2343.49 | 2337.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 2339.90 | 2343.49 | 2337.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 2339.90 | 2343.49 | 2337.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 2326.55 | 2340.10 | 2336.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 2326.55 | 2340.10 | 2336.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 2323.85 | 2336.85 | 2335.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 2323.85 | 2336.85 | 2335.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 2322.05 | 2333.89 | 2334.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 2317.20 | 2330.55 | 2332.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 2335.00 | 2330.02 | 2331.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 2335.00 | 2330.02 | 2331.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 2335.00 | 2330.02 | 2331.84 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 2343.25 | 2334.08 | 2333.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 15:15:00 | 2344.00 | 2336.06 | 2334.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 2341.90 | 2343.57 | 2339.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 2341.90 | 2343.57 | 2339.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 2341.90 | 2343.57 | 2339.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 2340.45 | 2343.57 | 2339.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 2345.10 | 2343.88 | 2340.45 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 2333.00 | 2339.23 | 2339.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 2308.55 | 2333.09 | 2336.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 2324.70 | 2314.40 | 2322.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 2324.70 | 2314.40 | 2322.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 2324.70 | 2314.40 | 2322.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 2323.20 | 2314.40 | 2322.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 2333.30 | 2318.18 | 2323.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 2333.30 | 2318.18 | 2323.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 2328.65 | 2320.27 | 2323.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:30:00 | 2335.15 | 2320.27 | 2323.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 15:15:00 | 2331.00 | 2326.66 | 2326.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 10:15:00 | 2334.65 | 2328.64 | 2327.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 2382.25 | 2403.13 | 2381.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 2382.25 | 2403.13 | 2381.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 2382.25 | 2403.13 | 2381.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 2382.25 | 2403.13 | 2381.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 2405.75 | 2403.66 | 2383.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:30:00 | 2422.20 | 2410.73 | 2388.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:45:00 | 2421.70 | 2420.10 | 2403.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 10:30:00 | 2419.00 | 2422.16 | 2405.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 2471.85 | 2495.37 | 2495.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 2471.85 | 2495.37 | 2495.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 2466.15 | 2489.53 | 2493.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 2468.15 | 2445.64 | 2453.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 09:15:00 | 2468.15 | 2445.64 | 2453.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 2468.15 | 2445.64 | 2453.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 2466.25 | 2445.64 | 2453.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 2461.15 | 2448.74 | 2454.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:30:00 | 2468.90 | 2448.74 | 2454.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 2484.55 | 2461.14 | 2459.32 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 2457.95 | 2466.06 | 2466.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 2439.95 | 2459.07 | 2463.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 12:15:00 | 2453.55 | 2452.74 | 2458.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 13:00:00 | 2453.55 | 2452.74 | 2458.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 2473.05 | 2457.12 | 2459.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 2473.05 | 2457.12 | 2459.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 2474.90 | 2460.68 | 2461.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 2485.05 | 2460.68 | 2461.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 2503.50 | 2469.24 | 2465.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 09:15:00 | 2528.00 | 2493.23 | 2481.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 2509.00 | 2509.65 | 2496.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:15:00 | 2502.50 | 2509.65 | 2496.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 2496.40 | 2507.00 | 2496.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 2496.40 | 2507.00 | 2496.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 2492.25 | 2504.05 | 2496.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:45:00 | 2492.55 | 2504.05 | 2496.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 2504.05 | 2504.05 | 2497.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 13:00:00 | 2513.10 | 2505.86 | 2498.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 14:45:00 | 2510.20 | 2508.50 | 2501.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 10:45:00 | 2512.85 | 2506.66 | 2502.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 2471.30 | 2499.59 | 2499.28 | SL hit (close<static) qty=1.00 sl=2492.25 alert=retest2 |

### Cycle 95 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 2461.85 | 2492.04 | 2495.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 2446.00 | 2482.83 | 2491.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 2473.50 | 2473.42 | 2483.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:45:00 | 2475.90 | 2473.42 | 2483.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 2388.20 | 2374.35 | 2392.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 2377.05 | 2374.35 | 2392.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2376.30 | 2374.74 | 2390.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 2363.50 | 2374.34 | 2387.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 2363.15 | 2372.10 | 2385.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 2245.32 | 2272.48 | 2285.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 2244.99 | 2272.48 | 2285.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 13:15:00 | 2281.95 | 2272.00 | 2280.65 | SL hit (close>ema200) qty=0.50 sl=2272.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 11:15:00 | 2299.00 | 2284.46 | 2284.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 12:15:00 | 2315.00 | 2290.57 | 2286.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 2286.90 | 2295.15 | 2290.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 2286.90 | 2295.15 | 2290.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 2286.90 | 2295.15 | 2290.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 2286.90 | 2295.15 | 2290.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 2277.80 | 2291.68 | 2289.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 2273.10 | 2291.68 | 2289.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 2273.75 | 2288.10 | 2288.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 2262.20 | 2282.92 | 2285.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 2274.60 | 2268.89 | 2276.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 10:15:00 | 2274.60 | 2268.89 | 2276.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 2274.60 | 2268.89 | 2276.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 2274.60 | 2268.89 | 2276.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 2269.40 | 2268.99 | 2275.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:30:00 | 2286.00 | 2268.99 | 2275.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 2271.25 | 2269.44 | 2275.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 2271.25 | 2269.44 | 2275.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 2264.65 | 2268.48 | 2274.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 2272.70 | 2268.48 | 2274.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 2280.90 | 2267.98 | 2272.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 2280.90 | 2267.98 | 2272.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 2273.50 | 2269.08 | 2272.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:30:00 | 2266.70 | 2266.74 | 2271.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 15:00:00 | 2269.00 | 2261.48 | 2267.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:30:00 | 2260.60 | 2244.59 | 2248.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 12:15:00 | 2293.05 | 2254.28 | 2252.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 2293.05 | 2254.28 | 2252.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 13:15:00 | 2297.85 | 2262.99 | 2256.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 10:15:00 | 2277.55 | 2278.95 | 2267.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 11:00:00 | 2277.55 | 2278.95 | 2267.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 2320.00 | 2333.80 | 2317.47 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 2285.15 | 2311.90 | 2313.97 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 2320.55 | 2306.69 | 2306.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 2338.70 | 2315.38 | 2310.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 2302.05 | 2337.10 | 2327.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 2302.05 | 2337.10 | 2327.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 2302.05 | 2337.10 | 2327.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 2302.05 | 2337.10 | 2327.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 2331.40 | 2335.96 | 2327.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 2311.75 | 2335.96 | 2327.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 2322.10 | 2333.19 | 2327.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 2322.10 | 2333.19 | 2327.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 2325.95 | 2331.74 | 2327.26 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 2309.05 | 2322.57 | 2324.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 2292.55 | 2316.57 | 2321.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 2281.10 | 2280.06 | 2292.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 10:15:00 | 2290.00 | 2280.06 | 2292.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 2282.35 | 2280.52 | 2291.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 2287.75 | 2280.52 | 2291.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 2224.20 | 2201.44 | 2210.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 2224.20 | 2201.44 | 2210.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 2221.90 | 2205.53 | 2211.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 2226.95 | 2205.53 | 2211.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 2216.50 | 2209.02 | 2212.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:00:00 | 2216.50 | 2209.02 | 2212.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 2210.85 | 2209.39 | 2212.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:30:00 | 2216.70 | 2209.39 | 2212.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 2185.05 | 2204.52 | 2209.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 1967.15 | 2200.58 | 2207.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 1868.79 | 2147.27 | 2182.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 2068.10 | 2056.68 | 2107.51 | SL hit (close>ema200) qty=0.50 sl=2056.68 alert=retest2 |

### Cycle 102 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 2152.25 | 2117.37 | 2114.89 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 15:15:00 | 2116.00 | 2121.48 | 2121.81 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 2140.60 | 2125.30 | 2123.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 11:15:00 | 2155.25 | 2132.11 | 2126.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 2185.05 | 2191.39 | 2172.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 15:00:00 | 2185.05 | 2191.39 | 2172.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 2249.95 | 2265.94 | 2248.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:00:00 | 2249.95 | 2265.94 | 2248.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 2251.95 | 2263.14 | 2248.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 2246.95 | 2263.14 | 2248.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 2249.70 | 2260.45 | 2248.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:00:00 | 2249.70 | 2260.45 | 2248.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 2245.00 | 2257.36 | 2248.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:00:00 | 2245.00 | 2257.36 | 2248.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 2239.70 | 2253.83 | 2247.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:45:00 | 2239.00 | 2253.83 | 2247.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 2260.00 | 2251.90 | 2247.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 11:30:00 | 2264.00 | 2255.90 | 2250.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:00:00 | 2264.80 | 2257.68 | 2251.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 14:15:00 | 2268.20 | 2257.72 | 2251.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 2268.80 | 2259.93 | 2253.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 2259.05 | 2264.00 | 2259.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 13:45:00 | 2260.00 | 2264.00 | 2259.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 2257.35 | 2262.67 | 2258.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:30:00 | 2257.80 | 2262.67 | 2258.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 2257.05 | 2261.54 | 2258.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 2248.00 | 2258.84 | 2257.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 2263.85 | 2259.84 | 2258.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-10 11:15:00 | 2249.45 | 2258.65 | 2258.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 2249.45 | 2258.65 | 2258.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 2242.60 | 2255.44 | 2257.46 | Break + close below crossover candle low |

### Cycle 106 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 2282.00 | 2257.61 | 2257.33 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 2248.70 | 2257.32 | 2257.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 2231.20 | 2251.60 | 2255.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 10:15:00 | 2258.25 | 2252.93 | 2255.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 10:15:00 | 2258.25 | 2252.93 | 2255.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 2258.25 | 2252.93 | 2255.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:45:00 | 2258.70 | 2252.93 | 2255.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 2246.35 | 2251.62 | 2254.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 13:15:00 | 2241.45 | 2250.66 | 2253.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:00:00 | 2241.20 | 2240.04 | 2241.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:15:00 | 2241.70 | 2240.78 | 2241.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 14:15:00 | 2246.55 | 2243.02 | 2242.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 2246.55 | 2243.02 | 2242.56 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 2231.95 | 2240.60 | 2241.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 2215.25 | 2234.66 | 2238.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 11:15:00 | 2124.05 | 2122.74 | 2145.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 12:00:00 | 2124.05 | 2122.74 | 2145.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 2096.80 | 2096.82 | 2107.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 2102.25 | 2096.82 | 2107.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 2090.80 | 2086.24 | 2093.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:45:00 | 2090.10 | 2086.24 | 2093.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 2092.90 | 2087.57 | 2093.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:15:00 | 2085.45 | 2089.68 | 2093.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:30:00 | 2083.85 | 2083.81 | 2086.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 2083.00 | 2083.81 | 2086.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 2078.25 | 2061.21 | 2060.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 2078.25 | 2061.21 | 2060.16 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 2052.75 | 2061.61 | 2061.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 2038.00 | 2054.75 | 2058.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 2014.35 | 2009.10 | 2028.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 2014.35 | 2009.10 | 2028.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 2014.35 | 2009.10 | 2028.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 2006.50 | 2009.10 | 2028.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 2020.90 | 2011.06 | 2021.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 2020.90 | 2011.06 | 2021.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 2015.75 | 2012.00 | 2021.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 2009.00 | 2012.00 | 2021.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 2025.50 | 2014.70 | 2021.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 15:15:00 | 2005.60 | 2013.77 | 2018.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1905.32 | 1937.69 | 1961.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 1907.20 | 1893.46 | 1922.71 | SL hit (close>ema200) qty=0.50 sl=1893.46 alert=retest2 |

### Cycle 112 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 1956.85 | 1933.06 | 1931.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 2030.85 | 1969.84 | 1952.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 14:15:00 | 1986.95 | 1988.85 | 1970.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 15:00:00 | 1986.95 | 1988.85 | 1970.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1997.35 | 1990.04 | 1979.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:15:00 | 2005.15 | 1990.04 | 1979.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 10:45:00 | 2003.35 | 1998.93 | 1987.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:30:00 | 2003.05 | 2001.18 | 1990.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 09:45:00 | 2008.30 | 2018.81 | 2010.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 2002.80 | 2015.61 | 2009.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 2002.80 | 2015.61 | 2009.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 1986.10 | 2009.71 | 2007.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 1986.10 | 2009.71 | 2007.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-22 12:15:00 | 1981.60 | 2004.09 | 2005.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 1981.60 | 2004.09 | 2005.38 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 2021.35 | 2007.91 | 2006.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 14:15:00 | 2044.75 | 2020.34 | 2012.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 2035.95 | 2044.02 | 2033.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 2035.95 | 2044.02 | 2033.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 2035.95 | 2044.02 | 2033.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:30:00 | 2027.10 | 2044.02 | 2033.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 2034.15 | 2042.05 | 2033.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 11:15:00 | 2045.10 | 2042.05 | 2033.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 14:15:00 | 1999.75 | 2036.12 | 2033.81 | SL hit (close<static) qty=1.00 sl=2023.65 alert=retest2 |

### Cycle 115 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 1996.00 | 2028.10 | 2030.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1976.25 | 2017.73 | 2025.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 2040.65 | 2017.83 | 2023.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 12:15:00 | 2040.65 | 2017.83 | 2023.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 2040.65 | 2017.83 | 2023.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 2040.65 | 2017.83 | 2023.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 2023.00 | 2018.86 | 2023.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 2005.70 | 2015.49 | 2021.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 10:30:00 | 2009.10 | 2012.43 | 2018.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:00:00 | 2006.35 | 2012.43 | 2018.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:45:00 | 2004.80 | 2010.74 | 2016.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 2015.90 | 2011.78 | 2016.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:45:00 | 2028.05 | 2011.78 | 2016.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 2005.40 | 2010.50 | 2015.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:30:00 | 2015.85 | 2010.50 | 2015.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1984.25 | 2003.42 | 2011.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:45:00 | 1966.25 | 1995.77 | 2006.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 1966.40 | 1987.06 | 2000.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 2008.25 | 2000.05 | 1999.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 15:15:00 | 2008.25 | 2000.05 | 1999.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 2021.75 | 2004.39 | 2001.87 | Break + close above crossover candle high |

### Cycle 117 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1969.35 | 1999.75 | 2000.66 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 2038.50 | 2003.16 | 1998.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 2056.75 | 2030.26 | 2016.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 2015.55 | 2037.15 | 2027.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 2015.55 | 2037.15 | 2027.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 2015.55 | 2037.15 | 2027.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:45:00 | 2013.70 | 2037.15 | 2027.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 2014.35 | 2032.59 | 2026.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 2014.35 | 2032.59 | 2026.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 1995.00 | 2019.07 | 2021.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 1988.10 | 2012.88 | 2018.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 2001.80 | 1999.72 | 2006.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 15:00:00 | 2001.80 | 1999.72 | 2006.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1977.10 | 1995.12 | 2003.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:45:00 | 1972.85 | 1990.27 | 2000.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1874.21 | 1931.30 | 1955.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 11:15:00 | 1930.90 | 1929.55 | 1950.70 | SL hit (close>ema200) qty=0.50 sl=1929.55 alert=retest2 |

### Cycle 120 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 1889.20 | 1878.52 | 1878.38 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 1870.25 | 1877.41 | 1877.94 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 13:15:00 | 1884.60 | 1879.11 | 1878.49 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 1863.10 | 1877.06 | 1877.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 1848.85 | 1869.39 | 1873.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 1827.00 | 1826.64 | 1839.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 1827.00 | 1826.64 | 1839.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 1786.95 | 1809.60 | 1821.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 10:30:00 | 1783.00 | 1805.77 | 1818.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 14:15:00 | 1829.90 | 1821.00 | 1820.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 1829.90 | 1821.00 | 1820.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 1850.30 | 1828.98 | 1824.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1862.00 | 1867.37 | 1856.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 11:45:00 | 1871.40 | 1867.37 | 1856.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1900.55 | 1880.79 | 1867.53 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 1833.60 | 1866.31 | 1869.69 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 1868.10 | 1863.48 | 1863.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 1877.25 | 1867.20 | 1864.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 12:15:00 | 1880.50 | 1881.60 | 1874.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 12:45:00 | 1882.45 | 1881.60 | 1874.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 1893.80 | 1895.25 | 1888.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:45:00 | 1894.80 | 1895.25 | 1888.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 1896.85 | 1894.68 | 1889.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 1899.80 | 1892.12 | 1889.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:30:00 | 1900.50 | 1893.10 | 1890.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 11:15:00 | 1901.30 | 1893.10 | 1890.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:00:00 | 1901.75 | 1894.83 | 1891.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1925.35 | 1936.65 | 1928.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 1947.45 | 1936.65 | 1928.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1948.00 | 1938.92 | 1929.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 1954.00 | 1940.54 | 1931.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:15:00 | 1952.60 | 1939.52 | 1936.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 11:30:00 | 1950.20 | 1950.09 | 1944.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 1956.20 | 1943.19 | 1942.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1947.05 | 1943.96 | 1942.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:00:00 | 1964.65 | 1952.63 | 1948.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:00:00 | 1967.90 | 1974.96 | 1970.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1892.00 | 1956.30 | 1963.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1892.00 | 1956.30 | 1963.02 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 1979.00 | 1953.38 | 1951.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 13:15:00 | 1985.00 | 1970.83 | 1963.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 2049.10 | 2050.55 | 2033.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 2049.10 | 2050.55 | 2033.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 2056.00 | 2079.24 | 2074.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 2056.00 | 2079.24 | 2074.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 2055.00 | 2074.39 | 2073.13 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 2058.30 | 2071.17 | 2071.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 13:15:00 | 2050.20 | 2065.64 | 2069.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 2087.40 | 2066.32 | 2068.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 2087.40 | 2066.32 | 2068.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 2087.40 | 2066.32 | 2068.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:00:00 | 2087.40 | 2066.32 | 2068.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 10:15:00 | 2091.70 | 2071.39 | 2070.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 13:15:00 | 2110.00 | 2084.38 | 2076.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 14:15:00 | 2064.50 | 2080.40 | 2075.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 14:15:00 | 2064.50 | 2080.40 | 2075.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 2064.50 | 2080.40 | 2075.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 2064.50 | 2080.40 | 2075.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 2065.00 | 2077.32 | 2074.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 2022.60 | 2077.32 | 2074.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1959.90 | 2053.84 | 2064.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1943.50 | 2031.77 | 2053.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 1892.70 | 1876.41 | 1890.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 10:15:00 | 1892.70 | 1876.41 | 1890.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1892.70 | 1876.41 | 1890.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 1892.70 | 1876.41 | 1890.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1897.80 | 1880.69 | 1890.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1897.80 | 1880.69 | 1890.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1899.90 | 1884.53 | 1891.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:30:00 | 1903.00 | 1884.53 | 1891.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1886.60 | 1886.19 | 1891.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 1876.70 | 1886.15 | 1890.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1782.87 | 1824.13 | 1840.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 1810.10 | 1809.95 | 1824.72 | SL hit (close>ema200) qty=0.50 sl=1809.95 alert=retest2 |

### Cycle 132 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1854.00 | 1833.14 | 1832.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1864.70 | 1853.62 | 1844.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 1853.20 | 1857.44 | 1849.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 1853.20 | 1857.44 | 1849.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1920.00 | 1934.75 | 1924.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 1920.00 | 1934.75 | 1924.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 1920.50 | 1931.90 | 1924.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 1924.60 | 1931.90 | 1924.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1939.60 | 1933.44 | 1925.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 1941.70 | 1932.18 | 1926.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:15:00 | 1942.00 | 1933.84 | 1927.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 1942.20 | 1935.51 | 1929.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:45:00 | 1941.90 | 1935.67 | 1930.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1928.80 | 1934.29 | 1930.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 1928.80 | 1934.29 | 1930.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1932.70 | 1933.98 | 1930.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 1929.50 | 1933.98 | 1930.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1935.40 | 1934.26 | 1930.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:00:00 | 1937.50 | 1934.91 | 1931.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:30:00 | 1939.50 | 1937.07 | 1932.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:30:00 | 1937.00 | 1951.98 | 1950.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 1937.70 | 1951.98 | 1950.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 1936.10 | 1948.80 | 1949.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 1926.30 | 1933.89 | 1939.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 1926.90 | 1924.59 | 1931.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 1926.90 | 1924.59 | 1931.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1882.80 | 1884.24 | 1894.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:45:00 | 1872.10 | 1880.22 | 1889.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 13:30:00 | 1875.00 | 1879.36 | 1888.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:15:00 | 1875.00 | 1879.36 | 1888.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1872.30 | 1879.30 | 1886.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1878.00 | 1874.66 | 1880.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 1878.00 | 1874.66 | 1880.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1885.70 | 1876.92 | 1880.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 1887.00 | 1876.92 | 1880.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1891.10 | 1879.76 | 1881.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:00:00 | 1891.10 | 1879.76 | 1881.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 1890.90 | 1883.88 | 1883.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1890.90 | 1883.88 | 1883.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 1897.20 | 1888.42 | 1885.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 1913.30 | 1913.45 | 1906.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 15:00:00 | 1913.30 | 1913.45 | 1906.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1898.80 | 1912.24 | 1909.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1899.10 | 1912.24 | 1909.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1904.60 | 1910.71 | 1908.92 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 1890.00 | 1904.32 | 1906.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1886.60 | 1900.77 | 1904.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 1854.80 | 1851.30 | 1865.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 1854.80 | 1851.30 | 1865.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1862.20 | 1853.48 | 1864.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 1862.00 | 1853.48 | 1864.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1871.50 | 1857.09 | 1865.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1871.50 | 1857.09 | 1865.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1874.30 | 1860.53 | 1866.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1874.30 | 1860.53 | 1866.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1870.10 | 1862.44 | 1866.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1870.70 | 1862.44 | 1866.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1863.00 | 1863.14 | 1866.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:15:00 | 1872.80 | 1863.14 | 1866.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1868.80 | 1864.27 | 1866.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:30:00 | 1872.90 | 1864.27 | 1866.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 1863.70 | 1864.16 | 1866.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 1866.50 | 1864.16 | 1866.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 1865.70 | 1864.47 | 1866.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:45:00 | 1865.00 | 1864.47 | 1866.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 1859.70 | 1863.51 | 1865.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 1857.00 | 1863.51 | 1865.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 1852.80 | 1861.39 | 1863.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1858.80 | 1831.28 | 1831.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1858.80 | 1831.28 | 1831.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 1869.80 | 1838.98 | 1834.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1846.20 | 1846.32 | 1839.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1846.20 | 1846.32 | 1839.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1914.80 | 1918.23 | 1909.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 1911.10 | 1918.23 | 1909.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1951.50 | 1961.37 | 1953.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 1951.50 | 1961.37 | 1953.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1947.00 | 1958.50 | 1953.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 1945.80 | 1958.50 | 1953.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1957.20 | 1960.03 | 1956.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1957.20 | 1960.03 | 1956.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1949.90 | 1958.00 | 1955.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 1949.90 | 1958.00 | 1955.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1960.30 | 1958.46 | 1956.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 12:30:00 | 1965.50 | 1960.47 | 1957.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1971.40 | 1986.30 | 1987.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 10:15:00 | 1971.40 | 1986.30 | 1987.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 11:15:00 | 1970.00 | 1983.04 | 1985.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1986.00 | 1980.25 | 1982.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1986.00 | 1980.25 | 1982.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1986.00 | 1980.25 | 1982.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 1986.50 | 1980.25 | 1982.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1991.00 | 1982.40 | 1983.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 1991.00 | 1982.40 | 1983.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1986.80 | 1983.28 | 1983.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 12:30:00 | 1983.90 | 1982.61 | 1983.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 1985.00 | 1979.98 | 1981.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 1993.00 | 1983.77 | 1983.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1993.00 | 1983.77 | 1983.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 14:15:00 | 1996.50 | 1986.32 | 1984.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 1980.30 | 1985.27 | 1984.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1980.30 | 1985.27 | 1984.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1980.30 | 1985.27 | 1984.16 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 1979.70 | 1983.31 | 1983.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1973.20 | 1979.16 | 1981.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 15:15:00 | 1973.30 | 1973.02 | 1976.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:15:00 | 1975.40 | 1973.02 | 1976.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1974.60 | 1973.34 | 1976.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 10:30:00 | 1966.30 | 1971.27 | 1975.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 1969.00 | 1975.12 | 1975.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1970.10 | 1974.08 | 1975.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 1871.59 | 1913.12 | 1933.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 1867.98 | 1903.32 | 1927.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 1870.55 | 1903.32 | 1927.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1824.90 | 1813.94 | 1825.66 | SL hit (close>ema200) qty=0.50 sl=1813.94 alert=retest2 |

### Cycle 140 — BUY (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 09:15:00 | 1828.00 | 1804.21 | 1802.31 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 1802.70 | 1808.74 | 1808.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 1799.70 | 1806.93 | 1808.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 1790.50 | 1790.49 | 1795.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1790.50 | 1790.49 | 1795.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1790.50 | 1790.49 | 1795.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 1783.10 | 1789.89 | 1792.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 12:15:00 | 1785.00 | 1789.89 | 1792.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 1785.70 | 1789.87 | 1792.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 1785.20 | 1789.46 | 1791.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1791.30 | 1789.14 | 1791.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 1785.90 | 1788.72 | 1790.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 1785.80 | 1787.16 | 1789.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:45:00 | 1784.60 | 1787.11 | 1789.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 14:15:00 | 1785.10 | 1787.11 | 1789.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1840.10 | 1796.33 | 1792.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 1848.20 | 1820.12 | 1805.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 1859.50 | 1860.79 | 1850.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:45:00 | 1859.60 | 1860.79 | 1850.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1848.00 | 1857.96 | 1853.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:45:00 | 1848.90 | 1857.96 | 1853.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1850.00 | 1856.37 | 1852.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1845.30 | 1856.37 | 1852.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1833.90 | 1849.85 | 1850.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 1828.70 | 1845.62 | 1848.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1805.60 | 1804.89 | 1814.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 1805.60 | 1804.89 | 1814.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1804.30 | 1803.54 | 1809.22 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1817.00 | 1809.43 | 1809.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1824.20 | 1816.09 | 1812.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1814.90 | 1821.35 | 1816.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1814.90 | 1821.35 | 1816.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1814.90 | 1821.35 | 1816.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1814.90 | 1821.35 | 1816.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1824.00 | 1821.88 | 1817.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 1819.50 | 1821.88 | 1817.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1815.00 | 1820.50 | 1817.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1825.20 | 1820.50 | 1817.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:45:00 | 1832.70 | 1823.22 | 1818.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 12:15:00 | 1827.50 | 1833.56 | 1833.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 1827.50 | 1833.56 | 1833.98 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 1844.40 | 1834.15 | 1833.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 1850.00 | 1841.80 | 1838.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 1847.80 | 1849.32 | 1846.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 14:15:00 | 1847.80 | 1849.32 | 1846.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1847.80 | 1849.32 | 1846.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 1846.10 | 1849.32 | 1846.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1850.40 | 1849.54 | 1846.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1852.00 | 1849.54 | 1846.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1840.00 | 1847.98 | 1847.06 | SL hit (close<static) qty=1.00 sl=1846.70 alert=retest2 |

### Cycle 147 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 1837.20 | 1845.82 | 1846.16 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1854.40 | 1847.10 | 1846.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 1860.00 | 1851.68 | 1849.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 14:15:00 | 1866.00 | 1866.69 | 1861.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 15:00:00 | 1866.00 | 1866.69 | 1861.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1863.10 | 1866.02 | 1861.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 1865.40 | 1866.02 | 1861.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1861.90 | 1865.20 | 1861.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:45:00 | 1862.40 | 1865.20 | 1861.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1860.80 | 1864.32 | 1861.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 1860.80 | 1864.32 | 1861.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1858.90 | 1863.23 | 1861.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:30:00 | 1857.10 | 1863.23 | 1861.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 1854.80 | 1859.24 | 1859.78 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 1864.00 | 1860.11 | 1860.01 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 1850.00 | 1858.08 | 1859.10 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 1869.40 | 1860.76 | 1859.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 12:15:00 | 1875.50 | 1865.27 | 1862.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1886.00 | 1889.99 | 1879.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:45:00 | 1884.90 | 1889.99 | 1879.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1884.00 | 1888.80 | 1879.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 1891.20 | 1888.80 | 1879.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:00:00 | 1893.60 | 1889.76 | 1880.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:30:00 | 1890.80 | 1888.29 | 1881.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 13:15:00 | 1887.10 | 1886.94 | 1881.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1879.40 | 1885.02 | 1881.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 1877.50 | 1885.02 | 1881.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1879.50 | 1883.92 | 1881.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 1868.30 | 1883.92 | 1881.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1876.60 | 1882.46 | 1881.02 | SL hit (close<static) qty=1.00 sl=1878.30 alert=retest2 |

### Cycle 153 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1873.60 | 1878.96 | 1879.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 1864.50 | 1874.62 | 1877.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 1837.00 | 1834.67 | 1844.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 1837.00 | 1834.67 | 1844.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1825.50 | 1826.10 | 1836.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 1831.90 | 1826.10 | 1836.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1840.50 | 1828.50 | 1832.35 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1844.70 | 1835.00 | 1834.63 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 15:15:00 | 1829.90 | 1833.67 | 1834.16 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1844.30 | 1835.80 | 1835.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 1856.10 | 1843.83 | 1841.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1844.30 | 1850.43 | 1846.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 1844.30 | 1850.43 | 1846.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1844.30 | 1850.43 | 1846.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 1844.90 | 1850.43 | 1846.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1851.40 | 1850.62 | 1846.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 1857.00 | 1853.08 | 1848.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 1863.00 | 1867.65 | 1868.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1863.00 | 1867.65 | 1868.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 1855.30 | 1862.74 | 1864.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 1862.90 | 1862.06 | 1864.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 1862.90 | 1862.06 | 1864.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1862.90 | 1862.06 | 1864.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 1856.90 | 1862.36 | 1863.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 1854.90 | 1860.55 | 1862.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 1853.90 | 1859.22 | 1861.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 1859.00 | 1844.49 | 1843.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 15:15:00 | 1859.00 | 1844.49 | 1843.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 1863.10 | 1851.44 | 1848.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 1850.80 | 1852.40 | 1849.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 10:15:00 | 1850.80 | 1852.40 | 1849.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1850.80 | 1852.40 | 1849.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 1848.00 | 1852.40 | 1849.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1860.50 | 1853.77 | 1850.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 1865.50 | 1855.23 | 1851.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:15:00 | 1862.70 | 1869.90 | 1865.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 1864.00 | 1865.87 | 1864.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1862.00 | 1864.70 | 1863.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1863.00 | 1864.36 | 1863.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 1877.90 | 1864.30 | 1863.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1845.00 | 1871.01 | 1872.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 1845.00 | 1871.01 | 1872.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 1839.90 | 1864.79 | 1869.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1835.00 | 1831.61 | 1840.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:45:00 | 1836.70 | 1831.61 | 1840.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1846.00 | 1834.48 | 1840.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 1846.00 | 1834.48 | 1840.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1843.20 | 1836.23 | 1840.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 1850.00 | 1836.23 | 1840.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1839.50 | 1836.88 | 1840.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1836.70 | 1838.69 | 1841.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 1850.00 | 1842.85 | 1842.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1850.00 | 1842.85 | 1842.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1860.50 | 1847.41 | 1845.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 1852.40 | 1852.61 | 1849.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 1852.40 | 1852.61 | 1849.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1852.40 | 1852.61 | 1849.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:45:00 | 1850.80 | 1852.61 | 1849.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1854.70 | 1853.86 | 1850.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 1852.10 | 1853.86 | 1850.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1849.50 | 1852.98 | 1850.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 1849.50 | 1852.98 | 1850.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1847.70 | 1851.93 | 1850.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1847.70 | 1851.93 | 1850.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1845.10 | 1849.29 | 1849.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1839.00 | 1846.93 | 1848.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1844.10 | 1843.75 | 1846.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1844.10 | 1843.75 | 1846.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1844.10 | 1843.75 | 1846.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:45:00 | 1840.60 | 1844.24 | 1845.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 1841.10 | 1843.22 | 1844.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:30:00 | 1840.40 | 1839.53 | 1842.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:00:00 | 1840.00 | 1838.68 | 1841.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1852.20 | 1841.30 | 1841.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 1850.00 | 1841.30 | 1841.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1858.00 | 1844.64 | 1843.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 1858.00 | 1844.64 | 1843.01 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 1839.10 | 1844.36 | 1844.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 1828.00 | 1838.52 | 1841.50 | Break + close below crossover candle low |

### Cycle 164 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 1916.20 | 1845.47 | 1841.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 1930.00 | 1862.38 | 1849.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 11:15:00 | 1864.90 | 1865.67 | 1854.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 1864.90 | 1865.67 | 1854.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1874.90 | 1867.70 | 1859.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:45:00 | 1880.10 | 1871.96 | 1865.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 1856.10 | 1865.22 | 1866.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 1856.10 | 1865.22 | 1866.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 1850.00 | 1862.17 | 1864.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 1854.60 | 1852.12 | 1855.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 1854.60 | 1852.12 | 1855.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1854.60 | 1852.12 | 1855.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 1854.60 | 1852.12 | 1855.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1853.90 | 1851.48 | 1854.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1853.90 | 1851.48 | 1854.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1855.00 | 1852.18 | 1854.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1845.90 | 1852.18 | 1854.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1839.00 | 1849.54 | 1852.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 1837.00 | 1847.62 | 1851.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 1838.00 | 1844.67 | 1849.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:45:00 | 1838.30 | 1843.64 | 1848.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:00:00 | 1837.20 | 1842.35 | 1847.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1791.60 | 1789.70 | 1798.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 1786.10 | 1789.10 | 1796.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 1786.90 | 1781.94 | 1785.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 1784.80 | 1783.01 | 1785.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:15:00 | 1786.50 | 1783.01 | 1785.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1785.40 | 1783.49 | 1785.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:30:00 | 1783.80 | 1782.79 | 1785.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 10:15:00 | 1783.40 | 1779.84 | 1782.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 1775.70 | 1763.85 | 1762.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 1787.40 | 1769.39 | 1765.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1761.90 | 1769.05 | 1766.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1761.90 | 1769.05 | 1766.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1761.90 | 1769.05 | 1766.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1760.70 | 1769.05 | 1766.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1763.10 | 1767.86 | 1765.93 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 1757.10 | 1764.20 | 1764.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 1755.00 | 1762.36 | 1763.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 1736.40 | 1728.76 | 1732.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 1736.40 | 1728.76 | 1732.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 1736.40 | 1728.76 | 1732.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:45:00 | 1734.40 | 1728.76 | 1732.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1744.40 | 1731.89 | 1734.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 1744.40 | 1731.89 | 1734.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1730.00 | 1731.44 | 1733.28 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1738.10 | 1733.76 | 1733.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 11:15:00 | 1742.00 | 1735.86 | 1734.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 1738.20 | 1738.34 | 1736.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1748.90 | 1738.34 | 1736.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1742.20 | 1739.91 | 1737.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:30:00 | 1741.90 | 1739.91 | 1737.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1750.00 | 1759.96 | 1753.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 1750.00 | 1759.96 | 1753.87 | SL hit (close<ema400) qty=1.00 sl=1753.87 alert=retest1 |

### Cycle 169 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 1742.00 | 1753.31 | 1753.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1734.50 | 1747.32 | 1750.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 1704.40 | 1703.10 | 1715.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 1704.40 | 1703.10 | 1715.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1706.70 | 1705.47 | 1713.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1702.30 | 1705.30 | 1712.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 1719.00 | 1708.04 | 1713.33 | SL hit (close>static) qty=1.00 sl=1717.00 alert=retest2 |

### Cycle 170 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1725.60 | 1714.94 | 1714.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 1728.60 | 1717.67 | 1715.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 10:15:00 | 1737.30 | 1738.12 | 1730.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:30:00 | 1739.00 | 1738.12 | 1730.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1730.00 | 1736.07 | 1732.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1736.90 | 1736.07 | 1732.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1738.30 | 1736.51 | 1732.89 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 1720.90 | 1730.59 | 1730.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1706.80 | 1725.83 | 1728.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1706.80 | 1704.52 | 1714.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 1706.80 | 1704.52 | 1714.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 1714.20 | 1706.16 | 1713.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 1714.20 | 1706.16 | 1713.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1731.00 | 1711.13 | 1715.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1727.10 | 1715.00 | 1716.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1729.00 | 1719.40 | 1718.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 1733.60 | 1722.24 | 1719.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 1719.60 | 1724.83 | 1722.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1719.60 | 1724.83 | 1722.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1719.60 | 1724.83 | 1722.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:15:00 | 1716.00 | 1724.83 | 1722.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1719.00 | 1723.66 | 1721.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:45:00 | 1716.90 | 1723.66 | 1721.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1698.20 | 1716.79 | 1718.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1684.90 | 1710.41 | 1715.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1698.00 | 1696.94 | 1707.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 1686.00 | 1687.38 | 1695.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1686.00 | 1687.38 | 1695.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 14:15:00 | 1680.00 | 1689.66 | 1694.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 15:00:00 | 1684.50 | 1688.63 | 1693.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:00:00 | 1682.00 | 1680.29 | 1686.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:45:00 | 1679.70 | 1680.11 | 1685.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1635.70 | 1668.09 | 1676.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:00:00 | 1617.90 | 1643.94 | 1658.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:15:00 | 1600.27 | 1630.80 | 1648.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:30:00 | 1616.00 | 1630.80 | 1648.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 1596.00 | 1626.96 | 1645.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 1597.90 | 1626.96 | 1645.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 1595.71 | 1626.96 | 1645.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 1630.80 | 1626.29 | 1641.72 | SL hit (close>ema200) qty=0.50 sl=1626.29 alert=retest2 |

### Cycle 174 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1681.10 | 1650.27 | 1648.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 1687.00 | 1671.64 | 1662.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1677.00 | 1681.35 | 1671.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1677.00 | 1681.35 | 1671.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1670.80 | 1678.08 | 1671.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 1670.80 | 1678.08 | 1671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 1673.10 | 1677.09 | 1671.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 1670.00 | 1677.09 | 1671.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 1671.60 | 1675.99 | 1671.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 1671.60 | 1675.99 | 1671.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1677.50 | 1676.29 | 1672.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 1676.00 | 1676.29 | 1672.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 1674.90 | 1676.01 | 1672.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 1668.50 | 1676.01 | 1672.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1656.10 | 1672.03 | 1670.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1656.10 | 1672.03 | 1670.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 1660.40 | 1669.70 | 1670.03 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1676.00 | 1668.66 | 1668.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1690.10 | 1672.95 | 1670.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1694.10 | 1698.07 | 1689.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 1694.10 | 1698.07 | 1689.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1688.50 | 1695.08 | 1690.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1688.60 | 1695.08 | 1690.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1690.70 | 1694.20 | 1690.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:15:00 | 1687.60 | 1694.20 | 1690.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1688.00 | 1692.96 | 1690.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 1688.00 | 1692.96 | 1690.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1690.00 | 1692.37 | 1690.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 1695.00 | 1692.37 | 1690.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1681.30 | 1691.01 | 1690.63 | SL hit (close<static) qty=1.00 sl=1684.90 alert=retest2 |

### Cycle 177 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 1679.80 | 1688.77 | 1689.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 1676.00 | 1684.65 | 1687.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 1644.20 | 1638.51 | 1648.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 1644.20 | 1638.51 | 1648.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1638.50 | 1638.51 | 1647.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1640.00 | 1638.51 | 1647.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1644.40 | 1639.11 | 1646.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 1633.00 | 1637.23 | 1643.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 1634.10 | 1637.74 | 1640.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 1624.60 | 1621.70 | 1621.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1624.60 | 1621.70 | 1621.32 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 1617.30 | 1621.29 | 1621.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 1615.60 | 1620.15 | 1620.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1522.50 | 1519.08 | 1536.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 1526.30 | 1519.08 | 1536.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1531.00 | 1522.41 | 1534.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 1531.00 | 1522.41 | 1534.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1477.70 | 1472.74 | 1480.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 1474.30 | 1473.47 | 1479.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:30:00 | 1474.90 | 1473.98 | 1479.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 13:00:00 | 1474.30 | 1474.04 | 1479.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 1400.58 | 1419.90 | 1440.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 1401.15 | 1419.90 | 1440.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 1400.58 | 1419.90 | 1440.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 1378.60 | 1376.98 | 1390.92 | SL hit (close>ema200) qty=0.50 sl=1376.98 alert=retest2 |

### Cycle 180 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 1404.60 | 1394.39 | 1394.08 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1373.20 | 1393.41 | 1394.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 1366.60 | 1388.05 | 1391.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1370.80 | 1367.32 | 1377.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:00:00 | 1370.80 | 1367.32 | 1377.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1369.80 | 1368.12 | 1376.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1345.00 | 1375.21 | 1377.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1379.70 | 1345.08 | 1348.65 | SL hit (close>static) qty=1.00 sl=1376.40 alert=retest2 |

### Cycle 182 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1377.50 | 1351.57 | 1351.28 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1317.60 | 1348.54 | 1352.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1315.90 | 1332.31 | 1342.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1310.50 | 1284.62 | 1305.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1310.50 | 1284.62 | 1305.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1310.50 | 1284.62 | 1305.05 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 1329.00 | 1314.78 | 1313.58 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1304.10 | 1311.50 | 1312.22 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1319.80 | 1313.71 | 1313.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 1328.00 | 1316.56 | 1314.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1339.50 | 1343.06 | 1332.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 1339.50 | 1343.06 | 1332.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 1337.50 | 1341.43 | 1333.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 1335.00 | 1341.43 | 1333.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1389.60 | 1411.58 | 1401.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:30:00 | 1410.40 | 1406.73 | 1402.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1427.10 | 1406.73 | 1402.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 1426.60 | 1433.93 | 1434.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1426.60 | 1433.93 | 1434.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1410.80 | 1424.91 | 1429.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1428.70 | 1415.06 | 1420.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1428.70 | 1415.06 | 1420.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1428.70 | 1415.06 | 1420.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1432.80 | 1415.06 | 1420.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1433.00 | 1418.65 | 1421.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1433.00 | 1418.65 | 1421.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1454.90 | 1427.35 | 1425.03 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1410.20 | 1434.70 | 1435.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 1409.00 | 1429.56 | 1433.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 1425.60 | 1421.79 | 1426.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 10:15:00 | 1425.60 | 1421.79 | 1426.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1425.60 | 1421.79 | 1426.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 1425.60 | 1421.79 | 1426.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1425.00 | 1422.43 | 1426.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 1432.90 | 1422.43 | 1426.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 1411.00 | 1420.14 | 1424.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 1406.40 | 1420.14 | 1424.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:15:00 | 1396.50 | 1418.31 | 1423.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1422.40 | 1408.96 | 1407.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 1422.40 | 1408.96 | 1407.37 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 1400.10 | 1411.33 | 1411.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 1395.00 | 1408.07 | 1410.17 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-16 09:15:00 | 1860.50 | 2023-06-19 10:15:00 | 1841.95 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-06-27 13:30:00 | 1779.70 | 2023-06-28 09:15:00 | 1816.90 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2023-07-05 13:00:00 | 1815.00 | 2023-07-06 13:15:00 | 1815.90 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2023-07-05 13:30:00 | 1818.15 | 2023-07-06 13:15:00 | 1815.90 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2023-07-06 12:45:00 | 1815.75 | 2023-07-06 13:15:00 | 1815.90 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2023-07-11 12:30:00 | 1783.05 | 2023-07-17 10:15:00 | 1811.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-07-12 13:30:00 | 1785.05 | 2023-07-17 10:15:00 | 1811.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2023-07-13 10:45:00 | 1785.85 | 2023-07-17 10:15:00 | 1811.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-07-13 13:15:00 | 1785.25 | 2023-07-17 10:15:00 | 1811.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2023-07-19 09:15:00 | 1802.80 | 2023-07-20 09:15:00 | 1792.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-07-20 09:15:00 | 1809.00 | 2023-07-20 09:15:00 | 1792.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-08-30 11:00:00 | 1991.00 | 2023-08-31 09:15:00 | 1944.25 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2023-10-20 09:15:00 | 2020.75 | 2023-10-23 12:15:00 | 1919.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 09:15:00 | 2020.75 | 2023-10-26 13:15:00 | 1898.20 | STOP_HIT | 0.50 | 6.06% |
| SELL | retest2 | 2023-11-22 10:15:00 | 1827.60 | 2023-11-28 09:15:00 | 1866.05 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2023-11-30 14:00:00 | 1871.25 | 2023-12-05 09:15:00 | 2058.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-30 15:00:00 | 1876.35 | 2023-12-05 09:15:00 | 2063.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-19 10:30:00 | 2258.40 | 2024-01-20 14:15:00 | 2285.15 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-01-20 09:45:00 | 2253.75 | 2024-01-20 14:15:00 | 2285.15 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-01-20 12:30:00 | 2260.55 | 2024-01-20 14:15:00 | 2285.15 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-02-07 14:15:00 | 2509.90 | 2024-02-07 14:15:00 | 2506.35 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-02-08 13:00:00 | 2532.00 | 2024-02-08 13:15:00 | 2509.05 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-02-09 09:15:00 | 2548.65 | 2024-02-21 15:15:00 | 2660.00 | STOP_HIT | 1.00 | 4.37% |
| SELL | retest2 | 2024-03-07 10:30:00 | 2622.65 | 2024-03-11 09:15:00 | 2667.70 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-03-07 12:00:00 | 2623.45 | 2024-03-11 09:15:00 | 2667.70 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-03-12 09:15:00 | 2609.00 | 2024-03-13 09:15:00 | 2489.57 | PARTIAL | 0.50 | 4.58% |
| SELL | retest2 | 2024-03-12 10:30:00 | 2620.60 | 2024-03-13 10:15:00 | 2478.55 | PARTIAL | 0.50 | 5.42% |
| SELL | retest2 | 2024-03-12 09:15:00 | 2609.00 | 2024-03-13 14:15:00 | 2358.54 | TARGET_HIT | 0.50 | 9.60% |
| SELL | retest2 | 2024-03-12 10:30:00 | 2620.60 | 2024-03-14 09:15:00 | 2348.10 | TARGET_HIT | 0.50 | 10.40% |
| SELL | retest2 | 2024-03-15 10:15:00 | 2448.40 | 2024-03-21 13:15:00 | 2440.25 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2024-03-18 09:30:00 | 2449.90 | 2024-03-21 13:15:00 | 2440.25 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-03-18 10:30:00 | 2429.35 | 2024-03-21 13:15:00 | 2440.25 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-03-28 09:15:00 | 2469.95 | 2024-04-05 12:15:00 | 2577.55 | STOP_HIT | 1.00 | 4.36% |
| SELL | retest2 | 2024-04-18 10:45:00 | 2446.25 | 2024-04-23 13:15:00 | 2456.15 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-04-18 13:30:00 | 2446.50 | 2024-04-23 13:15:00 | 2456.15 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-04-23 12:30:00 | 2441.35 | 2024-04-23 13:15:00 | 2456.15 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-04-29 12:30:00 | 2558.80 | 2024-05-02 13:15:00 | 2528.10 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-05-07 11:15:00 | 2449.90 | 2024-05-14 12:15:00 | 2452.75 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-05-22 12:15:00 | 2537.00 | 2024-05-28 12:15:00 | 2573.45 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2024-05-22 13:15:00 | 2537.30 | 2024-05-28 12:15:00 | 2573.45 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2024-05-23 11:00:00 | 2544.95 | 2024-05-28 12:15:00 | 2573.45 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2024-06-19 11:15:00 | 2622.75 | 2024-06-19 13:15:00 | 2629.35 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2024-06-21 13:00:00 | 2603.35 | 2024-06-26 10:15:00 | 2633.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-06-21 14:15:00 | 2600.50 | 2024-06-26 10:15:00 | 2633.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-06-21 15:00:00 | 2587.85 | 2024-06-26 10:15:00 | 2633.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-07-01 09:15:00 | 2633.00 | 2024-07-05 09:15:00 | 2721.40 | STOP_HIT | 1.00 | 3.36% |
| SELL | retest2 | 2024-07-10 09:15:00 | 2673.00 | 2024-07-12 10:15:00 | 2677.35 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-07-11 10:15:00 | 2672.00 | 2024-07-12 10:15:00 | 2677.35 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-07-11 11:15:00 | 2675.15 | 2024-07-12 10:15:00 | 2677.35 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-07-11 12:00:00 | 2677.00 | 2024-07-12 10:15:00 | 2677.35 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-07-16 13:15:00 | 2713.25 | 2024-07-18 10:15:00 | 2679.05 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-07-16 14:15:00 | 2710.40 | 2024-07-18 10:15:00 | 2679.05 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-23 12:15:00 | 2602.40 | 2024-07-29 09:15:00 | 2640.75 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-07-23 13:30:00 | 2634.75 | 2024-07-29 09:15:00 | 2640.75 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-07-24 10:15:00 | 2638.60 | 2024-07-29 09:15:00 | 2640.75 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-07-24 11:15:00 | 2635.05 | 2024-07-29 09:15:00 | 2640.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-08-08 09:15:00 | 2377.60 | 2024-08-16 15:15:00 | 2339.80 | STOP_HIT | 1.00 | 1.59% |
| BUY | retest2 | 2024-08-21 09:15:00 | 2339.40 | 2024-08-21 12:15:00 | 2322.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-09-06 11:30:00 | 2422.20 | 2024-09-18 12:15:00 | 2471.85 | STOP_HIT | 1.00 | 2.05% |
| BUY | retest2 | 2024-09-09 09:45:00 | 2421.70 | 2024-09-18 12:15:00 | 2471.85 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest2 | 2024-09-09 10:30:00 | 2419.00 | 2024-09-18 12:15:00 | 2471.85 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2024-10-01 13:00:00 | 2513.10 | 2024-10-03 11:15:00 | 2471.30 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-10-01 14:45:00 | 2510.20 | 2024-10-03 11:15:00 | 2471.30 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-10-03 10:45:00 | 2512.85 | 2024-10-03 11:15:00 | 2471.30 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-10-09 11:45:00 | 2363.50 | 2024-10-18 09:15:00 | 2245.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 13:00:00 | 2363.15 | 2024-10-18 09:15:00 | 2244.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-09 11:45:00 | 2363.50 | 2024-10-18 13:15:00 | 2281.95 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2024-10-09 13:00:00 | 2363.15 | 2024-10-18 13:15:00 | 2281.95 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2024-10-24 11:30:00 | 2266.70 | 2024-10-28 12:15:00 | 2293.05 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-10-24 15:00:00 | 2269.00 | 2024-10-28 12:15:00 | 2293.05 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-10-28 11:30:00 | 2260.60 | 2024-10-28 12:15:00 | 2293.05 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-11-21 09:15:00 | 1967.15 | 2024-11-21 09:15:00 | 1868.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-21 09:15:00 | 1967.15 | 2024-11-22 09:15:00 | 2068.10 | STOP_HIT | 0.50 | -5.13% |
| BUY | retest2 | 2024-12-05 11:30:00 | 2264.00 | 2024-12-10 11:15:00 | 2249.45 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-12-05 13:00:00 | 2264.80 | 2024-12-10 11:15:00 | 2249.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-12-05 14:15:00 | 2268.20 | 2024-12-10 11:15:00 | 2249.45 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-12-05 15:00:00 | 2268.80 | 2024-12-10 11:15:00 | 2249.45 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-12 13:15:00 | 2241.45 | 2024-12-16 14:15:00 | 2246.55 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-12-16 11:00:00 | 2241.20 | 2024-12-16 14:15:00 | 2246.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-12-16 13:15:00 | 2241.70 | 2024-12-16 14:15:00 | 2246.55 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-12-27 11:15:00 | 2085.45 | 2025-01-02 13:15:00 | 2078.25 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2024-12-30 11:30:00 | 2083.85 | 2025-01-02 13:15:00 | 2078.25 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-12-30 12:15:00 | 2083.00 | 2025-01-02 13:15:00 | 2078.25 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-01-08 15:15:00 | 2005.60 | 2025-01-13 09:15:00 | 1905.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 15:15:00 | 2005.60 | 2025-01-14 09:15:00 | 1907.20 | STOP_HIT | 0.50 | 4.91% |
| BUY | retest2 | 2025-01-17 14:15:00 | 2005.15 | 2025-01-22 12:15:00 | 1981.60 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-01-20 10:45:00 | 2003.35 | 2025-01-22 12:15:00 | 1981.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-01-20 12:30:00 | 2003.05 | 2025-01-22 12:15:00 | 1981.60 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-01-22 09:45:00 | 2008.30 | 2025-01-22 12:15:00 | 1981.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-01-27 11:15:00 | 2045.10 | 2025-01-27 14:15:00 | 1999.75 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-01-28 14:45:00 | 2005.70 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-01-29 10:30:00 | 2009.10 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-01-29 11:00:00 | 2006.35 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-01-29 11:45:00 | 2004.80 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-01-30 10:45:00 | 1966.25 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-01-30 13:15:00 | 1966.40 | 2025-01-31 15:15:00 | 2008.25 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-02-10 10:45:00 | 1972.85 | 2025-02-12 09:15:00 | 1874.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 10:45:00 | 1972.85 | 2025-02-12 11:15:00 | 1930.90 | STOP_HIT | 0.50 | 2.13% |
| SELL | retest2 | 2025-03-03 10:30:00 | 1783.00 | 2025-03-04 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-03-21 09:15:00 | 1899.80 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-03-21 10:30:00 | 1900.50 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-03-21 11:15:00 | 1901.30 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-03-21 12:00:00 | 1901.75 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-03-26 10:45:00 | 1954.00 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-03-27 14:15:00 | 1952.60 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-03-28 11:30:00 | 1950.20 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-04-01 09:15:00 | 1956.20 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-04-02 12:00:00 | 1964.65 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-04-04 15:00:00 | 1967.90 | 2025-04-07 09:15:00 | 1892.00 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1876.70 | 2025-05-09 09:15:00 | 1782.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1876.70 | 2025-05-09 15:15:00 | 1810.10 | STOP_HIT | 0.50 | 3.55% |
| BUY | retest2 | 2025-05-21 13:00:00 | 1941.70 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-05-21 14:15:00 | 1942.00 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-05-21 15:00:00 | 1942.20 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-05-22 09:45:00 | 1941.90 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-05-22 14:00:00 | 1937.50 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-05-22 14:30:00 | 1939.50 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-05-27 10:30:00 | 1937.00 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-05-27 11:15:00 | 1937.70 | 2025-05-27 11:15:00 | 1936.10 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-06-03 12:45:00 | 1872.10 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-03 13:30:00 | 1875.00 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-03 14:15:00 | 1875.00 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-04 09:15:00 | 1872.30 | 2025-06-05 12:15:00 | 1890.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-17 15:15:00 | 1857.00 | 2025-06-24 10:15:00 | 1858.80 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-06-18 11:15:00 | 1852.80 | 2025-06-24 10:15:00 | 1858.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-08 12:30:00 | 1965.50 | 2025-07-14 10:15:00 | 1971.40 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-07-15 12:30:00 | 1983.90 | 2025-07-16 13:15:00 | 1993.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-16 12:15:00 | 1985.00 | 2025-07-16 13:15:00 | 1993.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-21 10:30:00 | 1966.30 | 2025-07-25 09:15:00 | 1871.59 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2025-07-22 09:45:00 | 1969.00 | 2025-07-25 10:15:00 | 1867.98 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1970.10 | 2025-07-25 10:15:00 | 1870.55 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-07-21 10:30:00 | 1966.30 | 2025-07-31 12:15:00 | 1824.90 | STOP_HIT | 0.50 | 7.19% |
| SELL | retest2 | 2025-07-22 09:45:00 | 1969.00 | 2025-07-31 12:15:00 | 1824.90 | STOP_HIT | 0.50 | 7.32% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1970.10 | 2025-07-31 12:15:00 | 1824.90 | STOP_HIT | 0.50 | 7.37% |
| SELL | retest2 | 2025-08-13 11:45:00 | 1783.10 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-08-13 12:15:00 | 1785.00 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-08-13 13:15:00 | 1785.70 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-08-13 15:15:00 | 1785.20 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-08-14 11:15:00 | 1785.90 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-08-14 12:45:00 | 1785.80 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-08-14 13:45:00 | 1784.60 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-08-14 14:15:00 | 1785.10 | 2025-08-18 09:15:00 | 1840.10 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1825.20 | 2025-09-05 12:15:00 | 1827.50 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-09-03 09:45:00 | 1832.70 | 2025-09-05 12:15:00 | 1827.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1852.00 | 2025-09-11 12:15:00 | 1840.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-09-23 09:15:00 | 1891.20 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-09-23 10:00:00 | 1893.60 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-23 10:30:00 | 1890.80 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-23 13:15:00 | 1887.10 | 2025-09-24 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-08 14:30:00 | 1857.00 | 2025-10-14 11:15:00 | 1863.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-10-16 13:15:00 | 1856.90 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-10-17 10:15:00 | 1854.90 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-10-17 11:00:00 | 1853.90 | 2025-10-23 15:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-10-28 15:15:00 | 1865.50 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-10-30 12:15:00 | 1862.70 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-10-30 15:00:00 | 1864.00 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1862.00 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-31 12:15:00 | 1877.90 | 2025-11-04 09:15:00 | 1845.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1836.70 | 2025-11-10 12:15:00 | 1850.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-11-18 09:45:00 | 1840.60 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-11-18 11:30:00 | 1841.10 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-19 09:30:00 | 1840.40 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-11-19 13:00:00 | 1840.00 | 2025-11-20 10:15:00 | 1858.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-11-26 14:45:00 | 1880.10 | 2025-11-28 11:15:00 | 1856.10 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-03 10:30:00 | 1837.00 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.34% |
| SELL | retest2 | 2025-12-03 12:45:00 | 1838.00 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.39% |
| SELL | retest2 | 2025-12-03 13:45:00 | 1838.30 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.41% |
| SELL | retest2 | 2025-12-03 15:00:00 | 1837.20 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 3.35% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1786.10 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-12-12 10:15:00 | 1786.90 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-12-12 10:45:00 | 1784.80 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2025-12-12 11:15:00 | 1786.50 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2025-12-12 12:30:00 | 1783.80 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-12-15 10:15:00 | 1783.40 | 2025-12-22 11:15:00 | 1775.70 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest1 | 2026-01-02 09:15:00 | 1748.90 | 2026-01-06 10:15:00 | 1750.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2026-01-07 14:45:00 | 1758.00 | 2026-01-08 09:15:00 | 1742.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-07 15:15:00 | 1758.70 | 2026-01-08 09:15:00 | 1742.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1702.30 | 2026-01-13 11:15:00 | 1719.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-01-28 14:15:00 | 1680.00 | 2026-02-02 10:15:00 | 1600.27 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-28 15:00:00 | 1684.50 | 2026-02-02 11:15:00 | 1596.00 | PARTIAL | 0.50 | 5.25% |
| SELL | retest2 | 2026-01-29 14:00:00 | 1682.00 | 2026-02-02 11:15:00 | 1597.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 14:45:00 | 1679.70 | 2026-02-02 11:15:00 | 1595.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 14:15:00 | 1680.00 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2026-01-28 15:00:00 | 1684.50 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-01-29 14:00:00 | 1682.00 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2026-01-29 14:45:00 | 1679.70 | 2026-02-02 13:15:00 | 1630.80 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2026-02-01 15:00:00 | 1617.90 | 2026-02-03 11:15:00 | 1681.10 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2026-02-02 10:30:00 | 1616.00 | 2026-02-03 11:15:00 | 1681.10 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2026-02-02 14:30:00 | 1624.30 | 2026-02-03 11:15:00 | 1681.10 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2026-02-11 13:15:00 | 1695.00 | 2026-02-12 10:15:00 | 1681.30 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1633.00 | 2026-02-25 11:15:00 | 1624.60 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2026-02-19 11:45:00 | 1634.10 | 2026-02-25 11:15:00 | 1624.60 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2026-03-11 10:30:00 | 1474.30 | 2026-03-13 12:15:00 | 1400.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:30:00 | 1474.90 | 2026-03-13 12:15:00 | 1401.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:00:00 | 1474.30 | 2026-03-13 12:15:00 | 1400.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:30:00 | 1474.30 | 2026-03-17 13:15:00 | 1378.60 | STOP_HIT | 0.50 | 6.49% |
| SELL | retest2 | 2026-03-11 11:30:00 | 1474.90 | 2026-03-17 13:15:00 | 1378.60 | STOP_HIT | 0.50 | 6.53% |
| SELL | retest2 | 2026-03-11 13:00:00 | 1474.30 | 2026-03-17 13:15:00 | 1378.60 | STOP_HIT | 0.50 | 6.49% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1345.00 | 2026-03-25 09:15:00 | 1379.70 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-04-13 14:30:00 | 1410.40 | 2026-04-23 11:15:00 | 1426.60 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1427.10 | 2026-04-23 11:15:00 | 1426.60 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2026-05-04 13:15:00 | 1406.40 | 2026-05-07 09:15:00 | 1422.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-05-04 14:15:00 | 1396.50 | 2026-05-07 09:15:00 | 1422.40 | STOP_HIT | 1.00 | -1.85% |
