# Aavas Financiers Ltd. (AAVAS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1446.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 57 |
| ALERT1 | 37 |
| ALERT2 | 37 |
| ALERT2_SKIP | 18 |
| ALERT3 | 76 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 59 |
| PARTIAL | 13 |
| TARGET_HIT | 11 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 34 / 38
- **Target hits / Stop hits / Partials:** 11 / 48 / 13
- **Avg / median % per leg:** 2.23% / -0.28%
- **Sum % (uncompounded):** 160.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 8 | 33.3% | 7 | 17 | 0 | 1.75% | 42.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.29% | -3.9% |
| BUY @ 3rd Alert (retest2) | 21 | 8 | 38.1% | 7 | 14 | 0 | 2.19% | 46.0% |
| SELL (all) | 48 | 26 | 54.2% | 4 | 31 | 13 | 2.46% | 118.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 48 | 26 | 54.2% | 4 | 31 | 13 | 2.46% | 118.2% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.29% | -3.9% |
| retest2 (combined) | 69 | 34 | 49.3% | 11 | 45 | 13 | 2.38% | 164.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1855.00 | 1783.35 | 1777.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1872.90 | 1801.26 | 1786.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 1819.20 | 1833.64 | 1813.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:00:00 | 1819.20 | 1833.64 | 1813.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1810.10 | 1828.93 | 1813.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 1810.10 | 1828.93 | 1813.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1797.90 | 1822.72 | 1811.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 1797.90 | 1822.72 | 1811.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 1805.00 | 1809.92 | 1808.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:45:00 | 1816.10 | 1810.28 | 1808.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:00:00 | 1809.50 | 1815.16 | 1812.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 12:15:00 | 1804.40 | 1810.31 | 1810.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 12:15:00 | 1804.40 | 1810.31 | 1810.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 09:15:00 | 1788.00 | 1802.04 | 1806.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 15:15:00 | 1794.00 | 1792.18 | 1798.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:15:00 | 1796.00 | 1792.18 | 1798.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1798.50 | 1793.44 | 1798.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 13:45:00 | 1785.00 | 1789.57 | 1794.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 1821.20 | 1792.21 | 1794.43 | SL hit (close>static) qty=1.00 sl=1806.90 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1833.00 | 1800.36 | 1797.94 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1781.00 | 1799.09 | 1800.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 12:15:00 | 1779.70 | 1795.21 | 1798.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 1781.00 | 1779.85 | 1788.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 1781.00 | 1779.85 | 1788.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1781.00 | 1779.85 | 1788.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 1781.00 | 1779.85 | 1788.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1772.00 | 1777.84 | 1786.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:00:00 | 1765.10 | 1778.96 | 1782.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 1765.50 | 1776.27 | 1781.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:45:00 | 1762.80 | 1767.66 | 1774.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:45:00 | 1763.60 | 1767.15 | 1771.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1779.00 | 1770.02 | 1772.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 1794.30 | 1770.02 | 1772.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1801.00 | 1776.22 | 1774.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1801.00 | 1776.22 | 1774.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 1816.80 | 1784.34 | 1778.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 1799.70 | 1821.06 | 1813.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 1799.70 | 1821.06 | 1813.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1799.70 | 1821.06 | 1813.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 1800.00 | 1821.06 | 1813.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1790.20 | 1814.88 | 1811.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:45:00 | 1789.60 | 1814.88 | 1811.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 12:15:00 | 1790.20 | 1806.17 | 1807.56 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 1818.00 | 1806.76 | 1806.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 1829.60 | 1818.64 | 1813.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 1921.70 | 1922.74 | 1891.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 13:15:00 | 1899.60 | 1911.78 | 1895.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1899.60 | 1911.78 | 1895.65 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1847.80 | 1884.34 | 1888.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1832.20 | 1856.31 | 1871.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1826.40 | 1825.90 | 1842.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 1826.40 | 1825.90 | 1842.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1846.60 | 1830.04 | 1843.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1846.60 | 1830.04 | 1843.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1855.00 | 1835.03 | 1844.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1837.80 | 1835.03 | 1844.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 12:00:00 | 1839.10 | 1834.52 | 1841.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1877.30 | 1845.09 | 1845.21 | SL hit (close>static) qty=1.00 sl=1861.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 1893.80 | 1854.83 | 1849.63 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 1830.10 | 1845.58 | 1846.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 1825.10 | 1837.27 | 1842.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 1850.70 | 1830.79 | 1834.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 1850.70 | 1830.79 | 1834.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1850.70 | 1830.79 | 1834.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:45:00 | 1827.00 | 1833.27 | 1834.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 13:15:00 | 1826.50 | 1833.27 | 1834.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1827.30 | 1825.02 | 1829.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:00:00 | 1830.00 | 1826.02 | 1829.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1830.10 | 1826.83 | 1829.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 1830.10 | 1826.83 | 1829.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1828.40 | 1822.49 | 1826.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1828.40 | 1822.49 | 1826.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1830.80 | 1824.15 | 1826.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 1828.90 | 1824.15 | 1826.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1863.80 | 1832.08 | 1830.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1863.80 | 1832.08 | 1830.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 1872.10 | 1851.94 | 1842.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 1883.00 | 1892.87 | 1875.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 15:00:00 | 1883.00 | 1892.87 | 1875.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1879.00 | 1890.09 | 1875.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 1893.00 | 1893.93 | 1878.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-30 09:15:00 | 2082.30 | 2001.41 | 1958.53 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 1992.30 | 2027.91 | 2029.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 1964.80 | 2004.01 | 2016.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 1928.90 | 1925.31 | 1950.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 10:00:00 | 1928.90 | 1925.31 | 1950.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1961.90 | 1910.40 | 1927.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1961.90 | 1910.40 | 1927.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1971.90 | 1922.70 | 1931.15 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1992.20 | 1946.25 | 1940.99 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 1946.00 | 1954.61 | 1955.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 1931.70 | 1950.03 | 1953.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 1951.00 | 1935.32 | 1943.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 13:15:00 | 1951.00 | 1935.32 | 1943.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1951.00 | 1935.32 | 1943.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 1951.00 | 1935.32 | 1943.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 2006.10 | 1949.48 | 1949.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 2006.10 | 1949.48 | 1949.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 15:15:00 | 1999.00 | 1959.38 | 1954.05 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 1947.10 | 1960.72 | 1961.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 09:15:00 | 1931.00 | 1951.76 | 1957.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 1939.20 | 1927.63 | 1938.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1939.20 | 1927.63 | 1938.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1939.20 | 1927.63 | 1938.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 1918.00 | 1928.76 | 1933.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 1915.40 | 1918.77 | 1928.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:30:00 | 1917.20 | 1916.32 | 1919.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 1917.20 | 1916.32 | 1919.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1932.30 | 1919.52 | 1920.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 1932.30 | 1919.52 | 1920.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1923.40 | 1920.29 | 1920.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1919.30 | 1920.29 | 1920.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1923.00 | 1920.83 | 1920.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:45:00 | 1905.50 | 1912.89 | 1916.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 1900.30 | 1907.21 | 1912.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 1822.10 | 1838.78 | 1855.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 1819.63 | 1838.78 | 1855.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 1821.34 | 1838.78 | 1855.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 1821.34 | 1838.78 | 1855.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 1810.22 | 1833.25 | 1851.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 15:15:00 | 1805.28 | 1823.29 | 1843.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 1778.20 | 1768.64 | 1792.26 | SL hit (close>ema200) qty=0.50 sl=1768.64 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 14:15:00 | 1710.00 | 1694.95 | 1693.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 1734.00 | 1705.33 | 1698.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 14:15:00 | 1710.90 | 1714.11 | 1706.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 15:00:00 | 1710.90 | 1714.11 | 1706.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1714.00 | 1714.09 | 1707.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 1725.10 | 1714.09 | 1707.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 13:45:00 | 1726.30 | 1715.23 | 1710.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 14:15:00 | 1680.30 | 1711.85 | 1712.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 1680.30 | 1711.85 | 1712.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 1666.00 | 1692.97 | 1702.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1700.50 | 1675.49 | 1687.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1700.50 | 1675.49 | 1687.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1700.50 | 1675.49 | 1687.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1700.50 | 1675.49 | 1687.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1672.90 | 1674.97 | 1685.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 1670.90 | 1674.83 | 1682.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1702.50 | 1677.83 | 1676.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 1702.50 | 1677.83 | 1676.77 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 1660.80 | 1675.44 | 1677.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 1654.60 | 1671.27 | 1675.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 11:15:00 | 1651.80 | 1650.39 | 1659.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 12:00:00 | 1651.80 | 1650.39 | 1659.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1659.90 | 1652.95 | 1659.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 1659.90 | 1652.95 | 1659.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 1648.80 | 1652.12 | 1658.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 09:45:00 | 1635.50 | 1648.53 | 1655.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1553.72 | 1592.38 | 1616.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1535.70 | 1532.80 | 1554.18 | SL hit (close>ema200) qty=0.50 sl=1532.80 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1609.00 | 1552.89 | 1545.43 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 1583.10 | 1586.10 | 1586.41 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 1589.40 | 1586.76 | 1586.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1596.00 | 1588.81 | 1587.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 11:15:00 | 1586.50 | 1589.88 | 1588.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 1586.50 | 1589.88 | 1588.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1586.50 | 1589.88 | 1588.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 1586.50 | 1589.88 | 1588.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1600.70 | 1592.04 | 1589.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 1613.70 | 1601.15 | 1594.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 12:15:00 | 1640.90 | 1650.37 | 1651.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 1640.90 | 1650.37 | 1651.37 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 1661.30 | 1653.35 | 1652.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 1681.80 | 1667.63 | 1661.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 1664.40 | 1672.18 | 1666.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 1664.40 | 1672.18 | 1666.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1664.40 | 1672.18 | 1666.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 1665.30 | 1672.18 | 1666.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1664.20 | 1670.58 | 1666.48 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 1655.00 | 1662.95 | 1663.82 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 12:15:00 | 1670.40 | 1664.25 | 1663.88 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1640.40 | 1661.45 | 1662.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 1632.20 | 1651.91 | 1658.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1620.70 | 1593.93 | 1613.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1620.70 | 1593.93 | 1613.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1620.70 | 1593.93 | 1613.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:45:00 | 1593.90 | 1595.30 | 1611.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1597.00 | 1595.44 | 1610.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 1640.40 | 1618.94 | 1616.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 1640.40 | 1618.94 | 1616.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1652.40 | 1635.76 | 1629.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 1667.00 | 1667.00 | 1657.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:30:00 | 1667.00 | 1667.00 | 1657.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1672.00 | 1667.28 | 1659.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:30:00 | 1672.60 | 1668.55 | 1660.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1676.40 | 1669.14 | 1661.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1637.10 | 1661.59 | 1662.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 1637.10 | 1661.59 | 1662.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 1632.80 | 1655.83 | 1659.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 1636.60 | 1635.96 | 1645.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 10:15:00 | 1638.70 | 1635.96 | 1645.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1622.10 | 1628.71 | 1637.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1637.00 | 1628.71 | 1637.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1621.80 | 1627.33 | 1636.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:45:00 | 1611.00 | 1619.64 | 1627.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 1606.90 | 1602.46 | 1601.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1606.90 | 1602.46 | 1601.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1630.50 | 1608.91 | 1605.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 14:15:00 | 1688.60 | 1690.38 | 1668.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 14:30:00 | 1687.40 | 1690.38 | 1668.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1672.10 | 1686.34 | 1670.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 1672.10 | 1686.34 | 1670.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1658.90 | 1680.86 | 1669.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 1658.90 | 1680.86 | 1669.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1659.00 | 1676.48 | 1668.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 1653.10 | 1676.48 | 1668.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 15:15:00 | 1659.20 | 1663.90 | 1664.09 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 1665.70 | 1661.52 | 1661.19 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1651.00 | 1660.05 | 1660.62 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 1673.20 | 1660.76 | 1660.44 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1650.00 | 1662.36 | 1662.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 14:15:00 | 1646.60 | 1658.46 | 1660.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 15:15:00 | 1654.00 | 1646.41 | 1651.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 15:15:00 | 1654.00 | 1646.41 | 1651.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1654.00 | 1646.41 | 1651.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 1622.60 | 1646.41 | 1651.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1659.90 | 1600.21 | 1608.04 | SL hit (close>static) qty=1.00 sl=1654.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1666.10 | 1613.39 | 1613.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1683.40 | 1639.05 | 1632.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 1716.50 | 1727.73 | 1705.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 10:00:00 | 1716.50 | 1727.73 | 1705.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1704.90 | 1723.16 | 1705.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 1703.40 | 1723.16 | 1705.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1718.20 | 1722.17 | 1706.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 1719.90 | 1722.17 | 1706.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 1700.80 | 1713.76 | 1708.01 | SL hit (close<static) qty=1.00 sl=1702.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 11:15:00 | 1672.60 | 1699.53 | 1702.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1657.30 | 1685.27 | 1693.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 14:15:00 | 1674.10 | 1673.76 | 1683.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 14:45:00 | 1677.30 | 1673.76 | 1683.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1664.00 | 1633.51 | 1635.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1664.00 | 1633.51 | 1635.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1650.00 | 1636.81 | 1637.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:45:00 | 1635.70 | 1634.25 | 1635.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 12:15:00 | 1553.91 | 1573.62 | 1590.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-03 12:15:00 | 1472.13 | 1493.72 | 1516.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 13:15:00 | 1498.00 | 1491.66 | 1491.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 14:15:00 | 1499.30 | 1493.19 | 1492.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 1515.50 | 1532.92 | 1523.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 15:15:00 | 1515.50 | 1532.92 | 1523.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1515.50 | 1532.92 | 1523.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1537.90 | 1532.92 | 1523.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1541.90 | 1534.71 | 1525.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 1547.00 | 1534.71 | 1525.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:30:00 | 1550.40 | 1556.12 | 1551.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1501.00 | 1542.30 | 1546.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 1501.00 | 1542.30 | 1546.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 1476.40 | 1522.35 | 1536.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 1457.00 | 1450.29 | 1463.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 1457.00 | 1450.29 | 1463.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1465.00 | 1453.24 | 1463.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1463.40 | 1453.24 | 1463.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1460.80 | 1454.75 | 1463.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 1455.50 | 1455.42 | 1462.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:00:00 | 1456.00 | 1455.79 | 1461.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:00:00 | 1455.60 | 1455.26 | 1460.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 12:15:00 | 1475.30 | 1464.47 | 1463.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 12:15:00 | 1475.30 | 1464.47 | 1463.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 09:15:00 | 1497.10 | 1478.37 | 1471.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 11:15:00 | 1496.00 | 1499.55 | 1489.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:45:00 | 1498.30 | 1499.55 | 1489.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1480.00 | 1494.81 | 1489.81 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1457.10 | 1484.90 | 1486.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 1445.10 | 1458.19 | 1463.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 15:15:00 | 1445.00 | 1443.16 | 1451.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:15:00 | 1439.00 | 1443.16 | 1451.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1440.90 | 1439.90 | 1444.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:45:00 | 1443.80 | 1439.90 | 1444.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1451.90 | 1442.74 | 1445.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:45:00 | 1452.60 | 1442.74 | 1445.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1456.00 | 1445.40 | 1446.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 1456.00 | 1445.40 | 1446.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 13:15:00 | 1465.50 | 1449.42 | 1448.10 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1441.50 | 1454.22 | 1454.76 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 1456.20 | 1448.36 | 1447.65 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 13:15:00 | 1443.00 | 1446.56 | 1446.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 09:15:00 | 1435.80 | 1443.60 | 1445.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 13:15:00 | 1391.40 | 1384.73 | 1397.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 14:00:00 | 1391.40 | 1384.73 | 1397.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 1388.00 | 1385.39 | 1396.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:15:00 | 1380.00 | 1385.39 | 1396.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1380.00 | 1384.31 | 1394.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 1402.50 | 1384.31 | 1394.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1407.80 | 1389.01 | 1396.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1404.10 | 1389.01 | 1396.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1405.70 | 1392.35 | 1396.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 1410.00 | 1392.35 | 1396.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 1431.00 | 1402.57 | 1400.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 1436.90 | 1409.43 | 1404.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 1465.00 | 1469.35 | 1447.14 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1495.40 | 1469.35 | 1447.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 11:00:00 | 1486.40 | 1476.71 | 1454.58 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 14:30:00 | 1488.70 | 1482.79 | 1464.84 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1483.20 | 1483.23 | 1468.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:30:00 | 1489.80 | 1484.52 | 1470.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:00:00 | 1489.70 | 1484.52 | 1470.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 1471.00 | 1489.02 | 1480.15 | SL hit (close<ema400) qty=1.00 sl=1480.15 alert=retest1 |

### Cycle 48 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 1462.50 | 1473.56 | 1474.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 1450.00 | 1465.35 | 1469.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 11:15:00 | 1466.30 | 1463.68 | 1467.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1466.30 | 1463.68 | 1467.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1466.30 | 1463.68 | 1467.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1466.30 | 1463.68 | 1467.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1465.00 | 1463.94 | 1467.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1473.50 | 1463.94 | 1467.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1457.60 | 1462.68 | 1466.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:15:00 | 1453.40 | 1462.68 | 1466.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 1456.10 | 1442.63 | 1449.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 09:45:00 | 1453.00 | 1442.64 | 1449.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 11:15:00 | 1383.29 | 1405.17 | 1418.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 12:15:00 | 1380.73 | 1400.56 | 1415.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 12:15:00 | 1380.35 | 1400.56 | 1415.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-06 09:15:00 | 1308.06 | 1375.16 | 1398.18 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1304.40 | 1301.46 | 1301.24 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1295.80 | 1301.83 | 1302.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 1292.10 | 1299.89 | 1301.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 1287.70 | 1277.95 | 1283.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 1287.70 | 1277.95 | 1283.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1287.70 | 1277.95 | 1283.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 1287.70 | 1277.95 | 1283.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1280.00 | 1278.36 | 1283.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1261.00 | 1278.36 | 1283.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 1293.40 | 1274.46 | 1275.18 | SL hit (close>static) qty=1.00 sl=1288.10 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 1282.90 | 1276.15 | 1275.88 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1244.70 | 1277.86 | 1280.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 10:15:00 | 1216.40 | 1231.05 | 1243.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 1226.20 | 1226.13 | 1235.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:15:00 | 1233.00 | 1226.13 | 1235.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1234.40 | 1229.57 | 1234.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 1234.40 | 1229.57 | 1234.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1234.00 | 1230.45 | 1234.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:30:00 | 1232.40 | 1230.45 | 1234.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1228.50 | 1230.06 | 1233.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1169.50 | 1230.06 | 1233.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 1212.10 | 1212.69 | 1213.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:15:00 | 1151.49 | 1169.98 | 1185.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 14:15:00 | 1172.40 | 1165.43 | 1178.94 | SL hit (close>ema200) qty=0.50 sl=1165.43 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 1117.50 | 1096.34 | 1094.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1152.50 | 1113.06 | 1103.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1113.90 | 1126.57 | 1117.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1113.90 | 1126.57 | 1117.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1113.90 | 1126.57 | 1117.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1113.90 | 1126.57 | 1117.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1118.80 | 1125.02 | 1117.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 1111.60 | 1125.02 | 1117.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1112.70 | 1122.55 | 1117.05 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1080.80 | 1108.31 | 1111.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 1076.40 | 1092.91 | 1101.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1118.70 | 1098.07 | 1103.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1118.70 | 1098.07 | 1103.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1118.70 | 1098.07 | 1103.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:15:00 | 1117.00 | 1098.07 | 1103.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1117.20 | 1101.89 | 1104.59 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 1126.00 | 1109.72 | 1107.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 1135.40 | 1116.92 | 1111.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1117.40 | 1117.62 | 1112.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 1117.40 | 1117.62 | 1112.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1117.40 | 1117.62 | 1112.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:15:00 | 1133.10 | 1120.36 | 1115.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:45:00 | 1140.90 | 1127.63 | 1121.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:30:00 | 1134.60 | 1136.95 | 1132.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1145.20 | 1131.58 | 1130.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 15:15:00 | 1246.41 | 1233.32 | 1212.66 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 10:15:00 | 1387.90 | 1403.68 | 1403.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 1371.10 | 1387.43 | 1394.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 14:15:00 | 1377.30 | 1370.89 | 1382.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 1377.30 | 1370.89 | 1382.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1365.00 | 1370.74 | 1380.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1358.30 | 1377.54 | 1379.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:30:00 | 1361.20 | 1369.15 | 1374.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 1381.10 | 1377.20 | 1376.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1381.10 | 1377.20 | 1376.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 1395.30 | 1384.96 | 1381.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 15:15:00 | 1422.50 | 1432.55 | 1418.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 15:15:00 | 1422.50 | 1432.55 | 1418.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1422.50 | 1432.55 | 1418.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 1440.20 | 1432.55 | 1418.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 1416.60 | 1427.51 | 1418.18 | SL hit (close<static) qty=1.00 sl=1417.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 12:45:00 | 1816.10 | 2025-05-15 12:15:00 | 1804.40 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-05-15 10:00:00 | 1809.50 | 2025-05-15 12:15:00 | 1804.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-05-19 13:45:00 | 1785.00 | 2025-05-20 09:15:00 | 1821.20 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-05-26 10:00:00 | 1765.10 | 2025-05-28 09:15:00 | 1801.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-05-26 11:00:00 | 1765.50 | 2025-05-28 09:15:00 | 1801.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-05-27 09:45:00 | 1762.80 | 2025-05-28 09:15:00 | 1801.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-05-27 13:45:00 | 1763.60 | 2025-05-28 09:15:00 | 1801.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1837.80 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-06-16 12:00:00 | 1839.10 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-19 12:45:00 | 1827.00 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-06-19 13:15:00 | 1826.50 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1827.30 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-06-20 13:00:00 | 1830.00 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-06-26 09:30:00 | 1893.00 | 2025-06-30 09:15:00 | 2082.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-18 10:15:00 | 1918.00 | 2025-07-28 12:15:00 | 1822.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1915.40 | 2025-07-28 12:15:00 | 1819.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 13:30:00 | 1917.20 | 2025-07-28 12:15:00 | 1821.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 14:15:00 | 1917.20 | 2025-07-28 12:15:00 | 1821.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 14:45:00 | 1905.50 | 2025-07-28 13:15:00 | 1810.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 10:30:00 | 1900.30 | 2025-07-28 15:15:00 | 1805.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:15:00 | 1918.00 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 7.29% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1915.40 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 7.16% |
| SELL | retest2 | 2025-07-21 13:30:00 | 1917.20 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 7.25% |
| SELL | retest2 | 2025-07-21 14:15:00 | 1917.20 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 7.25% |
| SELL | retest2 | 2025-07-22 14:45:00 | 1905.50 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 6.68% |
| SELL | retest2 | 2025-07-23 10:30:00 | 1900.30 | 2025-07-30 11:15:00 | 1778.20 | STOP_HIT | 0.50 | 6.43% |
| BUY | retest2 | 2025-08-12 09:15:00 | 1725.10 | 2025-08-13 14:15:00 | 1680.30 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-08-12 13:45:00 | 1726.30 | 2025-08-13 14:15:00 | 1680.30 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-08-18 14:45:00 | 1670.90 | 2025-08-20 09:15:00 | 1702.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-08-25 09:45:00 | 1635.50 | 2025-08-28 09:15:00 | 1553.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 09:45:00 | 1635.50 | 2025-09-01 09:15:00 | 1535.70 | STOP_HIT | 0.50 | 6.10% |
| BUY | retest2 | 2025-09-11 09:30:00 | 1613.70 | 2025-09-18 12:15:00 | 1640.90 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2025-09-29 10:45:00 | 1593.90 | 2025-09-30 10:15:00 | 1640.40 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1597.00 | 2025-09-30 10:15:00 | 1640.40 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-10-07 14:30:00 | 1672.60 | 2025-10-09 09:15:00 | 1637.10 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1676.40 | 2025-10-09 09:15:00 | 1637.10 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-10-14 09:45:00 | 1611.00 | 2025-10-17 13:15:00 | 1606.90 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1622.60 | 2025-11-10 10:15:00 | 1659.90 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-11-14 12:15:00 | 1719.90 | 2025-11-17 09:15:00 | 1700.80 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-11-17 09:45:00 | 1719.30 | 2025-11-17 10:15:00 | 1676.30 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-11-25 09:45:00 | 1635.70 | 2025-11-28 12:15:00 | 1553.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 09:45:00 | 1635.70 | 2025-12-03 12:15:00 | 1472.13 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-11 10:15:00 | 1547.00 | 2025-12-16 09:15:00 | 1501.00 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-12-15 11:30:00 | 1550.40 | 2025-12-16 09:15:00 | 1501.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-12-22 11:15:00 | 1455.50 | 2025-12-23 12:15:00 | 1475.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-12-22 13:00:00 | 1456.00 | 2025-12-23 12:15:00 | 1475.30 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-22 15:00:00 | 1455.60 | 2025-12-23 12:15:00 | 1475.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest1 | 2026-01-27 09:15:00 | 1495.40 | 2026-01-29 09:15:00 | 1471.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest1 | 2026-01-27 11:00:00 | 1486.40 | 2026-01-29 09:15:00 | 1471.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest1 | 2026-01-27 14:30:00 | 1488.70 | 2026-01-29 09:15:00 | 1471.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-28 10:30:00 | 1489.80 | 2026-01-29 13:15:00 | 1462.50 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-01-28 11:00:00 | 1489.70 | 2026-01-29 13:15:00 | 1462.50 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-01 14:15:00 | 1453.40 | 2026-02-05 11:15:00 | 1383.29 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2026-02-03 09:15:00 | 1456.10 | 2026-02-05 12:15:00 | 1380.73 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2026-02-03 09:45:00 | 1453.00 | 2026-02-05 12:15:00 | 1380.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 14:15:00 | 1453.40 | 2026-02-06 09:15:00 | 1308.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 09:15:00 | 1456.10 | 2026-02-06 09:15:00 | 1310.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 09:45:00 | 1453.00 | 2026-02-06 09:15:00 | 1307.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1261.00 | 2026-02-25 14:15:00 | 1293.40 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1169.50 | 2026-03-13 11:15:00 | 1151.49 | PARTIAL | 0.50 | 1.54% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1169.50 | 2026-03-13 14:15:00 | 1172.40 | STOP_HIT | 0.50 | -0.25% |
| SELL | retest2 | 2026-03-11 09:15:00 | 1212.10 | 2026-03-17 09:15:00 | 1111.02 | PARTIAL | 0.50 | 8.34% |
| SELL | retest2 | 2026-03-11 09:15:00 | 1212.10 | 2026-03-18 14:15:00 | 1124.60 | STOP_HIT | 0.50 | 7.22% |
| BUY | retest2 | 2026-04-02 14:15:00 | 1133.10 | 2026-04-10 15:15:00 | 1246.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:45:00 | 1140.90 | 2026-04-13 09:15:00 | 1254.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 12:30:00 | 1134.60 | 2026-04-13 09:15:00 | 1248.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1145.20 | 2026-04-13 09:15:00 | 1259.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 15:00:00 | 1285.20 | 2026-04-21 10:15:00 | 1413.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 10:00:00 | 1289.00 | 2026-04-21 11:15:00 | 1417.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1358.30 | 2026-05-04 10:15:00 | 1381.10 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-04-30 11:30:00 | 1361.20 | 2026-05-04 10:15:00 | 1381.10 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-05-07 09:15:00 | 1440.20 | 2026-05-07 10:15:00 | 1416.60 | STOP_HIT | 1.00 | -1.64% |
