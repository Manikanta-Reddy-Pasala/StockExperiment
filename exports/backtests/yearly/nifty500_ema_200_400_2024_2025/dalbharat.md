# Dalmia Bharat Ltd. (DALBHARAT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1840.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT2_SKIP | 6 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 23
- **Target hits / Stop hits / Partials:** 0 / 23 / 1
- **Avg / median % per leg:** -1.62% / -1.68%
- **Sum % (uncompounded):** -38.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.80% | -16.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.80% | -16.2% |
| SELL (all) | 15 | 1 | 6.7% | 0 | 14 | 1 | -1.52% | -22.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 1 | 6.7% | 0 | 14 | 1 | -1.52% | -22.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 1 | 4.2% | 0 | 23 | 1 | -1.62% | -38.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 14:15:00 | 1913.35 | 1831.05 | 1830.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 1931.75 | 1853.09 | 1844.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 12:15:00 | 1867.45 | 1887.27 | 1865.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 12:15:00 | 1867.45 | 1887.27 | 1865.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 1867.45 | 1887.27 | 1865.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:00:00 | 1867.45 | 1887.27 | 1865.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 1853.85 | 1886.94 | 1865.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 14:00:00 | 1853.85 | 1886.94 | 1865.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1852.05 | 1886.59 | 1865.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 14:30:00 | 1849.00 | 1886.59 | 1865.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1897.25 | 1886.54 | 1866.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 10:15:00 | 1900.15 | 1886.54 | 1866.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 15:15:00 | 1905.00 | 1886.68 | 1866.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 12:15:00 | 1852.85 | 1886.26 | 1866.90 | SL hit (close<static) qty=1.00 sl=1853.75 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 1789.10 | 1856.45 | 1856.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 14:15:00 | 1771.95 | 1854.15 | 1855.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 12:15:00 | 1846.50 | 1844.89 | 1850.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 13:00:00 | 1846.50 | 1844.89 | 1850.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 1846.35 | 1844.91 | 1850.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 1846.35 | 1844.91 | 1850.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 1843.95 | 1844.90 | 1850.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:45:00 | 1848.95 | 1844.90 | 1850.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 1840.55 | 1844.62 | 1850.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:30:00 | 1840.10 | 1844.62 | 1850.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1836.55 | 1844.54 | 1850.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 1836.55 | 1844.54 | 1850.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1840.60 | 1844.13 | 1849.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1840.60 | 1844.13 | 1849.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1816.65 | 1796.21 | 1819.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 1816.65 | 1796.21 | 1819.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1811.70 | 1796.36 | 1819.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:30:00 | 1818.45 | 1796.36 | 1819.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1807.00 | 1796.65 | 1819.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 15:15:00 | 1806.00 | 1796.65 | 1819.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 1853.30 | 1797.31 | 1819.38 | SL hit (close>static) qty=1.00 sl=1819.75 alert=retest2 |

### Cycle 3 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 1925.85 | 1833.76 | 1833.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 1949.40 | 1845.03 | 1839.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 1857.05 | 1873.14 | 1856.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 1857.05 | 1873.14 | 1856.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1857.05 | 1873.14 | 1856.03 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 12:15:00 | 1734.95 | 1841.85 | 1842.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 1724.95 | 1814.37 | 1825.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 09:15:00 | 1791.80 | 1782.86 | 1805.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 09:45:00 | 1798.95 | 1782.86 | 1805.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1798.50 | 1783.33 | 1804.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:45:00 | 1799.55 | 1783.33 | 1804.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1804.10 | 1783.66 | 1804.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1804.10 | 1783.66 | 1804.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1818.50 | 1784.00 | 1804.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 1821.85 | 1784.00 | 1804.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1814.00 | 1784.30 | 1804.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 10:15:00 | 1796.00 | 1787.42 | 1805.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 11:30:00 | 1797.80 | 1787.65 | 1805.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 12:00:00 | 1797.75 | 1787.65 | 1805.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 14:15:00 | 1795.00 | 1787.97 | 1805.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 1787.65 | 1788.15 | 1805.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 1828.00 | 1788.96 | 1805.18 | SL hit (close>static) qty=1.00 sl=1826.65 alert=retest2 |

### Cycle 5 — BUY (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 15:15:00 | 1870.00 | 1817.03 | 1816.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 10:15:00 | 1872.55 | 1818.00 | 1817.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 11:15:00 | 1815.35 | 1819.91 | 1818.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 11:15:00 | 1815.35 | 1819.91 | 1818.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1815.35 | 1819.91 | 1818.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 1815.35 | 1819.91 | 1818.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 1818.55 | 1819.90 | 1818.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 13:15:00 | 1822.25 | 1819.90 | 1818.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 14:15:00 | 1822.50 | 1819.90 | 1818.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 1806.05 | 1819.83 | 1818.42 | SL hit (close<static) qty=1.00 sl=1810.80 alert=retest2 |

### Cycle 6 — SELL (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 12:15:00 | 1803.70 | 1816.99 | 1817.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 1776.60 | 1815.39 | 1816.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 1748.00 | 1724.13 | 1757.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 12:00:00 | 1748.00 | 1724.13 | 1757.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 1744.50 | 1722.48 | 1754.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:30:00 | 1749.05 | 1722.48 | 1754.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1753.80 | 1723.04 | 1754.32 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 15:15:00 | 1811.35 | 1772.66 | 1772.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1853.00 | 1773.46 | 1772.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 2048.40 | 2056.71 | 1991.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 15:15:00 | 2042.90 | 2056.71 | 1991.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2293.10 | 2369.34 | 2304.37 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 2184.00 | 2270.17 | 2270.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 09:15:00 | 2150.00 | 2268.98 | 2269.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 2049.30 | 2030.56 | 2093.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:45:00 | 2056.00 | 2030.56 | 2093.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 2090.20 | 2034.30 | 2092.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 2091.40 | 2034.30 | 2092.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 2093.80 | 2034.89 | 2092.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 2081.90 | 2036.90 | 2092.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 2104.70 | 2038.65 | 2082.97 | SL hit (close>static) qty=1.00 sl=2101.90 alert=retest2 |

### Cycle 9 — BUY (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 15:15:00 | 2179.70 | 2105.61 | 2105.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 09:15:00 | 2198.80 | 2106.54 | 2105.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 2113.10 | 2116.62 | 2111.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 2113.10 | 2116.62 | 2111.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2113.10 | 2116.62 | 2111.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:15:00 | 2114.50 | 2116.62 | 2111.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 2133.00 | 2116.78 | 2111.28 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 2048.30 | 2106.44 | 2106.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2034.40 | 2105.47 | 2106.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 2098.80 | 2097.89 | 2102.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 11:15:00 | 2098.80 | 2097.89 | 2102.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 2098.80 | 2097.89 | 2102.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:45:00 | 2102.50 | 2097.89 | 2102.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 2103.60 | 2097.95 | 2102.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 13:00:00 | 2103.60 | 2097.95 | 2102.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 13:15:00 | 2111.20 | 2098.08 | 2102.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 13:45:00 | 2106.00 | 2098.08 | 2102.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 2110.10 | 2098.20 | 2102.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:15:00 | 2105.70 | 2098.20 | 2102.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 2105.70 | 2098.27 | 2102.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 2136.30 | 2098.27 | 2102.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2133.40 | 2098.62 | 2102.43 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 2162.10 | 2106.32 | 2106.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 2176.80 | 2107.02 | 2106.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 2116.10 | 2125.00 | 2116.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 13:15:00 | 2116.10 | 2125.00 | 2116.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 2116.10 | 2125.00 | 2116.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 2116.10 | 2125.00 | 2116.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 2115.00 | 2124.90 | 2116.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 2107.00 | 2124.90 | 2116.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 2116.10 | 2124.81 | 2116.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 2124.00 | 2124.81 | 2116.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2124.80 | 2124.63 | 2116.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 13:00:00 | 2131.90 | 2124.70 | 2116.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 2128.90 | 2124.79 | 2117.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 2129.90 | 2124.79 | 2117.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:45:00 | 2127.90 | 2124.91 | 2117.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 2107.50 | 2124.76 | 2117.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 2107.50 | 2124.76 | 2117.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 2100.20 | 2124.51 | 2117.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 2100.20 | 2124.51 | 2117.06 | SL hit (close<static) qty=1.00 sl=2106.20 alert=retest2 |

### Cycle 12 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 2046.80 | 2110.31 | 2110.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 2019.40 | 2106.15 | 2108.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1943.20 | 1889.82 | 1959.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 1943.20 | 1889.82 | 1959.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1947.00 | 1894.04 | 1956.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1920.00 | 1897.30 | 1956.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:00:00 | 1915.00 | 1897.47 | 1956.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1985.90 | 1901.11 | 1956.59 | SL hit (close>static) qty=1.00 sl=1957.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-08 10:15:00 | 1900.15 | 2024-10-09 12:15:00 | 1852.85 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-10-08 15:15:00 | 1905.00 | 2024-10-09 12:15:00 | 1852.85 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-10-14 11:45:00 | 1911.30 | 2024-10-17 09:15:00 | 1842.20 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2024-11-22 15:15:00 | 1806.00 | 2024-11-25 09:15:00 | 1853.30 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-11-27 10:45:00 | 1805.00 | 2024-11-27 11:15:00 | 1821.60 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-01-27 10:15:00 | 1796.00 | 2025-01-28 12:15:00 | 1828.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-01-27 11:30:00 | 1797.80 | 2025-01-28 12:15:00 | 1828.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-01-27 12:00:00 | 1797.75 | 2025-01-28 12:15:00 | 1828.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-01-27 14:15:00 | 1795.00 | 2025-01-28 12:15:00 | 1828.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-02-10 13:15:00 | 1822.25 | 2025-02-11 09:15:00 | 1806.05 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-02-10 14:15:00 | 1822.50 | 2025-02-11 09:15:00 | 1806.05 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-12-16 09:15:00 | 2081.90 | 2025-12-24 09:15:00 | 2104.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-08 11:00:00 | 2082.80 | 2026-01-13 12:15:00 | 2103.20 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-01-13 10:30:00 | 2083.90 | 2026-01-13 12:15:00 | 2103.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-02-18 13:00:00 | 2131.90 | 2026-02-19 12:15:00 | 2100.20 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-02-18 14:45:00 | 2128.90 | 2026-02-19 12:15:00 | 2100.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-19 09:15:00 | 2129.90 | 2026-02-19 12:15:00 | 2100.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-02-19 09:45:00 | 2127.90 | 2026-02-19 12:15:00 | 2100.20 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1920.00 | 2026-04-15 09:15:00 | 1985.90 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2026-04-13 10:00:00 | 1915.00 | 2026-04-15 09:15:00 | 1985.90 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-04-24 11:00:00 | 1924.90 | 2026-04-24 14:15:00 | 1961.30 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-04-28 13:30:00 | 1925.40 | 2026-04-29 10:15:00 | 1964.30 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1927.50 | 2026-05-04 10:15:00 | 1989.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2026-05-08 10:00:00 | 1918.00 | 2026-05-08 14:15:00 | 1822.10 | PARTIAL | 0.50 | 5.00% |
