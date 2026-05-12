# Sobha Ltd. (SOBHA)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 1425.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 134 |
| ALERT1 | 93 |
| ALERT2 | 93 |
| ALERT2_SKIP | 46 |
| ALERT3 | 265 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 120 |
| PARTIAL | 19 |
| TARGET_HIT | 5 |
| STOP_HIT | 119 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 141 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 85
- **Target hits / Stop hits / Partials:** 5 / 118 / 18
- **Avg / median % per leg:** 0.29% / -1.01%
- **Sum % (uncompounded):** 40.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 20 | 35.1% | 3 | 51 | 3 | 0.14% | 8.3% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.61% | 27.7% |
| BUY @ 3rd Alert (retest2) | 51 | 14 | 27.5% | 3 | 48 | 0 | -0.38% | -19.4% |
| SELL (all) | 84 | 36 | 42.9% | 2 | 67 | 15 | 0.39% | 32.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 84 | 36 | 42.9% | 2 | 67 | 15 | 0.39% | 32.7% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.61% | 27.7% |
| retest2 (combined) | 135 | 50 | 37.0% | 5 | 115 | 15 | 0.10% | 13.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 10:15:00 | 1728.30 | 1674.18 | 1668.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 13:15:00 | 1742.46 | 1702.06 | 1684.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 1721.83 | 1726.08 | 1706.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 1721.83 | 1726.08 | 1706.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 1681.97 | 1717.26 | 1704.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 1681.97 | 1717.26 | 1704.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1662.41 | 1706.29 | 1700.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 1662.41 | 1706.29 | 1700.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 1686.79 | 1702.39 | 1699.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 1708.25 | 1702.39 | 1699.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 12:15:00 | 1705.96 | 1729.48 | 1730.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 1705.96 | 1729.48 | 1730.64 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 11:15:00 | 1737.60 | 1730.65 | 1730.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 14:15:00 | 1752.00 | 1736.98 | 1733.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 14:15:00 | 1938.01 | 1947.52 | 1893.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 15:00:00 | 1938.01 | 1947.52 | 1893.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 1889.68 | 1924.30 | 1902.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:00:00 | 1889.68 | 1924.30 | 1902.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 1888.03 | 1917.05 | 1900.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 1888.03 | 1917.05 | 1900.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 1839.55 | 1883.76 | 1887.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 15:15:00 | 1815.27 | 1842.44 | 1862.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 14:15:00 | 1817.41 | 1808.18 | 1832.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 15:00:00 | 1817.41 | 1808.18 | 1832.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 1836.29 | 1813.80 | 1833.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 1843.25 | 1813.80 | 1833.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1810.26 | 1813.09 | 1830.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:15:00 | 1809.19 | 1813.09 | 1830.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1880.82 | 1843.62 | 1839.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1880.82 | 1843.62 | 1839.97 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1798.48 | 1846.56 | 1849.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1508.28 | 1778.91 | 1818.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 14:15:00 | 1764.61 | 1763.16 | 1800.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 15:00:00 | 1764.61 | 1763.16 | 1800.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 1756.72 | 1751.08 | 1779.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 1756.72 | 1751.08 | 1779.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1765.14 | 1753.89 | 1777.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 1765.14 | 1753.89 | 1777.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1767.19 | 1756.55 | 1776.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 1768.55 | 1756.55 | 1776.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 1786.07 | 1762.45 | 1777.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 1859.99 | 1762.45 | 1777.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1866.61 | 1783.29 | 1785.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 1866.86 | 1783.29 | 1785.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 1871.63 | 1800.95 | 1793.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 12:15:00 | 1875.57 | 1827.13 | 1807.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 13:15:00 | 2024.88 | 2027.07 | 1973.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:00:00 | 2024.88 | 2027.07 | 1973.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 2006.04 | 2012.36 | 1997.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 2132.24 | 2011.04 | 2003.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 10:15:00 | 2040.00 | 2081.79 | 2085.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 2040.00 | 2081.79 | 2085.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 11:15:00 | 2024.55 | 2070.34 | 2079.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 2053.15 | 2044.97 | 2061.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 2053.15 | 2044.97 | 2061.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 2053.15 | 2044.97 | 2061.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 14:45:00 | 2015.45 | 2041.41 | 2048.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 11:15:00 | 2052.55 | 2048.16 | 2047.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 11:15:00 | 2052.55 | 2048.16 | 2047.91 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 2029.45 | 2045.71 | 2047.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 2009.95 | 2038.56 | 2043.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 15:15:00 | 2023.35 | 2020.44 | 2030.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 09:15:00 | 2006.55 | 2020.44 | 2030.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 2033.95 | 2023.15 | 2031.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:00:00 | 2033.95 | 2023.15 | 2031.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 2022.60 | 2023.04 | 2030.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:45:00 | 2031.45 | 2023.04 | 2030.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 1978.00 | 1953.36 | 1975.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 1978.00 | 1953.36 | 1975.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1990.15 | 1960.72 | 1977.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:00:00 | 1990.15 | 1960.72 | 1977.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 1990.05 | 1966.58 | 1978.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:30:00 | 1990.00 | 1966.58 | 1978.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1963.00 | 1957.35 | 1966.05 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 12:15:00 | 2032.75 | 1978.36 | 1973.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 09:15:00 | 2058.10 | 2016.07 | 1994.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 14:15:00 | 2023.00 | 2031.69 | 2012.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 15:00:00 | 2023.00 | 2031.69 | 2012.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 2030.00 | 2031.35 | 2013.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 2045.00 | 2031.35 | 2013.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:00:00 | 2034.65 | 2032.01 | 2015.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:30:00 | 2038.45 | 2032.68 | 2017.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 14:45:00 | 2039.60 | 2033.10 | 2022.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 2026.85 | 2031.85 | 2023.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 2042.00 | 2031.85 | 2023.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 15:15:00 | 2014.85 | 2021.12 | 2021.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 15:15:00 | 2014.85 | 2021.12 | 2021.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 12:15:00 | 1980.00 | 2000.51 | 2007.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 13:15:00 | 1950.10 | 1929.09 | 1947.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 13:15:00 | 1950.10 | 1929.09 | 1947.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1950.10 | 1929.09 | 1947.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 1950.10 | 1929.09 | 1947.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1965.50 | 1936.37 | 1949.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 1913.50 | 1941.10 | 1950.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1900.85 | 1931.54 | 1938.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 1817.82 | 1864.97 | 1894.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 1805.81 | 1864.97 | 1894.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 09:15:00 | 1839.80 | 1820.05 | 1852.13 | SL hit (close>ema200) qty=0.50 sl=1820.05 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 11:15:00 | 1861.20 | 1844.99 | 1843.90 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 09:15:00 | 1800.70 | 1840.69 | 1843.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 1732.15 | 1778.13 | 1790.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 14:15:00 | 1751.15 | 1693.91 | 1722.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 14:15:00 | 1751.15 | 1693.91 | 1722.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 1751.15 | 1693.91 | 1722.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 15:00:00 | 1751.15 | 1693.91 | 1722.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 1704.00 | 1695.93 | 1720.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:15:00 | 1731.95 | 1695.93 | 1720.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1759.90 | 1708.72 | 1724.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:15:00 | 1762.90 | 1708.72 | 1724.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1758.75 | 1718.73 | 1727.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:15:00 | 1745.00 | 1728.41 | 1730.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:15:00 | 1747.00 | 1730.22 | 1730.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 15:15:00 | 1747.00 | 1733.57 | 1732.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 15:15:00 | 1747.00 | 1733.57 | 1732.42 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 15:15:00 | 1726.00 | 1732.38 | 1733.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 1693.50 | 1724.60 | 1729.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 1733.75 | 1706.23 | 1714.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 1733.75 | 1706.23 | 1714.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1733.75 | 1706.23 | 1714.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:30:00 | 1733.10 | 1706.23 | 1714.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 1732.00 | 1711.38 | 1715.80 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 1751.65 | 1719.44 | 1719.06 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 1706.55 | 1719.19 | 1719.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 1696.45 | 1708.95 | 1713.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 15:15:00 | 1695.00 | 1694.14 | 1702.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 09:15:00 | 1684.60 | 1694.14 | 1702.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1690.75 | 1693.46 | 1701.38 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 1720.95 | 1702.69 | 1701.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 1725.40 | 1715.73 | 1711.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 1709.05 | 1714.39 | 1711.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 10:15:00 | 1709.05 | 1714.39 | 1711.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 1709.05 | 1714.39 | 1711.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 1713.00 | 1714.39 | 1711.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1709.85 | 1713.48 | 1711.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:45:00 | 1716.85 | 1713.48 | 1711.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 1702.00 | 1711.19 | 1710.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:00:00 | 1702.00 | 1711.19 | 1710.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 1701.30 | 1709.21 | 1709.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 09:15:00 | 1687.60 | 1703.33 | 1706.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 10:15:00 | 1703.10 | 1692.95 | 1697.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 10:15:00 | 1703.10 | 1692.95 | 1697.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 1703.10 | 1692.95 | 1697.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 1703.10 | 1692.95 | 1697.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 1706.00 | 1695.56 | 1698.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 12:00:00 | 1706.00 | 1695.56 | 1698.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 1694.00 | 1695.25 | 1698.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 15:15:00 | 1693.00 | 1695.37 | 1697.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 10:45:00 | 1691.70 | 1696.15 | 1697.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 11:45:00 | 1688.00 | 1693.59 | 1696.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 1692.95 | 1688.87 | 1692.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1684.00 | 1687.89 | 1691.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-26 11:15:00 | 1708.80 | 1692.07 | 1693.23 | SL hit (close>static) qty=1.00 sl=1707.60 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 1726.45 | 1698.95 | 1696.25 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 1700.10 | 1711.26 | 1711.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 1696.50 | 1708.30 | 1710.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 1710.05 | 1708.65 | 1710.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 14:15:00 | 1710.05 | 1708.65 | 1710.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 1710.05 | 1708.65 | 1710.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 1710.05 | 1708.65 | 1710.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1708.60 | 1708.64 | 1710.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 1709.15 | 1708.64 | 1710.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1712.85 | 1709.48 | 1710.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 1712.85 | 1709.48 | 1710.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1713.60 | 1710.31 | 1710.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 1712.00 | 1710.31 | 1710.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 1690.50 | 1706.35 | 1708.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:30:00 | 1712.00 | 1706.35 | 1708.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 1709.00 | 1703.87 | 1707.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:00:00 | 1709.00 | 1703.87 | 1707.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1686.05 | 1700.31 | 1705.23 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 13:15:00 | 1720.10 | 1705.91 | 1705.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 14:15:00 | 1740.05 | 1712.74 | 1708.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 15:15:00 | 1731.00 | 1737.32 | 1727.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 15:15:00 | 1731.00 | 1737.32 | 1727.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1731.00 | 1737.32 | 1727.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 1724.60 | 1737.32 | 1727.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1744.65 | 1738.79 | 1728.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:45:00 | 1751.50 | 1742.78 | 1734.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 1758.75 | 1742.63 | 1735.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:45:00 | 1752.50 | 1744.14 | 1736.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 15:15:00 | 1723.10 | 1732.95 | 1733.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 15:15:00 | 1723.10 | 1732.95 | 1733.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 1697.35 | 1725.83 | 1730.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 13:15:00 | 1714.00 | 1713.37 | 1722.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 14:00:00 | 1714.00 | 1713.37 | 1722.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1687.05 | 1707.18 | 1716.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 11:45:00 | 1664.95 | 1693.81 | 1708.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 1732.05 | 1700.87 | 1699.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 1732.05 | 1700.87 | 1699.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 1739.00 | 1708.50 | 1703.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 10:15:00 | 1705.90 | 1708.00 | 1703.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 10:15:00 | 1705.90 | 1708.00 | 1703.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 1705.90 | 1708.00 | 1703.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:00:00 | 1705.90 | 1708.00 | 1703.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 1740.00 | 1743.68 | 1733.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 1824.00 | 1743.68 | 1733.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 10:15:00 | 1757.70 | 1769.32 | 1765.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 12:15:00 | 1752.10 | 1761.76 | 1762.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 1752.10 | 1761.76 | 1762.74 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 13:15:00 | 1812.85 | 1771.98 | 1767.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 14:15:00 | 1850.00 | 1787.58 | 1774.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 10:15:00 | 1840.00 | 1848.21 | 1824.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 10:45:00 | 1845.95 | 1848.21 | 1824.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1833.00 | 1845.17 | 1825.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 1818.00 | 1845.17 | 1825.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 1821.95 | 1840.53 | 1825.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 1821.95 | 1840.53 | 1825.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 1861.30 | 1844.68 | 1828.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 14:45:00 | 1875.15 | 1848.48 | 1831.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 1881.45 | 1850.18 | 1834.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:45:00 | 1876.65 | 1856.34 | 1838.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-23 09:15:00 | 2062.67 | 1953.18 | 1900.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1964.90 | 1990.79 | 1991.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 1956.20 | 1980.14 | 1985.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 1825.65 | 1824.52 | 1860.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:00:00 | 1825.65 | 1824.52 | 1860.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1752.00 | 1744.46 | 1763.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1749.55 | 1744.46 | 1763.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1749.95 | 1745.56 | 1762.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:30:00 | 1733.30 | 1743.77 | 1759.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 14:15:00 | 1726.90 | 1721.38 | 1720.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 1726.90 | 1721.38 | 1720.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 15:15:00 | 1729.50 | 1723.00 | 1721.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 1789.55 | 1796.84 | 1777.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 10:00:00 | 1789.55 | 1796.84 | 1777.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1778.15 | 1793.10 | 1777.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 1772.00 | 1793.10 | 1777.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 1769.55 | 1788.39 | 1776.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 1769.20 | 1788.39 | 1776.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 1773.05 | 1785.32 | 1776.22 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 1730.40 | 1764.85 | 1768.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 1672.95 | 1708.33 | 1728.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1676.95 | 1674.87 | 1697.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:15:00 | 1677.05 | 1674.87 | 1697.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 1533.20 | 1547.32 | 1573.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 1521.00 | 1541.97 | 1568.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 11:15:00 | 1584.10 | 1561.55 | 1564.88 | SL hit (close>static) qty=1.00 sl=1578.60 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 13:15:00 | 1572.75 | 1567.41 | 1567.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 14:15:00 | 1607.70 | 1575.47 | 1570.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 1576.60 | 1581.54 | 1574.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 10:00:00 | 1576.60 | 1581.54 | 1574.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 1570.15 | 1579.26 | 1574.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 1570.15 | 1579.26 | 1574.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1572.65 | 1577.94 | 1574.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:30:00 | 1578.80 | 1577.94 | 1574.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1572.80 | 1576.91 | 1574.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 1571.55 | 1576.91 | 1574.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 1570.45 | 1575.62 | 1573.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:30:00 | 1570.85 | 1575.62 | 1573.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1590.35 | 1578.57 | 1575.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 1605.75 | 1585.83 | 1579.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 11:15:00 | 1558.75 | 1579.64 | 1578.55 | SL hit (close<static) qty=1.00 sl=1569.60 alert=retest2 |

### Cycle 32 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 1566.80 | 1577.07 | 1577.48 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 1585.05 | 1571.98 | 1571.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1635.45 | 1584.67 | 1577.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 1649.40 | 1650.38 | 1625.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:30:00 | 1641.00 | 1650.38 | 1625.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1642.30 | 1648.61 | 1635.37 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 1603.10 | 1628.01 | 1628.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 1590.25 | 1620.46 | 1625.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 1621.65 | 1610.73 | 1618.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 1621.65 | 1610.73 | 1618.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 1621.65 | 1610.73 | 1618.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 1621.65 | 1610.73 | 1618.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1613.55 | 1611.29 | 1618.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 1618.55 | 1611.29 | 1618.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 1623.80 | 1613.79 | 1618.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:00:00 | 1623.80 | 1613.79 | 1618.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 1616.45 | 1614.32 | 1618.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 14:45:00 | 1614.00 | 1614.93 | 1618.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:15:00 | 1602.00 | 1614.93 | 1618.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 1634.15 | 1616.70 | 1618.54 | SL hit (close>static) qty=1.00 sl=1624.40 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 1632.20 | 1619.80 | 1619.78 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 1616.90 | 1619.66 | 1619.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 1594.80 | 1614.69 | 1617.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1567.50 | 1561.88 | 1580.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 1567.50 | 1561.88 | 1580.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1562.45 | 1550.55 | 1564.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 15:15:00 | 1518.00 | 1541.02 | 1554.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:30:00 | 1516.75 | 1538.64 | 1547.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 12:30:00 | 1517.50 | 1531.41 | 1541.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 14:00:00 | 1515.00 | 1528.13 | 1539.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1562.15 | 1531.61 | 1537.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 10:45:00 | 1536.10 | 1532.80 | 1537.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 11:15:00 | 1615.40 | 1549.32 | 1544.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 1615.40 | 1549.32 | 1544.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 10:15:00 | 1635.60 | 1614.22 | 1594.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 1650.00 | 1650.76 | 1636.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 12:30:00 | 1652.20 | 1650.76 | 1636.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1645.00 | 1649.12 | 1640.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:30:00 | 1647.40 | 1649.12 | 1640.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1649.40 | 1649.18 | 1640.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:00:00 | 1666.95 | 1653.23 | 1644.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 1670.00 | 1660.93 | 1650.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 1680.25 | 1654.20 | 1651.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 15:15:00 | 1673.00 | 1676.53 | 1676.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 15:15:00 | 1673.00 | 1676.53 | 1676.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 09:15:00 | 1654.00 | 1672.02 | 1674.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 13:15:00 | 1665.00 | 1662.19 | 1668.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 13:15:00 | 1665.00 | 1662.19 | 1668.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 1665.00 | 1662.19 | 1668.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 13:45:00 | 1665.45 | 1662.19 | 1668.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 1653.00 | 1660.35 | 1666.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:30:00 | 1657.10 | 1660.35 | 1666.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1660.00 | 1658.78 | 1664.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:15:00 | 1676.05 | 1658.78 | 1664.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1664.10 | 1659.85 | 1664.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 12:30:00 | 1655.10 | 1659.13 | 1663.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 1655.65 | 1662.73 | 1664.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:15:00 | 1572.34 | 1606.07 | 1622.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:15:00 | 1572.87 | 1606.07 | 1622.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-16 09:15:00 | 1598.65 | 1592.42 | 1607.02 | SL hit (close>ema200) qty=0.50 sl=1592.42 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 1655.80 | 1611.40 | 1608.13 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 1608.80 | 1615.39 | 1615.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 1588.20 | 1606.06 | 1610.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 12:15:00 | 1559.90 | 1559.74 | 1576.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 13:15:00 | 1580.95 | 1559.74 | 1576.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 1577.00 | 1563.19 | 1576.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 13:30:00 | 1573.70 | 1563.19 | 1576.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 1580.65 | 1566.69 | 1577.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 15:00:00 | 1580.65 | 1566.69 | 1577.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 15:15:00 | 1583.00 | 1569.95 | 1577.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:15:00 | 1561.65 | 1569.95 | 1577.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1585.50 | 1572.96 | 1577.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 1589.20 | 1572.96 | 1577.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 1596.10 | 1577.59 | 1579.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:00:00 | 1596.10 | 1577.59 | 1579.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 12:15:00 | 1595.00 | 1581.07 | 1580.78 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 11:15:00 | 1571.00 | 1585.05 | 1585.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 12:15:00 | 1570.05 | 1582.05 | 1583.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 13:15:00 | 1595.30 | 1584.70 | 1584.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 13:15:00 | 1595.30 | 1584.70 | 1584.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1595.30 | 1584.70 | 1584.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 1595.30 | 1584.70 | 1584.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 1605.10 | 1588.78 | 1586.81 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 1557.25 | 1583.31 | 1584.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 1542.75 | 1558.57 | 1569.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 12:15:00 | 1565.00 | 1555.78 | 1565.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 12:15:00 | 1565.00 | 1555.78 | 1565.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 1565.00 | 1555.78 | 1565.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:00:00 | 1565.00 | 1555.78 | 1565.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 1567.45 | 1558.11 | 1565.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:30:00 | 1573.05 | 1558.11 | 1565.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1577.00 | 1561.89 | 1566.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 15:00:00 | 1577.00 | 1561.89 | 1566.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 1573.45 | 1564.20 | 1566.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 1581.55 | 1564.20 | 1566.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 1571.20 | 1567.79 | 1568.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:15:00 | 1574.20 | 1567.79 | 1568.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 1570.70 | 1568.37 | 1568.49 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 1574.05 | 1569.51 | 1569.00 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 1552.55 | 1566.62 | 1567.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 10:15:00 | 1550.25 | 1563.35 | 1566.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 1510.90 | 1494.96 | 1510.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 13:15:00 | 1510.90 | 1494.96 | 1510.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1510.90 | 1494.96 | 1510.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 1510.90 | 1494.96 | 1510.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1512.25 | 1498.42 | 1511.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:45:00 | 1515.00 | 1498.42 | 1511.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 1510.00 | 1500.74 | 1510.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 1474.45 | 1500.74 | 1510.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 09:15:00 | 1400.73 | 1440.68 | 1469.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-10 09:15:00 | 1327.01 | 1387.03 | 1425.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 13:15:00 | 1321.00 | 1307.76 | 1306.84 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 1284.15 | 1305.15 | 1306.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 1281.10 | 1294.62 | 1300.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 10:15:00 | 1227.95 | 1225.34 | 1240.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 11:00:00 | 1227.95 | 1225.34 | 1240.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 12:15:00 | 1224.90 | 1173.00 | 1194.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 13:00:00 | 1224.90 | 1173.00 | 1194.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 1217.70 | 1181.94 | 1196.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 13:30:00 | 1233.50 | 1181.94 | 1196.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 1173.50 | 1182.62 | 1194.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 1182.75 | 1182.62 | 1194.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 1202.55 | 1186.61 | 1195.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 10:00:00 | 1202.55 | 1186.61 | 1195.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 1228.35 | 1194.96 | 1198.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 1228.35 | 1194.96 | 1198.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 1236.80 | 1203.32 | 1201.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 12:15:00 | 1240.20 | 1210.70 | 1205.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 1291.40 | 1296.06 | 1272.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 1290.45 | 1296.06 | 1272.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 1349.00 | 1352.74 | 1341.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 1373.20 | 1352.74 | 1341.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 10:15:00 | 1329.10 | 1366.66 | 1359.76 | SL hit (close<static) qty=1.00 sl=1340.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 12:15:00 | 1318.50 | 1350.65 | 1353.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 15:15:00 | 1308.20 | 1325.22 | 1334.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 1330.75 | 1325.62 | 1333.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 10:15:00 | 1330.75 | 1325.62 | 1333.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1330.75 | 1325.62 | 1333.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 1330.75 | 1325.62 | 1333.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1315.05 | 1323.51 | 1331.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 1308.85 | 1323.51 | 1331.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:15:00 | 1243.41 | 1290.77 | 1311.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 09:15:00 | 1177.96 | 1214.36 | 1256.90 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 15:15:00 | 1176.20 | 1148.13 | 1146.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 1203.00 | 1159.11 | 1151.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 1189.65 | 1193.83 | 1180.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 13:00:00 | 1189.65 | 1193.83 | 1180.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 1190.05 | 1195.77 | 1189.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 1188.05 | 1195.77 | 1189.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 1193.10 | 1195.24 | 1189.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:45:00 | 1188.90 | 1195.24 | 1189.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 1185.00 | 1193.19 | 1189.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 1181.35 | 1193.19 | 1189.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1187.50 | 1192.05 | 1188.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 1207.30 | 1194.22 | 1190.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 14:15:00 | 1188.05 | 1190.35 | 1190.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 14:15:00 | 1188.05 | 1190.35 | 1190.53 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 14:15:00 | 1197.00 | 1190.18 | 1190.07 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 1172.85 | 1187.51 | 1188.93 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 15:15:00 | 1191.50 | 1188.77 | 1188.60 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 1150.25 | 1181.07 | 1185.11 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 1198.15 | 1184.31 | 1183.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 12:15:00 | 1209.00 | 1199.85 | 1195.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 1216.95 | 1231.87 | 1221.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 1216.95 | 1231.87 | 1221.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1216.95 | 1231.87 | 1221.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 1216.95 | 1231.87 | 1221.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1219.45 | 1229.38 | 1221.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:30:00 | 1215.10 | 1229.38 | 1221.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1200.00 | 1215.64 | 1216.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 1193.55 | 1211.22 | 1214.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 1217.80 | 1212.54 | 1215.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 1217.80 | 1212.54 | 1215.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1217.80 | 1212.54 | 1215.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 1217.80 | 1212.54 | 1215.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1220.95 | 1214.22 | 1215.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 1220.95 | 1214.22 | 1215.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 1210.60 | 1213.50 | 1215.16 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 1225.10 | 1216.86 | 1216.46 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 1211.25 | 1216.38 | 1216.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 1205.65 | 1214.23 | 1215.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 12:15:00 | 1214.30 | 1214.25 | 1215.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 13:00:00 | 1214.30 | 1214.25 | 1215.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 1213.80 | 1214.16 | 1215.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:45:00 | 1216.05 | 1214.16 | 1215.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1219.65 | 1215.26 | 1215.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1219.65 | 1215.26 | 1215.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 1219.00 | 1216.01 | 1215.94 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 1213.40 | 1215.92 | 1215.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 1207.90 | 1214.32 | 1215.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1216.45 | 1209.90 | 1212.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 1216.45 | 1209.90 | 1212.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1216.45 | 1209.90 | 1212.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 1217.20 | 1209.90 | 1212.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1209.55 | 1209.83 | 1211.96 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 1223.25 | 1214.43 | 1213.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1229.80 | 1218.40 | 1215.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 13:15:00 | 1217.35 | 1220.39 | 1217.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 13:15:00 | 1217.35 | 1220.39 | 1217.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 1217.35 | 1220.39 | 1217.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 1230.50 | 1220.32 | 1218.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 1238.75 | 1250.43 | 1251.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 1238.75 | 1250.43 | 1251.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 1231.05 | 1244.95 | 1248.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 15:15:00 | 1215.00 | 1213.23 | 1222.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 09:15:00 | 1218.90 | 1213.23 | 1222.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1221.20 | 1214.83 | 1222.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:45:00 | 1197.55 | 1213.90 | 1218.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 1192.15 | 1210.67 | 1216.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 15:15:00 | 1193.00 | 1205.12 | 1212.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:45:00 | 1197.00 | 1199.70 | 1206.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1207.10 | 1201.18 | 1206.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-02 15:15:00 | 1231.85 | 1210.73 | 1209.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 1231.85 | 1210.73 | 1209.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 15:15:00 | 1239.00 | 1227.55 | 1220.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1196.05 | 1221.25 | 1217.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1196.05 | 1221.25 | 1217.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1196.05 | 1221.25 | 1217.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1196.05 | 1221.25 | 1217.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1207.65 | 1218.53 | 1217.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 1202.80 | 1218.53 | 1217.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 1195.75 | 1213.97 | 1215.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 14:15:00 | 1190.15 | 1205.34 | 1210.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 1120.70 | 1114.48 | 1142.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 1120.70 | 1114.48 | 1142.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1157.20 | 1126.56 | 1142.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:00:00 | 1157.20 | 1126.56 | 1142.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1151.80 | 1131.61 | 1143.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1132.85 | 1135.96 | 1144.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 1166.40 | 1132.87 | 1132.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 1166.40 | 1132.87 | 1132.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 12:15:00 | 1180.60 | 1151.65 | 1142.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 10:15:00 | 1287.10 | 1288.45 | 1270.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 11:00:00 | 1287.10 | 1288.45 | 1270.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1270.10 | 1281.78 | 1271.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:45:00 | 1270.50 | 1281.78 | 1271.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 1293.10 | 1284.05 | 1273.46 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1257.00 | 1267.90 | 1268.56 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 1287.60 | 1263.90 | 1263.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 10:15:00 | 1290.70 | 1269.26 | 1265.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 09:15:00 | 1311.70 | 1312.33 | 1298.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1311.70 | 1312.33 | 1298.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1311.70 | 1312.33 | 1298.04 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 1297.20 | 1308.65 | 1308.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 1262.80 | 1294.68 | 1302.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 12:15:00 | 1273.00 | 1269.57 | 1284.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 13:00:00 | 1273.00 | 1269.57 | 1284.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1261.50 | 1270.42 | 1281.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1287.40 | 1270.42 | 1281.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1285.10 | 1273.35 | 1281.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 1270.00 | 1274.78 | 1280.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 1259.40 | 1270.33 | 1277.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1206.50 | 1253.28 | 1268.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 1230.30 | 1226.79 | 1245.16 | SL hit (close>ema200) qty=0.50 sl=1226.79 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1293.00 | 1255.94 | 1255.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1296.10 | 1263.97 | 1259.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 1298.00 | 1300.50 | 1287.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:30:00 | 1316.60 | 1304.22 | 1290.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 09:15:00 | 1382.43 | 1351.43 | 1329.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1389.00 | 1372.68 | 1353.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 1362.80 | 1373.48 | 1360.30 | SL hit (close<ema200) qty=0.50 sl=1373.48 alert=retest1 |

### Cycle 72 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 1355.00 | 1361.19 | 1361.71 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 1383.80 | 1363.02 | 1361.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 1390.50 | 1368.52 | 1364.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 1362.20 | 1369.49 | 1365.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 12:15:00 | 1362.20 | 1369.49 | 1365.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1362.20 | 1369.49 | 1365.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 1362.20 | 1369.49 | 1365.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1361.00 | 1367.79 | 1365.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:45:00 | 1360.20 | 1367.79 | 1365.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1370.00 | 1368.24 | 1365.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 1375.00 | 1367.61 | 1365.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 1370.10 | 1369.37 | 1366.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 1371.00 | 1371.08 | 1368.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 1356.30 | 1370.37 | 1369.87 | SL hit (close<static) qty=1.00 sl=1358.50 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 1355.20 | 1367.33 | 1368.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 1344.20 | 1360.97 | 1365.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 1365.00 | 1360.66 | 1364.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 1365.00 | 1360.66 | 1364.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1365.00 | 1360.66 | 1364.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 1364.40 | 1360.66 | 1364.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1350.50 | 1358.63 | 1363.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:15:00 | 1348.20 | 1358.63 | 1363.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 14:45:00 | 1348.80 | 1355.58 | 1360.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1387.10 | 1361.63 | 1362.69 | SL hit (close>static) qty=1.00 sl=1367.30 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 1395.30 | 1368.36 | 1365.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1425.00 | 1394.26 | 1384.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 15:15:00 | 1420.00 | 1422.93 | 1405.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1443.10 | 1422.93 | 1405.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:45:00 | 1446.40 | 1434.11 | 1415.69 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:15:00 | 1515.25 | 1478.77 | 1448.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:15:00 | 1518.72 | 1478.77 | 1448.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-04 10:15:00 | 1511.00 | 1513.46 | 1483.07 | SL hit (close<ema200) qty=0.50 sl=1513.46 alert=retest1 |

### Cycle 76 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 1619.90 | 1635.96 | 1636.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 1610.80 | 1630.93 | 1633.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 1581.00 | 1565.57 | 1578.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 12:15:00 | 1581.00 | 1565.57 | 1578.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1581.00 | 1565.57 | 1578.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1581.00 | 1565.57 | 1578.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1589.20 | 1570.30 | 1579.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1589.20 | 1570.30 | 1579.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1584.00 | 1573.04 | 1579.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 1591.20 | 1573.04 | 1579.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1600.00 | 1580.10 | 1581.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 1587.40 | 1580.10 | 1581.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 1603.00 | 1584.68 | 1583.89 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 1567.80 | 1586.59 | 1587.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 1556.20 | 1577.94 | 1583.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1547.00 | 1546.98 | 1560.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:15:00 | 1552.60 | 1546.98 | 1560.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1521.50 | 1508.75 | 1521.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:30:00 | 1524.60 | 1508.75 | 1521.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1518.50 | 1510.70 | 1521.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:15:00 | 1520.80 | 1510.70 | 1521.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 1520.00 | 1512.56 | 1521.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 12:30:00 | 1512.60 | 1513.77 | 1520.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:15:00 | 1516.40 | 1513.77 | 1520.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:00:00 | 1512.00 | 1509.42 | 1516.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 1509.90 | 1513.16 | 1515.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1464.40 | 1464.92 | 1473.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 1491.00 | 1479.52 | 1478.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 13:15:00 | 1491.00 | 1479.52 | 1478.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 1506.30 | 1490.41 | 1485.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 1508.10 | 1510.93 | 1502.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 1508.10 | 1510.93 | 1502.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1508.10 | 1510.93 | 1502.74 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1500.00 | 1511.95 | 1511.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 1496.20 | 1508.80 | 1510.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 1537.00 | 1509.01 | 1509.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 11:15:00 | 1537.00 | 1509.01 | 1509.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1537.00 | 1509.01 | 1509.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 1537.00 | 1509.01 | 1509.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 1527.40 | 1512.69 | 1510.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1583.00 | 1539.20 | 1524.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 15:15:00 | 1685.10 | 1688.15 | 1668.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:15:00 | 1695.80 | 1688.15 | 1668.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1674.00 | 1683.60 | 1670.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:45:00 | 1673.00 | 1683.60 | 1670.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1668.40 | 1686.36 | 1682.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 1668.40 | 1686.36 | 1682.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1671.00 | 1683.29 | 1681.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1642.10 | 1683.29 | 1681.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 1643.70 | 1675.37 | 1677.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 10:15:00 | 1626.00 | 1645.83 | 1658.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 1615.00 | 1613.51 | 1628.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 15:00:00 | 1615.00 | 1613.51 | 1628.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1599.50 | 1611.59 | 1625.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:30:00 | 1580.50 | 1597.38 | 1611.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:30:00 | 1580.20 | 1574.58 | 1582.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 1560.50 | 1576.79 | 1582.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 13:15:00 | 1602.10 | 1582.17 | 1582.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 1602.10 | 1582.17 | 1582.17 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 1566.50 | 1582.07 | 1583.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 11:15:00 | 1552.50 | 1572.44 | 1578.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 1578.00 | 1573.55 | 1578.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 1578.00 | 1573.55 | 1578.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1578.00 | 1573.55 | 1578.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1578.00 | 1573.55 | 1578.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1581.40 | 1575.12 | 1579.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1581.40 | 1575.12 | 1579.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1566.30 | 1573.36 | 1577.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 1582.40 | 1573.36 | 1577.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1562.80 | 1569.91 | 1575.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 1578.10 | 1569.91 | 1575.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1587.00 | 1573.39 | 1576.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 1587.00 | 1573.39 | 1576.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1587.90 | 1576.29 | 1577.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 1587.90 | 1576.29 | 1577.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 13:15:00 | 1586.10 | 1578.25 | 1577.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 15:15:00 | 1610.00 | 1586.07 | 1581.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 1579.90 | 1584.83 | 1581.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 1579.90 | 1584.83 | 1581.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1579.90 | 1584.83 | 1581.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1579.90 | 1584.83 | 1581.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1577.50 | 1583.37 | 1581.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 1576.20 | 1583.37 | 1581.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1575.00 | 1581.69 | 1580.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:45:00 | 1580.10 | 1581.69 | 1580.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1575.60 | 1580.47 | 1580.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:45:00 | 1573.40 | 1580.47 | 1580.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1578.70 | 1580.12 | 1580.00 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 1560.00 | 1576.10 | 1578.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 1555.10 | 1571.90 | 1576.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1555.60 | 1552.17 | 1561.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 1553.80 | 1552.17 | 1561.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1556.10 | 1552.95 | 1561.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1535.00 | 1552.95 | 1561.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 1521.40 | 1517.53 | 1517.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 1521.40 | 1517.53 | 1517.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 1539.30 | 1521.88 | 1519.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 1521.80 | 1526.34 | 1522.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 15:15:00 | 1521.80 | 1526.34 | 1522.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1521.80 | 1526.34 | 1522.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 1514.90 | 1526.34 | 1522.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1527.90 | 1526.65 | 1523.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 1514.90 | 1526.65 | 1523.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1514.10 | 1524.14 | 1522.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 1512.00 | 1524.14 | 1522.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 1508.30 | 1520.97 | 1521.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 12:15:00 | 1505.60 | 1517.90 | 1519.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 12:15:00 | 1501.00 | 1499.07 | 1504.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 12:15:00 | 1501.00 | 1499.07 | 1504.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 1501.00 | 1499.07 | 1504.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 1501.00 | 1499.07 | 1504.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 1503.70 | 1500.00 | 1503.97 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1523.00 | 1506.58 | 1506.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 1534.40 | 1512.15 | 1508.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 1517.60 | 1519.60 | 1514.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 15:00:00 | 1517.60 | 1519.60 | 1514.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1525.20 | 1521.23 | 1515.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:15:00 | 1531.90 | 1521.23 | 1515.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 1508.90 | 1518.57 | 1515.59 | SL hit (close<static) qty=1.00 sl=1511.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 1494.10 | 1510.70 | 1512.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 1474.60 | 1503.48 | 1508.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 1500.00 | 1497.24 | 1504.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 1500.00 | 1497.24 | 1504.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1500.00 | 1497.24 | 1504.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1500.00 | 1497.24 | 1504.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1504.30 | 1498.65 | 1504.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 1504.30 | 1498.65 | 1504.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1505.00 | 1499.92 | 1504.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1483.00 | 1502.20 | 1504.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 1510.40 | 1497.19 | 1500.02 | SL hit (close>static) qty=1.00 sl=1508.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1452.00 | 1435.47 | 1434.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1463.20 | 1443.68 | 1438.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 1593.90 | 1594.76 | 1570.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 1593.90 | 1594.76 | 1570.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1612.00 | 1623.66 | 1612.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:30:00 | 1611.90 | 1623.66 | 1612.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1609.90 | 1620.91 | 1612.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:30:00 | 1611.60 | 1620.91 | 1612.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1607.50 | 1618.23 | 1612.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:30:00 | 1606.70 | 1618.23 | 1612.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1603.90 | 1610.36 | 1609.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 1603.20 | 1610.36 | 1609.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1610.60 | 1610.38 | 1609.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1610.00 | 1610.38 | 1609.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1603.10 | 1608.92 | 1609.22 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 12:15:00 | 1613.80 | 1609.90 | 1609.64 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 1598.30 | 1607.58 | 1608.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1592.00 | 1604.46 | 1607.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1578.60 | 1574.20 | 1580.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1578.60 | 1574.20 | 1580.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1578.60 | 1574.20 | 1580.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:30:00 | 1558.00 | 1573.89 | 1578.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 1565.20 | 1573.89 | 1578.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:00:00 | 1563.20 | 1571.76 | 1577.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1549.60 | 1568.66 | 1574.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1537.30 | 1520.18 | 1532.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 1537.30 | 1520.18 | 1532.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 1553.40 | 1526.82 | 1534.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 1532.50 | 1526.82 | 1534.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 12:15:00 | 1480.10 | 1504.36 | 1516.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 12:15:00 | 1486.94 | 1504.36 | 1516.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 12:15:00 | 1485.04 | 1504.36 | 1516.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 13:15:00 | 1472.12 | 1498.51 | 1512.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 1473.50 | 1473.24 | 1484.86 | SL hit (close>ema200) qty=0.50 sl=1473.24 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 1502.00 | 1447.14 | 1445.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 1537.20 | 1465.15 | 1453.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1527.50 | 1527.86 | 1506.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 15:15:00 | 1519.80 | 1526.39 | 1515.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 1519.80 | 1526.39 | 1515.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:30:00 | 1547.80 | 1528.57 | 1517.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 11:15:00 | 1503.50 | 1521.22 | 1515.56 | SL hit (close<static) qty=1.00 sl=1510.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1533.00 | 1550.11 | 1550.22 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 1558.30 | 1551.75 | 1550.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 1565.00 | 1557.57 | 1554.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1552.20 | 1556.50 | 1554.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1552.20 | 1556.50 | 1554.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1552.20 | 1556.50 | 1554.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1552.20 | 1556.50 | 1554.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1555.20 | 1556.24 | 1554.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 1552.90 | 1556.24 | 1554.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1572.00 | 1559.39 | 1555.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:15:00 | 1578.40 | 1559.39 | 1555.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 13:30:00 | 1576.50 | 1565.19 | 1559.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:15:00 | 1576.10 | 1567.15 | 1560.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 1615.80 | 1649.58 | 1654.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1615.80 | 1649.58 | 1654.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 14:15:00 | 1610.10 | 1630.04 | 1642.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 1624.10 | 1618.20 | 1627.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1624.10 | 1618.20 | 1627.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1624.10 | 1618.20 | 1627.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 1631.50 | 1618.20 | 1627.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1626.80 | 1619.92 | 1627.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 1628.70 | 1619.92 | 1627.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1625.00 | 1620.94 | 1626.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 1624.10 | 1620.94 | 1626.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1604.00 | 1592.03 | 1602.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 1604.00 | 1592.03 | 1602.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1607.10 | 1595.05 | 1602.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 1609.90 | 1595.05 | 1602.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1607.00 | 1597.44 | 1602.94 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 1619.50 | 1608.02 | 1606.68 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1586.70 | 1603.76 | 1604.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 1577.60 | 1598.52 | 1602.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 1569.50 | 1567.76 | 1575.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 13:45:00 | 1566.50 | 1567.76 | 1575.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1557.30 | 1565.66 | 1573.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:30:00 | 1570.50 | 1565.66 | 1573.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1536.70 | 1556.73 | 1567.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:45:00 | 1531.50 | 1551.96 | 1564.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:45:00 | 1523.50 | 1540.70 | 1554.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 1526.00 | 1525.27 | 1533.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 15:15:00 | 1545.10 | 1538.46 | 1537.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 1545.10 | 1538.46 | 1537.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1559.50 | 1542.67 | 1539.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 1548.30 | 1559.00 | 1552.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 11:15:00 | 1548.30 | 1559.00 | 1552.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1548.30 | 1559.00 | 1552.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 1551.90 | 1559.00 | 1552.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1544.40 | 1556.08 | 1551.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:15:00 | 1543.00 | 1556.08 | 1551.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1545.10 | 1553.88 | 1551.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 1545.10 | 1553.88 | 1551.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 1535.60 | 1547.72 | 1548.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 1532.30 | 1542.65 | 1546.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 13:15:00 | 1543.60 | 1539.31 | 1543.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 1543.60 | 1539.31 | 1543.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 1543.60 | 1539.31 | 1543.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 1543.60 | 1539.31 | 1543.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1535.50 | 1538.55 | 1542.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:45:00 | 1542.00 | 1538.55 | 1542.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1534.00 | 1537.64 | 1541.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 1544.00 | 1537.64 | 1541.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1549.70 | 1540.05 | 1542.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 1537.30 | 1540.82 | 1542.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:45:00 | 1528.20 | 1539.13 | 1541.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1555.00 | 1540.80 | 1539.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 1555.00 | 1540.80 | 1539.87 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 1530.40 | 1538.72 | 1539.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1524.00 | 1535.78 | 1537.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 1534.50 | 1528.78 | 1532.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 10:15:00 | 1534.50 | 1528.78 | 1532.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1534.50 | 1528.78 | 1532.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1534.50 | 1528.78 | 1532.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1532.60 | 1529.54 | 1532.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:15:00 | 1535.20 | 1529.54 | 1532.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1536.30 | 1530.89 | 1532.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 1542.80 | 1530.89 | 1532.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1532.00 | 1531.11 | 1532.65 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 15:15:00 | 1543.00 | 1534.90 | 1534.19 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1505.40 | 1529.14 | 1532.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1496.40 | 1518.73 | 1526.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 1461.00 | 1444.99 | 1468.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:30:00 | 1453.30 | 1444.99 | 1468.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1430.80 | 1421.51 | 1436.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 1430.10 | 1421.51 | 1436.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1440.00 | 1425.76 | 1435.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 1433.80 | 1425.76 | 1435.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1438.60 | 1428.33 | 1436.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:45:00 | 1441.90 | 1428.33 | 1436.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1437.00 | 1433.72 | 1435.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 1437.00 | 1433.72 | 1435.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1430.00 | 1432.97 | 1435.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1416.10 | 1432.97 | 1435.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 1440.00 | 1432.94 | 1434.81 | SL hit (close>static) qty=1.00 sl=1438.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 12:15:00 | 1445.00 | 1437.17 | 1436.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 13:15:00 | 1458.40 | 1441.42 | 1438.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 1445.10 | 1449.95 | 1443.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 1445.10 | 1449.95 | 1443.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1445.10 | 1449.95 | 1443.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 1445.10 | 1449.95 | 1443.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1435.00 | 1446.96 | 1443.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 1451.00 | 1443.34 | 1442.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 1432.90 | 1442.48 | 1442.12 | SL hit (close<static) qty=1.00 sl=1433.20 alert=retest2 |

### Cycle 108 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1473.30 | 1477.23 | 1477.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 1467.10 | 1474.40 | 1476.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 1461.00 | 1459.10 | 1466.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 11:45:00 | 1461.00 | 1459.10 | 1466.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1469.40 | 1459.86 | 1465.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 1464.40 | 1459.86 | 1465.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1457.20 | 1459.33 | 1464.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 1468.60 | 1459.33 | 1464.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1466.70 | 1460.11 | 1463.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 1462.20 | 1460.11 | 1463.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1477.50 | 1463.59 | 1465.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1477.50 | 1463.59 | 1465.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 1480.90 | 1467.05 | 1466.56 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 12:15:00 | 1459.30 | 1465.50 | 1465.90 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 1474.80 | 1466.24 | 1465.29 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 1459.30 | 1464.13 | 1464.51 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1471.40 | 1465.59 | 1465.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1484.20 | 1469.31 | 1466.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 1550.70 | 1554.96 | 1526.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 1550.70 | 1554.96 | 1526.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1535.00 | 1544.99 | 1534.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 1535.00 | 1544.99 | 1534.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1534.60 | 1542.92 | 1534.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 1534.60 | 1542.92 | 1534.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1526.10 | 1539.55 | 1533.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:30:00 | 1510.40 | 1539.55 | 1533.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1526.90 | 1537.02 | 1533.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 1525.40 | 1537.02 | 1533.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1525.70 | 1533.55 | 1532.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1526.00 | 1533.55 | 1532.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1561.80 | 1539.20 | 1534.95 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 1537.90 | 1541.69 | 1541.94 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 14:15:00 | 1547.30 | 1542.81 | 1542.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 15:15:00 | 1570.00 | 1548.25 | 1544.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 09:15:00 | 1530.10 | 1544.62 | 1543.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 1530.10 | 1544.62 | 1543.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1530.10 | 1544.62 | 1543.59 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 10:15:00 | 1531.00 | 1541.69 | 1543.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 13:15:00 | 1520.70 | 1535.55 | 1539.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 1534.30 | 1530.54 | 1535.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 11:15:00 | 1534.30 | 1530.54 | 1535.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1534.30 | 1530.54 | 1535.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 1536.90 | 1530.54 | 1535.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1553.90 | 1535.21 | 1537.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:45:00 | 1553.60 | 1535.21 | 1537.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1546.80 | 1537.53 | 1537.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 1546.80 | 1537.53 | 1537.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 1550.10 | 1540.04 | 1539.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1561.30 | 1545.89 | 1541.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1540.20 | 1548.59 | 1544.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 1540.20 | 1548.59 | 1544.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1540.20 | 1548.59 | 1544.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:45:00 | 1542.30 | 1548.59 | 1544.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1526.90 | 1544.25 | 1543.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 1526.90 | 1544.25 | 1543.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1528.00 | 1541.00 | 1541.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 1500.60 | 1532.92 | 1538.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 1366.00 | 1362.60 | 1405.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 1375.70 | 1362.60 | 1405.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1369.40 | 1353.58 | 1368.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 1368.20 | 1353.58 | 1368.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1373.00 | 1357.46 | 1369.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 1373.00 | 1357.46 | 1369.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1393.40 | 1364.65 | 1371.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1362.30 | 1364.65 | 1371.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 1364.60 | 1366.06 | 1371.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:30:00 | 1371.20 | 1362.96 | 1367.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:00:00 | 1366.90 | 1364.55 | 1367.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1368.20 | 1365.28 | 1367.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:30:00 | 1369.10 | 1365.28 | 1367.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 1399.50 | 1372.13 | 1370.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 1399.50 | 1372.13 | 1370.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 1408.00 | 1383.36 | 1375.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 1389.90 | 1394.79 | 1383.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 1389.90 | 1394.79 | 1383.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1384.50 | 1392.73 | 1383.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 1382.40 | 1392.73 | 1383.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 1380.60 | 1390.30 | 1383.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 1380.60 | 1390.30 | 1383.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 1396.80 | 1391.60 | 1384.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 1404.90 | 1394.00 | 1386.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 1406.00 | 1401.07 | 1391.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1366.10 | 1415.21 | 1413.13 | SL hit (close<static) qty=1.00 sl=1378.10 alert=retest2 |

### Cycle 120 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1375.00 | 1407.16 | 1409.67 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1421.50 | 1408.43 | 1408.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1484.80 | 1425.55 | 1416.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1475.10 | 1478.69 | 1456.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:30:00 | 1469.80 | 1478.69 | 1456.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1497.80 | 1486.98 | 1477.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:00:00 | 1507.50 | 1491.30 | 1483.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1501.20 | 1525.46 | 1526.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1501.20 | 1525.46 | 1526.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 1490.40 | 1510.46 | 1517.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 12:15:00 | 1488.10 | 1487.68 | 1497.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 12:45:00 | 1490.30 | 1487.68 | 1497.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1492.90 | 1488.73 | 1496.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 1492.90 | 1488.73 | 1496.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1479.40 | 1481.38 | 1488.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 1475.10 | 1481.64 | 1486.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 1500.50 | 1483.36 | 1485.71 | SL hit (close>static) qty=1.00 sl=1488.60 alert=retest2 |

### Cycle 123 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 1517.20 | 1492.18 | 1488.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 1527.60 | 1507.65 | 1497.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 14:15:00 | 1513.90 | 1514.66 | 1505.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 15:00:00 | 1513.90 | 1514.66 | 1505.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1497.30 | 1510.92 | 1505.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1493.40 | 1510.92 | 1505.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1497.40 | 1508.22 | 1504.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 1497.40 | 1508.22 | 1504.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1490.20 | 1504.61 | 1503.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 1490.20 | 1504.61 | 1503.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 1479.70 | 1499.63 | 1500.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 1469.80 | 1493.67 | 1498.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 1468.50 | 1458.12 | 1470.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 1468.50 | 1458.12 | 1470.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1468.50 | 1458.12 | 1470.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 1465.40 | 1458.12 | 1470.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1347.80 | 1339.17 | 1362.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1307.50 | 1353.49 | 1355.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 15:00:00 | 1323.50 | 1315.54 | 1323.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 1339.40 | 1328.45 | 1327.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 1339.40 | 1328.45 | 1327.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 14:15:00 | 1354.50 | 1337.13 | 1332.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 1331.40 | 1339.65 | 1334.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 1331.40 | 1339.65 | 1334.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1331.40 | 1339.65 | 1334.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 1356.70 | 1342.34 | 1336.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1312.60 | 1333.80 | 1334.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1312.60 | 1333.80 | 1334.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 1302.60 | 1327.56 | 1331.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1259.90 | 1257.04 | 1279.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 1259.90 | 1257.04 | 1279.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1276.00 | 1261.61 | 1274.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 1276.00 | 1261.61 | 1274.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1276.00 | 1264.49 | 1274.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:00:00 | 1276.00 | 1264.49 | 1274.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1282.00 | 1267.99 | 1275.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1295.90 | 1273.01 | 1277.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1296.30 | 1277.67 | 1278.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 1296.80 | 1277.67 | 1278.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 1288.10 | 1279.76 | 1279.61 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1262.90 | 1279.56 | 1280.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 1252.00 | 1269.81 | 1275.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 13:15:00 | 1266.40 | 1263.55 | 1270.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 14:15:00 | 1268.40 | 1263.55 | 1270.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1228.60 | 1215.63 | 1227.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 1230.40 | 1215.63 | 1227.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1208.10 | 1214.12 | 1226.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 1200.20 | 1214.12 | 1226.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1248.30 | 1218.73 | 1225.93 | SL hit (close>static) qty=1.00 sl=1229.40 alert=retest2 |

### Cycle 129 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 1222.60 | 1191.21 | 1187.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 15:15:00 | 1230.00 | 1198.97 | 1191.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1277.60 | 1282.86 | 1256.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 1276.70 | 1282.86 | 1256.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1289.20 | 1298.64 | 1289.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 1294.80 | 1299.33 | 1290.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 1309.00 | 1318.29 | 1319.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1309.00 | 1318.29 | 1319.16 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1369.50 | 1328.53 | 1323.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 13:15:00 | 1390.80 | 1368.60 | 1354.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 1407.20 | 1411.55 | 1392.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 11:00:00 | 1407.20 | 1411.55 | 1392.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 1402.10 | 1408.29 | 1394.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:00:00 | 1402.10 | 1408.29 | 1394.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1417.00 | 1423.54 | 1413.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:45:00 | 1418.00 | 1423.54 | 1413.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1438.10 | 1424.26 | 1416.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 1426.20 | 1424.26 | 1416.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1433.20 | 1443.42 | 1434.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1456.00 | 1436.41 | 1434.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 1453.20 | 1446.29 | 1440.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 1498.40 | 1443.83 | 1441.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1457.00 | 1453.92 | 1450.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1444.00 | 1451.94 | 1449.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 1444.00 | 1451.94 | 1449.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1437.50 | 1449.05 | 1448.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:30:00 | 1436.00 | 1449.05 | 1448.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1436.60 | 1446.56 | 1447.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 11:15:00 | 1436.60 | 1446.56 | 1447.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 09:15:00 | 1421.30 | 1440.13 | 1444.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 11:15:00 | 1440.00 | 1438.08 | 1442.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 11:15:00 | 1440.00 | 1438.08 | 1442.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 1440.00 | 1438.08 | 1442.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:00:00 | 1440.00 | 1438.08 | 1442.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1449.00 | 1440.27 | 1442.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:45:00 | 1448.60 | 1440.27 | 1442.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 1449.40 | 1442.09 | 1443.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:15:00 | 1446.00 | 1442.09 | 1443.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 14:15:00 | 1455.00 | 1444.68 | 1444.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 1455.00 | 1444.68 | 1444.56 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 1430.50 | 1442.69 | 1443.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 11:15:00 | 1428.20 | 1437.92 | 1441.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 12:15:00 | 1442.50 | 1438.84 | 1441.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 12:15:00 | 1442.50 | 1438.84 | 1441.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1442.50 | 1438.84 | 1441.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 1442.50 | 1438.84 | 1441.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1435.70 | 1438.21 | 1440.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 1435.70 | 1438.21 | 1440.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-15 10:15:00 | 1710.00 | 2024-05-15 10:15:00 | 1728.30 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-05-17 09:15:00 | 1708.25 | 2024-05-21 12:15:00 | 1705.96 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-05-31 10:15:00 | 1809.19 | 2024-06-03 09:15:00 | 1880.82 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2024-06-13 09:15:00 | 2132.24 | 2024-06-19 10:15:00 | 2040.00 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2024-06-24 14:45:00 | 2015.45 | 2024-06-26 11:15:00 | 2052.55 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-07-08 09:15:00 | 2045.00 | 2024-07-09 15:15:00 | 2014.85 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-07-08 10:00:00 | 2034.65 | 2024-07-09 15:15:00 | 2014.85 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-08 10:30:00 | 2038.45 | 2024-07-09 15:15:00 | 2014.85 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-07-08 14:45:00 | 2039.60 | 2024-07-09 15:15:00 | 2014.85 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-07-09 09:15:00 | 2042.00 | 2024-07-09 15:15:00 | 2014.85 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-07-18 09:15:00 | 1913.50 | 2024-07-22 09:15:00 | 1817.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1900.85 | 2024-07-22 09:15:00 | 1805.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 09:15:00 | 1913.50 | 2024-07-23 09:15:00 | 1839.80 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1900.85 | 2024-07-23 09:15:00 | 1839.80 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2024-08-06 12:15:00 | 1745.00 | 2024-08-06 15:15:00 | 1747.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-08-06 15:15:00 | 1747.00 | 2024-08-06 15:15:00 | 1747.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-08-22 15:15:00 | 1693.00 | 2024-08-26 11:15:00 | 1708.80 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-08-23 10:45:00 | 1691.70 | 2024-08-26 11:15:00 | 1708.80 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-08-23 11:45:00 | 1688.00 | 2024-08-26 11:15:00 | 1708.80 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-08-26 09:30:00 | 1692.95 | 2024-08-26 11:15:00 | 1708.80 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-09-04 14:45:00 | 1751.50 | 2024-09-05 15:15:00 | 1723.10 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-09-05 09:15:00 | 1758.75 | 2024-09-05 15:15:00 | 1723.10 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-09-05 09:45:00 | 1752.50 | 2024-09-05 15:15:00 | 1723.10 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-09-09 11:45:00 | 1664.95 | 2024-09-10 14:15:00 | 1732.05 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2024-09-13 09:15:00 | 1824.00 | 2024-09-17 12:15:00 | 1752.10 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2024-09-17 10:15:00 | 1757.70 | 2024-09-17 12:15:00 | 1752.10 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-09-19 14:45:00 | 1875.15 | 2024-09-23 09:15:00 | 2062.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-20 09:15:00 | 1881.45 | 2024-09-23 09:15:00 | 2069.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-20 09:45:00 | 1876.65 | 2024-09-23 09:15:00 | 2064.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-09 11:30:00 | 1733.30 | 2024-10-14 14:15:00 | 1726.90 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2024-10-29 10:30:00 | 1521.00 | 2024-10-30 11:15:00 | 1584.10 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2024-11-01 18:00:00 | 1605.75 | 2024-11-04 11:15:00 | 1558.75 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-11-11 14:45:00 | 1614.00 | 2024-11-12 09:15:00 | 1634.15 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-11-11 15:15:00 | 1602.00 | 2024-11-12 09:15:00 | 1634.15 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-11-18 15:15:00 | 1518.00 | 2024-11-22 11:15:00 | 1615.40 | STOP_HIT | 1.00 | -6.42% |
| SELL | retest2 | 2024-11-21 09:30:00 | 1516.75 | 2024-11-22 11:15:00 | 1615.40 | STOP_HIT | 1.00 | -6.50% |
| SELL | retest2 | 2024-11-21 12:30:00 | 1517.50 | 2024-11-22 11:15:00 | 1615.40 | STOP_HIT | 1.00 | -6.45% |
| SELL | retest2 | 2024-11-21 14:00:00 | 1515.00 | 2024-11-22 11:15:00 | 1615.40 | STOP_HIT | 1.00 | -6.63% |
| SELL | retest2 | 2024-11-22 10:45:00 | 1536.10 | 2024-11-22 11:15:00 | 1615.40 | STOP_HIT | 1.00 | -5.16% |
| BUY | retest2 | 2024-11-29 14:00:00 | 1666.95 | 2024-12-05 15:15:00 | 1673.00 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-12-02 09:30:00 | 1670.00 | 2024-12-05 15:15:00 | 1673.00 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-12-03 09:15:00 | 1680.25 | 2024-12-05 15:15:00 | 1673.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-12-09 12:30:00 | 1655.10 | 2024-12-13 10:15:00 | 1572.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 09:15:00 | 1655.65 | 2024-12-13 10:15:00 | 1572.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 12:30:00 | 1655.10 | 2024-12-16 09:15:00 | 1598.65 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2024-12-10 09:15:00 | 1655.65 | 2024-12-16 09:15:00 | 1598.65 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1474.45 | 2025-01-09 09:15:00 | 1400.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1474.45 | 2025-01-10 09:15:00 | 1327.01 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-04 09:15:00 | 1373.20 | 2025-02-05 10:15:00 | 1329.10 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-02-05 10:30:00 | 1349.10 | 2025-02-05 11:15:00 | 1326.80 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-02-07 12:15:00 | 1308.85 | 2025-02-10 09:15:00 | 1243.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:15:00 | 1308.85 | 2025-02-11 09:15:00 | 1177.96 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-24 11:30:00 | 1207.30 | 2025-02-25 14:15:00 | 1188.05 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-03-19 09:15:00 | 1230.50 | 2025-03-25 11:15:00 | 1238.75 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-04-01 10:45:00 | 1197.55 | 2025-04-02 15:15:00 | 1231.85 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-04-01 11:30:00 | 1192.15 | 2025-04-02 15:15:00 | 1231.85 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-04-01 15:15:00 | 1193.00 | 2025-04-02 15:15:00 | 1231.85 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-04-02 12:45:00 | 1197.00 | 2025-04-02 15:15:00 | 1231.85 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1132.85 | 2025-04-15 09:15:00 | 1166.40 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-05-08 13:00:00 | 1270.00 | 2025-05-09 09:15:00 | 1206.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:00:00 | 1270.00 | 2025-05-09 15:15:00 | 1230.30 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-05-08 14:45:00 | 1259.40 | 2025-05-12 11:15:00 | 1293.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest1 | 2025-05-14 09:30:00 | 1316.60 | 2025-05-16 09:15:00 | 1382.43 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-14 09:30:00 | 1316.60 | 2025-05-19 13:15:00 | 1362.80 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2025-05-20 10:30:00 | 1396.70 | 2025-05-21 09:15:00 | 1355.00 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-05-23 09:15:00 | 1375.00 | 2025-05-26 13:15:00 | 1356.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-05-23 11:15:00 | 1370.10 | 2025-05-26 13:15:00 | 1356.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-23 13:30:00 | 1371.00 | 2025-05-26 13:15:00 | 1356.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-05-27 13:15:00 | 1348.20 | 2025-05-28 09:15:00 | 1387.10 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-05-27 14:45:00 | 1348.80 | 2025-05-28 09:15:00 | 1387.10 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest1 | 2025-06-02 09:15:00 | 1443.10 | 2025-06-03 11:15:00 | 1515.25 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-02 11:45:00 | 1446.40 | 2025-06-03 11:15:00 | 1518.72 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-02 09:15:00 | 1443.10 | 2025-06-04 10:15:00 | 1511.00 | STOP_HIT | 0.50 | 4.71% |
| BUY | retest1 | 2025-06-02 11:45:00 | 1446.40 | 2025-06-04 10:15:00 | 1511.00 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2025-06-24 12:30:00 | 1512.60 | 2025-07-01 13:15:00 | 1491.00 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2025-06-24 13:15:00 | 1516.40 | 2025-07-01 13:15:00 | 1491.00 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-06-25 10:00:00 | 1512.00 | 2025-07-01 13:15:00 | 1491.00 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2025-06-26 09:30:00 | 1509.90 | 2025-07-01 13:15:00 | 1491.00 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2025-07-29 09:30:00 | 1580.50 | 2025-07-31 13:15:00 | 1602.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-07-30 14:30:00 | 1580.20 | 2025-07-31 13:15:00 | 1602.10 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1560.50 | 2025-07-31 13:15:00 | 1602.10 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1535.00 | 2025-08-14 10:15:00 | 1521.40 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-08-22 10:15:00 | 1531.90 | 2025-08-22 11:15:00 | 1508.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1483.00 | 2025-08-26 13:15:00 | 1510.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-08-26 15:00:00 | 1479.20 | 2025-09-05 14:15:00 | 1405.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 15:00:00 | 1479.20 | 2025-09-08 09:15:00 | 1428.30 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2025-09-25 12:30:00 | 1558.00 | 2025-10-03 12:15:00 | 1480.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1565.20 | 2025-10-03 12:15:00 | 1486.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 14:00:00 | 1563.20 | 2025-10-03 12:15:00 | 1485.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1549.60 | 2025-10-03 13:15:00 | 1472.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:30:00 | 1558.00 | 2025-10-07 12:15:00 | 1473.50 | STOP_HIT | 0.50 | 5.42% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1565.20 | 2025-10-07 12:15:00 | 1473.50 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2025-09-25 14:00:00 | 1563.20 | 2025-10-07 12:15:00 | 1473.50 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1549.60 | 2025-10-07 12:15:00 | 1473.50 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1532.50 | 2025-10-08 15:15:00 | 1455.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1532.50 | 2025-10-09 09:15:00 | 1468.70 | STOP_HIT | 0.50 | 4.16% |
| BUY | retest2 | 2025-10-20 09:30:00 | 1547.80 | 2025-10-20 11:15:00 | 1503.50 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-10-20 12:45:00 | 1533.90 | 2025-10-23 10:15:00 | 1508.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-23 11:15:00 | 1530.00 | 2025-10-29 09:15:00 | 1533.00 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-10-24 11:30:00 | 1535.90 | 2025-10-29 09:15:00 | 1533.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-10-27 09:15:00 | 1552.00 | 2025-10-29 09:15:00 | 1533.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-10-30 12:15:00 | 1578.40 | 2025-11-11 10:15:00 | 1615.80 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2025-10-30 13:30:00 | 1576.50 | 2025-11-11 10:15:00 | 1615.80 | STOP_HIT | 1.00 | 2.49% |
| BUY | retest2 | 2025-10-30 15:15:00 | 1576.10 | 2025-11-11 10:15:00 | 1615.80 | STOP_HIT | 1.00 | 2.52% |
| SELL | retest2 | 2025-11-21 10:45:00 | 1531.50 | 2025-11-25 15:15:00 | 1545.10 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-11-21 14:45:00 | 1523.50 | 2025-11-25 15:15:00 | 1545.10 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-25 10:45:00 | 1526.00 | 2025-11-25 15:15:00 | 1545.10 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-12-01 14:15:00 | 1537.30 | 2025-12-02 15:15:00 | 1555.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-01 14:45:00 | 1528.20 | 2025-12-02 15:15:00 | 1555.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1416.10 | 2025-12-16 10:15:00 | 1440.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-17 15:15:00 | 1451.00 | 2025-12-18 09:15:00 | 1432.90 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-12-18 11:15:00 | 1449.90 | 2025-12-29 09:15:00 | 1473.30 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2025-12-19 09:15:00 | 1456.40 | 2025-12-29 09:15:00 | 1473.30 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2025-12-19 12:45:00 | 1452.00 | 2025-12-29 09:15:00 | 1473.30 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-12-26 13:30:00 | 1481.90 | 2025-12-29 09:15:00 | 1473.30 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-12-26 14:45:00 | 1481.80 | 2025-12-29 09:15:00 | 1473.30 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-01-27 09:15:00 | 1362.30 | 2026-01-28 11:15:00 | 1399.50 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2026-01-27 10:15:00 | 1364.60 | 2026-01-28 11:15:00 | 1399.50 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-01-27 14:30:00 | 1371.20 | 2026-01-28 11:15:00 | 1399.50 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-01-28 10:00:00 | 1366.90 | 2026-01-28 11:15:00 | 1399.50 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-01-29 13:30:00 | 1404.90 | 2026-02-01 14:15:00 | 1366.10 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2026-01-30 09:30:00 | 1406.00 | 2026-02-01 14:15:00 | 1366.10 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2026-02-06 14:00:00 | 1507.50 | 2026-02-13 09:15:00 | 1501.20 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-02-19 11:45:00 | 1475.10 | 2026-02-19 14:15:00 | 1500.50 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1307.50 | 2026-03-11 11:15:00 | 1339.40 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-03-10 15:00:00 | 1323.50 | 2026-03-11 11:15:00 | 1339.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-03-12 11:30:00 | 1356.70 | 2026-03-13 09:15:00 | 1312.60 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-03-24 15:15:00 | 1200.20 | 2026-03-25 09:15:00 | 1248.30 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2026-03-27 09:15:00 | 1201.80 | 2026-03-27 15:15:00 | 1245.00 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2026-03-30 09:15:00 | 1200.30 | 2026-04-01 09:15:00 | 1229.90 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-03-30 10:00:00 | 1195.90 | 2026-04-01 09:15:00 | 1229.90 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-04-01 11:15:00 | 1203.10 | 2026-04-02 09:15:00 | 1142.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 12:45:00 | 1200.00 | 2026-04-02 09:15:00 | 1140.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 11:15:00 | 1203.10 | 2026-04-02 15:15:00 | 1167.90 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2026-04-01 12:45:00 | 1200.00 | 2026-04-02 15:15:00 | 1167.90 | STOP_HIT | 0.50 | 2.67% |
| BUY | retest2 | 2026-04-13 10:30:00 | 1294.80 | 2026-04-20 15:15:00 | 1309.00 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1456.00 | 2026-05-06 11:15:00 | 1436.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-05-04 12:30:00 | 1453.20 | 2026-05-06 11:15:00 | 1436.60 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-05-05 09:15:00 | 1498.40 | 2026-05-06 11:15:00 | 1436.60 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2026-05-06 09:15:00 | 1457.00 | 2026-05-06 11:15:00 | 1436.60 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-05-07 14:15:00 | 1446.00 | 2026-05-07 14:15:00 | 1455.00 | STOP_HIT | 1.00 | -0.62% |
