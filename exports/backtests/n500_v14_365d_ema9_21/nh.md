# Narayana Hrudayalaya Ltd. (NH)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1820.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 53 |
| ALERT2 | 54 |
| ALERT2_SKIP | 33 |
| ALERT3 | 151 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 64 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 50
- **Target hits / Stop hits / Partials:** 6 / 59 / 7
- **Avg / median % per leg:** 0.24% / -0.94%
- **Sum % (uncompounded):** 17.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 4 | 21.1% | 2 | 17 | 0 | 0.06% | 1.1% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.65% | 1.7% |
| BUY @ 3rd Alert (retest2) | 18 | 3 | 16.7% | 2 | 16 | 0 | -0.03% | -0.6% |
| SELL (all) | 53 | 18 | 34.0% | 4 | 42 | 7 | 0.30% | 16.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 53 | 18 | 34.0% | 4 | 42 | 7 | 0.30% | 16.1% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.65% | 1.7% |
| retest2 (combined) | 71 | 21 | 29.6% | 6 | 58 | 7 | 0.22% | 15.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 1796.00 | 1784.28 | 1783.44 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 1779.50 | 1782.56 | 1782.76 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 1792.20 | 1783.67 | 1783.14 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 1775.50 | 1783.73 | 1784.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 13:15:00 | 1770.00 | 1779.37 | 1782.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 1762.00 | 1756.14 | 1764.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1762.00 | 1756.14 | 1764.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1762.00 | 1756.14 | 1764.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1761.00 | 1756.14 | 1764.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1756.90 | 1756.29 | 1763.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 1762.00 | 1756.29 | 1763.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1754.40 | 1745.03 | 1753.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1752.50 | 1745.03 | 1753.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1753.20 | 1746.66 | 1753.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 1744.40 | 1745.01 | 1752.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 1822.30 | 1746.90 | 1740.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1822.30 | 1746.90 | 1740.38 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 1744.00 | 1755.60 | 1756.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 1723.70 | 1749.22 | 1753.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1745.50 | 1733.96 | 1741.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1745.50 | 1733.96 | 1741.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1745.50 | 1733.96 | 1741.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 1745.50 | 1733.96 | 1741.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1750.00 | 1737.17 | 1741.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 1750.00 | 1737.17 | 1741.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1756.30 | 1740.99 | 1743.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:30:00 | 1761.80 | 1740.99 | 1743.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 1772.60 | 1747.32 | 1745.95 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 1737.70 | 1749.20 | 1750.66 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1785.80 | 1753.30 | 1751.04 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 1740.70 | 1754.23 | 1755.63 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 1772.00 | 1753.79 | 1751.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 1792.80 | 1761.59 | 1755.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 1830.10 | 1832.51 | 1816.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 11:00:00 | 1830.10 | 1832.51 | 1816.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1813.50 | 1826.77 | 1816.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1813.50 | 1826.77 | 1816.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1811.40 | 1823.70 | 1815.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1811.40 | 1823.70 | 1815.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1820.20 | 1822.18 | 1816.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 1846.90 | 1822.18 | 1816.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1847.90 | 1827.33 | 1819.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:45:00 | 1886.50 | 1847.31 | 1831.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 1878.70 | 1900.87 | 1902.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 1878.70 | 1900.87 | 1902.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 14:15:00 | 1872.20 | 1891.94 | 1897.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 1890.50 | 1885.11 | 1891.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 1890.50 | 1885.11 | 1891.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1890.50 | 1885.11 | 1891.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1890.50 | 1885.11 | 1891.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1874.20 | 1882.92 | 1890.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:00:00 | 1863.00 | 1878.94 | 1887.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1903.00 | 1883.75 | 1889.20 | SL hit (close>static) qty=1.00 sl=1893.80 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 1926.10 | 1898.64 | 1895.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1987.80 | 1922.56 | 1907.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 2063.40 | 2068.53 | 2033.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 2063.40 | 2068.53 | 2033.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2036.10 | 2062.60 | 2036.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 2036.10 | 2062.60 | 2036.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 2041.70 | 2058.42 | 2037.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 2045.90 | 2058.42 | 2037.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 12:45:00 | 2044.00 | 2050.66 | 2036.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-27 14:15:00 | 2250.49 | 2115.81 | 2069.45 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-06-27 14:15:00 | 2248.40 | 2115.81 | 2069.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 2054.20 | 2104.15 | 2109.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 2026.90 | 2088.70 | 2101.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 2075.70 | 2032.67 | 2061.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 2075.70 | 2032.67 | 2061.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2075.70 | 2032.67 | 2061.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 2075.70 | 2032.67 | 2061.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2059.00 | 2037.93 | 2061.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 2044.20 | 2041.95 | 2060.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 2009.90 | 1984.27 | 1980.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 10:15:00 | 2009.90 | 1984.27 | 1980.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 2038.70 | 2007.75 | 1995.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 14:15:00 | 2019.30 | 2019.70 | 2007.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 2019.30 | 2019.70 | 2007.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1990.00 | 2013.29 | 2006.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 1990.00 | 2013.29 | 2006.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1997.10 | 2010.05 | 2005.51 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 12:15:00 | 1987.70 | 2000.82 | 2001.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 1965.30 | 1985.06 | 1993.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 1972.70 | 1967.12 | 1979.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1972.70 | 1967.12 | 1979.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1972.70 | 1967.12 | 1979.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 1949.20 | 1965.23 | 1973.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 13:15:00 | 1940.70 | 1960.82 | 1969.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 14:15:00 | 1934.60 | 1959.48 | 1967.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 1935.30 | 1957.94 | 1965.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1965.70 | 1958.84 | 1963.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 1962.00 | 1958.84 | 1963.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1933.70 | 1953.81 | 1961.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 1929.90 | 1948.93 | 1958.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 1971.10 | 1944.55 | 1953.13 | SL hit (close>static) qty=1.00 sl=1968.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 1971.50 | 1957.92 | 1957.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 1971.50 | 1957.92 | 1957.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 1971.50 | 1957.92 | 1957.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 1971.50 | 1957.92 | 1957.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 1971.50 | 1957.92 | 1957.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 1996.30 | 1974.55 | 1966.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 1980.00 | 1992.34 | 1985.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 1980.00 | 1992.34 | 1985.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1980.00 | 1992.34 | 1985.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 1980.00 | 1992.34 | 1985.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1980.00 | 1989.87 | 1984.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1982.40 | 1989.87 | 1984.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1994.20 | 1987.88 | 1984.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1999.60 | 1989.77 | 1986.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 1972.50 | 1986.11 | 1986.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 09:15:00 | 1972.50 | 1986.11 | 1986.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 10:15:00 | 1968.60 | 1977.44 | 1981.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 09:15:00 | 1960.90 | 1955.38 | 1966.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1960.90 | 1955.38 | 1966.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1960.90 | 1955.38 | 1966.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 1968.30 | 1955.38 | 1966.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1946.00 | 1953.51 | 1964.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 1943.00 | 1953.51 | 1964.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 1944.50 | 1951.61 | 1962.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 12:30:00 | 1941.00 | 1951.86 | 1961.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:30:00 | 1935.70 | 1946.39 | 1958.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1944.60 | 1938.68 | 1948.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 1944.20 | 1938.68 | 1948.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1947.60 | 1940.46 | 1948.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 1947.60 | 1940.46 | 1948.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1952.00 | 1942.77 | 1948.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:30:00 | 1950.50 | 1942.77 | 1948.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1942.70 | 1942.76 | 1948.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:45:00 | 1961.00 | 1942.76 | 1948.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1941.10 | 1942.42 | 1947.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1861.30 | 1942.42 | 1947.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 1845.85 | 1918.92 | 1936.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 1847.27 | 1918.92 | 1936.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 1843.95 | 1918.92 | 1936.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 1838.91 | 1918.92 | 1936.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:15:00 | 1768.23 | 1814.28 | 1846.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-07 09:15:00 | 1748.70 | 1776.78 | 1809.09 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-07 09:15:00 | 1750.05 | 1776.78 | 1809.09 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-07 09:15:00 | 1746.90 | 1776.78 | 1809.09 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-07 09:15:00 | 1742.13 | 1776.78 | 1809.09 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 12:15:00 | 1780.30 | 1770.02 | 1797.26 | SL hit (close>ema200) qty=0.50 sl=1770.02 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 1797.50 | 1769.50 | 1768.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 1809.80 | 1777.56 | 1771.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1811.50 | 1814.47 | 1802.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1811.50 | 1814.47 | 1802.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1814.00 | 1817.50 | 1808.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 1831.60 | 1818.81 | 1810.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1823.60 | 1835.63 | 1836.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1823.60 | 1835.63 | 1836.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 1820.70 | 1832.64 | 1834.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 1829.00 | 1824.84 | 1829.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 1829.00 | 1824.84 | 1829.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1829.00 | 1824.84 | 1829.32 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 1848.00 | 1833.04 | 1832.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 1849.70 | 1836.37 | 1833.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1817.00 | 1835.13 | 1834.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1817.00 | 1835.13 | 1834.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1817.00 | 1835.13 | 1834.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1830.90 | 1835.13 | 1834.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1803.60 | 1828.82 | 1831.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 1793.70 | 1821.80 | 1827.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1768.80 | 1766.58 | 1783.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 1770.10 | 1766.58 | 1783.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1767.30 | 1760.09 | 1772.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1767.30 | 1760.09 | 1772.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1777.90 | 1764.45 | 1772.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 1777.90 | 1764.45 | 1772.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1769.70 | 1765.50 | 1772.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:15:00 | 1786.50 | 1765.50 | 1772.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1790.10 | 1770.42 | 1773.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 1790.10 | 1770.42 | 1773.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1788.40 | 1774.02 | 1775.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:15:00 | 1782.90 | 1774.02 | 1775.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 1782.90 | 1775.79 | 1775.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 1782.90 | 1775.79 | 1775.75 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 1772.40 | 1775.74 | 1775.82 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1795.00 | 1775.81 | 1775.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 1807.40 | 1784.73 | 1779.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 1797.40 | 1798.54 | 1790.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 1797.40 | 1798.54 | 1790.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1797.40 | 1798.54 | 1790.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 1797.40 | 1798.54 | 1790.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1798.60 | 1812.97 | 1805.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 1800.60 | 1812.97 | 1805.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1797.30 | 1809.83 | 1804.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 1797.30 | 1809.83 | 1804.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 1785.00 | 1799.28 | 1800.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 1766.30 | 1792.69 | 1797.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 14:15:00 | 1782.50 | 1778.39 | 1787.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 1782.50 | 1778.39 | 1787.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1782.50 | 1778.39 | 1787.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 1789.90 | 1778.39 | 1787.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 1780.40 | 1778.79 | 1786.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 1785.00 | 1778.79 | 1786.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1771.00 | 1777.23 | 1785.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 1763.40 | 1772.15 | 1781.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 1763.50 | 1770.89 | 1773.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:00:00 | 1754.10 | 1770.89 | 1773.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:00:00 | 1764.10 | 1754.46 | 1758.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 1764.60 | 1756.49 | 1759.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1781.20 | 1762.97 | 1761.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1781.20 | 1762.97 | 1761.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1781.20 | 1762.97 | 1761.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1781.20 | 1762.97 | 1761.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1781.20 | 1762.97 | 1761.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 1789.60 | 1773.91 | 1768.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 1794.50 | 1801.01 | 1791.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 1794.50 | 1801.01 | 1791.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1794.50 | 1801.01 | 1791.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1794.50 | 1801.01 | 1791.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1795.80 | 1799.97 | 1792.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:15:00 | 1793.10 | 1799.97 | 1792.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1794.30 | 1798.84 | 1792.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:15:00 | 1792.50 | 1798.84 | 1792.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1784.10 | 1795.89 | 1791.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1784.10 | 1795.89 | 1791.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1787.00 | 1794.11 | 1791.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 1788.90 | 1794.11 | 1791.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1786.90 | 1792.67 | 1790.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1793.10 | 1791.76 | 1790.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 1777.20 | 1788.84 | 1789.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 1777.20 | 1788.84 | 1789.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 1770.20 | 1782.20 | 1785.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1727.10 | 1725.60 | 1735.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:15:00 | 1726.10 | 1725.60 | 1735.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1733.40 | 1728.13 | 1735.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:30:00 | 1734.40 | 1728.13 | 1735.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1722.10 | 1725.21 | 1730.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 1722.10 | 1725.21 | 1730.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 1725.20 | 1724.20 | 1728.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:30:00 | 1728.10 | 1724.20 | 1728.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 1724.10 | 1724.18 | 1728.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 1742.60 | 1724.18 | 1728.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1743.70 | 1728.08 | 1729.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 1751.20 | 1728.08 | 1729.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 10:15:00 | 1742.70 | 1731.01 | 1730.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 14:15:00 | 1772.70 | 1743.50 | 1736.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 12:15:00 | 1755.60 | 1760.44 | 1749.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 13:00:00 | 1755.60 | 1760.44 | 1749.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1741.80 | 1756.71 | 1748.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:30:00 | 1742.00 | 1756.71 | 1748.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1737.00 | 1752.77 | 1747.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 1737.00 | 1752.77 | 1747.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 1724.20 | 1744.05 | 1744.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 11:15:00 | 1718.70 | 1738.98 | 1742.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 15:15:00 | 1737.90 | 1735.42 | 1739.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 09:15:00 | 1776.00 | 1735.42 | 1739.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 31 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1770.60 | 1742.45 | 1741.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 1804.40 | 1763.25 | 1753.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 13:15:00 | 1791.20 | 1797.85 | 1776.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:30:00 | 1795.00 | 1797.85 | 1776.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1790.00 | 1792.49 | 1780.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 1787.50 | 1792.49 | 1780.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1787.90 | 1791.58 | 1780.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 1784.20 | 1791.58 | 1780.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1783.90 | 1790.04 | 1781.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 1785.00 | 1790.04 | 1781.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1769.20 | 1785.87 | 1780.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 1769.20 | 1785.87 | 1780.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1781.00 | 1784.90 | 1780.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1783.60 | 1784.32 | 1780.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 11:15:00 | 1771.70 | 1777.87 | 1777.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 1771.70 | 1777.87 | 1777.97 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 1784.20 | 1779.13 | 1778.54 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1777.50 | 1778.08 | 1778.12 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 1785.00 | 1779.46 | 1778.74 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 1754.00 | 1773.58 | 1776.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1740.70 | 1756.75 | 1763.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1753.20 | 1749.53 | 1756.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1753.20 | 1749.53 | 1756.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1753.20 | 1749.53 | 1756.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1758.70 | 1749.53 | 1756.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1765.80 | 1752.78 | 1757.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1765.80 | 1752.78 | 1757.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1755.40 | 1753.30 | 1757.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 1752.00 | 1753.30 | 1757.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1768.40 | 1754.49 | 1755.90 | SL hit (close>static) qty=1.00 sl=1767.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 1754.10 | 1755.93 | 1756.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:30:00 | 1755.20 | 1756.07 | 1756.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 1764.10 | 1757.67 | 1757.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 1764.10 | 1757.67 | 1757.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 1764.10 | 1757.67 | 1757.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 11:15:00 | 1782.00 | 1764.48 | 1760.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 13:15:00 | 1759.20 | 1763.75 | 1761.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 13:15:00 | 1759.20 | 1763.75 | 1761.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1759.20 | 1763.75 | 1761.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 1759.00 | 1763.75 | 1761.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1755.80 | 1762.16 | 1760.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 1755.80 | 1762.16 | 1760.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 1757.00 | 1761.13 | 1760.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 1766.00 | 1761.13 | 1760.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 1756.00 | 1762.71 | 1763.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 1756.00 | 1762.71 | 1763.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 1743.90 | 1757.00 | 1760.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 13:15:00 | 1739.10 | 1737.44 | 1744.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 13:15:00 | 1739.10 | 1737.44 | 1744.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1739.10 | 1737.44 | 1744.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 1739.10 | 1737.44 | 1744.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1747.10 | 1739.37 | 1744.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 1747.10 | 1739.37 | 1744.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1750.10 | 1741.52 | 1745.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1763.90 | 1741.52 | 1745.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 1776.00 | 1752.51 | 1749.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1780.70 | 1770.94 | 1762.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 11:15:00 | 1784.10 | 1789.38 | 1778.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 11:15:00 | 1784.10 | 1789.38 | 1778.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1784.10 | 1789.38 | 1778.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:30:00 | 1782.00 | 1789.38 | 1778.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1768.00 | 1786.42 | 1781.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1768.00 | 1786.42 | 1781.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1747.00 | 1778.53 | 1778.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1747.00 | 1778.53 | 1778.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1745.30 | 1771.89 | 1775.37 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 1786.90 | 1769.99 | 1768.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 1835.70 | 1785.19 | 1776.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 1821.60 | 1821.82 | 1803.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:30:00 | 1826.40 | 1821.82 | 1803.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1808.10 | 1818.79 | 1811.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:00:00 | 1824.10 | 1813.60 | 1810.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1781.10 | 1804.35 | 1807.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1781.10 | 1804.35 | 1807.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 1740.80 | 1768.41 | 1785.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 14:15:00 | 1758.70 | 1755.84 | 1771.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 15:00:00 | 1758.70 | 1755.84 | 1771.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1773.80 | 1759.99 | 1770.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1773.80 | 1759.99 | 1770.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1774.80 | 1762.95 | 1771.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1774.80 | 1762.95 | 1771.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1794.80 | 1769.32 | 1773.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:00:00 | 1794.80 | 1769.32 | 1773.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1765.30 | 1772.76 | 1774.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:45:00 | 1757.10 | 1768.31 | 1771.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 1756.30 | 1765.90 | 1770.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 1921.90 | 1792.22 | 1780.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 1921.90 | 1792.22 | 1780.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 1921.90 | 1792.22 | 1780.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 1942.00 | 1854.70 | 1814.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 12:15:00 | 1937.40 | 1942.45 | 1890.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 13:00:00 | 1937.40 | 1942.45 | 1890.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 2026.30 | 2010.06 | 1984.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:00:00 | 2035.40 | 2017.95 | 1992.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:00:00 | 2052.00 | 2024.76 | 1997.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1981.80 | 2006.36 | 2003.88 | SL hit (close<static) qty=1.00 sl=1982.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1981.80 | 2006.36 | 2003.88 | SL hit (close<static) qty=1.00 sl=1982.70 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 1982.20 | 2001.53 | 2001.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 1948.90 | 1987.24 | 1995.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1970.30 | 1966.62 | 1978.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:30:00 | 1971.90 | 1966.62 | 1978.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1978.30 | 1968.95 | 1978.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1980.80 | 1968.95 | 1978.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1968.90 | 1968.94 | 1977.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:00:00 | 1959.90 | 1966.04 | 1974.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:30:00 | 1952.00 | 1966.21 | 1971.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 1959.80 | 1949.91 | 1956.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:00:00 | 1948.80 | 1926.66 | 1928.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 1956.60 | 1934.51 | 1931.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 1956.60 | 1934.51 | 1931.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 1956.60 | 1934.51 | 1931.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 1956.60 | 1934.51 | 1931.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 1956.60 | 1934.51 | 1931.92 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 1914.60 | 1934.02 | 1936.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 1893.80 | 1919.57 | 1928.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 1923.40 | 1918.82 | 1926.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 14:15:00 | 1923.40 | 1918.82 | 1926.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 1923.40 | 1918.82 | 1926.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 15:00:00 | 1923.40 | 1918.82 | 1926.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 1922.30 | 1919.52 | 1925.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 1920.80 | 1919.52 | 1925.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 1918.00 | 1919.21 | 1925.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 10:45:00 | 1891.50 | 1914.49 | 1922.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 13:15:00 | 1901.00 | 1885.69 | 1884.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 13:15:00 | 1901.00 | 1885.69 | 1884.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 15:15:00 | 1905.00 | 1892.02 | 1887.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1886.40 | 1890.89 | 1887.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1886.40 | 1890.89 | 1887.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1886.40 | 1890.89 | 1887.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1893.20 | 1890.89 | 1887.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1913.00 | 1895.32 | 1889.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:15:00 | 1919.10 | 1895.32 | 1889.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 1872.00 | 1892.71 | 1891.95 | SL hit (close<static) qty=1.00 sl=1880.80 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 1872.20 | 1888.61 | 1890.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1853.00 | 1874.29 | 1881.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1873.80 | 1856.57 | 1866.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 1873.80 | 1856.57 | 1866.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1873.80 | 1856.57 | 1866.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 1873.80 | 1856.57 | 1866.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1878.20 | 1860.90 | 1867.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 1882.90 | 1860.90 | 1867.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1865.40 | 1863.73 | 1868.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 1878.30 | 1863.73 | 1868.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1874.80 | 1865.95 | 1868.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 1874.80 | 1865.95 | 1868.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1888.90 | 1870.54 | 1870.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 1920.00 | 1880.43 | 1874.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1883.40 | 1892.90 | 1886.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1883.40 | 1892.90 | 1886.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1883.40 | 1892.90 | 1886.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1876.10 | 1892.90 | 1886.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1880.00 | 1890.32 | 1885.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 1882.60 | 1890.32 | 1885.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 1870.20 | 1883.77 | 1883.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 1867.70 | 1875.58 | 1878.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 1877.00 | 1875.86 | 1878.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 10:15:00 | 1877.00 | 1875.86 | 1878.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1877.00 | 1875.86 | 1878.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1877.00 | 1875.86 | 1878.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1877.00 | 1876.09 | 1878.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 1877.00 | 1876.09 | 1878.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 1869.40 | 1874.75 | 1877.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 14:15:00 | 1865.00 | 1873.56 | 1876.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:45:00 | 1866.00 | 1868.04 | 1873.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 1863.30 | 1865.95 | 1871.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 14:45:00 | 1833.50 | 1835.40 | 1849.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1851.20 | 1839.30 | 1848.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 1851.20 | 1839.30 | 1848.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1882.00 | 1847.84 | 1851.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 1882.00 | 1847.84 | 1851.71 | SL hit (close>static) qty=1.00 sl=1881.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 1882.00 | 1847.84 | 1851.71 | SL hit (close>static) qty=1.00 sl=1881.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 1882.00 | 1847.84 | 1851.71 | SL hit (close>static) qty=1.00 sl=1881.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 1882.00 | 1847.84 | 1851.71 | SL hit (close>static) qty=1.00 sl=1881.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1882.00 | 1847.84 | 1851.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1877.00 | 1853.67 | 1854.01 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1867.20 | 1856.38 | 1855.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 1890.00 | 1863.10 | 1858.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 14:15:00 | 1916.90 | 1920.71 | 1905.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 15:00:00 | 1916.90 | 1920.71 | 1905.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1892.80 | 1915.37 | 1905.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 1892.80 | 1915.37 | 1905.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1923.60 | 1917.01 | 1907.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 1931.00 | 1917.01 | 1907.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1929.40 | 1906.62 | 1905.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:00:00 | 1930.70 | 1912.91 | 1908.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 1937.10 | 1917.75 | 1911.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1922.70 | 1926.05 | 1919.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 1916.00 | 1926.05 | 1919.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1924.10 | 1925.66 | 1919.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 1923.80 | 1925.66 | 1919.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1923.10 | 1924.63 | 1920.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 1923.10 | 1924.63 | 1920.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1917.50 | 1923.20 | 1920.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1925.80 | 1923.20 | 1920.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1935.90 | 1925.74 | 1921.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1904.00 | 1918.44 | 1919.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1904.00 | 1918.44 | 1919.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1904.00 | 1918.44 | 1919.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1904.00 | 1918.44 | 1919.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1904.00 | 1918.44 | 1919.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1894.90 | 1913.74 | 1916.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 15:15:00 | 1920.00 | 1914.23 | 1916.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 15:15:00 | 1920.00 | 1914.23 | 1916.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1920.00 | 1914.23 | 1916.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 1899.30 | 1914.23 | 1916.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 1896.40 | 1910.88 | 1914.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 1937.90 | 1897.24 | 1891.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 1937.90 | 1897.24 | 1891.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1937.90 | 1897.24 | 1891.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 1944.20 | 1912.61 | 1900.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 1922.60 | 1926.66 | 1912.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 1916.80 | 1922.40 | 1914.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1916.80 | 1922.40 | 1914.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:30:00 | 1914.10 | 1922.40 | 1914.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1916.00 | 1921.12 | 1914.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:15:00 | 1915.00 | 1921.12 | 1914.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1915.00 | 1919.89 | 1914.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1902.60 | 1919.89 | 1914.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1900.90 | 1916.10 | 1913.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 1898.60 | 1916.10 | 1913.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1885.00 | 1909.88 | 1910.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 1867.40 | 1901.38 | 1906.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 1817.20 | 1801.49 | 1827.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 1817.20 | 1801.49 | 1827.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1803.40 | 1804.84 | 1824.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1794.40 | 1802.54 | 1819.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 1798.70 | 1792.05 | 1803.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 1708.76 | 1765.89 | 1786.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1769.70 | 1741.32 | 1759.23 | SL hit (close>ema200) qty=0.50 sl=1741.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 1766.80 | 1757.09 | 1755.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 1766.80 | 1757.09 | 1755.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 11:15:00 | 1789.90 | 1768.51 | 1762.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 1741.70 | 1777.61 | 1771.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 1741.70 | 1777.61 | 1771.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1741.70 | 1777.61 | 1771.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 1741.70 | 1777.61 | 1771.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 1728.00 | 1767.69 | 1767.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 1728.00 | 1767.69 | 1767.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1738.20 | 1761.79 | 1764.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1694.00 | 1728.38 | 1740.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 1716.00 | 1714.41 | 1728.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:00:00 | 1716.00 | 1714.41 | 1728.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1735.20 | 1719.08 | 1727.32 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1742.40 | 1732.87 | 1732.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1788.90 | 1749.10 | 1740.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 13:15:00 | 1862.20 | 1862.34 | 1833.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 13:45:00 | 1866.20 | 1862.34 | 1833.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1843.00 | 1859.52 | 1839.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:15:00 | 1836.10 | 1859.52 | 1839.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1845.40 | 1856.70 | 1840.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:00:00 | 1862.00 | 1857.76 | 1842.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 1828.80 | 1849.93 | 1841.16 | SL hit (close<static) qty=1.00 sl=1831.10 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 1815.40 | 1834.68 | 1835.63 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1839.10 | 1834.96 | 1834.72 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 14:15:00 | 1828.60 | 1833.53 | 1834.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 09:15:00 | 1824.10 | 1830.44 | 1832.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 10:15:00 | 1833.20 | 1830.99 | 1832.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 10:15:00 | 1833.20 | 1830.99 | 1832.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1833.20 | 1830.99 | 1832.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 1833.20 | 1830.99 | 1832.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 1851.50 | 1835.09 | 1834.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1852.90 | 1844.09 | 1839.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1836.10 | 1844.35 | 1841.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1836.10 | 1844.35 | 1841.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1836.10 | 1844.35 | 1841.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 1836.10 | 1844.35 | 1841.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1845.00 | 1844.48 | 1841.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1817.90 | 1844.48 | 1841.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1818.00 | 1839.18 | 1839.63 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 1873.20 | 1840.92 | 1839.08 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 1817.90 | 1842.13 | 1844.11 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 1847.60 | 1839.86 | 1839.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1864.50 | 1844.79 | 1841.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 1853.10 | 1853.43 | 1847.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:45:00 | 1852.70 | 1853.43 | 1847.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 1850.20 | 1853.16 | 1848.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:30:00 | 1851.10 | 1853.16 | 1848.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1853.60 | 1853.83 | 1849.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 1837.60 | 1853.83 | 1849.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1837.40 | 1850.55 | 1848.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 1839.40 | 1850.55 | 1848.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 1825.00 | 1845.44 | 1846.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 1821.20 | 1840.59 | 1844.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 15:15:00 | 1853.90 | 1839.02 | 1842.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 1853.90 | 1839.02 | 1842.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1853.90 | 1839.02 | 1842.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 1797.90 | 1839.02 | 1842.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1708.01 | 1747.36 | 1755.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 11:15:00 | 1756.90 | 1747.45 | 1754.34 | SL hit (close>ema200) qty=0.50 sl=1747.45 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1773.80 | 1757.96 | 1757.24 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 1749.30 | 1756.29 | 1756.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 1745.00 | 1751.82 | 1754.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 15:15:00 | 1645.00 | 1644.25 | 1668.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:15:00 | 1650.40 | 1644.25 | 1668.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1653.80 | 1649.81 | 1661.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 1663.70 | 1649.81 | 1661.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1665.10 | 1652.87 | 1661.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1681.00 | 1652.87 | 1661.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1695.00 | 1661.29 | 1664.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1697.90 | 1661.29 | 1664.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1688.10 | 1666.66 | 1666.65 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1648.80 | 1671.64 | 1672.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1612.00 | 1652.15 | 1660.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1642.40 | 1632.89 | 1643.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 1642.40 | 1632.89 | 1643.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1642.40 | 1632.89 | 1643.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 1644.50 | 1632.89 | 1643.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 1640.80 | 1634.47 | 1643.21 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 1666.50 | 1648.86 | 1648.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1672.50 | 1655.27 | 1651.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1678.70 | 1690.29 | 1675.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1678.70 | 1690.29 | 1675.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1678.70 | 1690.29 | 1675.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1678.70 | 1690.29 | 1675.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1655.70 | 1683.37 | 1673.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1655.70 | 1683.37 | 1673.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1627.10 | 1672.12 | 1669.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 1627.10 | 1672.12 | 1669.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1622.10 | 1662.11 | 1664.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1609.50 | 1636.82 | 1650.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1630.80 | 1622.15 | 1635.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1630.80 | 1622.15 | 1635.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1630.80 | 1622.15 | 1635.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1618.60 | 1622.15 | 1635.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 1627.00 | 1626.80 | 1633.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1596.90 | 1630.20 | 1634.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 1612.90 | 1618.49 | 1623.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1635.40 | 1622.10 | 1624.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 1635.40 | 1622.10 | 1624.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 1653.30 | 1628.34 | 1626.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 1653.30 | 1628.34 | 1626.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 1653.30 | 1628.34 | 1626.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 1653.30 | 1628.34 | 1626.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1653.30 | 1628.34 | 1626.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 1670.20 | 1640.88 | 1633.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1721.00 | 1724.22 | 1710.08 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1743.40 | 1724.22 | 1710.08 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1748.20 | 1747.89 | 1732.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1754.30 | 1747.89 | 1732.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 1772.20 | 1789.62 | 1777.38 | SL hit (close<ema400) qty=1.00 sl=1777.38 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 1793.20 | 1804.73 | 1806.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1793.20 | 1804.73 | 1806.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1782.20 | 1797.02 | 1801.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1802.00 | 1784.85 | 1791.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1802.00 | 1784.85 | 1791.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1802.00 | 1784.85 | 1791.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1806.20 | 1784.85 | 1791.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1803.90 | 1788.66 | 1792.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1804.60 | 1788.66 | 1792.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1803.10 | 1795.46 | 1795.20 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 1788.20 | 1794.01 | 1794.56 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1799.00 | 1795.01 | 1794.96 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 1786.40 | 1793.29 | 1794.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 10:15:00 | 1783.70 | 1791.37 | 1793.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 11:15:00 | 1776.80 | 1772.73 | 1779.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 12:00:00 | 1776.80 | 1772.73 | 1779.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1775.10 | 1773.21 | 1779.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 1771.70 | 1773.25 | 1778.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 1769.00 | 1773.25 | 1778.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:45:00 | 1768.80 | 1769.92 | 1775.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:00:00 | 1770.50 | 1766.28 | 1771.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1768.30 | 1766.68 | 1771.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 1768.70 | 1766.68 | 1771.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1773.00 | 1767.95 | 1771.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1783.70 | 1767.95 | 1771.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1804.40 | 1775.24 | 1774.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1804.40 | 1775.24 | 1774.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1804.40 | 1775.24 | 1774.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1804.40 | 1775.24 | 1774.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1804.40 | 1775.24 | 1774.57 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 1762.30 | 1777.42 | 1779.05 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 1802.60 | 1782.24 | 1780.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 1822.90 | 1790.37 | 1784.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 1832.10 | 1848.70 | 1831.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 1832.10 | 1848.70 | 1831.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1832.10 | 1848.70 | 1831.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 1832.10 | 1848.70 | 1831.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 1823.80 | 1843.72 | 1830.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 1824.30 | 1843.72 | 1830.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1825.00 | 1839.98 | 1830.05 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 11:15:00 | 1771.10 | 2025-05-14 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-05-21 11:45:00 | 1744.40 | 2025-05-26 09:15:00 | 1822.30 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2025-06-13 12:45:00 | 1886.50 | 2025-06-19 12:15:00 | 1878.70 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-06-20 14:00:00 | 1863.00 | 2025-06-20 14:15:00 | 1903.00 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-06-27 11:15:00 | 2045.90 | 2025-06-27 14:15:00 | 2250.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-27 12:45:00 | 2044.00 | 2025-06-27 14:15:00 | 2248.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-03 12:15:00 | 2044.20 | 2025-07-11 10:15:00 | 2009.90 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-07-18 10:15:00 | 1949.20 | 2025-07-22 09:15:00 | 1971.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-18 13:15:00 | 1940.70 | 2025-07-22 12:15:00 | 1971.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-07-18 14:15:00 | 1934.60 | 2025-07-22 12:15:00 | 1971.50 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-07-21 09:15:00 | 1935.30 | 2025-07-22 12:15:00 | 1971.50 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-07-21 13:45:00 | 1929.90 | 2025-07-22 12:15:00 | 1971.50 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-07-28 09:15:00 | 1999.60 | 2025-07-29 09:15:00 | 1972.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-07-31 11:15:00 | 1943.00 | 2025-08-04 09:15:00 | 1845.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 11:45:00 | 1944.50 | 2025-08-04 09:15:00 | 1847.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 12:30:00 | 1941.00 | 2025-08-04 09:15:00 | 1843.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 13:30:00 | 1935.70 | 2025-08-04 09:15:00 | 1838.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-04 09:15:00 | 1861.30 | 2025-08-06 09:15:00 | 1768.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 11:15:00 | 1943.00 | 2025-08-07 09:15:00 | 1748.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 11:45:00 | 1944.50 | 2025-08-07 09:15:00 | 1750.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 12:30:00 | 1941.00 | 2025-08-07 09:15:00 | 1746.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 13:30:00 | 1935.70 | 2025-08-07 09:15:00 | 1742.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-04 09:15:00 | 1861.30 | 2025-08-07 12:15:00 | 1780.30 | STOP_HIT | 0.50 | 4.35% |
| BUY | retest2 | 2025-08-18 09:30:00 | 1831.60 | 2025-08-22 10:15:00 | 1823.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-01 15:15:00 | 1782.90 | 2025-09-01 15:15:00 | 1782.90 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-09-09 11:30:00 | 1763.40 | 2025-09-15 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-11 09:30:00 | 1763.50 | 2025-09-15 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-11 10:00:00 | 1754.10 | 2025-09-15 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-09-12 13:00:00 | 1764.10 | 2025-09-15 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-09-19 09:15:00 | 1793.10 | 2025-09-19 09:15:00 | 1777.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1783.60 | 2025-10-08 11:15:00 | 1771.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-10-15 12:15:00 | 1752.00 | 2025-10-16 09:15:00 | 1768.40 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-16 11:30:00 | 1754.10 | 2025-10-16 13:15:00 | 1764.10 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-16 12:30:00 | 1755.20 | 2025-10-16 13:15:00 | 1764.10 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-20 09:15:00 | 1766.00 | 2025-10-23 11:15:00 | 1756.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-11-10 10:00:00 | 1824.10 | 2025-11-11 09:15:00 | 1781.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-11-14 11:45:00 | 1757.10 | 2025-11-17 09:15:00 | 1921.90 | STOP_HIT | 1.00 | -9.38% |
| SELL | retest2 | 2025-11-14 13:00:00 | 1756.30 | 2025-11-17 09:15:00 | 1921.90 | STOP_HIT | 1.00 | -9.43% |
| BUY | retest2 | 2025-11-21 12:00:00 | 2035.40 | 2025-11-24 13:15:00 | 1981.80 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-11-21 13:00:00 | 2052.00 | 2025-11-24 13:15:00 | 1981.80 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-11-26 14:00:00 | 1959.90 | 2025-12-03 14:15:00 | 1956.60 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-11-27 13:30:00 | 1952.00 | 2025-12-03 14:15:00 | 1956.60 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-12-01 09:30:00 | 1959.80 | 2025-12-03 14:15:00 | 1956.60 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-12-03 13:00:00 | 1948.80 | 2025-12-03 14:15:00 | 1956.60 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-12-09 10:45:00 | 1891.50 | 2025-12-15 13:15:00 | 1901.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-16 11:15:00 | 1919.10 | 2025-12-17 09:15:00 | 1872.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-12-26 14:15:00 | 1865.00 | 2025-12-31 10:15:00 | 1882.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-29 09:45:00 | 1866.00 | 2025-12-31 10:15:00 | 1882.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-29 10:30:00 | 1863.30 | 2025-12-31 10:15:00 | 1882.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-12-30 14:45:00 | 1833.50 | 2025-12-31 10:15:00 | 1882.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-01-05 11:15:00 | 1931.00 | 2026-01-08 12:15:00 | 1904.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1929.40 | 2026-01-08 12:15:00 | 1904.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-01-06 12:00:00 | 1930.70 | 2026-01-08 12:15:00 | 1904.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-01-06 13:00:00 | 1937.10 | 2026-01-08 12:15:00 | 1904.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-01-09 09:15:00 | 1899.30 | 2026-01-14 10:15:00 | 1937.90 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-01-09 13:45:00 | 1896.40 | 2026-01-14 10:15:00 | 1937.90 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1794.40 | 2026-01-27 09:15:00 | 1708.76 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1794.40 | 2026-01-28 09:15:00 | 1769.70 | STOP_HIT | 0.50 | 1.38% |
| SELL | retest2 | 2026-01-23 11:30:00 | 1798.70 | 2026-01-30 12:15:00 | 1766.80 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2026-02-13 12:00:00 | 1862.00 | 2026-02-13 13:15:00 | 1828.80 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1797.90 | 2026-03-09 09:15:00 | 1708.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1797.90 | 2026-03-09 11:15:00 | 1756.90 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1618.60 | 2026-04-06 11:15:00 | 1653.30 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-04-01 13:30:00 | 1627.00 | 2026-04-06 11:15:00 | 1653.30 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1596.90 | 2026-04-06 11:15:00 | 1653.30 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-04-06 09:15:00 | 1612.90 | 2026-04-06 11:15:00 | 1653.30 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1743.40 | 2026-04-16 11:15:00 | 1772.20 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1754.30 | 2026-04-23 11:15:00 | 1793.20 | STOP_HIT | 1.00 | 2.22% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1771.70 | 2026-05-04 09:15:00 | 1804.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-04-29 15:15:00 | 1769.00 | 2026-05-04 09:15:00 | 1804.40 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-04-30 09:45:00 | 1768.80 | 2026-05-04 09:15:00 | 1804.40 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-04-30 14:00:00 | 1770.50 | 2026-05-04 09:15:00 | 1804.40 | STOP_HIT | 1.00 | -1.91% |
