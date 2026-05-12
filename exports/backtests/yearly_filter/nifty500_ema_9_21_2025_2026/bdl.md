# Bharat Dynamics Ltd. (BDL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1447.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 60 |
| ALERT1 | 48 |
| ALERT2 | 48 |
| ALERT2_SKIP | 24 |
| ALERT3 | 111 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 64 |
| PARTIAL | 18 |
| TARGET_HIT | 0 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 42
- **Target hits / Stop hits / Partials:** 0 / 67 / 18
- **Avg / median % per leg:** 1.17% / 0.49%
- **Sum % (uncompounded):** 99.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 10 | 28.6% | 0 | 35 | 0 | -0.62% | -21.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.69% | -2.1% |
| BUY @ 3rd Alert (retest2) | 32 | 10 | 31.2% | 0 | 32 | 0 | -0.61% | -19.5% |
| SELL (all) | 50 | 33 | 66.0% | 0 | 32 | 18 | 2.42% | 121.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 50 | 33 | 66.0% | 0 | 32 | 18 | 2.42% | 121.1% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.69% | -2.1% |
| retest2 (combined) | 82 | 43 | 52.4% | 0 | 64 | 18 | 1.24% | 101.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 1767.50 | 1799.53 | 1800.76 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 1813.10 | 1802.24 | 1801.88 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1798.50 | 1801.15 | 1801.42 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 1845.90 | 1808.64 | 1804.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 14:15:00 | 1881.40 | 1843.96 | 1824.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 1886.90 | 1895.61 | 1869.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 09:45:00 | 1886.10 | 1895.61 | 1869.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1894.00 | 1908.41 | 1891.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 1890.50 | 1908.41 | 1891.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1906.00 | 1907.92 | 1892.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 11:15:00 | 1917.00 | 1907.92 | 1892.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 12:00:00 | 1917.60 | 1909.86 | 1895.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 12:45:00 | 1914.50 | 1910.47 | 1896.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 1960.60 | 1910.24 | 1900.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1959.40 | 1920.07 | 1905.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 2003.00 | 1947.67 | 1935.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:45:00 | 1992.50 | 2010.85 | 1987.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 10:45:00 | 1993.00 | 2005.70 | 1987.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 2011.00 | 1988.38 | 1984.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1991.40 | 1993.61 | 1988.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:45:00 | 1989.10 | 1993.61 | 1988.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1971.80 | 1989.25 | 1986.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 1971.80 | 1989.25 | 1986.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 1975.00 | 1986.40 | 1985.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:45:00 | 1971.70 | 1986.40 | 1985.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-03 14:15:00 | 1978.70 | 1984.86 | 1985.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 1978.70 | 1984.86 | 1985.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 1953.90 | 1977.66 | 1981.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 12:15:00 | 1950.50 | 1949.16 | 1961.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 12:15:00 | 1950.50 | 1949.16 | 1961.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 1950.50 | 1949.16 | 1961.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 15:15:00 | 1932.00 | 1947.14 | 1954.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 1926.50 | 1943.23 | 1950.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 13:00:00 | 1932.00 | 1940.08 | 1948.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 12:15:00 | 1957.10 | 1949.54 | 1949.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 12:15:00 | 1957.10 | 1949.54 | 1949.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 1969.10 | 1954.36 | 1951.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 1937.80 | 1953.07 | 1951.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1937.80 | 1953.07 | 1951.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1937.80 | 1953.07 | 1951.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 1937.80 | 1953.07 | 1951.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 1934.10 | 1949.28 | 1950.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 1924.00 | 1940.90 | 1945.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 1919.00 | 1892.74 | 1908.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 1919.00 | 1892.74 | 1908.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1919.00 | 1892.74 | 1908.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:45:00 | 1923.00 | 1892.74 | 1908.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 1921.10 | 1898.41 | 1909.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 1923.90 | 1898.41 | 1909.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 1894.00 | 1898.60 | 1908.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1872.90 | 1899.85 | 1906.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 1917.20 | 1889.07 | 1892.74 | SL hit (close>static) qty=1.00 sl=1909.70 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 1937.50 | 1898.76 | 1896.81 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 1876.70 | 1902.42 | 1904.06 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 1915.00 | 1898.62 | 1898.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1947.00 | 1912.66 | 1904.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 1858.90 | 1913.26 | 1909.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1858.90 | 1913.26 | 1909.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1858.90 | 1913.26 | 1909.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 1858.90 | 1913.26 | 1909.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 10:15:00 | 1862.20 | 1903.05 | 1905.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 09:15:00 | 1839.20 | 1867.07 | 1884.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 1827.00 | 1823.11 | 1840.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 14:45:00 | 1829.00 | 1823.11 | 1840.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1854.00 | 1829.89 | 1840.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 1846.60 | 1829.89 | 1840.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1863.00 | 1836.51 | 1842.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:45:00 | 1868.90 | 1836.51 | 1842.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 12:15:00 | 1866.90 | 1847.10 | 1846.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 13:15:00 | 1876.00 | 1852.88 | 1849.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 1971.60 | 1971.99 | 1942.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 09:30:00 | 1957.00 | 1971.99 | 1942.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1969.30 | 1972.52 | 1952.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 15:00:00 | 1975.00 | 1973.02 | 1954.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 12:30:00 | 1976.70 | 1968.88 | 1959.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1993.20 | 1967.99 | 1961.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 1944.80 | 1966.72 | 1967.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 1944.80 | 1966.72 | 1967.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 1938.00 | 1953.10 | 1959.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 1954.30 | 1950.88 | 1956.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 1954.30 | 1950.88 | 1956.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1954.30 | 1950.88 | 1956.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 1956.40 | 1950.88 | 1956.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1958.80 | 1952.46 | 1956.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 1961.90 | 1952.46 | 1956.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1973.50 | 1956.67 | 1958.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 1973.50 | 1956.67 | 1958.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 1970.00 | 1959.34 | 1959.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 1983.00 | 1964.07 | 1961.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1946.30 | 1973.03 | 1969.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1946.30 | 1973.03 | 1969.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1946.30 | 1973.03 | 1969.28 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 1927.10 | 1963.85 | 1965.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 1918.60 | 1954.80 | 1961.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 1688.20 | 1686.98 | 1723.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 1688.20 | 1686.98 | 1723.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1717.50 | 1699.59 | 1720.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 1717.50 | 1699.59 | 1720.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1722.00 | 1704.07 | 1720.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1721.00 | 1704.07 | 1720.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1729.60 | 1709.18 | 1721.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:15:00 | 1701.30 | 1717.85 | 1721.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:00:00 | 1704.30 | 1709.39 | 1716.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 1696.80 | 1704.11 | 1711.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 1616.23 | 1637.40 | 1659.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 1619.08 | 1637.40 | 1659.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 1611.96 | 1637.40 | 1659.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 1628.00 | 1619.94 | 1639.45 | SL hit (close>ema200) qty=0.50 sl=1619.94 alert=retest2 |

### Cycle 16 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 1559.40 | 1526.46 | 1522.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 1587.00 | 1538.57 | 1528.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1567.40 | 1593.03 | 1580.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1567.40 | 1593.03 | 1580.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1567.40 | 1593.03 | 1580.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 1567.80 | 1593.03 | 1580.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1569.00 | 1588.23 | 1579.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:45:00 | 1564.00 | 1588.23 | 1579.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 1551.00 | 1573.32 | 1574.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 15:15:00 | 1549.80 | 1565.03 | 1570.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1543.50 | 1540.46 | 1551.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:45:00 | 1544.40 | 1540.46 | 1551.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1523.60 | 1522.54 | 1530.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 1518.40 | 1523.57 | 1528.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1499.60 | 1523.04 | 1527.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:15:00 | 1442.48 | 1460.52 | 1482.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1424.62 | 1449.87 | 1473.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1449.80 | 1442.55 | 1457.11 | SL hit (close>ema200) qty=0.50 sl=1442.55 alert=retest2 |

### Cycle 18 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 1476.50 | 1461.59 | 1460.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 1486.90 | 1466.65 | 1462.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1464.80 | 1469.73 | 1465.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 10:15:00 | 1464.80 | 1469.73 | 1465.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1464.80 | 1469.73 | 1465.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 1464.80 | 1469.73 | 1465.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1461.00 | 1467.98 | 1465.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:00:00 | 1461.00 | 1467.98 | 1465.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1467.00 | 1465.54 | 1464.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 1459.70 | 1465.54 | 1464.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 1456.00 | 1463.63 | 1464.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 10:15:00 | 1446.50 | 1460.20 | 1462.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 1445.80 | 1438.23 | 1445.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 1445.80 | 1438.23 | 1445.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1445.80 | 1438.23 | 1445.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 1445.80 | 1438.23 | 1445.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1441.40 | 1438.86 | 1445.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 1450.40 | 1438.86 | 1445.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 1450.40 | 1441.17 | 1445.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 1443.70 | 1441.17 | 1445.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1452.40 | 1443.42 | 1446.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1439.20 | 1444.98 | 1446.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 1437.90 | 1443.56 | 1445.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 1481.10 | 1447.39 | 1443.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 1481.10 | 1447.39 | 1443.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 12:15:00 | 1485.50 | 1455.01 | 1447.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 1477.00 | 1482.50 | 1468.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 1477.00 | 1482.50 | 1468.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1617.40 | 1617.01 | 1602.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1631.50 | 1611.79 | 1604.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 1621.00 | 1617.39 | 1608.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 1621.90 | 1618.99 | 1611.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 1600.00 | 1620.79 | 1618.93 | SL hit (close<static) qty=1.00 sl=1601.20 alert=retest2 |

### Cycle 21 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 1583.80 | 1613.39 | 1615.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 1580.00 | 1606.71 | 1612.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1582.60 | 1575.14 | 1586.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1582.60 | 1575.14 | 1586.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1582.60 | 1575.14 | 1586.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:15:00 | 1599.00 | 1575.14 | 1586.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1598.00 | 1579.72 | 1587.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 1598.00 | 1579.72 | 1587.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1584.60 | 1580.69 | 1587.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:45:00 | 1582.70 | 1580.77 | 1586.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1503.57 | 1528.90 | 1552.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 1495.00 | 1493.59 | 1509.39 | SL hit (close>ema200) qty=0.50 sl=1493.59 alert=retest2 |

### Cycle 22 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1541.40 | 1515.97 | 1514.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1567.90 | 1531.00 | 1522.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 1554.80 | 1555.15 | 1543.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:45:00 | 1552.80 | 1555.15 | 1543.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1544.40 | 1554.15 | 1547.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 1544.40 | 1554.15 | 1547.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1538.00 | 1550.92 | 1546.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1538.00 | 1550.92 | 1546.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1529.50 | 1546.63 | 1545.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 1529.50 | 1546.63 | 1545.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 1532.60 | 1543.83 | 1543.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1509.60 | 1531.53 | 1537.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1504.00 | 1500.77 | 1511.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:00:00 | 1504.00 | 1500.77 | 1511.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1519.00 | 1504.42 | 1512.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1519.00 | 1504.42 | 1512.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1524.10 | 1508.36 | 1513.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1519.00 | 1508.36 | 1513.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1532.00 | 1517.30 | 1517.06 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 1502.10 | 1517.33 | 1518.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1484.30 | 1505.01 | 1510.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1508.80 | 1496.61 | 1502.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 1508.80 | 1496.61 | 1502.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1508.80 | 1496.61 | 1502.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1506.80 | 1496.61 | 1502.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1511.80 | 1499.65 | 1503.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 1513.60 | 1499.65 | 1503.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1502.60 | 1503.25 | 1504.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 1496.10 | 1501.82 | 1503.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1546.70 | 1511.57 | 1507.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1546.70 | 1511.57 | 1507.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 1558.00 | 1520.86 | 1511.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 10:15:00 | 1534.10 | 1535.28 | 1525.45 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 12:30:00 | 1542.20 | 1537.69 | 1528.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-21 13:45:00 | 1547.30 | 1538.30 | 1531.45 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:15:00 | 1544.60 | 1538.64 | 1532.23 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1540.60 | 1539.03 | 1532.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-23 13:15:00 | 1534.00 | 1538.90 | 1535.01 | SL hit (close<ema400) qty=1.00 sl=1535.01 alert=retest1 |

### Cycle 27 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 1515.00 | 1529.88 | 1531.32 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 1550.20 | 1533.95 | 1533.04 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1526.80 | 1536.01 | 1536.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 1521.60 | 1531.65 | 1534.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 13:15:00 | 1522.60 | 1519.06 | 1524.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 1522.60 | 1519.06 | 1524.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1522.60 | 1519.06 | 1524.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:45:00 | 1512.30 | 1519.66 | 1523.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:30:00 | 1511.00 | 1518.04 | 1521.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 14:45:00 | 1511.70 | 1517.54 | 1521.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 1536.40 | 1521.05 | 1522.29 | SL hit (close>static) qty=1.00 sl=1529.40 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 1531.00 | 1523.62 | 1523.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 1543.10 | 1527.51 | 1525.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 1526.10 | 1527.23 | 1525.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 1526.10 | 1527.23 | 1525.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1534.10 | 1528.60 | 1525.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1522.10 | 1528.60 | 1525.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1518.60 | 1526.60 | 1525.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 1524.50 | 1526.60 | 1525.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1520.20 | 1525.32 | 1524.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1517.80 | 1525.32 | 1524.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 1517.70 | 1523.80 | 1524.19 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 1536.80 | 1526.46 | 1525.30 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1506.70 | 1522.79 | 1524.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1501.60 | 1518.55 | 1522.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 1446.60 | 1441.03 | 1458.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:30:00 | 1453.40 | 1441.03 | 1458.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1502.80 | 1454.18 | 1461.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 1500.30 | 1454.18 | 1461.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1505.60 | 1464.46 | 1465.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1500.00 | 1464.46 | 1465.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1509.10 | 1473.39 | 1469.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 1517.90 | 1488.21 | 1476.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 14:15:00 | 1534.40 | 1537.49 | 1523.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 14:15:00 | 1534.40 | 1537.49 | 1523.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1534.40 | 1537.49 | 1523.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:30:00 | 1543.00 | 1535.33 | 1524.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 12:15:00 | 1521.00 | 1528.49 | 1523.68 | SL hit (close<static) qty=1.00 sl=1521.20 alert=retest2 |

### Cycle 35 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 1558.00 | 1572.54 | 1574.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1546.00 | 1564.68 | 1570.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1563.20 | 1548.91 | 1557.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1563.20 | 1548.91 | 1557.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1563.20 | 1548.91 | 1557.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1563.20 | 1548.91 | 1557.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1561.00 | 1551.33 | 1557.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:15:00 | 1558.20 | 1551.33 | 1557.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 1559.20 | 1554.85 | 1558.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:45:00 | 1560.00 | 1555.14 | 1557.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 1482.00 | 1513.90 | 1531.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 1480.29 | 1495.34 | 1516.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 1481.24 | 1495.34 | 1516.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1479.00 | 1474.82 | 1490.75 | SL hit (close>ema200) qty=0.50 sl=1474.82 alert=retest2 |

### Cycle 36 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 1495.80 | 1492.20 | 1491.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 15:15:00 | 1507.00 | 1498.01 | 1494.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 1530.00 | 1530.68 | 1520.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:00:00 | 1530.00 | 1530.68 | 1520.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1522.80 | 1529.10 | 1520.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 1525.00 | 1529.10 | 1520.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 1522.60 | 1527.80 | 1520.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 1529.00 | 1528.04 | 1521.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 1505.40 | 1522.50 | 1520.17 | SL hit (close<static) qty=1.00 sl=1517.30 alert=retest2 |

### Cycle 37 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 1491.30 | 1516.26 | 1517.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 11:15:00 | 1484.80 | 1509.97 | 1514.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1496.40 | 1495.71 | 1504.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:30:00 | 1498.50 | 1495.71 | 1504.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1501.30 | 1496.83 | 1504.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:45:00 | 1502.90 | 1496.83 | 1504.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1513.50 | 1500.16 | 1505.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 1513.50 | 1500.16 | 1505.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1514.00 | 1502.93 | 1505.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 1516.30 | 1502.93 | 1505.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 1526.40 | 1509.32 | 1508.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 15:15:00 | 1537.90 | 1515.03 | 1511.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 1514.00 | 1514.83 | 1511.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1514.00 | 1514.83 | 1511.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1514.00 | 1514.83 | 1511.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 1500.00 | 1514.83 | 1511.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1522.10 | 1516.28 | 1512.31 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1472.20 | 1505.94 | 1509.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 1461.30 | 1497.01 | 1504.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 1430.20 | 1428.89 | 1452.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 1430.20 | 1428.89 | 1452.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1414.50 | 1411.03 | 1422.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:30:00 | 1425.30 | 1411.03 | 1422.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1426.20 | 1415.06 | 1421.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:30:00 | 1418.50 | 1414.67 | 1420.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:15:00 | 1347.58 | 1362.37 | 1381.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 1333.70 | 1330.39 | 1349.64 | SL hit (close>ema200) qty=0.50 sl=1330.39 alert=retest2 |

### Cycle 40 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1373.60 | 1353.26 | 1351.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1387.00 | 1363.17 | 1356.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 1486.30 | 1487.93 | 1463.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 14:00:00 | 1486.30 | 1487.93 | 1463.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1467.90 | 1484.64 | 1474.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 1467.90 | 1484.64 | 1474.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1470.20 | 1481.75 | 1473.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 1478.00 | 1481.75 | 1473.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 1456.70 | 1476.14 | 1472.58 | SL hit (close<static) qty=1.00 sl=1462.40 alert=retest2 |

### Cycle 41 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 1441.30 | 1469.17 | 1469.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 1430.80 | 1461.50 | 1466.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1468.00 | 1458.57 | 1462.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 1468.00 | 1458.57 | 1462.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1468.00 | 1458.57 | 1462.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1475.00 | 1458.57 | 1462.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1465.10 | 1459.87 | 1463.01 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1469.90 | 1464.98 | 1464.86 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 1451.30 | 1462.54 | 1463.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 1447.70 | 1456.20 | 1460.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 1484.50 | 1461.86 | 1462.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 14:15:00 | 1484.50 | 1461.86 | 1462.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1484.50 | 1461.86 | 1462.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1484.50 | 1461.86 | 1462.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 15:15:00 | 1485.60 | 1466.61 | 1464.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1497.20 | 1472.73 | 1467.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 12:15:00 | 1541.70 | 1542.91 | 1522.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 12:45:00 | 1547.60 | 1542.91 | 1522.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1538.90 | 1540.75 | 1527.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 1557.20 | 1543.15 | 1534.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 1560.40 | 1543.15 | 1534.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 1563.30 | 1544.61 | 1539.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 1520.00 | 1539.34 | 1539.09 | SL hit (close<static) qty=1.00 sl=1526.50 alert=retest2 |

### Cycle 45 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 1522.00 | 1535.87 | 1537.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1508.30 | 1527.82 | 1533.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 1531.70 | 1524.25 | 1529.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 13:15:00 | 1531.70 | 1524.25 | 1529.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1531.70 | 1524.25 | 1529.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:45:00 | 1530.60 | 1524.25 | 1529.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1530.20 | 1525.44 | 1529.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1541.50 | 1525.44 | 1529.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1535.00 | 1527.35 | 1529.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1531.30 | 1527.35 | 1529.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1524.00 | 1517.54 | 1522.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 1524.00 | 1517.54 | 1522.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1520.20 | 1518.07 | 1522.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 1529.40 | 1518.07 | 1522.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1520.40 | 1518.54 | 1522.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 1510.70 | 1522.44 | 1523.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 1513.80 | 1520.71 | 1522.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1511.80 | 1519.67 | 1521.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 1511.30 | 1519.88 | 1521.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1517.10 | 1516.24 | 1519.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 1517.00 | 1516.24 | 1519.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1512.50 | 1515.28 | 1518.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:15:00 | 1502.00 | 1509.43 | 1513.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1435.16 | 1463.71 | 1483.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1438.11 | 1463.71 | 1483.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1436.21 | 1463.71 | 1483.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1435.73 | 1463.71 | 1483.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1426.90 | 1463.71 | 1483.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1454.10 | 1437.41 | 1457.12 | SL hit (close>ema200) qty=0.50 sl=1437.41 alert=retest2 |

### Cycle 46 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 1465.40 | 1442.78 | 1442.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 1479.40 | 1450.10 | 1445.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 13:15:00 | 1537.60 | 1538.22 | 1512.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 14:00:00 | 1537.60 | 1538.22 | 1512.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1547.50 | 1537.61 | 1518.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:00:00 | 1551.10 | 1540.31 | 1521.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:00:00 | 1550.40 | 1542.33 | 1524.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 1440.20 | 1509.29 | 1516.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1440.20 | 1509.29 | 1516.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 1397.60 | 1474.99 | 1498.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 1342.80 | 1333.47 | 1378.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 11:30:00 | 1341.80 | 1333.47 | 1378.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1279.30 | 1267.79 | 1281.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 1280.90 | 1267.79 | 1281.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1285.40 | 1271.32 | 1282.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 1286.60 | 1271.32 | 1282.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 1290.00 | 1275.05 | 1282.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 1290.00 | 1275.05 | 1282.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 1286.30 | 1277.30 | 1283.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:15:00 | 1288.50 | 1277.30 | 1283.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 1295.90 | 1281.02 | 1284.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 1293.60 | 1281.02 | 1284.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 1304.00 | 1289.07 | 1287.62 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1278.60 | 1289.38 | 1290.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 11:15:00 | 1273.70 | 1286.24 | 1288.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 1283.40 | 1282.64 | 1286.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 15:00:00 | 1283.40 | 1282.64 | 1286.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 1280.60 | 1274.60 | 1279.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1255.00 | 1273.70 | 1278.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 1268.10 | 1259.95 | 1258.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 1268.10 | 1259.95 | 1258.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 1276.80 | 1265.25 | 1261.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1273.50 | 1284.80 | 1278.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1273.50 | 1284.80 | 1278.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1273.50 | 1284.80 | 1278.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 1273.50 | 1284.80 | 1278.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1270.00 | 1281.84 | 1277.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 1285.40 | 1281.84 | 1277.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 1262.90 | 1288.80 | 1288.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 1262.90 | 1288.80 | 1288.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 1252.00 | 1273.51 | 1280.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1253.80 | 1251.75 | 1263.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 1253.80 | 1251.75 | 1263.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1252.70 | 1246.38 | 1254.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:15:00 | 1258.00 | 1246.38 | 1254.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1262.30 | 1249.56 | 1255.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 1262.30 | 1249.56 | 1255.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1263.00 | 1252.25 | 1255.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:15:00 | 1259.10 | 1252.25 | 1255.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1258.80 | 1253.68 | 1256.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 14:15:00 | 1273.00 | 1259.51 | 1258.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 1273.00 | 1259.51 | 1258.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 09:15:00 | 1292.90 | 1268.97 | 1263.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 12:15:00 | 1266.40 | 1270.46 | 1266.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 12:15:00 | 1266.40 | 1270.46 | 1266.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 1266.40 | 1270.46 | 1266.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:45:00 | 1265.60 | 1270.46 | 1266.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 1268.70 | 1270.11 | 1266.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:45:00 | 1253.00 | 1270.11 | 1266.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1270.50 | 1270.19 | 1266.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:30:00 | 1267.90 | 1270.19 | 1266.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1270.50 | 1270.25 | 1267.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 1269.30 | 1270.25 | 1267.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1278.50 | 1271.90 | 1268.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 11:15:00 | 1303.80 | 1277.99 | 1272.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 12:15:00 | 1296.50 | 1280.33 | 1274.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 1324.60 | 1283.73 | 1278.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1302.90 | 1338.23 | 1342.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1302.90 | 1338.23 | 1342.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 1293.20 | 1315.37 | 1327.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1290.10 | 1287.34 | 1303.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 13:15:00 | 1303.50 | 1290.73 | 1299.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1303.50 | 1290.73 | 1299.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 1303.50 | 1290.73 | 1299.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1298.90 | 1292.37 | 1299.66 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1313.00 | 1302.96 | 1302.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 14:15:00 | 1321.10 | 1308.27 | 1305.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1282.60 | 1304.37 | 1303.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1282.60 | 1304.37 | 1303.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1282.60 | 1304.37 | 1303.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1282.60 | 1304.37 | 1303.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1283.00 | 1300.10 | 1302.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 1272.60 | 1291.45 | 1297.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1179.80 | 1175.32 | 1204.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 1180.90 | 1175.32 | 1204.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1196.60 | 1179.67 | 1197.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 1199.10 | 1179.67 | 1197.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1163.80 | 1177.89 | 1188.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 10:15:00 | 1158.00 | 1177.89 | 1188.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 12:15:00 | 1160.10 | 1170.10 | 1182.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 13:00:00 | 1154.50 | 1166.98 | 1180.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 13:15:00 | 1102.09 | 1125.00 | 1148.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 14:15:00 | 1100.10 | 1119.98 | 1144.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 14:15:00 | 1096.77 | 1119.98 | 1144.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 1193.90 | 1130.29 | 1144.24 | SL hit (close>ema200) qty=0.50 sl=1130.29 alert=retest2 |

### Cycle 56 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1207.30 | 1152.25 | 1152.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1221.40 | 1198.48 | 1185.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1318.80 | 1336.73 | 1315.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1318.80 | 1336.73 | 1315.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1318.80 | 1336.73 | 1315.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 1338.60 | 1336.27 | 1319.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 1336.30 | 1336.28 | 1320.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 1338.90 | 1335.94 | 1321.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1344.80 | 1334.94 | 1323.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 1348.50 | 1352.59 | 1345.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 1353.80 | 1353.18 | 1346.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 12:15:00 | 1389.00 | 1394.73 | 1395.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 1389.00 | 1394.73 | 1395.19 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 15:15:00 | 1401.30 | 1395.18 | 1395.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1410.40 | 1398.22 | 1396.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1392.50 | 1400.37 | 1398.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1392.50 | 1400.37 | 1398.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1392.50 | 1400.37 | 1398.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1392.50 | 1400.37 | 1398.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1392.50 | 1398.80 | 1397.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:45:00 | 1385.50 | 1398.80 | 1397.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1363.50 | 1391.71 | 1394.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 1359.70 | 1385.30 | 1391.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 1373.00 | 1370.97 | 1380.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 1368.60 | 1370.97 | 1380.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1382.00 | 1373.18 | 1381.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 1382.00 | 1373.18 | 1381.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1385.00 | 1375.54 | 1381.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 1385.00 | 1375.54 | 1381.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1371.80 | 1374.79 | 1380.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 1387.70 | 1374.79 | 1380.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 1377.00 | 1373.42 | 1377.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 1379.50 | 1376.30 | 1378.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1378.80 | 1376.80 | 1378.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1375.30 | 1376.80 | 1378.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1396.50 | 1381.43 | 1380.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 1396.50 | 1381.43 | 1380.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1413.60 | 1394.20 | 1387.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 1444.00 | 1445.63 | 1427.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:45:00 | 1443.10 | 1445.63 | 1427.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-26 11:15:00 | 1917.00 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2025-05-26 12:00:00 | 1917.60 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | 3.19% |
| BUY | retest2 | 2025-05-26 12:45:00 | 1914.50 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | 3.35% |
| BUY | retest2 | 2025-05-27 09:15:00 | 1960.60 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2025-05-30 09:15:00 | 2003.00 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-06-02 09:45:00 | 1992.50 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-06-02 10:45:00 | 1993.00 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-03 09:15:00 | 2011.00 | 2025-06-03 14:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-06-06 15:15:00 | 1932.00 | 2025-06-10 12:15:00 | 1957.10 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-09 10:45:00 | 1926.50 | 2025-06-10 12:15:00 | 1957.10 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-06-09 13:00:00 | 1932.00 | 2025-06-10 12:15:00 | 1957.10 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1872.90 | 2025-06-17 09:15:00 | 1917.20 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-07-02 15:00:00 | 1975.00 | 2025-07-07 10:15:00 | 1944.80 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-07-03 12:30:00 | 1976.70 | 2025-07-07 10:15:00 | 1944.80 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-07-04 09:15:00 | 1993.20 | 2025-07-07 10:15:00 | 1944.80 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-07-23 10:15:00 | 1701.30 | 2025-07-28 13:15:00 | 1616.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 13:00:00 | 1704.30 | 2025-07-28 13:15:00 | 1619.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 09:45:00 | 1696.80 | 2025-07-28 13:15:00 | 1611.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 10:15:00 | 1701.30 | 2025-07-29 12:15:00 | 1628.00 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2025-07-23 13:00:00 | 1704.30 | 2025-07-29 12:15:00 | 1628.00 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2025-07-24 09:45:00 | 1696.80 | 2025-07-29 12:15:00 | 1628.00 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2025-08-25 14:15:00 | 1518.40 | 2025-08-28 14:15:00 | 1442.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1499.60 | 2025-08-29 09:15:00 | 1424.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 14:15:00 | 1518.40 | 2025-09-01 09:15:00 | 1449.80 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1499.60 | 2025-09-01 09:15:00 | 1449.80 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-09-09 09:15:00 | 1439.20 | 2025-09-10 11:15:00 | 1481.10 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-09-09 10:00:00 | 1437.90 | 2025-09-10 11:15:00 | 1481.10 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-09-19 09:15:00 | 1631.50 | 2025-09-23 09:15:00 | 1600.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-09-19 11:15:00 | 1621.00 | 2025-09-23 09:15:00 | 1600.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-19 13:45:00 | 1621.90 | 2025-09-23 09:15:00 | 1600.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-09-25 12:45:00 | 1582.70 | 2025-09-26 14:15:00 | 1503.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:45:00 | 1582.70 | 2025-09-30 15:15:00 | 1495.00 | STOP_HIT | 0.50 | 5.54% |
| SELL | retest2 | 2025-10-16 11:00:00 | 1496.10 | 2025-10-17 09:15:00 | 1546.70 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest1 | 2025-10-20 12:30:00 | 1542.20 | 2025-10-23 13:15:00 | 1534.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-10-21 13:45:00 | 1547.30 | 2025-10-23 13:15:00 | 1534.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest1 | 2025-10-23 09:15:00 | 1544.60 | 2025-10-23 13:15:00 | 1534.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-30 12:45:00 | 1512.30 | 2025-10-31 09:15:00 | 1536.40 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-10-30 13:30:00 | 1511.00 | 2025-10-31 09:15:00 | 1536.40 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-10-30 14:45:00 | 1511.70 | 2025-10-31 09:15:00 | 1536.40 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-13 09:30:00 | 1543.00 | 2025-11-13 12:15:00 | 1521.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-11-14 09:15:00 | 1619.30 | 2025-11-18 14:15:00 | 1558.00 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-11-20 11:15:00 | 1558.20 | 2025-11-24 10:15:00 | 1482.00 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-11-20 12:30:00 | 1559.20 | 2025-11-24 14:15:00 | 1480.29 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-11-20 14:45:00 | 1560.00 | 2025-11-24 14:15:00 | 1481.24 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-11-20 11:15:00 | 1558.20 | 2025-11-26 09:15:00 | 1479.00 | STOP_HIT | 0.50 | 5.08% |
| SELL | retest2 | 2025-11-20 12:30:00 | 1559.20 | 2025-11-26 09:15:00 | 1479.00 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2025-11-20 14:45:00 | 1560.00 | 2025-11-26 09:15:00 | 1479.00 | STOP_HIT | 0.50 | 5.19% |
| BUY | retest2 | 2025-12-02 15:00:00 | 1529.00 | 2025-12-03 09:15:00 | 1505.40 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-12-12 10:30:00 | 1418.50 | 2025-12-17 09:15:00 | 1347.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:30:00 | 1418.50 | 2025-12-18 11:15:00 | 1333.70 | STOP_HIT | 0.50 | 5.98% |
| BUY | retest2 | 2025-12-29 15:15:00 | 1478.00 | 2025-12-30 09:15:00 | 1456.70 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-01-08 09:30:00 | 1557.20 | 2026-01-09 13:15:00 | 1520.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-01-08 10:00:00 | 1560.40 | 2026-01-09 13:15:00 | 1520.00 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2026-01-09 09:45:00 | 1563.30 | 2026-01-09 13:15:00 | 1520.00 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-01-14 14:15:00 | 1510.70 | 2026-01-21 09:15:00 | 1435.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 1513.80 | 2026-01-21 09:15:00 | 1438.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1511.80 | 2026-01-21 09:15:00 | 1436.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:00:00 | 1511.30 | 2026-01-21 09:15:00 | 1435.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 1502.00 | 2026-01-21 09:15:00 | 1426.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 14:15:00 | 1510.70 | 2026-01-22 09:15:00 | 1454.10 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2026-01-14 15:00:00 | 1513.80 | 2026-01-22 09:15:00 | 1454.10 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1511.80 | 2026-01-22 09:15:00 | 1454.10 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2026-01-16 13:00:00 | 1511.30 | 2026-01-22 09:15:00 | 1454.10 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2026-01-19 15:15:00 | 1502.00 | 2026-01-22 09:15:00 | 1454.10 | STOP_HIT | 0.50 | 3.19% |
| BUY | retest2 | 2026-01-30 11:00:00 | 1551.10 | 2026-02-01 11:15:00 | 1440.20 | STOP_HIT | 1.00 | -7.15% |
| BUY | retest2 | 2026-01-30 12:00:00 | 1550.40 | 2026-02-01 11:15:00 | 1440.20 | STOP_HIT | 1.00 | -7.11% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1255.00 | 2026-02-18 09:15:00 | 1268.10 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-20 09:15:00 | 1285.40 | 2026-02-23 11:15:00 | 1262.90 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-02-26 12:15:00 | 1259.10 | 2026-02-26 14:15:00 | 1273.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-02-26 12:45:00 | 1258.80 | 2026-02-26 14:15:00 | 1273.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-03-05 11:15:00 | 1303.80 | 2026-03-13 09:15:00 | 1302.90 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2026-03-05 12:15:00 | 1296.50 | 2026-03-13 09:15:00 | 1302.90 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2026-03-06 09:15:00 | 1324.60 | 2026-03-13 09:15:00 | 1302.90 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-03-27 10:15:00 | 1158.00 | 2026-03-30 13:15:00 | 1102.09 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2026-03-27 12:15:00 | 1160.10 | 2026-03-30 14:15:00 | 1100.10 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-03-27 13:00:00 | 1154.50 | 2026-03-30 14:15:00 | 1096.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 10:15:00 | 1158.00 | 2026-04-01 09:15:00 | 1193.90 | STOP_HIT | 0.50 | -3.10% |
| SELL | retest2 | 2026-03-27 12:15:00 | 1160.10 | 2026-04-01 09:15:00 | 1193.90 | STOP_HIT | 0.50 | -2.91% |
| SELL | retest2 | 2026-03-27 13:00:00 | 1154.50 | 2026-04-01 09:15:00 | 1193.90 | STOP_HIT | 0.50 | -3.41% |
| BUY | retest2 | 2026-04-13 11:30:00 | 1338.60 | 2026-04-28 12:15:00 | 1389.00 | STOP_HIT | 1.00 | 3.77% |
| BUY | retest2 | 2026-04-13 13:00:00 | 1336.30 | 2026-04-28 12:15:00 | 1389.00 | STOP_HIT | 1.00 | 3.94% |
| BUY | retest2 | 2026-04-13 13:30:00 | 1338.90 | 2026-04-28 12:15:00 | 1389.00 | STOP_HIT | 1.00 | 3.74% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1344.80 | 2026-04-28 12:15:00 | 1389.00 | STOP_HIT | 1.00 | 3.29% |
| BUY | retest2 | 2026-04-16 14:30:00 | 1353.80 | 2026-04-28 12:15:00 | 1389.00 | STOP_HIT | 1.00 | 2.60% |
| SELL | retest2 | 2026-05-05 11:15:00 | 1375.30 | 2026-05-05 12:15:00 | 1396.50 | STOP_HIT | 1.00 | -1.54% |
