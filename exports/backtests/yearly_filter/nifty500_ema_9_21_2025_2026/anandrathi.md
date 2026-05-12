# Anand Rathi Wealth Ltd. (ANANDRATHI)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 3602.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 41 |
| ALERT2 | 39 |
| ALERT2_SKIP | 26 |
| ALERT3 | 119 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 90 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 93 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 99 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 38
- **Target hits / Stop hits / Partials:** 4 / 91 / 4
- **Avg / median % per leg:** 1.55% / 1.47%
- **Sum % (uncompounded):** 153.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 47 | 63.5% | 1 | 73 | 0 | 1.54% | 113.9% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 5 | 0 | 0.72% | 3.6% |
| BUY @ 3rd Alert (retest2) | 69 | 44 | 63.8% | 1 | 68 | 0 | 1.60% | 110.3% |
| SELL (all) | 25 | 14 | 56.0% | 3 | 18 | 4 | 1.57% | 39.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 14 | 56.0% | 3 | 18 | 4 | 1.57% | 39.2% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 5 | 0 | 0.72% | 3.6% |
| retest2 (combined) | 94 | 58 | 61.7% | 4 | 86 | 4 | 1.59% | 149.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1722.10 | 1691.00 | 1689.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1745.30 | 1722.65 | 1708.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 1790.50 | 1791.86 | 1776.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 13:30:00 | 1814.20 | 1797.87 | 1783.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 1790.00 | 1800.16 | 1787.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 1828.50 | 1800.16 | 1787.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:15:00 | 1823.00 | 1827.62 | 1818.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 1831.00 | 1829.96 | 1820.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 1820.40 | 1827.46 | 1821.31 | SL hit (close<ema400) qty=1.00 sl=1821.31 alert=retest1 |

### Cycle 2 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 1876.00 | 1907.16 | 1908.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 11:15:00 | 1875.00 | 1896.00 | 1903.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 1899.90 | 1896.78 | 1902.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 12:15:00 | 1899.90 | 1896.78 | 1902.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1899.90 | 1896.78 | 1902.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 1903.60 | 1896.78 | 1902.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1892.90 | 1890.99 | 1897.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 1887.60 | 1890.99 | 1897.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1900.90 | 1891.37 | 1896.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 1907.70 | 1891.37 | 1896.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 1892.60 | 1891.62 | 1896.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 1871.20 | 1887.61 | 1893.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 1962.30 | 1901.33 | 1898.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 1962.30 | 1901.33 | 1898.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 1993.10 | 1972.55 | 1961.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 1977.60 | 1980.43 | 1969.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 15:00:00 | 1977.60 | 1980.43 | 1969.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1987.90 | 1981.93 | 1971.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1997.70 | 1977.36 | 1976.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 11:45:00 | 2006.60 | 1983.27 | 1979.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 2053.30 | 2071.08 | 2071.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 2053.30 | 2071.08 | 2071.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 2020.00 | 2060.87 | 2066.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 12:15:00 | 2072.70 | 2060.30 | 2064.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 12:15:00 | 2072.70 | 2060.30 | 2064.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 2072.70 | 2060.30 | 2064.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:45:00 | 2091.10 | 2060.30 | 2064.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 2072.10 | 2062.66 | 2065.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:15:00 | 2075.00 | 2062.66 | 2065.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 2064.00 | 2063.26 | 2064.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 2078.50 | 2063.26 | 2064.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 2071.10 | 2064.82 | 2065.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 2071.10 | 2064.82 | 2065.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 2086.60 | 2069.18 | 2067.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 2097.70 | 2075.77 | 2070.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 11:15:00 | 2086.60 | 2092.17 | 2082.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 11:15:00 | 2086.60 | 2092.17 | 2082.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 2086.60 | 2092.17 | 2082.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 2086.60 | 2092.17 | 2082.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 2076.20 | 2088.76 | 2082.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 2076.20 | 2088.76 | 2082.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 2083.70 | 2087.74 | 2082.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:15:00 | 2070.80 | 2087.74 | 2082.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 2070.80 | 2084.36 | 2081.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:15:00 | 2081.70 | 2084.36 | 2081.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 2079.70 | 2083.42 | 2081.50 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 10:15:00 | 2060.70 | 2078.88 | 2079.61 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 2087.20 | 2074.71 | 2074.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 2120.10 | 2086.25 | 2079.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 2090.90 | 2093.30 | 2085.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 2090.90 | 2093.30 | 2085.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 2090.90 | 2093.30 | 2085.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 2090.90 | 2093.30 | 2085.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 2069.90 | 2088.62 | 2083.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 2107.00 | 2091.84 | 2086.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:15:00 | 2106.00 | 2115.31 | 2105.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:30:00 | 2106.10 | 2110.52 | 2105.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 2074.00 | 2101.63 | 2101.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 2074.00 | 2101.63 | 2101.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 2048.80 | 2091.06 | 2097.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 2092.50 | 2089.88 | 2095.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 12:15:00 | 2092.50 | 2089.88 | 2095.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 2092.50 | 2089.88 | 2095.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 2096.70 | 2089.88 | 2095.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 2102.00 | 2092.31 | 2096.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 2108.00 | 2092.31 | 2096.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 2102.00 | 2094.25 | 2096.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 2105.00 | 2094.25 | 2096.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 2107.30 | 2098.58 | 2098.25 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 2092.30 | 2097.01 | 2097.57 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 2105.00 | 2099.24 | 2098.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 2108.00 | 2102.01 | 2100.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 2102.20 | 2103.25 | 2101.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 11:15:00 | 2102.20 | 2103.25 | 2101.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 2102.20 | 2103.25 | 2101.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 2101.50 | 2103.25 | 2101.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 2118.90 | 2106.38 | 2102.64 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 2086.30 | 2100.58 | 2102.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 2067.10 | 2090.62 | 2097.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 2097.30 | 2087.34 | 2092.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 2097.30 | 2087.34 | 2092.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 2097.30 | 2087.34 | 2092.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 2095.00 | 2087.34 | 2092.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 2088.40 | 2087.55 | 2092.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 2082.70 | 2093.56 | 2094.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 15:15:00 | 2100.00 | 2094.85 | 2094.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 15:15:00 | 2100.00 | 2094.85 | 2094.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 2119.40 | 2099.76 | 2096.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 09:15:00 | 2609.40 | 2617.29 | 2571.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 2609.40 | 2617.29 | 2571.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 2609.40 | 2617.29 | 2571.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:30:00 | 2641.50 | 2619.69 | 2580.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 2651.90 | 2627.51 | 2599.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 2645.00 | 2645.97 | 2625.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:00:00 | 2633.80 | 2636.08 | 2634.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 2628.20 | 2634.50 | 2634.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 2628.20 | 2634.50 | 2634.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 2637.50 | 2635.10 | 2634.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 15:15:00 | 2624.00 | 2632.88 | 2633.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 2624.00 | 2632.88 | 2633.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 2599.90 | 2625.60 | 2629.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2650.00 | 2618.22 | 2623.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2650.00 | 2618.22 | 2623.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2650.00 | 2618.22 | 2623.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 2650.00 | 2618.22 | 2623.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 2668.00 | 2628.18 | 2627.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 2688.40 | 2669.23 | 2654.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 2666.00 | 2675.98 | 2664.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 14:15:00 | 2666.00 | 2675.98 | 2664.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 2666.00 | 2675.98 | 2664.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 2666.00 | 2675.98 | 2664.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 2669.90 | 2674.76 | 2665.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 2683.60 | 2674.76 | 2665.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2657.50 | 2671.31 | 2664.67 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 2631.50 | 2657.96 | 2659.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 2611.30 | 2641.08 | 2650.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 12:15:00 | 2634.00 | 2632.44 | 2643.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 13:00:00 | 2634.00 | 2632.44 | 2643.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 2664.80 | 2628.94 | 2637.51 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 2647.20 | 2642.06 | 2641.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 2656.00 | 2644.85 | 2642.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 2635.70 | 2643.02 | 2642.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 2635.70 | 2643.02 | 2642.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 2635.70 | 2643.02 | 2642.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 2635.70 | 2643.02 | 2642.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 2638.80 | 2642.18 | 2642.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 11:15:00 | 2647.00 | 2642.18 | 2642.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 12:00:00 | 2643.20 | 2642.38 | 2642.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 2639.60 | 2641.83 | 2641.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 2639.60 | 2641.83 | 2641.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 2613.10 | 2634.76 | 2638.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 14:15:00 | 2632.00 | 2624.25 | 2630.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 14:15:00 | 2632.00 | 2624.25 | 2630.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 2632.00 | 2624.25 | 2630.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 2632.00 | 2624.25 | 2630.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 2624.30 | 2624.26 | 2629.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 2611.00 | 2615.21 | 2625.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 2651.30 | 2598.20 | 2601.50 | SL hit (close>static) qty=1.00 sl=2640.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 2690.50 | 2616.66 | 2609.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 2759.00 | 2682.46 | 2649.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 2765.00 | 2765.72 | 2730.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 15:00:00 | 2765.00 | 2765.72 | 2730.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 2689.30 | 2748.88 | 2728.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 2685.90 | 2748.88 | 2728.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2682.20 | 2735.55 | 2724.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 2753.90 | 2720.04 | 2719.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 2765.10 | 2795.26 | 2799.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 2765.10 | 2795.26 | 2799.06 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 2801.60 | 2787.67 | 2785.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 2919.30 | 2815.24 | 2799.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 2916.50 | 2919.32 | 2882.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 2916.50 | 2919.32 | 2882.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 2895.60 | 2917.87 | 2891.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 2895.60 | 2917.87 | 2891.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 2902.60 | 2914.82 | 2892.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:30:00 | 2902.80 | 2914.82 | 2892.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 2894.10 | 2910.67 | 2892.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:30:00 | 2900.40 | 2910.67 | 2892.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 2896.30 | 2907.80 | 2892.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 2892.10 | 2907.80 | 2892.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 2907.00 | 2907.64 | 2894.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:45:00 | 2906.60 | 2907.64 | 2894.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 2900.40 | 2911.70 | 2899.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 2883.40 | 2911.70 | 2899.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 2882.00 | 2905.76 | 2898.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 2882.00 | 2905.76 | 2898.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 2914.20 | 2907.45 | 2899.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:30:00 | 2931.00 | 2917.94 | 2905.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 2954.00 | 2949.17 | 2936.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 2920.10 | 2938.78 | 2940.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 2920.10 | 2938.78 | 2940.28 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 2958.30 | 2942.13 | 2941.34 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 2913.80 | 2937.98 | 2939.86 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 14:15:00 | 2949.80 | 2937.82 | 2937.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 2992.30 | 2949.07 | 2942.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 15:15:00 | 3046.20 | 3049.34 | 3033.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 09:15:00 | 3045.40 | 3049.34 | 3033.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 3061.60 | 3057.90 | 3046.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 3074.00 | 3058.46 | 3047.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:30:00 | 3070.40 | 3060.77 | 3049.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 3070.00 | 3060.77 | 3049.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 15:15:00 | 3030.20 | 3051.64 | 3048.91 | SL hit (close<static) qty=1.00 sl=3035.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 3033.00 | 3050.34 | 3050.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 3008.70 | 3038.07 | 3044.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 2900.50 | 2886.05 | 2916.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 2900.50 | 2886.05 | 2916.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 2900.50 | 2886.05 | 2916.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:45:00 | 2895.10 | 2886.05 | 2916.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 2896.50 | 2888.14 | 2915.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 2906.40 | 2888.14 | 2915.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 2841.40 | 2821.09 | 2849.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:15:00 | 2790.00 | 2815.79 | 2830.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 2871.00 | 2828.57 | 2834.27 | SL hit (close>static) qty=1.00 sl=2855.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 2885.00 | 2839.86 | 2838.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 2898.10 | 2866.62 | 2856.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 2920.40 | 2935.82 | 2911.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 2920.40 | 2935.82 | 2911.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 2920.40 | 2935.82 | 2911.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 2900.80 | 2935.82 | 2911.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 2910.00 | 2930.66 | 2911.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 2966.30 | 2930.66 | 2911.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 2937.20 | 2945.30 | 2946.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 2937.20 | 2945.30 | 2946.03 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 2956.30 | 2947.50 | 2946.96 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 2922.80 | 2944.73 | 2945.93 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 3095.10 | 2974.33 | 2958.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 10:15:00 | 3233.00 | 3026.07 | 2983.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 09:15:00 | 3124.80 | 3130.65 | 3068.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 09:45:00 | 3125.00 | 3130.65 | 3068.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 3104.40 | 3117.87 | 3090.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 3104.30 | 3117.87 | 3090.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 3074.50 | 3105.70 | 3091.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 3074.50 | 3105.70 | 3091.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 3078.60 | 3100.28 | 3090.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:30:00 | 3075.80 | 3100.28 | 3090.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 3156.90 | 3112.25 | 3098.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:30:00 | 3218.60 | 3142.52 | 3124.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 3100.30 | 3144.13 | 3144.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 3100.30 | 3144.13 | 3144.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 3082.00 | 3115.89 | 3129.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 3121.10 | 3092.20 | 3109.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 3121.10 | 3092.20 | 3109.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 3121.10 | 3092.20 | 3109.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 3121.10 | 3092.20 | 3109.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 3100.00 | 3093.76 | 3108.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 3191.70 | 3093.76 | 3108.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3200.30 | 3115.07 | 3116.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 3225.30 | 3115.07 | 3116.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 3202.40 | 3132.54 | 3124.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 3221.20 | 3170.67 | 3145.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 3182.00 | 3187.00 | 3160.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 3182.00 | 3187.00 | 3160.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 34 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 3066.00 | 3159.96 | 3160.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 3056.20 | 3098.41 | 3126.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 3140.60 | 3105.83 | 3124.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 3140.60 | 3105.83 | 3124.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 3140.60 | 3105.83 | 3124.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 3148.40 | 3105.83 | 3124.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 3150.40 | 3114.74 | 3126.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 3150.40 | 3114.74 | 3126.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 3121.60 | 3116.11 | 3126.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:30:00 | 3146.10 | 3116.11 | 3126.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 3114.40 | 3115.77 | 3125.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:00:00 | 3102.10 | 3116.85 | 3123.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 3135.50 | 3106.61 | 3112.74 | SL hit (close>static) qty=1.00 sl=3126.20 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 3135.00 | 3113.19 | 3111.68 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 3100.30 | 3110.62 | 3110.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 10:15:00 | 3091.20 | 3104.42 | 3107.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 3114.00 | 3106.34 | 3107.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 3114.00 | 3106.34 | 3107.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 3114.00 | 3106.34 | 3107.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 3114.00 | 3106.34 | 3107.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3117.00 | 3108.47 | 3108.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 3117.00 | 3108.47 | 3108.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 3129.40 | 3112.66 | 3110.51 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 11:15:00 | 3100.00 | 3110.64 | 3110.71 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 3120.90 | 3111.23 | 3110.78 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 3091.00 | 3108.62 | 3109.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 3071.30 | 3101.15 | 3106.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 3074.70 | 3070.39 | 3082.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 13:00:00 | 3074.70 | 3070.39 | 3082.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 3078.80 | 3072.07 | 3082.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 3078.80 | 3072.07 | 3082.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 3097.60 | 3077.18 | 3083.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 3097.60 | 3077.18 | 3083.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 3081.00 | 3077.94 | 3083.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 3074.30 | 3077.94 | 3083.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:15:00 | 2920.59 | 2974.41 | 3004.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 11:15:00 | 2888.90 | 2881.39 | 2914.26 | SL hit (close>ema200) qty=0.50 sl=2881.39 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 2900.40 | 2883.80 | 2882.91 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 2878.20 | 2884.38 | 2884.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 2873.10 | 2882.12 | 2883.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 14:15:00 | 2895.30 | 2884.76 | 2884.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 14:15:00 | 2895.30 | 2884.76 | 2884.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 2895.30 | 2884.76 | 2884.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 2895.30 | 2884.76 | 2884.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 2880.00 | 2883.81 | 2884.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 2862.30 | 2881.02 | 2883.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 2861.30 | 2881.02 | 2883.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 2929.00 | 2889.57 | 2885.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 2929.00 | 2889.57 | 2885.08 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 2900.80 | 2907.76 | 2907.86 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 2933.70 | 2911.84 | 2909.47 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 2893.10 | 2907.91 | 2908.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 12:15:00 | 2889.00 | 2904.13 | 2906.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 2910.00 | 2902.39 | 2905.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 15:15:00 | 2910.00 | 2902.39 | 2905.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 2910.00 | 2902.39 | 2905.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 2876.10 | 2895.37 | 2901.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:00:00 | 2883.80 | 2858.16 | 2865.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:45:00 | 2880.50 | 2868.12 | 2868.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 14:15:00 | 2876.70 | 2869.83 | 2869.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 14:15:00 | 2876.70 | 2869.83 | 2869.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 2892.60 | 2875.78 | 2872.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 15:15:00 | 2880.80 | 2882.47 | 2877.44 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:30:00 | 2909.10 | 2888.37 | 2880.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 12:15:00 | 2898.60 | 2892.93 | 2884.24 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 2989.00 | 2988.91 | 2980.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 2989.00 | 2988.91 | 2980.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 3009.60 | 2993.70 | 2984.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 2980.40 | 2991.51 | 2985.89 | SL hit (close<ema400) qty=1.00 sl=2985.89 alert=retest1 |

### Cycle 48 — SELL (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 13:15:00 | 3080.50 | 3086.49 | 3086.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 15:15:00 | 3079.00 | 3083.95 | 3085.65 | Break + close below crossover candle low |

### Cycle 49 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 3105.10 | 3088.18 | 3087.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 10:15:00 | 3134.80 | 3097.50 | 3091.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 3116.00 | 3116.42 | 3104.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:15:00 | 3141.30 | 3116.42 | 3104.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 3154.60 | 3124.06 | 3109.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 3165.90 | 3150.67 | 3133.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 3170.00 | 3150.67 | 3133.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 3167.70 | 3152.14 | 3135.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 3181.00 | 3162.39 | 3147.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 3152.70 | 3159.68 | 3150.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 3152.70 | 3159.68 | 3150.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 3149.40 | 3157.63 | 3150.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 3140.00 | 3157.63 | 3150.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 3142.50 | 3154.60 | 3149.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 3142.50 | 3154.60 | 3149.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 3140.00 | 3151.68 | 3148.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 3115.00 | 3151.68 | 3148.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 3127.00 | 3142.97 | 3144.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 3127.00 | 3142.97 | 3144.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 11:15:00 | 3101.30 | 3124.39 | 3134.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 3130.00 | 3121.20 | 3129.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 15:15:00 | 3130.00 | 3121.20 | 3129.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 3130.00 | 3121.20 | 3129.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 3162.60 | 3121.20 | 3129.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 3167.70 | 3130.50 | 3132.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:30:00 | 3163.20 | 3130.50 | 3132.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 3175.70 | 3139.54 | 3136.56 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 3098.10 | 3132.11 | 3134.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 3078.50 | 3121.39 | 3129.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 3115.80 | 3101.18 | 3113.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 3115.80 | 3101.18 | 3113.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 3115.80 | 3101.18 | 3113.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:45:00 | 3113.30 | 3101.18 | 3113.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 3121.10 | 3105.16 | 3114.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 3121.10 | 3105.16 | 3114.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 3125.00 | 3109.13 | 3115.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 3124.40 | 3109.13 | 3115.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 3094.00 | 3101.42 | 3109.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 3067.10 | 3101.42 | 3109.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 13:15:00 | 3081.00 | 3103.29 | 3108.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:15:00 | 3078.80 | 3099.00 | 3105.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2913.74 | 2960.29 | 2995.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2926.95 | 2960.29 | 2995.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2924.86 | 2960.29 | 2995.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 11:15:00 | 2760.39 | 2895.61 | 2958.60 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 2910.00 | 2901.19 | 2900.59 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 2870.10 | 2894.97 | 2897.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 13:15:00 | 2838.10 | 2873.30 | 2885.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 14:15:00 | 2882.00 | 2875.04 | 2885.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 14:15:00 | 2882.00 | 2875.04 | 2885.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 2882.00 | 2875.04 | 2885.46 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 11:15:00 | 2902.30 | 2890.70 | 2890.21 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 2847.80 | 2889.39 | 2891.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 2830.00 | 2877.51 | 2885.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 2878.20 | 2861.01 | 2873.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 2878.20 | 2861.01 | 2873.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 2878.20 | 2861.01 | 2873.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 2878.20 | 2861.01 | 2873.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2877.00 | 2864.21 | 2873.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2900.90 | 2864.21 | 2873.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2930.00 | 2877.37 | 2878.98 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 2926.00 | 2887.09 | 2883.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 2951.30 | 2899.94 | 2889.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 2942.00 | 2951.70 | 2933.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 09:15:00 | 2962.50 | 2951.70 | 2933.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 10:15:00 | 2965.50 | 2951.44 | 2934.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 2939.80 | 2949.11 | 2935.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 2939.80 | 2949.11 | 2935.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 2934.50 | 2946.19 | 2935.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-05 11:15:00 | 2934.50 | 2946.19 | 2935.30 | SL hit (close<ema400) qty=1.00 sl=2935.30 alert=retest1 |

### Cycle 58 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 3127.40 | 3170.65 | 3171.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 3100.90 | 3127.61 | 3138.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 3070.30 | 3051.56 | 3074.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 3070.30 | 3051.56 | 3074.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 3070.30 | 3051.56 | 3074.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 3073.90 | 3051.56 | 3074.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 3048.50 | 3050.95 | 3072.46 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 3088.90 | 3069.54 | 3067.34 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 13:15:00 | 3058.20 | 3065.03 | 3065.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 3028.40 | 3057.82 | 3062.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 3009.50 | 2995.97 | 3017.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 3009.50 | 2995.97 | 3017.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 3009.50 | 2995.97 | 3017.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 3009.50 | 2995.97 | 3017.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 3021.90 | 3001.16 | 3017.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 2985.00 | 3001.16 | 3017.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 3009.50 | 2979.88 | 2977.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 3009.50 | 2979.88 | 2977.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 14:15:00 | 3032.40 | 2996.13 | 2989.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 13:15:00 | 3014.10 | 3018.59 | 3005.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 13:45:00 | 3015.50 | 3018.59 | 3005.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 3000.40 | 3014.95 | 3004.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 3000.40 | 3014.95 | 3004.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 3010.50 | 3014.06 | 3005.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 3101.20 | 3014.06 | 3005.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 3411.32 | 3385.30 | 3321.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 3650.00 | 3653.24 | 3653.30 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 3656.50 | 3653.89 | 3653.59 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 3642.40 | 3651.59 | 3652.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 3632.00 | 3646.62 | 3650.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 3638.90 | 3635.83 | 3643.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 3638.90 | 3635.83 | 3643.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 3638.90 | 3635.83 | 3643.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:30:00 | 3599.10 | 3625.84 | 3635.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:30:00 | 3596.50 | 3616.81 | 3628.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:00:00 | 3606.80 | 3573.98 | 3586.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:00:00 | 3606.50 | 3580.48 | 3588.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 3591.40 | 3582.92 | 3587.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 3591.40 | 3582.92 | 3587.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 3570.00 | 3580.34 | 3586.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-29 14:15:00 | 3589.60 | 3587.51 | 3587.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 3589.60 | 3587.51 | 3587.50 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 3575.00 | 3585.01 | 3586.36 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 09:15:00 | 3616.30 | 3591.26 | 3589.08 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 3605.00 | 3616.54 | 3616.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 3602.30 | 3611.38 | 3613.99 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 13:30:00 | 1814.20 | 2025-05-21 11:15:00 | 1820.40 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-05-19 09:15:00 | 1828.50 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 2.60% |
| BUY | retest2 | 2025-05-20 15:15:00 | 1823.00 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2025-05-21 09:30:00 | 1831.00 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2025-05-21 12:15:00 | 1825.90 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest2 | 2025-05-21 14:45:00 | 1836.90 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-05-22 09:30:00 | 1840.40 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2025-05-23 09:15:00 | 1842.10 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2025-05-30 15:00:00 | 1871.20 | 2025-06-02 09:15:00 | 1962.30 | STOP_HIT | 1.00 | -4.87% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1997.70 | 2025-06-19 14:15:00 | 2053.30 | STOP_HIT | 1.00 | 2.78% |
| BUY | retest2 | 2025-06-11 11:45:00 | 2006.60 | 2025-06-19 14:15:00 | 2053.30 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2025-06-30 12:15:00 | 2107.00 | 2025-07-02 09:15:00 | 2074.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-07-01 12:15:00 | 2106.00 | 2025-07-02 09:15:00 | 2074.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-01 14:30:00 | 2106.10 | 2025-07-02 09:15:00 | 2074.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-07-09 14:30:00 | 2082.70 | 2025-07-09 15:15:00 | 2100.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-21 11:30:00 | 2641.50 | 2025-07-24 15:15:00 | 2624.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-22 09:45:00 | 2651.90 | 2025-07-24 15:15:00 | 2624.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-23 09:30:00 | 2645.00 | 2025-07-24 15:15:00 | 2624.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-24 13:00:00 | 2633.80 | 2025-07-24 15:15:00 | 2624.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-08-05 11:15:00 | 2647.00 | 2025-08-05 12:15:00 | 2639.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-08-05 12:00:00 | 2643.20 | 2025-08-05 12:15:00 | 2639.60 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-08-07 09:30:00 | 2611.00 | 2025-08-11 09:15:00 | 2651.30 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-08-18 09:15:00 | 2753.90 | 2025-08-26 09:15:00 | 2765.10 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2025-09-04 12:30:00 | 2931.00 | 2025-09-09 13:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-09-08 09:30:00 | 2954.00 | 2025-09-09 13:15:00 | 2920.10 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-19 10:30:00 | 3074.00 | 2025-09-19 15:15:00 | 3030.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-09-19 11:30:00 | 3070.40 | 2025-09-19 15:15:00 | 3030.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-19 12:00:00 | 3070.00 | 2025-09-19 15:15:00 | 3030.20 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-22 09:15:00 | 3070.00 | 2025-09-22 15:15:00 | 3033.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-01 12:15:00 | 2790.00 | 2025-10-01 13:15:00 | 2871.00 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-10-08 09:15:00 | 2966.30 | 2025-10-13 10:15:00 | 2937.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-20 09:30:00 | 3218.60 | 2025-10-23 11:15:00 | 3100.30 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-10-31 10:00:00 | 3102.10 | 2025-11-03 09:15:00 | 3135.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-11-03 11:30:00 | 3105.30 | 2025-11-04 11:15:00 | 3134.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-11-03 13:15:00 | 3106.10 | 2025-11-04 11:15:00 | 3134.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-11-13 09:15:00 | 3074.30 | 2025-11-19 09:15:00 | 2920.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 09:15:00 | 3074.30 | 2025-11-21 11:15:00 | 2888.90 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2025-11-28 09:45:00 | 2862.30 | 2025-12-01 09:15:00 | 2929.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-11-28 10:15:00 | 2861.30 | 2025-12-01 09:15:00 | 2929.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-12-08 09:30:00 | 2876.10 | 2025-12-10 14:15:00 | 2876.70 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-12-10 10:00:00 | 2883.80 | 2025-12-10 14:15:00 | 2876.70 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-12-10 13:45:00 | 2880.50 | 2025-12-10 14:15:00 | 2876.70 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest1 | 2025-12-12 09:30:00 | 2909.10 | 2025-12-19 12:15:00 | 2980.40 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest1 | 2025-12-12 12:15:00 | 2898.60 | 2025-12-19 12:15:00 | 2980.40 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2025-12-23 14:00:00 | 3026.50 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2025-12-23 14:45:00 | 3039.30 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2025-12-24 09:15:00 | 3056.10 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2025-12-26 15:00:00 | 3049.50 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2025-12-29 10:45:00 | 3113.80 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-31 09:45:00 | 3098.80 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-31 12:00:00 | 3096.30 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-31 12:30:00 | 3101.80 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-01-07 10:45:00 | 3165.90 | 2026-01-09 10:15:00 | 3127.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-01-07 11:15:00 | 3170.00 | 2026-01-09 10:15:00 | 3127.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-01-07 12:15:00 | 3167.70 | 2026-01-09 10:15:00 | 3127.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-01-08 09:30:00 | 3181.00 | 2026-01-09 10:15:00 | 3127.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-19 09:15:00 | 3067.10 | 2026-01-27 09:15:00 | 2913.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 13:15:00 | 3081.00 | 2026-01-27 09:15:00 | 2926.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 3078.80 | 2026-01-27 09:15:00 | 2924.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 3067.10 | 2026-01-27 11:15:00 | 2760.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 13:15:00 | 3081.00 | 2026-01-27 11:15:00 | 2772.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 3078.80 | 2026-01-27 11:15:00 | 2770.92 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-02-05 09:15:00 | 2962.50 | 2026-02-05 11:15:00 | 2934.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest1 | 2026-02-05 10:15:00 | 2965.50 | 2026-02-05 11:15:00 | 2934.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-06 11:45:00 | 2959.90 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.66% |
| BUY | retest2 | 2026-02-09 09:15:00 | 2961.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.62% |
| BUY | retest2 | 2026-02-09 14:15:00 | 2977.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.05% |
| BUY | retest2 | 2026-02-10 11:15:00 | 2964.30 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.50% |
| BUY | retest2 | 2026-02-11 11:15:00 | 2974.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.15% |
| BUY | retest2 | 2026-02-12 14:45:00 | 2989.90 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 4.60% |
| BUY | retest2 | 2026-02-13 10:45:00 | 2974.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.15% |
| BUY | retest2 | 2026-02-13 12:00:00 | 2976.70 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.06% |
| BUY | retest2 | 2026-02-13 14:00:00 | 3006.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 4.04% |
| BUY | retest2 | 2026-02-16 09:45:00 | 3003.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 4.14% |
| BUY | retest2 | 2026-02-16 12:15:00 | 3008.90 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.94% |
| BUY | retest2 | 2026-02-16 15:15:00 | 2998.90 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 4.28% |
| BUY | retest2 | 2026-02-19 09:30:00 | 3027.30 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.31% |
| BUY | retest2 | 2026-02-19 11:30:00 | 3034.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2026-02-20 09:30:00 | 3021.70 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.50% |
| BUY | retest2 | 2026-02-20 10:30:00 | 3025.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.38% |
| BUY | retest2 | 2026-02-23 09:15:00 | 3039.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2026-02-23 11:15:00 | 3025.90 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.35% |
| BUY | retest2 | 2026-02-24 12:15:00 | 3025.80 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.36% |
| BUY | retest2 | 2026-02-24 14:00:00 | 3034.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2026-02-25 10:45:00 | 3064.40 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2026-02-25 11:45:00 | 3061.60 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2026-02-25 13:00:00 | 3063.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2026-02-25 14:15:00 | 3060.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2026-02-26 11:00:00 | 3074.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2026-02-26 11:45:00 | 3073.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2026-02-26 12:30:00 | 3074.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2026-02-26 14:15:00 | 3078.80 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2026-03-02 10:45:00 | 3082.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2026-03-23 09:15:00 | 2985.00 | 2026-03-25 10:15:00 | 3009.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-04-01 09:15:00 | 3101.20 | 2026-04-09 09:15:00 | 3411.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-23 13:30:00 | 3599.10 | 2026-04-29 14:15:00 | 3589.60 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2026-04-24 09:30:00 | 3596.50 | 2026-04-29 14:15:00 | 3589.60 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2026-04-28 10:00:00 | 3606.80 | 2026-04-29 14:15:00 | 3589.60 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2026-04-28 11:00:00 | 3606.50 | 2026-04-29 14:15:00 | 3589.60 | STOP_HIT | 1.00 | 0.47% |
