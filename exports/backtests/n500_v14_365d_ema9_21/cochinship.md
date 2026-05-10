# Cochin Shipyard Ltd. (COCHINSHIP)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1769.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 61 |
| ALERT1 | 45 |
| ALERT2 | 45 |
| ALERT2_SKIP | 24 |
| ALERT3 | 103 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 55 |
| PARTIAL | 16 |
| TARGET_HIT | 7 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 36
- **Target hits / Stop hits / Partials:** 7 / 51 / 16
- **Avg / median % per leg:** 1.51% / 1.49%
- **Sum % (uncompounded):** 111.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 6 | 40.0% | 4 | 10 | 1 | 2.32% | 34.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.85% | 11.4% |
| BUY @ 3rd Alert (retest2) | 11 | 4 | 36.4% | 3 | 8 | 0 | 2.12% | 23.4% |
| SELL (all) | 59 | 32 | 54.2% | 3 | 41 | 15 | 1.31% | 77.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 59 | 32 | 54.2% | 3 | 41 | 15 | 1.31% | 77.2% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.85% | 11.4% |
| retest2 (combined) | 70 | 36 | 51.4% | 6 | 49 | 15 | 1.44% | 100.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1508.30 | 1481.24 | 1480.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1570.80 | 1520.16 | 1502.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 14:15:00 | 2001.40 | 2006.64 | 1933.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 15:00:00 | 2001.40 | 2006.64 | 1933.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1858.80 | 1973.12 | 1930.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 1858.80 | 1973.12 | 1930.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1866.90 | 1951.87 | 1924.66 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1833.70 | 1899.29 | 1904.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1817.40 | 1882.91 | 1896.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 15:15:00 | 1844.90 | 1841.12 | 1861.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:15:00 | 1842.40 | 1841.12 | 1861.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1896.40 | 1852.17 | 1864.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 1896.40 | 1852.17 | 1864.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1880.30 | 1857.80 | 1866.27 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 1917.90 | 1879.49 | 1874.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 10:15:00 | 1937.90 | 1904.32 | 1889.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 1910.40 | 1913.36 | 1899.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 1910.40 | 1913.36 | 1899.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1881.30 | 1907.05 | 1898.74 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 1884.30 | 1894.78 | 1895.13 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 1925.80 | 1899.74 | 1897.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 11:15:00 | 1968.90 | 1936.69 | 1920.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 1964.10 | 1967.82 | 1952.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1964.10 | 1967.82 | 1952.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1964.10 | 1967.82 | 1952.99 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 1920.50 | 1945.84 | 1947.55 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 2020.80 | 1958.70 | 1950.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 11:15:00 | 2042.20 | 1975.40 | 1958.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 09:15:00 | 2062.50 | 2073.22 | 2039.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 12:30:00 | 2151.00 | 2101.26 | 2060.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 13:15:00 | 2258.55 | 2148.09 | 2085.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-06-05 15:15:00 | 2366.10 | 2221.81 | 2132.01 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2268.00 | 2308.66 | 2284.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 2272.00 | 2308.66 | 2284.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 2262.00 | 2299.33 | 2282.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:15:00 | 2252.20 | 2299.33 | 2282.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 2261.80 | 2272.74 | 2273.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 2197.90 | 2257.77 | 2266.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 2223.60 | 2214.59 | 2235.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 2223.60 | 2214.59 | 2235.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 2223.60 | 2214.59 | 2235.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 2211.10 | 2214.59 | 2235.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 2203.00 | 2174.25 | 2198.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 2203.00 | 2174.25 | 2198.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 2169.90 | 2173.38 | 2195.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:30:00 | 2163.80 | 2172.68 | 2193.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 13:15:00 | 2150.80 | 2172.68 | 2193.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 14:15:00 | 2156.80 | 2174.34 | 2192.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 2130.00 | 2164.90 | 2183.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 2185.60 | 2169.04 | 2183.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 2185.60 | 2169.04 | 2183.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2201.10 | 2175.45 | 2184.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 2202.20 | 2175.45 | 2184.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 2190.00 | 2178.36 | 2185.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 2184.00 | 2178.36 | 2185.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:45:00 | 2180.50 | 2182.71 | 2186.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 2208.90 | 2188.96 | 2188.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 2208.90 | 2188.96 | 2188.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 2208.90 | 2188.96 | 2188.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 2208.90 | 2188.96 | 2188.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 2208.90 | 2188.96 | 2188.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 2208.90 | 2188.96 | 2188.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 2208.90 | 2188.96 | 2188.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 10:15:00 | 2223.80 | 2195.92 | 2191.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 2197.50 | 2198.83 | 2193.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 12:15:00 | 2197.50 | 2198.83 | 2193.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 2197.50 | 2198.83 | 2193.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 2197.50 | 2198.83 | 2193.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 2198.60 | 2199.64 | 2195.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 2198.60 | 2199.64 | 2195.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 2194.00 | 2198.51 | 2195.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 2180.00 | 2198.51 | 2195.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2185.50 | 2195.91 | 2194.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 2181.40 | 2195.91 | 2194.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 2165.00 | 2189.73 | 2191.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 2154.20 | 2178.99 | 2186.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 2176.60 | 2168.43 | 2178.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 2176.60 | 2168.43 | 2178.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 2176.60 | 2168.43 | 2178.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 2176.60 | 2168.43 | 2178.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 2149.90 | 2164.73 | 2175.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 2141.30 | 2158.78 | 2172.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 2122.60 | 2118.53 | 2139.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 2204.00 | 2147.44 | 2146.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 2204.00 | 2147.44 | 2146.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 2204.00 | 2147.44 | 2146.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 2211.60 | 2169.54 | 2157.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 2176.90 | 2206.06 | 2185.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 2176.90 | 2206.06 | 2185.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2176.90 | 2206.06 | 2185.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 2181.10 | 2206.06 | 2185.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 2179.20 | 2200.69 | 2185.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:15:00 | 2171.90 | 2200.69 | 2185.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 2184.60 | 2197.47 | 2185.30 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 09:15:00 | 2135.80 | 2173.85 | 2177.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 11:15:00 | 2128.00 | 2158.38 | 2169.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 2109.80 | 2107.27 | 2125.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 2109.80 | 2107.27 | 2125.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2109.80 | 2107.27 | 2125.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:30:00 | 2093.00 | 2103.64 | 2120.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:30:00 | 2093.00 | 2100.36 | 2116.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 2061.90 | 2042.39 | 2041.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 2061.90 | 2042.39 | 2041.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 11:15:00 | 2061.90 | 2042.39 | 2041.78 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 2025.60 | 2043.37 | 2045.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 2012.40 | 2037.18 | 2042.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 2060.00 | 2033.05 | 2037.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 2060.00 | 2033.05 | 2037.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 2060.00 | 2033.05 | 2037.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 2060.00 | 2033.05 | 2037.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 2048.00 | 2036.04 | 2038.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:45:00 | 2045.40 | 2038.21 | 2038.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 2044.90 | 2038.21 | 2038.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 2045.60 | 2039.69 | 2039.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 2045.60 | 2039.69 | 2039.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 2045.60 | 2039.69 | 2039.59 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 2035.00 | 2038.75 | 2039.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 09:15:00 | 2020.20 | 2034.60 | 2037.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 11:15:00 | 1935.40 | 1927.71 | 1943.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-16 11:45:00 | 1937.80 | 1927.71 | 1943.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1936.00 | 1931.70 | 1941.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:45:00 | 1940.40 | 1931.70 | 1941.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1923.00 | 1931.32 | 1939.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 1915.10 | 1925.59 | 1935.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 1819.34 | 1842.18 | 1851.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-29 09:15:00 | 1723.59 | 1768.53 | 1797.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 1791.00 | 1779.58 | 1779.49 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 1770.00 | 1778.50 | 1779.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 1745.80 | 1771.96 | 1776.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1738.20 | 1738.06 | 1751.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 1738.20 | 1738.06 | 1751.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1742.20 | 1739.68 | 1747.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 1723.90 | 1735.18 | 1741.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 1637.70 | 1663.22 | 1683.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 1664.80 | 1663.54 | 1681.35 | SL hit (close>ema200) qty=0.50 sl=1663.54 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 1687.00 | 1673.80 | 1672.62 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 1660.50 | 1671.90 | 1672.42 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 1689.50 | 1674.93 | 1673.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 1707.70 | 1684.06 | 1677.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 1711.90 | 1713.16 | 1702.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:00:00 | 1711.90 | 1713.16 | 1702.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1705.70 | 1711.58 | 1703.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 1705.20 | 1713.36 | 1705.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 1713.50 | 1717.71 | 1710.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 1713.50 | 1717.71 | 1710.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1712.60 | 1716.69 | 1710.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 1712.60 | 1716.69 | 1710.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1744.00 | 1722.04 | 1714.05 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1703.80 | 1716.14 | 1716.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 1698.00 | 1707.32 | 1710.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1630.80 | 1621.87 | 1641.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:30:00 | 1633.20 | 1621.87 | 1641.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1636.00 | 1624.70 | 1641.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1642.80 | 1624.70 | 1641.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1626.50 | 1619.17 | 1631.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1641.10 | 1619.17 | 1631.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1640.90 | 1623.33 | 1630.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1640.90 | 1623.33 | 1630.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1645.00 | 1627.66 | 1631.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 1645.00 | 1627.66 | 1631.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 1669.20 | 1635.97 | 1635.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1701.50 | 1656.12 | 1644.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 1735.40 | 1736.76 | 1713.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 09:15:00 | 1718.00 | 1736.76 | 1713.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1707.40 | 1730.89 | 1712.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 1707.40 | 1730.89 | 1712.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1702.90 | 1725.29 | 1711.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 1702.90 | 1725.29 | 1711.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1692.80 | 1718.79 | 1710.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 1692.80 | 1718.79 | 1710.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1678.00 | 1700.76 | 1703.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 1673.00 | 1695.21 | 1700.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 1669.10 | 1660.07 | 1671.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 12:15:00 | 1669.10 | 1660.07 | 1671.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1669.10 | 1660.07 | 1671.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 1669.10 | 1660.07 | 1671.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 1656.50 | 1659.36 | 1670.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 1650.00 | 1658.49 | 1668.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 1693.40 | 1653.35 | 1652.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 1693.40 | 1653.35 | 1652.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 1713.90 | 1676.92 | 1665.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1888.60 | 1896.60 | 1870.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 13:30:00 | 1912.00 | 1891.68 | 1875.75 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:15:00 | 1946.00 | 1894.62 | 1879.96 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1894.00 | 1915.60 | 1903.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 1894.00 | 1915.60 | 1903.29 | SL hit (close<ema400) qty=1.00 sl=1903.29 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 1894.00 | 1915.60 | 1903.29 | SL hit (close<ema400) qty=1.00 sl=1903.29 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 1882.00 | 1915.60 | 1903.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1886.10 | 1909.70 | 1901.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 1886.10 | 1909.70 | 1901.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1885.40 | 1904.84 | 1900.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 1884.00 | 1904.84 | 1900.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 1882.30 | 1894.74 | 1896.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 1869.00 | 1884.79 | 1890.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 14:15:00 | 1894.00 | 1885.96 | 1890.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 14:15:00 | 1894.00 | 1885.96 | 1890.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1894.00 | 1885.96 | 1890.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 1894.00 | 1885.96 | 1890.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1898.00 | 1888.37 | 1890.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 1952.30 | 1888.37 | 1890.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 1942.60 | 1899.22 | 1895.49 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 1886.70 | 1905.67 | 1907.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 1871.00 | 1898.74 | 1904.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 1795.00 | 1790.20 | 1817.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 1814.20 | 1790.20 | 1817.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1838.70 | 1799.90 | 1819.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 1838.70 | 1799.90 | 1819.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1858.00 | 1811.52 | 1823.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 1889.60 | 1811.52 | 1823.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1855.00 | 1832.55 | 1830.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1862.20 | 1838.48 | 1833.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1855.00 | 1857.77 | 1849.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 1855.00 | 1857.77 | 1849.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1855.00 | 1857.77 | 1849.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 1851.90 | 1857.77 | 1849.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1842.50 | 1854.72 | 1849.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 1840.70 | 1854.72 | 1849.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1839.20 | 1851.61 | 1848.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 1839.20 | 1851.61 | 1848.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1848.10 | 1851.63 | 1849.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1848.10 | 1851.63 | 1849.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1844.80 | 1850.27 | 1849.20 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 1841.30 | 1847.30 | 1847.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1812.50 | 1839.17 | 1844.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 1783.10 | 1769.41 | 1779.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 1783.10 | 1769.41 | 1779.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1783.10 | 1769.41 | 1779.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 1795.20 | 1769.41 | 1779.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1773.20 | 1770.17 | 1778.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 1771.60 | 1770.17 | 1778.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:15:00 | 1769.90 | 1771.04 | 1778.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 1790.50 | 1775.87 | 1779.50 | SL hit (close>static) qty=1.00 sl=1788.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 1790.50 | 1775.87 | 1779.50 | SL hit (close>static) qty=1.00 sl=1788.30 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1793.00 | 1783.20 | 1782.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 1805.00 | 1790.15 | 1785.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 1792.10 | 1793.37 | 1789.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 1792.10 | 1793.37 | 1789.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1792.10 | 1793.37 | 1789.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 1792.10 | 1793.37 | 1789.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1788.90 | 1792.48 | 1789.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:45:00 | 1788.70 | 1792.48 | 1789.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1785.50 | 1791.08 | 1788.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 1785.50 | 1791.08 | 1788.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1792.00 | 1791.27 | 1789.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 1800.70 | 1793.25 | 1790.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 1801.00 | 1792.96 | 1790.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 15:00:00 | 1795.00 | 1793.04 | 1791.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 1797.60 | 1792.24 | 1791.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1783.00 | 1790.39 | 1790.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1783.00 | 1790.39 | 1790.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1783.00 | 1790.39 | 1790.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1783.00 | 1790.39 | 1790.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1783.00 | 1790.39 | 1790.53 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 1795.50 | 1790.74 | 1790.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 1800.80 | 1793.93 | 1792.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1787.00 | 1807.08 | 1802.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 1787.00 | 1807.08 | 1802.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1787.00 | 1807.08 | 1802.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1787.00 | 1807.08 | 1802.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1792.00 | 1804.06 | 1801.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1820.00 | 1804.06 | 1801.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 1810.20 | 1812.85 | 1813.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 1810.20 | 1812.85 | 1813.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 1803.30 | 1809.89 | 1811.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 1791.20 | 1790.43 | 1798.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:45:00 | 1789.00 | 1790.43 | 1798.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1785.60 | 1789.47 | 1797.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 1790.60 | 1789.47 | 1797.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1789.00 | 1789.37 | 1796.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1778.00 | 1789.65 | 1792.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1689.10 | 1713.16 | 1739.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 1713.00 | 1708.77 | 1730.61 | SL hit (close>ema200) qty=0.50 sl=1708.77 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1765.00 | 1741.02 | 1738.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1796.70 | 1766.80 | 1757.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1728.00 | 1773.92 | 1768.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1728.00 | 1773.92 | 1768.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1728.00 | 1773.92 | 1768.34 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 1721.50 | 1763.43 | 1764.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 11:15:00 | 1707.20 | 1752.19 | 1758.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 1745.80 | 1729.68 | 1740.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 11:15:00 | 1745.80 | 1729.68 | 1740.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1745.80 | 1729.68 | 1740.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:00:00 | 1745.80 | 1729.68 | 1740.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1724.60 | 1728.66 | 1739.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:15:00 | 1723.10 | 1728.66 | 1739.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:15:00 | 1721.00 | 1729.15 | 1734.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 15:00:00 | 1721.60 | 1727.64 | 1732.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1752.80 | 1722.18 | 1718.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1752.80 | 1722.18 | 1718.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1752.80 | 1722.18 | 1718.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1752.80 | 1722.18 | 1718.52 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1700.00 | 1717.65 | 1718.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 1666.90 | 1697.92 | 1707.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1689.50 | 1676.77 | 1688.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1689.50 | 1676.77 | 1688.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1689.50 | 1676.77 | 1688.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 1693.80 | 1676.77 | 1688.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1696.00 | 1680.62 | 1689.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 1696.00 | 1680.62 | 1689.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 1667.30 | 1677.95 | 1687.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 1662.50 | 1677.95 | 1687.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 1664.90 | 1668.95 | 1673.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 13:00:00 | 1664.50 | 1668.06 | 1672.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 1663.90 | 1668.48 | 1671.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1655.30 | 1665.11 | 1669.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:30:00 | 1652.60 | 1662.89 | 1668.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:30:00 | 1652.50 | 1657.74 | 1664.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1579.38 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1581.65 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1581.27 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1580.70 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1569.97 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1569.88 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1619.10 | 1614.01 | 1621.04 | SL hit (close>ema200) qty=0.50 sl=1614.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1619.10 | 1614.01 | 1621.04 | SL hit (close>ema200) qty=0.50 sl=1614.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1619.10 | 1614.01 | 1621.04 | SL hit (close>ema200) qty=0.50 sl=1614.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1619.10 | 1614.01 | 1621.04 | SL hit (close>ema200) qty=0.50 sl=1614.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1619.10 | 1614.01 | 1621.04 | SL hit (close>ema200) qty=0.50 sl=1614.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1619.10 | 1614.01 | 1621.04 | SL hit (close>ema200) qty=0.50 sl=1614.01 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 1551.00 | 1531.06 | 1530.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1577.40 | 1540.33 | 1534.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 1647.70 | 1651.58 | 1632.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:45:00 | 1647.80 | 1651.58 | 1632.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1652.00 | 1657.41 | 1648.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1670.40 | 1657.41 | 1648.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:30:00 | 1655.50 | 1657.55 | 1650.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1635.00 | 1653.04 | 1649.19 | SL hit (close<static) qty=1.00 sl=1647.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1635.00 | 1653.04 | 1649.19 | SL hit (close<static) qty=1.00 sl=1647.70 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 1636.10 | 1646.68 | 1646.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1617.20 | 1640.04 | 1643.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1621.00 | 1620.36 | 1630.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 1630.40 | 1620.36 | 1630.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1621.10 | 1620.51 | 1629.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:45:00 | 1613.70 | 1619.43 | 1624.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 1612.10 | 1617.65 | 1623.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 1613.30 | 1616.78 | 1622.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 1613.00 | 1616.47 | 1621.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1621.30 | 1617.43 | 1621.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1621.30 | 1617.43 | 1621.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1632.00 | 1620.35 | 1622.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1634.70 | 1620.35 | 1622.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1632.00 | 1622.68 | 1623.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1635.50 | 1625.24 | 1624.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1635.50 | 1625.24 | 1624.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1635.50 | 1625.24 | 1624.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1635.50 | 1625.24 | 1624.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1635.50 | 1625.24 | 1624.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 1663.00 | 1634.48 | 1629.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1627.00 | 1636.69 | 1632.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 1627.00 | 1636.69 | 1632.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1627.00 | 1636.69 | 1632.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1627.00 | 1636.69 | 1632.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1629.30 | 1635.21 | 1632.09 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 1613.40 | 1630.33 | 1630.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1603.00 | 1617.26 | 1623.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 1608.20 | 1603.22 | 1611.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 1608.20 | 1603.22 | 1611.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1612.50 | 1605.08 | 1611.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1619.00 | 1605.08 | 1611.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1621.60 | 1608.38 | 1612.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1607.30 | 1609.12 | 1612.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 1604.50 | 1607.60 | 1611.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:00:00 | 1602.00 | 1595.99 | 1603.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:15:00 | 1606.30 | 1599.05 | 1603.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 1590.80 | 1597.40 | 1602.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:30:00 | 1585.50 | 1596.10 | 1601.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:30:00 | 1583.40 | 1591.12 | 1598.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1526.93 | 1575.49 | 1589.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1524.27 | 1575.49 | 1589.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1521.90 | 1575.49 | 1589.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1525.98 | 1575.49 | 1589.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 1567.60 | 1564.36 | 1576.31 | SL hit (close>ema200) qty=0.50 sl=1564.36 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 1567.60 | 1564.36 | 1576.31 | SL hit (close>ema200) qty=0.50 sl=1564.36 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 1567.60 | 1564.36 | 1576.31 | SL hit (close>ema200) qty=0.50 sl=1564.36 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 1567.60 | 1564.36 | 1576.31 | SL hit (close>ema200) qty=0.50 sl=1564.36 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1506.22 | 1527.06 | 1538.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1504.23 | 1527.06 | 1538.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 1426.95 | 1464.70 | 1490.91 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 1425.06 | 1464.70 | 1490.91 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1505.00 | 1480.48 | 1477.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 1506.40 | 1487.98 | 1483.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 12:15:00 | 1488.60 | 1489.74 | 1485.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 12:45:00 | 1487.90 | 1489.74 | 1485.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 1483.10 | 1488.41 | 1485.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:00:00 | 1483.10 | 1488.41 | 1485.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1511.00 | 1492.93 | 1487.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 1542.00 | 1492.93 | 1487.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1553.30 | 1602.40 | 1605.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 1553.30 | 1602.40 | 1605.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 1536.00 | 1577.34 | 1592.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 1488.40 | 1486.26 | 1498.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:00:00 | 1488.40 | 1486.26 | 1498.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1495.00 | 1487.75 | 1494.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 1518.60 | 1487.75 | 1494.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1509.00 | 1492.00 | 1495.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 1511.20 | 1492.00 | 1495.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1514.70 | 1496.54 | 1497.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 1517.00 | 1496.54 | 1497.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 1514.90 | 1500.21 | 1498.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 1520.80 | 1504.33 | 1500.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1530.90 | 1531.00 | 1519.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 1530.90 | 1531.00 | 1519.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1513.50 | 1526.15 | 1520.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1510.40 | 1526.15 | 1520.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1512.80 | 1523.48 | 1519.55 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 1514.10 | 1516.99 | 1517.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1507.30 | 1514.74 | 1516.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 1475.50 | 1469.31 | 1479.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 15:15:00 | 1475.50 | 1469.31 | 1479.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1475.50 | 1469.31 | 1479.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 1541.70 | 1469.31 | 1479.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1537.70 | 1482.99 | 1484.58 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 1529.60 | 1492.31 | 1488.67 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 09:15:00 | 1505.70 | 1516.35 | 1517.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 1500.80 | 1511.12 | 1514.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 1495.00 | 1493.60 | 1500.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 09:15:00 | 1496.40 | 1493.60 | 1500.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1491.90 | 1493.26 | 1499.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:30:00 | 1484.30 | 1489.80 | 1496.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1480.30 | 1487.94 | 1493.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1483.50 | 1488.95 | 1493.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 1507.00 | 1492.56 | 1494.93 | SL hit (close>static) qty=1.00 sl=1504.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 1507.00 | 1492.56 | 1494.93 | SL hit (close>static) qty=1.00 sl=1504.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 1507.00 | 1492.56 | 1494.93 | SL hit (close>static) qty=1.00 sl=1504.60 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 12:15:00 | 1503.10 | 1497.30 | 1496.83 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1481.60 | 1495.98 | 1496.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 1477.00 | 1489.95 | 1493.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 15:15:00 | 1502.00 | 1487.31 | 1490.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 1502.00 | 1487.31 | 1490.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1502.00 | 1487.31 | 1490.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1521.20 | 1487.31 | 1490.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1481.20 | 1486.09 | 1489.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 1424.00 | 1462.50 | 1474.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 10:15:00 | 1424.50 | 1456.32 | 1470.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:30:00 | 1425.50 | 1417.91 | 1435.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 1502.40 | 1453.15 | 1447.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 1502.40 | 1453.15 | 1447.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 1502.40 | 1453.15 | 1447.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 1502.40 | 1453.15 | 1447.98 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 1445.70 | 1459.83 | 1460.96 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1478.30 | 1462.86 | 1461.81 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 1457.10 | 1464.83 | 1465.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1429.10 | 1457.68 | 1462.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1358.00 | 1349.17 | 1378.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:45:00 | 1354.90 | 1349.17 | 1378.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1392.90 | 1360.71 | 1369.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 1398.30 | 1360.71 | 1369.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1402.70 | 1369.11 | 1372.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 1403.90 | 1369.11 | 1372.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1402.80 | 1380.18 | 1377.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1407.80 | 1385.70 | 1380.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1383.10 | 1393.10 | 1385.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1383.10 | 1393.10 | 1385.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1383.10 | 1393.10 | 1385.55 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 1359.00 | 1383.15 | 1383.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 1335.90 | 1363.69 | 1372.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1295.70 | 1292.51 | 1321.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 1295.70 | 1292.51 | 1321.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1295.70 | 1292.51 | 1321.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 1287.20 | 1291.69 | 1318.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1348.90 | 1309.01 | 1315.47 | SL hit (close>static) qty=1.00 sl=1321.90 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 1327.90 | 1321.07 | 1320.17 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1286.60 | 1313.96 | 1317.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1281.60 | 1307.48 | 1314.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1345.20 | 1245.53 | 1261.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1345.20 | 1245.53 | 1261.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1345.20 | 1245.53 | 1261.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1370.40 | 1245.53 | 1261.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1343.50 | 1265.13 | 1269.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 1343.50 | 1265.13 | 1269.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1352.30 | 1282.56 | 1276.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1382.50 | 1348.64 | 1332.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1421.50 | 1448.94 | 1427.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1421.50 | 1448.94 | 1427.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1421.50 | 1448.94 | 1427.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 1442.40 | 1445.53 | 1429.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 1440.40 | 1442.58 | 1430.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 1440.00 | 1440.46 | 1430.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 09:15:00 | 1584.44 | 1550.94 | 1533.50 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-21 09:15:00 | 1584.00 | 1550.94 | 1533.50 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-22 14:15:00 | 1586.64 | 1570.02 | 1557.21 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 1705.20 | 1717.20 | 1718.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 13:15:00 | 1701.50 | 1711.82 | 1715.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 15:15:00 | 1718.00 | 1712.96 | 1715.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 15:15:00 | 1718.00 | 1712.96 | 1715.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 1718.00 | 1712.96 | 1715.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 1727.20 | 1712.96 | 1715.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1746.90 | 1719.75 | 1718.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1762.60 | 1728.32 | 1722.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 1782.50 | 1789.83 | 1772.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 1782.50 | 1789.83 | 1772.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1770.20 | 1784.44 | 1772.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 1770.20 | 1784.44 | 1772.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1769.40 | 1781.43 | 1772.64 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-05 12:30:00 | 2151.00 | 2025-06-05 13:15:00 | 2258.55 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-05 12:30:00 | 2151.00 | 2025-06-05 15:15:00 | 2366.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-13 12:30:00 | 2163.80 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-13 13:15:00 | 2150.80 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-06-13 14:15:00 | 2156.80 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-06-16 09:30:00 | 2130.00 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-06-16 13:15:00 | 2184.00 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-06-16 14:45:00 | 2180.50 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-19 11:30:00 | 2141.30 | 2025-06-20 15:15:00 | 2204.00 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-06-20 12:15:00 | 2122.60 | 2025-06-20 15:15:00 | 2204.00 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-06-27 11:30:00 | 2093.00 | 2025-07-04 11:15:00 | 2061.90 | STOP_HIT | 1.00 | 1.49% |
| SELL | retest2 | 2025-06-27 13:30:00 | 2093.00 | 2025-07-04 11:15:00 | 2061.90 | STOP_HIT | 1.00 | 1.49% |
| SELL | retest2 | 2025-07-09 11:45:00 | 2045.40 | 2025-07-09 12:15:00 | 2045.60 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-07-09 12:15:00 | 2044.90 | 2025-07-09 12:15:00 | 2045.60 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-07-17 11:30:00 | 1915.10 | 2025-07-25 10:15:00 | 1819.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 11:30:00 | 1915.10 | 2025-07-29 09:15:00 | 1723.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-06 09:15:00 | 1723.90 | 2025-08-08 15:15:00 | 1637.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 09:15:00 | 1723.90 | 2025-08-11 09:15:00 | 1664.80 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2025-09-08 15:15:00 | 1650.00 | 2025-09-11 09:15:00 | 1693.40 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest1 | 2025-09-19 13:30:00 | 1912.00 | 2025-09-23 09:15:00 | 1894.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest1 | 2025-09-22 09:15:00 | 1946.00 | 2025-09-23 09:15:00 | 1894.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-10-14 11:15:00 | 1771.60 | 2025-10-14 13:15:00 | 1790.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-14 12:15:00 | 1769.90 | 2025-10-14 13:15:00 | 1790.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-10-17 09:30:00 | 1800.70 | 2025-10-20 09:15:00 | 1783.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-17 11:15:00 | 1801.00 | 2025-10-20 09:15:00 | 1783.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-17 15:00:00 | 1795.00 | 2025-10-20 09:15:00 | 1783.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-10-20 09:15:00 | 1797.60 | 2025-10-20 09:15:00 | 1783.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1820.00 | 2025-10-28 14:15:00 | 1810.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1778.00 | 2025-11-07 09:15:00 | 1689.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1778.00 | 2025-11-07 12:15:00 | 1713.00 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2025-11-14 13:15:00 | 1723.10 | 2025-11-20 11:15:00 | 1752.80 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-11-17 14:15:00 | 1721.00 | 2025-11-20 11:15:00 | 1752.80 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-11-17 15:00:00 | 1721.60 | 2025-11-20 11:15:00 | 1752.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-11-25 12:15:00 | 1662.50 | 2025-12-09 09:15:00 | 1579.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:45:00 | 1664.90 | 2025-12-09 09:15:00 | 1581.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 13:00:00 | 1664.50 | 2025-12-09 09:15:00 | 1581.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 15:15:00 | 1663.90 | 2025-12-09 09:15:00 | 1580.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1652.60 | 2025-12-09 09:15:00 | 1569.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 12:30:00 | 1652.50 | 2025-12-09 09:15:00 | 1569.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 12:15:00 | 1662.50 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2025-12-01 11:45:00 | 1664.90 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2025-12-01 13:00:00 | 1664.50 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-12-01 15:15:00 | 1663.90 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.69% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1652.60 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2025-12-02 12:30:00 | 1652.50 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.02% |
| BUY | retest2 | 2025-12-29 09:15:00 | 1670.40 | 2025-12-29 12:15:00 | 1635.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-12-29 11:30:00 | 1655.50 | 2025-12-29 12:15:00 | 1635.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-01 09:45:00 | 1613.70 | 2026-01-02 10:15:00 | 1635.50 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-01-01 12:00:00 | 1612.10 | 2026-01-02 10:15:00 | 1635.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-01 13:00:00 | 1613.30 | 2026-01-02 10:15:00 | 1635.50 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-01-01 14:15:00 | 1613.00 | 2026-01-02 10:15:00 | 1635.50 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1607.30 | 2026-01-12 09:15:00 | 1526.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 1604.50 | 2026-01-12 09:15:00 | 1524.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 10:00:00 | 1602.00 | 2026-01-12 09:15:00 | 1521.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:15:00 | 1606.30 | 2026-01-12 09:15:00 | 1525.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1607.30 | 2026-01-12 15:15:00 | 1567.60 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2026-01-08 11:45:00 | 1604.50 | 2026-01-12 15:15:00 | 1567.60 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2026-01-09 10:00:00 | 1602.00 | 2026-01-12 15:15:00 | 1567.60 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2026-01-09 11:15:00 | 1606.30 | 2026-01-12 15:15:00 | 1567.60 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2026-01-09 12:30:00 | 1585.50 | 2026-01-19 09:15:00 | 1506.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 13:30:00 | 1583.40 | 2026-01-19 09:15:00 | 1504.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 12:30:00 | 1585.50 | 2026-01-20 15:15:00 | 1426.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-09 13:30:00 | 1583.40 | 2026-01-20 15:15:00 | 1425.06 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-27 15:15:00 | 1542.00 | 2026-02-01 14:15:00 | 1553.30 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2026-02-25 12:30:00 | 1484.30 | 2026-02-26 10:15:00 | 1507.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-02-26 09:15:00 | 1480.30 | 2026-02-26 10:15:00 | 1507.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-02-26 09:45:00 | 1483.50 | 2026-02-26 10:15:00 | 1507.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1424.00 | 2026-03-06 09:15:00 | 1502.40 | STOP_HIT | 1.00 | -5.51% |
| SELL | retest2 | 2026-03-04 10:15:00 | 1424.50 | 2026-03-06 09:15:00 | 1502.40 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest2 | 2026-03-05 11:30:00 | 1425.50 | 2026-03-06 09:15:00 | 1502.40 | STOP_HIT | 1.00 | -5.39% |
| SELL | retest2 | 2026-03-24 10:30:00 | 1287.20 | 2026-03-25 09:15:00 | 1348.90 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2026-04-13 11:30:00 | 1442.40 | 2026-04-21 09:15:00 | 1584.44 | TARGET_HIT | 1.00 | 9.85% |
| BUY | retest2 | 2026-04-13 13:45:00 | 1440.40 | 2026-04-21 09:15:00 | 1584.00 | TARGET_HIT | 1.00 | 9.97% |
| BUY | retest2 | 2026-04-13 15:15:00 | 1440.00 | 2026-04-22 14:15:00 | 1586.64 | TARGET_HIT | 1.00 | 10.18% |
