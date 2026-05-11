# Adani Enterprises Ltd. (ADANIENT)

## Backtest Summary

- **Window:** 2025-12-22 09:15:00 → 2026-05-08 15:15:00 (644 bars)
- **Last close:** 2502.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 23 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT2_SKIP | 5 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 4 |
| TARGET_HIT | 8 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 9
- **Target hits / Stop hits / Partials:** 8 / 11 / 4
- **Avg / median % per leg:** 3.93% / 5.00%
- **Sum % (uncompounded):** 90.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 6 | 2 | 0 | 7.02% | 56.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 6 | 75.0% | 6 | 2 | 0 | 7.02% | 56.1% |
| SELL (all) | 15 | 8 | 53.3% | 2 | 9 | 4 | 2.29% | 34.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 8 | 53.3% | 2 | 9 | 4 | 2.29% | 34.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 14 | 60.9% | 8 | 11 | 4 | 3.93% | 90.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 2238.70 | 2225.45 | 2224.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 2272.00 | 2240.08 | 2231.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 2280.00 | 2280.67 | 2268.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:15:00 | 2278.50 | 2279.96 | 2270.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2265.00 | 2277.07 | 2270.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 2265.00 | 2277.07 | 2270.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 2271.10 | 2275.87 | 2270.90 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 2262.00 | 2267.21 | 2267.91 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 2276.10 | 2268.58 | 2268.39 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 2253.10 | 2268.31 | 2268.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 2234.90 | 2261.63 | 2265.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 2176.00 | 2162.44 | 2189.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 2175.00 | 2162.44 | 2189.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2163.70 | 2166.49 | 2182.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 2148.10 | 2161.32 | 2175.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 2151.20 | 2158.81 | 2170.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:15:00 | 2151.70 | 2159.16 | 2167.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 2152.00 | 2157.20 | 2165.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2179.30 | 2160.79 | 2165.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 2179.30 | 2160.79 | 2165.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 2185.80 | 2165.79 | 2167.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 2184.40 | 2165.79 | 2167.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 2189.20 | 2170.47 | 2169.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 2189.20 | 2170.47 | 2169.36 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 2156.30 | 2167.79 | 2168.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 2146.60 | 2162.15 | 2165.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 2092.10 | 2055.79 | 2079.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2070.20 | 2058.67 | 2078.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 2063.50 | 2073.61 | 2078.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:15:00 | 2064.00 | 2073.61 | 2078.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 1960.32 | 2046.48 | 2065.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 1960.80 | 2046.48 | 2065.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 14:15:00 | 1857.15 | 1985.50 | 2032.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 2018.50 | 1991.22 | 1990.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 2030.00 | 2006.04 | 1998.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:15:00 | 2027.00 | 2015.07 | 2008.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 1965.80 | 2000.32 | 2003.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1965.80 | 2000.32 | 2003.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1943.60 | 1988.98 | 1997.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 1982.80 | 1971.89 | 1983.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1978.70 | 1973.25 | 1983.36 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2181.90 | 2021.22 | 2003.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 2239.10 | 2189.37 | 2124.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 2214.50 | 2215.40 | 2172.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 12:00:00 | 2214.50 | 2215.40 | 2172.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 2215.70 | 2219.19 | 2200.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 2208.90 | 2219.19 | 2200.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 2217.50 | 2240.51 | 2229.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 2209.00 | 2240.51 | 2229.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 2245.50 | 2241.51 | 2230.80 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 2222.50 | 2227.25 | 2227.74 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 2235.40 | 2229.41 | 2228.68 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 2215.00 | 2226.62 | 2227.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 2208.30 | 2222.96 | 2225.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 2171.70 | 2154.15 | 2174.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 2171.70 | 2154.15 | 2174.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 2180.30 | 2159.38 | 2175.20 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 2199.80 | 2183.36 | 2181.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 2234.20 | 2193.53 | 2186.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 2210.40 | 2214.87 | 2200.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 2210.40 | 2214.87 | 2200.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2197.60 | 2211.41 | 2200.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 2198.60 | 2211.41 | 2200.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 2207.70 | 2210.67 | 2200.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:00:00 | 2211.80 | 2209.74 | 2202.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 2193.30 | 2206.98 | 2202.81 | SL hit (close<static) qty=1.00 sl=2193.80 alert=retest2 |

### Cycle 14 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 2166.90 | 2198.96 | 2199.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 2156.20 | 2178.44 | 2188.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 2165.00 | 2164.43 | 2175.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 2165.00 | 2164.43 | 2175.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2192.20 | 2169.31 | 2174.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 2198.60 | 2169.31 | 2174.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 2188.90 | 2173.23 | 2176.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 2180.00 | 2174.76 | 2176.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 2180.60 | 2176.93 | 2177.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 2192.90 | 2178.68 | 2178.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 2192.90 | 2178.68 | 2178.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 2200.90 | 2185.55 | 2182.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 2208.90 | 2211.94 | 2201.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:45:00 | 2207.40 | 2211.94 | 2201.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 2203.40 | 2209.51 | 2201.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 2203.40 | 2209.51 | 2201.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 2216.60 | 2210.93 | 2203.15 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 2190.90 | 2200.52 | 2200.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 2160.20 | 2191.39 | 2196.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 2127.50 | 2126.24 | 2153.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 2127.50 | 2126.24 | 2153.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 2080.50 | 2064.03 | 2088.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 2080.50 | 2064.03 | 2088.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 2080.00 | 2067.23 | 2087.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 2084.20 | 2069.38 | 2086.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2062.90 | 2068.09 | 2084.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 2059.00 | 2068.27 | 2081.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 2061.50 | 2066.45 | 2079.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1956.05 | 2040.69 | 2063.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1958.42 | 2040.69 | 2063.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 2000.10 | 1997.12 | 2016.67 | SL hit (close>ema200) qty=0.50 sl=1997.12 alert=retest2 |

### Cycle 17 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 2010.00 | 1978.91 | 1976.35 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1948.40 | 1984.01 | 1984.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 1941.40 | 1970.47 | 1977.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 1964.50 | 1955.14 | 1966.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1852.20 | 1856.22 | 1892.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 1848.30 | 1856.22 | 1892.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 1907.00 | 1857.89 | 1870.56 | SL hit (close>static) qty=1.00 sl=1893.00 alert=retest2 |

### Cycle 19 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 1891.50 | 1877.76 | 1877.72 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1849.00 | 1874.30 | 1876.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 1832.80 | 1861.87 | 1870.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1850.40 | 1803.28 | 1822.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1844.60 | 1811.54 | 1824.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 1856.00 | 1811.54 | 1824.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1854.90 | 1835.31 | 1833.76 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1786.20 | 1827.32 | 1830.69 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1839.20 | 1827.57 | 1826.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1863.40 | 1834.74 | 1829.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1868.70 | 1870.30 | 1852.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 10:45:00 | 1866.90 | 1870.30 | 1852.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2124.80 | 2082.59 | 2043.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2144.90 | 2082.59 | 2043.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2132.10 | 2093.87 | 2052.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 2130.00 | 2117.77 | 2078.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 2130.20 | 2117.77 | 2078.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2249.80 | 2247.03 | 2235.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 2261.90 | 2247.03 | 2235.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 2261.70 | 2263.29 | 2259.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-28 09:15:00 | 2359.39 | 2328.13 | 2301.77 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-13 14:00:00 | 2148.10 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-01-14 09:30:00 | 2151.20 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-01-14 13:15:00 | 2151.70 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-01-14 15:15:00 | 2152.00 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-01-23 10:45:00 | 2063.50 | 2026-01-23 12:15:00 | 1960.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 2064.00 | 2026-01-23 12:15:00 | 1960.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 2063.50 | 2026-01-23 14:15:00 | 1857.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 2064.00 | 2026-01-23 14:15:00 | 1857.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-01 11:15:00 | 2027.00 | 2026-02-01 13:15:00 | 1965.80 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-02-18 15:00:00 | 2211.80 | 2026-02-19 09:15:00 | 2193.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-23 11:45:00 | 2180.00 | 2026-02-23 14:15:00 | 2192.90 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-02-23 13:15:00 | 2180.60 | 2026-02-23 14:15:00 | 2192.90 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-03-06 13:15:00 | 2059.00 | 2026-03-09 09:15:00 | 1956.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2061.50 | 2026-03-09 09:15:00 | 1958.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:15:00 | 2059.00 | 2026-03-10 13:15:00 | 2000.10 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2061.50 | 2026-03-10 13:15:00 | 2000.10 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-03-24 10:15:00 | 1848.30 | 2026-03-25 10:15:00 | 1907.00 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2144.90 | 2026-04-28 09:15:00 | 2359.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2132.10 | 2026-04-28 09:15:00 | 2345.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 14:45:00 | 2130.00 | 2026-04-28 09:15:00 | 2343.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 15:15:00 | 2130.20 | 2026-04-28 09:15:00 | 2343.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-23 10:15:00 | 2261.90 | 2026-05-04 09:15:00 | 2488.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-24 14:00:00 | 2261.70 | 2026-05-04 09:15:00 | 2487.87 | TARGET_HIT | 1.00 | 10.00% |
