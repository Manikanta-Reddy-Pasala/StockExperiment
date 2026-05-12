# BEML Ltd. (BEML)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1952.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 19 |
| ALERT3 | 149 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 55 |
| PARTIAL | 5 |
| TARGET_HIT | 6 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 48
- **Target hits / Stop hits / Partials:** 6 / 53 / 5
- **Avg / median % per leg:** 0.11% / -1.10%
- **Sum % (uncompounded):** 6.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 6 | 35.3% | 6 | 11 | 0 | 2.14% | 36.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.43% | -1.4% |
| BUY @ 3rd Alert (retest2) | 16 | 6 | 37.5% | 6 | 10 | 0 | 2.36% | 37.7% |
| SELL (all) | 47 | 10 | 21.3% | 0 | 42 | 5 | -0.62% | -29.3% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.16% | -0.6% |
| SELL @ 3rd Alert (retest2) | 43 | 8 | 18.6% | 0 | 39 | 4 | -0.67% | -28.7% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.41% | -2.0% |
| retest2 (combined) | 59 | 14 | 23.7% | 6 | 49 | 4 | 0.15% | 9.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1592.20 | 1558.14 | 1554.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1642.60 | 1595.98 | 1576.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1677.05 | 1685.71 | 1658.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:30:00 | 1679.65 | 1685.71 | 1658.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1785.85 | 1837.92 | 1802.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 1783.90 | 1837.92 | 1802.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1811.05 | 1832.55 | 1803.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 1826.50 | 1805.74 | 1799.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:30:00 | 1815.20 | 1807.23 | 1802.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 1814.65 | 1808.22 | 1803.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 1827.00 | 1813.86 | 1806.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1809.40 | 1812.75 | 1807.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 1804.25 | 1812.75 | 1807.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1805.80 | 1811.36 | 1807.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 1805.30 | 1811.36 | 1807.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1809.60 | 1811.01 | 1807.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:15:00 | 1797.60 | 1811.01 | 1807.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1814.65 | 1811.74 | 1808.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 1799.75 | 1811.74 | 1808.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1814.50 | 1813.21 | 1809.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 1812.20 | 1813.21 | 1809.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 1826.80 | 1815.93 | 1811.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 1852.00 | 1829.18 | 1818.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-26 11:15:00 | 2009.15 | 1890.04 | 1852.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 14:15:00 | 2183.45 | 2192.58 | 2192.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 10:15:00 | 2171.65 | 2182.62 | 2186.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 2155.75 | 2131.57 | 2149.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 2155.75 | 2131.57 | 2149.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2155.75 | 2131.57 | 2149.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:00:00 | 2155.75 | 2131.57 | 2149.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 2180.10 | 2141.28 | 2152.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 2180.10 | 2141.28 | 2152.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 2152.00 | 2143.42 | 2152.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:30:00 | 2177.65 | 2143.42 | 2152.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 2144.85 | 2143.71 | 2151.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 2120.60 | 2144.53 | 2149.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 12:15:00 | 2175.00 | 2150.35 | 2151.10 | SL hit (close>static) qty=1.00 sl=2163.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 2178.00 | 2155.88 | 2153.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 2193.90 | 2163.49 | 2157.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 2185.00 | 2190.02 | 2178.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 09:15:00 | 2177.55 | 2190.02 | 2178.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2215.00 | 2195.02 | 2181.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 2225.50 | 2197.75 | 2184.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 2225.00 | 2204.80 | 2188.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 13:15:00 | 2224.50 | 2208.44 | 2191.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 2148.45 | 2195.34 | 2195.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 2148.45 | 2195.34 | 2195.60 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 2334.00 | 2205.48 | 2196.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 13:15:00 | 2339.75 | 2251.54 | 2219.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 2248.40 | 2343.17 | 2304.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 2248.40 | 2343.17 | 2304.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2248.40 | 2343.17 | 2304.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 2248.40 | 2343.17 | 2304.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 2250.00 | 2324.54 | 2299.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:15:00 | 2242.90 | 2324.54 | 2299.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 14:15:00 | 2224.30 | 2275.17 | 2281.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 09:15:00 | 2219.90 | 2255.78 | 2271.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 10:15:00 | 2226.05 | 2225.61 | 2243.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:30:00 | 2228.25 | 2225.61 | 2243.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 2224.50 | 2224.38 | 2235.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 2240.35 | 2224.38 | 2235.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2240.05 | 2227.51 | 2236.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 2253.10 | 2227.51 | 2236.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 2232.30 | 2228.47 | 2235.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 2227.65 | 2228.47 | 2235.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:45:00 | 2222.00 | 2228.14 | 2235.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 09:45:00 | 2229.00 | 2223.36 | 2229.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 10:45:00 | 2228.80 | 2224.22 | 2229.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 2234.65 | 2226.30 | 2230.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 2234.65 | 2226.30 | 2230.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 2229.05 | 2226.85 | 2229.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 2223.25 | 2226.85 | 2229.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 15:15:00 | 2227.05 | 2226.27 | 2229.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 2244.15 | 2229.97 | 2230.27 | SL hit (close>static) qty=1.00 sl=2242.20 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 15:15:00 | 2226.00 | 2213.44 | 2211.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 2322.15 | 2235.18 | 2221.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 15:15:00 | 2280.50 | 2283.35 | 2266.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 2294.45 | 2283.35 | 2266.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2264.05 | 2279.49 | 2265.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 2264.05 | 2279.49 | 2265.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 2263.80 | 2276.35 | 2265.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:30:00 | 2258.50 | 2276.35 | 2265.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2260.05 | 2273.09 | 2265.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 2260.05 | 2273.09 | 2265.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 2262.70 | 2271.02 | 2264.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 2293.90 | 2265.47 | 2263.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 2227.35 | 2292.24 | 2294.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 2227.35 | 2292.24 | 2294.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 2217.50 | 2244.74 | 2266.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 2252.50 | 2241.93 | 2261.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 2252.50 | 2241.93 | 2261.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 2252.50 | 2241.93 | 2261.34 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 2307.00 | 2264.32 | 2263.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 2328.20 | 2277.10 | 2269.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 2311.15 | 2311.90 | 2300.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 14:45:00 | 2311.40 | 2311.90 | 2300.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 2302.70 | 2310.06 | 2300.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 2303.95 | 2310.06 | 2300.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 2262.50 | 2300.55 | 2296.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 2262.50 | 2300.55 | 2296.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 2282.00 | 2296.84 | 2295.60 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 2279.50 | 2293.37 | 2294.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 13:15:00 | 2270.10 | 2286.90 | 2290.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 2159.50 | 2153.88 | 2173.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 2159.50 | 2153.88 | 2173.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 2159.50 | 2153.88 | 2173.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:45:00 | 2154.90 | 2156.48 | 2167.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 2047.15 | 2095.45 | 2123.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 2019.80 | 2017.49 | 2053.64 | SL hit (close>ema200) qty=0.50 sl=2017.49 alert=retest2 |

### Cycle 11 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 1996.15 | 1981.38 | 1980.59 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 1969.80 | 1982.14 | 1983.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1953.00 | 1972.37 | 1977.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 1962.80 | 1939.62 | 1947.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 1962.80 | 1939.62 | 1947.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1962.80 | 1939.62 | 1947.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:15:00 | 1985.00 | 1939.62 | 1947.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1999.35 | 1951.57 | 1952.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 2015.45 | 1951.57 | 1952.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 1982.90 | 1957.84 | 1955.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 2005.75 | 1972.14 | 1966.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 2013.05 | 2015.47 | 1995.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 2013.05 | 2015.47 | 1995.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 2042.40 | 2059.83 | 2052.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 2042.40 | 2059.83 | 2052.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 2043.35 | 2056.53 | 2051.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 2058.90 | 2056.53 | 2051.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 12:30:00 | 2046.50 | 2054.27 | 2052.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 13:15:00 | 2031.50 | 2049.72 | 2050.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 2031.50 | 2049.72 | 2050.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 2013.40 | 2042.45 | 2046.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1945.55 | 1928.47 | 1947.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1945.55 | 1928.47 | 1947.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1938.25 | 1930.43 | 1947.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1944.60 | 1930.43 | 1947.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1933.15 | 1925.37 | 1937.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1941.65 | 1925.37 | 1937.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1933.85 | 1927.07 | 1937.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 1932.55 | 1927.07 | 1937.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1931.50 | 1927.96 | 1936.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 1934.00 | 1927.96 | 1936.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1936.85 | 1929.73 | 1936.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 1936.00 | 1929.73 | 1936.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1931.80 | 1930.15 | 1936.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:15:00 | 1936.50 | 1930.15 | 1936.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1940.90 | 1932.30 | 1936.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:30:00 | 1941.45 | 1932.30 | 1936.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1966.00 | 1939.04 | 1939.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1992.10 | 1939.04 | 1939.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1993.70 | 1949.97 | 1944.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 2050.90 | 1981.11 | 1960.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 2016.05 | 2034.20 | 2012.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 2016.05 | 2034.20 | 2012.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 2010.30 | 2029.42 | 2012.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 2010.00 | 2029.42 | 2012.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 2007.25 | 2024.99 | 2012.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 2007.25 | 2024.99 | 2012.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1994.65 | 2018.92 | 2010.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 1994.65 | 2018.92 | 2010.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1976.00 | 2005.40 | 2005.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 1971.05 | 1998.53 | 2002.36 | Break + close below crossover candle low |

### Cycle 17 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 2042.00 | 2007.23 | 2005.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 10:15:00 | 2062.95 | 2018.37 | 2011.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 13:15:00 | 2046.45 | 2046.57 | 2034.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 14:00:00 | 2046.45 | 2046.57 | 2034.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2041.95 | 2045.84 | 2036.98 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 2020.50 | 2031.60 | 2032.97 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 2039.15 | 2033.97 | 2033.87 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 2020.05 | 2033.32 | 2034.17 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 2086.70 | 2041.03 | 2037.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 2178.75 | 2079.45 | 2055.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 2176.50 | 2181.68 | 2147.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 13:15:00 | 2212.45 | 2182.85 | 2168.01 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2180.90 | 2190.80 | 2182.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 2180.90 | 2190.80 | 2182.24 | SL hit (close<ema400) qty=1.00 sl=2182.24 alert=retest1 |

### Cycle 22 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 2173.25 | 2187.94 | 2189.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 12:15:00 | 2149.35 | 2173.61 | 2181.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 2168.00 | 2167.60 | 2174.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 11:30:00 | 2163.00 | 2167.60 | 2174.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 2162.50 | 2165.85 | 2172.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 2174.10 | 2165.85 | 2172.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2181.35 | 2166.32 | 2170.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 2191.10 | 2166.32 | 2170.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 2212.05 | 2175.46 | 2174.51 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 2135.00 | 2171.94 | 2174.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 2096.35 | 2141.44 | 2157.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 10:15:00 | 2073.40 | 2066.42 | 2096.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 11:00:00 | 2073.40 | 2066.42 | 2096.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 2100.00 | 2072.76 | 2087.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 2123.60 | 2072.76 | 2087.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 2157.15 | 2089.64 | 2094.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 2157.45 | 2089.64 | 2094.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 2152.60 | 2102.23 | 2099.35 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 2129.40 | 2159.68 | 2162.95 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 2190.60 | 2153.92 | 2153.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 2194.40 | 2172.00 | 2168.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 2183.75 | 2185.85 | 2176.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 2183.75 | 2185.85 | 2176.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 2183.75 | 2185.85 | 2176.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 2183.75 | 2185.85 | 2176.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 2159.90 | 2180.66 | 2175.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 2159.90 | 2180.66 | 2175.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 2156.70 | 2175.87 | 2173.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 2156.70 | 2175.87 | 2173.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 2180.15 | 2176.73 | 2174.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:30:00 | 2198.15 | 2188.34 | 2180.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 2174.95 | 2208.33 | 2212.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 2174.95 | 2208.33 | 2212.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 13:15:00 | 2166.75 | 2186.75 | 2199.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 2189.95 | 2182.83 | 2194.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 2179.60 | 2182.83 | 2194.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 2205.15 | 2187.29 | 2195.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 2205.15 | 2187.29 | 2195.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 2202.70 | 2190.37 | 2195.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 2208.00 | 2190.37 | 2195.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 2210.00 | 2194.30 | 2197.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 2210.00 | 2194.30 | 2197.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 12:15:00 | 2210.50 | 2199.73 | 2199.19 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 2194.35 | 2198.65 | 2198.75 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 2226.10 | 2204.17 | 2201.24 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 2183.50 | 2208.36 | 2210.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 2180.90 | 2202.87 | 2207.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 2157.35 | 2154.38 | 2170.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 14:30:00 | 2157.45 | 2154.38 | 2170.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2161.25 | 2156.66 | 2169.15 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 12:15:00 | 2204.95 | 2179.17 | 2177.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 2233.55 | 2202.45 | 2189.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 2203.30 | 2206.41 | 2195.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 12:15:00 | 2203.30 | 2206.41 | 2195.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 2203.30 | 2206.41 | 2195.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:45:00 | 2198.65 | 2206.41 | 2195.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 2200.00 | 2205.15 | 2196.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 2200.00 | 2205.15 | 2196.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 2203.50 | 2204.82 | 2197.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 2177.20 | 2204.82 | 2197.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 2181.20 | 2200.10 | 2195.83 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 12:15:00 | 2183.60 | 2192.03 | 2192.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 2172.40 | 2188.37 | 2190.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 2002.30 | 2001.88 | 2054.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:45:00 | 1999.50 | 2001.88 | 2054.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 2032.10 | 2008.41 | 2040.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 2034.00 | 2008.41 | 2040.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2036.50 | 2014.03 | 2040.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 2035.50 | 2014.03 | 2040.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2005.00 | 2009.17 | 2026.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 2017.70 | 2009.17 | 2026.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 2015.80 | 2011.06 | 2024.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 2015.80 | 2011.06 | 2024.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 2023.00 | 2013.66 | 2021.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 2025.20 | 2013.66 | 2021.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 2035.00 | 2017.93 | 2022.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 2035.00 | 2017.93 | 2022.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 2017.90 | 2017.92 | 2021.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:15:00 | 2012.60 | 2017.92 | 2021.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 2013.90 | 2017.12 | 2021.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 2014.70 | 2020.27 | 2022.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 2013.00 | 2018.82 | 2021.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 2001.20 | 2013.08 | 2018.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:45:00 | 1994.10 | 2009.07 | 2015.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:30:00 | 1987.50 | 1993.59 | 2003.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 2029.40 | 2000.75 | 2006.07 | SL hit (close>static) qty=1.00 sl=2026.80 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 2016.00 | 2009.49 | 2009.20 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 2004.00 | 2008.39 | 2008.73 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 2010.20 | 2009.01 | 2008.97 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 2008.50 | 2008.91 | 2008.93 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 2010.00 | 2009.13 | 2009.02 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 2007.70 | 2008.84 | 2008.90 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 2010.00 | 2009.07 | 2009.00 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1983.60 | 2003.98 | 2006.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1979.00 | 1989.99 | 1998.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1972.90 | 1961.44 | 1975.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1972.90 | 1961.44 | 1975.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1972.90 | 1961.44 | 1975.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1972.90 | 1961.44 | 1975.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1957.50 | 1960.65 | 1973.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1946.50 | 1964.06 | 1970.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 1849.17 | 1903.72 | 1933.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1850.90 | 1845.98 | 1867.32 | SL hit (close>ema200) qty=0.50 sl=1845.98 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1683.90 | 1679.83 | 1679.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 1701.10 | 1687.34 | 1683.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1714.50 | 1722.71 | 1707.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 1711.20 | 1719.16 | 1708.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1711.20 | 1719.16 | 1708.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 1709.30 | 1719.16 | 1708.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1706.40 | 1716.61 | 1708.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 1707.30 | 1716.61 | 1708.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1705.00 | 1714.29 | 1707.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:15:00 | 1703.00 | 1714.29 | 1707.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1705.10 | 1712.45 | 1707.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 1702.20 | 1712.45 | 1707.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1703.00 | 1710.56 | 1707.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1704.30 | 1710.56 | 1707.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1702.40 | 1708.93 | 1706.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:15:00 | 1692.10 | 1708.93 | 1706.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1697.70 | 1706.68 | 1706.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 1694.70 | 1706.68 | 1706.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 1696.00 | 1704.55 | 1705.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 1688.00 | 1701.24 | 1703.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1683.30 | 1675.47 | 1683.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 1683.30 | 1675.47 | 1683.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1683.30 | 1675.47 | 1683.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 1691.90 | 1675.47 | 1683.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1686.00 | 1677.58 | 1683.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 1682.20 | 1677.58 | 1683.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1689.80 | 1680.02 | 1684.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 1689.80 | 1680.02 | 1684.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1681.40 | 1680.30 | 1684.19 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1720.90 | 1692.39 | 1689.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1765.10 | 1712.08 | 1699.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 1875.00 | 1881.35 | 1854.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 15:00:00 | 1875.00 | 1881.35 | 1854.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1853.40 | 1882.86 | 1869.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1853.40 | 1882.86 | 1869.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1856.70 | 1877.63 | 1868.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 1840.20 | 1877.63 | 1868.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 1823.00 | 1859.98 | 1861.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 1820.30 | 1852.04 | 1857.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1866.70 | 1847.89 | 1852.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 1866.70 | 1847.89 | 1852.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1866.70 | 1847.89 | 1852.67 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1857.30 | 1855.13 | 1855.13 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 1845.20 | 1854.90 | 1855.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 1840.50 | 1849.96 | 1852.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 1849.00 | 1848.40 | 1851.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 15:00:00 | 1849.00 | 1848.40 | 1851.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1852.00 | 1849.12 | 1851.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1863.00 | 1849.12 | 1851.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1859.80 | 1851.25 | 1852.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:15:00 | 1873.20 | 1851.25 | 1852.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1882.00 | 1857.40 | 1854.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 1902.70 | 1872.71 | 1864.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1873.30 | 1883.27 | 1875.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 1873.30 | 1883.27 | 1875.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1873.30 | 1883.27 | 1875.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1872.50 | 1883.27 | 1875.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1863.70 | 1879.36 | 1874.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 1861.30 | 1879.36 | 1874.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 1843.50 | 1866.35 | 1869.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 1838.80 | 1854.22 | 1858.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 1785.40 | 1783.97 | 1803.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 1789.00 | 1783.97 | 1803.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1817.00 | 1790.57 | 1805.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 1779.20 | 1790.12 | 1803.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1784.20 | 1788.38 | 1801.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1781.10 | 1776.15 | 1787.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1831.80 | 1789.33 | 1792.04 | SL hit (close>static) qty=1.00 sl=1820.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 1819.00 | 1795.27 | 1794.49 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 1791.00 | 1798.48 | 1798.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 1780.40 | 1794.86 | 1797.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 15:15:00 | 1779.00 | 1778.97 | 1786.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:15:00 | 1751.20 | 1778.97 | 1786.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1663.64 | 1714.84 | 1743.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1713.60 | 1692.82 | 1717.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1713.60 | 1692.82 | 1717.27 | SL hit (close>ema200) qty=0.50 sl=1692.82 alert=retest1 |

### Cycle 53 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 1741.90 | 1679.53 | 1675.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 1778.40 | 1699.31 | 1684.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 1785.60 | 1787.07 | 1765.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:15:00 | 1783.80 | 1787.07 | 1765.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1788.30 | 1794.01 | 1777.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 1788.30 | 1794.01 | 1777.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1750.20 | 1785.25 | 1774.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1750.20 | 1785.25 | 1774.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1713.40 | 1770.88 | 1769.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1719.40 | 1770.88 | 1769.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1699.10 | 1756.52 | 1762.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1682.70 | 1741.76 | 1755.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 1712.00 | 1706.40 | 1727.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 1712.00 | 1706.40 | 1727.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1734.00 | 1711.92 | 1728.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1734.00 | 1711.92 | 1728.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1744.00 | 1718.33 | 1729.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1755.50 | 1718.33 | 1729.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1758.60 | 1738.59 | 1737.20 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 1742.00 | 1746.30 | 1746.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1715.90 | 1739.64 | 1743.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 1745.80 | 1739.79 | 1742.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 1745.80 | 1739.79 | 1742.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 1745.80 | 1739.79 | 1742.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 1752.00 | 1739.79 | 1742.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1739.00 | 1739.64 | 1742.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 14:15:00 | 1604.00 | 1739.57 | 1742.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 13:15:00 | 1758.30 | 1713.90 | 1719.74 | SL hit (close>static) qty=1.00 sl=1748.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 1759.00 | 1730.36 | 1726.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1775.80 | 1739.45 | 1731.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1749.70 | 1757.66 | 1746.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 1749.70 | 1757.66 | 1746.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1749.70 | 1757.66 | 1746.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1745.40 | 1757.66 | 1746.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1771.00 | 1760.33 | 1749.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:30:00 | 1765.00 | 1760.33 | 1749.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1743.20 | 1760.40 | 1754.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 1743.20 | 1760.40 | 1754.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1751.60 | 1758.64 | 1754.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 1745.00 | 1758.64 | 1754.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 1749.00 | 1756.71 | 1753.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 1749.00 | 1756.71 | 1753.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 1743.10 | 1752.91 | 1752.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:45:00 | 1738.00 | 1752.91 | 1752.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 1739.00 | 1750.13 | 1751.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 1733.00 | 1746.71 | 1749.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 1742.90 | 1740.51 | 1745.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:00:00 | 1742.90 | 1740.51 | 1745.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1744.80 | 1741.37 | 1745.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 1737.50 | 1740.87 | 1744.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 1735.60 | 1739.82 | 1744.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:45:00 | 1736.00 | 1729.64 | 1734.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 1732.30 | 1728.55 | 1733.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1729.60 | 1728.07 | 1732.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:00:00 | 1729.60 | 1728.07 | 1732.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1731.80 | 1728.81 | 1732.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:45:00 | 1734.00 | 1728.81 | 1732.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1737.00 | 1730.45 | 1733.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 1737.00 | 1730.45 | 1733.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1736.50 | 1731.66 | 1733.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 1745.00 | 1731.66 | 1733.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 1757.00 | 1736.73 | 1735.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 1757.00 | 1736.73 | 1735.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 1762.50 | 1741.88 | 1737.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 1743.30 | 1758.07 | 1751.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 1743.30 | 1758.07 | 1751.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1743.30 | 1758.07 | 1751.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1743.30 | 1758.07 | 1751.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1743.30 | 1755.11 | 1750.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 1747.10 | 1755.11 | 1750.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 13:15:00 | 1736.30 | 1751.35 | 1749.24 | SL hit (close<static) qty=1.00 sl=1738.50 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 1719.80 | 1745.04 | 1746.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1716.60 | 1739.35 | 1743.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 13:15:00 | 1696.90 | 1693.62 | 1704.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 14:00:00 | 1696.90 | 1693.62 | 1704.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1691.50 | 1693.21 | 1702.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 1713.30 | 1693.21 | 1702.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1722.60 | 1699.09 | 1703.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:45:00 | 1729.80 | 1699.09 | 1703.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1721.00 | 1703.47 | 1705.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 1723.90 | 1703.47 | 1705.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1722.60 | 1707.30 | 1707.03 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 1703.00 | 1709.49 | 1709.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 1656.90 | 1697.32 | 1703.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 1688.00 | 1686.42 | 1695.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 14:00:00 | 1688.00 | 1686.42 | 1695.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1669.30 | 1681.23 | 1690.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:15:00 | 1597.70 | 1642.24 | 1662.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 12:45:00 | 1599.00 | 1627.97 | 1652.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 13:15:00 | 1599.10 | 1627.97 | 1652.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 13:45:00 | 1598.40 | 1623.22 | 1647.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 1633.60 | 1617.01 | 1632.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 1637.80 | 1617.01 | 1632.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1612.80 | 1616.17 | 1630.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-06 10:15:00 | 1689.60 | 1647.08 | 1641.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 1689.60 | 1647.08 | 1641.50 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1593.80 | 1644.53 | 1644.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 1572.90 | 1604.51 | 1623.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 1599.10 | 1595.18 | 1611.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:00:00 | 1599.10 | 1595.18 | 1611.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 1609.00 | 1599.16 | 1608.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 15:00:00 | 1609.00 | 1599.16 | 1608.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1612.00 | 1601.73 | 1608.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 1626.00 | 1601.73 | 1608.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1612.30 | 1603.84 | 1609.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 1603.70 | 1603.67 | 1608.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:00:00 | 1603.00 | 1603.67 | 1608.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 12:15:00 | 1624.60 | 1612.87 | 1612.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 12:15:00 | 1624.60 | 1612.87 | 1612.20 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1578.70 | 1606.19 | 1609.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1553.20 | 1597.67 | 1603.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1514.70 | 1507.01 | 1533.39 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1498.60 | 1506.41 | 1530.72 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1497.60 | 1505.72 | 1528.20 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1556.30 | 1519.40 | 1526.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 1556.30 | 1519.40 | 1526.40 | SL hit (close>ema400) qty=1.00 sl=1526.40 alert=retest1 |

### Cycle 67 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1553.90 | 1534.92 | 1532.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 14:15:00 | 1561.20 | 1542.97 | 1536.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1536.20 | 1544.98 | 1538.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1536.20 | 1544.98 | 1538.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1536.20 | 1544.98 | 1538.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1537.50 | 1544.98 | 1538.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1537.80 | 1543.54 | 1538.87 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1515.10 | 1535.23 | 1536.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1499.90 | 1528.16 | 1532.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 1539.00 | 1527.18 | 1530.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 1539.00 | 1527.18 | 1530.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1539.00 | 1527.18 | 1530.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 1539.00 | 1527.18 | 1530.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1519.70 | 1525.68 | 1529.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 1513.60 | 1525.68 | 1529.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 1514.10 | 1521.29 | 1526.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1437.92 | 1502.09 | 1516.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1438.39 | 1502.09 | 1516.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 1430.90 | 1430.73 | 1465.31 | SL hit (close>ema200) qty=0.50 sl=1430.73 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1510.50 | 1466.51 | 1465.75 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1432.60 | 1471.15 | 1472.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1413.30 | 1441.01 | 1454.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1484.00 | 1412.82 | 1428.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1484.00 | 1412.82 | 1428.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1484.00 | 1412.82 | 1428.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1484.00 | 1412.82 | 1428.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1462.20 | 1422.70 | 1431.49 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 1490.00 | 1446.80 | 1441.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1504.70 | 1478.99 | 1464.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 13:15:00 | 1619.20 | 1619.27 | 1594.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 14:00:00 | 1619.20 | 1619.27 | 1594.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1593.80 | 1614.96 | 1598.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 1605.30 | 1615.37 | 1600.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 1765.83 | 1734.50 | 1697.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 14:15:00 | 1790.80 | 1800.29 | 1801.09 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 1819.00 | 1803.19 | 1802.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 1838.00 | 1814.97 | 1808.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 14:15:00 | 1818.00 | 1820.74 | 1812.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-27 15:00:00 | 1818.00 | 1820.74 | 1812.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1822.60 | 1821.47 | 1814.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:00:00 | 1822.60 | 1821.47 | 1814.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1818.60 | 1821.62 | 1815.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 1820.00 | 1821.62 | 1815.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1810.90 | 1819.48 | 1815.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:30:00 | 1813.10 | 1819.48 | 1815.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1809.50 | 1817.48 | 1814.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 1809.00 | 1817.48 | 1814.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1842.50 | 1822.49 | 1817.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 1854.30 | 1825.99 | 1819.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1783.10 | 1818.71 | 1820.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1783.10 | 1818.71 | 1820.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 1779.30 | 1810.83 | 1816.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 1804.90 | 1801.01 | 1809.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 1804.90 | 1801.01 | 1809.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1804.90 | 1801.01 | 1809.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 1813.80 | 1801.01 | 1809.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1801.40 | 1801.09 | 1808.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1788.00 | 1801.09 | 1808.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1792.20 | 1799.31 | 1807.03 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 1829.80 | 1809.61 | 1809.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 14:15:00 | 1830.50 | 1813.79 | 1811.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 1950.10 | 1956.57 | 1927.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:45:00 | 1952.00 | 1956.57 | 1927.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 10:00:00 | 1826.50 | 2025-05-26 11:15:00 | 2009.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 12:30:00 | 1815.20 | 2025-05-26 11:15:00 | 1996.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 15:00:00 | 1814.65 | 2025-05-26 11:15:00 | 1996.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 09:30:00 | 1827.00 | 2025-05-26 11:15:00 | 2009.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-23 13:30:00 | 1852.00 | 2025-05-26 11:15:00 | 2037.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-16 09:30:00 | 2120.60 | 2025-06-16 12:15:00 | 2175.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-06-18 10:30:00 | 2225.50 | 2025-06-19 12:15:00 | 2148.45 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2025-06-18 11:45:00 | 2225.00 | 2025-06-19 12:15:00 | 2148.45 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-06-18 13:15:00 | 2224.50 | 2025-06-19 12:15:00 | 2148.45 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-06-27 11:15:00 | 2227.65 | 2025-07-01 09:15:00 | 2244.15 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-06-27 11:45:00 | 2222.00 | 2025-07-01 09:15:00 | 2244.15 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-30 09:45:00 | 2229.00 | 2025-07-01 09:15:00 | 2244.15 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-06-30 10:45:00 | 2228.80 | 2025-07-01 09:15:00 | 2244.15 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-06-30 13:15:00 | 2223.25 | 2025-07-03 13:15:00 | 2228.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-06-30 15:15:00 | 2227.05 | 2025-07-03 13:15:00 | 2228.50 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-07-01 10:45:00 | 2226.00 | 2025-07-03 13:15:00 | 2228.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-07-01 12:45:00 | 2223.95 | 2025-07-03 13:15:00 | 2228.50 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-07-02 12:00:00 | 2194.65 | 2025-07-03 15:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-07-02 12:30:00 | 2195.00 | 2025-07-03 15:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-07-02 13:00:00 | 2190.00 | 2025-07-03 15:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-07-03 10:15:00 | 2192.15 | 2025-07-03 15:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-07-09 09:15:00 | 2293.90 | 2025-07-11 09:15:00 | 2227.35 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-07-24 14:45:00 | 2154.90 | 2025-07-28 09:15:00 | 2047.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 14:45:00 | 2154.90 | 2025-07-29 12:15:00 | 2019.80 | STOP_HIT | 0.50 | 6.27% |
| BUY | retest2 | 2025-08-21 09:15:00 | 2058.90 | 2025-08-21 13:15:00 | 2031.50 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-08-21 12:30:00 | 2046.50 | 2025-08-21 13:15:00 | 2031.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2025-09-17 13:15:00 | 2212.45 | 2025-09-18 13:15:00 | 2180.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-09-22 09:15:00 | 2189.30 | 2025-09-22 15:15:00 | 2157.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-15 09:30:00 | 2198.15 | 2025-10-20 09:15:00 | 2174.95 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-11-12 11:15:00 | 2012.60 | 2025-11-14 10:15:00 | 2029.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-11-12 12:00:00 | 2013.90 | 2025-11-14 10:15:00 | 2029.40 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-12 14:15:00 | 2014.70 | 2025-11-14 15:15:00 | 2016.00 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-11-12 15:00:00 | 2013.00 | 2025-11-14 15:15:00 | 2016.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-11-13 10:45:00 | 1994.10 | 2025-11-14 15:15:00 | 2016.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-11-14 09:30:00 | 1987.50 | 2025-11-14 15:15:00 | 2016.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1946.50 | 2025-11-24 10:15:00 | 1849.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1946.50 | 2025-11-26 09:15:00 | 1850.90 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2026-01-13 09:30:00 | 1779.20 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1784.20 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1781.10 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest1 | 2026-01-20 09:15:00 | 1751.20 | 2026-01-21 10:15:00 | 1663.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-20 09:15:00 | 1751.20 | 2026-01-22 09:15:00 | 1713.60 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1691.90 | 2026-01-28 10:15:00 | 1741.90 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-01-22 13:15:00 | 1696.00 | 2026-01-28 10:15:00 | 1741.90 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-02-06 14:15:00 | 1604.00 | 2026-02-09 13:15:00 | 1758.30 | STOP_HIT | 1.00 | -9.62% |
| SELL | retest2 | 2026-02-13 14:15:00 | 1737.50 | 2026-02-18 09:15:00 | 1757.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-13 15:00:00 | 1735.60 | 2026-02-18 09:15:00 | 1757.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-02-17 09:45:00 | 1736.00 | 2026-02-18 09:15:00 | 1757.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-02-17 10:30:00 | 1732.30 | 2026-02-18 09:15:00 | 1757.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-02-19 13:15:00 | 1747.10 | 2026-02-19 13:15:00 | 1736.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-03-04 11:15:00 | 1597.70 | 2026-03-06 10:15:00 | 1689.60 | STOP_HIT | 1.00 | -5.75% |
| SELL | retest2 | 2026-03-04 12:45:00 | 1599.00 | 2026-03-06 10:15:00 | 1689.60 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2026-03-04 13:15:00 | 1599.10 | 2026-03-06 10:15:00 | 1689.60 | STOP_HIT | 1.00 | -5.66% |
| SELL | retest2 | 2026-03-04 13:45:00 | 1598.40 | 2026-03-06 10:15:00 | 1689.60 | STOP_HIT | 1.00 | -5.71% |
| SELL | retest2 | 2026-03-11 10:30:00 | 1603.70 | 2026-03-11 12:15:00 | 1624.60 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-03-11 11:00:00 | 1603.00 | 2026-03-11 12:15:00 | 1624.60 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest1 | 2026-03-17 11:15:00 | 1498.60 | 2026-03-18 09:15:00 | 1556.30 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest1 | 2026-03-17 12:15:00 | 1497.60 | 2026-03-18 09:15:00 | 1556.30 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1513.60 | 2026-03-23 09:15:00 | 1437.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1514.10 | 2026-03-23 09:15:00 | 1438.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1513.60 | 2026-03-24 09:15:00 | 1430.90 | STOP_HIT | 0.50 | 5.46% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1514.10 | 2026-03-24 09:15:00 | 1430.90 | STOP_HIT | 0.50 | 5.50% |
| BUY | retest2 | 2026-04-13 10:30:00 | 1605.30 | 2026-04-17 09:15:00 | 1765.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 09:15:00 | 1854.30 | 2026-04-30 09:15:00 | 1783.10 | STOP_HIT | 1.00 | -3.84% |
