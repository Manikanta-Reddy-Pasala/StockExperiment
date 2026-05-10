# Poly Medicure Ltd. (POLYMED)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1649.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 48 |
| ALERT2 | 48 |
| ALERT2_SKIP | 24 |
| ALERT3 | 115 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 48 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 25 / 30
- **Target hits / Stop hits / Partials:** 4 / 41 / 10
- **Avg / median % per leg:** 1.24% / -0.63%
- **Sum % (uncompounded):** 68.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 4 | 26.7% | 4 | 11 | 0 | 1.43% | 21.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 4 | 26.7% | 4 | 11 | 0 | 1.43% | 21.5% |
| SELL (all) | 40 | 21 | 52.5% | 0 | 30 | 10 | 1.17% | 46.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -6.98% | -7.0% |
| SELL @ 3rd Alert (retest2) | 39 | 21 | 53.8% | 0 | 29 | 10 | 1.38% | 53.9% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -6.98% | -7.0% |
| retest2 (combined) | 54 | 25 | 46.3% | 4 | 40 | 10 | 1.40% | 75.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 14:15:00 | 2442.00 | 2402.95 | 2398.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 2472.30 | 2423.23 | 2409.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 2440.10 | 2444.45 | 2425.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 14:00:00 | 2440.10 | 2444.45 | 2425.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2467.10 | 2477.10 | 2462.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 2467.10 | 2477.10 | 2462.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 2465.60 | 2474.80 | 2462.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 2461.80 | 2474.80 | 2462.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 2445.60 | 2468.96 | 2460.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 2445.60 | 2468.96 | 2460.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 2444.50 | 2464.07 | 2459.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:00:00 | 2444.50 | 2464.07 | 2459.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 2427.70 | 2451.82 | 2454.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 13:15:00 | 2408.90 | 2436.63 | 2446.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 2435.60 | 2418.22 | 2427.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 2435.60 | 2418.22 | 2427.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 2435.60 | 2418.22 | 2427.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 2435.60 | 2418.22 | 2427.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 2432.00 | 2420.97 | 2428.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 2391.60 | 2420.97 | 2428.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 09:15:00 | 2272.02 | 2300.87 | 2330.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 2239.10 | 2235.03 | 2256.28 | SL hit (close>ema200) qty=0.50 sl=2235.03 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 2274.60 | 2243.13 | 2241.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 2293.90 | 2269.83 | 2258.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 2264.10 | 2271.04 | 2261.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 12:15:00 | 2264.10 | 2271.04 | 2261.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 2264.10 | 2271.04 | 2261.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 2264.10 | 2271.04 | 2261.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 2270.40 | 2270.91 | 2262.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 2270.30 | 2270.91 | 2262.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 2301.10 | 2276.86 | 2267.34 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 2250.30 | 2263.57 | 2264.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 2240.40 | 2256.45 | 2261.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 2248.00 | 2244.44 | 2252.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 2248.00 | 2244.44 | 2252.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2248.00 | 2244.44 | 2252.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:00:00 | 2248.00 | 2244.44 | 2252.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 2255.90 | 2246.73 | 2252.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 2255.90 | 2246.73 | 2252.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 2214.50 | 2240.28 | 2249.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 14:45:00 | 2209.00 | 2230.08 | 2241.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 2208.60 | 2229.46 | 2240.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:45:00 | 2209.10 | 2224.49 | 2237.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 2098.55 | 2127.18 | 2154.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 2098.17 | 2127.18 | 2154.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 2098.64 | 2127.18 | 2154.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 2088.90 | 2082.25 | 2115.81 | SL hit (close>ema200) qty=0.50 sl=2082.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 2088.90 | 2082.25 | 2115.81 | SL hit (close>ema200) qty=0.50 sl=2082.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 2088.90 | 2082.25 | 2115.81 | SL hit (close>ema200) qty=0.50 sl=2082.25 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 15:15:00 | 2135.00 | 2100.59 | 2097.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 2189.60 | 2136.58 | 2121.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 2187.40 | 2194.59 | 2172.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 2187.40 | 2194.59 | 2172.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 2187.40 | 2194.59 | 2172.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:45:00 | 2183.30 | 2194.59 | 2172.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2162.70 | 2187.48 | 2172.84 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 2160.10 | 2167.29 | 2167.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 10:15:00 | 2147.20 | 2163.00 | 2165.49 | Break + close below crossover candle low |

### Cycle 7 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 2197.40 | 2167.38 | 2166.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 2233.10 | 2180.53 | 2172.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 2237.40 | 2256.41 | 2239.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 2237.40 | 2256.41 | 2239.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2237.40 | 2256.41 | 2239.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 2237.40 | 2256.41 | 2239.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2233.00 | 2251.73 | 2238.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 2233.00 | 2251.73 | 2238.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 2249.00 | 2250.92 | 2240.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 2244.50 | 2250.92 | 2240.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 2252.60 | 2251.26 | 2241.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 11:15:00 | 2260.70 | 2245.93 | 2241.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 12:30:00 | 2262.20 | 2246.16 | 2242.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 2236.30 | 2244.19 | 2241.88 | SL hit (close<static) qty=1.00 sl=2240.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 2236.30 | 2244.19 | 2241.88 | SL hit (close<static) qty=1.00 sl=2240.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 2267.90 | 2243.26 | 2241.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 2237.30 | 2242.30 | 2241.68 | SL hit (close<static) qty=1.00 sl=2240.90 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 2229.90 | 2239.82 | 2240.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 2220.00 | 2235.85 | 2238.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 2213.00 | 2210.61 | 2220.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 2210.00 | 2209.82 | 2218.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 2210.00 | 2209.82 | 2218.48 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 2230.30 | 2221.57 | 2220.91 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 2220.00 | 2220.61 | 2220.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 2217.00 | 2219.78 | 2220.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 2219.80 | 2219.79 | 2220.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 11:15:00 | 2219.80 | 2219.79 | 2220.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 2219.80 | 2219.79 | 2220.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:45:00 | 2220.70 | 2219.79 | 2220.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 2217.10 | 2219.25 | 2219.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 2218.80 | 2219.25 | 2219.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 2217.30 | 2218.86 | 2219.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 2217.70 | 2218.86 | 2219.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 2221.00 | 2219.29 | 2219.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 2221.00 | 2219.29 | 2219.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 2215.30 | 2218.49 | 2219.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 2189.70 | 2218.49 | 2219.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:45:00 | 2210.00 | 2211.06 | 2214.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 2222.00 | 2213.25 | 2215.52 | SL hit (close>static) qty=1.00 sl=2221.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 2222.00 | 2213.25 | 2215.52 | SL hit (close>static) qty=1.00 sl=2221.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 2205.10 | 2211.63 | 2214.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 09:15:00 | 2094.84 | 2126.06 | 2156.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 2131.70 | 2120.51 | 2145.40 | SL hit (close>ema200) qty=0.50 sl=2120.51 alert=retest2 |

### Cycle 11 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 2021.70 | 1959.94 | 1959.13 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1958.20 | 1971.78 | 1972.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 1935.90 | 1964.60 | 1969.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 1938.20 | 1936.53 | 1947.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 1945.20 | 1936.53 | 1947.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1936.00 | 1936.43 | 1946.49 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 1987.50 | 1947.85 | 1942.46 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 1925.30 | 1942.87 | 1944.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 15:15:00 | 1918.00 | 1937.90 | 1942.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 1903.70 | 1873.31 | 1896.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 1903.70 | 1873.31 | 1896.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1903.70 | 1873.31 | 1896.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 1897.60 | 1873.31 | 1896.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1903.50 | 1879.35 | 1897.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 1903.00 | 1879.35 | 1897.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1905.70 | 1897.48 | 1900.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 1883.90 | 1899.11 | 1900.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 1914.90 | 1900.17 | 1899.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 1914.90 | 1900.17 | 1899.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 1944.00 | 1911.05 | 1905.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 1965.00 | 1966.83 | 1947.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 15:00:00 | 1965.00 | 1966.83 | 1947.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 2100.00 | 2096.40 | 2073.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 12:00:00 | 2101.00 | 2096.59 | 2079.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 13:15:00 | 2102.30 | 2096.15 | 2080.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 2071.50 | 2081.22 | 2081.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 2071.50 | 2081.22 | 2081.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 10:15:00 | 2071.50 | 2081.22 | 2081.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 12:15:00 | 2068.00 | 2077.04 | 2079.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 2077.40 | 2077.11 | 2079.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 13:15:00 | 2077.40 | 2077.11 | 2079.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 2077.40 | 2077.11 | 2079.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 2075.90 | 2077.11 | 2079.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 2071.80 | 2076.05 | 2078.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 2079.60 | 2076.05 | 2078.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 2065.10 | 2072.57 | 2076.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:00:00 | 2059.10 | 2068.35 | 2073.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:15:00 | 2053.80 | 2065.70 | 2071.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 15:15:00 | 2082.00 | 2071.02 | 2073.12 | SL hit (close>static) qty=1.00 sl=2079.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 15:15:00 | 2082.00 | 2071.02 | 2073.12 | SL hit (close>static) qty=1.00 sl=2079.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:00:00 | 2060.00 | 2066.92 | 2069.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:45:00 | 2053.80 | 2057.73 | 2064.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 2055.70 | 2054.49 | 2061.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 2055.70 | 2054.49 | 2061.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 2053.90 | 2054.37 | 2060.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 2053.20 | 2054.37 | 2060.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 12:15:00 | 1957.00 | 1983.75 | 2005.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 12:15:00 | 1951.11 | 1983.75 | 2005.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 2000.40 | 1978.33 | 1994.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 2000.40 | 1978.33 | 1994.74 | SL hit (close>ema200) qty=0.50 sl=1978.33 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 2000.40 | 1978.33 | 1994.74 | SL hit (close>ema200) qty=0.50 sl=1978.33 alert=retest2 |

### Cycle 17 — BUY (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 11:15:00 | 2012.00 | 1997.09 | 1995.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 2014.20 | 2000.51 | 1997.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 1997.50 | 2008.54 | 2002.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 1997.50 | 2008.54 | 2002.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1997.50 | 2008.54 | 2002.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 1999.60 | 2008.54 | 2002.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1995.80 | 2005.99 | 2002.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 1996.90 | 2005.99 | 2002.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 1985.50 | 1999.73 | 1999.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 13:15:00 | 1980.30 | 1995.85 | 1998.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 1998.10 | 1992.33 | 1995.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 11:15:00 | 1998.10 | 1992.33 | 1995.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1998.10 | 1992.33 | 1995.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 1992.40 | 1992.33 | 1995.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1988.70 | 1991.60 | 1994.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 1988.00 | 1992.11 | 1993.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:00:00 | 1987.40 | 1991.17 | 1992.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 15:15:00 | 2000.00 | 1993.92 | 1993.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 15:15:00 | 2000.00 | 1993.92 | 1993.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 2000.00 | 1993.92 | 1993.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 2059.20 | 2006.97 | 1999.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 2052.40 | 2055.68 | 2034.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 12:15:00 | 2040.50 | 2049.92 | 2037.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 2040.50 | 2049.92 | 2037.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 2038.40 | 2049.92 | 2037.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 2032.30 | 2046.40 | 2036.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 2032.30 | 2046.40 | 2036.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 2000.00 | 2037.12 | 2033.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 2000.00 | 2037.12 | 2033.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 1989.00 | 2027.49 | 2029.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 1972.00 | 2016.39 | 2024.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 1959.60 | 1959.39 | 1979.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 1959.60 | 1959.39 | 1979.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1974.90 | 1960.20 | 1974.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1974.90 | 1960.20 | 1974.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1961.80 | 1960.52 | 1973.53 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 2035.00 | 1988.52 | 1982.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 12:15:00 | 2050.20 | 2016.46 | 1998.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 1956.30 | 2012.08 | 2003.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 1956.30 | 2012.08 | 2003.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1956.30 | 2012.08 | 2003.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 1956.30 | 2012.08 | 2003.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1940.00 | 1997.66 | 1997.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 1938.50 | 1997.66 | 1997.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1930.30 | 1984.19 | 1991.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 09:15:00 | 1897.10 | 1939.59 | 1948.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 1886.30 | 1874.86 | 1891.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:15:00 | 1886.90 | 1874.86 | 1891.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1910.20 | 1881.93 | 1892.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1910.20 | 1881.93 | 1892.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1912.50 | 1888.04 | 1894.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 1911.00 | 1888.04 | 1894.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 1911.10 | 1899.90 | 1899.14 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 1897.40 | 1900.53 | 1900.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 14:15:00 | 1894.00 | 1899.01 | 1899.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 15:15:00 | 1904.00 | 1900.01 | 1900.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 15:15:00 | 1904.00 | 1900.01 | 1900.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1904.00 | 1900.01 | 1900.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1883.50 | 1900.01 | 1900.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 1925.00 | 1867.14 | 1865.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1925.00 | 1867.14 | 1865.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 1928.70 | 1879.45 | 1871.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1892.10 | 1900.77 | 1887.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 1892.10 | 1900.77 | 1887.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1901.70 | 1900.96 | 1888.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 1906.90 | 1900.96 | 1888.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:00:00 | 1903.50 | 1901.18 | 1891.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:30:00 | 1905.50 | 1901.50 | 1892.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1874.70 | 1896.06 | 1891.86 | SL hit (close<static) qty=1.00 sl=1888.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1874.70 | 1896.06 | 1891.86 | SL hit (close<static) qty=1.00 sl=1888.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1874.70 | 1896.06 | 1891.86 | SL hit (close<static) qty=1.00 sl=1888.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:30:00 | 1906.70 | 1893.89 | 1891.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 1879.20 | 1890.95 | 1890.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 1879.20 | 1890.95 | 1890.45 | SL hit (close<static) qty=1.00 sl=1888.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 1879.20 | 1890.95 | 1890.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 1881.00 | 1888.96 | 1889.59 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 1958.00 | 1901.32 | 1895.01 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 1893.20 | 1906.42 | 1907.47 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 1913.50 | 1907.67 | 1907.48 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 1905.20 | 1907.18 | 1907.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 1901.90 | 1906.12 | 1906.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 13:15:00 | 1907.30 | 1906.36 | 1906.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 13:15:00 | 1907.30 | 1906.36 | 1906.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1907.30 | 1906.36 | 1906.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 1920.50 | 1906.36 | 1906.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1904.90 | 1906.07 | 1906.66 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1913.00 | 1907.45 | 1907.23 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1894.40 | 1904.84 | 1906.07 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1914.00 | 1907.98 | 1907.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 1944.20 | 1915.22 | 1910.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 1967.40 | 1967.55 | 1952.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 13:00:00 | 1967.40 | 1967.55 | 1952.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1955.20 | 1975.95 | 1965.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:30:00 | 1959.40 | 1975.95 | 1965.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1949.60 | 1970.68 | 1964.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:45:00 | 1949.00 | 1970.68 | 1964.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 14:15:00 | 1915.00 | 1955.90 | 1958.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 1905.90 | 1931.80 | 1940.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 1898.00 | 1893.58 | 1907.03 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1860.10 | 1893.58 | 1907.03 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1990.00 | 1889.37 | 1891.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1990.00 | 1889.37 | 1891.92 | SL hit (close>ema400) qty=1.00 sl=1891.92 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-11-11 09:45:00 | 1997.40 | 1889.37 | 1891.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 2004.40 | 1912.38 | 1902.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 2034.00 | 1936.70 | 1914.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 12:15:00 | 2007.80 | 2011.21 | 1976.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:45:00 | 2008.80 | 2011.21 | 1976.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1990.10 | 2012.89 | 1988.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 1987.20 | 2012.89 | 1988.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1978.80 | 2006.07 | 1988.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1978.80 | 2006.07 | 1988.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1978.20 | 2000.50 | 1987.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 1975.40 | 2000.50 | 1987.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1980.90 | 1996.58 | 1986.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 1977.50 | 1996.58 | 1986.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1947.80 | 1974.94 | 1978.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1943.50 | 1959.89 | 1969.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 1944.80 | 1943.18 | 1953.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 15:00:00 | 1944.80 | 1943.18 | 1953.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1906.20 | 1936.07 | 1948.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:15:00 | 1905.20 | 1936.07 | 1948.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:00:00 | 1893.10 | 1927.48 | 1943.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:15:00 | 1903.20 | 1918.84 | 1936.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1994.00 | 1901.28 | 1896.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1994.00 | 1901.28 | 1896.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1994.00 | 1901.28 | 1896.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 09:15:00 | 1994.00 | 1901.28 | 1896.68 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 15:15:00 | 1907.00 | 1918.83 | 1920.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 09:15:00 | 1898.50 | 1914.77 | 1918.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 15:15:00 | 1907.80 | 1902.14 | 1908.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:15:00 | 1891.00 | 1902.14 | 1908.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1891.00 | 1899.91 | 1907.14 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 1951.10 | 1915.59 | 1913.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 12:15:00 | 1974.60 | 1927.39 | 1918.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1936.50 | 1941.25 | 1929.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 1936.50 | 1941.25 | 1929.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1936.50 | 1941.25 | 1929.48 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1918.10 | 1931.98 | 1932.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1898.30 | 1916.61 | 1923.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 1916.60 | 1915.51 | 1921.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 14:00:00 | 1916.60 | 1915.51 | 1921.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1917.00 | 1915.80 | 1921.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 1917.00 | 1915.80 | 1921.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1908.90 | 1914.13 | 1919.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:15:00 | 1915.40 | 1914.13 | 1919.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1916.00 | 1914.51 | 1919.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:30:00 | 1903.20 | 1913.91 | 1918.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 1906.60 | 1912.81 | 1917.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 1901.10 | 1906.47 | 1912.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1918.80 | 1891.10 | 1888.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1918.80 | 1891.10 | 1888.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1918.80 | 1891.10 | 1888.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1918.80 | 1891.10 | 1888.83 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 1887.10 | 1888.15 | 1888.15 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 14:15:00 | 1891.50 | 1888.82 | 1888.46 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1884.00 | 1887.86 | 1888.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 1877.60 | 1885.81 | 1887.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 13:15:00 | 1886.40 | 1881.67 | 1884.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 13:15:00 | 1886.40 | 1881.67 | 1884.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1886.40 | 1881.67 | 1884.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 1886.40 | 1881.67 | 1884.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1882.80 | 1881.89 | 1884.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:30:00 | 1878.80 | 1881.75 | 1883.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 1893.30 | 1884.42 | 1884.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 1893.30 | 1884.42 | 1884.34 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1871.20 | 1882.49 | 1883.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 1861.50 | 1873.37 | 1877.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 1852.00 | 1851.79 | 1859.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 09:15:00 | 1836.50 | 1851.79 | 1859.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1827.00 | 1846.83 | 1856.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:00:00 | 1818.20 | 1835.19 | 1847.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:15:00 | 1727.29 | 1741.80 | 1758.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 14:15:00 | 1758.70 | 1710.44 | 1721.41 | SL hit (close>ema200) qty=0.50 sl=1710.44 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1762.10 | 1733.60 | 1730.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 1770.10 | 1753.59 | 1744.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 1748.10 | 1756.11 | 1748.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 1748.10 | 1756.11 | 1748.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1748.10 | 1756.11 | 1748.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 1748.10 | 1756.11 | 1748.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1734.40 | 1751.77 | 1747.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 1733.20 | 1751.77 | 1747.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1724.50 | 1746.32 | 1745.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 1726.90 | 1746.32 | 1745.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 1724.30 | 1741.91 | 1743.35 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 1810.00 | 1750.60 | 1746.47 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 1775.80 | 1780.90 | 1781.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 1756.60 | 1774.36 | 1777.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 12:15:00 | 1648.00 | 1643.41 | 1666.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 13:00:00 | 1648.00 | 1643.41 | 1666.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1655.00 | 1647.30 | 1662.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1649.30 | 1647.30 | 1662.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1646.10 | 1647.06 | 1660.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 1638.40 | 1644.50 | 1658.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:00:00 | 1635.00 | 1640.06 | 1653.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 11:15:00 | 1556.48 | 1609.45 | 1631.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1591.00 | 1587.03 | 1610.44 | SL hit (close>ema200) qty=0.50 sl=1587.03 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 1553.25 | 1583.23 | 1595.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 1511.40 | 1511.12 | 1535.95 | SL hit (close>ema200) qty=0.50 sl=1511.12 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1562.50 | 1501.20 | 1499.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 1587.50 | 1546.59 | 1523.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1551.30 | 1556.65 | 1535.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:45:00 | 1552.50 | 1556.65 | 1535.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1527.10 | 1547.11 | 1535.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 1530.10 | 1547.11 | 1535.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1522.90 | 1542.27 | 1534.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 1522.90 | 1542.27 | 1534.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 1480.00 | 1525.69 | 1528.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1433.50 | 1495.68 | 1510.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1440.10 | 1421.97 | 1454.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:45:00 | 1441.10 | 1421.97 | 1454.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 1454.30 | 1434.53 | 1452.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 1452.80 | 1434.53 | 1452.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 1464.40 | 1440.50 | 1453.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:45:00 | 1463.80 | 1440.50 | 1453.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 1461.00 | 1444.60 | 1454.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 1480.80 | 1444.60 | 1454.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 1515.00 | 1468.03 | 1463.71 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1425.00 | 1465.34 | 1469.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 11:15:00 | 1398.70 | 1452.01 | 1462.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 1302.50 | 1299.82 | 1329.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:30:00 | 1300.70 | 1299.82 | 1329.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1319.20 | 1304.01 | 1313.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 1319.20 | 1304.01 | 1313.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1322.70 | 1307.75 | 1314.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 1322.70 | 1307.75 | 1314.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1321.00 | 1310.40 | 1314.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 1337.50 | 1310.40 | 1314.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1317.00 | 1309.93 | 1312.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 1317.00 | 1309.93 | 1312.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1307.60 | 1309.47 | 1312.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 1299.60 | 1309.77 | 1312.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1284.10 | 1274.57 | 1273.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 14:15:00 | 1284.10 | 1274.57 | 1273.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 1295.00 | 1281.82 | 1277.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 15:15:00 | 1287.00 | 1289.53 | 1283.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 15:15:00 | 1287.00 | 1289.53 | 1283.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1287.00 | 1289.53 | 1283.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 1280.70 | 1287.77 | 1283.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1282.90 | 1286.79 | 1283.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 1285.00 | 1286.79 | 1283.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1272.30 | 1283.89 | 1282.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 1271.70 | 1283.89 | 1282.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 1264.40 | 1280.00 | 1280.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 14:15:00 | 1261.00 | 1273.89 | 1277.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 1252.90 | 1251.82 | 1262.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 13:30:00 | 1254.10 | 1251.82 | 1262.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1251.10 | 1251.67 | 1261.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:15:00 | 1266.50 | 1251.67 | 1261.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1266.50 | 1254.64 | 1262.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:30:00 | 1286.60 | 1263.01 | 1265.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 10:15:00 | 1291.00 | 1268.61 | 1267.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 14:15:00 | 1305.00 | 1283.36 | 1275.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 10:15:00 | 1362.00 | 1368.06 | 1337.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 1362.00 | 1368.06 | 1337.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1341.00 | 1359.51 | 1340.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:45:00 | 1345.10 | 1359.51 | 1340.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1356.30 | 1358.87 | 1342.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 1370.00 | 1358.87 | 1342.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 1367.00 | 1363.93 | 1351.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1335.30 | 1355.51 | 1351.51 | SL hit (close<static) qty=1.00 sl=1341.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1335.30 | 1355.51 | 1351.51 | SL hit (close<static) qty=1.00 sl=1341.10 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 1325.00 | 1346.41 | 1347.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 14:15:00 | 1319.70 | 1336.06 | 1342.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 09:15:00 | 1321.90 | 1304.55 | 1317.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 1321.90 | 1304.55 | 1317.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1321.90 | 1304.55 | 1317.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 1321.90 | 1304.55 | 1317.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 1333.00 | 1310.24 | 1319.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:30:00 | 1325.40 | 1310.24 | 1319.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 1334.60 | 1315.11 | 1320.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 1348.00 | 1315.11 | 1320.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1333.80 | 1321.70 | 1322.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1333.80 | 1321.70 | 1322.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1321.40 | 1321.64 | 1322.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 1295.80 | 1321.64 | 1322.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 1335.30 | 1318.92 | 1320.26 | SL hit (close>static) qty=1.00 sl=1333.60 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 1333.80 | 1321.90 | 1321.49 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 1309.20 | 1319.26 | 1320.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 1306.20 | 1316.65 | 1319.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 15:15:00 | 1256.80 | 1256.59 | 1273.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:15:00 | 1252.60 | 1256.59 | 1273.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1261.90 | 1248.68 | 1258.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 1275.10 | 1248.68 | 1258.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1281.00 | 1255.14 | 1260.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 1281.00 | 1255.14 | 1260.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 1281.30 | 1260.37 | 1262.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 1281.30 | 1260.37 | 1262.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1286.00 | 1265.50 | 1264.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1288.10 | 1270.02 | 1266.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 1272.20 | 1273.06 | 1269.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:15:00 | 1257.30 | 1273.06 | 1269.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1257.90 | 1270.03 | 1268.04 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1247.30 | 1265.48 | 1266.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1241.10 | 1257.28 | 1261.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 1267.00 | 1252.85 | 1257.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 1267.00 | 1252.85 | 1257.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1267.00 | 1252.85 | 1257.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 1267.00 | 1252.85 | 1257.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1276.90 | 1257.66 | 1259.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:00:00 | 1276.90 | 1257.66 | 1259.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1241.00 | 1255.60 | 1258.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1213.00 | 1253.12 | 1256.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 1232.80 | 1235.23 | 1236.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1296.00 | 1247.00 | 1241.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1296.00 | 1247.00 | 1241.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1296.00 | 1247.00 | 1241.36 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1233.00 | 1256.01 | 1257.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1202.10 | 1241.10 | 1250.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1251.00 | 1218.48 | 1230.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1251.00 | 1218.48 | 1230.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1251.00 | 1218.48 | 1230.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1251.00 | 1218.48 | 1230.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1252.80 | 1225.34 | 1232.65 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 1264.90 | 1240.10 | 1238.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 13:15:00 | 1270.20 | 1246.12 | 1241.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1256.30 | 1257.80 | 1248.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 1256.30 | 1257.80 | 1248.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1256.30 | 1257.80 | 1248.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 1280.10 | 1260.52 | 1250.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-07 09:15:00 | 1408.11 | 1363.45 | 1328.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 09:15:00 | 1448.00 | 1471.67 | 1471.75 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 1487.40 | 1467.88 | 1467.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 1513.70 | 1481.02 | 1476.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 12:15:00 | 1486.90 | 1487.52 | 1480.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 13:00:00 | 1486.90 | 1487.52 | 1480.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1483.80 | 1487.32 | 1482.89 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1440.00 | 1472.57 | 1476.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1425.80 | 1463.22 | 1472.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 1440.00 | 1438.60 | 1454.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:00:00 | 1440.00 | 1438.60 | 1454.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1489.00 | 1449.61 | 1456.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 1501.80 | 1449.61 | 1456.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1491.20 | 1457.93 | 1459.69 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1492.70 | 1464.88 | 1462.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1508.00 | 1473.51 | 1466.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1510.40 | 1514.84 | 1504.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 1510.40 | 1514.84 | 1504.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1510.40 | 1514.84 | 1504.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1545.70 | 1513.38 | 1507.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:30:00 | 1538.10 | 1525.51 | 1515.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 13:00:00 | 1530.10 | 1526.43 | 1516.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 15:15:00 | 1532.00 | 1524.87 | 1517.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 1532.00 | 1526.29 | 1518.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:30:00 | 1551.00 | 1534.25 | 1523.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:00:00 | 1558.10 | 1540.97 | 1528.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 13:30:00 | 1547.30 | 1544.28 | 1532.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-07 10:15:00 | 1683.11 | 1644.37 | 1605.73 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-05-07 10:15:00 | 1685.20 | 1644.37 | 1605.73 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-05-07 11:15:00 | 1691.91 | 1650.70 | 1612.12 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-27 09:15:00 | 2391.60 | 2025-05-30 09:15:00 | 2272.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-27 09:15:00 | 2391.60 | 2025-06-03 13:15:00 | 2239.10 | STOP_HIT | 0.50 | 6.38% |
| SELL | retest2 | 2025-06-13 14:45:00 | 2209.00 | 2025-06-19 10:15:00 | 2098.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 2208.60 | 2025-06-19 10:15:00 | 2098.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-16 09:45:00 | 2209.10 | 2025-06-19 10:15:00 | 2098.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 14:45:00 | 2209.00 | 2025-06-20 09:15:00 | 2088.90 | STOP_HIT | 0.50 | 5.44% |
| SELL | retest2 | 2025-06-16 09:15:00 | 2208.60 | 2025-06-20 09:15:00 | 2088.90 | STOP_HIT | 0.50 | 5.42% |
| SELL | retest2 | 2025-06-16 09:45:00 | 2209.10 | 2025-06-20 09:15:00 | 2088.90 | STOP_HIT | 0.50 | 5.44% |
| BUY | retest2 | 2025-07-04 11:15:00 | 2260.70 | 2025-07-04 13:15:00 | 2236.30 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-04 12:30:00 | 2262.20 | 2025-07-04 13:15:00 | 2236.30 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-07-07 09:15:00 | 2267.90 | 2025-07-07 10:15:00 | 2237.30 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-07-14 09:15:00 | 2189.70 | 2025-07-14 13:15:00 | 2222.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-07-14 12:45:00 | 2210.00 | 2025-07-14 13:15:00 | 2222.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-07-15 09:30:00 | 2205.10 | 2025-07-17 09:15:00 | 2094.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 09:30:00 | 2205.10 | 2025-07-17 12:15:00 | 2131.70 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2025-08-19 09:15:00 | 1883.90 | 2025-08-19 11:15:00 | 1914.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-08-28 12:00:00 | 2101.00 | 2025-09-01 10:15:00 | 2071.50 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-08-28 13:15:00 | 2102.30 | 2025-09-01 10:15:00 | 2071.50 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-02 12:00:00 | 2059.10 | 2025-09-02 15:15:00 | 2082.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-09-02 14:15:00 | 2053.80 | 2025-09-02 15:15:00 | 2082.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-09-04 13:00:00 | 2060.00 | 2025-09-10 12:15:00 | 1957.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 14:45:00 | 2053.80 | 2025-09-10 12:15:00 | 1951.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 13:00:00 | 2060.00 | 2025-09-11 09:15:00 | 2000.40 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2025-09-04 14:45:00 | 2053.80 | 2025-09-11 09:15:00 | 2000.40 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest2 | 2025-09-17 12:30:00 | 1988.00 | 2025-09-17 15:15:00 | 2000.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-17 14:00:00 | 1987.40 | 2025-09-17 15:15:00 | 2000.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1883.50 | 2025-10-16 10:15:00 | 1925.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-10-17 11:15:00 | 1906.90 | 2025-10-20 09:15:00 | 1874.70 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-10-17 14:00:00 | 1903.50 | 2025-10-20 09:15:00 | 1874.70 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-10-17 14:30:00 | 1905.50 | 2025-10-20 09:15:00 | 1874.70 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-10-20 12:30:00 | 1906.70 | 2025-10-20 13:15:00 | 1879.20 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest1 | 2025-11-10 09:15:00 | 1860.10 | 2025-11-11 09:15:00 | 1990.00 | STOP_HIT | 1.00 | -6.98% |
| SELL | retest2 | 2025-11-18 10:15:00 | 1905.20 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2025-11-18 11:00:00 | 1893.10 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2025-11-18 13:15:00 | 1903.20 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2025-12-04 12:30:00 | 1903.20 | 2025-12-09 14:15:00 | 1918.80 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-12-04 13:30:00 | 1906.60 | 2025-12-09 14:15:00 | 1918.80 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-12-05 11:15:00 | 1901.10 | 2025-12-09 14:15:00 | 1918.80 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-12 10:30:00 | 1878.80 | 2025-12-12 13:15:00 | 1893.30 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-18 13:00:00 | 1818.20 | 2025-12-30 09:15:00 | 1727.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-18 13:00:00 | 1818.20 | 2025-12-31 14:15:00 | 1758.70 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2026-01-20 10:30:00 | 1638.40 | 2026-01-21 11:15:00 | 1556.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 10:30:00 | 1638.40 | 2026-01-22 09:15:00 | 1591.00 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-01-20 13:00:00 | 1635.00 | 2026-01-27 09:15:00 | 1553.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 13:00:00 | 1635.00 | 2026-01-28 14:15:00 | 1511.40 | STOP_HIT | 0.50 | 7.56% |
| SELL | retest2 | 2026-02-19 09:15:00 | 1299.60 | 2026-02-24 14:15:00 | 1284.10 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2026-03-05 15:15:00 | 1370.00 | 2026-03-09 09:15:00 | 1335.30 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-03-06 13:15:00 | 1367.00 | 2026-03-09 09:15:00 | 1335.30 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2026-03-12 09:15:00 | 1295.80 | 2026-03-12 11:15:00 | 1335.30 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1213.00 | 2026-03-25 09:15:00 | 1296.00 | STOP_HIT | 1.00 | -6.84% |
| SELL | retest2 | 2026-03-24 15:15:00 | 1232.80 | 2026-03-25 09:15:00 | 1296.00 | STOP_HIT | 1.00 | -5.13% |
| BUY | retest2 | 2026-04-02 11:15:00 | 1280.10 | 2026-04-07 09:15:00 | 1408.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1545.70 | 2026-05-07 10:15:00 | 1683.11 | TARGET_HIT | 1.00 | 8.89% |
| BUY | retest2 | 2026-05-04 11:30:00 | 1538.10 | 2026-05-07 10:15:00 | 1685.20 | TARGET_HIT | 1.00 | 9.56% |
| BUY | retest2 | 2026-05-04 13:00:00 | 1530.10 | 2026-05-07 11:15:00 | 1691.91 | TARGET_HIT | 1.00 | 10.58% |
