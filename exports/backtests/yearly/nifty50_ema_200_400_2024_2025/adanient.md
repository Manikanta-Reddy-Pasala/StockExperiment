# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2024-03-12 09:15:00 → 2026-05-08 15:15:00 (3717 bars)
- **Last close:** 2502.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 0 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 24
- **Target hits / Stop hits / Partials:** 3 / 26 / 3
- **Avg / median % per leg:** 0.12% / -1.50%
- **Sum % (uncompounded):** 3.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| SELL (all) | 30 | 6 | 20.0% | 1 | 26 | 3 | -0.54% | -16.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 6 | 20.0% | 1 | 26 | 3 | -0.54% | -16.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 8 | 25.0% | 3 | 26 | 3 | 0.12% | 3.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 15:15:00 | 2916.16 | 3042.01 | 3042.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 13:15:00 | 2906.08 | 3035.77 | 3039.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 10:15:00 | 3009.67 | 3004.91 | 3021.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-29 11:00:00 | 3009.67 | 3004.91 | 3021.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 3023.34 | 3004.34 | 3020.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:00:00 | 3023.34 | 3004.34 | 3020.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 3045.93 | 3004.75 | 3021.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:45:00 | 3046.51 | 3004.75 | 3021.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 3102.35 | 3017.94 | 3026.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:00:00 | 3102.35 | 3017.94 | 3026.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 3032.16 | 3016.84 | 3025.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 3032.16 | 3016.84 | 3025.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 3028.96 | 3016.96 | 3025.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 2995.71 | 3017.10 | 3025.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 3040.74 | 3016.56 | 3025.01 | SL hit (close>static) qty=1.00 sl=3039.09 alert=retest2 |

### Cycle 2 — BUY (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 12:15:00 | 3037.93 | 2983.10 | 2982.83 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 2914.85 | 2983.19 | 2983.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 2907.39 | 2981.87 | 2982.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 2889.07 | 2884.27 | 2927.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 12:00:00 | 2889.07 | 2884.27 | 2927.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 2930.41 | 2873.28 | 2915.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 12:00:00 | 2930.41 | 2873.28 | 2915.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 2943.11 | 2873.97 | 2915.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 2943.11 | 2873.97 | 2915.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 2493.37 | 2440.08 | 2559.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:45:00 | 2499.19 | 2440.08 | 2559.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 2259.05 | 2174.36 | 2246.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 10:00:00 | 2231.46 | 2179.51 | 2246.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 14:15:00 | 2267.38 | 2183.39 | 2247.00 | SL hit (close>static) qty=1.00 sl=2266.90 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 2371.36 | 2269.13 | 2268.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 2403.06 | 2275.61 | 2272.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 15:15:00 | 2278.29 | 2281.31 | 2275.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-07 09:15:00 | 2293.22 | 2281.31 | 2275.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2262.10 | 2281.12 | 2275.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 2309.89 | 2272.42 | 2271.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:45:00 | 2328.70 | 2272.83 | 2272.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-10 11:15:00 | 2540.88 | 2407.52 | 2361.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 2196.08 | 2430.63 | 2430.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 2172.13 | 2428.06 | 2429.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 2308.54 | 2272.40 | 2320.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 2308.54 | 2272.40 | 2320.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 2333.36 | 2273.01 | 2320.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 2334.42 | 2273.01 | 2320.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 2336.56 | 2273.64 | 2320.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:30:00 | 2317.84 | 2274.00 | 2320.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 2342.86 | 2276.18 | 2320.89 | SL hit (close>static) qty=1.00 sl=2341.21 alert=retest2 |

### Cycle 6 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 2536.66 | 2348.59 | 2348.48 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 2256.30 | 2396.43 | 2397.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2249.00 | 2394.97 | 2396.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 2272.00 | 2270.61 | 2310.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 09:45:00 | 2277.90 | 2270.61 | 2310.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2181.90 | 2117.03 | 2195.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:30:00 | 2180.00 | 2117.03 | 2195.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 2208.10 | 2117.93 | 2195.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:30:00 | 2215.30 | 2117.93 | 2195.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 2215.90 | 2118.91 | 2195.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:30:00 | 2226.60 | 2118.91 | 2195.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 2206.50 | 2122.42 | 2195.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 2205.00 | 2122.42 | 2195.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 2198.50 | 2124.13 | 2195.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 2168.60 | 2162.97 | 2202.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 2191.10 | 2162.90 | 2199.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 2234.20 | 2163.97 | 2199.66 | SL hit (close>static) qty=1.00 sl=2226.90 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 2281.60 | 2094.05 | 2093.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 2309.40 | 2096.19 | 2095.04 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-08-06 14:00:00 | 2995.71 | 2024-08-07 09:15:00 | 3040.74 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-08-12 09:15:00 | 2996.73 | 2024-08-12 10:15:00 | 3050.87 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-08-13 15:00:00 | 2997.79 | 2024-08-22 11:15:00 | 3045.44 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-08-20 11:15:00 | 2993.38 | 2024-08-22 11:15:00 | 3045.44 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-08-21 11:30:00 | 3022.42 | 2024-09-04 14:15:00 | 2871.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-22 14:00:00 | 3023.78 | 2024-09-04 14:15:00 | 2872.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-21 11:30:00 | 3022.42 | 2024-09-20 12:15:00 | 2926.05 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2024-08-22 14:00:00 | 3023.78 | 2024-09-20 12:15:00 | 2926.05 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2024-09-26 11:45:00 | 3023.97 | 2024-09-30 10:15:00 | 3087.23 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-09-26 12:15:00 | 3022.90 | 2024-09-30 10:15:00 | 3087.23 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-03-20 10:00:00 | 2231.46 | 2025-03-20 14:15:00 | 2267.38 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-03-27 10:30:00 | 2237.67 | 2025-03-27 14:15:00 | 2291.72 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-04-07 09:15:00 | 2155.75 | 2025-04-15 09:15:00 | 2313.19 | STOP_HIT | 1.00 | -7.30% |
| SELL | retest2 | 2025-04-11 10:30:00 | 2239.27 | 2025-04-15 09:15:00 | 2313.19 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-04-11 13:15:00 | 2242.95 | 2025-04-15 09:15:00 | 2313.19 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-05-12 09:15:00 | 2309.89 | 2025-06-10 11:15:00 | 2540.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:45:00 | 2328.70 | 2025-06-10 11:15:00 | 2561.57 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-11 12:30:00 | 2317.84 | 2025-09-12 09:15:00 | 2342.86 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-12 11:30:00 | 2329.67 | 2025-09-18 10:15:00 | 2351.97 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-09-17 11:15:00 | 2331.90 | 2025-09-18 10:15:00 | 2351.97 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-17 12:00:00 | 2330.35 | 2025-09-18 10:15:00 | 2351.97 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-02-13 09:15:00 | 2168.60 | 2026-02-17 12:15:00 | 2234.20 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-02-17 10:30:00 | 2191.10 | 2026-02-17 12:15:00 | 2234.20 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-02-18 10:30:00 | 2190.50 | 2026-02-25 09:15:00 | 2200.90 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-02-19 09:45:00 | 2195.30 | 2026-02-25 09:15:00 | 2200.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-02-23 11:45:00 | 2180.00 | 2026-02-25 09:15:00 | 2200.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-02-23 13:15:00 | 2180.60 | 2026-02-25 14:15:00 | 2230.70 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-02-24 12:00:00 | 2178.10 | 2026-02-25 14:15:00 | 2230.70 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-02-27 15:00:00 | 2160.20 | 2026-03-04 09:15:00 | 2052.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 2160.20 | 2026-03-09 09:15:00 | 1944.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-08 12:45:00 | 2030.10 | 2026-04-10 10:15:00 | 2064.70 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-04-09 09:30:00 | 2021.00 | 2026-04-10 10:15:00 | 2064.70 | STOP_HIT | 1.00 | -2.16% |
