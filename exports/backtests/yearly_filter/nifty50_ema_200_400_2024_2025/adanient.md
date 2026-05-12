# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2502.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 30
- **Target hits / Stop hits / Partials:** 3 / 32 / 3
- **Avg / median % per leg:** -0.29% / -1.70%
- **Sum % (uncompounded):** -11.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.74% | 13.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.74% | 13.7% |
| SELL (all) | 33 | 6 | 18.2% | 1 | 29 | 3 | -0.75% | -24.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 6 | 18.2% | 1 | 29 | 3 | -0.75% | -24.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 38 | 8 | 21.1% | 3 | 32 | 3 | -0.29% | -11.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 14:15:00 | 3282.29 | 2981.68 | 2980.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 3306.82 | 3041.55 | 3013.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3019.99 | 3079.62 | 3034.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 3019.99 | 3079.62 | 3034.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3019.99 | 3079.62 | 3034.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 3019.99 | 3079.62 | 3034.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 2827.21 | 3077.11 | 3033.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 2827.21 | 3077.11 | 3033.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 2884.02 | 3072.40 | 3031.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 2884.02 | 3072.40 | 3031.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 2998.62 | 3060.41 | 3027.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 3056.16 | 3060.41 | 3027.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:30:00 | 3042.24 | 3087.71 | 3065.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 2982.72 | 3084.83 | 3064.89 | SL hit (close<static) qty=1.00 sl=2995.71 alert=retest2 |

### Cycle 2 — SELL (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 15:15:00 | 2995.71 | 3049.76 | 3049.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 2970.99 | 3048.98 | 3049.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 10:15:00 | 3009.67 | 3005.01 | 3024.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-29 11:00:00 | 3009.67 | 3005.01 | 3024.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 3023.34 | 3004.42 | 3023.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:00:00 | 3023.34 | 3004.42 | 3023.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 3045.93 | 3004.84 | 3023.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:45:00 | 3046.51 | 3004.84 | 3023.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 3040.30 | 3005.19 | 3023.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:45:00 | 3043.02 | 3005.19 | 3023.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 3102.35 | 3018.01 | 3028.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:00:00 | 3102.35 | 3018.01 | 3028.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 3032.16 | 3016.90 | 3027.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 3032.16 | 3016.90 | 3027.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 3028.96 | 3017.02 | 3027.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 2995.71 | 3017.16 | 3027.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 3040.74 | 3016.61 | 3027.25 | SL hit (close>static) qty=1.00 sl=3039.09 alert=retest2 |

### Cycle 3 — BUY (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 13:15:00 | 3033.32 | 2983.60 | 2983.54 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 2921.06 | 2983.88 | 2983.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 10:15:00 | 2877.19 | 2977.80 | 2980.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 2889.07 | 2884.27 | 2927.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 12:00:00 | 2889.07 | 2884.27 | 2927.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 2930.41 | 2873.28 | 2915.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 12:00:00 | 2930.41 | 2873.28 | 2915.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 2943.11 | 2873.97 | 2916.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 2943.11 | 2873.97 | 2916.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 2493.37 | 2440.08 | 2559.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:45:00 | 2499.19 | 2440.08 | 2559.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 2259.05 | 2174.36 | 2246.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 10:00:00 | 2231.46 | 2179.51 | 2246.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 14:15:00 | 2267.38 | 2183.39 | 2247.01 | SL hit (close>static) qty=1.00 sl=2266.90 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 2371.36 | 2269.13 | 2268.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 2403.06 | 2275.61 | 2272.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 15:15:00 | 2278.29 | 2281.31 | 2275.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-07 09:15:00 | 2293.22 | 2281.31 | 2275.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2262.10 | 2281.12 | 2275.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 2309.89 | 2272.42 | 2271.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:45:00 | 2328.70 | 2272.83 | 2272.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-10 11:15:00 | 2540.88 | 2407.52 | 2361.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-08-07 09:15:00)

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

### Cycle 7 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 2536.66 | 2348.59 | 2348.48 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-11-27 14:15:00)

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

### Cycle 9 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 2281.60 | 2094.05 | 2093.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 2309.40 | 2096.19 | 2095.04 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-15 12:00:00 | 2969.82 | 2024-05-21 11:15:00 | 3051.89 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-05-15 13:30:00 | 2966.33 | 2024-05-21 11:15:00 | 3051.89 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-05-18 09:45:00 | 2964.88 | 2024-05-21 11:15:00 | 3051.89 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-06-06 09:15:00 | 3056.16 | 2024-07-08 11:15:00 | 2982.72 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-07-05 14:30:00 | 3042.24 | 2024-07-08 11:15:00 | 2982.72 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-07-09 09:15:00 | 3052.91 | 2024-07-10 13:15:00 | 2993.62 | STOP_HIT | 1.00 | -1.94% |
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
