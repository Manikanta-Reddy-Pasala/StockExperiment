# ASIANPAINT (ASIANPAINT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 2519.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 5 |
| ALERT3 | 11 |
| PENDING | 43 |
| PENDING_CANCEL | 12 |
| ENTRY1 | 7 |
| ENTRY2 | 24 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 30
- **Target hits / Stop hits / Partials:** 0 / 31 / 0
- **Avg / median % per leg:** -2.07% / -1.47%
- **Sum % (uncompounded):** -64.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.36% | -2.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.36% | -2.7% |
| SELL (all) | 29 | 1 | 3.4% | 0 | 29 | 0 | -2.11% | -61.3% |
| SELL @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.97% | -20.8% |
| SELL @ 3rd Alert (retest2) | 22 | 1 | 4.5% | 0 | 22 | 0 | -1.84% | -40.5% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.97% | -20.8% |
| retest2 (combined) | 24 | 1 | 4.2% | 0 | 24 | 0 | -1.80% | -43.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 12:15:00 | 3231.00 | 3159.07 | 3158.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 13:15:00 | 3242.95 | 3163.42 | 3161.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 09:15:00 | 3291.85 | 3297.58 | 3247.29 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 3249.35 | 3294.04 | 3250.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 3249.35 | 3294.04 | 3250.44 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-01-16 09:15:00 | 3298.00 | 3291.12 | 3251.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 10:15:00 | 3300.40 | 3291.21 | 3252.11 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-17 14:15:00 | 3244.50 | 3290.27 | 3253.73 | SL hit qty=1.00 sl=3244.50 alert=retest2 |

### Cycle 2 — SELL (started 2024-01-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 15:15:00 | 2949.10 | 3222.40 | 3223.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 09:15:00 | 2944.45 | 3171.65 | 3196.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 13:15:00 | 2881.75 | 2880.13 | 2951.27 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-04-03 14:15:00 | 2865.30 | 2879.98 | 2950.85 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-03 15:15:00 | 2871.70 | 2879.90 | 2950.45 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-04 09:15:00 | 2852.00 | 2879.62 | 2949.96 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:15:00 | 2866.00 | 2879.49 | 2949.54 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-09 13:15:00 | 2848.05 | 2882.34 | 2943.18 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-09 14:15:00 | 2855.65 | 2882.08 | 2942.74 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-12 09:15:00 | 2852.40 | 2881.79 | 2939.92 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 10:15:00 | 2862.55 | 2881.60 | 2939.54 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-24 09:15:00 | 2866.75 | 2865.05 | 2917.87 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-24 10:15:00 | 2875.00 | 2865.15 | 2917.65 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-24 13:15:00 | 2866.50 | 2865.27 | 2916.93 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-24 14:15:00 | 2870.65 | 2865.32 | 2916.70 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-25 09:15:00 | 2841.20 | 2865.13 | 2916.09 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:15:00 | 2840.30 | 2864.88 | 2915.71 | SELL ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 2906.00 | 2864.69 | 2910.52 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-04-30 10:15:00 | 2910.52 | 2864.69 | 2910.52 | SL hit qty=1.00 sl=2910.52 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-30 10:15:00 | 2910.52 | 2864.69 | 2910.52 | SL hit qty=1.00 sl=2910.52 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-30 10:15:00 | 2910.52 | 2864.69 | 2910.52 | SL hit qty=1.00 sl=2910.52 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-30 10:15:00 | 2910.52 | 2864.69 | 2910.52 | SL hit qty=1.00 sl=2910.52 alert=retest1 |
| Cross detected — sustain check pending | 2024-04-30 13:15:00 | 2889.95 | 2865.57 | 2910.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 14:15:00 | 2872.65 | 2865.64 | 2910.09 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 2911.90 | 2866.28 | 2909.97 | SL hit qty=1.00 sl=2911.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-08 09:15:00 | 2879.85 | 2882.63 | 2913.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 10:15:00 | 2836.30 | 2882.16 | 2912.70 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 2911.90 | 2857.98 | 2889.13 | SL hit qty=1.00 sl=2911.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-23 10:15:00 | 2891.00 | 2858.31 | 2889.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 11:15:00 | 2885.25 | 2858.58 | 2889.12 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-24 09:15:00 | 2883.25 | 2860.37 | 2889.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:15:00 | 2888.00 | 2860.64 | 2889.26 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 2887.15 | 2860.91 | 2889.25 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-05-24 12:15:00 | 2873.40 | 2861.03 | 2889.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:15:00 | 2879.05 | 2861.21 | 2889.12 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 2894.00 | 2862.17 | 2888.78 | SL hit qty=1.00 sl=2894.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-27 14:15:00 | 2872.85 | 2862.49 | 2888.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 15:15:00 | 2872.30 | 2862.59 | 2888.60 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-28 11:15:00 | 2894.00 | 2863.13 | 2888.49 | SL hit qty=1.00 sl=2894.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 2911.90 | 2864.23 | 2888.66 | SL hit qty=1.00 sl=2911.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 2911.90 | 2864.23 | 2888.66 | SL hit qty=1.00 sl=2911.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-30 14:15:00 | 2879.55 | 2868.96 | 2889.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-30 15:15:00 | 2883.70 | 2869.11 | 2889.44 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-31 10:15:00 | 2873.45 | 2869.30 | 2889.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-31 11:15:00 | 2893.20 | 2869.54 | 2889.35 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-31 13:15:00 | 2878.50 | 2869.90 | 2889.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-31 14:15:00 | 2885.45 | 2870.06 | 2889.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-31 15:15:00 | 2881.20 | 2870.17 | 2889.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-03 09:15:00 | 2890.05 | 2870.37 | 2889.28 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2024-06-03 12:15:00 | 2873.95 | 2870.67 | 2889.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:15:00 | 2872.00 | 2870.68 | 2889.07 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 2894.00 | 2870.83 | 2888.87 | SL hit qty=1.00 sl=2894.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-04 10:15:00 | 2842.20 | 2870.54 | 2888.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:15:00 | 2851.95 | 2870.36 | 2888.45 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 2894.00 | 2870.36 | 2888.45 | SL hit qty=1.00 sl=2894.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 2992.15 | 2871.13 | 2888.39 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-25 13:15:00 | 2869.35 | 2893.57 | 2896.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 14:15:00 | 2857.50 | 2893.21 | 2896.32 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| CROSSOVER_SKIP | 2024-07-03 14:15:00 | 2924.45 | 2898.37 | 2898.29 | HTF filter: close below htf_sma |
| Stop hit — per-position SL triggered | 2024-07-10 14:15:00 | 2995.35 | 2906.29 | 2902.63 | SL hit qty=1.00 sl=2995.35 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 14:15:00 | 2987.40 | 3120.09 | 3120.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 2959.95 | 3114.78 | 3117.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 2301.20 | 2298.95 | 2414.55 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-02-04 09:15:00 | 2274.20 | 2299.05 | 2409.53 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-04 10:15:00 | 2293.40 | 2299.00 | 2408.95 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 13:15:00 | 2395.70 | 2300.28 | 2407.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 2395.70 | 2300.28 | 2407.95 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-05 09:15:00 | 2262.65 | 2300.99 | 2406.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 10:15:00 | 2267.40 | 2300.66 | 2406.02 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-24 15:15:00 | 2317.95 | 2257.68 | 2299.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-25 09:15:00 | 2326.85 | 2258.36 | 2299.73 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-03-26 11:15:00 | 2307.10 | 2263.93 | 2300.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 12:15:00 | 2311.30 | 2264.40 | 2300.81 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-26 15:15:00 | 2317.25 | 2266.07 | 2301.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 2313.30 | 2266.54 | 2301.17 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-04-01 13:15:00 | 2310.20 | 2277.76 | 2304.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 14:15:00 | 2316.35 | 2278.14 | 2304.11 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 2316.00 | 2278.52 | 2304.17 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-02 09:15:00 | 2301.55 | 2278.75 | 2304.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 10:15:00 | 2304.00 | 2279.00 | 2304.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 2318.55 | 2280.65 | 2304.25 | SL hit qty=1.00 sl=2318.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-04 09:15:00 | 2308.85 | 2284.55 | 2305.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-04 10:15:00 | 2321.40 | 2284.92 | 2305.51 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-04-08 13:15:00 | 2417.10 | 2296.21 | 2309.66 | SL hit qty=1.00 sl=2417.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 13:15:00 | 2417.10 | 2296.21 | 2309.66 | SL hit qty=1.00 sl=2417.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 13:15:00 | 2417.10 | 2296.21 | 2309.66 | SL hit qty=1.00 sl=2417.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 13:15:00 | 2417.10 | 2296.21 | 2309.66 | SL hit qty=1.00 sl=2417.10 alert=retest2 |
| CROSSOVER_SKIP | 2025-04-16 11:15:00 | 2442.00 | 2322.15 | 2321.81 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-05-08 14:15:00 | 2305.70 | 2386.18 | 2363.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 15:15:00 | 2300.90 | 2385.33 | 2363.23 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 2318.55 | 2384.53 | 2362.93 | SL hit qty=1.00 sl=2318.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-09 11:15:00 | 2310.80 | 2383.11 | 2362.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 12:15:00 | 2295.40 | 2382.24 | 2362.10 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 2318.55 | 2379.65 | 2361.19 | SL hit qty=1.00 sl=2318.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-14 09:15:00 | 2292.90 | 2374.24 | 2359.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 10:15:00 | 2294.80 | 2373.45 | 2359.34 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 2318.55 | 2365.76 | 2356.08 | SL hit qty=1.00 sl=2318.55 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2355.70 | 2363.17 | 2355.22 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-19 10:15:00 | 2347.00 | 2363.01 | 2355.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 11:15:00 | 2341.30 | 2362.80 | 2355.11 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 2319.50 | 2348.46 | 2348.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 2319.50 | 2348.46 | 2348.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 2309.10 | 2348.07 | 2348.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 2348.00 | 2346.84 | 2347.72 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 2348.00 | 2346.84 | 2347.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 2348.00 | 2346.84 | 2347.72 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-27 13:15:00 | 2329.60 | 2346.62 | 2347.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 14:15:00 | 2326.10 | 2346.41 | 2347.49 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 2351.60 | 2281.99 | 2301.03 | SL hit qty=1.00 sl=2351.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-30 11:15:00 | 2335.00 | 2285.21 | 2302.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 2328.00 | 2285.63 | 2302.32 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 2351.60 | 2288.03 | 2303.20 | SL hit qty=1.00 sl=2351.60 alert=retest2 |
| CROSSOVER_SKIP | 2025-07-04 15:15:00 | 2424.20 | 2316.55 | 2316.38 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-07-25 11:15:00 | 2332.00 | 2368.87 | 2351.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 12:15:00 | 2332.60 | 2368.51 | 2351.34 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 2351.60 | 2367.24 | 2351.12 | SL hit qty=1.00 sl=2351.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-01 09:15:00 | 2322.70 | 2465.84 | 2456.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 2327.40 | 2464.46 | 2455.66 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 2351.60 | 2456.86 | 2452.06 | SL hit qty=1.00 sl=2351.60 alert=retest2 |

### Cycle 5 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 2346.40 | 2446.58 | 2447.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 2336.00 | 2435.12 | 2441.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 2408.40 | 2404.31 | 2422.54 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 2514.70 | 2405.47 | 2422.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2514.70 | 2405.47 | 2422.94 | EMA400 retest candle locked |

### Cycle 6 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 2518.80 | 2438.26 | 2437.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 2524.60 | 2443.70 | 2440.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 14:15:00 | 2794.50 | 2797.80 | 2683.55 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 14:15:00 | 2756.40 | 2805.29 | 2754.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2756.40 | 2805.29 | 2754.27 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-19 09:15:00 | 2765.60 | 2804.41 | 2754.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 2779.20 | 2804.15 | 2754.46 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-19 15:15:00 | 2750.90 | 2802.17 | 2754.69 | SL hit qty=1.00 sl=2750.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-23 10:15:00 | 2782.00 | 2780.07 | 2748.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-23 11:15:00 | 2755.60 | 2779.82 | 2748.16 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-23 12:15:00 | 2764.10 | 2779.67 | 2748.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-23 13:15:00 | 2729.00 | 2779.16 | 2748.15 | ENTRY2 sustain failed after 60m |

### Cycle 7 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 2431.50 | 2720.59 | 2721.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 2407.70 | 2706.31 | 2713.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2280.80 | 2264.25 | 2369.63 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 10:15:00 | 2269.10 | 2264.29 | 2369.13 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 11:15:00 | 2255.70 | 2264.21 | 2368.56 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 2271.40 | 2264.81 | 2366.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:15:00 | 2272.20 | 2264.89 | 2365.82 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 2272.10 | 2265.10 | 2364.92 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 13:15:00 | 2265.10 | 2265.10 | 2364.43 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 2364.40 | 2266.14 | 2363.47 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2363.47 | 2266.14 | 2363.47 | SL hit qty=1.00 sl=2363.47 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2363.47 | 2266.14 | 2363.47 | SL hit qty=1.00 sl=2363.47 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2363.47 | 2266.14 | 2363.47 | SL hit qty=1.00 sl=2363.47 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-16 10:15:00 | 3300.40 | 2024-01-17 14:15:00 | 3244.50 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest1 | 2024-04-04 10:15:00 | 2866.00 | 2024-04-30 10:15:00 | 2910.52 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest1 | 2024-04-09 14:15:00 | 2855.65 | 2024-04-30 10:15:00 | 2910.52 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest1 | 2024-04-12 10:15:00 | 2862.55 | 2024-04-30 10:15:00 | 2910.52 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest1 | 2024-04-25 10:15:00 | 2840.30 | 2024-04-30 10:15:00 | 2910.52 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-04-30 14:15:00 | 2872.65 | 2024-05-02 09:15:00 | 2911.90 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-05-08 10:15:00 | 2836.30 | 2024-05-23 09:15:00 | 2911.90 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-05-23 11:15:00 | 2885.25 | 2024-05-27 12:15:00 | 2894.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-05-24 10:15:00 | 2888.00 | 2024-05-28 11:15:00 | 2894.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-05-24 13:15:00 | 2879.05 | 2024-05-28 14:15:00 | 2911.90 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-05-27 15:15:00 | 2872.30 | 2024-05-28 14:15:00 | 2911.90 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-06-03 13:15:00 | 2872.00 | 2024-06-04 09:15:00 | 2894.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-06-04 11:15:00 | 2851.95 | 2024-06-04 11:15:00 | 2894.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-06-25 14:15:00 | 2857.50 | 2024-07-10 14:15:00 | 2995.35 | STOP_HIT | 1.00 | -4.82% |
| SELL | retest2 | 2025-02-05 10:15:00 | 2267.40 | 2025-04-03 09:15:00 | 2318.55 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-03-26 12:15:00 | 2311.30 | 2025-04-08 13:15:00 | 2417.10 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2025-03-27 09:15:00 | 2313.30 | 2025-04-08 13:15:00 | 2417.10 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-04-01 14:15:00 | 2316.35 | 2025-04-08 13:15:00 | 2417.10 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2025-04-02 10:15:00 | 2304.00 | 2025-04-08 13:15:00 | 2417.10 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2025-05-08 15:15:00 | 2300.90 | 2025-05-09 09:15:00 | 2318.55 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-05-09 12:15:00 | 2295.40 | 2025-05-12 09:15:00 | 2318.55 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-05-14 10:15:00 | 2294.80 | 2025-05-15 13:15:00 | 2318.55 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-05-19 11:15:00 | 2341.30 | 2025-05-26 10:15:00 | 2319.50 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-05-27 14:15:00 | 2326.10 | 2025-06-27 13:15:00 | 2351.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-30 12:15:00 | 2328.00 | 2025-07-01 09:15:00 | 2351.60 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-25 12:15:00 | 2332.60 | 2025-07-28 10:15:00 | 2351.60 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-10-01 10:15:00 | 2327.40 | 2025-10-03 09:15:00 | 2351.60 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-01-19 10:15:00 | 2779.20 | 2026-01-19 15:15:00 | 2750.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest1 | 2026-04-08 11:15:00 | 2255.70 | 2026-04-10 09:15:00 | 2363.47 | STOP_HIT | 1.00 | -4.78% |
| SELL | retest1 | 2026-04-09 10:15:00 | 2272.20 | 2026-04-10 09:15:00 | 2363.47 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest1 | 2026-04-09 13:15:00 | 2265.10 | 2026-04-10 09:15:00 | 2363.47 | STOP_HIT | 1.00 | -4.34% |
