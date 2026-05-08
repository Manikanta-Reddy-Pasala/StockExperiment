# ASIANPAINT (ASIANPAINT)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 2599.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 8 |
| ALERT3 | 13 |
| PENDING | 41 |
| PENDING_CANCEL | 13 |
| ENTRY1 | 7 |
| ENTRY2 | 21 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 27
- **Target hits / Stop hits / Partials:** 1 / 27 / 0
- **Avg / median % per leg:** -2.58% / -2.40%
- **Sum % (uncompounded):** -72.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 1 | 5 | 0 | -1.04% | -6.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 1 | 5 | 0 | -1.04% | -6.2% |
| SELL (all) | 22 | 0 | 0.0% | 0 | 22 | 0 | -3.00% | -65.9% |
| SELL @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.17% | -22.2% |
| SELL @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -2.92% | -43.7% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.17% | -22.2% |
| retest2 (combined) | 21 | 1 | 4.8% | 1 | 20 | 0 | -2.38% | -50.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 3244.55 | 3164.21 | 3163.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 3263.85 | 3170.83 | 3167.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 09:15:00 | 3291.85 | 3297.58 | 3248.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 3249.35 | 3294.03 | 3251.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 3249.35 | 3294.03 | 3251.64 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-16 09:15:00 | 3298.00 | 3291.12 | 3252.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 10:15:00 | 3300.40 | 3291.21 | 3253.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 3145.30 | 3288.42 | 3254.22 | SL hit (close<static) qty=1.00 sl=3244.50 alert=retest2 |

### Cycle 2 — SELL (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 14:15:00 | 2948.15 | 3225.14 | 3225.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 12:15:00 | 2937.40 | 3164.92 | 3193.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 13:15:00 | 2881.75 | 2880.13 | 2951.46 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-04-03 14:15:00 | 2865.30 | 2879.98 | 2951.03 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-03 15:15:00 | 2871.70 | 2879.90 | 2950.63 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-04 09:15:00 | 2852.00 | 2879.62 | 2950.14 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:15:00 | 2866.00 | 2879.48 | 2949.72 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-09 13:15:00 | 2848.05 | 2882.34 | 2943.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-09 14:15:00 | 2855.65 | 2882.08 | 2942.90 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-12 09:15:00 | 2852.40 | 2881.79 | 2940.07 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 10:15:00 | 2862.55 | 2881.60 | 2939.69 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-24 09:15:00 | 2866.75 | 2865.05 | 2917.99 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-24 10:15:00 | 2875.00 | 2865.15 | 2917.77 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-24 13:15:00 | 2866.50 | 2865.27 | 2917.05 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-24 14:15:00 | 2870.65 | 2865.32 | 2916.82 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-25 09:15:00 | 2841.20 | 2865.13 | 2916.21 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:15:00 | 2840.30 | 2864.88 | 2915.83 | SELL ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 2906.00 | 2864.69 | 2910.62 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-04-30 13:15:00 | 2889.95 | 2865.57 | 2910.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 14:15:00 | 2872.65 | 2865.64 | 2910.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 2919.70 | 2866.28 | 2910.07 | SL hit (close>ema400) qty=1.00 sl=2910.07 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 2919.70 | 2866.28 | 2910.07 | SL hit (close>ema400) qty=1.00 sl=2910.07 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 2919.70 | 2866.28 | 2910.07 | SL hit (close>ema400) qty=1.00 sl=2910.07 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 2919.70 | 2866.28 | 2910.07 | SL hit (close>ema400) qty=1.00 sl=2910.07 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 2919.70 | 2866.28 | 2910.07 | SL hit (close>static) qty=1.00 sl=2911.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-08 09:15:00 | 2879.85 | 2882.62 | 2913.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 10:15:00 | 2836.30 | 2882.16 | 2912.79 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-23 10:15:00 | 2891.00 | 2858.31 | 2889.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 11:15:00 | 2885.25 | 2858.58 | 2889.18 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-24 09:15:00 | 2883.25 | 2860.37 | 2889.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:15:00 | 2888.00 | 2860.64 | 2889.32 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 2887.15 | 2860.91 | 2889.31 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-24 12:15:00 | 2873.40 | 2861.03 | 2889.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:15:00 | 2879.05 | 2861.21 | 2889.18 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-27 14:15:00 | 2872.85 | 2862.49 | 2888.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 15:15:00 | 2872.30 | 2862.59 | 2888.65 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-28 13:15:00 | 2906.00 | 2863.86 | 2888.65 | SL hit (close>static) qty=1.00 sl=2894.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-28 13:15:00 | 2906.00 | 2863.86 | 2888.65 | SL hit (close>static) qty=1.00 sl=2894.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-28 15:15:00 | 2920.00 | 2864.78 | 2888.87 | SL hit (close>static) qty=1.00 sl=2911.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-28 15:15:00 | 2920.00 | 2864.78 | 2888.87 | SL hit (close>static) qty=1.00 sl=2911.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-28 15:15:00 | 2920.00 | 2864.78 | 2888.87 | SL hit (close>static) qty=1.00 sl=2911.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-30 14:15:00 | 2879.55 | 2868.96 | 2889.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-30 15:15:00 | 2883.70 | 2869.11 | 2889.49 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-31 10:15:00 | 2873.45 | 2869.30 | 2889.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-31 11:15:00 | 2893.20 | 2869.54 | 2889.40 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-31 13:15:00 | 2878.50 | 2869.90 | 2889.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-31 14:15:00 | 2885.45 | 2870.06 | 2889.37 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-31 15:15:00 | 2881.20 | 2870.17 | 2889.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-03 09:15:00 | 2890.05 | 2870.37 | 2889.33 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2024-06-03 12:15:00 | 2873.95 | 2870.67 | 2889.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:15:00 | 2872.00 | 2870.68 | 2889.11 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-04 10:15:00 | 2842.20 | 2870.54 | 2888.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:15:00 | 2851.95 | 2870.36 | 2888.50 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 2992.15 | 2871.13 | 2888.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 2992.15 | 2871.13 | 2888.44 | SL hit (close>static) qty=1.00 sl=2894.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 2992.15 | 2871.13 | 2888.44 | SL hit (close>static) qty=1.00 sl=2894.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-25 13:15:00 | 2869.35 | 2893.57 | 2896.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 14:15:00 | 2857.50 | 2893.21 | 2896.35 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-03 14:15:00 | 2924.45 | 2898.37 | 2898.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 2924.45 | 2898.37 | 2898.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 2942.55 | 2899.08 | 2898.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 2898.75 | 2903.89 | 2901.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 2898.75 | 2903.89 | 2901.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 2898.75 | 2903.89 | 2901.20 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-07-10 13:15:00 | 2995.00 | 2905.39 | 2902.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:15:00 | 2996.30 | 2906.29 | 2902.65 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-19 09:15:00 | 2950.40 | 2927.29 | 2914.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 10:15:00 | 2959.10 | 2927.60 | 2915.19 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 2888.10 | 2927.56 | 2916.40 | SL hit (close<static) qty=1.00 sl=2890.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 2888.10 | 2927.56 | 2916.40 | SL hit (close<static) qty=1.00 sl=2890.70 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-26 09:15:00 | 2939.90 | 2925.27 | 2915.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 2943.55 | 2925.46 | 2916.11 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2024-09-04 10:15:00 | 3237.91 | 3086.54 | 3030.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 14:15:00 | 2987.40 | 3120.09 | 3120.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 2959.95 | 3114.78 | 3117.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 2301.20 | 2298.95 | 2414.55 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-04 09:15:00 | 2274.20 | 2299.05 | 2409.53 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-04 10:15:00 | 2293.40 | 2299.00 | 2408.95 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 13:15:00 | 2395.70 | 2300.28 | 2407.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 2395.70 | 2300.28 | 2407.95 | EMA400 retest candle locked (from downside) |
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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 2304.00 | 2279.00 | 2304.16 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-04-02 13:15:00 | 2296.65 | 2279.67 | 2304.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-02 14:15:00 | 2305.70 | 2279.93 | 2304.13 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-04-09 10:15:00 | 2419.20 | 2300.40 | 2311.51 | SL hit (close>static) qty=1.00 sl=2417.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-09 10:15:00 | 2419.20 | 2300.40 | 2311.51 | SL hit (close>static) qty=1.00 sl=2417.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-09 10:15:00 | 2419.20 | 2300.40 | 2311.51 | SL hit (close>static) qty=1.00 sl=2417.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-09 10:15:00 | 2419.20 | 2300.40 | 2311.51 | SL hit (close>static) qty=1.00 sl=2417.10 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 2442.00 | 2322.15 | 2321.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 2453.80 | 2323.46 | 2322.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 2378.70 | 2391.07 | 2364.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 14:15:00 | 2332.30 | 2390.27 | 2364.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 2332.30 | 2390.27 | 2364.74 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 2319.50 | 2348.46 | 2348.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 2309.10 | 2348.07 | 2348.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 2348.00 | 2346.84 | 2347.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 2348.00 | 2346.84 | 2347.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 2348.00 | 2346.84 | 2347.72 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-05-27 13:15:00 | 2329.60 | 2346.62 | 2347.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 14:15:00 | 2326.10 | 2346.41 | 2347.49 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 2351.80 | 2281.99 | 2301.03 | SL hit (close>static) qty=1.00 sl=2351.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-30 11:15:00 | 2335.00 | 2285.21 | 2302.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 2328.00 | 2285.63 | 2302.32 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 2370.00 | 2288.03 | 2303.20 | SL hit (close>static) qty=1.00 sl=2351.60 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 15:15:00 | 2424.20 | 2316.55 | 2316.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 2448.00 | 2317.85 | 2317.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 2369.70 | 2371.84 | 2350.24 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-21 15:15:00 | 2378.80 | 2371.90 | 2350.59 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-22 09:15:00 | 2367.80 | 2371.86 | 2350.68 | ENTRY1 sustain failed after 1080m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 11:15:00 | 2344.00 | 2370.65 | 2351.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 2344.00 | 2370.65 | 2351.68 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-29 13:15:00 | 2398.20 | 2366.20 | 2351.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:15:00 | 2399.80 | 2366.54 | 2351.61 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 2322.70 | 2465.84 | 2456.30 | SL hit (close<static) qty=1.00 sl=2341.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 2346.40 | 2446.58 | 2447.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 2336.00 | 2435.12 | 2441.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 2408.40 | 2404.31 | 2422.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 2514.70 | 2405.47 | 2422.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2514.70 | 2405.47 | 2422.94 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 2518.80 | 2438.26 | 2437.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 2524.60 | 2443.70 | 2440.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 14:15:00 | 2794.50 | 2797.80 | 2683.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 14:15:00 | 2756.40 | 2805.29 | 2754.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2756.40 | 2805.29 | 2754.27 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-19 09:15:00 | 2765.60 | 2804.41 | 2754.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 2779.20 | 2804.15 | 2754.46 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 2715.50 | 2801.31 | 2754.49 | SL hit (close<static) qty=1.00 sl=2750.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-23 10:15:00 | 2782.00 | 2780.07 | 2748.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-23 11:15:00 | 2755.60 | 2779.82 | 2748.16 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-23 12:15:00 | 2764.10 | 2779.67 | 2748.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-23 13:15:00 | 2729.00 | 2779.16 | 2748.15 | ENTRY2 sustain failed after 60m |

### Cycle 10 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 2431.50 | 2720.59 | 2721.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 2407.70 | 2706.31 | 2713.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2280.80 | 2264.25 | 2369.63 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 10:15:00 | 2269.10 | 2264.29 | 2369.13 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 11:15:00 | 2255.70 | 2264.21 | 2368.56 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 2271.40 | 2264.81 | 2366.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:15:00 | 2272.20 | 2264.89 | 2365.82 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 2272.10 | 2265.10 | 2364.92 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 13:15:00 | 2265.10 | 2265.10 | 2364.43 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 2364.40 | 2266.14 | 2363.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.40 | 2266.14 | 2363.47 | SL hit (close>ema400) qty=1.00 sl=2363.47 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.40 | 2266.14 | 2363.47 | SL hit (close>ema400) qty=1.00 sl=2363.47 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.40 | 2266.14 | 2363.47 | SL hit (close>ema400) qty=1.00 sl=2363.47 alert=retest1 |

### Cycle 11 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 2557.00 | 2412.89 | 2412.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 10:15:00 | 2597.00 | 2414.72 | 2413.74 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-16 10:15:00 | 3300.40 | 2024-01-18 09:15:00 | 3145.30 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest1 | 2024-04-04 10:15:00 | 2866.00 | 2024-05-02 09:15:00 | 2919.70 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest1 | 2024-04-09 14:15:00 | 2855.65 | 2024-05-02 09:15:00 | 2919.70 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest1 | 2024-04-12 10:15:00 | 2862.55 | 2024-05-02 09:15:00 | 2919.70 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest1 | 2024-04-25 10:15:00 | 2840.30 | 2024-05-02 09:15:00 | 2919.70 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2024-04-30 14:15:00 | 2872.65 | 2024-05-02 09:15:00 | 2919.70 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-05-08 10:15:00 | 2836.30 | 2024-05-28 13:15:00 | 2906.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-05-23 11:15:00 | 2885.25 | 2024-05-28 13:15:00 | 2906.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-05-24 10:15:00 | 2888.00 | 2024-05-28 15:15:00 | 2920.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-05-24 13:15:00 | 2879.05 | 2024-05-28 15:15:00 | 2920.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-05-27 15:15:00 | 2872.30 | 2024-05-28 15:15:00 | 2920.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-06-03 13:15:00 | 2872.00 | 2024-06-05 09:15:00 | 2992.15 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2024-06-04 11:15:00 | 2851.95 | 2024-06-05 09:15:00 | 2992.15 | STOP_HIT | 1.00 | -4.92% |
| SELL | retest2 | 2024-06-25 14:15:00 | 2857.50 | 2024-07-03 14:15:00 | 2924.45 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-07-10 14:15:00 | 2996.30 | 2024-07-24 09:15:00 | 2888.10 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2024-07-19 10:15:00 | 2959.10 | 2024-07-24 09:15:00 | 2888.10 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-07-26 10:15:00 | 2943.55 | 2024-09-04 10:15:00 | 3237.91 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-05 10:15:00 | 2267.40 | 2025-04-09 10:15:00 | 2419.20 | STOP_HIT | 1.00 | -6.69% |
| SELL | retest2 | 2025-03-26 12:15:00 | 2311.30 | 2025-04-09 10:15:00 | 2419.20 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-03-27 09:15:00 | 2313.30 | 2025-04-09 10:15:00 | 2419.20 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2025-04-01 14:15:00 | 2316.35 | 2025-04-09 10:15:00 | 2419.20 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-05-27 14:15:00 | 2326.10 | 2025-06-27 13:15:00 | 2351.80 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-30 12:15:00 | 2328.00 | 2025-07-01 09:15:00 | 2370.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-07-29 14:15:00 | 2399.80 | 2025-10-01 09:15:00 | 2322.70 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2026-01-19 10:15:00 | 2779.20 | 2026-01-20 09:15:00 | 2715.50 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest1 | 2026-04-08 11:15:00 | 2255.70 | 2026-04-10 09:15:00 | 2364.40 | STOP_HIT | 1.00 | -4.82% |
| SELL | retest1 | 2026-04-09 10:15:00 | 2272.20 | 2026-04-10 09:15:00 | 2364.40 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest1 | 2026-04-09 13:15:00 | 2265.10 | 2026-04-10 09:15:00 | 2364.40 | STOP_HIT | 1.00 | -4.38% |
