# ADANIENT (ADANIENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 2540.30
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 10 |
| PENDING | 24 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 15 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 16
- **Target hits / Stop hits / Partials:** 2 / 18 / 5
- **Avg / median % per leg:** 3.95% / -1.54%
- **Sum % (uncompounded):** 98.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 0 | 2 | 0 | -2.27% | -4.5% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.00% | 1.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.54% | -5.5% |
| SELL (all) | 23 | 8 | 34.8% | 2 | 16 | 5 | 4.49% | 103.3% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 4 | 2 | 4.53% | 27.2% |
| SELL @ 3rd Alert (retest2) | 17 | 5 | 29.4% | 2 | 12 | 3 | 4.47% | 76.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 5 | 2 | 4.03% | 28.2% |
| retest2 (combined) | 18 | 5 | 27.8% | 2 | 13 | 3 | 3.92% | 70.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 2445.50 | 2480.32 | 2480.47 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 2529.00 | 2480.99 | 2480.79 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 10:15:00 | 2449.00 | 2480.71 | 2480.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 13:15:00 | 2444.00 | 2479.82 | 2480.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 2373.20 | 2268.03 | 2335.21 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 2373.20 | 2268.03 | 2335.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 2373.20 | 2268.03 | 2335.21 | EMA400 retest candle locked |

### Cycle 4 — BUY (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 14:15:00 | 2891.00 | 2383.34 | 2382.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 13:15:00 | 2924.20 | 2546.51 | 2472.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 2895.00 | 2908.15 | 2769.51 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-29 09:15:00 | 3046.70 | 2905.26 | 2778.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:15:00 | 3045.95 | 2906.66 | 2779.98 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 3033.35 | 3197.75 | 3076.27 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-03-13 09:15:00 | 3076.27 | 3197.75 | 3076.27 | SL hit qty=1.00 sl=3076.27 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-26 09:15:00 | 3141.00 | 3141.69 | 3073.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-26 10:15:00 | 3130.00 | 3141.57 | 3073.80 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-28 10:15:00 | 3155.60 | 3139.21 | 3077.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:15:00 | 3156.40 | 3139.38 | 3077.55 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-19 09:15:00 | 2981.40 | 3167.87 | 3119.19 | SL hit qty=1.00 sl=2981.40 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 2835.00 | 3088.49 | 3089.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 2816.20 | 3057.67 | 3073.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 12:15:00 | 3027.80 | 3017.76 | 3050.68 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 3090.75 | 3018.92 | 3050.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 3090.75 | 3018.92 | 3050.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-05-16 09:15:00 | 3014.05 | 3021.24 | 3050.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 10:15:00 | 3014.10 | 3021.17 | 3050.53 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-21 11:15:00 | 3118.50 | 3025.97 | 3050.85 | SL hit qty=1.00 sl=3118.50 alert=retest2 |

### Cycle 6 — BUY (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 14:15:00 | 3385.60 | 3074.56 | 3073.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 3410.90 | 3136.60 | 3107.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.95 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.95 | EMA400 retest candle locked |

### Cycle 7 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 3064.50 | 3145.00 | 3145.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 3037.35 | 3143.93 | 3144.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 10:15:00 | 3105.00 | 3099.62 | 3119.83 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-07-29 12:15:00 | 3089.90 | 3099.53 | 3119.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 13:15:00 | 3082.30 | 3099.36 | 3119.40 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 3119.60 | 3099.04 | 3118.74 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-07-30 11:15:00 | 3118.74 | 3099.04 | 3118.74 | SL hit qty=1.00 sl=3118.74 alert=retest1 |
| Cross detected — sustain check pending | 2024-08-05 10:15:00 | 3047.50 | 3115.55 | 3125.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 11:15:00 | 3046.05 | 3114.85 | 3124.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 3120.65 | 3111.94 | 3122.95 | SL hit qty=1.00 sl=3120.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-06 14:15:00 | 3064.70 | 3111.74 | 3122.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:15:00 | 3072.70 | 3111.35 | 3122.34 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 3120.65 | 3111.61 | 3122.42 | SL hit qty=1.00 sl=3120.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-12 09:15:00 | 3071.15 | 3123.81 | 3127.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-12 10:15:00 | 3146.90 | 3124.04 | 3127.86 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-13 15:15:00 | 3082.10 | 3123.72 | 3127.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 3070.50 | 3123.19 | 3127.20 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2024-08-16 13:15:00 | 3120.65 | 3116.19 | 3123.38 | SL hit qty=1.00 sl=3120.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-20 11:15:00 | 3078.40 | 3115.11 | 3122.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 12:15:00 | 3073.45 | 3114.69 | 3122.17 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 3124.00 | 3113.52 | 3121.43 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 3120.65 | 3113.52 | 3121.43 | SL hit qty=1.00 sl=3120.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-26 10:15:00 | 3058.00 | 3111.15 | 3119.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 11:15:00 | 3070.50 | 3110.74 | 3119.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-27 13:15:00 | 3069.80 | 3107.56 | 3117.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:15:00 | 3068.00 | 3107.17 | 3116.92 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-26 11:15:00 | 3127.90 | 3030.51 | 3057.91 | SL hit qty=1.00 sl=3127.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-26 11:15:00 | 3127.90 | 3030.51 | 3057.91 | SL hit qty=1.00 sl=3127.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-07 09:15:00 | 3064.10 | 3063.90 | 3071.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 3024.75 | 3063.51 | 3071.15 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-08 12:15:00 | 3127.90 | 3062.30 | 3070.17 | SL hit qty=1.00 sl=3127.90 alert=retest2 |

### Cycle 8 — BUY (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 13:15:00 | 3128.80 | 3077.40 | 3077.34 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 3006.60 | 3077.15 | 3077.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 2998.90 | 3075.78 | 3076.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 2980.00 | 2975.07 | 3020.06 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-10-30 15:15:00 | 2959.00 | 2974.78 | 3019.02 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-31 09:15:00 | 2963.95 | 2974.67 | 3018.75 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2024-10-31 10:15:00 | 2957.00 | 2974.50 | 3018.44 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:15:00 | 2961.00 | 2974.36 | 3018.15 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 3022.65 | 2964.08 | 3008.34 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 3008.34 | 2964.08 | 3008.34 | SL hit qty=1.00 sl=3008.34 alert=retest1 |
| Cross detected — sustain check pending | 2024-11-07 09:15:00 | 2965.95 | 2967.16 | 3008.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:15:00 | 2965.10 | 2967.14 | 3008.59 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-07 13:15:00 | 2976.35 | 2967.38 | 3008.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 14:15:00 | 2969.75 | 2967.40 | 3007.90 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-21 09:15:00 | 2520.34 | 2921.50 | 2973.89 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-21 09:15:00 | 2524.29 | 2921.50 | 2973.89 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-11-22 09:15:00 | 2075.57 | 2873.45 | 2947.66 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2024-11-22 09:15:00 | 2078.82 | 2873.45 | 2947.66 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-04-24 10:15:00 | 2451.50 | 2342.97 | 2342.53 | HTF filter: close below htf_sma |

### Cycle 10 — SELL (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 15:15:00 | 2255.00 | 2344.08 | 2344.24 | EMA200 below EMA400 |
| CROSSOVER_SKIP | 2025-05-12 09:15:00 | 2387.20 | 2344.51 | 2344.45 | HTF filter: close below htf_sma |

### Cycle 11 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 2265.20 | 2507.17 | 2507.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 2240.50 | 2504.52 | 2506.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 2381.20 | 2343.88 | 2393.71 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 10:15:00 | 2406.80 | 2344.51 | 2393.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 2406.80 | 2344.51 | 2393.77 | EMA400 retest candle locked |

### Cycle 12 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 2616.50 | 2422.51 | 2422.42 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 2456.80 | 2469.85 | 2469.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 15:15:00 | 2446.10 | 2468.80 | 2469.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 2279.60 | 2277.23 | 2331.21 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-06 10:15:00 | 2265.00 | 2277.60 | 2328.52 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-06 11:15:00 | 2271.20 | 2277.54 | 2328.23 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-06 13:15:00 | 2254.80 | 2277.25 | 2327.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 14:15:00 | 2258.90 | 2277.07 | 2327.24 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-08 09:15:00 | 2253.10 | 2276.40 | 2324.69 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:15:00 | 2234.90 | 2275.98 | 2324.24 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-01-23 13:15:00 | 1920.06 | 2192.93 | 2262.15 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-01-23 13:15:00 | 1899.66 | 2192.93 | 2262.15 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2182.00 | 2128.62 | 2214.71 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-04 11:15:00 | 2234.90 | 2135.86 | 2214.62 | SL hit qty=0.50 sl=2234.90 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 2258.90 | 2151.39 | 2215.62 | SL hit qty=0.50 sl=2258.90 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 2154.50 | 2169.61 | 2217.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:15:00 | 2157.10 | 2169.49 | 2216.78 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 2232.10 | 2169.65 | 2213.18 | SL hit qty=1.00 sl=2232.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-19 14:15:00 | 2155.70 | 2173.84 | 2212.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 2156.70 | 2173.67 | 2211.78 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 2232.10 | 2176.58 | 2208.27 | SL hit qty=1.00 sl=2232.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-27 15:15:00 | 2157.00 | 2179.33 | 2207.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 2093.00 | 2178.47 | 2207.00 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 3960m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-30 14:15:00 | 1779.05 | 1998.11 | 2081.61 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 2093.00 | 1961.99 | 2049.91 | SL hit qty=0.50 sl=2093.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-15 13:15:00 | 2152.80 | 1997.51 | 2056.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 14:15:00 | 2144.00 | 1998.96 | 2056.55 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 2139.90 | 2000.37 | 2056.97 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 2232.10 | 2014.79 | 2062.10 | SL hit qty=1.00 sl=2232.10 alert=retest2 |

### Cycle 14 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 2323.20 | 2101.36 | 2100.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 2394.20 | 2106.45 | 2103.03 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-29 10:15:00 | 3045.95 | 2024-03-13 09:15:00 | 3076.27 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2024-03-28 11:15:00 | 3156.40 | 2024-04-19 09:15:00 | 2981.40 | STOP_HIT | 1.00 | -5.54% |
| SELL | retest2 | 2024-05-16 10:15:00 | 3014.10 | 2024-05-21 11:15:00 | 3118.50 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest1 | 2024-07-29 13:15:00 | 3082.30 | 2024-07-30 11:15:00 | 3118.74 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-08-05 11:15:00 | 3046.05 | 2024-08-06 09:15:00 | 3120.65 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-08-06 15:15:00 | 3072.70 | 2024-08-07 09:15:00 | 3120.65 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-08-14 09:15:00 | 3070.50 | 2024-08-16 13:15:00 | 3120.65 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-08-20 12:15:00 | 3073.45 | 2024-08-21 09:15:00 | 3120.65 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-08-26 11:15:00 | 3070.50 | 2024-09-26 11:15:00 | 3127.90 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-08-27 14:15:00 | 3068.00 | 2024-09-26 11:15:00 | 3127.90 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-10-07 10:15:00 | 3024.75 | 2024-10-08 12:15:00 | 3127.90 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest1 | 2024-10-31 11:15:00 | 2961.00 | 2024-11-06 11:15:00 | 3008.34 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-11-07 10:15:00 | 2965.10 | 2024-11-21 09:15:00 | 2520.34 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-11-07 14:15:00 | 2969.75 | 2024-11-21 09:15:00 | 2524.29 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-11-07 10:15:00 | 2965.10 | 2024-11-22 09:15:00 | 2075.57 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2024-11-07 14:15:00 | 2969.75 | 2024-11-22 09:15:00 | 2078.82 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest1 | 2026-01-06 14:15:00 | 2258.90 | 2026-01-23 13:15:00 | 1920.06 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2026-01-08 10:15:00 | 2234.90 | 2026-01-23 13:15:00 | 1899.66 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2026-01-06 14:15:00 | 2258.90 | 2026-02-04 11:15:00 | 2234.90 | STOP_HIT | 0.50 | 1.06% |
| SELL | retest1 | 2026-01-08 10:15:00 | 2234.90 | 2026-02-09 09:15:00 | 2258.90 | STOP_HIT | 0.50 | -1.07% |
| SELL | retest2 | 2026-02-13 10:15:00 | 2157.10 | 2026-02-17 12:15:00 | 2232.10 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-02-19 15:15:00 | 2156.70 | 2026-02-25 15:15:00 | 2232.10 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2093.00 | 2026-03-30 14:15:00 | 1779.05 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2093.00 | 2026-04-08 10:15:00 | 2093.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest2 | 2026-04-15 14:15:00 | 2144.00 | 2026-04-17 09:15:00 | 2232.10 | STOP_HIT | 1.00 | -4.11% |
