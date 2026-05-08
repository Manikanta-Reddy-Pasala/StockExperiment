# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:15:00 (4996 bars)
- **Last close:** 2505.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 5 |
| ALERT3 | 11 |
| PENDING | 27 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 16 |
| PARTIAL | 4 |
| TARGET_HIT | 6 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 14
- **Target hits / Stop hits / Partials:** 6 / 16 / 4
- **Avg / median % per leg:** 1.58% / -1.21%
- **Sum % (uncompounded):** 41.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 0 | 4 | 2 | 0.46% | 2.8% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.54% | 14.2% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.70% | -11.4% |
| SELL (all) | 20 | 8 | 40.0% | 6 | 12 | 2 | 1.92% | 38.4% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.45% | 26.7% |
| SELL @ 3rd Alert (retest2) | 14 | 4 | 28.6% | 4 | 10 | 0 | 0.84% | 11.7% |
| retest1 (combined) | 10 | 8 | 80.0% | 2 | 4 | 4 | 4.09% | 40.9% |
| retest2 (combined) | 16 | 4 | 25.0% | 4 | 12 | 0 | 0.02% | 0.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 2386.25 | 2464.52 | 2464.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 2349.30 | 2461.31 | 2463.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 2373.20 | 2268.02 | 2332.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 2373.20 | 2268.02 | 2332.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 2373.20 | 2268.02 | 2332.02 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 13:15:00 | 2909.35 | 2378.24 | 2377.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 10:15:00 | 2963.55 | 2561.25 | 2479.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 2895.00 | 2908.15 | 2768.68 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-29 09:15:00 | 3046.70 | 2905.26 | 2777.89 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:15:00 | 3045.95 | 2906.66 | 2779.23 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 13:15:00 | 3198.25 | 2949.18 | 2816.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-12 10:15:00 | 3170.00 | 3202.44 | 3074.61 | SL hit (close<ema200) qty=0.50 sl=3202.44 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 3033.35 | 3197.75 | 3076.02 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-03-26 09:15:00 | 3141.00 | 3141.69 | 3073.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-26 10:15:00 | 3130.00 | 3141.57 | 3073.61 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-28 10:15:00 | 3155.60 | 3139.21 | 3076.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:15:00 | 3156.40 | 3139.38 | 3077.37 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-06 09:15:00 | 2861.15 | 3111.18 | 3100.48 | SL hit (close<static) qty=1.00 sl=2981.40 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 2835.00 | 3088.49 | 3089.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 2816.20 | 3057.67 | 3073.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 12:15:00 | 3027.80 | 3017.76 | 3050.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 3090.75 | 3018.92 | 3050.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 3090.75 | 3018.92 | 3050.55 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-16 09:15:00 | 3014.05 | 3021.24 | 3050.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 10:15:00 | 3014.10 | 3021.17 | 3050.47 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-21 11:15:00 | 3145.95 | 3025.97 | 3050.80 | SL hit (close>static) qty=1.00 sl=3118.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 14:15:00 | 3385.60 | 3074.56 | 3073.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 3410.90 | 3136.60 | 3107.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.91 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 3064.50 | 3145.00 | 3145.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 3037.35 | 3143.93 | 3144.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 10:15:00 | 3105.00 | 3099.62 | 3119.82 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-07-29 12:15:00 | 3089.90 | 3099.53 | 3119.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 13:15:00 | 3082.30 | 3099.36 | 3119.39 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 3119.60 | 3099.04 | 3118.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-30 11:15:00 | 3119.60 | 3099.04 | 3118.73 | SL hit (close>ema400) qty=1.00 sl=3118.73 alert=retest1 |
| Cross detected — sustain check pending | 2024-08-05 10:15:00 | 3047.50 | 3115.55 | 3125.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 11:15:00 | 3046.05 | 3114.85 | 3124.67 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 3127.25 | 3111.94 | 3122.95 | SL hit (close>static) qty=1.00 sl=3120.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-06 14:15:00 | 3064.70 | 3111.74 | 3122.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:15:00 | 3072.70 | 3111.35 | 3122.33 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 3137.45 | 3111.61 | 3122.41 | SL hit (close>static) qty=1.00 sl=3120.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-12 09:15:00 | 3071.15 | 3123.81 | 3127.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-12 10:15:00 | 3146.90 | 3124.04 | 3127.85 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-13 15:15:00 | 3082.10 | 3123.72 | 3127.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 3070.50 | 3123.19 | 3127.19 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 3122.80 | 3116.08 | 3123.21 | SL hit (close>static) qty=1.00 sl=3120.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-20 11:15:00 | 3078.40 | 3115.11 | 3122.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 12:15:00 | 3073.45 | 3114.69 | 3122.16 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 3124.00 | 3113.52 | 3121.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 3124.00 | 3113.52 | 3121.42 | SL hit (close>static) qty=1.00 sl=3120.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-08-26 10:15:00 | 3058.00 | 3111.15 | 3119.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 11:15:00 | 3070.50 | 3110.74 | 3119.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-27 13:15:00 | 3069.80 | 3107.56 | 3117.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:15:00 | 3068.00 | 3107.17 | 3116.91 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 3131.95 | 3034.71 | 3059.36 | SL hit (close>static) qty=1.00 sl=3127.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 3131.95 | 3034.71 | 3059.36 | SL hit (close>static) qty=1.00 sl=3127.90 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-07 09:15:00 | 3064.10 | 3063.90 | 3071.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 3024.75 | 3063.51 | 3071.15 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-08 12:15:00 | 3144.00 | 3062.30 | 3070.17 | SL hit (close>static) qty=1.00 sl=3127.90 alert=retest2 |

### Cycle 6 — BUY (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 13:15:00 | 3128.80 | 3077.40 | 3077.34 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 3006.60 | 3077.15 | 3077.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 2998.90 | 3075.78 | 3076.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 2980.00 | 2975.07 | 3020.06 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-10-30 15:15:00 | 2959.00 | 2974.78 | 3019.02 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-31 09:15:00 | 2963.95 | 2974.67 | 3018.75 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2024-10-31 10:15:00 | 2957.00 | 2974.50 | 3018.44 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:15:00 | 2961.00 | 2974.36 | 3018.15 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 3022.65 | 2964.08 | 3008.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 3022.65 | 2964.08 | 3008.34 | SL hit (close>ema400) qty=1.00 sl=3008.34 alert=retest1 |
| Cross detected — sustain check pending | 2024-11-07 09:15:00 | 2965.95 | 2967.16 | 3008.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:15:00 | 2965.10 | 2967.14 | 3008.59 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-07 13:15:00 | 2976.35 | 2967.38 | 3008.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 14:15:00 | 2969.75 | 2967.40 | 3007.90 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2024-11-21 09:15:00 | 2668.59 | 2921.50 | 2973.89 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2024-11-21 09:15:00 | 2672.78 | 2921.50 | 2973.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 10:15:00 | 2451.50 | 2342.97 | 2342.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 2478.70 | 2347.46 | 2345.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 15:15:00 | 2349.00 | 2353.29 | 2348.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 15:15:00 | 2349.00 | 2353.29 | 2348.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 2349.00 | 2353.29 | 2348.48 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 15:15:00 | 2255.00 | 2344.08 | 2344.24 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2387.20 | 2344.51 | 2344.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 2416.70 | 2345.23 | 2344.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 2454.80 | 2463.35 | 2420.16 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-04 10:15:00 | 2478.00 | 2463.50 | 2420.45 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-04 11:15:00 | 2472.00 | 2463.58 | 2420.71 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-04 12:15:00 | 2478.60 | 2463.73 | 2420.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 13:15:00 | 2495.00 | 2464.05 | 2421.36 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 11:15:00 | 2619.75 | 2483.31 | 2436.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-13 10:15:00 | 2497.60 | 2499.89 | 2449.90 | SL hit (close<ema200) qty=0.50 sl=2499.89 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 2459.00 | 2501.24 | 2456.09 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 2475.60 | 2491.46 | 2455.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 2479.50 | 2491.35 | 2455.18 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 2428.60 | 2563.59 | 2533.26 | SL hit (close<static) qty=1.00 sl=2450.10 alert=retest2 |

### Cycle 11 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 2265.20 | 2507.17 | 2507.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 2240.50 | 2504.52 | 2506.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 2381.20 | 2343.88 | 2393.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 10:15:00 | 2406.80 | 2344.51 | 2393.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 2406.80 | 2344.51 | 2393.77 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 2616.50 | 2422.51 | 2422.42 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 2456.80 | 2469.85 | 2469.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 15:15:00 | 2446.10 | 2468.80 | 2469.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 2279.60 | 2277.23 | 2331.21 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-06 10:15:00 | 2265.00 | 2277.60 | 2328.52 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-06 11:15:00 | 2271.20 | 2277.54 | 2328.23 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-06 13:15:00 | 2254.80 | 2277.25 | 2327.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 14:15:00 | 2258.90 | 2277.07 | 2327.24 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-08 09:15:00 | 2253.10 | 2276.40 | 2324.69 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:15:00 | 2234.90 | 2275.98 | 2324.24 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 2145.95 | 2267.07 | 2317.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 2123.15 | 2264.81 | 2315.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-01-21 10:15:00 | 2033.01 | 2219.95 | 2281.20 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-01-21 10:15:00 | 2011.41 | 2219.95 | 2281.20 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2182.00 | 2128.62 | 2214.71 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 2154.50 | 2169.61 | 2217.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:15:00 | 2157.10 | 2169.49 | 2216.78 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 2234.20 | 2169.65 | 2213.18 | SL hit (close>static) qty=1.00 sl=2232.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-19 14:15:00 | 2155.70 | 2173.84 | 2212.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 2156.70 | 2173.67 | 2211.78 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-27 15:15:00 | 2157.00 | 2179.33 | 2207.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 2093.00 | 2178.47 | 2207.00 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 3960m) |
| Target hit | 2026-03-19 12:15:00 | 1941.03 | 2076.02 | 2136.24 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 1883.70 | 2061.07 | 2125.35 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-15 13:15:00 | 2152.80 | 1997.51 | 2056.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 14:15:00 | 2144.00 | 1998.96 | 2056.55 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 2233.80 | 2030.07 | 2068.03 | SL hit (close>static) qty=1.00 sl=2232.10 alert=retest2 |

### Cycle 14 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 2323.20 | 2101.36 | 2100.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 2394.20 | 2106.45 | 2103.03 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-29 10:15:00 | 3045.95 | 2024-02-01 13:15:00 | 3198.25 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-01-29 10:15:00 | 3045.95 | 2024-03-12 10:15:00 | 3170.00 | STOP_HIT | 0.50 | 4.07% |
| BUY | retest2 | 2024-03-28 11:15:00 | 3156.40 | 2024-05-06 09:15:00 | 2861.15 | STOP_HIT | 1.00 | -9.35% |
| SELL | retest2 | 2024-05-16 10:15:00 | 3014.10 | 2024-05-21 11:15:00 | 3145.95 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest1 | 2024-07-29 13:15:00 | 3082.30 | 2024-07-30 11:15:00 | 3119.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-08-05 11:15:00 | 3046.05 | 2024-08-06 09:15:00 | 3127.25 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-08-06 15:15:00 | 3072.70 | 2024-08-07 09:15:00 | 3137.45 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-08-14 09:15:00 | 3070.50 | 2024-08-19 09:15:00 | 3122.80 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-08-20 12:15:00 | 3073.45 | 2024-08-21 09:15:00 | 3124.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-08-26 11:15:00 | 3070.50 | 2024-09-27 09:15:00 | 3131.95 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-08-27 14:15:00 | 3068.00 | 2024-09-27 09:15:00 | 3131.95 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-10-07 10:15:00 | 3024.75 | 2024-10-08 12:15:00 | 3144.00 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest1 | 2024-10-31 11:15:00 | 2961.00 | 2024-11-06 11:15:00 | 3022.65 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-11-07 10:15:00 | 2965.10 | 2024-11-21 09:15:00 | 2668.59 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-07 14:15:00 | 2969.75 | 2024-11-21 09:15:00 | 2672.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-06-04 13:15:00 | 2495.00 | 2025-06-10 11:15:00 | 2619.75 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-04 13:15:00 | 2495.00 | 2025-06-13 10:15:00 | 2497.60 | STOP_HIT | 0.50 | 0.10% |
| BUY | retest2 | 2025-06-23 11:15:00 | 2479.50 | 2025-07-31 14:15:00 | 2428.60 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest1 | 2026-01-06 14:15:00 | 2258.90 | 2026-01-09 14:15:00 | 2145.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-08 10:15:00 | 2234.90 | 2026-01-12 09:15:00 | 2123.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-06 14:15:00 | 2258.90 | 2026-01-21 10:15:00 | 2033.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-01-08 10:15:00 | 2234.90 | 2026-01-21 10:15:00 | 2011.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 10:15:00 | 2157.10 | 2026-02-17 12:15:00 | 2234.20 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2026-02-19 15:15:00 | 2156.70 | 2026-03-19 12:15:00 | 1941.03 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2093.00 | 2026-03-23 09:15:00 | 1883.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-15 14:15:00 | 2144.00 | 2026-04-20 14:15:00 | 2233.80 | STOP_HIT | 1.00 | -4.19% |
