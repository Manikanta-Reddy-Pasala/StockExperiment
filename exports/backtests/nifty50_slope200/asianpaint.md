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
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 5 |
| ALERT3 | 7 |
| PENDING | 22 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 17
- **Target hits / Stop hits / Partials:** 0 / 17 / 0
- **Avg / median % per leg:** -2.24% / -1.69%
- **Sum % (uncompounded):** -38.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.13% | -10.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.13% | -10.6% |
| SELL (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.28% | -27.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.28% | -27.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 0 | 0.0% | 0 | 17 | 0 | -2.24% | -38.0% |

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
| CROSSOVER_SKIP | 2024-01-25 15:15:00 | 2949.10 | 3222.40 | 3223.49 | slope filter: EMA200 not falling 2.00% over 1400 bars |
| CROSSOVER_SKIP | 2024-07-03 14:15:00 | 2924.45 | 2898.37 | 2898.29 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2024-09-10 11:15:00 | 3302.95 | 3130.54 | 3061.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 12:15:00 | 3308.50 | 3132.31 | 3062.93 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-11 09:15:00 | 3361.00 | 3139.45 | 3067.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 10:15:00 | 3372.10 | 3141.76 | 3069.42 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-19 09:15:00 | 3305.70 | 3206.04 | 3118.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 3318.00 | 3207.15 | 3119.58 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 3244.50 | 3225.25 | 3139.54 | SL hit qty=1.00 sl=3244.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 3244.50 | 3225.25 | 3139.54 | SL hit qty=1.00 sl=3244.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 3244.50 | 3225.25 | 3139.54 | SL hit qty=1.00 sl=3244.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 3148.55 | 3241.24 | 3163.04 | EMA400 retest candle locked |

### Cycle 2 — SELL (started 2024-10-23 14:15:00)

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
| ALERT3_SKIP | 2025-05-19 09:15:00 | 2355.70 | 2363.17 | 2355.22 | max_alert3_locks_per_cycle=2 reached — end cycle |

### Cycle 3 — SELL (started 2025-05-26 10:15:00)

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

### Cycle 4 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 2346.40 | 2446.58 | 2447.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 2336.00 | 2435.12 | 2441.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 2408.40 | 2404.31 | 2422.54 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 2514.70 | 2405.47 | 2422.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2514.70 | 2405.47 | 2422.94 | EMA400 retest candle locked |

### Cycle 5 — BUY (started 2025-10-27 15:15:00)

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
| CROSSOVER_SKIP | 2026-01-30 11:15:00 | 2431.50 | 2720.59 | 2721.19 | slope filter: EMA200 not falling 2.00% over 1400 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-16 10:15:00 | 3300.40 | 2024-01-17 14:15:00 | 3244.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-09-10 12:15:00 | 3308.50 | 2024-09-24 14:15:00 | 3244.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-09-11 10:15:00 | 3372.10 | 2024-09-24 14:15:00 | 3244.50 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2024-09-19 10:15:00 | 3318.00 | 2024-09-24 14:15:00 | 3244.50 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-02-05 10:15:00 | 2267.40 | 2025-04-03 09:15:00 | 2318.55 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-03-26 12:15:00 | 2311.30 | 2025-04-08 13:15:00 | 2417.10 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2025-03-27 09:15:00 | 2313.30 | 2025-04-08 13:15:00 | 2417.10 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-04-01 14:15:00 | 2316.35 | 2025-04-08 13:15:00 | 2417.10 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2025-04-02 10:15:00 | 2304.00 | 2025-04-08 13:15:00 | 2417.10 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2025-05-08 15:15:00 | 2300.90 | 2025-05-09 09:15:00 | 2318.55 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-05-09 12:15:00 | 2295.40 | 2025-05-12 09:15:00 | 2318.55 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-05-14 10:15:00 | 2294.80 | 2025-05-15 13:15:00 | 2318.55 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-05-27 14:15:00 | 2326.10 | 2025-06-27 13:15:00 | 2351.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-30 12:15:00 | 2328.00 | 2025-07-01 09:15:00 | 2351.60 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-25 12:15:00 | 2332.60 | 2025-07-28 10:15:00 | 2351.60 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-10-01 10:15:00 | 2327.40 | 2025-10-03 09:15:00 | 2351.60 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-01-19 10:15:00 | 2779.20 | 2026-01-19 15:15:00 | 2750.90 | STOP_HIT | 1.00 | -1.02% |
