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
| ALERT3 | 5 |
| PENDING | 12 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / Stop hits / Partials:** 0 / 9 / 1
- **Avg / median % per leg:** 1.01% / -1.02%
- **Sum % (uncompounded):** 10.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 0 | 9 | 1 | 1.01% | 10.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 3 | 30.0% | 0 | 9 | 1 | 1.01% | 10.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 3 | 30.0% | 0 | 9 | 1 | 1.01% | 10.1% |

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
| CROSSOVER_SKIP | 2024-01-25 15:15:00 | 2949.10 | 3222.40 | 3223.49 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 2 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 2924.45 | 2898.37 | 2898.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 2942.55 | 2899.08 | 2898.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 2898.75 | 2903.89 | 2901.18 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 2898.75 | 2903.89 | 2901.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 2898.75 | 2903.89 | 2901.18 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-07-10 13:15:00 | 2995.00 | 2905.39 | 2902.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:15:00 | 2996.30 | 2906.29 | 2902.63 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-18 09:15:00 | 2890.70 | 2927.26 | 2914.51 | SL hit qty=1.00 sl=2890.70 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-19 09:15:00 | 2950.40 | 2927.29 | 2914.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 10:15:00 | 2959.10 | 2927.60 | 2915.18 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 2890.70 | 2928.82 | 2916.78 | SL hit qty=1.00 sl=2890.70 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-26 09:15:00 | 2939.90 | 2925.27 | 2915.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 2943.55 | 2925.46 | 2916.10 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-09-12 09:15:00 | 3385.08 | 3154.88 | 3078.22 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2024-10-23 14:15:00 | 2987.40 | 3120.09 | 3120.21 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Stop hit — per-position SL triggered | 2024-10-29 10:15:00 | 2943.55 | 3091.23 | 3104.89 | SL hit qty=0.50 sl=2943.55 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 2442.00 | 2322.15 | 2321.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 2453.80 | 2323.46 | 2322.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 2378.70 | 2391.07 | 2364.48 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 14:15:00 | 2332.30 | 2390.27 | 2364.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 2332.30 | 2390.27 | 2364.74 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-05-26 10:15:00 | 2319.50 | 2348.46 | 2348.56 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-07-02 11:15:00 | 2389.90 | 2294.77 | 2305.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 2390.80 | 2295.73 | 2306.40 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-04 15:15:00 | 2424.20 | 2316.55 | 2316.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 15:15:00 | 2424.20 | 2316.55 | 2316.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 2448.00 | 2317.85 | 2317.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 2369.70 | 2371.84 | 2350.24 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-21 15:15:00 | 2378.80 | 2371.90 | 2350.59 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-22 09:15:00 | 2367.80 | 2371.86 | 2350.68 | ENTRY1 sustain failed after 1080m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 11:15:00 | 2344.00 | 2370.65 | 2351.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 2344.00 | 2370.65 | 2351.68 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-29 13:15:00 | 2398.20 | 2366.20 | 2351.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:15:00 | 2399.80 | 2366.54 | 2351.61 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-26 15:15:00 | 2341.00 | 2485.11 | 2464.83 | SL hit qty=1.00 sl=2341.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-06 12:15:00 | 2346.40 | 2446.58 | 2447.02 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-10-07 12:15:00 | 2369.90 | 2440.53 | 2443.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:15:00 | 2368.80 | 2439.82 | 2443.56 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 2341.00 | 2436.11 | 2441.61 | SL hit qty=1.00 sl=2341.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-15 12:15:00 | 2373.60 | 2406.22 | 2424.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 13:15:00 | 2378.70 | 2405.95 | 2424.09 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-27 15:15:00 | 2518.80 | 2438.26 | 2437.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

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
| CROSSOVER_SKIP | 2026-01-30 11:15:00 | 2431.50 | 2720.59 | 2721.19 | slope filter: EMA200 not falling 0.50% over 350 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-16 10:15:00 | 3300.40 | 2024-01-17 14:15:00 | 3244.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-07-10 14:15:00 | 2996.30 | 2024-07-18 09:15:00 | 2890.70 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2024-07-19 10:15:00 | 2959.10 | 2024-07-23 12:15:00 | 2890.70 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-07-26 10:15:00 | 2943.55 | 2024-09-12 09:15:00 | 3385.08 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-07-26 10:15:00 | 2943.55 | 2024-10-29 10:15:00 | 2943.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2025-07-02 12:15:00 | 2390.80 | 2025-07-04 15:15:00 | 2424.20 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2025-07-29 14:15:00 | 2399.80 | 2025-09-26 15:15:00 | 2341.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-10-07 13:15:00 | 2368.80 | 2025-10-08 10:15:00 | 2341.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-10-15 13:15:00 | 2378.70 | 2025-10-27 15:15:00 | 2518.80 | STOP_HIT | 1.00 | 5.89% |
| BUY | retest2 | 2026-01-19 10:15:00 | 2779.20 | 2026-01-19 15:15:00 | 2750.90 | STOP_HIT | 1.00 | -1.02% |
