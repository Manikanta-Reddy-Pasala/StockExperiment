# HINDUNILVR (HINDUNILVR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 2317.10
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 5 |
| PENDING | 18 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 10 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 11
- **Target hits / Stop hits / Partials:** 0 / 15 / 2
- **Avg / median % per leg:** 1.33% / -0.92%
- **Sum % (uncompounded):** 22.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 6 | 35.3% | 0 | 15 | 2 | 1.33% | 22.6% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 0 | 5 | 2 | 5.31% | 37.2% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 0 | 10 | 0 | -1.46% | -14.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 0 | 5 | 2 | 5.31% | 37.2% |
| retest2 (combined) | 10 | 1 | 10.0% | 0 | 10 | 0 | -1.46% | -14.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 14:15:00 | 2566.65 | 2524.61 | 2524.50 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2023-12-11 09:15:00 | 2513.00 | 2524.34 | 2524.38 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 2 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 2561.15 | 2524.04 | 2523.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-21 13:15:00 | 2576.30 | 2527.95 | 2525.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 12:15:00 | 2566.55 | 2572.18 | 2552.63 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-08 14:15:00 | 2577.75 | 2572.23 | 2552.85 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 15:15:00 | 2579.00 | 2572.29 | 2552.98 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 2544.75 | 2572.46 | 2554.84 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-11 13:15:00 | 2554.84 | 2572.46 | 2554.84 | SL hit qty=1.00 sl=2554.84 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-15 13:15:00 | 2563.55 | 2568.58 | 2554.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 14:15:00 | 2572.65 | 2568.62 | 2554.11 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-17 10:15:00 | 2544.00 | 2568.77 | 2554.90 | SL hit qty=1.00 sl=2544.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-17 11:15:00 | 2560.60 | 2568.69 | 2554.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-17 12:15:00 | 2552.95 | 2568.54 | 2554.92 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-17 13:15:00 | 2559.75 | 2568.45 | 2554.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 14:15:00 | 2564.15 | 2568.41 | 2554.99 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 2544.00 | 2568.14 | 2554.99 | SL hit qty=1.00 sl=2544.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-01-19 13:15:00 | 2559.70 | 2566.26 | 2554.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 14:15:00 | 2567.65 | 2566.28 | 2554.79 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 2544.00 | 2564.99 | 2554.26 | SL hit qty=1.00 sl=2544.00 alert=retest2 |
| CROSSOVER_SKIP | 2024-01-25 11:15:00 | 2457.00 | 2543.39 | 2543.81 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 3 — BUY (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 11:15:00 | 2438.75 | 2334.12 | 2333.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 13:15:00 | 2497.05 | 2337.09 | 2335.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 11:15:00 | 2435.30 | 2436.74 | 2397.61 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-06-21 13:15:00 | 2445.50 | 2436.82 | 2398.04 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-21 14:15:00 | 2440.05 | 2436.85 | 2398.25 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-21 15:15:00 | 2443.50 | 2436.92 | 2398.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 09:15:00 | 2445.50 | 2437.00 | 2398.71 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 3960m) |
| Cross detected — sustain check pending | 2024-06-24 14:15:00 | 2443.15 | 2437.50 | 2399.90 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-24 15:15:00 | 2438.55 | 2437.51 | 2400.10 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-26 09:15:00 | 2446.05 | 2437.42 | 2401.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:15:00 | 2455.00 | 2437.59 | 2401.78 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-08-23 13:15:00 | 2812.32 | 2707.57 | 2628.21 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-08-26 10:15:00 | 2823.25 | 2711.94 | 2631.98 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 2815.05 | 2878.80 | 2804.14 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-24 11:15:00 | 2455.00 | 2792.55 | 2779.34 | SL hit qty=0.50 sl=2455.00 alert=retest1 |
| CROSSOVER_SKIP | 2024-10-25 14:15:00 | 2527.95 | 2766.03 | 2766.38 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 2445.50 | 2615.59 | 2676.53 | SL hit qty=0.50 sl=2445.50 alert=retest1 |

### Cycle 4 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 2347.20 | 2313.33 | 2313.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 2367.70 | 2314.53 | 2313.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2318.40 | 2324.04 | 2318.91 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 2318.40 | 2324.04 | 2318.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 2318.40 | 2324.04 | 2318.91 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 2385.50 | 2324.94 | 2319.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 2378.90 | 2325.48 | 2319.84 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-14 14:15:00 | 2353.20 | 2331.64 | 2323.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 2351.50 | 2331.84 | 2323.71 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-21 09:15:00 | 2365.90 | 2340.83 | 2329.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:15:00 | 2369.20 | 2341.11 | 2329.81 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-23 09:15:00 | 2350.50 | 2341.51 | 2330.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 2361.00 | 2341.70 | 2330.90 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 2346.00 | 2361.47 | 2347.15 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 2302.90 | 2360.03 | 2346.78 | SL hit qty=1.00 sl=2302.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 2302.90 | 2360.03 | 2346.78 | SL hit qty=1.00 sl=2302.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 2302.90 | 2360.03 | 2346.78 | SL hit qty=1.00 sl=2302.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 2302.90 | 2360.03 | 2346.78 | SL hit qty=1.00 sl=2302.90 alert=retest2 |
| CROSSOVER_SKIP | 2025-06-24 12:15:00 | 2278.10 | 2336.79 | 2336.95 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-07-07 09:15:00 | 2384.90 | 2321.51 | 2327.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:15:00 | 2400.00 | 2322.29 | 2328.08 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 2431.70 | 2334.16 | 2333.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 2431.70 | 2334.16 | 2333.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 2516.00 | 2344.48 | 2339.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 2408.50 | 2415.21 | 2383.41 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-28 09:15:00 | 2440.60 | 2415.45 | 2384.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:15:00 | 2443.00 | 2415.72 | 2384.60 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-30 12:15:00 | 2436.30 | 2418.83 | 2388.61 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 13:15:00 | 2440.40 | 2419.04 | 2388.87 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 2535.60 | 2588.37 | 2541.03 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 2541.03 | 2588.37 | 2541.03 | SL hit qty=1.00 sl=2541.03 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 2541.03 | 2588.37 | 2541.03 | SL hit qty=1.00 sl=2541.03 alert=retest1 |
| Cross detected — sustain check pending | 2025-09-24 11:15:00 | 2566.00 | 2584.31 | 2540.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:15:00 | 2572.40 | 2584.19 | 2540.97 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-25 14:15:00 | 2533.30 | 2581.28 | 2541.38 | SL hit qty=1.00 sl=2533.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-16 14:15:00 | 2562.30 | 2541.38 | 2532.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 2560.00 | 2541.56 | 2532.23 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 2533.30 | 2554.25 | 2540.03 | SL hit qty=1.00 sl=2533.30 alert=retest2 |
| CROSSOVER_SKIP | 2025-11-03 13:15:00 | 2456.10 | 2528.67 | 2528.99 | HTF filter: close above htf_sma |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-08 15:15:00 | 2579.00 | 2024-01-11 13:15:00 | 2554.84 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-01-15 14:15:00 | 2572.65 | 2024-01-17 10:15:00 | 2544.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-01-17 14:15:00 | 2564.15 | 2024-01-18 09:15:00 | 2544.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-01-19 14:15:00 | 2567.65 | 2024-01-23 09:15:00 | 2544.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest1 | 2024-06-24 09:15:00 | 2445.50 | 2024-08-23 13:15:00 | 2812.32 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2024-06-26 10:15:00 | 2455.00 | 2024-08-26 10:15:00 | 2823.25 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2024-06-24 09:15:00 | 2445.50 | 2024-10-24 11:15:00 | 2455.00 | STOP_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-06-26 10:15:00 | 2455.00 | 2024-11-14 09:15:00 | 2445.50 | STOP_HIT | 0.50 | -0.39% |
| BUY | retest2 | 2025-05-12 10:15:00 | 2378.90 | 2025-06-13 09:15:00 | 2302.90 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-05-14 15:15:00 | 2351.50 | 2025-06-13 09:15:00 | 2302.90 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-05-21 10:15:00 | 2369.20 | 2025-06-13 09:15:00 | 2302.90 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-05-23 10:15:00 | 2361.00 | 2025-06-13 09:15:00 | 2302.90 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-07-07 10:15:00 | 2400.00 | 2025-07-09 11:15:00 | 2431.70 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest1 | 2025-07-28 10:15:00 | 2443.00 | 2025-09-23 10:15:00 | 2541.03 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest1 | 2025-07-30 13:15:00 | 2440.40 | 2025-09-23 10:15:00 | 2541.03 | STOP_HIT | 1.00 | 4.12% |
| BUY | retest2 | 2025-09-24 12:15:00 | 2572.40 | 2025-09-25 14:15:00 | 2533.30 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-10-16 15:15:00 | 2560.00 | 2025-10-24 09:15:00 | 2533.30 | STOP_HIT | 1.00 | -1.04% |
