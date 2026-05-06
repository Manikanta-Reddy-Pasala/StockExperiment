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
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 12 |
| PENDING | 37 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 10 |
| ENTRY2 | 20 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 13 / 19
- **Target hits / Stop hits / Partials:** 0 / 25 / 7
- **Avg / median % per leg:** 2.85% / -0.79%
- **Sum % (uncompounded):** 91.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 6 | 37.5% | 0 | 14 | 2 | 1.54% | 24.7% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 0 | 5 | 2 | 5.79% | 40.6% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.77% | -15.9% |
| SELL (all) | 16 | 7 | 43.8% | 0 | 11 | 5 | 4.17% | 66.7% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 5 | 0 | 0.22% | 1.1% |
| SELL @ 3rd Alert (retest2) | 11 | 5 | 45.5% | 0 | 6 | 5 | 5.96% | 65.6% |
| retest1 (combined) | 12 | 8 | 66.7% | 0 | 10 | 2 | 3.47% | 41.7% |
| retest2 (combined) | 20 | 5 | 25.0% | 0 | 15 | 5 | 2.48% | 49.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 14:15:00 | 2566.65 | 2524.61 | 2524.50 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 09:15:00 | 2513.00 | 2524.34 | 2524.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 12:15:00 | 2503.90 | 2523.86 | 2524.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 14:15:00 | 2524.00 | 2519.96 | 2521.98 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 14:15:00 | 2524.00 | 2519.96 | 2521.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 2524.00 | 2519.96 | 2521.98 | EMA400 retest candle locked |

### Cycle 3 — BUY (started 2023-12-20 09:15:00)

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

### Cycle 4 — SELL (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 11:15:00 | 2457.00 | 2543.39 | 2543.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 12:15:00 | 2440.00 | 2542.36 | 2543.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 2436.35 | 2430.06 | 2465.98 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-03-05 09:15:00 | 2399.85 | 2428.65 | 2462.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 10:15:00 | 2397.05 | 2428.33 | 2462.49 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-11 11:15:00 | 2402.75 | 2423.57 | 2456.40 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 12:15:00 | 2402.30 | 2423.35 | 2456.13 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 2355.00 | 2260.49 | 2311.07 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 2311.07 | 2260.49 | 2311.07 | SL hit qty=1.00 sl=2311.07 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 2311.07 | 2260.49 | 2311.07 | SL hit qty=1.00 sl=2311.07 alert=retest1 |

### Cycle 5 — BUY (started 2024-06-04 11:15:00)

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
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 2527.95 | 2766.03 | 2766.38 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest1 |

### Cycle 6 — SELL (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 14:15:00 | 2527.95 | 2766.03 | 2766.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2519.15 | 2710.56 | 2736.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 13:15:00 | 2401.60 | 2398.37 | 2477.33 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-06 09:15:00 | 2383.75 | 2398.36 | 2476.15 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:15:00 | 2370.80 | 2398.09 | 2475.63 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-08 09:15:00 | 2383.65 | 2396.38 | 2469.86 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-08 10:15:00 | 2389.65 | 2396.31 | 2469.46 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 2470.00 | 2397.59 | 2467.59 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-09 10:15:00 | 2467.59 | 2397.59 | 2467.59 | SL hit qty=1.00 sl=2467.59 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-09 13:15:00 | 2434.85 | 2398.99 | 2467.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:15:00 | 2435.55 | 2399.35 | 2467.09 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-10 11:15:00 | 2437.65 | 2400.86 | 2466.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-10 12:15:00 | 2449.85 | 2401.35 | 2466.42 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-10 14:15:00 | 2439.35 | 2402.19 | 2466.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 15:15:00 | 2440.00 | 2402.57 | 2466.07 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-13 10:15:00 | 2436.00 | 2403.38 | 2465.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 11:15:00 | 2431.85 | 2403.66 | 2465.68 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-13 15:15:00 | 2446.00 | 2405.08 | 2465.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 09:15:00 | 2420.20 | 2405.23 | 2464.94 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 2429.40 | 2384.23 | 2429.60 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 2472.30 | 2388.31 | 2430.33 | SL hit qty=1.00 sl=2472.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 2472.30 | 2388.31 | 2430.33 | SL hit qty=1.00 sl=2472.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 2472.30 | 2388.31 | 2430.33 | SL hit qty=1.00 sl=2472.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 2472.30 | 2388.31 | 2430.33 | SL hit qty=1.00 sl=2472.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-05 13:15:00 | 2406.20 | 2395.57 | 2430.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 14:15:00 | 2390.40 | 2395.52 | 2430.15 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-24 09:15:00 | 2439.90 | 2304.08 | 2309.17 | SL hit qty=1.00 sl=2439.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-24 10:15:00 | 2332.00 | 2304.36 | 2309.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 11:15:00 | 2331.10 | 2304.63 | 2309.39 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 2347.20 | 2313.33 | 2313.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-05 13:15:00)

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

### Cycle 8 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 2278.10 | 2336.79 | 2336.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 2269.80 | 2336.12 | 2336.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 2321.00 | 2320.27 | 2327.52 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-03 14:15:00 | 2314.20 | 2320.21 | 2327.38 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 15:15:00 | 2314.80 | 2320.16 | 2327.32 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 2321.00 | 2320.17 | 2327.29 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 2327.29 | 2320.17 | 2327.29 | SL hit qty=1.00 sl=2327.29 alert=retest1 |

### Cycle 9 — BUY (started 2025-07-09 11:15:00)

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

### Cycle 10 — SELL (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 13:15:00 | 2456.10 | 2528.67 | 2528.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 2449.90 | 2526.54 | 2527.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 2459.10 | 2458.64 | 2483.89 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-11-27 12:15:00 | 2451.70 | 2458.57 | 2483.73 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 13:15:00 | 2442.20 | 2458.41 | 2483.53 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2478.00 | 2458.82 | 2482.51 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 2482.51 | 2458.82 | 2482.51 | SL hit qty=1.00 sl=2482.51 alert=retest1 |
| Cross detected — sustain check pending | 2025-12-01 14:15:00 | 2461.90 | 2459.60 | 2482.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 2459.00 | 2459.60 | 2482.20 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 09:15:00 | 2454.90 | 2460.48 | 2481.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-03 10:15:00 | 2469.50 | 2460.57 | 2481.71 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 12:15:00 | 2463.50 | 2460.65 | 2481.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-03 13:15:00 | 2466.00 | 2460.70 | 2481.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 14:15:00 | 2449.80 | 2460.59 | 2481.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 2448.00 | 2460.47 | 2481.14 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-04 14:15:00 | 2462.50 | 2460.36 | 2480.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 2464.10 | 2460.40 | 2480.39 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-11 11:15:00 | 2463.00 | 2380.56 | 2387.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:15:00 | 2460.10 | 2381.36 | 2388.19 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 2465.00 | 2382.19 | 2388.57 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-12 10:15:00 | 2376.70 | 2384.51 | 2389.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 2414.50 | 2384.81 | 2389.75 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-19 12:15:00 | 2094.49 | 2255.57 | 2306.88 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-19 12:15:00 | 2091.09 | 2255.57 | 2306.88 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-19 13:15:00 | 2090.15 | 2253.83 | 2305.75 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-19 13:15:00 | 2080.80 | 2253.83 | 2305.75 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-23 09:15:00 | 2052.32 | 2237.26 | 2294.72 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-08 15:15:00 | 2579.00 | 2024-01-11 13:15:00 | 2554.84 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-01-15 14:15:00 | 2572.65 | 2024-01-17 10:15:00 | 2544.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-01-17 14:15:00 | 2564.15 | 2024-01-18 09:15:00 | 2544.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-01-19 14:15:00 | 2567.65 | 2024-01-23 09:15:00 | 2544.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest1 | 2024-03-05 10:15:00 | 2397.05 | 2024-05-07 09:15:00 | 2311.07 | STOP_HIT | 1.00 | 3.59% |
| SELL | retest1 | 2024-03-11 12:15:00 | 2402.30 | 2024-05-07 09:15:00 | 2311.07 | STOP_HIT | 1.00 | 3.80% |
| BUY | retest1 | 2024-06-24 09:15:00 | 2445.50 | 2024-08-23 13:15:00 | 2812.32 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2024-06-26 10:15:00 | 2455.00 | 2024-08-26 10:15:00 | 2823.25 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2024-06-24 09:15:00 | 2445.50 | 2024-10-24 11:15:00 | 2455.00 | STOP_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-06-26 10:15:00 | 2455.00 | 2024-10-25 14:15:00 | 2527.95 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest1 | 2025-01-06 10:15:00 | 2370.80 | 2025-01-09 10:15:00 | 2467.59 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2025-01-09 14:15:00 | 2435.55 | 2025-01-31 15:15:00 | 2472.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-01-10 15:15:00 | 2440.00 | 2025-01-31 15:15:00 | 2472.30 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-01-13 11:15:00 | 2431.85 | 2025-01-31 15:15:00 | 2472.30 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-01-14 09:15:00 | 2420.20 | 2025-01-31 15:15:00 | 2472.30 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-02-05 14:15:00 | 2390.40 | 2025-04-24 09:15:00 | 2439.90 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-04-24 11:15:00 | 2331.10 | 2025-05-05 13:15:00 | 2347.20 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-05-12 10:15:00 | 2378.90 | 2025-06-13 09:15:00 | 2302.90 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-05-14 15:15:00 | 2351.50 | 2025-06-13 09:15:00 | 2302.90 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-05-21 10:15:00 | 2369.20 | 2025-06-13 09:15:00 | 2302.90 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-05-23 10:15:00 | 2361.00 | 2025-06-13 09:15:00 | 2302.90 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest1 | 2025-07-03 15:15:00 | 2314.80 | 2025-07-04 09:15:00 | 2327.29 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-07-28 10:15:00 | 2443.00 | 2025-09-23 10:15:00 | 2541.03 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest1 | 2025-07-30 13:15:00 | 2440.40 | 2025-09-23 10:15:00 | 2541.03 | STOP_HIT | 1.00 | 4.12% |
| BUY | retest2 | 2025-09-24 12:15:00 | 2572.40 | 2025-09-25 14:15:00 | 2533.30 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-10-16 15:15:00 | 2560.00 | 2025-10-24 09:15:00 | 2533.30 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest1 | 2025-11-27 13:15:00 | 2442.20 | 2025-12-01 09:15:00 | 2482.51 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-12-01 15:15:00 | 2459.00 | 2026-03-19 12:15:00 | 2094.49 | PARTIAL | 0.50 | 14.82% |
| SELL | retest2 | 2025-12-03 15:15:00 | 2448.00 | 2026-03-19 12:15:00 | 2091.09 | PARTIAL | 0.50 | 14.58% |
| SELL | retest2 | 2025-12-04 15:15:00 | 2464.10 | 2026-03-19 13:15:00 | 2090.15 | PARTIAL | 0.50 | 15.18% |
| SELL | retest2 | 2026-02-11 12:15:00 | 2460.10 | 2026-03-19 13:15:00 | 2080.80 | PARTIAL | 0.50 | 15.42% |
| SELL | retest2 | 2026-02-12 11:15:00 | 2414.50 | 2026-03-23 09:15:00 | 2052.32 | PARTIAL | 0.50 | 15.00% |
