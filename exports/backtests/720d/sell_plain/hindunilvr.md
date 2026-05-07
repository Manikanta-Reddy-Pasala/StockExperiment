# HINDUNILVR (HINDUNILVR)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 2273.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 7 |
| PENDING | 36 |
| PENDING_CANCEL | 13 |
| ENTRY1 | 2 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 16
- **Target hits / Stop hits / Partials:** 0 / 21 / 0
- **Avg / median % per leg:** -1.21% / -1.57%
- **Sum % (uncompounded):** -25.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 21 | 5 | 23.8% | 0 | 21 | 0 | -1.21% | -25.5% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.61% | -5.2% |
| SELL @ 3rd Alert (retest2) | 19 | 5 | 26.3% | 0 | 19 | 0 | -1.07% | -20.3% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.61% | -5.2% |
| retest2 (combined) | 19 | 5 | 26.3% | 0 | 19 | 0 | -1.07% | -20.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 14:15:00 | 2485.93 | 2720.93 | 2721.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2476.39 | 2663.02 | 2690.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 13:15:00 | 2362.38 | 2359.08 | 2436.48 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-06 09:15:00 | 2344.83 | 2359.07 | 2435.32 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:15:00 | 2342.12 | 2358.64 | 2434.35 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-08 09:15:00 | 2345.37 | 2357.17 | 2429.18 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-08 10:15:00 | 2350.63 | 2357.10 | 2428.78 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 2429.86 | 2358.38 | 2426.96 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-09 10:15:00 | 2429.86 | 2358.38 | 2426.96 | SL hit (close>ema400) qty=1.00 sl=2426.96 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-09 13:15:00 | 2395.09 | 2359.74 | 2426.63 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 2400.16 | 2360.50 | 2426.34 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-10 11:15:00 | 2397.85 | 2361.63 | 2425.93 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-10 12:15:00 | 2410.54 | 2362.12 | 2425.85 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-10 14:15:00 | 2399.52 | 2362.96 | 2425.64 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-13 09:15:00 | 2410.98 | 2363.80 | 2425.44 | ENTRY2 sustain failed after 4020m |
| Cross detected — sustain check pending | 2025-01-13 10:15:00 | 2396.22 | 2364.12 | 2425.30 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 12:15:00 | 2384.96 | 2364.61 | 2424.93 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-13 15:15:00 | 2406.06 | 2365.80 | 2424.63 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 09:15:00 | 2380.68 | 2365.95 | 2424.41 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 2434.59 | 2349.37 | 2390.51 | SL hit (close>static) qty=1.00 sl=2433.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 2434.59 | 2349.37 | 2390.51 | SL hit (close>static) qty=1.00 sl=2433.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 2434.59 | 2349.37 | 2390.51 | SL hit (close>static) qty=1.00 sl=2433.26 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-03 11:15:00 | 2388.30 | 2359.43 | 2393.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 13:15:00 | 2393.12 | 2360.10 | 2393.67 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 2400.26 | 2360.50 | 2393.70 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-04 09:15:00 | 2375.66 | 2361.09 | 2393.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-04 10:15:00 | 2390.91 | 2361.39 | 2393.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-04 11:15:00 | 2381.52 | 2361.59 | 2393.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-04 12:15:00 | 2394.21 | 2361.91 | 2393.60 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-05 09:15:00 | 2372.52 | 2362.99 | 2393.51 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 11:15:00 | 2380.73 | 2363.29 | 2393.36 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-24 09:15:00 | 2418.85 | 2266.79 | 2271.98 | SL hit (close>static) qty=1.00 sl=2409.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-24 10:15:00 | 2294.71 | 2267.07 | 2272.10 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 12:15:00 | 2294.41 | 2267.59 | 2272.31 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 2240.31 | 2298.59 | 2298.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 2240.31 | 2298.59 | 2298.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 2240.31 | 2298.59 | 2298.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 2232.74 | 2297.93 | 2298.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 2283.10 | 2282.41 | 2289.59 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-03 14:15:00 | 2276.31 | 2282.35 | 2289.46 | ENTRY1 cross detected — sustain check pending (75m) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 2283.20 | 2282.30 | 2289.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 2283.20 | 2282.30 | 2289.36 | EMA400 retest candle locked |

### Cycle 3 — SELL (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 13:15:00 | 2416.78 | 2487.39 | 2487.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 2409.99 | 2485.29 | 2486.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 2418.95 | 2418.52 | 2443.35 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-11-27 12:15:00 | 2411.67 | 2418.46 | 2443.19 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 14:15:00 | 2410.09 | 2418.21 | 2442.83 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2437.54 | 2418.77 | 2442.02 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 2445.41 | 2419.03 | 2442.04 | SL hit (close>ema400) qty=1.00 sl=2442.04 alert=retest1 |
| Cross detected — sustain check pending | 2025-12-01 14:15:00 | 2421.80 | 2419.54 | 2441.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-02 09:15:00 | 2431.63 | 2419.65 | 2441.68 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2025-12-03 09:15:00 | 2414.81 | 2420.39 | 2441.30 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-03 10:15:00 | 2429.18 | 2420.48 | 2441.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-03 14:15:00 | 2409.80 | 2420.50 | 2440.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 2412.16 | 2420.23 | 2440.50 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Cross detected — sustain check pending | 2025-12-04 14:15:00 | 2422.29 | 2420.21 | 2439.99 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-04 15:15:00 | 2423.86 | 2420.25 | 2439.91 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-05 09:15:00 | 2361.60 | 2419.67 | 2439.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 2338.00 | 2418.18 | 2438.57 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-01-06 15:15:00 | 2421.00 | 2337.37 | 2370.17 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 2411.10 | 2338.10 | 2370.37 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-01-20 12:15:00 | 2421.40 | 2359.68 | 2374.46 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 14:15:00 | 2377.10 | 2360.38 | 2374.67 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 2382.00 | 2360.60 | 2374.71 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-21 13:15:00 | 2364.90 | 2361.61 | 2374.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-21 14:15:00 | 2368.90 | 2361.68 | 2374.84 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-21 15:15:00 | 2365.90 | 2361.72 | 2374.79 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-22 09:15:00 | 2381.20 | 2361.91 | 2374.83 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-01-22 12:15:00 | 2366.70 | 2362.30 | 2374.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-22 13:15:00 | 2369.60 | 2362.37 | 2374.80 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 09:15:00 | 2362.60 | 2368.85 | 2377.18 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-28 10:15:00 | 2370.10 | 2368.86 | 2377.14 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 11:15:00 | 2361.90 | 2368.79 | 2377.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 2348.90 | 2368.54 | 2376.86 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 2386.00 | 2368.81 | 2376.91 | SL hit (close>static) qty=1.00 sl=2384.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-29 09:15:00 | 2326.40 | 2368.39 | 2376.66 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 2330.00 | 2367.56 | 2376.16 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-01-30 10:15:00 | 2361.30 | 2366.79 | 2375.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 12:15:00 | 2362.70 | 2366.68 | 2375.37 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-01-30 15:15:00 | 2365.00 | 2366.63 | 2375.22 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 2364.00 | 2366.61 | 2375.16 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 2520m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 2371.70 | 2366.66 | 2375.15 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-01 11:15:00 | 2347.30 | 2366.46 | 2375.01 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:15:00 | 2348.60 | 2366.07 | 2374.73 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 2385.80 | 2364.28 | 2373.19 | SL hit (close>static) qty=1.00 sl=2384.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 2385.80 | 2364.28 | 2373.19 | SL hit (close>static) qty=1.00 sl=2384.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 2385.80 | 2364.28 | 2373.19 | SL hit (close>static) qty=1.00 sl=2384.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 2385.80 | 2364.28 | 2373.19 | SL hit (close>static) qty=1.00 sl=2384.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-05 13:15:00 | 2353.90 | 2365.44 | 2373.18 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 2355.00 | 2365.22 | 2373.00 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-06 13:15:00 | 2402.00 | 2365.59 | 2372.99 | SL hit (close>static) qty=1.00 sl=2384.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 2449.90 | 2371.47 | 2375.64 | SL hit (close>static) qty=1.00 sl=2447.37 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 2449.90 | 2371.47 | 2375.64 | SL hit (close>static) qty=1.00 sl=2447.37 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 2449.90 | 2371.47 | 2375.64 | SL hit (close>static) qty=1.00 sl=2447.37 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 2449.90 | 2371.47 | 2375.64 | SL hit (close>static) qty=1.00 sl=2447.37 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 2352.60 | 2383.44 | 2381.58 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 2352.50 | 2382.89 | 2381.32 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 2296.30 | 2379.30 | 2379.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 2296.30 | 2379.30 | 2379.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 2283.40 | 2364.72 | 2371.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 11:15:00 | 2360.10 | 2358.38 | 2367.76 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 12:15:00 | 2367.80 | 2358.47 | 2367.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 2367.80 | 2358.47 | 2367.76 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-25 11:15:00 | 2352.20 | 2358.60 | 2367.55 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-25 12:15:00 | 2359.50 | 2358.61 | 2367.51 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-27 09:15:00 | 2343.30 | 2359.86 | 2367.68 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 11:15:00 | 2347.10 | 2359.65 | 2367.50 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-27 14:15:00 | 2339.50 | 2359.44 | 2367.28 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 2315.30 | 2358.81 | 2366.88 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 4020m) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 2371.70 | 2177.39 | 2221.48 | SL hit (close>static) qty=1.00 sl=2369.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 2371.70 | 2177.39 | 2221.48 | SL hit (close>static) qty=1.00 sl=2369.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-23 15:15:00 | 2353.00 | 2199.62 | 2230.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 2343.40 | 2201.05 | 2230.93 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-01-06 11:15:00 | 2342.12 | 2025-01-09 10:15:00 | 2429.86 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-01-09 15:15:00 | 2400.16 | 2025-01-31 15:15:00 | 2434.59 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-01-13 12:15:00 | 2384.96 | 2025-01-31 15:15:00 | 2434.59 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-01-14 09:15:00 | 2380.68 | 2025-01-31 15:15:00 | 2434.59 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-02-03 13:15:00 | 2393.12 | 2025-04-24 09:15:00 | 2418.85 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-02-05 11:15:00 | 2380.73 | 2025-06-24 12:15:00 | 2240.31 | STOP_HIT | 1.00 | 5.90% |
| SELL | retest2 | 2025-04-24 12:15:00 | 2294.41 | 2025-06-24 12:15:00 | 2240.31 | STOP_HIT | 1.00 | 2.36% |
| SELL | retest1 | 2025-11-27 14:15:00 | 2410.09 | 2025-12-01 10:15:00 | 2445.41 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-12-04 09:15:00 | 2412.16 | 2026-01-28 15:15:00 | 2386.00 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2025-12-05 11:15:00 | 2338.00 | 2026-02-03 13:15:00 | 2385.80 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2026-01-07 09:15:00 | 2411.10 | 2026-02-03 13:15:00 | 2385.80 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2026-01-20 14:15:00 | 2377.10 | 2026-02-03 13:15:00 | 2385.80 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2026-01-28 13:15:00 | 2348.90 | 2026-02-03 13:15:00 | 2385.80 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-01-29 11:15:00 | 2330.00 | 2026-02-06 13:15:00 | 2402.00 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2026-01-30 12:15:00 | 2362.70 | 2026-02-10 09:15:00 | 2449.90 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2026-02-01 09:15:00 | 2364.00 | 2026-02-10 09:15:00 | 2449.90 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2026-02-01 13:15:00 | 2348.60 | 2026-02-10 09:15:00 | 2449.90 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2026-02-05 15:15:00 | 2355.00 | 2026-02-10 09:15:00 | 2449.90 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2026-02-13 11:15:00 | 2352.50 | 2026-02-16 09:15:00 | 2296.30 | STOP_HIT | 1.00 | 2.39% |
| SELL | retest2 | 2026-02-27 11:15:00 | 2347.10 | 2026-04-22 10:15:00 | 2371.70 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2315.30 | 2026-04-22 10:15:00 | 2371.70 | STOP_HIT | 1.00 | -2.44% |
