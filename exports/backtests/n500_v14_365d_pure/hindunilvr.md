# Hindustan Unilever Ltd. (HINDUNILVR)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 2286.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 20
- **Target hits / Stop hits / Partials:** 1 / 25 / 6
- **Avg / median % per leg:** 0.83% / -0.76%
- **Sum % (uncompounded):** 26.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.78% | -8.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.78% | -8.9% |
| SELL (all) | 27 | 12 | 44.4% | 1 | 20 | 6 | 1.31% | 35.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 12 | 44.4% | 1 | 20 | 6 | 1.31% | 35.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 12 | 37.5% | 1 | 25 | 6 | 0.83% | 26.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 2240.31 | 2298.59 | 2298.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 2232.74 | 2297.93 | 2298.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 2283.10 | 2282.41 | 2289.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 12:00:00 | 2283.10 | 2282.41 | 2289.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 2283.20 | 2282.30 | 2289.36 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 2391.99 | 2296.04 | 2295.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 2475.01 | 2306.18 | 2300.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 2368.88 | 2375.75 | 2344.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 12:00:00 | 2368.88 | 2375.75 | 2344.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 2494.20 | 2546.03 | 2499.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 2494.20 | 2546.03 | 2499.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 2502.16 | 2545.60 | 2499.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 2497.54 | 2545.60 | 2499.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 2493.70 | 2545.08 | 2499.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 2493.70 | 2545.08 | 2499.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 2484.75 | 2544.48 | 2499.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 2484.75 | 2544.48 | 2499.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 2524.10 | 2542.07 | 2499.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:15:00 | 2528.03 | 2542.07 | 2499.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 2531.67 | 2541.95 | 2499.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 2473.44 | 2538.13 | 2499.75 | SL hit (close<static) qty=1.00 sl=2496.46 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 2473.44 | 2538.13 | 2499.75 | SL hit (close<static) qty=1.00 sl=2496.46 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 2530.20 | 2500.07 | 2490.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 2469.01 | 2512.61 | 2498.59 | SL hit (close<static) qty=1.00 sl=2496.46 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 13:15:00 | 2416.78 | 2487.39 | 2487.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 2409.99 | 2485.29 | 2486.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 2418.95 | 2418.52 | 2443.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 12:00:00 | 2418.95 | 2418.52 | 2443.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2437.54 | 2418.77 | 2442.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 2440.88 | 2418.77 | 2442.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 2445.41 | 2419.03 | 2442.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 2445.41 | 2419.03 | 2442.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 2443.64 | 2419.28 | 2442.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 2443.64 | 2419.28 | 2442.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 2428.19 | 2419.37 | 2441.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:00:00 | 2421.80 | 2419.54 | 2441.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 2370.06 | 2420.45 | 2441.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 2421.99 | 2420.48 | 2441.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:15:00 | 2423.96 | 2420.53 | 2441.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 2422.29 | 2420.21 | 2439.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 2433.60 | 2420.21 | 2439.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 2361.60 | 2419.67 | 2439.52 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 2300.71 | 2419.67 | 2439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 2300.89 | 2419.67 | 2439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 2302.76 | 2419.67 | 2439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 2462.19 | 2419.67 | 2439.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 09:15:00 | 2251.56 | 2388.13 | 2419.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 2335.20 | 2327.67 | 2369.08 | SL hit (close>ema200) qty=0.50 sl=2327.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 2335.20 | 2327.67 | 2369.08 | SL hit (close>ema200) qty=0.50 sl=2327.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 2335.20 | 2327.67 | 2369.08 | SL hit (close>ema200) qty=0.50 sl=2327.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 2335.20 | 2327.67 | 2369.08 | SL hit (close>ema200) qty=0.50 sl=2327.67 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 2378.30 | 2329.31 | 2368.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 2378.30 | 2329.31 | 2368.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 2391.60 | 2329.93 | 2368.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 2389.10 | 2329.93 | 2368.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2382.60 | 2333.46 | 2369.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 2381.50 | 2333.46 | 2369.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2410.80 | 2343.02 | 2371.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 2410.80 | 2343.02 | 2371.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 2372.50 | 2345.99 | 2372.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 2376.80 | 2345.99 | 2372.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 2371.40 | 2346.25 | 2372.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:30:00 | 2377.00 | 2346.25 | 2372.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 2370.70 | 2346.49 | 2372.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:30:00 | 2362.90 | 2346.66 | 2372.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 2381.00 | 2347.57 | 2372.17 | SL hit (close>static) qty=1.00 sl=2380.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:15:00 | 2365.30 | 2353.02 | 2373.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:30:00 | 2365.00 | 2353.20 | 2372.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 2358.00 | 2353.34 | 2372.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2382.90 | 2353.91 | 2372.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2382.90 | 2353.91 | 2372.42 | SL hit (close>static) qty=1.00 sl=2380.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2382.90 | 2353.91 | 2372.42 | SL hit (close>static) qty=1.00 sl=2380.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2382.90 | 2353.91 | 2372.42 | SL hit (close>static) qty=1.00 sl=2380.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:30:00 | 2347.40 | 2368.86 | 2377.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:30:00 | 2352.40 | 2368.54 | 2376.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 2344.60 | 2368.81 | 2376.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:00:00 | 2352.00 | 2367.12 | 2375.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 2372.10 | 2366.85 | 2375.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 2372.10 | 2366.85 | 2375.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 2370.60 | 2366.65 | 2375.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:45:00 | 2374.00 | 2366.65 | 2375.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 2365.00 | 2366.63 | 2375.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 10:00:00 | 2364.00 | 2366.61 | 2375.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:00:00 | 2347.30 | 2366.46 | 2375.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 2382.40 | 2363.90 | 2373.09 | SL hit (close>static) qty=1.00 sl=2381.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 2382.40 | 2363.90 | 2373.09 | SL hit (close>static) qty=1.00 sl=2381.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 15:00:00 | 2364.80 | 2364.29 | 2373.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 11:30:00 | 2362.50 | 2364.52 | 2373.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 2371.70 | 2364.71 | 2373.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 2408.80 | 2365.21 | 2373.23 | SL hit (close>static) qty=1.00 sl=2401.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 2408.80 | 2365.21 | 2373.23 | SL hit (close>static) qty=1.00 sl=2401.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 2408.80 | 2365.21 | 2373.23 | SL hit (close>static) qty=1.00 sl=2401.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 2408.80 | 2365.21 | 2373.23 | SL hit (close>static) qty=1.00 sl=2401.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 2408.80 | 2365.21 | 2373.23 | SL hit (close>static) qty=1.00 sl=2381.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 2408.80 | 2365.21 | 2373.23 | SL hit (close>static) qty=1.00 sl=2381.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 13:00:00 | 2367.90 | 2365.55 | 2373.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 12:15:00 | 2381.50 | 2365.22 | 2372.84 | SL hit (close>static) qty=1.00 sl=2377.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 2465.00 | 2379.87 | 2379.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 2466.90 | 2382.31 | 2380.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 2376.70 | 2382.25 | 2380.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 2414.50 | 2382.57 | 2381.10 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 2296.30 | 2379.30 | 2379.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 2283.40 | 2364.72 | 2371.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 11:15:00 | 2360.10 | 2358.38 | 2367.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 12:00:00 | 2360.10 | 2358.38 | 2367.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 2367.80 | 2358.47 | 2367.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 2367.80 | 2358.47 | 2367.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 2363.00 | 2358.52 | 2367.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:00:00 | 2363.00 | 2358.52 | 2367.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 2367.10 | 2358.65 | 2367.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:45:00 | 2358.70 | 2358.66 | 2367.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:45:00 | 2357.30 | 2359.10 | 2367.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 14:15:00 | 2383.60 | 2359.83 | 2367.74 | SL hit (close>static) qty=1.00 sl=2379.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 14:15:00 | 2383.60 | 2359.83 | 2367.74 | SL hit (close>static) qty=1.00 sl=2379.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 2356.30 | 2360.03 | 2367.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:15:00 | 2238.49 | 2350.04 | 2361.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-12 12:15:00 | 2120.67 | 2299.81 | 2332.61 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:30:00 | 2358.00 | 2192.74 | 2227.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 2240.10 | 2229.75 | 2242.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 2230.20 | 2229.75 | 2242.34 | SL hit (close>static) qty=0.50 sl=2229.75 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 2248.20 | 2229.94 | 2242.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 2248.20 | 2229.94 | 2242.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 2249.40 | 2230.13 | 2242.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:45:00 | 2253.20 | 2230.13 | 2242.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 2260.00 | 2230.43 | 2242.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 2260.00 | 2230.43 | 2242.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 2246.50 | 2230.59 | 2242.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 2261.00 | 2230.59 | 2242.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-22 11:45:00 | 2301.99 | 2025-05-22 12:15:00 | 2287.23 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-05-23 09:15:00 | 2304.25 | 2025-06-13 09:15:00 | 2272.38 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-24 12:15:00 | 2528.03 | 2025-09-26 09:15:00 | 2473.44 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-09-24 12:45:00 | 2531.67 | 2025-09-26 09:15:00 | 2473.44 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-10-17 09:15:00 | 2530.20 | 2025-10-24 09:15:00 | 2469.01 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-12-01 15:00:00 | 2421.80 | 2025-12-05 09:15:00 | 2300.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 2370.06 | 2025-12-05 09:15:00 | 2300.89 | PARTIAL | 0.50 | 2.92% |
| SELL | retest2 | 2025-12-03 10:30:00 | 2421.99 | 2025-12-05 09:15:00 | 2302.76 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-12-03 12:15:00 | 2423.96 | 2025-12-12 09:15:00 | 2251.56 | PARTIAL | 0.50 | 7.11% |
| SELL | retest2 | 2025-12-01 15:00:00 | 2421.80 | 2026-01-02 09:15:00 | 2335.20 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-12-03 09:15:00 | 2370.06 | 2026-01-02 09:15:00 | 2335.20 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2025-12-03 10:30:00 | 2421.99 | 2026-01-02 09:15:00 | 2335.20 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-12-03 12:15:00 | 2423.96 | 2026-01-02 09:15:00 | 2335.20 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2026-01-09 13:30:00 | 2362.90 | 2026-01-12 09:15:00 | 2381.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-01-14 10:15:00 | 2365.30 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-16 10:30:00 | 2365.00 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-16 12:15:00 | 2358.00 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-28 10:30:00 | 2347.40 | 2026-02-03 11:15:00 | 2382.40 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-28 13:30:00 | 2352.40 | 2026-02-03 11:15:00 | 2382.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-01-29 09:15:00 | 2344.60 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-01-29 14:00:00 | 2352.00 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-02-01 10:00:00 | 2364.00 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-02-01 12:00:00 | 2347.30 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2026-02-03 15:00:00 | 2364.80 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-02-04 11:30:00 | 2362.50 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-02-05 13:00:00 | 2367.90 | 2026-02-06 12:15:00 | 2381.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-02-25 10:45:00 | 2358.70 | 2026-02-26 14:15:00 | 2383.60 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-26 09:45:00 | 2357.30 | 2026-02-26 14:15:00 | 2383.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-27 09:15:00 | 2356.30 | 2026-03-05 09:15:00 | 2238.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 2356.30 | 2026-03-12 12:15:00 | 2120.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-23 11:30:00 | 2358.00 | 2026-04-30 10:15:00 | 2240.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 11:30:00 | 2358.00 | 2026-04-30 10:15:00 | 2230.20 | STOP_HIT | 0.50 | 5.42% |
