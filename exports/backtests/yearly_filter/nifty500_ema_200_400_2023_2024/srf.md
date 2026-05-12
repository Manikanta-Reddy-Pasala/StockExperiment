# SRF Ltd. (SRF)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2778.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 4 |
| ALERT3 | 59 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 57 |
| PARTIAL | 15 |
| TARGET_HIT | 10 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 34
- **Target hits / Stop hits / Partials:** 10 / 47 / 15
- **Avg / median % per leg:** 1.52% / 0.64%
- **Sum % (uncompounded):** 109.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 8 | 23.5% | 5 | 29 | 0 | -0.19% | -6.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 34 | 8 | 23.5% | 5 | 29 | 0 | -0.19% | -6.6% |
| SELL (all) | 38 | 30 | 78.9% | 5 | 18 | 15 | 3.05% | 116.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 30 | 78.9% | 5 | 18 | 15 | 3.05% | 116.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 38 | 52.8% | 10 | 47 | 15 | 1.52% | 109.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 11:15:00 | 2466.70 | 2331.16 | 2330.99 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 14:15:00 | 2239.95 | 2337.70 | 2337.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-27 09:15:00 | 2234.85 | 2335.74 | 2336.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 12:15:00 | 2285.65 | 2281.77 | 2303.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-17 12:45:00 | 2290.10 | 2281.77 | 2303.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 2262.75 | 2233.09 | 2266.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 14:00:00 | 2262.75 | 2233.09 | 2266.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 2304.55 | 2234.20 | 2266.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 09:45:00 | 2305.45 | 2234.20 | 2266.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 10:15:00 | 2311.20 | 2234.97 | 2266.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 10:45:00 | 2314.30 | 2234.97 | 2266.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 12:15:00 | 2333.00 | 2289.41 | 2289.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 09:15:00 | 2339.30 | 2291.08 | 2290.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 2370.00 | 2387.07 | 2352.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 15:00:00 | 2370.00 | 2387.07 | 2352.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 2398.45 | 2386.92 | 2352.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 10:00:00 | 2413.95 | 2387.63 | 2354.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-09 11:15:00 | 2342.30 | 2431.35 | 2391.88 | SL hit (close<static) qty=1.00 sl=2344.60 alert=retest2 |

### Cycle 4 — SELL (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 12:15:00 | 2279.20 | 2365.71 | 2366.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 14:15:00 | 2271.60 | 2358.74 | 2362.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 2324.10 | 2318.88 | 2336.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 15:00:00 | 2324.10 | 2318.88 | 2336.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 2337.30 | 2319.03 | 2336.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:00:00 | 2337.30 | 2319.03 | 2336.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 2336.05 | 2319.20 | 2336.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:00:00 | 2336.05 | 2319.20 | 2336.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 2337.55 | 2319.38 | 2336.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:00:00 | 2337.55 | 2319.38 | 2336.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 2358.00 | 2319.76 | 2336.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:00:00 | 2358.00 | 2319.76 | 2336.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 2358.00 | 2320.14 | 2336.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:15:00 | 2364.25 | 2320.14 | 2336.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 2341.00 | 2328.59 | 2339.38 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 12:15:00 | 2381.00 | 2348.63 | 2348.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 14:15:00 | 2391.20 | 2349.42 | 2348.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 14:15:00 | 2519.30 | 2527.17 | 2467.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-16 15:00:00 | 2519.30 | 2527.17 | 2467.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 2471.20 | 2573.04 | 2515.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:30:00 | 2473.00 | 2573.04 | 2515.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 2429.70 | 2571.62 | 2515.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:00:00 | 2429.70 | 2571.62 | 2515.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 2290.15 | 2470.31 | 2470.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 2283.00 | 2466.67 | 2469.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 10:15:00 | 2343.55 | 2331.80 | 2381.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-10 11:00:00 | 2343.55 | 2331.80 | 2381.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 2391.30 | 2335.63 | 2379.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 2391.30 | 2335.63 | 2379.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 2381.80 | 2336.09 | 2379.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:30:00 | 2405.55 | 2336.09 | 2379.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 2383.90 | 2336.56 | 2379.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:45:00 | 2385.65 | 2336.56 | 2379.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 12:15:00 | 2385.00 | 2337.05 | 2379.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 12:30:00 | 2384.00 | 2337.05 | 2379.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 2401.25 | 2337.69 | 2379.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:00:00 | 2401.25 | 2337.69 | 2379.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 2457.00 | 2353.64 | 2382.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:15:00 | 2442.65 | 2368.43 | 2388.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 2449.65 | 2380.81 | 2392.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 09:15:00 | 2445.05 | 2388.43 | 2395.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 2320.52 | 2387.31 | 2392.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 2327.17 | 2387.31 | 2392.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 2322.80 | 2387.31 | 2392.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 10:15:00 | 2387.70 | 2385.45 | 2391.49 | SL hit (close>ema200) qty=0.50 sl=2385.45 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 2510.65 | 2396.52 | 2396.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 11:15:00 | 2534.40 | 2399.16 | 2397.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 2431.75 | 2442.65 | 2420.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:00:00 | 2431.75 | 2442.65 | 2420.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 2461.90 | 2442.84 | 2421.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 14:00:00 | 2465.50 | 2443.08 | 2421.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:00:00 | 2463.25 | 2480.40 | 2446.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 11:15:00 | 2463.70 | 2480.64 | 2449.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 10:00:00 | 2465.00 | 2480.31 | 2449.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 2484.80 | 2525.72 | 2488.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 2482.05 | 2525.72 | 2488.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 2497.00 | 2525.43 | 2488.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 11:45:00 | 2515.15 | 2525.20 | 2488.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 13:15:00 | 2508.05 | 2524.98 | 2488.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 14:15:00 | 2481.40 | 2525.34 | 2491.44 | SL hit (close<static) qty=1.00 sl=2481.80 alert=retest2 |

### Cycle 8 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 2377.05 | 2471.24 | 2471.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 2348.10 | 2470.01 | 2470.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 2324.80 | 2316.66 | 2369.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 09:30:00 | 2327.10 | 2316.66 | 2369.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 2370.60 | 2318.41 | 2368.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:45:00 | 2373.20 | 2318.41 | 2368.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 2374.05 | 2318.96 | 2368.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 2374.05 | 2318.96 | 2368.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 2372.55 | 2319.50 | 2368.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:30:00 | 2372.35 | 2319.50 | 2368.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 2363.05 | 2319.93 | 2368.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:00:00 | 2358.10 | 2322.34 | 2368.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 15:15:00 | 2240.19 | 2316.11 | 2361.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-27 12:15:00 | 2282.60 | 2267.52 | 2321.31 | SL hit (close>ema200) qty=0.50 sl=2267.52 alert=retest2 |

### Cycle 9 — BUY (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 12:15:00 | 2596.35 | 2316.96 | 2315.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 2617.10 | 2377.68 | 2348.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 2879.10 | 2889.80 | 2779.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 10:00:00 | 2879.10 | 2889.80 | 2779.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 2680.15 | 2887.00 | 2792.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 2769.60 | 2876.03 | 2789.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 2778.50 | 2874.74 | 2789.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 14:15:00 | 2776.45 | 2870.35 | 2788.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 2839.50 | 2859.26 | 2786.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-22 09:15:00 | 3046.56 | 2897.82 | 2819.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 2843.00 | 3058.51 | 3058.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 2831.00 | 2980.92 | 3012.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2961.50 | 2938.82 | 2979.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 2961.50 | 2938.82 | 2979.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2961.50 | 2938.82 | 2979.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 2938.40 | 2946.84 | 2978.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 2929.80 | 2946.68 | 2978.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 2929.50 | 2946.62 | 2978.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 2934.50 | 2946.62 | 2978.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 2982.00 | 2946.94 | 2977.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 2982.00 | 2946.94 | 2977.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 2980.00 | 2947.27 | 2977.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:15:00 | 2983.10 | 2947.27 | 2977.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 2971.10 | 2947.51 | 2977.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 2966.10 | 2947.74 | 2977.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 2817.79 | 2931.51 | 2962.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 2791.48 | 2924.79 | 2957.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 2783.31 | 2912.66 | 2949.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 2783.03 | 2912.66 | 2949.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 2787.78 | 2912.66 | 2949.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 2914.90 | 2910.45 | 2946.98 | SL hit (close>ema200) qty=0.50 sl=2910.45 alert=retest2 |

### Cycle 11 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 3200.50 | 2969.90 | 2969.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3204.10 | 2972.23 | 2970.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 3009.50 | 3013.44 | 2994.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:45:00 | 3014.00 | 3013.44 | 2994.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2988.80 | 3014.95 | 2996.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 2988.80 | 3014.95 | 2996.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 2992.90 | 3014.73 | 2996.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 2973.60 | 3014.73 | 2996.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 2964.10 | 3014.23 | 2995.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 2964.10 | 3014.23 | 2995.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 2978.00 | 3013.87 | 2995.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 2981.90 | 3013.11 | 2995.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 2955.10 | 3012.21 | 2995.31 | SL hit (close<static) qty=1.00 sl=2962.20 alert=retest2 |

### Cycle 12 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 2902.50 | 2981.90 | 2981.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 2885.80 | 2979.54 | 2980.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 2919.80 | 2895.86 | 2930.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 2919.80 | 2895.86 | 2930.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2919.80 | 2895.86 | 2930.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:45:00 | 2922.60 | 2895.86 | 2930.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 2921.80 | 2896.11 | 2930.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 2922.40 | 2896.11 | 2930.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 2945.50 | 2896.61 | 2930.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:00:00 | 2945.50 | 2896.61 | 2930.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 2918.40 | 2896.82 | 2930.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:30:00 | 2945.80 | 2896.82 | 2930.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 2911.10 | 2896.96 | 2930.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:30:00 | 2920.00 | 2896.96 | 2930.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 2933.00 | 2897.61 | 2930.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 2913.60 | 2897.61 | 2930.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2897.70 | 2897.61 | 2930.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:30:00 | 2890.20 | 2898.26 | 2929.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 13:45:00 | 2895.00 | 2883.61 | 2915.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:45:00 | 2894.20 | 2883.70 | 2915.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 11:15:00 | 2946.00 | 2885.32 | 2915.76 | SL hit (close>static) qty=1.00 sl=2939.90 alert=retest2 |

### Cycle 13 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 3083.00 | 2938.21 | 2937.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 3126.90 | 2940.09 | 2938.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 2976.30 | 3009.61 | 2980.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 2976.30 | 3009.61 | 2980.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2976.30 | 3009.61 | 2980.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 2976.30 | 3009.61 | 2980.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 3015.80 | 3009.67 | 2980.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 3019.90 | 3009.67 | 2980.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 3036.20 | 3018.54 | 2988.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 09:45:00 | 3032.00 | 3019.00 | 2989.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 11:45:00 | 3020.00 | 3019.08 | 2989.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2994.00 | 3023.52 | 2995.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 2994.00 | 3023.52 | 2995.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 2982.60 | 3023.11 | 2995.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:45:00 | 2982.00 | 3023.11 | 2995.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 2968.30 | 3022.56 | 2995.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 2969.40 | 3022.56 | 2995.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-19 12:15:00 | 2948.30 | 3021.82 | 2995.18 | SL hit (close<static) qty=1.00 sl=2955.00 alert=retest2 |

### Cycle 14 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2706.60 | 2971.95 | 2972.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 2693.80 | 2954.55 | 2963.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2929.60 | 2895.15 | 2928.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 2929.60 | 2895.15 | 2928.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2929.60 | 2895.15 | 2928.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:15:00 | 2912.00 | 2896.37 | 2928.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 2916.80 | 2897.26 | 2928.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 13:45:00 | 2921.80 | 2898.48 | 2928.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 14:45:00 | 2921.30 | 2898.71 | 2928.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 2902.20 | 2899.02 | 2928.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:30:00 | 2866.90 | 2899.11 | 2927.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 2963.30 | 2899.68 | 2926.55 | SL hit (close>static) qty=1.00 sl=2937.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-22 10:00:00 | 2413.95 | 2024-01-09 11:15:00 | 2342.30 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-06-24 09:15:00 | 2442.65 | 2024-07-22 09:15:00 | 2320.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-28 09:15:00 | 2449.65 | 2024-07-22 09:15:00 | 2327.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-02 09:15:00 | 2445.05 | 2024-07-22 09:15:00 | 2322.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-24 09:15:00 | 2442.65 | 2024-07-23 10:15:00 | 2387.70 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2024-06-28 09:15:00 | 2449.65 | 2024-07-23 10:15:00 | 2387.70 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2024-07-02 09:15:00 | 2445.05 | 2024-07-23 10:15:00 | 2387.70 | STOP_HIT | 0.50 | 2.35% |
| SELL | retest2 | 2024-07-26 13:45:00 | 2452.95 | 2024-07-29 09:15:00 | 2488.95 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-08-05 14:00:00 | 2465.50 | 2024-09-11 14:15:00 | 2481.40 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2024-08-14 12:00:00 | 2463.25 | 2024-09-11 14:15:00 | 2481.40 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2024-08-19 11:15:00 | 2463.70 | 2024-09-17 10:15:00 | 2410.45 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-08-20 10:00:00 | 2465.00 | 2024-09-17 10:15:00 | 2410.45 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-09-09 11:45:00 | 2515.15 | 2024-09-17 10:15:00 | 2410.45 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2024-09-09 13:15:00 | 2508.05 | 2024-09-17 10:15:00 | 2410.45 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2024-09-30 09:15:00 | 2517.95 | 2024-10-01 09:15:00 | 2460.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-09-30 12:30:00 | 2505.00 | 2024-10-01 09:15:00 | 2460.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-10-03 09:30:00 | 2495.40 | 2024-10-03 13:15:00 | 2421.85 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-11-08 11:00:00 | 2358.10 | 2024-11-12 15:15:00 | 2240.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 11:00:00 | 2358.10 | 2024-11-27 12:15:00 | 2282.60 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2024-12-11 11:45:00 | 2355.05 | 2024-12-19 09:15:00 | 2237.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 11:45:00 | 2355.05 | 2024-12-20 09:15:00 | 2300.70 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2025-01-08 14:30:00 | 2354.95 | 2025-01-09 09:15:00 | 2647.30 | STOP_HIT | 1.00 | -12.41% |
| BUY | retest2 | 2025-04-08 09:15:00 | 2769.60 | 2025-04-22 09:15:00 | 3046.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-08 10:15:00 | 2778.50 | 2025-04-22 09:15:00 | 3056.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-08 14:15:00 | 2776.45 | 2025-04-22 09:15:00 | 3054.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-11 09:15:00 | 2839.50 | 2025-05-15 11:15:00 | 2874.50 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2025-05-09 10:15:00 | 2984.30 | 2025-05-15 11:15:00 | 2874.50 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2025-05-12 12:45:00 | 2992.80 | 2025-05-26 15:15:00 | 2891.80 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2025-05-19 10:15:00 | 2980.90 | 2025-05-26 15:15:00 | 2891.80 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-05-19 15:15:00 | 2980.00 | 2025-05-27 09:15:00 | 2882.80 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-05-23 09:15:00 | 2918.80 | 2025-05-27 09:15:00 | 2882.80 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-05-23 10:00:00 | 2919.40 | 2025-05-27 09:15:00 | 2882.80 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-05-23 10:45:00 | 2909.20 | 2025-05-27 09:15:00 | 2882.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-05-23 14:00:00 | 2909.10 | 2025-05-28 15:15:00 | 2890.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-26 09:15:00 | 2911.50 | 2025-05-28 15:15:00 | 2890.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-05-26 10:45:00 | 2917.00 | 2025-05-30 12:15:00 | 2867.50 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-05-28 10:45:00 | 2916.70 | 2025-05-30 12:15:00 | 2867.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-28 11:45:00 | 2911.60 | 2025-06-05 10:15:00 | 3123.45 | TARGET_HIT | 1.00 | 7.28% |
| BUY | retest2 | 2025-06-03 09:15:00 | 2921.20 | 2025-06-30 10:15:00 | 3213.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 14:15:00 | 2905.20 | 2025-08-08 09:15:00 | 2885.60 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-08-07 15:00:00 | 2901.70 | 2025-08-08 09:15:00 | 2885.60 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-09-16 09:45:00 | 2938.40 | 2025-09-26 13:15:00 | 2817.79 | PARTIAL | 0.50 | 4.10% |
| SELL | retest2 | 2025-09-16 10:45:00 | 2929.80 | 2025-09-29 12:15:00 | 2791.48 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2025-09-16 11:45:00 | 2929.50 | 2025-10-01 09:15:00 | 2783.31 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-09-16 12:15:00 | 2934.50 | 2025-10-01 09:15:00 | 2783.03 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-09-17 13:15:00 | 2966.10 | 2025-10-01 09:15:00 | 2787.78 | PARTIAL | 0.50 | 6.01% |
| SELL | retest2 | 2025-09-16 09:45:00 | 2938.40 | 2025-10-03 12:15:00 | 2914.90 | STOP_HIT | 0.50 | 0.80% |
| SELL | retest2 | 2025-09-16 10:45:00 | 2929.80 | 2025-10-03 12:15:00 | 2914.90 | STOP_HIT | 0.50 | 0.51% |
| SELL | retest2 | 2025-09-16 11:45:00 | 2929.50 | 2025-10-03 12:15:00 | 2914.90 | STOP_HIT | 0.50 | 0.50% |
| SELL | retest2 | 2025-09-16 12:15:00 | 2934.50 | 2025-10-03 12:15:00 | 2914.90 | STOP_HIT | 0.50 | 0.67% |
| SELL | retest2 | 2025-09-17 13:15:00 | 2966.10 | 2025-10-03 12:15:00 | 2914.90 | STOP_HIT | 0.50 | 1.73% |
| SELL | retest2 | 2025-10-08 10:15:00 | 2968.80 | 2025-10-08 11:15:00 | 2995.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-08 15:15:00 | 2968.20 | 2025-10-09 09:15:00 | 2993.80 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-30 15:00:00 | 2981.90 | 2025-10-31 09:15:00 | 2955.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-12-02 09:30:00 | 2890.20 | 2025-12-10 11:15:00 | 2946.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-12-09 13:45:00 | 2895.00 | 2025-12-10 11:15:00 | 2946.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-09 14:45:00 | 2894.20 | 2025-12-10 11:15:00 | 2946.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-01-06 12:15:00 | 3019.90 | 2026-01-19 12:15:00 | 2948.30 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-01-09 09:45:00 | 3036.20 | 2026-01-19 12:15:00 | 2948.30 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2026-01-12 09:45:00 | 3032.00 | 2026-01-19 12:15:00 | 2948.30 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2026-01-12 11:45:00 | 3020.00 | 2026-01-19 12:15:00 | 2948.30 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-03 13:15:00 | 2912.00 | 2026-02-09 10:15:00 | 2963.30 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-02-04 10:15:00 | 2916.80 | 2026-02-13 09:15:00 | 2766.40 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2026-02-04 13:45:00 | 2921.80 | 2026-02-13 09:15:00 | 2770.96 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2026-02-04 14:45:00 | 2921.30 | 2026-02-13 09:15:00 | 2775.71 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2026-02-06 09:30:00 | 2866.90 | 2026-02-13 09:15:00 | 2775.24 | PARTIAL | 0.50 | 3.20% |
| SELL | retest2 | 2026-02-12 11:45:00 | 2862.50 | 2026-02-18 09:15:00 | 2719.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 10:15:00 | 2916.80 | 2026-02-23 15:15:00 | 2625.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 13:45:00 | 2921.80 | 2026-02-23 15:15:00 | 2629.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 14:45:00 | 2921.30 | 2026-02-23 15:15:00 | 2629.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-06 09:30:00 | 2866.90 | 2026-02-24 09:15:00 | 2620.80 | TARGET_HIT | 0.50 | 8.58% |
| SELL | retest2 | 2026-02-12 11:45:00 | 2862.50 | 2026-02-24 11:15:00 | 2576.25 | TARGET_HIT | 0.50 | 10.00% |
