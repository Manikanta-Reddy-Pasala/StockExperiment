# DOMS Industries Ltd. (DOMS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 2320.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 84 |
| ALERT1 | 55 |
| ALERT2 | 55 |
| ALERT2_SKIP | 34 |
| ALERT3 | 141 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 64 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 45
- **Target hits / Stop hits / Partials:** 3 / 64 / 9
- **Avg / median % per leg:** 0.40% / -0.71%
- **Sum % (uncompounded):** 30.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 3 | 14.3% | 2 | 19 | 0 | -0.26% | -5.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 3 | 14.3% | 2 | 19 | 0 | -0.26% | -5.5% |
| SELL (all) | 55 | 28 | 50.9% | 1 | 45 | 9 | 0.65% | 35.6% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 5.07% | 30.4% |
| SELL @ 3rd Alert (retest2) | 49 | 22 | 44.9% | 1 | 42 | 6 | 0.11% | 5.2% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 5.07% | 30.4% |
| retest2 (combined) | 70 | 25 | 35.7% | 3 | 61 | 6 | -0.00% | -0.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2767.10 | 2710.79 | 2703.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 2800.00 | 2738.89 | 2718.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 2790.00 | 2794.62 | 2768.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 09:15:00 | 2851.50 | 2794.62 | 2768.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 2823.50 | 2838.09 | 2817.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:45:00 | 2819.90 | 2838.09 | 2817.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 2817.40 | 2831.73 | 2818.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 2818.10 | 2831.73 | 2818.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 2824.20 | 2830.23 | 2818.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 2863.70 | 2830.06 | 2819.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 2837.90 | 2830.83 | 2821.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:45:00 | 2839.80 | 2831.18 | 2822.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:15:00 | 2841.30 | 2831.18 | 2822.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2820.50 | 2838.22 | 2831.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 2790.10 | 2829.03 | 2829.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 2790.10 | 2829.03 | 2829.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 2648.20 | 2784.08 | 2808.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 2557.30 | 2556.07 | 2613.12 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 11:15:00 | 2535.00 | 2556.07 | 2613.12 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 13:30:00 | 2541.80 | 2555.14 | 2576.91 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 15:15:00 | 2537.00 | 2555.76 | 2575.21 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2526.80 | 2546.96 | 2567.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 2508.40 | 2535.04 | 2556.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 15:15:00 | 2408.25 | 2445.22 | 2487.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 15:15:00 | 2414.71 | 2445.22 | 2487.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 15:15:00 | 2410.15 | 2445.22 | 2487.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 15:15:00 | 2382.98 | 2445.22 | 2487.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 2407.70 | 2376.88 | 2401.80 | SL hit (close>ema200) qty=0.50 sl=2376.88 alert=retest1 |

### Cycle 3 — BUY (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 15:15:00 | 2452.00 | 2412.80 | 2410.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 2468.50 | 2441.94 | 2429.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 2454.20 | 2457.46 | 2445.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:45:00 | 2457.80 | 2457.46 | 2445.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 2457.20 | 2457.41 | 2446.51 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 2440.10 | 2443.72 | 2443.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 15:15:00 | 2435.00 | 2441.38 | 2442.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 2429.80 | 2424.64 | 2430.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 2429.80 | 2424.64 | 2430.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 2429.80 | 2424.64 | 2430.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:30:00 | 2423.50 | 2423.83 | 2429.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:45:00 | 2424.50 | 2423.90 | 2429.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 13:00:00 | 2424.10 | 2423.94 | 2428.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 2413.00 | 2427.02 | 2429.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 2413.00 | 2424.21 | 2427.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 2414.00 | 2424.21 | 2427.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2407.00 | 2420.77 | 2426.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 13:15:00 | 2401.70 | 2413.32 | 2420.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 10:30:00 | 2400.30 | 2406.18 | 2414.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 2302.32 | 2337.35 | 2356.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 2303.28 | 2337.35 | 2356.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 2302.89 | 2337.35 | 2356.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 2292.35 | 2337.35 | 2356.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 2335.00 | 2331.31 | 2346.70 | SL hit (close>ema200) qty=0.50 sl=2331.31 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 2380.90 | 2359.44 | 2356.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 13:15:00 | 2408.00 | 2376.27 | 2365.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 2413.30 | 2423.70 | 2404.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 2413.30 | 2423.70 | 2404.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 2389.70 | 2416.90 | 2403.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 2390.00 | 2416.90 | 2403.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 2375.00 | 2408.52 | 2400.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 2375.00 | 2408.52 | 2400.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 2402.50 | 2405.94 | 2400.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 2405.00 | 2404.87 | 2400.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 2455.40 | 2488.44 | 2489.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 2455.40 | 2488.44 | 2489.43 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 12:15:00 | 2506.90 | 2491.22 | 2490.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 15:15:00 | 2528.00 | 2504.78 | 2497.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 2485.00 | 2500.83 | 2496.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 2485.00 | 2500.83 | 2496.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 2485.00 | 2500.83 | 2496.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:45:00 | 2491.00 | 2500.83 | 2496.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 2489.90 | 2498.64 | 2495.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:15:00 | 2494.70 | 2498.64 | 2495.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 2482.70 | 2492.04 | 2493.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 2482.70 | 2492.04 | 2493.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 2480.10 | 2488.38 | 2491.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 15:15:00 | 2456.00 | 2453.46 | 2468.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 15:15:00 | 2456.00 | 2453.46 | 2468.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 2456.00 | 2453.46 | 2468.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 2430.40 | 2453.85 | 2457.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 2430.20 | 2447.96 | 2454.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 2428.00 | 2434.62 | 2445.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 2429.80 | 2435.21 | 2442.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 2441.70 | 2436.44 | 2441.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 2443.00 | 2436.44 | 2441.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2433.00 | 2435.76 | 2440.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 2412.10 | 2434.10 | 2439.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 2411.80 | 2376.70 | 2376.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 2411.80 | 2376.70 | 2376.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 2421.50 | 2391.88 | 2383.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 2393.50 | 2399.53 | 2390.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:00:00 | 2393.50 | 2399.53 | 2390.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 2387.30 | 2397.08 | 2389.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 2401.10 | 2390.74 | 2389.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:00:00 | 2401.00 | 2392.79 | 2390.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 2384.00 | 2390.95 | 2391.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 2384.00 | 2390.95 | 2391.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 2373.00 | 2382.21 | 2386.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 2371.50 | 2371.47 | 2379.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 2371.50 | 2371.47 | 2379.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 2374.20 | 2371.46 | 2377.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 2362.30 | 2368.99 | 2375.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 2390.70 | 2374.57 | 2375.96 | SL hit (close>static) qty=1.00 sl=2384.70 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 2398.00 | 2379.26 | 2377.96 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 2372.50 | 2376.99 | 2377.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 13:15:00 | 2366.90 | 2374.97 | 2376.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 2388.70 | 2375.03 | 2375.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 2388.70 | 2375.03 | 2375.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 2388.70 | 2375.03 | 2375.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 2388.70 | 2375.03 | 2375.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 2389.00 | 2377.82 | 2376.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 15:15:00 | 2399.90 | 2388.69 | 2383.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 2371.00 | 2385.15 | 2381.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 2371.00 | 2385.15 | 2381.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 2371.00 | 2385.15 | 2381.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 2365.00 | 2385.15 | 2381.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 2364.40 | 2381.00 | 2380.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 2364.20 | 2381.00 | 2380.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 2371.20 | 2379.04 | 2379.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 2357.80 | 2370.95 | 2374.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 2348.00 | 2324.57 | 2341.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 2348.00 | 2324.57 | 2341.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2348.00 | 2324.57 | 2341.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 2348.00 | 2324.57 | 2341.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2344.00 | 2328.46 | 2341.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 2346.00 | 2328.46 | 2341.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 2365.00 | 2335.77 | 2343.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 2365.00 | 2335.77 | 2343.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 2348.00 | 2338.21 | 2344.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:30:00 | 2342.80 | 2339.13 | 2343.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 2342.00 | 2338.92 | 2341.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 2356.70 | 2343.99 | 2343.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 2356.70 | 2343.99 | 2343.62 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 2340.10 | 2344.33 | 2344.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 2316.90 | 2334.32 | 2339.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 2339.50 | 2330.31 | 2335.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 10:15:00 | 2339.50 | 2330.31 | 2335.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 2339.50 | 2330.31 | 2335.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 2339.50 | 2330.31 | 2335.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 2341.40 | 2332.52 | 2336.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 2341.40 | 2332.52 | 2336.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 2338.50 | 2333.72 | 2336.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 2338.90 | 2333.72 | 2336.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 2339.10 | 2334.80 | 2336.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 2341.80 | 2334.80 | 2336.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 2369.00 | 2341.64 | 2339.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 2370.00 | 2347.31 | 2342.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 11:15:00 | 2352.20 | 2352.88 | 2346.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 11:45:00 | 2353.50 | 2352.88 | 2346.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 2386.70 | 2359.50 | 2350.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:30:00 | 2357.80 | 2359.50 | 2350.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 2373.50 | 2376.89 | 2365.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 2387.00 | 2376.89 | 2365.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 2352.50 | 2379.76 | 2372.04 | SL hit (close<static) qty=1.00 sl=2360.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 2339.90 | 2365.10 | 2366.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 2303.90 | 2352.86 | 2360.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 2446.40 | 2335.73 | 2338.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 2446.40 | 2335.73 | 2338.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2446.40 | 2335.73 | 2338.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:00:00 | 2446.40 | 2335.73 | 2338.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 2517.00 | 2371.98 | 2354.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 2545.10 | 2406.61 | 2372.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 2431.40 | 2490.62 | 2455.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 2431.40 | 2490.62 | 2455.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 2431.40 | 2490.62 | 2455.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 2431.40 | 2490.62 | 2455.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 2404.10 | 2473.32 | 2450.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 2415.50 | 2473.32 | 2450.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 2395.40 | 2437.77 | 2438.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 12:15:00 | 2356.00 | 2421.41 | 2430.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 2396.90 | 2395.73 | 2411.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:00:00 | 2396.90 | 2395.73 | 2411.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 2395.00 | 2394.07 | 2404.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 2500.50 | 2394.07 | 2404.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 2513.20 | 2417.90 | 2414.45 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 2470.50 | 2481.44 | 2482.72 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 2492.20 | 2483.60 | 2483.58 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 2460.50 | 2480.70 | 2483.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 2457.40 | 2476.04 | 2480.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 2440.60 | 2437.32 | 2445.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 10:15:00 | 2440.60 | 2437.32 | 2445.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2440.60 | 2437.32 | 2445.49 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 2486.10 | 2450.08 | 2448.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 2503.40 | 2477.80 | 2465.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 11:15:00 | 2661.50 | 2669.19 | 2611.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 12:00:00 | 2661.50 | 2669.19 | 2611.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2605.50 | 2645.19 | 2620.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 2606.90 | 2645.19 | 2620.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 2605.10 | 2637.17 | 2619.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 2604.70 | 2637.17 | 2619.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 2600.30 | 2610.13 | 2610.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 2572.00 | 2602.51 | 2607.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 2598.00 | 2592.45 | 2598.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 15:15:00 | 2598.00 | 2592.45 | 2598.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 2598.00 | 2592.45 | 2598.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 2589.40 | 2592.45 | 2598.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2600.00 | 2593.96 | 2598.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:30:00 | 2564.60 | 2593.24 | 2594.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 2610.50 | 2595.56 | 2595.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 13:15:00 | 2610.50 | 2595.56 | 2595.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 2621.60 | 2602.40 | 2598.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 10:15:00 | 2616.40 | 2626.10 | 2615.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 2616.40 | 2626.10 | 2615.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 2616.40 | 2626.10 | 2615.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 2616.40 | 2626.10 | 2615.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 2617.20 | 2624.32 | 2616.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 2617.20 | 2624.32 | 2616.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 2603.30 | 2620.11 | 2614.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:00:00 | 2603.30 | 2620.11 | 2614.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 2606.80 | 2617.45 | 2614.11 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 2595.00 | 2609.69 | 2610.96 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 2619.50 | 2612.84 | 2611.94 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 2608.00 | 2611.55 | 2611.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 2588.40 | 2606.99 | 2609.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 13:15:00 | 2607.00 | 2606.99 | 2609.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 13:15:00 | 2607.00 | 2606.99 | 2609.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2607.00 | 2606.99 | 2609.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 2607.00 | 2606.99 | 2609.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 2593.90 | 2604.38 | 2607.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 2592.20 | 2600.73 | 2605.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:30:00 | 2592.70 | 2598.21 | 2603.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 2631.30 | 2600.53 | 2603.01 | SL hit (close>static) qty=1.00 sl=2608.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 2632.60 | 2606.95 | 2605.70 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 2603.50 | 2612.57 | 2612.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 2590.00 | 2605.05 | 2608.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 2520.10 | 2510.19 | 2542.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 2520.10 | 2510.19 | 2542.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 2505.30 | 2517.20 | 2534.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 2536.00 | 2517.20 | 2534.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 2530.00 | 2519.76 | 2533.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 2491.10 | 2516.71 | 2531.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 2492.90 | 2491.49 | 2508.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 2498.10 | 2492.65 | 2500.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 2494.70 | 2492.71 | 2495.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 2498.20 | 2491.90 | 2494.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 2496.10 | 2491.90 | 2494.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 2488.70 | 2491.26 | 2494.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 13:15:00 | 2485.00 | 2491.26 | 2494.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 15:00:00 | 2485.90 | 2489.94 | 2492.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 2631.60 | 2518.12 | 2505.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 2631.60 | 2518.12 | 2505.19 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 2547.60 | 2561.47 | 2562.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 2535.70 | 2548.51 | 2553.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 2556.70 | 2536.51 | 2542.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 2556.70 | 2536.51 | 2542.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 2556.70 | 2536.51 | 2542.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 2556.60 | 2536.51 | 2542.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 2543.20 | 2537.85 | 2542.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 2525.20 | 2537.85 | 2542.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:45:00 | 2534.50 | 2497.50 | 2508.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 14:15:00 | 2523.80 | 2499.40 | 2496.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 2523.80 | 2499.40 | 2496.91 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 2486.70 | 2496.15 | 2496.25 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 10:15:00 | 2508.00 | 2496.16 | 2495.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 2520.10 | 2507.98 | 2502.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 2505.90 | 2511.60 | 2506.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 2505.90 | 2511.60 | 2506.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 2505.90 | 2511.60 | 2506.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 2505.90 | 2511.60 | 2506.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 2513.70 | 2512.02 | 2507.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 2532.20 | 2515.01 | 2509.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:30:00 | 2529.90 | 2523.24 | 2517.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 2528.30 | 2523.24 | 2517.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:30:00 | 2529.90 | 2527.92 | 2522.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2555.10 | 2560.01 | 2553.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 2556.00 | 2560.01 | 2553.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 2552.30 | 2558.47 | 2553.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 2509.60 | 2545.47 | 2548.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 2509.60 | 2545.47 | 2548.29 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 14:15:00 | 2551.20 | 2546.72 | 2546.55 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 2524.90 | 2542.35 | 2544.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 2468.00 | 2527.48 | 2537.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 2499.40 | 2487.47 | 2505.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 10:15:00 | 2499.40 | 2487.47 | 2505.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2499.40 | 2487.47 | 2505.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:45:00 | 2503.10 | 2487.47 | 2505.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 2514.30 | 2492.84 | 2506.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 2510.00 | 2492.84 | 2506.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 2509.40 | 2496.15 | 2506.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:30:00 | 2501.00 | 2498.34 | 2506.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 2537.20 | 2506.11 | 2509.67 | SL hit (close>static) qty=1.00 sl=2527.90 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 2552.90 | 2516.43 | 2513.39 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 2514.80 | 2531.10 | 2531.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 11:15:00 | 2508.20 | 2526.52 | 2529.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 2571.30 | 2526.78 | 2526.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 2571.30 | 2526.78 | 2526.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 2571.30 | 2526.78 | 2526.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 2571.30 | 2526.78 | 2526.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 2600.00 | 2541.42 | 2533.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 14:15:00 | 2621.90 | 2572.89 | 2551.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 15:15:00 | 2593.60 | 2596.65 | 2578.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 2571.00 | 2591.52 | 2577.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2571.00 | 2591.52 | 2577.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 2571.00 | 2591.52 | 2577.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 2573.50 | 2587.91 | 2577.43 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 2557.00 | 2570.44 | 2572.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 2539.90 | 2564.33 | 2569.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 2558.50 | 2553.98 | 2561.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 2558.50 | 2553.98 | 2561.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 2567.30 | 2556.64 | 2561.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 2563.30 | 2556.64 | 2561.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2541.70 | 2553.66 | 2559.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:45:00 | 2540.10 | 2551.09 | 2557.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 2539.40 | 2549.71 | 2556.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:45:00 | 2540.00 | 2547.77 | 2554.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 2539.50 | 2545.62 | 2553.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 2530.00 | 2539.19 | 2548.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:15:00 | 2528.20 | 2539.19 | 2548.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:00:00 | 2514.40 | 2534.24 | 2545.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 2559.30 | 2526.46 | 2523.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 2559.30 | 2526.46 | 2523.28 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 2521.30 | 2531.80 | 2531.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2520.10 | 2529.46 | 2530.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 2500.00 | 2497.45 | 2506.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 10:45:00 | 2499.20 | 2497.45 | 2506.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 2499.00 | 2497.38 | 2505.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 2500.00 | 2497.38 | 2505.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 2504.70 | 2499.39 | 2504.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 2504.70 | 2499.39 | 2504.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 2502.10 | 2499.93 | 2504.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 2637.90 | 2499.93 | 2504.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 09:15:00 | 2624.50 | 2524.85 | 2515.38 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 2569.00 | 2590.88 | 2590.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 2557.80 | 2584.26 | 2587.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 15:15:00 | 2598.90 | 2586.67 | 2588.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 15:15:00 | 2598.90 | 2586.67 | 2588.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 2598.90 | 2586.67 | 2588.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 2536.60 | 2586.67 | 2588.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:15:00 | 2573.30 | 2553.88 | 2564.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 2569.30 | 2545.28 | 2544.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 2569.30 | 2545.28 | 2544.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 2572.00 | 2550.62 | 2546.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 2540.00 | 2572.79 | 2566.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 2540.00 | 2572.79 | 2566.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2540.00 | 2572.79 | 2566.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 2540.00 | 2572.79 | 2566.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2554.00 | 2569.03 | 2565.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 2538.60 | 2569.03 | 2565.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 2546.90 | 2560.57 | 2562.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 2543.00 | 2553.45 | 2557.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 2534.00 | 2530.24 | 2539.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:00:00 | 2534.00 | 2530.24 | 2539.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2535.10 | 2531.21 | 2539.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 2536.00 | 2531.21 | 2539.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 2548.90 | 2534.75 | 2540.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 2548.90 | 2534.75 | 2540.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 2527.50 | 2533.30 | 2538.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 2543.30 | 2533.30 | 2538.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2554.70 | 2537.58 | 2540.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 2553.20 | 2537.58 | 2540.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2545.30 | 2539.12 | 2540.79 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 2554.20 | 2542.14 | 2542.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 2562.70 | 2546.25 | 2543.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 2571.20 | 2571.68 | 2561.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 13:00:00 | 2571.20 | 2571.68 | 2561.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2579.10 | 2576.11 | 2566.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 2570.00 | 2576.11 | 2566.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2561.60 | 2573.21 | 2566.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:45:00 | 2562.30 | 2573.21 | 2566.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2576.70 | 2573.90 | 2567.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 2575.10 | 2573.90 | 2567.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2571.10 | 2573.34 | 2567.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:45:00 | 2565.00 | 2573.34 | 2567.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 2575.20 | 2573.72 | 2568.32 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 2545.10 | 2562.60 | 2564.47 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 14:15:00 | 2577.20 | 2565.77 | 2565.17 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 2553.00 | 2563.07 | 2564.43 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 2582.00 | 2566.86 | 2566.03 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 2553.80 | 2564.26 | 2565.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 2542.50 | 2558.01 | 2562.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 2543.00 | 2537.37 | 2547.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 13:00:00 | 2543.00 | 2537.37 | 2547.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 2543.20 | 2538.54 | 2546.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 2543.20 | 2538.54 | 2546.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2553.30 | 2541.91 | 2546.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 2548.90 | 2541.91 | 2546.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2563.40 | 2546.21 | 2548.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 2561.30 | 2546.21 | 2548.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 2584.70 | 2553.91 | 2551.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 2617.30 | 2571.50 | 2560.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 2586.00 | 2589.11 | 2575.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:45:00 | 2585.90 | 2589.11 | 2575.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 2601.00 | 2595.13 | 2583.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 11:45:00 | 2618.10 | 2600.61 | 2588.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 2600.00 | 2635.01 | 2638.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 2600.00 | 2635.01 | 2638.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 2586.50 | 2610.21 | 2623.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 2492.50 | 2467.26 | 2486.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 2492.50 | 2467.26 | 2486.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2492.50 | 2467.26 | 2486.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 2498.40 | 2467.26 | 2486.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2518.30 | 2477.46 | 2489.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 2518.30 | 2477.46 | 2489.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 2521.00 | 2486.17 | 2492.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 2511.00 | 2486.17 | 2492.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:45:00 | 2515.00 | 2491.94 | 2494.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 13:15:00 | 2516.00 | 2496.75 | 2496.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 2516.00 | 2496.75 | 2496.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 15:15:00 | 2524.90 | 2504.98 | 2500.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 2525.20 | 2526.28 | 2515.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 2525.20 | 2526.28 | 2515.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 2525.20 | 2526.28 | 2515.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 2505.00 | 2526.28 | 2515.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2557.00 | 2532.43 | 2519.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:45:00 | 2562.70 | 2544.89 | 2528.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 2496.80 | 2527.81 | 2528.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 2496.80 | 2527.81 | 2528.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 2482.10 | 2518.67 | 2524.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2423.50 | 2414.57 | 2448.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:30:00 | 2413.70 | 2414.57 | 2448.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2431.70 | 2416.02 | 2432.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 2431.70 | 2416.02 | 2432.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 2420.60 | 2416.94 | 2431.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:45:00 | 2422.00 | 2416.94 | 2431.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 2374.10 | 2389.32 | 2408.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:45:00 | 2398.00 | 2389.32 | 2408.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 2383.90 | 2370.89 | 2383.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 2383.90 | 2370.89 | 2383.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 2340.70 | 2364.85 | 2379.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 2326.90 | 2351.66 | 2369.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 2380.50 | 2346.55 | 2345.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 2380.50 | 2346.55 | 2345.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 2395.00 | 2364.79 | 2355.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 2332.50 | 2369.20 | 2363.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 2332.50 | 2369.20 | 2363.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 2332.50 | 2369.20 | 2363.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 2440.00 | 2373.69 | 2369.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 10:00:00 | 2430.00 | 2419.81 | 2402.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-05 15:15:00 | 2684.00 | 2433.26 | 2427.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 2411.30 | 2421.50 | 2422.69 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 2439.90 | 2425.68 | 2424.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 2442.90 | 2431.08 | 2426.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 14:15:00 | 2427.00 | 2431.66 | 2427.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 14:15:00 | 2427.00 | 2431.66 | 2427.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 2427.00 | 2431.66 | 2427.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 2427.00 | 2431.66 | 2427.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 2415.70 | 2428.47 | 2426.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 2427.70 | 2428.47 | 2426.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 2437.40 | 2430.25 | 2427.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 11:00:00 | 2449.20 | 2434.04 | 2429.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 2418.20 | 2448.68 | 2450.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 2418.20 | 2448.68 | 2450.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 2410.00 | 2428.12 | 2438.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 2380.00 | 2376.26 | 2398.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 2380.00 | 2376.26 | 2398.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 2367.00 | 2358.71 | 2369.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 2382.60 | 2358.71 | 2369.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2364.10 | 2359.79 | 2369.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:30:00 | 2373.00 | 2359.79 | 2369.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 2360.40 | 2359.91 | 2368.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:45:00 | 2363.20 | 2359.91 | 2368.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2338.40 | 2348.03 | 2358.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:00:00 | 2330.00 | 2340.22 | 2352.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 2348.00 | 2320.31 | 2319.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 2348.00 | 2320.31 | 2319.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 2380.60 | 2355.54 | 2340.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 2363.40 | 2370.82 | 2357.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 2363.40 | 2370.82 | 2357.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 2363.40 | 2370.82 | 2357.81 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 2342.30 | 2350.53 | 2351.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 2320.00 | 2344.34 | 2348.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 14:15:00 | 2336.40 | 2329.97 | 2338.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 2336.40 | 2329.97 | 2338.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 2336.40 | 2329.97 | 2338.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 2336.40 | 2329.97 | 2338.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 2333.70 | 2330.72 | 2338.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 2280.00 | 2330.72 | 2338.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:15:00 | 2166.00 | 2203.02 | 2239.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 2052.00 | 2102.06 | 2146.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 09:15:00 | 2239.30 | 2124.99 | 2114.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 10:15:00 | 2356.40 | 2171.27 | 2136.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 14:15:00 | 2156.00 | 2188.09 | 2157.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 14:15:00 | 2156.00 | 2188.09 | 2157.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 2156.00 | 2188.09 | 2157.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 15:00:00 | 2156.00 | 2188.09 | 2157.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 2144.50 | 2179.38 | 2156.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:30:00 | 2109.80 | 2164.68 | 2151.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 2101.10 | 2151.96 | 2147.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 2093.90 | 2151.96 | 2147.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 2103.80 | 2142.33 | 2143.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 2095.70 | 2133.01 | 2138.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 11:15:00 | 2110.00 | 2097.37 | 2115.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 11:15:00 | 2110.00 | 2097.37 | 2115.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 2110.00 | 2097.37 | 2115.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 2110.00 | 2097.37 | 2115.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 2116.00 | 2101.09 | 2115.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 2116.00 | 2101.09 | 2115.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 2125.80 | 2106.04 | 2116.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:30:00 | 2125.90 | 2106.04 | 2116.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 2144.50 | 2113.73 | 2118.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 2139.60 | 2113.73 | 2118.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 2146.30 | 2123.97 | 2122.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 10:15:00 | 2163.60 | 2131.89 | 2126.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2164.90 | 2208.28 | 2184.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 2164.90 | 2208.28 | 2184.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 2164.90 | 2208.28 | 2184.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 2164.90 | 2208.28 | 2184.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 2157.00 | 2198.02 | 2181.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 2157.00 | 2198.02 | 2181.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 2175.10 | 2187.02 | 2180.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:45:00 | 2174.10 | 2187.02 | 2180.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2250.00 | 2198.35 | 2186.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:45:00 | 2275.10 | 2214.42 | 2195.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:45:00 | 2277.50 | 2226.54 | 2202.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 13:15:00 | 2187.60 | 2210.00 | 2211.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 13:15:00 | 2187.60 | 2210.00 | 2211.86 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 09:15:00 | 2223.50 | 2212.82 | 2212.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 2337.70 | 2281.53 | 2251.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2287.40 | 2314.37 | 2288.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2287.40 | 2314.37 | 2288.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2287.40 | 2314.37 | 2288.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 2286.00 | 2314.37 | 2288.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 2279.00 | 2307.29 | 2287.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 2278.70 | 2307.29 | 2287.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 2279.10 | 2301.65 | 2286.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 2279.10 | 2301.65 | 2286.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 2271.70 | 2293.91 | 2285.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 2271.70 | 2293.91 | 2285.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 2258.00 | 2286.73 | 2283.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 2258.00 | 2286.73 | 2283.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 2252.00 | 2279.78 | 2280.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2182.50 | 2260.33 | 2271.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 2300.00 | 2242.04 | 2254.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 2300.00 | 2242.04 | 2254.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 2300.00 | 2242.04 | 2254.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 2300.00 | 2242.04 | 2254.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 2296.10 | 2252.85 | 2257.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 2322.10 | 2252.85 | 2257.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 2320.60 | 2266.40 | 2263.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 2384.00 | 2355.89 | 2341.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 15:15:00 | 2370.00 | 2372.03 | 2357.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:15:00 | 2386.00 | 2372.03 | 2357.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 2369.00 | 2371.43 | 2358.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 2369.00 | 2371.43 | 2358.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2385.00 | 2402.90 | 2391.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2393.90 | 2402.90 | 2391.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 2372.80 | 2387.13 | 2387.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 2372.80 | 2387.13 | 2387.69 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 2421.30 | 2393.14 | 2390.28 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 2378.10 | 2389.25 | 2389.98 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 2429.80 | 2394.34 | 2391.25 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 2382.90 | 2391.70 | 2391.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 2367.00 | 2386.76 | 2389.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 2389.60 | 2383.11 | 2386.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 2389.60 | 2383.11 | 2386.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 2389.60 | 2383.11 | 2386.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 2387.70 | 2383.11 | 2386.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 2404.40 | 2387.37 | 2388.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:00:00 | 2404.40 | 2387.37 | 2388.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 2399.40 | 2389.77 | 2389.43 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 2354.90 | 2383.72 | 2387.00 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 2406.70 | 2392.17 | 2390.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 2417.80 | 2397.29 | 2392.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 2396.50 | 2399.17 | 2394.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 2396.50 | 2399.17 | 2394.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2396.50 | 2399.17 | 2394.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 2396.70 | 2399.17 | 2394.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 2390.90 | 2397.51 | 2394.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 2389.50 | 2397.51 | 2394.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 2390.90 | 2396.19 | 2393.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 2391.00 | 2396.19 | 2393.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 2385.40 | 2394.03 | 2393.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:30:00 | 2385.00 | 2394.03 | 2393.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 2385.20 | 2392.27 | 2392.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 2383.50 | 2390.51 | 2391.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 13:15:00 | 2324.60 | 2324.40 | 2342.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 14:00:00 | 2324.60 | 2324.40 | 2342.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 2324.80 | 2306.19 | 2318.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 2324.80 | 2306.19 | 2318.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 2320.10 | 2308.97 | 2318.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:00:00 | 2315.20 | 2311.66 | 2318.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 2313.80 | 2312.33 | 2318.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 2344.90 | 2310.93 | 2310.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 2344.90 | 2310.93 | 2310.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 2350.20 | 2330.19 | 2325.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 2356.50 | 2360.24 | 2350.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 2356.50 | 2360.24 | 2350.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 2354.00 | 2359.00 | 2351.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 2344.60 | 2359.00 | 2351.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 2363.60 | 2359.92 | 2352.22 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 2326.20 | 2345.59 | 2347.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-11 09:15:00 | 2325.90 | 2339.99 | 2344.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-11 12:15:00 | 2336.70 | 2335.44 | 2340.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-11 12:15:00 | 2336.70 | 2335.44 | 2340.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 12:15:00 | 2336.70 | 2335.44 | 2340.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-11 12:30:00 | 2338.70 | 2335.44 | 2340.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 13:15:00 | 2330.00 | 2334.35 | 2339.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-11 13:30:00 | 2332.50 | 2334.35 | 2339.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 14:15:00 | 2323.20 | 2332.12 | 2338.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-11 15:00:00 | 2323.20 | 2332.12 | 2338.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 09:15:00 | 2863.70 | 2025-05-19 13:15:00 | 2790.10 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-05-16 10:15:00 | 2837.90 | 2025-05-19 13:15:00 | 2790.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-05-16 10:45:00 | 2839.80 | 2025-05-19 13:15:00 | 2790.10 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-05-16 11:15:00 | 2841.30 | 2025-05-19 13:15:00 | 2790.10 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest1 | 2025-05-22 11:15:00 | 2535.00 | 2025-05-27 15:15:00 | 2408.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-23 13:30:00 | 2541.80 | 2025-05-27 15:15:00 | 2414.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-23 15:15:00 | 2537.00 | 2025-05-27 15:15:00 | 2410.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-26 13:15:00 | 2508.40 | 2025-05-27 15:15:00 | 2382.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-22 11:15:00 | 2535.00 | 2025-05-30 09:15:00 | 2407.70 | STOP_HIT | 0.50 | 5.02% |
| SELL | retest1 | 2025-05-23 13:30:00 | 2541.80 | 2025-05-30 09:15:00 | 2407.70 | STOP_HIT | 0.50 | 5.28% |
| SELL | retest1 | 2025-05-23 15:15:00 | 2537.00 | 2025-05-30 09:15:00 | 2407.70 | STOP_HIT | 0.50 | 5.10% |
| SELL | retest2 | 2025-05-26 13:15:00 | 2508.40 | 2025-05-30 09:15:00 | 2407.70 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2025-06-09 10:30:00 | 2423.50 | 2025-06-16 09:15:00 | 2302.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 11:45:00 | 2424.50 | 2025-06-16 09:15:00 | 2303.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 13:00:00 | 2424.10 | 2025-06-16 09:15:00 | 2302.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 15:15:00 | 2413.00 | 2025-06-16 09:15:00 | 2292.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 10:30:00 | 2423.50 | 2025-06-16 13:15:00 | 2335.00 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-06-09 11:45:00 | 2424.50 | 2025-06-16 13:15:00 | 2335.00 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-06-09 13:00:00 | 2424.10 | 2025-06-16 13:15:00 | 2335.00 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2025-06-09 15:15:00 | 2413.00 | 2025-06-16 13:15:00 | 2335.00 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-06-10 13:15:00 | 2401.70 | 2025-06-17 10:15:00 | 2380.90 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-06-11 10:30:00 | 2400.30 | 2025-06-17 10:15:00 | 2380.90 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-06-19 15:15:00 | 2405.00 | 2025-06-30 09:15:00 | 2455.40 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2025-07-01 11:15:00 | 2494.70 | 2025-07-01 13:15:00 | 2482.70 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-08 09:45:00 | 2430.40 | 2025-07-15 11:15:00 | 2411.80 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2430.20 | 2025-07-15 11:15:00 | 2411.80 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2025-07-08 15:00:00 | 2428.00 | 2025-07-15 11:15:00 | 2411.80 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-07-09 12:15:00 | 2429.80 | 2025-07-15 11:15:00 | 2411.80 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-07-10 09:15:00 | 2412.10 | 2025-07-15 11:15:00 | 2411.80 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-07-17 09:15:00 | 2401.10 | 2025-07-18 10:15:00 | 2384.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-17 10:00:00 | 2401.00 | 2025-07-18 10:15:00 | 2384.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-22 10:45:00 | 2362.30 | 2025-07-22 14:15:00 | 2390.70 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-07-30 09:30:00 | 2342.80 | 2025-07-31 11:15:00 | 2356.70 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-07-31 10:00:00 | 2342.00 | 2025-07-31 11:15:00 | 2356.70 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-08-06 12:15:00 | 2387.00 | 2025-08-07 09:15:00 | 2352.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-12 09:30:00 | 2564.60 | 2025-09-12 13:15:00 | 2610.50 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-09-19 10:30:00 | 2592.20 | 2025-09-19 14:15:00 | 2631.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-19 11:30:00 | 2592.70 | 2025-09-19 14:15:00 | 2631.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-09-30 10:15:00 | 2491.10 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2025-10-01 09:15:00 | 2492.90 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.56% |
| SELL | retest2 | 2025-10-03 09:15:00 | 2498.10 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-10-06 09:30:00 | 2494.70 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.49% |
| SELL | retest2 | 2025-10-06 13:15:00 | 2485.00 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2025-10-06 15:00:00 | 2485.90 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2025-10-14 11:15:00 | 2525.20 | 2025-10-21 14:15:00 | 2523.80 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-10-16 09:45:00 | 2534.50 | 2025-10-21 14:15:00 | 2523.80 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-10-28 14:00:00 | 2532.20 | 2025-11-04 13:15:00 | 2509.60 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-29 12:30:00 | 2529.90 | 2025-11-04 13:15:00 | 2509.60 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-10-29 13:15:00 | 2528.30 | 2025-11-04 13:15:00 | 2509.60 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-30 10:30:00 | 2529.90 | 2025-11-04 13:15:00 | 2509.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-10 13:30:00 | 2501.00 | 2025-11-10 14:15:00 | 2537.20 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-11-10 15:15:00 | 2490.00 | 2025-11-11 10:15:00 | 2552.90 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-11-20 11:45:00 | 2540.10 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-20 13:15:00 | 2539.40 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-20 13:45:00 | 2540.00 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-20 14:30:00 | 2539.50 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-21 10:15:00 | 2528.20 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-11-21 11:00:00 | 2514.40 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-12-09 09:15:00 | 2536.60 | 2025-12-12 11:15:00 | 2569.30 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-12-10 10:15:00 | 2573.30 | 2025-12-12 11:15:00 | 2569.30 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2026-01-02 11:45:00 | 2618.10 | 2026-01-07 10:15:00 | 2600.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-14 12:15:00 | 2511.00 | 2026-01-14 13:15:00 | 2516.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-01-14 12:45:00 | 2515.00 | 2026-01-14 13:15:00 | 2516.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2026-01-19 12:45:00 | 2562.70 | 2026-01-20 11:15:00 | 2496.80 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-01-29 09:30:00 | 2326.90 | 2026-01-30 13:15:00 | 2380.50 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-02-03 09:15:00 | 2440.00 | 2026-02-05 15:15:00 | 2684.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 10:00:00 | 2430.00 | 2026-02-05 15:15:00 | 2673.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-10 11:00:00 | 2449.20 | 2026-02-12 10:15:00 | 2418.20 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-02-19 13:00:00 | 2330.00 | 2026-02-24 10:15:00 | 2348.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2280.00 | 2026-03-05 10:15:00 | 2166.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2280.00 | 2026-03-09 09:15:00 | 2052.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-20 10:45:00 | 2275.10 | 2026-03-23 13:15:00 | 2187.60 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2026-03-20 11:45:00 | 2277.50 | 2026-03-23 13:15:00 | 2187.60 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2393.90 | 2026-04-13 14:15:00 | 2372.80 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-29 13:00:00 | 2315.20 | 2026-05-04 10:15:00 | 2344.90 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-04-29 13:30:00 | 2313.80 | 2026-05-04 10:15:00 | 2344.90 | STOP_HIT | 1.00 | -1.34% |
