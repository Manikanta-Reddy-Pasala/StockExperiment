# Deepak Nitrite Ltd. (DEEPAKNTR)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1875.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 90 |
| ALERT2 | 90 |
| ALERT2_SKIP | 42 |
| ALERT3 | 254 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 105 |
| PARTIAL | 13 |
| TARGET_HIT | 8 |
| STOP_HIT | 98 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 119 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 49 / 70
- **Target hits / Stop hits / Partials:** 8 / 98 / 13
- **Avg / median % per leg:** 1.14% / -0.42%
- **Sum % (uncompounded):** 136.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 16 | 35.6% | 6 | 39 | 0 | 1.05% | 47.3% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.89% | 0.9% |
| BUY @ 3rd Alert (retest2) | 44 | 15 | 34.1% | 6 | 38 | 0 | 1.05% | 46.4% |
| SELL (all) | 74 | 33 | 44.6% | 2 | 59 | 13 | 1.20% | 89.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.09% | -1.1% |
| SELL @ 3rd Alert (retest2) | 73 | 33 | 45.2% | 2 | 58 | 13 | 1.23% | 90.0% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.10% | -0.2% |
| retest2 (combined) | 117 | 48 | 41.0% | 8 | 96 | 13 | 1.17% | 136.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 2485.00 | 2470.18 | 2469.30 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 2452.40 | 2467.55 | 2469.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 11:15:00 | 2437.45 | 2458.25 | 2464.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 2441.30 | 2439.33 | 2449.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 12:00:00 | 2441.30 | 2439.33 | 2449.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 2447.25 | 2439.17 | 2446.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 2447.25 | 2439.17 | 2446.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 2453.00 | 2441.94 | 2447.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 2470.65 | 2441.94 | 2447.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 2468.95 | 2447.34 | 2449.22 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 2482.90 | 2454.45 | 2452.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 2518.00 | 2479.16 | 2466.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 12:15:00 | 2512.35 | 2513.23 | 2499.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 13:00:00 | 2512.35 | 2513.23 | 2499.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 2495.15 | 2509.61 | 2499.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 2495.15 | 2509.61 | 2499.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 2471.50 | 2501.99 | 2496.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:45:00 | 2468.00 | 2501.99 | 2496.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 2460.00 | 2493.59 | 2493.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 2406.50 | 2493.59 | 2493.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 2387.00 | 2472.27 | 2483.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 10:15:00 | 2352.50 | 2448.32 | 2471.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 12:15:00 | 2371.50 | 2366.63 | 2401.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 12:45:00 | 2370.60 | 2366.63 | 2401.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 2222.65 | 2198.89 | 2220.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 2178.20 | 2227.19 | 2227.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:45:00 | 2191.15 | 2216.84 | 2222.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 2069.29 | 2191.78 | 2209.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 2081.59 | 2191.78 | 2209.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 2197.80 | 2176.39 | 2195.17 | SL hit (close>ema200) qty=0.50 sl=2176.39 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 2228.30 | 2207.95 | 2205.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 2267.35 | 2222.51 | 2212.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 13:15:00 | 2412.65 | 2413.19 | 2384.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 14:00:00 | 2412.65 | 2413.19 | 2384.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 2393.30 | 2409.75 | 2390.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 2393.30 | 2409.75 | 2390.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 2408.90 | 2409.58 | 2391.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 12:00:00 | 2417.05 | 2411.08 | 2394.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 14:15:00 | 2415.90 | 2414.75 | 2398.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 15:15:00 | 2505.95 | 2524.02 | 2525.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 2505.95 | 2524.02 | 2525.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 2500.55 | 2519.32 | 2523.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 11:15:00 | 2480.80 | 2479.51 | 2494.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 11:15:00 | 2480.80 | 2479.51 | 2494.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 2480.80 | 2479.51 | 2494.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:45:00 | 2483.60 | 2479.51 | 2494.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 2508.40 | 2486.17 | 2495.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:00:00 | 2508.40 | 2486.17 | 2495.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 2509.50 | 2490.84 | 2496.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:15:00 | 2522.95 | 2490.84 | 2496.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 2522.95 | 2497.26 | 2499.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 2500.00 | 2497.26 | 2499.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 2521.25 | 2502.06 | 2501.04 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 2481.20 | 2497.43 | 2499.24 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 2511.00 | 2501.90 | 2501.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 2553.00 | 2516.89 | 2510.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 10:15:00 | 2663.00 | 2669.91 | 2628.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 11:00:00 | 2663.00 | 2669.91 | 2628.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 2645.10 | 2659.81 | 2640.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:45:00 | 2685.15 | 2662.50 | 2645.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:45:00 | 2687.75 | 2666.12 | 2648.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:45:00 | 2680.45 | 2669.03 | 2651.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:30:00 | 2718.90 | 2678.17 | 2660.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 2698.40 | 2692.22 | 2678.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 2675.85 | 2692.22 | 2678.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 2689.25 | 2692.47 | 2681.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:30:00 | 2682.45 | 2692.47 | 2681.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 2674.80 | 2688.93 | 2681.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:30:00 | 2678.60 | 2688.93 | 2681.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 2694.80 | 2690.11 | 2682.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 2712.65 | 2691.09 | 2683.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 13:30:00 | 2729.35 | 2705.72 | 2695.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 2661.55 | 2694.61 | 2692.95 | SL hit (close<static) qty=1.00 sl=2670.05 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 2678.00 | 2691.29 | 2691.59 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 11:15:00 | 2712.65 | 2695.56 | 2693.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 12:15:00 | 2720.20 | 2700.49 | 2695.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 15:15:00 | 2765.00 | 2769.14 | 2750.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 09:15:00 | 2772.80 | 2769.14 | 2750.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2765.55 | 2768.42 | 2752.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 2761.70 | 2768.42 | 2752.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 2800.00 | 2805.66 | 2792.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 2825.00 | 2805.66 | 2792.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 2823.15 | 2809.16 | 2794.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 10:15:00 | 2843.65 | 2823.21 | 2810.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 15:15:00 | 2764.90 | 2798.98 | 2802.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 15:15:00 | 2764.90 | 2798.98 | 2802.90 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 2817.65 | 2805.98 | 2805.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 12:15:00 | 2852.10 | 2815.20 | 2809.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 2823.75 | 2830.74 | 2820.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 2823.75 | 2830.74 | 2820.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 2823.75 | 2830.74 | 2820.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:00:00 | 2823.75 | 2830.74 | 2820.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 2864.50 | 2837.50 | 2824.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 10:00:00 | 2873.00 | 2841.22 | 2830.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-01 09:15:00 | 3160.30 | 3098.66 | 3057.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 2983.55 | 3078.52 | 3079.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 2975.45 | 3043.84 | 3062.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 3030.30 | 3008.84 | 3036.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 3030.30 | 3008.84 | 3036.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 3030.30 | 3008.84 | 3036.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:30:00 | 3024.00 | 3008.84 | 3036.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 3068.00 | 3020.67 | 3039.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 3068.00 | 3020.67 | 3039.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 3063.40 | 3029.21 | 3041.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 12:00:00 | 3063.40 | 3029.21 | 3041.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 3056.15 | 3034.60 | 3043.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 3041.05 | 3035.63 | 3042.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 3039.75 | 3035.63 | 3042.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 3035.40 | 3032.10 | 3040.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:30:00 | 3048.20 | 3029.59 | 3035.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 3076.30 | 3038.93 | 3039.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 12:15:00 | 3076.30 | 3038.93 | 3039.62 | SL hit (close>static) qty=1.00 sl=3068.95 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 3107.25 | 3052.59 | 3045.77 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 3028.35 | 3067.09 | 3070.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 3005.50 | 3045.99 | 3059.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 2878.50 | 2875.98 | 2931.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 13:00:00 | 2878.50 | 2875.98 | 2931.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 2896.65 | 2877.52 | 2900.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 2896.65 | 2877.52 | 2900.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 2903.10 | 2882.64 | 2900.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:45:00 | 2904.55 | 2882.64 | 2900.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 2902.00 | 2886.51 | 2900.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 2900.50 | 2886.51 | 2900.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 2886.45 | 2886.50 | 2899.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 2875.25 | 2886.50 | 2899.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 13:30:00 | 2876.55 | 2881.67 | 2893.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 09:30:00 | 2872.30 | 2878.30 | 2888.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 10:15:00 | 2900.85 | 2888.88 | 2887.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 2900.85 | 2888.88 | 2887.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 14:15:00 | 2952.70 | 2910.15 | 2898.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 2955.20 | 2966.74 | 2941.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:45:00 | 2949.05 | 2966.74 | 2941.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 2941.60 | 2961.71 | 2941.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 2895.00 | 2961.71 | 2941.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 2856.50 | 2940.67 | 2933.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 2856.50 | 2940.67 | 2933.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 2869.20 | 2926.38 | 2927.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 2821.05 | 2880.15 | 2903.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 2867.00 | 2837.14 | 2858.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 2867.00 | 2837.14 | 2858.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 2867.00 | 2837.14 | 2858.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 2877.75 | 2837.14 | 2858.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 2880.00 | 2845.72 | 2860.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 2877.00 | 2845.72 | 2860.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 2869.25 | 2864.41 | 2865.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 2854.95 | 2864.41 | 2865.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 2865.90 | 2864.71 | 2865.92 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 2891.60 | 2870.09 | 2868.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 2911.95 | 2899.42 | 2889.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 2902.00 | 2906.45 | 2896.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 12:00:00 | 2902.00 | 2906.45 | 2896.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 2913.15 | 2907.79 | 2897.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 2907.65 | 2907.79 | 2897.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 2941.60 | 2916.33 | 2905.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 2956.90 | 2927.31 | 2916.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 2928.85 | 2967.68 | 2969.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 2928.85 | 2967.68 | 2969.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 2913.15 | 2951.58 | 2961.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 11:15:00 | 2950.40 | 2949.75 | 2958.67 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 12:15:00 | 2935.00 | 2949.75 | 2958.67 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 2967.00 | 2947.90 | 2953.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 2967.00 | 2947.90 | 2953.19 | SL hit (close>ema400) qty=1.00 sl=2953.19 alert=retest1 |

### Cycle 21 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 2975.60 | 2958.20 | 2956.58 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 2931.95 | 2953.82 | 2955.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 2916.00 | 2946.26 | 2952.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 11:15:00 | 2937.15 | 2935.55 | 2944.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-12 12:00:00 | 2937.15 | 2935.55 | 2944.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 2954.60 | 2933.49 | 2939.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:45:00 | 2968.00 | 2933.49 | 2939.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 2945.10 | 2935.81 | 2939.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 11:15:00 | 2960.50 | 2935.81 | 2939.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 2934.45 | 2939.86 | 2940.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 15:15:00 | 2925.15 | 2939.86 | 2940.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 14:30:00 | 2926.35 | 2929.30 | 2933.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 2927.20 | 2930.44 | 2933.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 15:15:00 | 2869.00 | 2862.92 | 2862.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 15:15:00 | 2869.00 | 2862.92 | 2862.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 2890.55 | 2868.44 | 2864.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 13:15:00 | 2870.00 | 2872.60 | 2868.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 13:15:00 | 2870.00 | 2872.60 | 2868.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 2870.00 | 2872.60 | 2868.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:00:00 | 2870.00 | 2872.60 | 2868.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 2860.70 | 2870.22 | 2867.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 2860.70 | 2870.22 | 2867.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 2859.00 | 2867.98 | 2867.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 2856.50 | 2867.98 | 2867.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 2841.00 | 2862.58 | 2864.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 12:15:00 | 2820.40 | 2849.74 | 2858.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 2817.80 | 2812.74 | 2830.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 15:00:00 | 2817.80 | 2812.74 | 2830.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 2823.50 | 2814.89 | 2829.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 2861.15 | 2814.89 | 2829.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 2829.65 | 2817.85 | 2829.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 2839.70 | 2817.85 | 2829.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 2881.00 | 2830.48 | 2834.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 2881.00 | 2830.48 | 2834.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 11:15:00 | 2872.20 | 2838.82 | 2837.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 13:15:00 | 2902.65 | 2858.76 | 2847.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 2936.30 | 2944.30 | 2921.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 11:00:00 | 2936.30 | 2944.30 | 2921.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 2902.05 | 2935.85 | 2919.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 2902.05 | 2935.85 | 2919.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 2890.00 | 2926.68 | 2916.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 2890.00 | 2926.68 | 2916.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 14:15:00 | 2886.00 | 2909.15 | 2910.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 2855.95 | 2891.91 | 2900.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 2804.65 | 2792.01 | 2824.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 2804.65 | 2792.01 | 2824.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2809.45 | 2796.04 | 2812.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 2809.45 | 2796.04 | 2812.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 2806.70 | 2798.18 | 2811.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 2818.35 | 2798.18 | 2811.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 2799.80 | 2798.50 | 2810.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 2791.60 | 2797.12 | 2808.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 2822.00 | 2798.85 | 2805.48 | SL hit (close>static) qty=1.00 sl=2814.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 2830.80 | 2801.30 | 2798.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 14:15:00 | 2846.00 | 2821.42 | 2811.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 2924.35 | 2956.68 | 2921.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 2924.35 | 2956.68 | 2921.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 2924.35 | 2956.68 | 2921.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 2924.35 | 2956.68 | 2921.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 2886.45 | 2942.63 | 2918.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 2886.45 | 2942.63 | 2918.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 2863.70 | 2926.84 | 2913.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 2863.70 | 2926.84 | 2913.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 2857.55 | 2904.40 | 2905.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 2840.00 | 2884.17 | 2895.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 2711.00 | 2708.04 | 2757.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 2711.00 | 2708.04 | 2757.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 2732.25 | 2719.05 | 2753.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:30:00 | 2715.25 | 2736.28 | 2751.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 2727.10 | 2734.45 | 2748.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:00:00 | 2718.45 | 2731.25 | 2746.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 10:15:00 | 2728.10 | 2702.80 | 2700.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 2728.10 | 2702.80 | 2700.75 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 2666.30 | 2697.08 | 2700.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 11:15:00 | 2654.55 | 2688.58 | 2696.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-04 14:15:00 | 2645.25 | 2640.68 | 2655.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-04 15:00:00 | 2645.25 | 2640.68 | 2655.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 2640.60 | 2640.79 | 2653.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:30:00 | 2644.85 | 2640.79 | 2653.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 2653.25 | 2643.29 | 2653.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:45:00 | 2656.85 | 2643.29 | 2653.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 2631.30 | 2640.89 | 2651.10 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 2693.00 | 2659.22 | 2657.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 2752.10 | 2684.32 | 2669.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 12:15:00 | 2809.00 | 2810.95 | 2763.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 13:00:00 | 2809.00 | 2810.95 | 2763.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 2786.55 | 2806.31 | 2776.71 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 2692.50 | 2755.44 | 2759.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 2680.90 | 2740.53 | 2752.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 2578.25 | 2536.81 | 2575.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 2578.25 | 2536.81 | 2575.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 2578.25 | 2536.81 | 2575.66 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 14:15:00 | 2641.80 | 2599.29 | 2595.03 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 10:15:00 | 2547.45 | 2584.65 | 2589.49 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 2640.40 | 2590.91 | 2590.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 2662.55 | 2605.24 | 2596.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 2649.95 | 2651.84 | 2629.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 2649.95 | 2651.84 | 2629.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 2649.95 | 2651.84 | 2629.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 13:00:00 | 2683.00 | 2649.23 | 2638.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 15:15:00 | 2676.35 | 2658.71 | 2645.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 12:15:00 | 2627.85 | 2723.41 | 2728.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 12:15:00 | 2627.85 | 2723.41 | 2728.51 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 2710.45 | 2704.17 | 2703.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 2716.10 | 2707.92 | 2705.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 11:15:00 | 2707.55 | 2708.56 | 2706.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 11:45:00 | 2708.55 | 2708.56 | 2706.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 2715.10 | 2709.87 | 2707.03 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 2663.65 | 2698.06 | 2702.34 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 2713.95 | 2697.27 | 2696.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 2728.95 | 2710.48 | 2703.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 2691.00 | 2724.12 | 2718.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 2691.00 | 2724.12 | 2718.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 2691.00 | 2724.12 | 2718.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 2691.00 | 2724.12 | 2718.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 2668.65 | 2713.03 | 2714.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 2650.30 | 2684.48 | 2697.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 2676.90 | 2665.83 | 2681.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 2676.90 | 2665.83 | 2681.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 2676.90 | 2665.83 | 2681.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 2676.90 | 2665.83 | 2681.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 2675.20 | 2667.71 | 2680.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 2669.75 | 2667.71 | 2680.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 2683.00 | 2670.77 | 2681.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 2687.70 | 2670.77 | 2681.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 2678.95 | 2672.40 | 2680.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:45:00 | 2683.95 | 2672.40 | 2680.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 2688.70 | 2676.35 | 2681.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 2688.70 | 2676.35 | 2681.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 2702.70 | 2681.62 | 2683.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 2702.70 | 2681.62 | 2683.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 2685.00 | 2684.30 | 2684.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 2700.00 | 2687.44 | 2685.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 2685.75 | 2687.10 | 2685.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 10:15:00 | 2685.75 | 2687.10 | 2685.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 2685.75 | 2687.10 | 2685.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 2695.00 | 2687.10 | 2685.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 2670.75 | 2683.83 | 2684.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 2666.20 | 2680.31 | 2682.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 2638.70 | 2637.54 | 2651.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 11:45:00 | 2636.60 | 2637.54 | 2651.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2622.60 | 2634.88 | 2645.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:30:00 | 2621.25 | 2629.29 | 2639.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 11:45:00 | 2618.40 | 2612.85 | 2627.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:30:00 | 2615.80 | 2610.03 | 2617.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 15:15:00 | 2490.19 | 2513.81 | 2533.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 09:15:00 | 2487.48 | 2511.64 | 2530.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 09:15:00 | 2485.01 | 2511.64 | 2530.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-01 10:15:00 | 2512.00 | 2511.71 | 2529.01 | SL hit (close>ema200) qty=0.50 sl=2511.71 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 2466.55 | 2452.83 | 2451.68 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 2424.45 | 2452.11 | 2452.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 2376.85 | 2421.32 | 2435.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 2364.45 | 2360.19 | 2388.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 2364.45 | 2360.19 | 2388.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 2364.65 | 2348.79 | 2367.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:00:00 | 2364.65 | 2348.79 | 2367.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 2366.00 | 2352.23 | 2367.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:45:00 | 2361.10 | 2352.23 | 2367.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 2360.40 | 2353.86 | 2366.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:30:00 | 2366.00 | 2353.86 | 2366.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 2358.00 | 2357.20 | 2364.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:45:00 | 2382.00 | 2357.20 | 2364.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 2345.85 | 2354.93 | 2362.69 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 09:15:00 | 2383.20 | 2364.20 | 2363.77 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 2358.30 | 2363.13 | 2363.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 2330.45 | 2356.57 | 2360.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 2351.75 | 2350.61 | 2356.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 13:00:00 | 2351.75 | 2350.61 | 2356.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 2355.90 | 2351.67 | 2356.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 2355.90 | 2351.67 | 2356.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 2360.75 | 2353.49 | 2356.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:45:00 | 2366.70 | 2353.49 | 2356.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 2360.00 | 2354.79 | 2357.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 2403.00 | 2354.79 | 2357.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 2367.55 | 2357.34 | 2358.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:30:00 | 2393.20 | 2357.34 | 2358.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 11:15:00 | 2367.50 | 2359.85 | 2359.06 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 2350.00 | 2358.57 | 2358.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 2346.50 | 2356.15 | 2357.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 2319.25 | 2313.35 | 2331.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 2319.25 | 2313.35 | 2331.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2363.90 | 2323.64 | 2332.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 2363.90 | 2323.64 | 2332.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2367.30 | 2332.38 | 2335.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 2367.30 | 2332.38 | 2335.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 2366.30 | 2343.47 | 2340.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 14:15:00 | 2370.05 | 2351.50 | 2344.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 2317.05 | 2346.77 | 2343.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 2317.05 | 2346.77 | 2343.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 2317.05 | 2346.77 | 2343.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 2317.05 | 2346.77 | 2343.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 2331.30 | 2343.68 | 2342.82 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 2329.75 | 2340.89 | 2341.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 2301.85 | 2330.63 | 2336.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 2238.95 | 2215.09 | 2241.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 2238.95 | 2215.09 | 2241.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 2238.95 | 2215.09 | 2241.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 2238.95 | 2215.09 | 2241.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 2259.45 | 2223.96 | 2243.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 2259.45 | 2223.96 | 2243.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 2262.90 | 2231.75 | 2244.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 2274.15 | 2231.75 | 2244.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 2280.40 | 2253.87 | 2252.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 2300.40 | 2266.54 | 2258.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 2241.60 | 2265.18 | 2261.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 2241.60 | 2265.18 | 2261.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 2241.60 | 2265.18 | 2261.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 2241.60 | 2265.18 | 2261.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 2271.20 | 2266.38 | 2262.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:30:00 | 2318.40 | 2274.81 | 2266.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:15:00 | 2313.75 | 2281.18 | 2270.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:45:00 | 2312.20 | 2319.06 | 2302.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:45:00 | 2311.10 | 2320.66 | 2304.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 2362.20 | 2330.01 | 2313.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 2378.70 | 2336.01 | 2317.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:45:00 | 2366.20 | 2354.65 | 2344.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 11:00:00 | 2378.95 | 2359.51 | 2347.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 12:15:00 | 2375.00 | 2360.15 | 2348.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 2335.55 | 2363.04 | 2355.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 2335.55 | 2363.04 | 2355.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 2349.60 | 2360.35 | 2355.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 2334.00 | 2360.35 | 2355.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 2347.85 | 2357.85 | 2354.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:00:00 | 2347.85 | 2357.85 | 2354.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 2349.20 | 2356.12 | 2353.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 2337.10 | 2352.32 | 2352.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 2337.10 | 2352.32 | 2352.43 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 14:15:00 | 2355.60 | 2352.97 | 2352.72 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 2309.15 | 2344.05 | 2348.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 2303.45 | 2335.93 | 2344.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 2218.90 | 2213.93 | 2250.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 2219.30 | 2213.93 | 2250.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2249.30 | 2217.05 | 2237.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 2249.30 | 2217.05 | 2237.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 2245.45 | 2222.73 | 2238.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 2245.45 | 2222.73 | 2238.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 2236.70 | 2225.52 | 2238.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 2228.15 | 2226.05 | 2237.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 2024.00 | 2229.15 | 2235.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-14 09:15:00 | 2005.34 | 2170.93 | 2208.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 1973.20 | 1944.79 | 1941.14 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 1930.85 | 1943.56 | 1943.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 1920.85 | 1935.81 | 1939.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 1935.00 | 1934.10 | 1938.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 1935.00 | 1934.10 | 1938.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 1935.00 | 1934.10 | 1938.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 1937.45 | 1934.10 | 1938.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1958.55 | 1938.99 | 1940.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 1958.55 | 1938.99 | 1940.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 13:15:00 | 1951.50 | 1941.49 | 1941.12 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 14:15:00 | 1924.00 | 1942.51 | 1943.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 15:15:00 | 1918.45 | 1937.69 | 1941.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 1847.95 | 1834.18 | 1860.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 1847.95 | 1834.18 | 1860.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1859.90 | 1845.12 | 1857.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1859.90 | 1845.12 | 1857.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1872.45 | 1850.58 | 1859.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 1872.45 | 1850.58 | 1859.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 1849.45 | 1851.91 | 1858.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:45:00 | 1861.60 | 1851.91 | 1858.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 1861.45 | 1853.82 | 1858.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 1861.45 | 1853.82 | 1858.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 1858.85 | 1854.83 | 1858.70 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 1903.00 | 1865.96 | 1863.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 1924.00 | 1877.57 | 1868.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 1971.40 | 1975.62 | 1958.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:00:00 | 1971.40 | 1975.62 | 1958.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1947.35 | 1967.52 | 1958.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 1947.35 | 1967.52 | 1958.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1940.00 | 1962.02 | 1956.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1958.55 | 1962.02 | 1956.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1964.95 | 1965.62 | 1960.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:45:00 | 1960.80 | 1965.62 | 1960.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 1951.05 | 1962.71 | 1959.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 1951.05 | 1962.71 | 1959.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 1953.10 | 1960.79 | 1958.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:30:00 | 1955.00 | 1960.79 | 1958.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 1960.90 | 1960.81 | 1958.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 1964.00 | 1960.28 | 1958.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 09:15:00 | 1944.10 | 1957.04 | 1957.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 1944.10 | 1957.04 | 1957.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 1927.00 | 1951.03 | 1954.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 1959.00 | 1943.24 | 1948.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 1959.00 | 1943.24 | 1948.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1959.00 | 1943.24 | 1948.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1959.00 | 1943.24 | 1948.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1958.00 | 1946.19 | 1949.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 1952.90 | 1946.19 | 1949.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1960.20 | 1950.31 | 1950.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:45:00 | 1954.85 | 1950.31 | 1950.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 11:15:00 | 1964.30 | 1953.11 | 1952.11 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 1947.00 | 1951.25 | 1951.60 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1985.70 | 1958.14 | 1954.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 2029.40 | 1996.97 | 1983.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 2031.00 | 2037.89 | 2021.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 2031.00 | 2037.89 | 2021.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 2032.55 | 2035.36 | 2023.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:00:00 | 2049.70 | 2037.04 | 2027.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 11:45:00 | 2049.95 | 2067.95 | 2059.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:00:00 | 2047.45 | 2058.87 | 2056.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 14:15:00 | 2035.00 | 2054.09 | 2054.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 2035.00 | 2054.09 | 2054.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 2025.00 | 2048.27 | 2051.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 15:15:00 | 2002.00 | 1998.44 | 2013.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 09:15:00 | 2021.95 | 1998.44 | 2013.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 2005.10 | 1999.78 | 2013.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 1983.10 | 2000.00 | 2011.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 1980.05 | 1979.01 | 1987.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 2020.10 | 1992.15 | 1991.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 2020.10 | 1992.15 | 1991.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 12:15:00 | 2025.90 | 1998.90 | 1994.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 1982.40 | 2008.74 | 2002.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 1982.40 | 2008.74 | 2002.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1982.40 | 2008.74 | 2002.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:45:00 | 1973.40 | 2008.74 | 2002.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1992.75 | 2005.54 | 2001.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:30:00 | 1995.10 | 2005.23 | 2001.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 1949.00 | 1992.43 | 1996.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1949.00 | 1992.43 | 1996.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 1940.00 | 1970.96 | 1984.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 1881.85 | 1824.36 | 1839.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 09:15:00 | 1881.85 | 1824.36 | 1839.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1881.85 | 1824.36 | 1839.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:45:00 | 1884.95 | 1824.36 | 1839.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 1906.70 | 1849.49 | 1848.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 1933.60 | 1880.89 | 1864.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 1961.20 | 1963.71 | 1948.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 1961.20 | 1963.71 | 1948.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1965.00 | 1962.74 | 1950.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:30:00 | 1976.60 | 1965.41 | 1952.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1953.90 | 2002.13 | 2007.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1953.90 | 2002.13 | 2007.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1932.40 | 1988.19 | 2000.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1974.70 | 1967.95 | 1982.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 1974.70 | 1967.95 | 1982.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 1975.30 | 1970.04 | 1979.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 1977.40 | 1970.04 | 1979.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 1975.00 | 1972.56 | 1978.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 1999.80 | 1972.56 | 1978.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1985.60 | 1975.17 | 1979.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:15:00 | 1977.10 | 1981.21 | 1981.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 14:15:00 | 1989.50 | 1982.87 | 1982.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 1989.50 | 1982.87 | 1982.21 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 11:15:00 | 1959.60 | 1977.93 | 1980.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 1945.10 | 1968.57 | 1975.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1992.30 | 1963.98 | 1970.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1992.30 | 1963.98 | 1970.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1992.30 | 1963.98 | 1970.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 1992.30 | 1963.98 | 1970.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 2001.20 | 1971.42 | 1973.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:00:00 | 2001.20 | 1971.42 | 1973.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 1978.90 | 1973.55 | 1974.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:00:00 | 1978.90 | 1973.55 | 1974.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 1986.30 | 1976.10 | 1975.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 14:15:00 | 1993.20 | 1979.52 | 1976.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 1990.20 | 2006.45 | 1997.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 1990.20 | 2006.45 | 1997.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1990.20 | 2006.45 | 1997.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 1990.20 | 2006.45 | 1997.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1988.50 | 2002.86 | 1996.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:15:00 | 1975.00 | 2002.86 | 1996.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 1970.10 | 1991.87 | 1992.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 1961.60 | 1985.81 | 1989.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 1984.00 | 1969.71 | 1978.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 1984.00 | 1969.71 | 1978.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1984.00 | 1969.71 | 1978.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 1984.00 | 1969.71 | 1978.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1978.70 | 1971.50 | 1978.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 1982.00 | 1971.50 | 1978.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 1986.10 | 1974.42 | 1978.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 1986.10 | 1974.42 | 1978.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1989.40 | 1977.42 | 1979.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:30:00 | 1981.90 | 1977.42 | 1979.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1985.50 | 1979.04 | 1980.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1979.30 | 1979.04 | 1980.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 1996.10 | 1982.45 | 1981.85 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 11:15:00 | 1968.30 | 1980.35 | 1981.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 12:15:00 | 1945.40 | 1973.36 | 1977.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1952.60 | 1920.23 | 1934.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1952.60 | 1920.23 | 1934.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1952.60 | 1920.23 | 1934.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 1982.70 | 1920.23 | 1934.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1973.30 | 1947.05 | 1944.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1981.80 | 1954.00 | 1948.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1958.40 | 1964.31 | 1955.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:45:00 | 1957.40 | 1964.31 | 1955.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 1954.80 | 1962.41 | 1955.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:00:00 | 1954.80 | 1962.41 | 1955.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1946.00 | 1959.13 | 1954.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 1946.00 | 1959.13 | 1954.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1945.00 | 1956.30 | 1953.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:15:00 | 1945.60 | 1956.30 | 1953.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1972.30 | 1958.04 | 1955.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:30:00 | 1982.00 | 1966.44 | 1960.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:00:00 | 1981.50 | 1969.46 | 1962.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1998.00 | 1971.10 | 1963.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 2060.50 | 2088.82 | 2090.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 2060.50 | 2088.82 | 2090.53 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 2129.70 | 2096.85 | 2093.56 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 14:15:00 | 2069.90 | 2090.07 | 2092.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 2058.80 | 2071.15 | 2077.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 2095.50 | 2067.40 | 2073.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 2095.50 | 2067.40 | 2073.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2095.50 | 2067.40 | 2073.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 2092.00 | 2067.40 | 2073.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 2102.00 | 2074.32 | 2076.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 2109.50 | 2074.32 | 2076.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 2071.10 | 2056.42 | 2064.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:00:00 | 2071.10 | 2056.42 | 2064.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 2048.30 | 2054.80 | 2062.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 11:15:00 | 2042.40 | 2054.80 | 2062.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 2108.80 | 2048.68 | 2051.85 | SL hit (close>static) qty=1.00 sl=2077.90 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 2121.00 | 2063.14 | 2058.14 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 2026.80 | 2061.74 | 2063.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1991.00 | 2031.30 | 2046.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 2014.90 | 2007.09 | 2023.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 2014.90 | 2007.09 | 2023.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 2014.90 | 2007.09 | 2023.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:00:00 | 2014.90 | 2007.09 | 2023.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 2026.10 | 2010.89 | 2023.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 2026.10 | 2010.89 | 2023.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 2025.50 | 2013.81 | 2023.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:30:00 | 2030.00 | 2013.81 | 2023.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 2017.70 | 2014.59 | 2023.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 2008.00 | 2017.74 | 2022.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:00:00 | 2007.80 | 2002.33 | 2008.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1907.60 | 1934.16 | 1945.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1907.41 | 1934.16 | 1945.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 1912.30 | 1908.52 | 1919.19 | SL hit (close>ema200) qty=0.50 sl=1908.52 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 1895.10 | 1888.86 | 1888.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 15:15:00 | 1900.00 | 1892.97 | 1890.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 1971.20 | 1976.06 | 1960.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 09:45:00 | 1971.80 | 1976.06 | 1960.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1974.90 | 1980.20 | 1971.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1974.90 | 1980.20 | 1971.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1962.20 | 1976.60 | 1970.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 1961.50 | 1976.60 | 1970.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1960.10 | 1973.30 | 1969.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:45:00 | 1963.60 | 1973.30 | 1969.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1955.20 | 1969.68 | 1968.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 1955.20 | 1969.68 | 1968.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 1960.80 | 1966.53 | 1967.17 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 09:15:00 | 1982.10 | 1969.65 | 1968.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 2005.30 | 1978.13 | 1972.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 11:15:00 | 1986.90 | 1991.02 | 1983.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 11:45:00 | 1987.80 | 1991.02 | 1983.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1968.70 | 1985.98 | 1982.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 1968.70 | 1985.98 | 1982.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1970.40 | 1982.86 | 1981.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 1973.60 | 1982.86 | 1981.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 1974.90 | 1979.23 | 1979.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 1964.10 | 1976.20 | 1978.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 11:15:00 | 1965.90 | 1963.55 | 1968.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 1965.90 | 1963.55 | 1968.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1965.90 | 1963.55 | 1968.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 1967.70 | 1963.55 | 1968.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1965.00 | 1963.84 | 1968.54 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 1979.10 | 1972.63 | 1971.77 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 1969.10 | 1981.16 | 1981.84 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1989.50 | 1980.69 | 1980.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 1991.00 | 1982.75 | 1981.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 1977.80 | 1982.12 | 1981.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 14:15:00 | 1977.80 | 1982.12 | 1981.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1977.80 | 1982.12 | 1981.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1977.80 | 1982.12 | 1981.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1977.90 | 1981.28 | 1980.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1987.70 | 1981.28 | 1980.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 1975.00 | 1980.02 | 1980.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 1975.00 | 1980.02 | 1980.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 1972.00 | 1978.42 | 1979.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1935.60 | 1933.46 | 1939.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 10:00:00 | 1935.60 | 1933.46 | 1939.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1940.30 | 1931.53 | 1934.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 1938.30 | 1931.53 | 1934.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1934.10 | 1932.04 | 1934.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 1940.50 | 1932.04 | 1934.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1922.50 | 1930.13 | 1933.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 1919.20 | 1927.37 | 1931.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 1920.00 | 1925.89 | 1930.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1912.30 | 1925.31 | 1929.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 1823.24 | 1840.67 | 1856.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 1824.00 | 1840.67 | 1856.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 1816.68 | 1840.67 | 1856.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 14:15:00 | 1823.70 | 1816.83 | 1825.43 | SL hit (close>ema200) qty=0.50 sl=1816.83 alert=retest2 |

### Cycle 89 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 1825.10 | 1815.19 | 1814.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 1835.10 | 1821.32 | 1817.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 1852.00 | 1853.79 | 1841.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:45:00 | 1850.50 | 1853.79 | 1841.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1819.20 | 1857.13 | 1851.05 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 1832.30 | 1846.72 | 1847.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 09:15:00 | 1818.90 | 1839.04 | 1843.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 15:15:00 | 1815.00 | 1814.57 | 1818.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:15:00 | 1809.80 | 1814.57 | 1818.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1818.80 | 1813.25 | 1815.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 1818.80 | 1813.25 | 1815.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1815.20 | 1813.64 | 1815.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 1813.00 | 1813.64 | 1815.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 14:30:00 | 1812.50 | 1812.02 | 1814.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 1831.30 | 1816.24 | 1815.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 1831.30 | 1816.24 | 1815.55 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 1812.50 | 1819.10 | 1819.29 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 1846.00 | 1824.48 | 1821.72 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 1811.40 | 1820.37 | 1820.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 1802.00 | 1811.99 | 1816.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 1798.30 | 1798.18 | 1805.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 11:00:00 | 1798.30 | 1798.18 | 1805.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1778.50 | 1791.83 | 1799.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:45:00 | 1774.70 | 1786.72 | 1795.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:00:00 | 1775.00 | 1784.38 | 1793.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 1768.40 | 1749.53 | 1749.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1768.40 | 1749.53 | 1749.35 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 1743.10 | 1748.15 | 1748.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 13:15:00 | 1737.70 | 1746.06 | 1747.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 1748.20 | 1744.61 | 1746.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1748.20 | 1744.61 | 1746.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1748.20 | 1744.61 | 1746.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 1743.80 | 1744.88 | 1746.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1765.50 | 1750.52 | 1748.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1765.50 | 1750.52 | 1748.50 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 1746.70 | 1749.28 | 1749.45 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1759.90 | 1750.08 | 1749.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 1782.00 | 1758.79 | 1753.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 10:15:00 | 1824.10 | 1828.56 | 1806.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:45:00 | 1824.80 | 1828.56 | 1806.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1816.00 | 1821.57 | 1808.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1833.60 | 1819.51 | 1809.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1804.50 | 1823.57 | 1817.37 | SL hit (close<static) qty=1.00 sl=1806.50 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 1812.80 | 1816.94 | 1816.97 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 1821.10 | 1817.77 | 1817.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 1828.80 | 1820.82 | 1818.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 15:15:00 | 1820.80 | 1820.95 | 1819.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 09:15:00 | 1822.40 | 1820.95 | 1819.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1826.70 | 1822.10 | 1820.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 1836.00 | 1822.10 | 1820.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 10:15:00 | 1832.50 | 1844.58 | 1845.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 10:15:00 | 1832.50 | 1844.58 | 1845.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 1829.00 | 1835.49 | 1839.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 1836.00 | 1833.78 | 1837.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 14:15:00 | 1836.00 | 1833.78 | 1837.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1836.00 | 1833.78 | 1837.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:45:00 | 1834.20 | 1833.78 | 1837.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 1833.50 | 1833.72 | 1837.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 1841.00 | 1833.72 | 1837.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1831.70 | 1833.32 | 1836.66 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1846.50 | 1838.00 | 1837.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 1856.90 | 1844.52 | 1840.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 1829.40 | 1844.79 | 1843.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 11:15:00 | 1829.40 | 1844.79 | 1843.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1829.40 | 1844.79 | 1843.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 1827.50 | 1844.79 | 1843.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 1830.10 | 1841.86 | 1841.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 09:15:00 | 1816.30 | 1835.73 | 1838.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 12:15:00 | 1784.00 | 1776.73 | 1788.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 13:00:00 | 1784.00 | 1776.73 | 1788.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1788.50 | 1779.99 | 1787.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 1788.00 | 1779.99 | 1787.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1790.00 | 1781.99 | 1787.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1786.40 | 1781.99 | 1787.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:00:00 | 1781.60 | 1769.38 | 1770.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 1779.00 | 1771.30 | 1770.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1779.00 | 1771.30 | 1770.90 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 1768.50 | 1770.53 | 1770.61 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 1772.00 | 1770.83 | 1770.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 15:15:00 | 1776.20 | 1772.50 | 1771.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 1766.00 | 1771.28 | 1771.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 1766.00 | 1771.28 | 1771.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1766.00 | 1771.28 | 1771.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 1766.00 | 1771.28 | 1771.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 1765.80 | 1770.19 | 1770.68 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1775.00 | 1771.02 | 1770.78 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 1764.10 | 1769.64 | 1770.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 1761.70 | 1767.63 | 1769.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 1773.00 | 1765.48 | 1767.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 1773.00 | 1765.48 | 1767.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1773.00 | 1765.48 | 1767.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1772.80 | 1765.48 | 1767.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1769.00 | 1767.23 | 1767.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 10:30:00 | 1767.20 | 1767.08 | 1767.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 1772.80 | 1768.22 | 1768.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 1772.80 | 1768.22 | 1768.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 12:15:00 | 1775.10 | 1769.60 | 1768.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 1767.10 | 1769.10 | 1768.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 13:15:00 | 1767.10 | 1769.10 | 1768.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1767.10 | 1769.10 | 1768.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 1767.10 | 1769.10 | 1768.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1764.50 | 1768.18 | 1768.29 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 1772.10 | 1768.50 | 1768.39 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1766.60 | 1768.12 | 1768.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 1758.00 | 1766.10 | 1767.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1759.10 | 1752.84 | 1756.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1759.10 | 1752.84 | 1756.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1759.10 | 1752.84 | 1756.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 1747.20 | 1751.29 | 1754.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 1746.40 | 1750.55 | 1753.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:30:00 | 1747.10 | 1749.13 | 1750.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 1747.20 | 1749.13 | 1750.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1739.00 | 1743.49 | 1747.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:30:00 | 1741.30 | 1743.49 | 1747.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1737.80 | 1734.33 | 1739.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1737.80 | 1734.33 | 1739.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1737.90 | 1735.04 | 1739.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:00:00 | 1735.30 | 1735.97 | 1739.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:15:00 | 1734.00 | 1735.99 | 1738.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 15:15:00 | 1734.90 | 1736.56 | 1738.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 1732.00 | 1736.22 | 1738.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 14:15:00 | 1768.80 | 1741.09 | 1739.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 14:15:00 | 1768.80 | 1741.09 | 1739.47 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 1731.30 | 1738.41 | 1738.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1717.90 | 1732.69 | 1735.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 13:15:00 | 1727.80 | 1722.77 | 1726.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 13:15:00 | 1727.80 | 1722.77 | 1726.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 1727.80 | 1722.77 | 1726.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 1727.80 | 1722.77 | 1726.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1725.50 | 1723.32 | 1726.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 1718.00 | 1723.32 | 1726.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 1721.70 | 1723.52 | 1726.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:30:00 | 1722.80 | 1724.27 | 1726.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:30:00 | 1724.50 | 1725.25 | 1726.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1724.50 | 1725.10 | 1726.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:30:00 | 1729.00 | 1725.10 | 1726.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1730.00 | 1726.08 | 1726.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 1730.80 | 1726.08 | 1726.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1727.30 | 1726.32 | 1726.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-12 10:15:00 | 1729.00 | 1726.86 | 1726.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1729.00 | 1726.86 | 1726.71 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 09:15:00 | 1713.80 | 1725.23 | 1726.28 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 1740.40 | 1725.54 | 1725.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 1746.00 | 1736.97 | 1731.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 10:15:00 | 1730.00 | 1735.57 | 1731.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 10:15:00 | 1730.00 | 1735.57 | 1731.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1730.00 | 1735.57 | 1731.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 1730.00 | 1735.57 | 1731.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1731.30 | 1734.72 | 1731.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 1731.30 | 1734.72 | 1731.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1730.00 | 1733.78 | 1731.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 1730.00 | 1733.78 | 1731.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1731.70 | 1733.36 | 1731.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:30:00 | 1730.60 | 1733.36 | 1731.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1730.00 | 1732.69 | 1731.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 1730.00 | 1732.69 | 1731.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 1737.00 | 1733.55 | 1731.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 1726.10 | 1733.55 | 1731.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1717.00 | 1730.24 | 1730.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 11:15:00 | 1711.10 | 1724.31 | 1727.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1711.00 | 1710.31 | 1717.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:00:00 | 1711.00 | 1710.31 | 1717.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 1716.30 | 1712.02 | 1716.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 1716.30 | 1712.02 | 1716.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 1706.50 | 1710.91 | 1715.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 1727.70 | 1710.91 | 1715.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1727.50 | 1714.23 | 1716.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1727.50 | 1714.23 | 1716.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1725.60 | 1716.50 | 1717.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 1727.50 | 1716.50 | 1717.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1727.50 | 1718.70 | 1718.44 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 1703.90 | 1718.47 | 1718.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 1699.00 | 1710.65 | 1714.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 1580.10 | 1579.31 | 1601.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:15:00 | 1584.50 | 1579.31 | 1601.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1571.40 | 1563.71 | 1572.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 1573.70 | 1563.71 | 1572.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1567.10 | 1564.38 | 1571.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:30:00 | 1561.90 | 1563.91 | 1570.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 1561.50 | 1537.95 | 1536.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 1561.50 | 1537.95 | 1536.47 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 1541.50 | 1544.55 | 1544.60 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1548.00 | 1545.24 | 1544.91 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 1533.10 | 1542.81 | 1543.84 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 1549.20 | 1536.43 | 1534.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 1551.40 | 1539.42 | 1536.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1551.60 | 1557.76 | 1548.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 12:00:00 | 1567.10 | 1560.80 | 1551.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1587.50 | 1601.15 | 1587.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1587.50 | 1601.15 | 1587.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1581.00 | 1597.12 | 1586.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-18 13:15:00 | 1581.00 | 1597.12 | 1586.69 | SL hit (close<ema400) qty=1.00 sl=1586.69 alert=retest1 |

### Cycle 128 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 1732.80 | 1743.65 | 1744.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 13:15:00 | 1730.00 | 1738.03 | 1741.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 1706.80 | 1706.37 | 1716.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:00:00 | 1706.80 | 1706.37 | 1716.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1710.90 | 1707.28 | 1715.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1700.00 | 1707.42 | 1714.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 09:15:00 | 1615.00 | 1638.38 | 1660.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 1639.50 | 1633.66 | 1649.33 | SL hit (close>ema200) qty=0.50 sl=1633.66 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 10:15:00 | 1596.40 | 1557.80 | 1557.69 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1568.80 | 1584.14 | 1585.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 1561.80 | 1576.75 | 1581.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 1565.90 | 1555.78 | 1565.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 1565.90 | 1555.78 | 1565.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1565.90 | 1555.78 | 1565.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1560.70 | 1555.78 | 1565.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1558.60 | 1556.34 | 1564.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1545.90 | 1556.34 | 1564.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1563.00 | 1557.67 | 1564.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1563.00 | 1557.67 | 1564.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1568.40 | 1559.82 | 1564.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 1568.40 | 1559.82 | 1564.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 1566.50 | 1561.15 | 1565.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:00:00 | 1559.70 | 1564.48 | 1565.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 1575.90 | 1567.49 | 1566.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 1575.90 | 1567.49 | 1566.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 1584.90 | 1570.97 | 1568.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1624.70 | 1629.43 | 1608.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:30:00 | 1623.60 | 1629.43 | 1608.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1604.00 | 1623.33 | 1608.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1604.00 | 1623.33 | 1608.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1615.40 | 1621.75 | 1609.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1611.00 | 1621.75 | 1609.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1602.90 | 1617.98 | 1608.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 1602.90 | 1617.98 | 1608.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 1591.50 | 1612.68 | 1607.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 1591.50 | 1612.68 | 1607.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 1573.90 | 1600.94 | 1603.03 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1621.10 | 1604.39 | 1603.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1660.50 | 1618.35 | 1610.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 1627.00 | 1643.52 | 1630.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 1627.00 | 1643.52 | 1630.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1627.00 | 1643.52 | 1630.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 1627.00 | 1643.52 | 1630.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 1619.50 | 1638.71 | 1629.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 1619.50 | 1638.71 | 1629.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1626.50 | 1634.24 | 1629.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:45:00 | 1626.00 | 1634.24 | 1629.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1630.60 | 1633.51 | 1629.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:30:00 | 1626.90 | 1633.51 | 1629.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1625.00 | 1631.81 | 1628.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 1625.00 | 1631.81 | 1628.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 1629.40 | 1631.33 | 1629.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 1631.00 | 1631.33 | 1629.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1618.40 | 1628.74 | 1628.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 1616.60 | 1628.74 | 1628.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 1609.70 | 1624.93 | 1626.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1594.00 | 1611.12 | 1618.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 1612.80 | 1611.45 | 1617.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:45:00 | 1611.80 | 1611.45 | 1617.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 1607.70 | 1610.70 | 1616.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 1616.70 | 1610.70 | 1616.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1629.40 | 1614.44 | 1617.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 1629.40 | 1614.44 | 1617.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1626.00 | 1616.75 | 1618.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:45:00 | 1627.60 | 1616.75 | 1618.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1610.00 | 1615.87 | 1617.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 1647.80 | 1615.87 | 1617.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1637.00 | 1620.10 | 1619.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1654.00 | 1626.88 | 1622.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 10:15:00 | 1671.50 | 1690.89 | 1673.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 1671.50 | 1690.89 | 1673.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1671.50 | 1690.89 | 1673.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 1671.50 | 1690.89 | 1673.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1672.50 | 1687.21 | 1673.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 1668.70 | 1687.21 | 1673.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1672.00 | 1684.17 | 1673.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:30:00 | 1678.80 | 1683.14 | 1673.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 1680.10 | 1680.89 | 1673.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1652.20 | 1675.02 | 1672.35 | SL hit (close<static) qty=1.00 sl=1668.80 alert=retest2 |

### Cycle 136 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 1655.90 | 1668.49 | 1669.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1625.20 | 1656.28 | 1662.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 1658.40 | 1655.25 | 1661.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:00:00 | 1658.40 | 1655.25 | 1661.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1645.00 | 1652.81 | 1658.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 1634.00 | 1652.81 | 1658.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 1654.90 | 1642.85 | 1642.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1654.90 | 1642.85 | 1642.84 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 1632.30 | 1641.03 | 1642.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 11:15:00 | 1630.00 | 1638.82 | 1641.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 1642.30 | 1638.12 | 1640.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 1642.30 | 1638.12 | 1640.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1642.30 | 1638.12 | 1640.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 1642.30 | 1638.12 | 1640.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 1655.00 | 1641.50 | 1641.44 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1624.00 | 1639.30 | 1640.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 1620.10 | 1633.08 | 1637.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 13:15:00 | 1615.60 | 1614.97 | 1622.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 13:45:00 | 1614.40 | 1614.97 | 1622.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 1596.00 | 1582.30 | 1589.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 1596.00 | 1582.30 | 1589.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 1593.30 | 1584.50 | 1589.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:45:00 | 1596.00 | 1584.50 | 1589.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1588.60 | 1581.61 | 1585.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 1588.60 | 1581.61 | 1585.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1585.00 | 1582.28 | 1585.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 1578.00 | 1582.28 | 1585.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1499.10 | 1539.62 | 1558.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1536.50 | 1506.82 | 1526.77 | SL hit (close>ema200) qty=0.50 sl=1506.82 alert=retest2 |

### Cycle 141 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 1372.40 | 1354.65 | 1353.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1378.10 | 1359.34 | 1355.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 1361.00 | 1361.25 | 1357.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 15:15:00 | 1361.00 | 1361.25 | 1357.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1361.00 | 1361.25 | 1357.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 1370.00 | 1361.25 | 1357.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 1341.90 | 1357.38 | 1356.00 | SL hit (close<static) qty=1.00 sl=1355.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1340.90 | 1354.08 | 1354.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1311.10 | 1338.76 | 1345.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1338.60 | 1310.24 | 1323.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1338.60 | 1310.24 | 1323.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1338.60 | 1310.24 | 1323.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1340.40 | 1310.24 | 1323.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1335.40 | 1315.27 | 1324.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 1342.60 | 1315.27 | 1324.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1351.80 | 1332.99 | 1331.43 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1313.60 | 1330.25 | 1331.23 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 1346.00 | 1333.40 | 1332.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 1363.30 | 1339.38 | 1335.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 1354.60 | 1357.13 | 1346.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 1354.60 | 1357.13 | 1346.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1354.60 | 1357.13 | 1346.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 15:00:00 | 1391.10 | 1372.01 | 1358.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 1395.50 | 1377.88 | 1363.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:45:00 | 1392.10 | 1380.87 | 1366.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:00:00 | 1390.10 | 1386.35 | 1372.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1424.00 | 1430.67 | 1418.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 1425.10 | 1430.67 | 1418.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 1424.00 | 1429.34 | 1419.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1461.60 | 1429.34 | 1419.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 1530.21 | 1481.66 | 1463.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 1678.90 | 1688.61 | 1688.81 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 1723.10 | 1692.68 | 1689.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 1735.00 | 1701.14 | 1693.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 09:15:00 | 1729.30 | 1732.69 | 1721.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 10:00:00 | 1729.30 | 1732.69 | 1721.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1751.00 | 1747.86 | 1736.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 1765.10 | 1749.73 | 1739.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-14 09:15:00 | 2471.55 | 2024-05-14 10:15:00 | 2485.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-06-04 09:15:00 | 2178.20 | 2024-06-04 12:15:00 | 2069.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:45:00 | 2191.15 | 2024-06-04 12:15:00 | 2081.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 2178.20 | 2024-06-05 09:15:00 | 2197.80 | STOP_HIT | 0.50 | -0.90% |
| SELL | retest2 | 2024-06-04 10:45:00 | 2191.15 | 2024-06-05 09:15:00 | 2197.80 | STOP_HIT | 0.50 | -0.30% |
| BUY | retest2 | 2024-06-18 12:00:00 | 2417.05 | 2024-06-24 15:15:00 | 2505.95 | STOP_HIT | 1.00 | 3.68% |
| BUY | retest2 | 2024-06-18 14:15:00 | 2415.90 | 2024-06-24 15:15:00 | 2505.95 | STOP_HIT | 1.00 | 3.73% |
| BUY | retest2 | 2024-07-04 11:45:00 | 2685.15 | 2024-07-10 09:15:00 | 2661.55 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-07-04 12:45:00 | 2687.75 | 2024-07-10 09:15:00 | 2661.55 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-04 13:45:00 | 2680.45 | 2024-07-10 10:15:00 | 2678.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-07-05 09:30:00 | 2718.90 | 2024-07-10 10:15:00 | 2678.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-07-09 09:15:00 | 2712.65 | 2024-07-10 10:15:00 | 2678.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-07-09 13:30:00 | 2729.35 | 2024-07-10 10:15:00 | 2678.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-07-19 10:15:00 | 2843.65 | 2024-07-19 15:15:00 | 2764.90 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-07-24 10:00:00 | 2873.00 | 2024-08-01 09:15:00 | 3160.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-06 13:30:00 | 3041.05 | 2024-08-07 12:15:00 | 3076.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-08-06 14:00:00 | 3039.75 | 2024-08-07 12:15:00 | 3076.30 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-08-06 14:30:00 | 3035.40 | 2024-08-07 12:15:00 | 3076.30 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-08-07 11:30:00 | 3048.20 | 2024-08-07 12:15:00 | 3076.30 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-08-19 10:15:00 | 2875.25 | 2024-08-21 10:15:00 | 2900.85 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-08-19 13:30:00 | 2876.55 | 2024-08-21 10:15:00 | 2900.85 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-08-20 09:30:00 | 2872.30 | 2024-08-21 10:15:00 | 2900.85 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-09-04 09:15:00 | 2956.90 | 2024-09-06 14:15:00 | 2928.85 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest1 | 2024-09-09 12:15:00 | 2935.00 | 2024-09-10 09:15:00 | 2967.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-09-10 12:00:00 | 2951.00 | 2024-09-10 14:15:00 | 2975.60 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-09-13 15:15:00 | 2925.15 | 2024-09-23 15:15:00 | 2869.00 | STOP_HIT | 1.00 | 1.92% |
| SELL | retest2 | 2024-09-16 14:30:00 | 2926.35 | 2024-09-23 15:15:00 | 2869.00 | STOP_HIT | 1.00 | 1.96% |
| SELL | retest2 | 2024-09-17 09:15:00 | 2927.20 | 2024-09-23 15:15:00 | 2869.00 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2024-10-09 13:00:00 | 2791.60 | 2024-10-10 09:15:00 | 2822.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-10-10 11:00:00 | 2792.00 | 2024-10-11 14:15:00 | 2830.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-10-10 11:30:00 | 2789.95 | 2024-10-11 14:15:00 | 2830.80 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-10-10 12:15:00 | 2788.00 | 2024-10-11 14:15:00 | 2830.80 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-10-24 09:30:00 | 2715.25 | 2024-10-30 10:15:00 | 2728.10 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-10-24 11:00:00 | 2727.10 | 2024-10-30 10:15:00 | 2728.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-10-24 12:00:00 | 2718.45 | 2024-10-30 10:15:00 | 2728.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-11-22 13:00:00 | 2683.00 | 2024-12-03 12:15:00 | 2627.85 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-11-22 15:15:00 | 2676.35 | 2024-12-03 12:15:00 | 2627.85 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-12-20 14:30:00 | 2621.25 | 2024-12-31 15:15:00 | 2490.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-23 11:45:00 | 2618.40 | 2025-01-01 09:15:00 | 2487.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-24 11:30:00 | 2615.80 | 2025-01-01 09:15:00 | 2485.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 14:30:00 | 2621.25 | 2025-01-01 10:15:00 | 2512.00 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2024-12-23 11:45:00 | 2618.40 | 2025-01-01 10:15:00 | 2512.00 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2024-12-24 11:30:00 | 2615.80 | 2025-01-01 10:15:00 | 2512.00 | STOP_HIT | 0.50 | 3.97% |
| BUY | retest2 | 2025-01-31 09:30:00 | 2318.40 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-01-31 11:15:00 | 2313.75 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2025-02-01 12:45:00 | 2312.20 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2025-02-01 13:45:00 | 2311.10 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2025-02-03 10:45:00 | 2378.70 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-02-06 09:45:00 | 2366.20 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-02-06 11:00:00 | 2378.95 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-02-06 12:15:00 | 2375.00 | 2025-02-07 13:15:00 | 2337.10 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-02-13 13:00:00 | 2228.15 | 2025-02-14 09:15:00 | 2005.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 2024.00 | 2025-02-14 10:15:00 | 1922.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 2024.00 | 2025-02-17 09:15:00 | 1821.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-12 09:15:00 | 1964.00 | 2025-03-12 09:15:00 | 1944.10 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-03-21 14:00:00 | 2049.70 | 2025-03-25 14:15:00 | 2035.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-03-25 11:45:00 | 2049.95 | 2025-03-25 14:15:00 | 2035.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-03-25 14:00:00 | 2047.45 | 2025-03-25 14:15:00 | 2035.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-03-28 12:15:00 | 1983.10 | 2025-04-02 11:15:00 | 2020.10 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-04-02 09:15:00 | 1980.05 | 2025-04-02 11:15:00 | 2020.10 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-04-03 11:30:00 | 1995.10 | 2025-04-04 09:15:00 | 1949.00 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-04-21 10:30:00 | 1976.60 | 2025-04-25 09:15:00 | 1953.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-04-29 14:15:00 | 1977.10 | 2025-04-29 14:15:00 | 1989.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-05-14 13:30:00 | 1982.00 | 2025-05-21 11:15:00 | 2060.50 | STOP_HIT | 1.00 | 3.96% |
| BUY | retest2 | 2025-05-14 15:00:00 | 1981.50 | 2025-05-21 11:15:00 | 2060.50 | STOP_HIT | 1.00 | 3.99% |
| BUY | retest2 | 2025-05-15 09:15:00 | 1998.00 | 2025-05-21 11:15:00 | 2060.50 | STOP_HIT | 1.00 | 3.13% |
| SELL | retest2 | 2025-05-28 11:15:00 | 2042.40 | 2025-05-29 10:15:00 | 2108.80 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-06-04 09:15:00 | 2008.00 | 2025-06-13 09:15:00 | 1907.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 11:00:00 | 2007.80 | 2025-06-13 09:15:00 | 1907.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-04 09:15:00 | 2008.00 | 2025-06-16 14:15:00 | 1912.30 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2025-06-05 11:00:00 | 2007.80 | 2025-06-16 14:15:00 | 1912.30 | STOP_HIT | 0.50 | 4.76% |
| BUY | retest2 | 2025-07-16 09:15:00 | 1987.70 | 2025-07-16 09:15:00 | 1975.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-24 12:45:00 | 1919.20 | 2025-08-01 13:15:00 | 1823.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 14:00:00 | 1920.00 | 2025-08-01 13:15:00 | 1824.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:15:00 | 1912.30 | 2025-08-01 13:15:00 | 1816.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 12:45:00 | 1919.20 | 2025-08-05 14:15:00 | 1823.70 | STOP_HIT | 0.50 | 4.98% |
| SELL | retest2 | 2025-07-24 14:00:00 | 1920.00 | 2025-08-05 14:15:00 | 1823.70 | STOP_HIT | 0.50 | 5.02% |
| SELL | retest2 | 2025-07-25 09:15:00 | 1912.30 | 2025-08-05 14:15:00 | 1823.70 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2025-08-22 11:15:00 | 1813.00 | 2025-08-25 10:15:00 | 1831.30 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-22 14:30:00 | 1812.50 | 2025-08-25 10:15:00 | 1831.30 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-02 11:45:00 | 1774.70 | 2025-09-10 10:15:00 | 1768.40 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-09-02 13:00:00 | 1775.00 | 2025-09-10 10:15:00 | 1768.40 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-09-11 11:30:00 | 1743.80 | 2025-09-12 09:15:00 | 1765.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-09-19 09:15:00 | 1833.60 | 2025-09-19 14:15:00 | 1804.50 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-09-22 09:45:00 | 1819.70 | 2025-09-23 09:15:00 | 1812.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-09-22 10:45:00 | 1823.00 | 2025-09-23 09:15:00 | 1812.80 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-22 12:30:00 | 1819.90 | 2025-09-23 09:15:00 | 1812.80 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-09-24 10:15:00 | 1836.00 | 2025-09-29 10:15:00 | 1832.50 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1786.40 | 2025-10-16 10:15:00 | 1779.00 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-10-16 10:00:00 | 1781.60 | 2025-10-16 10:15:00 | 1779.00 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-10-23 10:30:00 | 1767.20 | 2025-10-23 11:15:00 | 1772.80 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-10-28 14:00:00 | 1747.20 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-29 09:45:00 | 1746.40 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-30 11:30:00 | 1747.10 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-30 12:00:00 | 1747.20 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-11-03 13:00:00 | 1735.30 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-11-03 14:15:00 | 1734.00 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-11-03 15:15:00 | 1734.90 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-11-04 10:15:00 | 1732.00 | 2025-11-04 14:15:00 | 1768.80 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-11-10 15:15:00 | 1718.00 | 2025-11-12 10:15:00 | 1729.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-11 09:30:00 | 1721.70 | 2025-11-12 10:15:00 | 1729.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-11-11 11:30:00 | 1722.80 | 2025-11-12 10:15:00 | 1729.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-11-11 13:30:00 | 1724.50 | 2025-11-12 10:15:00 | 1729.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-12-01 12:30:00 | 1561.90 | 2025-12-05 09:15:00 | 1561.50 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest1 | 2025-12-16 12:00:00 | 1567.10 | 2025-12-18 13:15:00 | 1581.00 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1700.00 | 2026-01-07 09:15:00 | 1615.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1700.00 | 2026-01-07 14:15:00 | 1639.50 | STOP_HIT | 0.50 | 3.56% |
| SELL | retest2 | 2026-01-29 10:00:00 | 1559.70 | 2026-01-29 13:15:00 | 1575.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-11 13:30:00 | 1678.80 | 2026-02-12 09:15:00 | 1652.20 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-02-11 15:15:00 | 1680.10 | 2026-02-12 09:15:00 | 1652.20 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-13 15:15:00 | 1634.00 | 2026-02-17 13:15:00 | 1654.90 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-02-27 15:15:00 | 1578.00 | 2026-03-04 09:15:00 | 1499.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:15:00 | 1578.00 | 2026-03-05 09:15:00 | 1536.50 | STOP_HIT | 0.50 | 2.63% |
| BUY | retest2 | 2026-03-27 09:15:00 | 1370.00 | 2026-03-27 09:15:00 | 1341.90 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-04-06 15:00:00 | 1391.10 | 2026-04-15 09:15:00 | 1530.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:15:00 | 1395.50 | 2026-04-15 09:15:00 | 1535.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:45:00 | 1392.10 | 2026-04-15 09:15:00 | 1531.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 14:00:00 | 1390.10 | 2026-04-15 09:15:00 | 1529.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-10 09:15:00 | 1461.60 | 2026-04-21 11:15:00 | 1607.76 | TARGET_HIT | 1.00 | 10.00% |
