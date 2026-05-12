# MphasiS Ltd. (MPHASIS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2214.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 5 |
| ALERT3 | 56 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 61 |
| PARTIAL | 10 |
| TARGET_HIT | 20 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 41
- **Target hits / Stop hits / Partials:** 20 / 41 / 10
- **Avg / median % per leg:** 1.99% / -1.15%
- **Sum % (uncompounded):** 141.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 13 | 32.5% | 13 | 27 | 0 | 1.58% | 63.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 40 | 13 | 32.5% | 13 | 27 | 0 | 1.58% | 63.1% |
| SELL (all) | 31 | 17 | 54.8% | 7 | 14 | 10 | 2.52% | 78.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 17 | 54.8% | 7 | 14 | 10 | 2.52% | 78.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 71 | 30 | 42.3% | 20 | 41 | 10 | 1.99% | 141.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 15:15:00 | 2147.00 | 2307.72 | 2308.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 09:15:00 | 2139.15 | 2306.04 | 2307.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 2260.60 | 2244.55 | 2270.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 2260.60 | 2244.55 | 2270.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 2260.60 | 2244.55 | 2270.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:45:00 | 2238.95 | 2244.55 | 2270.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 14:15:00 | 2267.70 | 2245.01 | 2269.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 14:45:00 | 2270.00 | 2245.01 | 2269.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 15:15:00 | 2271.50 | 2245.27 | 2270.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 09:15:00 | 2264.55 | 2245.27 | 2270.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 2261.00 | 2245.43 | 2269.96 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 10:15:00 | 2366.00 | 2288.72 | 2288.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 14:15:00 | 2376.35 | 2295.36 | 2291.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 14:15:00 | 2556.00 | 2564.94 | 2476.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-09 15:15:00 | 2552.10 | 2564.94 | 2476.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 2513.55 | 2581.61 | 2503.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:30:00 | 2496.00 | 2581.61 | 2503.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 13:15:00 | 2502.55 | 2579.54 | 2503.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 14:00:00 | 2502.55 | 2579.54 | 2503.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 14:15:00 | 2519.00 | 2578.94 | 2503.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 15:15:00 | 2520.00 | 2578.94 | 2503.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 14:30:00 | 2524.30 | 2574.33 | 2508.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 12:45:00 | 2522.60 | 2572.32 | 2509.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 15:00:00 | 2521.20 | 2568.58 | 2510.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 2545.15 | 2573.65 | 2521.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 09:45:00 | 2640.25 | 2569.69 | 2525.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 12:00:00 | 2645.65 | 2570.87 | 2526.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 09:15:00 | 2638.60 | 2572.38 | 2528.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 11:45:00 | 2639.50 | 2574.55 | 2531.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 2561.25 | 2577.97 | 2535.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:30:00 | 2540.50 | 2577.97 | 2535.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 2529.85 | 2577.49 | 2535.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:00:00 | 2529.85 | 2577.49 | 2535.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 2526.95 | 2576.98 | 2535.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 12:45:00 | 2540.20 | 2576.60 | 2535.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 14:00:00 | 2545.90 | 2576.29 | 2535.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-16 14:15:00 | 2772.00 | 2590.05 | 2545.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 15:15:00 | 2469.15 | 2558.05 | 2558.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 2446.80 | 2556.94 | 2557.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 10:15:00 | 2501.10 | 2492.59 | 2519.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-03 10:30:00 | 2496.00 | 2492.59 | 2519.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 2513.00 | 2492.99 | 2518.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:00:00 | 2513.00 | 2492.99 | 2518.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 2513.00 | 2493.19 | 2518.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:30:00 | 2523.75 | 2493.19 | 2518.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 15:15:00 | 2523.85 | 2493.50 | 2518.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 09:15:00 | 2499.30 | 2493.50 | 2518.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 11:00:00 | 2508.00 | 2493.76 | 2518.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 15:15:00 | 2503.70 | 2494.14 | 2518.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 09:45:00 | 2499.55 | 2491.82 | 2515.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 09:15:00 | 2374.34 | 2479.25 | 2506.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 09:15:00 | 2382.60 | 2479.25 | 2506.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 09:15:00 | 2378.51 | 2479.25 | 2506.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 09:15:00 | 2374.57 | 2479.25 | 2506.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-04-22 10:15:00 | 2249.37 | 2448.05 | 2486.98 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 2521.85 | 2400.12 | 2399.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 11:15:00 | 2529.60 | 2401.41 | 2400.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 2667.90 | 2732.24 | 2616.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 10:00:00 | 2667.90 | 2732.24 | 2616.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 2614.80 | 2731.07 | 2616.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 2614.80 | 2731.07 | 2616.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 2624.00 | 2730.00 | 2616.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 12:15:00 | 2632.20 | 2730.00 | 2616.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 13:30:00 | 2631.05 | 2727.92 | 2616.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-16 12:15:00 | 2895.42 | 2722.65 | 2639.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 13:15:00 | 2757.80 | 2924.61 | 2924.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 2753.70 | 2922.91 | 2924.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 2918.40 | 2902.68 | 2913.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 2918.40 | 2902.68 | 2913.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 2918.40 | 2902.68 | 2913.10 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 15:15:00 | 2963.00 | 2922.29 | 2922.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 2975.10 | 2922.82 | 2922.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 11:15:00 | 3016.80 | 3055.01 | 3003.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 12:00:00 | 3016.80 | 3055.01 | 3003.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 2994.40 | 3054.41 | 3003.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 2994.40 | 3054.41 | 3003.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 2984.40 | 3053.71 | 3002.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:00:00 | 2984.40 | 3053.71 | 3002.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 2970.00 | 3049.92 | 3001.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 2993.90 | 3049.92 | 3001.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 14:15:00 | 2981.30 | 3047.59 | 3001.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 09:15:00 | 2993.30 | 3046.19 | 3001.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 11:15:00 | 2987.25 | 3045.05 | 3001.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 2960.05 | 3044.21 | 3000.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:00:00 | 2960.05 | 3044.21 | 3000.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-24 14:15:00 | 2930.00 | 3041.08 | 3000.03 | SL hit (close<static) qty=1.00 sl=2940.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 11:15:00 | 2909.95 | 2969.38 | 2969.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 2890.85 | 2966.90 | 2968.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 2903.00 | 2889.58 | 2922.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 2903.00 | 2889.58 | 2922.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2916.85 | 2889.85 | 2922.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 2920.50 | 2889.85 | 2922.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 2932.50 | 2890.27 | 2922.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 2925.00 | 2890.27 | 2922.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 2940.95 | 2890.78 | 2922.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 2939.75 | 2890.78 | 2922.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 2922.65 | 2891.41 | 2922.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 2922.65 | 2891.41 | 2922.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 2911.05 | 2891.60 | 2922.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 2803.25 | 2891.60 | 2922.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 3025.40 | 2891.54 | 2922.42 | SL hit (close>static) qty=1.00 sl=2930.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 2557.50 | 2493.60 | 2493.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 2565.10 | 2494.31 | 2493.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 2465.10 | 2498.79 | 2496.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 2465.10 | 2498.79 | 2496.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 2465.10 | 2498.79 | 2496.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 10:45:00 | 2496.10 | 2498.71 | 2496.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:30:00 | 2488.80 | 2498.20 | 2495.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 14:15:00 | 2487.10 | 2498.20 | 2495.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 14:45:00 | 2487.80 | 2498.15 | 2495.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 2502.50 | 2498.09 | 2495.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 2534.00 | 2498.28 | 2496.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 2486.60 | 2500.25 | 2497.21 | SL hit (close<static) qty=1.00 sl=2490.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 2740.00 | 2780.22 | 2780.30 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 2806.20 | 2780.48 | 2780.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 2826.80 | 2782.39 | 2781.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 2793.60 | 2804.16 | 2793.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 2793.60 | 2804.16 | 2793.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 2793.60 | 2804.16 | 2793.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 2793.60 | 2804.16 | 2793.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 2770.80 | 2803.83 | 2793.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 2770.80 | 2803.83 | 2793.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 2762.50 | 2803.42 | 2793.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 2762.50 | 2803.42 | 2793.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 2763.10 | 2798.99 | 2791.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 2763.10 | 2798.99 | 2791.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 2775.00 | 2790.98 | 2788.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 2775.00 | 2790.98 | 2788.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 2777.40 | 2790.85 | 2787.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 2778.90 | 2790.85 | 2787.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 2775.70 | 2790.68 | 2787.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 2775.70 | 2790.68 | 2787.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 2775.00 | 2790.52 | 2787.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 2767.30 | 2790.30 | 2787.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 2775.20 | 2790.15 | 2787.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 2787.90 | 2789.59 | 2787.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:30:00 | 2785.10 | 2794.51 | 2790.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 2749.10 | 2794.06 | 2790.01 | SL hit (close<static) qty=1.00 sl=2762.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 2662.50 | 2785.91 | 2786.07 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 2819.30 | 2783.43 | 2783.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 15:15:00 | 2846.00 | 2784.90 | 2784.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 2852.50 | 2856.65 | 2830.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 14:00:00 | 2852.50 | 2856.65 | 2830.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 2824.50 | 2856.13 | 2831.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:00:00 | 2824.50 | 2856.13 | 2831.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 2818.00 | 2855.75 | 2830.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:15:00 | 2801.50 | 2855.75 | 2830.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 2820.00 | 2844.81 | 2827.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:30:00 | 2826.60 | 2844.42 | 2827.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 2805.30 | 2843.79 | 2827.49 | SL hit (close<static) qty=1.00 sl=2812.20 alert=retest2 |

### Cycle 13 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 2753.40 | 2822.74 | 2822.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 2735.00 | 2819.92 | 2821.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2846.00 | 2812.04 | 2817.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 2846.00 | 2812.04 | 2817.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2846.00 | 2812.04 | 2817.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:30:00 | 2830.70 | 2812.18 | 2817.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 11:45:00 | 2830.50 | 2812.35 | 2817.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 09:15:00 | 2689.16 | 2811.27 | 2816.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 09:15:00 | 2688.97 | 2811.27 | 2816.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-11 12:15:00 | 2547.63 | 2750.78 | 2782.85 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-18 15:15:00 | 2520.00 | 2024-02-16 14:15:00 | 2772.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-23 14:30:00 | 2524.30 | 2024-02-16 14:15:00 | 2776.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-24 12:45:00 | 2522.60 | 2024-02-16 14:15:00 | 2774.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-25 15:00:00 | 2521.20 | 2024-02-16 14:15:00 | 2773.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-08 09:45:00 | 2640.25 | 2024-02-16 14:15:00 | 2794.22 | TARGET_HIT | 1.00 | 5.83% |
| BUY | retest2 | 2024-02-08 12:00:00 | 2645.65 | 2024-02-16 14:15:00 | 2800.49 | TARGET_HIT | 1.00 | 5.85% |
| BUY | retest2 | 2024-02-09 09:15:00 | 2638.60 | 2024-03-06 09:15:00 | 2498.35 | STOP_HIT | 1.00 | -5.32% |
| BUY | retest2 | 2024-02-12 11:45:00 | 2639.50 | 2024-03-06 09:15:00 | 2498.35 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest2 | 2024-02-14 12:45:00 | 2540.20 | 2024-03-06 09:15:00 | 2498.35 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-02-14 14:00:00 | 2545.90 | 2024-03-06 09:15:00 | 2498.35 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-03-05 12:45:00 | 2540.10 | 2024-03-06 09:15:00 | 2498.35 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-04-05 09:15:00 | 2499.30 | 2024-04-16 09:15:00 | 2374.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 11:00:00 | 2508.00 | 2024-04-16 09:15:00 | 2382.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 15:15:00 | 2503.70 | 2024-04-16 09:15:00 | 2378.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-09 09:45:00 | 2499.55 | 2024-04-16 09:15:00 | 2374.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 09:15:00 | 2499.30 | 2024-04-22 10:15:00 | 2249.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-05 11:00:00 | 2508.00 | 2024-04-22 10:15:00 | 2257.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-05 15:15:00 | 2503.70 | 2024-04-22 10:15:00 | 2253.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-09 09:45:00 | 2499.55 | 2024-04-22 10:15:00 | 2249.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-22 15:00:00 | 2370.00 | 2024-05-23 11:15:00 | 2416.95 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-05-29 15:00:00 | 2367.10 | 2024-06-04 09:15:00 | 2248.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 09:15:00 | 2349.20 | 2024-06-04 11:15:00 | 2231.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 15:00:00 | 2367.10 | 2024-06-05 10:15:00 | 2406.85 | STOP_HIT | 0.50 | -1.68% |
| SELL | retest2 | 2024-05-30 09:15:00 | 2349.20 | 2024-06-05 10:15:00 | 2406.85 | STOP_HIT | 0.50 | -2.45% |
| SELL | retest2 | 2024-06-05 14:15:00 | 2362.65 | 2024-06-06 09:15:00 | 2452.05 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2024-06-10 09:45:00 | 2374.60 | 2024-06-24 10:15:00 | 2420.30 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-06-11 09:30:00 | 2390.80 | 2024-06-26 09:15:00 | 2418.25 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-06-11 11:00:00 | 2398.00 | 2024-07-01 09:15:00 | 2481.55 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2024-06-11 12:45:00 | 2396.30 | 2024-07-01 09:15:00 | 2481.55 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2024-06-24 09:15:00 | 2390.30 | 2024-07-01 09:15:00 | 2481.55 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2024-06-25 09:15:00 | 2388.20 | 2024-07-01 09:15:00 | 2481.55 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2024-08-05 12:15:00 | 2632.20 | 2024-08-16 12:15:00 | 2895.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-05 13:30:00 | 2631.05 | 2024-08-16 12:15:00 | 2894.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-23 11:15:00 | 2993.90 | 2024-12-24 14:15:00 | 2930.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-12-23 14:15:00 | 2981.30 | 2024-12-24 14:15:00 | 2930.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-12-24 09:15:00 | 2993.30 | 2024-12-24 14:15:00 | 2930.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-12-24 11:15:00 | 2987.25 | 2024-12-24 14:15:00 | 2930.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-01-24 09:15:00 | 2803.25 | 2025-01-24 11:15:00 | 3025.40 | STOP_HIT | 1.00 | -7.92% |
| SELL | retest2 | 2025-01-28 13:45:00 | 2905.20 | 2025-01-29 09:15:00 | 2958.10 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-01-28 14:30:00 | 2899.55 | 2025-01-29 09:15:00 | 2958.10 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-01-30 11:30:00 | 2901.45 | 2025-02-03 09:15:00 | 2756.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-30 11:30:00 | 2901.45 | 2025-02-13 13:15:00 | 2611.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-06 09:30:00 | 2434.30 | 2025-05-09 09:15:00 | 2312.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:30:00 | 2434.30 | 2025-05-12 09:15:00 | 2494.00 | STOP_HIT | 0.50 | -2.45% |
| BUY | retest2 | 2025-06-02 10:45:00 | 2496.10 | 2025-06-05 10:15:00 | 2486.60 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-06-02 13:30:00 | 2488.80 | 2025-06-25 10:15:00 | 2737.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-02 14:15:00 | 2487.10 | 2025-06-25 10:15:00 | 2735.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-02 14:45:00 | 2487.80 | 2025-06-25 10:15:00 | 2736.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 09:15:00 | 2534.00 | 2025-06-25 12:15:00 | 2745.71 | TARGET_HIT | 1.00 | 8.35% |
| BUY | retest2 | 2025-06-05 11:45:00 | 2518.90 | 2025-06-26 09:15:00 | 2770.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-11 15:15:00 | 2787.90 | 2025-11-14 12:15:00 | 2749.10 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-11-14 11:30:00 | 2785.10 | 2025-11-14 12:15:00 | 2749.10 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-01-01 14:30:00 | 2826.60 | 2026-01-02 09:15:00 | 2805.30 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-01-06 11:45:00 | 2824.60 | 2026-01-08 13:15:00 | 2806.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2026-01-07 09:15:00 | 2829.50 | 2026-01-08 13:15:00 | 2806.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-07 09:45:00 | 2825.60 | 2026-01-08 13:15:00 | 2806.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-01-09 09:30:00 | 2827.00 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-01-09 10:15:00 | 2837.80 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-01-12 09:45:00 | 2827.30 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-12 10:15:00 | 2826.00 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-01-13 10:45:00 | 2862.00 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-01-13 14:45:00 | 2865.00 | 2026-01-14 10:15:00 | 2782.70 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2026-01-16 09:15:00 | 2899.90 | 2026-01-21 09:15:00 | 2764.10 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2026-01-19 10:00:00 | 2867.70 | 2026-01-21 09:15:00 | 2764.10 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2026-01-22 09:15:00 | 2854.50 | 2026-01-22 14:15:00 | 2812.70 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-28 11:00:00 | 2850.90 | 2026-01-29 09:15:00 | 2780.00 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-01-28 11:45:00 | 2856.60 | 2026-01-29 09:15:00 | 2780.00 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-02-03 10:30:00 | 2830.70 | 2026-02-04 09:15:00 | 2689.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-03 11:45:00 | 2830.50 | 2026-02-04 09:15:00 | 2688.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-03 10:30:00 | 2830.70 | 2026-02-11 12:15:00 | 2547.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 11:45:00 | 2830.50 | 2026-02-11 12:15:00 | 2547.45 | TARGET_HIT | 0.50 | 10.00% |
