# Balkrishna Industries Ltd. (BALKRISIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2265.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 4 |
| ALERT3 | 52 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 30 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 26
- **Target hits / Stop hits / Partials:** 2 / 29 / 6
- **Avg / median % per leg:** 0.04% / -1.67%
- **Sum % (uncompounded):** 1.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 1 | 10.0% | 1 | 9 | 0 | -1.50% | -15.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 1 | 9 | 0 | -1.50% | -15.0% |
| SELL (all) | 27 | 10 | 37.0% | 1 | 20 | 6 | 0.61% | 16.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.82% | -5.8% |
| SELL @ 3rd Alert (retest2) | 26 | 10 | 38.5% | 1 | 19 | 6 | 0.86% | 22.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.82% | -5.8% |
| retest2 (combined) | 36 | 11 | 30.6% | 2 | 28 | 6 | 0.20% | 7.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 14:15:00 | 2859.35 | 3008.77 | 3009.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 09:15:00 | 2842.80 | 3005.57 | 3007.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 15:15:00 | 2955.80 | 2955.21 | 2978.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 09:15:00 | 2953.80 | 2955.21 | 2978.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 2959.80 | 2955.25 | 2978.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 2941.00 | 2956.24 | 2977.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 10:15:00 | 2980.75 | 2956.58 | 2977.14 | SL hit (close>static) qty=1.00 sl=2979.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 14:15:00 | 3070.15 | 2993.78 | 2993.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 13:15:00 | 3078.35 | 2997.54 | 2995.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 2985.55 | 3025.63 | 3011.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 2985.55 | 3025.63 | 3011.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 2985.55 | 3025.63 | 3011.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:30:00 | 3050.00 | 3024.23 | 3011.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 15:00:00 | 3049.25 | 3026.20 | 3012.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 15:15:00 | 3055.00 | 3026.28 | 3013.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 11:30:00 | 3053.70 | 3027.60 | 3014.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 3019.10 | 3028.52 | 3014.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-03 10:15:00 | 2952.50 | 3027.76 | 3014.52 | SL hit (close<static) qty=1.00 sl=2974.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 09:15:00 | 2873.85 | 3001.51 | 3002.14 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 15:15:00 | 3040.00 | 3002.68 | 3002.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 3051.90 | 3003.17 | 3002.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 2983.70 | 3006.20 | 3004.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 10:15:00 | 2983.70 | 3006.20 | 3004.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 2983.70 | 3006.20 | 3004.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 2983.70 | 3006.20 | 3004.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 3004.65 | 3006.18 | 3004.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 12:30:00 | 3019.15 | 3006.42 | 3004.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:45:00 | 3017.00 | 3006.55 | 3004.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 15:15:00 | 3039.00 | 3006.55 | 3004.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 14:15:00 | 3018.20 | 3006.34 | 3004.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 2991.45 | 3006.50 | 3004.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 2991.45 | 3006.50 | 3004.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 3020.65 | 3006.64 | 3004.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 15:15:00 | 3030.00 | 3006.74 | 3004.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 2952.00 | 3006.43 | 3004.83 | SL hit (close<static) qty=1.00 sl=2982.40 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 2972.95 | 3003.25 | 3003.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 2939.65 | 3000.66 | 3001.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 2819.40 | 2818.42 | 2877.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 2878.30 | 2820.35 | 2876.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 2878.30 | 2820.35 | 2876.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:00:00 | 2878.30 | 2820.35 | 2876.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 2885.50 | 2821.00 | 2876.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 2885.00 | 2821.00 | 2876.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 2884.10 | 2821.63 | 2876.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:30:00 | 2897.75 | 2821.63 | 2876.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 2886.70 | 2822.28 | 2876.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 2888.80 | 2822.28 | 2876.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 2875.00 | 2824.00 | 2876.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 2863.15 | 2824.00 | 2876.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 2857.80 | 2824.34 | 2876.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 10:30:00 | 2856.45 | 2824.54 | 2876.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 12:30:00 | 2850.00 | 2825.01 | 2876.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 09:15:00 | 2713.63 | 2825.11 | 2873.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 09:15:00 | 2707.50 | 2825.11 | 2873.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-09 10:15:00 | 2878.15 | 2825.64 | 2873.44 | SL hit (close>ema200) qty=0.50 sl=2825.64 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 2748.70 | 2609.96 | 2609.27 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 13:15:00 | 2476.30 | 2617.13 | 2617.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 2448.70 | 2595.16 | 2605.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 2551.90 | 2550.72 | 2578.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:45:00 | 2547.90 | 2550.72 | 2578.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2569.60 | 2480.62 | 2523.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 2569.60 | 2480.62 | 2523.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2575.40 | 2481.57 | 2523.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 2566.00 | 2481.57 | 2523.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 2670.20 | 2551.44 | 2551.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 2724.20 | 2559.99 | 2555.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 2625.20 | 2648.61 | 2610.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 10:00:00 | 2625.20 | 2648.61 | 2610.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 2597.20 | 2648.10 | 2610.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 2597.20 | 2648.10 | 2610.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 2612.00 | 2647.74 | 2610.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 2599.50 | 2647.74 | 2610.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 2605.00 | 2647.32 | 2610.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 2605.00 | 2647.32 | 2610.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 2590.00 | 2646.75 | 2610.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:45:00 | 2594.00 | 2646.75 | 2610.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 2413.60 | 2582.03 | 2582.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 2402.20 | 2578.59 | 2580.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 2409.30 | 2407.80 | 2466.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 2409.30 | 2407.80 | 2466.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 2460.90 | 2408.95 | 2462.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 2460.90 | 2408.95 | 2462.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 2455.20 | 2409.41 | 2462.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 2455.20 | 2409.41 | 2462.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 2452.70 | 2410.26 | 2462.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:45:00 | 2460.20 | 2410.26 | 2462.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 2468.30 | 2411.27 | 2462.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 2468.30 | 2411.27 | 2462.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 2470.00 | 2411.85 | 2462.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 2480.00 | 2411.85 | 2462.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 2484.00 | 2412.57 | 2462.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 2441.00 | 2440.09 | 2469.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:00:00 | 2455.40 | 2440.24 | 2469.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 2332.63 | 2430.70 | 2461.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 10:15:00 | 2318.95 | 2429.54 | 2461.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-14 09:15:00 | 2209.86 | 2356.79 | 2409.92 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 2413.30 | 2350.56 | 2350.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 2429.60 | 2357.25 | 2353.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 2376.50 | 2384.41 | 2370.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 2376.50 | 2384.41 | 2370.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 2376.50 | 2384.41 | 2370.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 2381.80 | 2384.41 | 2370.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 2356.30 | 2384.13 | 2370.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:45:00 | 2355.50 | 2384.13 | 2370.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 2367.90 | 2383.97 | 2370.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:45:00 | 2352.20 | 2383.97 | 2370.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 2362.80 | 2383.54 | 2370.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 2362.80 | 2383.54 | 2370.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 2365.00 | 2383.36 | 2370.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 2363.70 | 2383.36 | 2370.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 2339.00 | 2382.15 | 2369.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 2474.20 | 2366.65 | 2362.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 2721.62 | 2367.23 | 2363.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 2239.00 | 2423.36 | 2424.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 2233.50 | 2411.85 | 2418.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2221.40 | 2209.58 | 2287.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:45:00 | 2187.80 | 2209.99 | 2284.86 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 2277.20 | 2211.94 | 2282.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:30:00 | 2277.90 | 2211.94 | 2282.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 2281.80 | 2212.64 | 2282.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 2281.80 | 2212.64 | 2282.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 2271.10 | 2213.22 | 2282.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:00:00 | 2258.10 | 2213.66 | 2282.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 2262.00 | 2214.31 | 2282.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 2315.20 | 2219.34 | 2280.67 | SL hit (close>ema400) qty=1.00 sl=2280.67 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-09 09:15:00 | 2941.00 | 2024-09-09 10:15:00 | 2980.75 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-09-27 09:30:00 | 3050.00 | 2024-10-03 10:15:00 | 2952.50 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2024-09-27 15:00:00 | 3049.25 | 2024-10-03 10:15:00 | 2952.50 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-09-30 15:15:00 | 3055.00 | 2024-10-03 10:15:00 | 2952.50 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2024-10-01 11:30:00 | 3053.70 | 2024-10-03 10:15:00 | 2952.50 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2024-10-11 12:30:00 | 3019.15 | 2024-10-17 09:15:00 | 2952.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-10-11 14:45:00 | 3017.00 | 2024-10-17 09:15:00 | 2952.00 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-10-11 15:15:00 | 3039.00 | 2024-10-17 09:15:00 | 2952.00 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-10-15 14:15:00 | 3018.20 | 2024-10-17 09:15:00 | 2952.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-10-16 15:15:00 | 3030.00 | 2024-10-17 09:15:00 | 2952.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-12-05 10:30:00 | 2856.45 | 2024-12-09 09:15:00 | 2713.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-05 12:30:00 | 2850.00 | 2024-12-09 09:15:00 | 2707.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-05 10:30:00 | 2856.45 | 2024-12-09 10:15:00 | 2878.15 | STOP_HIT | 0.50 | -0.76% |
| SELL | retest2 | 2024-12-05 12:30:00 | 2850.00 | 2024-12-09 10:15:00 | 2878.15 | STOP_HIT | 0.50 | -0.99% |
| SELL | retest2 | 2024-12-09 13:15:00 | 2856.25 | 2024-12-27 13:15:00 | 2886.45 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-12-09 14:00:00 | 2851.20 | 2024-12-27 13:15:00 | 2886.45 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-12-16 13:30:00 | 2860.25 | 2024-12-27 13:15:00 | 2886.45 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-12-16 14:30:00 | 2861.70 | 2024-12-27 13:15:00 | 2886.45 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-17 10:30:00 | 2861.85 | 2024-12-31 13:15:00 | 2911.90 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-12-27 11:15:00 | 2857.75 | 2024-12-31 13:15:00 | 2911.90 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-12-30 12:45:00 | 2853.45 | 2024-12-31 13:15:00 | 2911.90 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-12-31 09:15:00 | 2857.65 | 2024-12-31 13:15:00 | 2911.90 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-12-31 10:45:00 | 2858.30 | 2024-12-31 13:15:00 | 2911.90 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-12-31 11:15:00 | 2855.60 | 2024-12-31 13:15:00 | 2911.90 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-01-03 09:45:00 | 2856.50 | 2025-01-06 14:15:00 | 2713.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 09:45:00 | 2856.50 | 2025-01-21 09:15:00 | 2780.05 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2441.00 | 2025-09-29 09:15:00 | 2332.63 | PARTIAL | 0.50 | 4.44% |
| SELL | retest2 | 2025-09-24 10:00:00 | 2455.40 | 2025-09-29 10:15:00 | 2318.95 | PARTIAL | 0.50 | 5.56% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2441.00 | 2025-10-14 09:15:00 | 2209.86 | TARGET_HIT | 0.50 | 9.47% |
| SELL | retest2 | 2025-09-24 10:00:00 | 2455.40 | 2025-10-23 09:15:00 | 2351.70 | STOP_HIT | 0.50 | 4.22% |
| BUY | retest2 | 2026-02-03 09:15:00 | 2474.20 | 2026-02-03 09:15:00 | 2721.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-04-09 09:45:00 | 2187.80 | 2026-04-15 14:15:00 | 2315.20 | STOP_HIT | 1.00 | -5.82% |
| SELL | retest2 | 2026-04-10 14:00:00 | 2258.10 | 2026-04-15 14:15:00 | 2315.20 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-04-10 15:15:00 | 2262.00 | 2026-04-15 14:15:00 | 2315.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-04-21 10:15:00 | 2264.20 | 2026-04-21 12:15:00 | 2302.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-04-23 12:30:00 | 2259.90 | 2026-04-30 10:15:00 | 2146.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 12:30:00 | 2259.90 | 2026-05-06 15:15:00 | 2236.00 | STOP_HIT | 0.50 | 1.06% |
