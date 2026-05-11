# Ajanta Pharmaceuticals Ltd. (AJANTPHARM)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 3033.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 6 |
| TARGET_HIT | 9 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 21
- **Target hits / Stop hits / Partials:** 9 / 22 / 6
- **Avg / median % per leg:** 2.07% / -1.08%
- **Sum % (uncompounded):** 76.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 6 | 40.0% | 6 | 9 | 0 | 2.38% | 35.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 6 | 40.0% | 6 | 9 | 0 | 2.38% | 35.7% |
| SELL (all) | 22 | 10 | 45.5% | 3 | 13 | 6 | 1.86% | 41.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 10 | 45.5% | 3 | 13 | 6 | 1.86% | 41.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 16 | 43.2% | 9 | 22 | 6 | 2.07% | 76.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 15:15:00 | 2272.00 | 2284.14 | 2284.15 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 09:15:00 | 2312.90 | 2284.43 | 2284.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 09:15:00 | 2327.00 | 2286.75 | 2285.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 11:15:00 | 3105.80 | 3118.16 | 2898.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 12:00:00 | 3105.80 | 3118.16 | 2898.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 3089.95 | 3234.25 | 3096.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 3089.95 | 3234.25 | 3096.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 3037.70 | 3232.29 | 3096.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 3027.00 | 3232.29 | 3096.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 3078.00 | 3131.00 | 3069.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 11:15:00 | 3096.05 | 3131.00 | 3069.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 12:15:00 | 3041.90 | 3129.41 | 3068.90 | SL hit (close<static) qty=1.00 sl=3056.55 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 14:15:00 | 2862.70 | 3030.78 | 3031.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 2828.95 | 2973.41 | 2996.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 11:15:00 | 2924.85 | 2894.16 | 2940.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 12:00:00 | 2924.85 | 2894.16 | 2940.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 2937.55 | 2894.59 | 2940.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:30:00 | 2929.90 | 2894.59 | 2940.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 2960.90 | 2895.25 | 2940.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 2960.90 | 2895.25 | 2940.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 3035.00 | 2896.64 | 2941.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 3035.00 | 2896.64 | 2941.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 2931.35 | 2898.68 | 2941.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 11:15:00 | 2889.70 | 2909.77 | 2942.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 12:15:00 | 2956.00 | 2909.59 | 2940.54 | SL hit (close>static) qty=1.00 sl=2951.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 2746.10 | 2607.04 | 2606.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 12:15:00 | 2775.80 | 2610.16 | 2608.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 13:15:00 | 2695.50 | 2701.95 | 2664.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 14:00:00 | 2695.50 | 2701.95 | 2664.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 2654.10 | 2701.34 | 2665.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 2696.40 | 2671.38 | 2655.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 2696.70 | 2671.78 | 2656.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 2701.70 | 2671.80 | 2656.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:15:00 | 2698.50 | 2672.04 | 2657.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 2655.00 | 2673.61 | 2658.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 2655.00 | 2673.61 | 2658.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 2637.30 | 2673.25 | 2658.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:45:00 | 2609.70 | 2673.25 | 2658.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 2702.30 | 2673.53 | 2658.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 15:15:00 | 2737.00 | 2673.53 | 2658.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 2627.40 | 2673.27 | 2660.05 | SL hit (close<static) qty=1.00 sl=2630.50 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 11:15:00 | 2486.40 | 2647.65 | 2648.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 2478.90 | 2645.97 | 2647.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 2615.90 | 2609.98 | 2627.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 10:00:00 | 2615.90 | 2609.98 | 2627.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 2530.20 | 2457.08 | 2501.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 2517.50 | 2457.08 | 2501.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 2542.90 | 2457.93 | 2501.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:45:00 | 2541.00 | 2457.93 | 2501.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2515.70 | 2478.74 | 2508.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 2515.70 | 2478.74 | 2508.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 2523.10 | 2479.18 | 2508.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 2522.60 | 2479.18 | 2508.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 2562.20 | 2482.41 | 2508.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 2562.20 | 2482.41 | 2508.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 2520.00 | 2497.12 | 2514.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 2500.80 | 2497.37 | 2514.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 2500.00 | 2498.14 | 2514.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:45:00 | 2500.50 | 2498.16 | 2513.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:00:00 | 2509.90 | 2496.15 | 2511.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 2518.40 | 2496.37 | 2511.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 2518.40 | 2496.37 | 2511.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 2511.00 | 2496.52 | 2511.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 2509.60 | 2496.52 | 2511.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2516.30 | 2496.71 | 2511.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 2494.00 | 2497.67 | 2511.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 2501.20 | 2496.42 | 2510.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:00:00 | 2500.90 | 2496.54 | 2510.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 2500.00 | 2496.78 | 2510.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 2500.00 | 2496.81 | 2510.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 2491.60 | 2496.81 | 2510.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 2521.90 | 2498.03 | 2510.33 | SL hit (close>static) qty=1.00 sl=2521.30 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 2608.00 | 2520.48 | 2520.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 2636.20 | 2526.96 | 2523.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 2570.80 | 2575.15 | 2552.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-18 09:45:00 | 2565.30 | 2575.15 | 2552.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 2548.10 | 2574.74 | 2552.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 2548.10 | 2574.74 | 2552.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2579.50 | 2574.79 | 2552.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 14:45:00 | 2601.80 | 2575.05 | 2553.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-01 11:15:00 | 2861.98 | 2637.27 | 2593.72 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 2793.20 | 2840.03 | 2840.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 14:15:00 | 2758.60 | 2836.32 | 2838.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 2833.30 | 2826.83 | 2832.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 12:15:00 | 2833.30 | 2826.83 | 2832.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 2833.30 | 2826.83 | 2832.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:00:00 | 2833.30 | 2826.83 | 2832.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 2820.20 | 2826.77 | 2832.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:45:00 | 2839.60 | 2826.77 | 2832.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2918.30 | 2827.68 | 2833.21 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 3062.70 | 2840.16 | 2839.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 3088.70 | 2845.04 | 2841.70 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-16 09:15:00 | 1283.10 | 2023-06-06 10:15:00 | 1411.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-16 13:30:00 | 1276.50 | 2023-06-06 10:15:00 | 1404.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-17 10:00:00 | 1278.60 | 2023-06-06 10:15:00 | 1406.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-17 11:45:00 | 1277.05 | 2023-06-06 10:15:00 | 1404.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-27 09:15:00 | 1747.10 | 2023-11-15 10:15:00 | 1921.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-31 11:15:00 | 3096.05 | 2024-10-31 12:15:00 | 3041.90 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-10-31 14:45:00 | 3099.00 | 2024-10-31 15:15:00 | 3052.55 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-11-01 18:00:00 | 3097.95 | 2024-11-04 09:15:00 | 2994.80 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2024-11-05 13:15:00 | 3109.90 | 2024-11-07 09:15:00 | 3040.40 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-01-03 11:15:00 | 2889.70 | 2025-01-06 12:15:00 | 2956.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-01-10 09:30:00 | 2888.80 | 2025-01-13 10:15:00 | 2744.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 12:30:00 | 2870.45 | 2025-01-13 10:15:00 | 2726.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:30:00 | 2888.80 | 2025-01-23 10:15:00 | 2913.40 | STOP_HIT | 0.50 | -0.85% |
| SELL | retest2 | 2025-01-10 12:30:00 | 2870.45 | 2025-01-23 10:15:00 | 2913.40 | STOP_HIT | 0.50 | -1.50% |
| SELL | retest2 | 2025-01-22 09:30:00 | 2766.65 | 2025-01-27 09:15:00 | 2711.59 | PARTIAL | 0.50 | 1.99% |
| SELL | retest2 | 2025-01-24 09:30:00 | 2854.30 | 2025-01-28 09:15:00 | 2628.32 | PARTIAL | 0.50 | 7.92% |
| SELL | retest2 | 2025-01-22 09:30:00 | 2766.65 | 2025-01-28 10:15:00 | 2568.87 | TARGET_HIT | 0.50 | 7.15% |
| SELL | retest2 | 2025-01-24 09:30:00 | 2854.30 | 2025-02-03 12:15:00 | 2815.65 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2025-02-04 09:15:00 | 2860.45 | 2025-02-06 12:15:00 | 2717.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 09:45:00 | 2850.00 | 2025-02-06 12:15:00 | 2707.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 09:15:00 | 2860.45 | 2025-02-11 09:15:00 | 2574.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-04 09:45:00 | 2850.00 | 2025-02-11 09:15:00 | 2565.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-12 15:00:00 | 2696.40 | 2025-08-22 10:15:00 | 2627.40 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-08-13 13:15:00 | 2696.70 | 2025-08-22 10:15:00 | 2627.40 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-08-14 09:15:00 | 2701.70 | 2025-08-22 10:15:00 | 2627.40 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-08-18 10:15:00 | 2698.50 | 2025-08-22 10:15:00 | 2627.40 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-08-19 15:15:00 | 2737.00 | 2025-08-26 09:15:00 | 2600.40 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2025-11-14 12:15:00 | 2500.80 | 2025-11-26 10:15:00 | 2521.90 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-17 09:15:00 | 2500.00 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-11-17 14:45:00 | 2500.50 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-11-19 14:00:00 | 2509.90 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-11-21 09:15:00 | 2494.00 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-11-24 11:15:00 | 2501.20 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-11-24 14:00:00 | 2500.90 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-11-24 15:15:00 | 2500.00 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-11-25 09:15:00 | 2491.60 | 2025-11-26 11:15:00 | 2537.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-12-18 14:45:00 | 2601.80 | 2026-01-01 11:15:00 | 2861.98 | TARGET_HIT | 1.00 | 10.00% |
