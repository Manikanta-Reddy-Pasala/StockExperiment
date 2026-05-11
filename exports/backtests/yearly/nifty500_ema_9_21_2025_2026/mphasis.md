# MphasiS Ltd. (MPHASIS)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1528 bars)
- **Last close:** 2214.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 58 |
| ALERT1 | 39 |
| ALERT2 | 37 |
| ALERT2_SKIP | 19 |
| ALERT3 | 83 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 42 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 4 / 35
- **Target hits / Stop hits / Partials:** 0 / 39 / 0
- **Avg / median % per leg:** -0.89% / -1.05%
- **Sum % (uncompounded):** -34.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 4 | 20.0% | 0 | 20 | 0 | -0.50% | -10.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.94% | -0.9% |
| BUY @ 3rd Alert (retest2) | 19 | 4 | 21.1% | 0 | 19 | 0 | -0.48% | -9.1% |
| SELL (all) | 19 | 0 | 0.0% | 0 | 19 | 0 | -1.30% | -24.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 0 | 0.0% | 0 | 19 | 0 | -1.30% | -24.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.94% | -0.9% |
| retest2 (combined) | 38 | 4 | 10.5% | 0 | 38 | 0 | -0.89% | -33.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2494.00 | 2406.55 | 2401.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 2543.10 | 2460.24 | 2429.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 2496.70 | 2503.03 | 2475.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 2496.70 | 2503.03 | 2475.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2521.40 | 2503.55 | 2481.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:15:00 | 2538.70 | 2508.48 | 2486.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:00:00 | 2537.20 | 2514.22 | 2490.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:00:00 | 2545.00 | 2520.38 | 2495.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 2539.90 | 2568.99 | 2569.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 2539.90 | 2568.99 | 2569.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 14:15:00 | 2535.70 | 2562.34 | 2566.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 09:15:00 | 2571.10 | 2559.07 | 2563.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 2571.10 | 2559.07 | 2563.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2571.10 | 2559.07 | 2563.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 2578.80 | 2559.07 | 2563.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2585.10 | 2564.28 | 2565.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 2585.10 | 2564.28 | 2565.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 2574.50 | 2566.32 | 2566.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 12:45:00 | 2553.10 | 2564.54 | 2565.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:30:00 | 2561.40 | 2548.14 | 2555.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 2583.50 | 2549.82 | 2549.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 2583.50 | 2549.82 | 2549.48 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 2538.00 | 2549.05 | 2550.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 2530.90 | 2545.16 | 2548.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 2546.20 | 2545.36 | 2547.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 2546.20 | 2545.36 | 2547.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 2546.20 | 2545.36 | 2547.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 2546.20 | 2545.36 | 2547.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 2540.00 | 2544.29 | 2546.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 2522.80 | 2544.29 | 2546.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2525.00 | 2540.43 | 2544.97 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 2544.80 | 2542.05 | 2541.82 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 2529.70 | 2540.09 | 2541.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 15:15:00 | 2528.50 | 2537.77 | 2539.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 2548.50 | 2539.91 | 2540.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 2548.50 | 2539.91 | 2540.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 2548.50 | 2539.91 | 2540.67 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 2562.20 | 2544.37 | 2542.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 2570.00 | 2557.36 | 2550.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2546.90 | 2555.27 | 2550.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 2546.90 | 2555.27 | 2550.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 2546.90 | 2555.27 | 2550.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 2546.90 | 2555.27 | 2550.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 2553.00 | 2554.81 | 2550.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 2554.70 | 2554.81 | 2550.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 2465.10 | 2538.15 | 2544.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 2465.10 | 2538.15 | 2544.97 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 2544.80 | 2520.53 | 2518.02 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 09:15:00 | 2494.30 | 2514.56 | 2516.89 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 2531.40 | 2517.60 | 2517.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 2548.60 | 2526.91 | 2522.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 2659.90 | 2671.95 | 2648.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 2659.90 | 2671.95 | 2648.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 2659.90 | 2671.95 | 2648.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 2651.90 | 2671.95 | 2648.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 2654.00 | 2670.87 | 2658.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 2646.70 | 2670.87 | 2658.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2670.00 | 2670.69 | 2659.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 2673.90 | 2670.69 | 2659.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 2646.40 | 2656.11 | 2656.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 14:15:00 | 2646.40 | 2656.11 | 2656.12 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 2670.20 | 2656.78 | 2656.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 2699.60 | 2668.00 | 2661.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 2694.70 | 2700.01 | 2687.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 15:00:00 | 2694.70 | 2700.01 | 2687.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 2679.10 | 2695.83 | 2686.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 2703.00 | 2695.83 | 2686.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 2697.00 | 2687.20 | 2685.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 2642.50 | 2679.83 | 2682.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 2642.50 | 2679.83 | 2682.27 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 2694.40 | 2674.59 | 2672.94 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 2658.80 | 2672.15 | 2672.16 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 2682.80 | 2673.74 | 2672.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 2713.00 | 2680.96 | 2676.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 2692.10 | 2694.61 | 2685.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 2698.80 | 2694.61 | 2685.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 2681.40 | 2691.96 | 2684.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:45:00 | 2679.00 | 2691.96 | 2684.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 2678.00 | 2689.17 | 2684.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 2704.40 | 2689.17 | 2684.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 2872.10 | 2891.51 | 2892.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 14:15:00 | 2872.10 | 2891.51 | 2892.44 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 2930.10 | 2896.71 | 2893.73 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 2853.20 | 2892.86 | 2893.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 2829.00 | 2858.15 | 2873.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 13:15:00 | 2860.10 | 2854.03 | 2866.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 14:00:00 | 2860.10 | 2854.03 | 2866.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 2871.00 | 2858.34 | 2866.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 2871.80 | 2858.34 | 2866.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 2850.20 | 2856.72 | 2865.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 2797.90 | 2838.47 | 2850.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 2819.00 | 2835.31 | 2846.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 2815.60 | 2825.84 | 2838.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 2876.50 | 2843.02 | 2842.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 2876.50 | 2843.02 | 2842.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 2891.00 | 2870.43 | 2857.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 2866.20 | 2876.30 | 2867.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 15:15:00 | 2866.20 | 2876.30 | 2867.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 2866.20 | 2876.30 | 2867.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:45:00 | 2864.90 | 2872.84 | 2866.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 2876.00 | 2873.47 | 2867.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 2856.90 | 2873.47 | 2867.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 2858.00 | 2870.38 | 2866.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 2857.20 | 2870.38 | 2866.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 2832.10 | 2862.72 | 2863.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 2818.70 | 2845.48 | 2854.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 09:15:00 | 2679.60 | 2675.45 | 2710.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 2679.60 | 2675.45 | 2710.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 2679.60 | 2675.45 | 2710.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 14:30:00 | 2617.30 | 2651.11 | 2683.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 14:15:00 | 2705.50 | 2695.14 | 2693.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 14:15:00 | 2705.50 | 2695.14 | 2693.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 11:15:00 | 2724.90 | 2705.33 | 2699.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 2753.70 | 2787.99 | 2763.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 2753.70 | 2787.99 | 2763.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2753.70 | 2787.99 | 2763.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 2753.70 | 2787.99 | 2763.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 2782.00 | 2786.80 | 2764.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:30:00 | 2749.00 | 2786.80 | 2764.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 2774.90 | 2786.34 | 2775.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 2771.20 | 2786.34 | 2775.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 2775.80 | 2784.23 | 2775.15 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 2753.50 | 2769.86 | 2770.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 2735.70 | 2763.03 | 2767.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 2743.30 | 2741.56 | 2753.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 2743.30 | 2741.56 | 2753.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 2758.60 | 2744.97 | 2754.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 2758.60 | 2744.97 | 2754.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 2746.00 | 2745.17 | 2753.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 2762.50 | 2745.17 | 2753.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 2667.40 | 2671.88 | 2688.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:30:00 | 2684.40 | 2671.88 | 2688.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2700.80 | 2678.72 | 2688.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2700.80 | 2678.72 | 2688.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2706.00 | 2684.18 | 2690.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 2684.30 | 2684.18 | 2690.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 2691.50 | 2684.81 | 2687.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 2718.60 | 2684.58 | 2684.67 | SL hit (close>static) qty=1.00 sl=2707.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 2714.70 | 2690.60 | 2687.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 2727.80 | 2703.47 | 2694.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 2707.40 | 2710.86 | 2701.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 11:00:00 | 2707.40 | 2710.86 | 2701.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2698.00 | 2708.29 | 2701.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 2698.00 | 2708.29 | 2701.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 2715.00 | 2709.63 | 2702.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 2727.20 | 2710.92 | 2705.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:15:00 | 2727.90 | 2711.73 | 2706.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 2726.30 | 2716.38 | 2710.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 2682.60 | 2708.26 | 2707.71 | SL hit (close<static) qty=1.00 sl=2696.40 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 12:15:00 | 2701.50 | 2706.91 | 2707.14 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 2731.90 | 2710.78 | 2708.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 2732.20 | 2715.06 | 2710.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 2848.80 | 2851.37 | 2814.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 2836.70 | 2851.37 | 2814.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 2884.60 | 2901.72 | 2887.53 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 2850.00 | 2876.20 | 2878.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 2835.40 | 2858.45 | 2868.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 2842.00 | 2819.26 | 2837.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 2842.00 | 2819.26 | 2837.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2842.00 | 2819.26 | 2837.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:30:00 | 2862.60 | 2819.26 | 2837.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2892.00 | 2833.81 | 2842.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 2892.00 | 2833.81 | 2842.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 2921.70 | 2851.38 | 2849.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 2925.80 | 2886.47 | 2868.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 2907.10 | 2914.01 | 2892.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 2907.10 | 2914.01 | 2892.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 2861.90 | 2899.81 | 2891.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 2861.90 | 2899.81 | 2891.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 2861.30 | 2892.11 | 2888.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 2861.30 | 2892.11 | 2888.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 2855.50 | 2884.78 | 2885.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 2781.20 | 2850.57 | 2866.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 2796.00 | 2793.12 | 2816.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:30:00 | 2789.30 | 2793.12 | 2816.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2818.00 | 2796.60 | 2810.55 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 2843.10 | 2819.74 | 2818.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 2961.40 | 2859.57 | 2838.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 2930.00 | 2937.50 | 2898.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:30:00 | 2941.20 | 2937.50 | 2898.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 2906.90 | 2926.83 | 2902.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 2910.70 | 2926.83 | 2902.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 2900.10 | 2921.49 | 2902.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 2900.10 | 2921.49 | 2902.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 2923.70 | 2921.93 | 2904.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 2946.00 | 2922.74 | 2906.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 2897.30 | 2914.37 | 2906.47 | SL hit (close<static) qty=1.00 sl=2900.10 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 2881.70 | 2901.67 | 2902.75 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 2913.10 | 2904.56 | 2903.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 2917.50 | 2907.14 | 2905.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 13:15:00 | 2986.00 | 2991.73 | 2972.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 14:00:00 | 2986.00 | 2991.73 | 2972.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 2978.90 | 2991.51 | 2981.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 2979.70 | 2991.51 | 2981.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 2977.10 | 2988.63 | 2981.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 2975.00 | 2988.63 | 2981.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 2873.60 | 2967.77 | 2973.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 2779.30 | 2865.50 | 2910.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 2749.60 | 2745.42 | 2784.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:45:00 | 2749.60 | 2745.42 | 2784.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 2687.30 | 2673.01 | 2689.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 2658.30 | 2672.81 | 2688.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:45:00 | 2661.20 | 2661.40 | 2676.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 2649.00 | 2655.27 | 2671.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:30:00 | 2660.30 | 2663.47 | 2672.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 2678.30 | 2666.44 | 2672.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 2690.00 | 2677.61 | 2676.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 2690.00 | 2677.61 | 2676.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 2706.10 | 2683.31 | 2679.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 13:15:00 | 2818.70 | 2821.16 | 2795.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:00:00 | 2818.70 | 2821.16 | 2795.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 2794.90 | 2815.91 | 2795.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 2794.90 | 2815.91 | 2795.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 2798.80 | 2812.49 | 2795.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 2811.80 | 2812.49 | 2795.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:15:00 | 2815.50 | 2811.71 | 2796.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 2810.00 | 2806.15 | 2799.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 2787.20 | 2802.98 | 2799.26 | SL hit (close<static) qty=1.00 sl=2793.50 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 2782.00 | 2796.08 | 2796.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 2757.90 | 2782.47 | 2789.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 2760.20 | 2758.19 | 2770.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 2760.20 | 2758.19 | 2770.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 2760.20 | 2758.19 | 2770.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 2779.20 | 2758.19 | 2770.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2791.40 | 2757.37 | 2762.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 2792.30 | 2757.37 | 2762.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 2784.20 | 2762.73 | 2764.50 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 2787.10 | 2767.61 | 2766.56 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 2721.90 | 2767.54 | 2770.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 13:15:00 | 2693.20 | 2730.83 | 2749.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 2717.60 | 2714.63 | 2734.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:00:00 | 2717.60 | 2714.63 | 2734.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 2743.70 | 2724.53 | 2734.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 2746.30 | 2724.53 | 2734.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 2750.30 | 2729.69 | 2736.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 2750.30 | 2729.69 | 2736.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 2748.00 | 2733.35 | 2737.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 2760.90 | 2735.68 | 2737.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 2806.20 | 2750.47 | 2744.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 2826.80 | 2797.46 | 2775.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 15:15:00 | 2812.10 | 2812.70 | 2794.35 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:15:00 | 2851.30 | 2812.70 | 2794.35 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 2848.00 | 2866.53 | 2842.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 2848.00 | 2866.53 | 2842.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 2824.60 | 2858.14 | 2841.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 2824.60 | 2858.14 | 2841.21 | SL hit (close<ema400) qty=1.00 sl=2841.21 alert=retest1 |

### Cycle 40 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 2801.60 | 2834.82 | 2835.24 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 2885.20 | 2843.48 | 2838.86 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 2793.60 | 2853.46 | 2854.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 2770.80 | 2836.92 | 2846.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 2786.10 | 2770.25 | 2793.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 15:00:00 | 2786.10 | 2770.25 | 2793.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2775.20 | 2774.29 | 2791.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 2763.10 | 2772.05 | 2789.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 2772.80 | 2766.56 | 2774.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 2771.60 | 2767.57 | 2774.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 2777.40 | 2763.97 | 2763.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 2777.40 | 2763.97 | 2763.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 2789.30 | 2769.04 | 2765.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 2768.00 | 2770.64 | 2767.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 2768.00 | 2770.64 | 2767.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2768.00 | 2770.64 | 2767.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 2767.30 | 2770.64 | 2767.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 2775.20 | 2771.55 | 2768.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 2787.90 | 2774.35 | 2770.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 2781.20 | 2805.04 | 2806.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 2781.20 | 2805.04 | 2806.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 2749.10 | 2791.34 | 2799.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 2709.50 | 2681.91 | 2707.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 2709.50 | 2681.91 | 2707.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2709.50 | 2681.91 | 2707.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 2709.50 | 2681.91 | 2707.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2704.00 | 2686.33 | 2707.19 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 2746.40 | 2714.30 | 2714.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 2755.00 | 2729.73 | 2721.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 14:15:00 | 2738.00 | 2748.87 | 2738.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 14:15:00 | 2738.00 | 2748.87 | 2738.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 2738.00 | 2748.87 | 2738.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 2738.00 | 2748.87 | 2738.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 2733.70 | 2745.84 | 2738.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 2795.00 | 2745.84 | 2738.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 15:15:00 | 2885.90 | 2903.83 | 2904.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 2885.90 | 2903.83 | 2904.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 2851.00 | 2893.27 | 2899.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2914.80 | 2895.08 | 2898.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 2914.80 | 2895.08 | 2898.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 2914.80 | 2895.08 | 2898.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 2914.80 | 2895.08 | 2898.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2887.20 | 2893.51 | 2897.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 2871.90 | 2891.47 | 2896.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 2863.90 | 2884.15 | 2891.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 2858.40 | 2868.23 | 2879.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 2893.90 | 2884.36 | 2884.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 2893.90 | 2884.36 | 2884.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 2901.30 | 2887.75 | 2885.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 15:15:00 | 2885.20 | 2899.60 | 2894.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 15:15:00 | 2885.20 | 2899.60 | 2894.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 2885.20 | 2899.60 | 2894.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 2879.50 | 2899.60 | 2894.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2879.10 | 2895.50 | 2893.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 2873.50 | 2895.50 | 2893.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2899.60 | 2896.32 | 2893.85 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 2849.40 | 2886.26 | 2890.18 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 2900.00 | 2877.40 | 2875.05 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 2116.10 | 2730.90 | 2811.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 2103.90 | 2427.99 | 2637.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2136.60 | 2113.32 | 2276.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 10:00:00 | 2136.60 | 2113.32 | 2276.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 2228.80 | 2158.96 | 2197.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 2228.80 | 2158.96 | 2197.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 2217.00 | 2170.56 | 2199.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:30:00 | 2218.00 | 2170.56 | 2199.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 2204.00 | 2188.39 | 2201.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 2198.90 | 2188.39 | 2201.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 2224.80 | 2203.41 | 2205.66 | SL hit (close>static) qty=1.00 sl=2220.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 2239.00 | 2210.53 | 2208.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 2273.10 | 2225.84 | 2216.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 09:15:00 | 2311.10 | 2357.53 | 2334.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 2311.10 | 2357.53 | 2334.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 2311.10 | 2357.53 | 2334.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:30:00 | 2286.80 | 2357.53 | 2334.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 2311.90 | 2348.41 | 2332.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 2325.90 | 2340.15 | 2331.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2325.50 | 2329.31 | 2328.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 2323.40 | 2328.13 | 2328.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 2323.40 | 2328.13 | 2328.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 13:15:00 | 2317.60 | 2325.72 | 2327.12 | Break + close below crossover candle low |

### Cycle 53 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 2373.00 | 2331.64 | 2329.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 2393.70 | 2351.23 | 2338.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 2436.00 | 2445.45 | 2424.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 2436.00 | 2445.45 | 2424.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 2436.00 | 2445.45 | 2424.96 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 2416.40 | 2423.68 | 2423.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 2388.50 | 2415.47 | 2420.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 2231.30 | 2202.31 | 2253.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:45:00 | 2232.60 | 2202.31 | 2253.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 2244.00 | 2210.65 | 2252.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 2248.50 | 2210.65 | 2252.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 2258.50 | 2220.22 | 2253.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 2258.50 | 2220.22 | 2253.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 2249.90 | 2226.15 | 2252.85 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 2267.20 | 2261.46 | 2261.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 2335.40 | 2277.61 | 2268.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 11:15:00 | 2256.00 | 2278.31 | 2270.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 11:15:00 | 2256.00 | 2278.31 | 2270.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 2256.00 | 2278.31 | 2270.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 2256.00 | 2278.31 | 2270.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 2253.40 | 2273.33 | 2269.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:30:00 | 2253.00 | 2273.33 | 2269.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 2254.90 | 2265.06 | 2265.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 2241.90 | 2260.43 | 2263.78 | Break + close below crossover candle low |

### Cycle 57 — BUY (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 09:15:00 | 2304.90 | 2269.32 | 2267.52 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 2256.50 | 2274.57 | 2276.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 2223.50 | 2261.59 | 2269.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 2233.90 | 2228.44 | 2246.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 2233.90 | 2228.44 | 2246.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 2233.90 | 2228.44 | 2246.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:30:00 | 2221.60 | 2226.19 | 2243.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:30:00 | 2218.30 | 2222.09 | 2240.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 15:15:00 | 2220.00 | 2216.44 | 2232.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 13:30:00 | 2217.70 | 2207.77 | 2212.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 2214.90 | 2209.19 | 2213.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 2225.00 | 2209.19 | 2213.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 2214.50 | 2210.25 | 2213.24 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 11:15:00 | 2538.70 | 2025-05-19 13:15:00 | 2539.90 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-05-14 12:00:00 | 2537.20 | 2025-05-19 13:15:00 | 2539.90 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-05-14 13:00:00 | 2545.00 | 2025-05-19 13:15:00 | 2539.90 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-05-20 12:45:00 | 2553.10 | 2025-05-23 09:15:00 | 2583.50 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-05-21 11:30:00 | 2561.40 | 2025-05-23 09:15:00 | 2583.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-30 11:15:00 | 2554.70 | 2025-06-02 09:15:00 | 2465.10 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2025-06-13 10:15:00 | 2673.90 | 2025-06-13 14:15:00 | 2646.40 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-06-18 09:15:00 | 2703.00 | 2025-06-19 09:15:00 | 2642.50 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-06-18 15:15:00 | 2697.00 | 2025-06-19 09:15:00 | 2642.50 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-06-25 09:15:00 | 2704.40 | 2025-07-07 14:15:00 | 2872.10 | STOP_HIT | 1.00 | 6.20% |
| SELL | retest2 | 2025-07-14 09:15:00 | 2797.90 | 2025-07-15 11:15:00 | 2876.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-07-14 11:45:00 | 2819.00 | 2025-07-15 11:15:00 | 2876.50 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-07-14 15:15:00 | 2815.60 | 2025-07-15 11:15:00 | 2876.50 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-07-25 14:30:00 | 2617.30 | 2025-07-28 14:15:00 | 2705.50 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-08-08 09:15:00 | 2684.30 | 2025-08-12 09:15:00 | 2718.60 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-08 15:00:00 | 2691.50 | 2025-08-12 09:15:00 | 2718.60 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-08-14 10:30:00 | 2727.20 | 2025-08-18 11:15:00 | 2682.60 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-08-14 14:15:00 | 2727.90 | 2025-08-18 11:15:00 | 2682.60 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-08-18 09:30:00 | 2726.30 | 2025-08-18 11:15:00 | 2682.60 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-12 09:15:00 | 2946.00 | 2025-09-12 11:15:00 | 2897.30 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-09-30 11:15:00 | 2658.30 | 2025-10-01 15:15:00 | 2690.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-09-30 14:45:00 | 2661.20 | 2025-10-01 15:15:00 | 2690.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-01 09:45:00 | 2649.00 | 2025-10-01 15:15:00 | 2690.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-10-01 11:30:00 | 2660.30 | 2025-10-01 15:15:00 | 2690.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-09 09:15:00 | 2811.80 | 2025-10-10 09:15:00 | 2787.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-10-09 10:15:00 | 2815.50 | 2025-10-10 09:15:00 | 2787.20 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-10-09 15:15:00 | 2810.00 | 2025-10-10 09:15:00 | 2787.20 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest1 | 2025-10-27 09:15:00 | 2851.30 | 2025-10-28 11:15:00 | 2824.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-04 11:00:00 | 2763.10 | 2025-11-10 12:15:00 | 2777.40 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-06 14:15:00 | 2772.80 | 2025-11-10 12:15:00 | 2777.40 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-11-06 15:00:00 | 2771.60 | 2025-11-10 12:15:00 | 2777.40 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-11-11 15:15:00 | 2787.90 | 2025-11-14 10:15:00 | 2781.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-11-24 09:15:00 | 2795.00 | 2025-12-08 15:15:00 | 2885.90 | STOP_HIT | 1.00 | 3.25% |
| SELL | retest2 | 2025-12-09 15:15:00 | 2871.90 | 2025-12-11 13:15:00 | 2893.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-10 10:45:00 | 2863.90 | 2025-12-11 13:15:00 | 2893.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-11 09:30:00 | 2858.40 | 2025-12-11 13:15:00 | 2893.90 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-04-06 11:15:00 | 2198.90 | 2026-04-06 13:15:00 | 2224.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-04-10 12:30:00 | 2325.90 | 2026-04-13 11:15:00 | 2323.40 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2325.50 | 2026-04-13 11:15:00 | 2323.40 | STOP_HIT | 1.00 | -0.09% |
