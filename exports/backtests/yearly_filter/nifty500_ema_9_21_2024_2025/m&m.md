# Mahindra & Mahindra Ltd. (M&M)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 3331.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 149 |
| ALERT1 | 98 |
| ALERT2 | 97 |
| ALERT2_SKIP | 49 |
| ALERT3 | 242 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 115 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 113 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 86
- **Target hits / Stop hits / Partials:** 4 / 113 / 11
- **Avg / median % per leg:** 0.16% / -0.47%
- **Sum % (uncompounded):** 20.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 16 | 26.7% | 3 | 56 | 1 | 0.13% | 7.9% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.58% | 7.2% |
| BUY @ 3rd Alert (retest2) | 58 | 14 | 24.1% | 3 | 55 | 0 | 0.01% | 0.7% |
| SELL (all) | 68 | 26 | 38.2% | 1 | 57 | 10 | 0.19% | 13.1% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 66 | 24 | 36.4% | 0 | 57 | 9 | -0.03% | -1.9% |
| retest1 (combined) | 4 | 4 | 100.0% | 1 | 1 | 2 | 5.54% | 22.2% |
| retest2 (combined) | 124 | 38 | 30.6% | 3 | 112 | 9 | -0.01% | -1.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 2216.85 | 2194.33 | 2194.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 2260.00 | 2207.46 | 2200.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 2294.00 | 2298.47 | 2275.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 2294.00 | 2298.47 | 2275.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 2558.00 | 2578.28 | 2559.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:45:00 | 2559.45 | 2578.28 | 2559.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 2559.30 | 2574.48 | 2559.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 2569.70 | 2574.25 | 2560.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 14:15:00 | 2565.00 | 2572.57 | 2562.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 14:15:00 | 2550.05 | 2568.06 | 2561.40 | SL hit (close<static) qty=1.00 sl=2553.90 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 2550.00 | 2558.96 | 2559.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 2512.00 | 2549.57 | 2555.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 2548.15 | 2514.51 | 2525.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 2548.15 | 2514.51 | 2525.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 2548.15 | 2514.51 | 2525.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 13:00:00 | 2503.85 | 2519.61 | 2525.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:00:00 | 2503.80 | 2516.18 | 2522.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 2635.60 | 2540.35 | 2532.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 2635.60 | 2540.35 | 2532.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 2725.00 | 2616.33 | 2590.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 2684.35 | 2703.01 | 2661.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 11:45:00 | 2688.25 | 2703.01 | 2661.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 2700.00 | 2698.51 | 2674.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 11:00:00 | 2734.85 | 2705.78 | 2679.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 3008.34 | 2924.54 | 2881.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 11:15:00 | 2887.40 | 2920.31 | 2921.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 12:15:00 | 2860.00 | 2908.25 | 2916.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 2893.15 | 2891.13 | 2904.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 10:00:00 | 2893.15 | 2891.13 | 2904.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 2888.75 | 2890.66 | 2902.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 2890.00 | 2890.66 | 2902.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 2911.90 | 2894.73 | 2901.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:00:00 | 2911.90 | 2894.73 | 2901.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 2832.85 | 2882.36 | 2895.37 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 2921.20 | 2895.45 | 2893.78 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 10:15:00 | 2875.60 | 2893.83 | 2895.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 2851.10 | 2880.21 | 2888.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 2890.90 | 2859.59 | 2870.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 2890.90 | 2859.59 | 2870.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 2890.90 | 2859.59 | 2870.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 2890.90 | 2859.59 | 2870.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 2885.00 | 2864.67 | 2871.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 2870.20 | 2864.67 | 2871.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 2881.05 | 2871.49 | 2872.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 2881.75 | 2873.14 | 2872.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 2881.75 | 2873.14 | 2872.51 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 2862.00 | 2871.14 | 2871.95 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 13:15:00 | 2879.10 | 2872.73 | 2872.60 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 14:15:00 | 2865.70 | 2871.32 | 2871.97 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 2885.00 | 2872.72 | 2872.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 2910.60 | 2882.26 | 2877.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 2850.35 | 2887.63 | 2884.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 2850.35 | 2887.63 | 2884.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 2850.35 | 2887.63 | 2884.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:30:00 | 2853.55 | 2887.63 | 2884.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 2850.00 | 2880.11 | 2881.67 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 2909.70 | 2878.87 | 2876.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 13:15:00 | 2932.50 | 2889.60 | 2881.46 | Break + close above crossover candle high |

### Cycle 14 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 2760.05 | 2873.72 | 2877.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 2732.00 | 2845.37 | 2864.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 2726.10 | 2712.26 | 2734.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 2726.10 | 2712.26 | 2734.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2726.10 | 2712.26 | 2734.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:45:00 | 2725.15 | 2712.26 | 2734.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 2732.05 | 2719.36 | 2729.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 2732.05 | 2719.36 | 2729.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 2730.00 | 2721.49 | 2729.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 2762.20 | 2721.49 | 2729.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 2749.05 | 2727.00 | 2731.51 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 2748.50 | 2734.96 | 2734.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 12:15:00 | 2753.95 | 2738.76 | 2736.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 2789.00 | 2789.81 | 2769.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 2789.00 | 2789.81 | 2769.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 2767.70 | 2783.75 | 2772.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:00:00 | 2767.70 | 2783.75 | 2772.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 2768.05 | 2780.61 | 2771.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:00:00 | 2768.05 | 2780.61 | 2771.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 2748.30 | 2774.15 | 2769.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 15:00:00 | 2748.30 | 2774.15 | 2769.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 2756.20 | 2770.56 | 2768.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 2740.45 | 2770.56 | 2768.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 2814.90 | 2809.71 | 2795.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 2782.10 | 2809.71 | 2795.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2778.50 | 2807.11 | 2799.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 2778.50 | 2807.11 | 2799.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 2783.90 | 2802.47 | 2797.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:30:00 | 2764.95 | 2802.47 | 2797.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 12:15:00 | 2784.40 | 2794.64 | 2794.94 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 2805.15 | 2795.67 | 2795.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 15:15:00 | 2815.70 | 2799.68 | 2797.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 2795.85 | 2798.91 | 2797.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 2795.85 | 2798.91 | 2797.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 2795.85 | 2798.91 | 2797.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:45:00 | 2805.60 | 2799.92 | 2797.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 2839.05 | 2901.58 | 2904.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 2839.05 | 2901.58 | 2904.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 2827.35 | 2886.74 | 2897.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2700.10 | 2697.41 | 2743.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 2700.10 | 2697.41 | 2743.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 2700.10 | 2697.41 | 2743.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:45:00 | 2681.95 | 2695.48 | 2731.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 09:45:00 | 2685.00 | 2668.47 | 2705.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 2680.30 | 2681.90 | 2690.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:45:00 | 2681.65 | 2682.39 | 2690.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 2684.20 | 2682.75 | 2689.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 2740.70 | 2693.93 | 2693.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 2740.70 | 2693.93 | 2693.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 15:15:00 | 2764.80 | 2733.98 | 2716.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 2719.75 | 2731.14 | 2716.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 09:45:00 | 2720.10 | 2731.14 | 2716.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 2726.30 | 2730.17 | 2717.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 11:15:00 | 2734.00 | 2730.17 | 2717.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:00:00 | 2730.60 | 2724.82 | 2719.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:45:00 | 2730.80 | 2725.49 | 2720.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:15:00 | 2734.95 | 2725.49 | 2720.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 2725.95 | 2726.04 | 2721.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 2725.95 | 2726.04 | 2721.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 2711.15 | 2723.07 | 2720.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 2711.15 | 2723.07 | 2720.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 2716.65 | 2721.78 | 2720.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 2748.40 | 2721.03 | 2720.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 2753.80 | 2775.19 | 2775.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 10:15:00 | 2753.80 | 2775.19 | 2775.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 12:15:00 | 2734.40 | 2754.38 | 2761.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 2748.00 | 2744.96 | 2753.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 2748.00 | 2744.96 | 2753.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 2748.00 | 2744.96 | 2753.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 2753.00 | 2744.96 | 2753.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 2760.00 | 2747.97 | 2754.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 2760.00 | 2747.97 | 2754.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 2752.10 | 2748.79 | 2754.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 2755.90 | 2748.79 | 2754.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 2764.35 | 2751.91 | 2755.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 2764.35 | 2751.91 | 2755.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 2763.45 | 2754.21 | 2755.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 2763.45 | 2754.21 | 2755.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 2761.00 | 2756.44 | 2756.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 2776.95 | 2756.44 | 2756.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 2781.35 | 2761.42 | 2758.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 14:15:00 | 2791.60 | 2773.91 | 2766.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 09:15:00 | 2776.95 | 2777.79 | 2769.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 09:45:00 | 2780.00 | 2777.79 | 2769.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 2788.50 | 2783.35 | 2775.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:30:00 | 2777.10 | 2783.35 | 2775.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 2780.35 | 2795.63 | 2789.47 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 2763.95 | 2784.83 | 2785.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 2750.35 | 2775.08 | 2780.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 2777.10 | 2771.01 | 2776.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 2777.10 | 2771.01 | 2776.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 2777.10 | 2771.01 | 2776.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 2777.10 | 2771.01 | 2776.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 2797.10 | 2776.23 | 2778.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 2797.10 | 2776.23 | 2778.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 2813.00 | 2783.58 | 2781.90 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 11:15:00 | 2759.45 | 2783.53 | 2785.03 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 12:15:00 | 2788.35 | 2783.71 | 2783.20 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 2742.70 | 2776.16 | 2780.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 12:15:00 | 2735.60 | 2758.26 | 2770.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 15:15:00 | 2755.20 | 2753.62 | 2764.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 09:15:00 | 2740.75 | 2753.62 | 2764.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 2705.20 | 2704.30 | 2713.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 2690.40 | 2704.30 | 2713.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:30:00 | 2704.75 | 2699.23 | 2706.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 13:15:00 | 2724.40 | 2685.57 | 2686.56 | SL hit (close>static) qty=1.00 sl=2714.30 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 2744.45 | 2697.34 | 2691.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 2767.45 | 2735.39 | 2718.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 2756.35 | 2758.18 | 2737.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 14:30:00 | 2755.00 | 2758.18 | 2737.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 2795.90 | 2811.82 | 2796.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 2795.90 | 2811.82 | 2796.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 2800.20 | 2809.50 | 2796.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 2833.00 | 2805.84 | 2797.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-25 09:15:00 | 3116.30 | 3067.57 | 3014.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 15:15:00 | 3115.00 | 3124.28 | 3124.86 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 3132.80 | 3125.98 | 3125.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 11:15:00 | 3179.75 | 3137.97 | 3131.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 3146.95 | 3154.34 | 3145.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 11:15:00 | 3146.95 | 3154.34 | 3145.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 3146.95 | 3154.34 | 3145.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:45:00 | 3137.10 | 3154.34 | 3145.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 3163.65 | 3156.20 | 3146.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:45:00 | 3140.10 | 3156.20 | 3146.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 3144.90 | 3153.94 | 3146.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:30:00 | 3143.85 | 3153.94 | 3146.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 3128.95 | 3148.94 | 3145.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:30:00 | 3133.95 | 3148.94 | 3145.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 3094.95 | 3136.33 | 3139.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 10:15:00 | 3079.00 | 3124.86 | 3134.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 14:15:00 | 3059.15 | 3041.56 | 3069.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 15:00:00 | 3059.15 | 3041.56 | 3069.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 3101.15 | 3056.75 | 3071.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:15:00 | 3130.85 | 3056.75 | 3071.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 3131.00 | 3071.60 | 3076.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:45:00 | 3127.15 | 3071.60 | 3076.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 11:15:00 | 3123.25 | 3081.93 | 3081.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 3168.15 | 3116.77 | 3098.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 3152.75 | 3157.33 | 3132.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 15:00:00 | 3152.75 | 3157.33 | 3132.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 3180.85 | 3185.07 | 3165.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 3172.25 | 3185.07 | 3165.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 3150.60 | 3178.17 | 3164.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 3149.05 | 3178.17 | 3164.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 3147.35 | 3172.01 | 3162.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:15:00 | 3139.85 | 3172.01 | 3162.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 3124.55 | 3155.81 | 3156.43 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 13:15:00 | 3160.75 | 3155.48 | 3155.08 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 3098.95 | 3144.00 | 3149.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 3096.15 | 3121.60 | 3137.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 13:15:00 | 3127.65 | 3122.81 | 3136.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-15 14:00:00 | 3127.65 | 3122.81 | 3136.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 3153.65 | 3128.98 | 3137.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 3153.65 | 3128.98 | 3137.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 3150.35 | 3133.25 | 3138.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 3103.10 | 3133.25 | 3138.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 2947.94 | 2982.02 | 3031.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-21 13:15:00 | 2988.60 | 2951.14 | 2978.19 | SL hit (close>ema200) qty=0.50 sl=2951.14 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 2798.85 | 2734.95 | 2734.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 18:15:00 | 2826.45 | 2753.25 | 2742.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 11:15:00 | 2832.05 | 2852.56 | 2818.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 11:45:00 | 2825.50 | 2852.56 | 2818.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 2875.85 | 2905.33 | 2879.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 2875.85 | 2905.33 | 2879.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 2896.45 | 2903.56 | 2880.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 2869.75 | 2903.56 | 2880.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 2897.30 | 2902.31 | 2882.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:15:00 | 2935.00 | 2902.31 | 2882.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:45:00 | 2930.00 | 2907.28 | 2886.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 10:45:00 | 2921.20 | 2906.09 | 2893.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 2918.00 | 2906.09 | 2893.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 2925.00 | 2937.37 | 2923.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 14:30:00 | 2936.00 | 2931.06 | 2922.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 15:00:00 | 2935.00 | 2931.06 | 2922.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 09:30:00 | 2936.00 | 2929.01 | 2922.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 10:15:00 | 2938.75 | 2929.01 | 2922.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 2921.70 | 2927.55 | 2922.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 2919.35 | 2927.55 | 2922.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 2922.65 | 2926.57 | 2922.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:45:00 | 2912.85 | 2926.57 | 2922.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 2904.95 | 2922.24 | 2921.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-12 12:15:00 | 2904.95 | 2922.24 | 2921.10 | SL hit (close<static) qty=1.00 sl=2913.30 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 2902.90 | 2918.37 | 2919.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 2804.25 | 2887.61 | 2904.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 14:15:00 | 2807.90 | 2807.56 | 2834.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 15:00:00 | 2807.90 | 2807.56 | 2834.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 2817.55 | 2811.27 | 2831.22 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 2948.90 | 2855.45 | 2844.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 2982.80 | 2907.01 | 2873.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 15:15:00 | 2925.05 | 2933.80 | 2913.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:15:00 | 2938.75 | 2933.80 | 2913.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 2955.00 | 2938.04 | 2917.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:45:00 | 2922.50 | 2938.04 | 2917.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 2933.10 | 2937.05 | 2918.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:45:00 | 2931.10 | 2937.05 | 2918.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 09:15:00 | 3085.69 | 3003.66 | 2961.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 3002.00 | 3036.41 | 3005.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-26 09:15:00 | 3002.00 | 3036.41 | 3005.31 | SL hit (close<ema200) qty=0.50 sl=3036.41 alert=retest1 |

### Cycle 38 — SELL (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 09:15:00 | 2952.15 | 2995.23 | 2998.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 2931.30 | 2982.45 | 2992.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 2935.60 | 2933.59 | 2959.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 09:45:00 | 2947.40 | 2933.59 | 2959.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 2960.70 | 2939.01 | 2959.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 2960.70 | 2939.01 | 2959.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 2953.10 | 2941.83 | 2958.63 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 2996.00 | 2968.76 | 2966.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 3018.00 | 2978.61 | 2970.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 15:15:00 | 3025.00 | 3029.77 | 3018.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:15:00 | 3040.90 | 3029.77 | 3018.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 3034.95 | 3030.81 | 3019.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 3026.20 | 3030.81 | 3019.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 3033.00 | 3031.25 | 3020.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 3023.00 | 3031.25 | 3020.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 3053.00 | 3035.60 | 3023.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:45:00 | 3023.55 | 3035.60 | 3023.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 3061.05 | 3072.39 | 3059.35 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 3019.15 | 3050.63 | 3053.73 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 3091.30 | 3058.25 | 3054.21 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 3020.20 | 3057.25 | 3061.60 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 3076.20 | 3063.93 | 3062.81 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 10:15:00 | 3045.90 | 3062.14 | 3062.55 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 3066.90 | 3063.09 | 3062.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 3084.75 | 3069.44 | 3066.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 3059.80 | 3070.82 | 3067.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 10:15:00 | 3059.80 | 3070.82 | 3067.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 3059.80 | 3070.82 | 3067.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:00:00 | 3059.80 | 3070.82 | 3067.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 3031.80 | 3063.02 | 3064.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 3004.90 | 3040.40 | 3048.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 2934.85 | 2923.57 | 2950.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 2934.85 | 2923.57 | 2950.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 2935.00 | 2925.86 | 2949.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:45:00 | 2938.10 | 2925.86 | 2949.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 2958.00 | 2933.59 | 2943.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:30:00 | 2948.15 | 2933.59 | 2943.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 2972.50 | 2941.37 | 2945.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:00:00 | 2972.50 | 2941.37 | 2945.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 2975.05 | 2948.10 | 2948.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:00:00 | 2975.05 | 2948.10 | 2948.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 2975.65 | 2953.61 | 2951.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 15:15:00 | 2984.20 | 2966.14 | 2957.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 3025.00 | 3030.67 | 3004.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 09:30:00 | 3019.50 | 3030.67 | 3004.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 2992.00 | 3025.53 | 3012.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 2992.00 | 3025.53 | 3012.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 3015.00 | 3023.42 | 3012.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 2983.05 | 3023.42 | 3012.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 2978.00 | 3014.34 | 3009.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:45:00 | 2980.75 | 3014.34 | 3009.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 3001.00 | 3010.89 | 3008.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:00:00 | 3001.00 | 3010.89 | 3008.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 3002.25 | 3009.16 | 3008.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:45:00 | 2998.40 | 3009.16 | 3008.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 3004.85 | 3008.38 | 3008.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:30:00 | 3004.95 | 3008.38 | 3008.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 3014.85 | 3009.68 | 3008.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 3006.00 | 3009.68 | 3008.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 3013.20 | 3010.38 | 3009.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:30:00 | 3026.00 | 3020.10 | 3013.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 14:15:00 | 3103.40 | 3142.28 | 3142.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 3103.40 | 3142.28 | 3142.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 3095.95 | 3133.02 | 3138.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 3118.65 | 3114.15 | 3123.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 3118.65 | 3114.15 | 3123.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 3118.65 | 3114.15 | 3123.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:00:00 | 3100.80 | 3113.30 | 3121.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:30:00 | 3096.00 | 3101.77 | 3111.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:45:00 | 3100.95 | 3108.36 | 3112.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 3151.60 | 3117.00 | 3116.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 12:15:00 | 3151.60 | 3117.00 | 3116.12 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 3078.85 | 3115.85 | 3116.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 3017.75 | 3086.75 | 3101.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 3072.95 | 3038.76 | 3061.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 3072.95 | 3038.76 | 3061.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 3072.95 | 3038.76 | 3061.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 3072.95 | 3038.76 | 3061.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 3066.95 | 3044.40 | 3062.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 14:15:00 | 3052.35 | 3050.35 | 3061.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 09:15:00 | 2899.73 | 2924.82 | 2954.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 2891.95 | 2842.90 | 2859.34 | SL hit (close>ema200) qty=0.50 sl=2842.90 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 2893.50 | 2867.45 | 2866.97 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 2834.80 | 2861.99 | 2865.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 2800.70 | 2844.25 | 2856.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 12:15:00 | 2834.00 | 2817.15 | 2835.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 13:00:00 | 2834.00 | 2817.15 | 2835.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 2829.30 | 2819.58 | 2835.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 14:00:00 | 2829.30 | 2819.58 | 2835.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 2831.10 | 2821.88 | 2834.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 2831.10 | 2821.88 | 2834.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 2830.00 | 2823.51 | 2834.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 2796.55 | 2823.51 | 2834.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 10:15:00 | 2854.50 | 2828.56 | 2834.64 | SL hit (close>static) qty=1.00 sl=2835.75 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 2894.95 | 2841.84 | 2840.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 12:15:00 | 2898.75 | 2853.22 | 2845.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 14:15:00 | 2989.00 | 2993.64 | 2962.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 15:00:00 | 2989.00 | 2993.64 | 2962.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 3134.40 | 3169.70 | 3151.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 3127.25 | 3169.70 | 3151.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 3139.15 | 3163.59 | 3150.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:30:00 | 3144.25 | 3163.59 | 3150.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 3160.15 | 3162.90 | 3151.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 3177.20 | 3152.51 | 3149.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:45:00 | 3174.85 | 3157.97 | 3152.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 3187.25 | 3163.19 | 3155.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:15:00 | 3200.00 | 3166.38 | 3158.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 3124.35 | 3157.97 | 3155.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 3124.35 | 3157.97 | 3155.15 | SL hit (close<static) qty=1.00 sl=3126.55 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 3130.75 | 3162.17 | 3164.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 3118.65 | 3153.47 | 3160.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 3026.75 | 3021.29 | 3064.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 13:30:00 | 2989.50 | 3008.46 | 3044.70 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 2840.03 | 2919.61 | 2969.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-20 09:15:00 | 2690.55 | 2771.84 | 2802.44 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 55 — BUY (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 15:15:00 | 2832.35 | 2817.45 | 2815.73 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 2694.05 | 2792.77 | 2804.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 2674.45 | 2769.11 | 2792.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 2699.05 | 2696.30 | 2732.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 11:45:00 | 2701.80 | 2696.30 | 2732.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 2774.90 | 2718.70 | 2729.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 2774.50 | 2718.70 | 2729.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 2790.00 | 2732.96 | 2735.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:45:00 | 2784.00 | 2732.96 | 2735.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 11:15:00 | 2798.00 | 2745.97 | 2741.02 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 2716.10 | 2746.92 | 2747.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 2598.45 | 2705.37 | 2726.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 2635.40 | 2628.75 | 2669.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 2635.40 | 2628.75 | 2669.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 2635.40 | 2628.75 | 2669.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:45:00 | 2597.00 | 2622.17 | 2658.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 13:45:00 | 2610.90 | 2621.01 | 2652.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 15:00:00 | 2611.75 | 2619.16 | 2648.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 2593.65 | 2618.12 | 2645.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 2704.55 | 2627.30 | 2632.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 2704.55 | 2627.30 | 2632.64 | SL hit (close>static) qty=1.00 sl=2695.95 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 2716.35 | 2645.11 | 2640.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 2731.90 | 2662.47 | 2648.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 2721.30 | 2730.79 | 2710.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 2721.30 | 2730.79 | 2710.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 2706.50 | 2724.73 | 2715.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:15:00 | 2727.30 | 2724.28 | 2715.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 2648.75 | 2702.11 | 2708.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 2648.75 | 2702.11 | 2708.77 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 2679.30 | 2664.61 | 2663.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 2705.55 | 2673.60 | 2667.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 14:15:00 | 2791.25 | 2793.84 | 2761.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 15:00:00 | 2791.25 | 2793.84 | 2761.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2801.85 | 2835.59 | 2817.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2801.85 | 2835.59 | 2817.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2787.20 | 2825.91 | 2814.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:15:00 | 2749.90 | 2825.91 | 2814.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 10:15:00 | 2725.00 | 2791.86 | 2800.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 09:15:00 | 2654.90 | 2719.37 | 2739.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 2706.00 | 2683.16 | 2705.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 2706.00 | 2683.16 | 2705.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 2706.00 | 2683.16 | 2705.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 2705.90 | 2683.16 | 2705.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 2642.05 | 2674.94 | 2700.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:45:00 | 2638.20 | 2670.33 | 2695.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:15:00 | 2640.75 | 2666.33 | 2691.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:45:00 | 2638.75 | 2660.39 | 2686.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:45:00 | 2637.00 | 2648.64 | 2669.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2506.29 | 2578.85 | 2608.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2508.71 | 2578.85 | 2608.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2506.81 | 2578.85 | 2608.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2505.15 | 2578.85 | 2608.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 11:15:00 | 2512.25 | 2508.66 | 2542.90 | SL hit (close>ema200) qty=0.50 sl=2508.66 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 2597.45 | 2547.70 | 2545.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 2600.60 | 2574.62 | 2561.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 2630.50 | 2639.61 | 2615.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 2630.50 | 2639.61 | 2615.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 2620.60 | 2633.54 | 2622.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 10:15:00 | 2635.00 | 2633.54 | 2622.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-23 09:15:00 | 2898.50 | 2805.82 | 2759.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 3021.90 | 3045.17 | 3048.32 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 3060.00 | 3036.87 | 3036.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 3083.60 | 3046.21 | 3041.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 3060.20 | 3074.31 | 3060.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 3060.20 | 3074.31 | 3060.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 3060.20 | 3074.31 | 3060.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 3058.20 | 3074.31 | 3060.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 3042.00 | 3067.85 | 3059.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 3042.00 | 3067.85 | 3059.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 3051.80 | 3064.64 | 3058.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 3078.80 | 3063.31 | 3058.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 10:15:00 | 3098.50 | 3121.02 | 3121.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 3098.50 | 3121.02 | 3121.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 3092.80 | 3115.38 | 3118.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 3109.80 | 3093.53 | 3104.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 3109.80 | 3093.53 | 3104.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 3109.80 | 3093.53 | 3104.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 3109.80 | 3093.53 | 3104.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 3111.70 | 3097.16 | 3105.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 3114.00 | 3097.16 | 3105.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 3086.10 | 3094.95 | 3103.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 3082.20 | 3094.95 | 3103.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 3042.70 | 3091.64 | 3099.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:15:00 | 3076.50 | 3029.20 | 3039.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:00:00 | 3076.00 | 3044.97 | 3045.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 3073.20 | 3050.62 | 3047.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 3073.20 | 3050.62 | 3047.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 3084.00 | 3057.29 | 3051.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 3036.90 | 3059.16 | 3054.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 3036.90 | 3059.16 | 3054.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 3036.90 | 3059.16 | 3054.14 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 3025.90 | 3048.29 | 3050.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 3007.90 | 3038.22 | 3045.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 3010.00 | 3004.76 | 3016.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 15:00:00 | 3010.00 | 3004.76 | 3016.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 3013.60 | 3006.53 | 3015.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 2998.20 | 3006.53 | 3015.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 3028.10 | 3003.49 | 3002.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 3028.10 | 3003.49 | 3002.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 13:15:00 | 3031.60 | 3009.11 | 3005.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 12:15:00 | 3052.10 | 3053.74 | 3038.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:30:00 | 3054.70 | 3053.74 | 3038.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 3027.70 | 3053.33 | 3044.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 3013.00 | 3053.33 | 3044.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 3069.90 | 3056.64 | 3046.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 3071.90 | 3052.04 | 3047.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:30:00 | 3075.50 | 3059.56 | 3051.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 15:15:00 | 3075.20 | 3084.48 | 3081.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 3058.40 | 3081.02 | 3082.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 3058.40 | 3081.02 | 3082.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 3038.30 | 3072.48 | 3078.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 3032.50 | 3012.19 | 3027.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 3032.50 | 3012.19 | 3027.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 3032.50 | 3012.19 | 3027.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 3026.20 | 3012.19 | 3027.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 3044.70 | 3018.69 | 3029.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 3044.70 | 3018.69 | 3029.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 3023.00 | 3019.55 | 3028.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 3006.50 | 3022.48 | 3027.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 3013.40 | 3016.24 | 3023.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 3067.00 | 3021.64 | 3022.12 | SL hit (close>static) qty=1.00 sl=3045.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 3051.50 | 3027.61 | 3024.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 12:15:00 | 3079.70 | 3058.04 | 3044.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 3132.70 | 3153.66 | 3117.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:45:00 | 3135.00 | 3153.66 | 3117.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 3147.40 | 3171.67 | 3150.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 3147.40 | 3171.67 | 3150.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 3149.50 | 3167.24 | 3150.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:15:00 | 3156.20 | 3167.24 | 3150.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 3178.00 | 3191.76 | 3193.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 3178.00 | 3191.76 | 3193.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 3163.70 | 3178.04 | 3183.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3203.60 | 3179.50 | 3182.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 3203.60 | 3179.50 | 3182.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 3203.60 | 3179.50 | 3182.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 3203.60 | 3179.50 | 3182.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 3211.90 | 3185.98 | 3185.44 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 3175.60 | 3186.05 | 3186.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 09:15:00 | 3163.00 | 3181.44 | 3184.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 3165.90 | 3162.28 | 3172.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 3165.90 | 3162.28 | 3172.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 3163.30 | 3162.44 | 3170.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 3145.20 | 3162.47 | 3167.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 3183.00 | 3157.57 | 3158.96 | SL hit (close>static) qty=1.00 sl=3170.80 alert=retest2 |

### Cycle 75 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 3177.60 | 3161.57 | 3160.65 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 3143.80 | 3162.39 | 3163.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 3093.90 | 3148.21 | 3156.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 3096.00 | 3093.35 | 3111.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 14:00:00 | 3096.00 | 3093.35 | 3111.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3119.10 | 3098.11 | 3109.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 3119.10 | 3098.11 | 3109.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 3120.00 | 3102.49 | 3110.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:45:00 | 3118.10 | 3102.49 | 3110.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 3131.90 | 3108.37 | 3112.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 3144.00 | 3108.37 | 3112.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 3143.80 | 3115.46 | 3114.96 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 3100.00 | 3115.47 | 3116.09 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 3194.90 | 3130.95 | 3122.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 15:15:00 | 3216.00 | 3185.59 | 3163.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 11:15:00 | 3189.10 | 3190.55 | 3171.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 11:30:00 | 3185.20 | 3190.55 | 3171.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 3219.80 | 3196.38 | 3181.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 3235.20 | 3212.41 | 3194.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 3237.20 | 3227.14 | 3209.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 10:30:00 | 3235.00 | 3249.63 | 3249.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 3231.30 | 3245.96 | 3247.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 3231.30 | 3245.96 | 3247.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 3214.00 | 3237.70 | 3242.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 3211.20 | 3209.50 | 3220.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:45:00 | 3208.50 | 3209.50 | 3220.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 3209.30 | 3186.62 | 3200.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:45:00 | 3209.50 | 3186.62 | 3200.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 3224.90 | 3194.28 | 3202.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 3167.40 | 3194.28 | 3202.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 3197.90 | 3195.00 | 3202.20 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 3217.80 | 3206.45 | 3206.25 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 3197.50 | 3206.26 | 3206.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 3164.40 | 3197.88 | 3202.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 3205.70 | 3179.29 | 3187.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 3205.70 | 3179.29 | 3187.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 3205.70 | 3179.29 | 3187.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 3195.90 | 3179.29 | 3187.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 3213.50 | 3186.13 | 3189.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 3213.50 | 3186.13 | 3189.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 3202.90 | 3193.25 | 3192.57 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 3181.20 | 3192.48 | 3192.55 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 3202.50 | 3194.57 | 3193.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 3215.00 | 3200.33 | 3196.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 3192.30 | 3200.75 | 3197.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 3192.30 | 3200.75 | 3197.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 3192.30 | 3200.75 | 3197.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 3192.30 | 3200.75 | 3197.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 3185.00 | 3197.60 | 3196.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 3185.00 | 3197.60 | 3196.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 3198.40 | 3197.76 | 3196.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 3188.30 | 3197.76 | 3196.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 3223.50 | 3202.91 | 3199.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 3240.10 | 3209.35 | 3202.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 3188.00 | 3208.35 | 3204.40 | SL hit (close<static) qty=1.00 sl=3197.20 alert=retest2 |

### Cycle 86 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 3180.20 | 3200.59 | 3201.75 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 3211.00 | 3202.68 | 3202.59 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 3195.30 | 3201.73 | 3202.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 11:15:00 | 3179.10 | 3194.82 | 3198.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 13:15:00 | 3179.90 | 3166.19 | 3176.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 13:15:00 | 3179.90 | 3166.19 | 3176.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 3179.90 | 3166.19 | 3176.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:45:00 | 3179.80 | 3166.19 | 3176.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 3186.50 | 3170.25 | 3177.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 3186.50 | 3170.25 | 3177.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 3184.10 | 3173.02 | 3178.05 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 3239.20 | 3186.26 | 3183.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 3267.50 | 3232.52 | 3212.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 3263.50 | 3272.69 | 3256.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 15:00:00 | 3263.50 | 3272.69 | 3256.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 3375.80 | 3368.48 | 3349.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 3383.60 | 3368.48 | 3349.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:30:00 | 3387.30 | 3373.63 | 3358.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 3379.90 | 3385.45 | 3373.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 3377.50 | 3383.64 | 3373.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 3377.50 | 3382.41 | 3373.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 3390.40 | 3382.41 | 3373.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:45:00 | 3387.00 | 3397.37 | 3392.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:15:00 | 3382.00 | 3391.77 | 3390.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 3371.50 | 3387.71 | 3389.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 3371.50 | 3387.71 | 3389.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 3326.70 | 3371.83 | 3381.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 3238.00 | 3230.20 | 3272.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 3238.00 | 3230.20 | 3272.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 3292.20 | 3247.19 | 3273.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 3292.20 | 3247.19 | 3273.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 3315.60 | 3260.87 | 3277.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 3315.60 | 3260.87 | 3277.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 3312.90 | 3276.47 | 3281.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 3312.90 | 3276.47 | 3281.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 3290.20 | 3285.88 | 3285.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 3299.20 | 3288.54 | 3286.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 3283.20 | 3288.49 | 3287.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 3283.20 | 3288.49 | 3287.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3283.20 | 3288.49 | 3287.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 3283.20 | 3288.49 | 3287.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 3238.40 | 3278.47 | 3282.64 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 3477.50 | 3314.23 | 3293.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 10:15:00 | 3562.30 | 3480.41 | 3407.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 09:15:00 | 3667.70 | 3685.52 | 3636.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 3619.50 | 3669.03 | 3637.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 3619.50 | 3669.03 | 3637.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 3619.50 | 3669.03 | 3637.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 3604.00 | 3656.03 | 3634.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 3610.00 | 3656.03 | 3634.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 3594.60 | 3624.46 | 3624.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 3584.00 | 3616.37 | 3620.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 3606.60 | 3604.71 | 3611.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 3606.60 | 3604.71 | 3611.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 3606.60 | 3604.71 | 3611.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 3607.50 | 3604.71 | 3611.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 3600.30 | 3603.83 | 3610.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 3597.00 | 3603.83 | 3610.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 3597.10 | 3602.76 | 3609.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 3594.30 | 3602.76 | 3609.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 3613.60 | 3581.07 | 3578.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 3613.60 | 3581.07 | 3578.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 3620.00 | 3588.86 | 3582.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 3619.40 | 3623.54 | 3609.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:00:00 | 3619.40 | 3623.54 | 3609.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 3622.00 | 3623.23 | 3610.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 3611.00 | 3623.23 | 3610.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 3612.60 | 3621.10 | 3610.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 3612.60 | 3621.10 | 3610.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 3649.10 | 3626.70 | 3614.27 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 3587.00 | 3610.60 | 3613.64 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 3627.70 | 3609.22 | 3608.03 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 3592.00 | 3606.89 | 3608.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 3571.70 | 3596.44 | 3602.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 3450.00 | 3441.96 | 3478.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:45:00 | 3454.10 | 3441.96 | 3478.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 3432.70 | 3427.69 | 3451.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:45:00 | 3445.50 | 3427.69 | 3451.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 3475.60 | 3438.40 | 3450.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 3457.90 | 3438.40 | 3450.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 3464.20 | 3443.56 | 3451.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:30:00 | 3455.30 | 3448.76 | 3452.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 3468.00 | 3456.73 | 3455.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 3468.00 | 3456.73 | 3455.67 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 3414.80 | 3448.34 | 3451.95 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 3460.00 | 3452.49 | 3452.15 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 3448.30 | 3451.65 | 3451.80 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 3464.50 | 3454.22 | 3452.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 3476.60 | 3459.88 | 3455.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 3479.40 | 3485.97 | 3476.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 3479.40 | 3485.97 | 3476.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 3461.90 | 3481.16 | 3474.93 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 3447.30 | 3468.87 | 3470.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 3425.80 | 3454.41 | 3462.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 3446.70 | 3445.61 | 3456.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 3446.70 | 3445.61 | 3456.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 3440.90 | 3443.08 | 3452.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 3440.90 | 3443.08 | 3452.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 3452.90 | 3443.48 | 3449.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 3448.10 | 3443.48 | 3449.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 3444.00 | 3443.59 | 3448.89 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 3464.80 | 3454.43 | 3453.05 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 3437.80 | 3450.99 | 3452.53 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 3484.50 | 3457.44 | 3454.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 3495.30 | 3472.55 | 3462.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 10:15:00 | 3613.70 | 3622.18 | 3583.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:00:00 | 3613.70 | 3622.18 | 3583.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 3600.90 | 3620.09 | 3607.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 3600.90 | 3620.09 | 3607.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 3617.60 | 3619.59 | 3608.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 3650.10 | 3621.17 | 3610.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:30:00 | 3636.50 | 3628.20 | 3615.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 3598.10 | 3616.44 | 3616.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 3598.10 | 3616.44 | 3616.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 3590.30 | 3604.74 | 3610.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 3540.80 | 3508.89 | 3521.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 3540.80 | 3508.89 | 3521.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3540.80 | 3508.89 | 3521.49 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 3548.60 | 3528.99 | 3528.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 3554.00 | 3534.00 | 3530.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 3597.00 | 3610.27 | 3589.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 3597.00 | 3610.27 | 3589.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 3597.00 | 3610.27 | 3589.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 3597.30 | 3610.27 | 3589.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 3636.70 | 3615.55 | 3593.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:30:00 | 3647.90 | 3622.78 | 3598.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 3710.00 | 3713.96 | 3714.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 3710.00 | 3713.96 | 3714.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 3680.00 | 3700.88 | 3707.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 10:15:00 | 3721.10 | 3703.50 | 3706.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 10:15:00 | 3721.10 | 3703.50 | 3706.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 3721.10 | 3703.50 | 3706.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 3721.10 | 3703.50 | 3706.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 3743.20 | 3711.44 | 3709.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 3747.60 | 3723.54 | 3715.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 3696.80 | 3721.10 | 3717.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 3696.80 | 3721.10 | 3717.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 3696.80 | 3721.10 | 3717.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 3696.80 | 3721.10 | 3717.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 3701.70 | 3717.22 | 3715.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 3695.20 | 3717.22 | 3715.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 3715.00 | 3717.22 | 3715.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 3710.90 | 3717.22 | 3715.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 3717.70 | 3717.31 | 3716.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:45:00 | 3715.70 | 3717.31 | 3716.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 3693.30 | 3712.51 | 3714.01 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 3734.80 | 3716.63 | 3715.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 3751.30 | 3727.67 | 3721.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 3717.00 | 3730.21 | 3725.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 3717.00 | 3730.21 | 3725.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 3717.00 | 3730.21 | 3725.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 3717.00 | 3730.21 | 3725.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 3711.80 | 3726.53 | 3724.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 3764.00 | 3726.53 | 3724.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 10:15:00 | 3719.50 | 3741.46 | 3737.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 3702.80 | 3729.32 | 3732.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 3702.80 | 3729.32 | 3732.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 12:15:00 | 3696.50 | 3722.76 | 3729.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 3692.90 | 3689.74 | 3704.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 3692.90 | 3689.74 | 3704.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 3702.00 | 3684.57 | 3694.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 3702.00 | 3684.57 | 3694.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 3697.70 | 3687.19 | 3694.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:30:00 | 3705.00 | 3687.19 | 3694.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 3688.20 | 3687.39 | 3693.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:45:00 | 3675.50 | 3684.65 | 3690.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 3670.90 | 3679.68 | 3687.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:00:00 | 3676.00 | 3678.94 | 3686.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:30:00 | 3677.40 | 3679.28 | 3685.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 3746.00 | 3692.58 | 3690.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 3746.00 | 3692.58 | 3690.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 3785.00 | 3746.71 | 3723.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 3746.70 | 3746.71 | 3725.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:45:00 | 3738.40 | 3746.71 | 3725.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 3733.00 | 3745.43 | 3733.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 3734.30 | 3745.43 | 3733.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 3723.40 | 3741.03 | 3732.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 3723.40 | 3741.03 | 3732.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 3720.00 | 3736.82 | 3731.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:30:00 | 3727.00 | 3730.48 | 3729.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 3728.90 | 3730.04 | 3729.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 14:15:00 | 3716.40 | 3727.31 | 3728.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 3716.40 | 3727.31 | 3728.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 3691.10 | 3719.04 | 3724.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 3677.00 | 3667.91 | 3687.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 11:00:00 | 3677.00 | 3667.91 | 3687.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 3694.90 | 3672.36 | 3680.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 3694.90 | 3672.36 | 3680.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 3704.80 | 3678.85 | 3682.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 3704.80 | 3678.85 | 3682.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 3704.10 | 3683.90 | 3684.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 3703.40 | 3683.90 | 3684.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 3710.90 | 3689.30 | 3687.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 3714.10 | 3698.05 | 3691.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 3692.90 | 3698.66 | 3693.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 3692.90 | 3698.66 | 3693.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 3692.90 | 3698.66 | 3693.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 3692.90 | 3698.66 | 3693.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 3692.70 | 3697.47 | 3693.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 3684.00 | 3697.47 | 3693.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 3689.90 | 3695.96 | 3692.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:45:00 | 3687.50 | 3695.96 | 3692.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 3696.10 | 3695.98 | 3693.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:15:00 | 3689.00 | 3695.98 | 3693.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 3682.40 | 3693.27 | 3692.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 3682.40 | 3693.27 | 3692.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 3683.60 | 3691.33 | 3691.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 3617.00 | 3674.49 | 3683.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 3687.00 | 3656.35 | 3666.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 3687.00 | 3656.35 | 3666.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 3687.00 | 3656.35 | 3666.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 3687.00 | 3656.35 | 3666.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 3680.40 | 3661.16 | 3667.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 3688.80 | 3661.16 | 3667.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 3635.20 | 3659.01 | 3665.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 3628.30 | 3659.01 | 3665.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 3631.40 | 3647.31 | 3657.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 3679.60 | 3657.60 | 3657.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 3679.60 | 3657.60 | 3657.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 10:15:00 | 3694.00 | 3664.88 | 3660.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 3633.90 | 3666.56 | 3664.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 3633.90 | 3666.56 | 3664.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 3633.90 | 3666.56 | 3664.63 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 3624.60 | 3658.17 | 3660.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 3610.00 | 3639.14 | 3650.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 11:15:00 | 3630.50 | 3625.35 | 3637.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 12:00:00 | 3630.50 | 3625.35 | 3637.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3616.80 | 3623.63 | 3632.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 3612.80 | 3623.63 | 3632.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 3612.50 | 3611.96 | 3622.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:45:00 | 3610.60 | 3595.31 | 3598.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 3610.30 | 3600.78 | 3600.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 3610.30 | 3600.78 | 3600.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 3620.70 | 3610.24 | 3606.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 3610.00 | 3610.20 | 3606.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 10:15:00 | 3610.00 | 3610.20 | 3606.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 3610.00 | 3610.20 | 3606.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 3609.10 | 3610.20 | 3606.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 3620.10 | 3612.18 | 3607.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 3611.70 | 3612.18 | 3607.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 3636.00 | 3623.43 | 3615.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 3640.20 | 3626.02 | 3617.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 3641.00 | 3629.02 | 3619.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 3593.00 | 3621.02 | 3623.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 3593.00 | 3621.02 | 3623.60 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 11:15:00 | 3636.90 | 3623.03 | 3622.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 3647.80 | 3629.13 | 3625.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 3793.40 | 3803.53 | 3775.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 3793.40 | 3803.53 | 3775.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 3774.00 | 3793.89 | 3777.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:15:00 | 3760.50 | 3793.89 | 3777.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 3768.60 | 3788.83 | 3777.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 3760.00 | 3788.83 | 3777.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 3783.20 | 3779.83 | 3775.63 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 3753.30 | 3773.88 | 3774.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 12:15:00 | 3732.50 | 3765.60 | 3770.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 3740.20 | 3732.53 | 3744.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 3740.20 | 3732.53 | 3744.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3740.20 | 3732.53 | 3744.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 3740.20 | 3732.53 | 3744.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 3715.30 | 3729.09 | 3742.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 3706.40 | 3723.79 | 3738.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 10:15:00 | 3715.80 | 3671.99 | 3669.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 3715.80 | 3671.99 | 3669.55 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 3660.00 | 3670.77 | 3670.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 3643.20 | 3665.26 | 3668.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 13:15:00 | 3658.40 | 3654.51 | 3661.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 13:15:00 | 3658.40 | 3654.51 | 3661.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 3658.40 | 3654.51 | 3661.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:45:00 | 3663.70 | 3654.51 | 3661.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 3660.80 | 3655.77 | 3661.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:30:00 | 3666.90 | 3655.77 | 3661.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 3662.90 | 3657.19 | 3661.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 3648.30 | 3657.19 | 3661.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 3648.10 | 3655.37 | 3660.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:00:00 | 3626.30 | 3649.56 | 3657.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 3632.60 | 3646.17 | 3654.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 3444.99 | 3532.10 | 3556.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 3450.97 | 3532.10 | 3556.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 3441.60 | 3430.22 | 3476.21 | SL hit (close>ema200) qty=0.50 sl=3430.22 alert=retest2 |

### Cycle 127 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 3476.00 | 3421.27 | 3416.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 3497.00 | 3436.42 | 3423.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 3437.50 | 3443.19 | 3429.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 3437.50 | 3443.19 | 3429.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 3437.50 | 3443.19 | 3429.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 3424.80 | 3443.19 | 3429.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 3429.00 | 3440.35 | 3429.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 3429.00 | 3440.35 | 3429.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 3366.50 | 3425.58 | 3423.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 3368.10 | 3425.58 | 3423.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 3362.00 | 3412.87 | 3418.19 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 3460.50 | 3417.70 | 3415.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 3543.80 | 3449.03 | 3430.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 3539.00 | 3560.65 | 3529.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 3539.00 | 3560.65 | 3529.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3539.00 | 3560.65 | 3529.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 3539.00 | 3560.65 | 3529.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 3550.40 | 3565.78 | 3548.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 3550.40 | 3565.78 | 3548.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 3572.00 | 3567.02 | 3550.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 3551.90 | 3567.02 | 3550.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 3545.10 | 3562.83 | 3551.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 3539.40 | 3562.83 | 3551.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 3564.80 | 3563.22 | 3552.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 3575.00 | 3566.30 | 3554.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 12:15:00 | 3607.70 | 3652.06 | 3652.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 3607.70 | 3652.06 | 3652.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 3603.10 | 3642.27 | 3648.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 3489.80 | 3488.80 | 3519.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 3489.80 | 3488.80 | 3519.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 3528.90 | 3499.24 | 3509.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 3528.90 | 3499.24 | 3509.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 3543.00 | 3507.99 | 3512.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 3526.00 | 3507.99 | 3512.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 3450.00 | 3429.81 | 3450.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 3450.50 | 3429.81 | 3450.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 3446.00 | 3433.05 | 3449.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 3426.00 | 3438.33 | 3446.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 3463.50 | 3436.28 | 3438.58 | SL hit (close>static) qty=1.00 sl=3457.30 alert=retest2 |

### Cycle 131 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 3486.90 | 3446.40 | 3442.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 3491.80 | 3455.48 | 3447.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 3470.60 | 3477.10 | 3464.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:00:00 | 3470.60 | 3477.10 | 3464.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 3481.90 | 3478.06 | 3466.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:30:00 | 3487.00 | 3480.19 | 3469.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 3485.40 | 3480.19 | 3469.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 3446.00 | 3473.32 | 3468.03 | SL hit (close<static) qty=1.00 sl=3461.60 alert=retest2 |

### Cycle 132 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 3445.00 | 3463.96 | 3464.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 3392.60 | 3444.32 | 3454.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 3293.90 | 3281.60 | 3322.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 3293.90 | 3281.60 | 3322.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3293.90 | 3281.60 | 3322.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 3315.20 | 3281.60 | 3322.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 3316.00 | 3292.82 | 3317.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:30:00 | 3310.10 | 3292.82 | 3317.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 3308.20 | 3295.89 | 3316.82 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 3349.70 | 3329.11 | 3327.45 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 3209.30 | 3311.12 | 3321.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 3175.70 | 3284.03 | 3308.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 3258.70 | 3219.26 | 3256.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 3258.70 | 3219.26 | 3256.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3258.70 | 3219.26 | 3256.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 3206.30 | 3248.84 | 3257.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:15:00 | 3045.99 | 3150.37 | 3198.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 2965.30 | 2963.40 | 3025.17 | SL hit (close>ema200) qty=0.50 sl=2963.40 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 3100.10 | 3050.02 | 3043.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 3117.00 | 3063.42 | 3049.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3124.30 | 3176.08 | 3138.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3124.30 | 3176.08 | 3138.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3124.30 | 3176.08 | 3138.01 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 3044.60 | 3109.98 | 3117.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 2968.00 | 3055.04 | 3080.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 2993.50 | 2982.77 | 3022.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 2993.50 | 2982.77 | 3022.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 2993.50 | 2982.77 | 3022.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 2981.00 | 2981.53 | 3018.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 11:45:00 | 2985.40 | 2978.53 | 3013.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 3046.70 | 3000.93 | 3018.39 | SL hit (close>static) qty=1.00 sl=3035.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 3115.40 | 3033.95 | 3029.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 3151.40 | 3057.44 | 3040.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 3059.00 | 3099.74 | 3075.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 3059.00 | 3099.74 | 3075.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 3059.00 | 3099.74 | 3075.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 3059.00 | 3099.74 | 3075.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3065.90 | 3092.97 | 3074.83 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 3037.30 | 3065.58 | 3065.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 3009.40 | 3050.89 | 3058.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3047.90 | 3001.86 | 3022.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3047.90 | 3001.86 | 3022.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3047.90 | 3001.86 | 3022.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 3031.50 | 3001.86 | 3022.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 3033.70 | 3008.23 | 3023.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 3030.50 | 3031.17 | 3031.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 3221.60 | 3041.69 | 3019.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 3221.60 | 3041.69 | 3019.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 10:15:00 | 3250.60 | 3190.39 | 3150.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 3192.10 | 3228.70 | 3192.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 3192.10 | 3228.70 | 3192.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3192.10 | 3228.70 | 3192.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 3201.90 | 3228.70 | 3192.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 3224.50 | 3227.86 | 3195.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 3228.40 | 3227.86 | 3195.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 3243.70 | 3226.85 | 3207.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:15:00 | 3228.00 | 3236.13 | 3233.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 15:15:00 | 3220.00 | 3230.39 | 3231.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 15:15:00 | 3220.00 | 3230.39 | 3231.18 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 3239.80 | 3232.27 | 3231.96 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 10:15:00 | 3228.60 | 3231.54 | 3231.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 11:15:00 | 3212.00 | 3227.63 | 3229.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 11:15:00 | 3214.70 | 3206.98 | 3215.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 11:15:00 | 3214.70 | 3206.98 | 3215.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 11:15:00 | 3214.70 | 3206.98 | 3215.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:00:00 | 3214.70 | 3206.98 | 3215.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 3212.30 | 3208.04 | 3215.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:45:00 | 3221.90 | 3208.04 | 3215.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 3228.80 | 3212.19 | 3216.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:00:00 | 3228.80 | 3212.19 | 3216.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 3224.80 | 3214.71 | 3217.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:30:00 | 3226.80 | 3214.71 | 3217.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 3242.40 | 3219.90 | 3219.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 3257.50 | 3227.42 | 3222.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 3206.10 | 3233.99 | 3229.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 3206.10 | 3233.99 | 3229.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 3206.10 | 3233.99 | 3229.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 3206.10 | 3233.99 | 3229.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 3201.10 | 3227.41 | 3227.18 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 11:15:00 | 3200.90 | 3222.11 | 3224.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 12:15:00 | 3191.80 | 3216.05 | 3221.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 3086.10 | 3084.51 | 3129.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 10:00:00 | 3086.10 | 3084.51 | 3129.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 3084.40 | 3061.10 | 3092.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 3083.60 | 3061.10 | 3092.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 3083.30 | 3065.54 | 3091.82 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 3117.50 | 3102.60 | 3100.68 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 3082.50 | 3098.58 | 3099.02 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 3175.40 | 3109.99 | 3103.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 3183.80 | 3124.75 | 3110.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 3060.00 | 3130.45 | 3123.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 3060.00 | 3130.45 | 3123.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 3060.00 | 3130.45 | 3123.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 3060.00 | 3130.45 | 3123.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 3050.40 | 3114.44 | 3116.94 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 3174.20 | 3114.94 | 3111.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 3213.90 | 3140.02 | 3123.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 3339.00 | 3348.17 | 3305.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 3339.00 | 3348.17 | 3305.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-27 11:45:00 | 2569.70 | 2024-05-27 14:15:00 | 2550.05 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-05-27 14:15:00 | 2565.00 | 2024-05-27 14:15:00 | 2550.05 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-05-28 13:00:00 | 2578.70 | 2024-05-28 14:15:00 | 2550.25 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-05-31 13:00:00 | 2503.85 | 2024-06-03 09:15:00 | 2635.60 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2024-05-31 15:00:00 | 2503.80 | 2024-06-03 09:15:00 | 2635.60 | STOP_HIT | 1.00 | -5.26% |
| BUY | retest2 | 2024-06-07 11:00:00 | 2734.85 | 2024-06-18 09:15:00 | 3008.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-28 09:15:00 | 2870.20 | 2024-07-01 13:15:00 | 2881.75 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-06-28 14:30:00 | 2881.05 | 2024-07-01 13:15:00 | 2881.75 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-07-25 11:45:00 | 2805.60 | 2024-08-01 10:15:00 | 2839.05 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2024-08-06 12:45:00 | 2681.95 | 2024-08-09 09:15:00 | 2740.70 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-08-07 09:45:00 | 2685.00 | 2024-08-09 09:15:00 | 2740.70 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-08-08 13:00:00 | 2680.30 | 2024-08-09 09:15:00 | 2740.70 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-08-08 13:45:00 | 2681.65 | 2024-08-09 09:15:00 | 2740.70 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-08-12 11:15:00 | 2734.00 | 2024-08-20 10:15:00 | 2753.80 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2024-08-13 10:00:00 | 2730.60 | 2024-08-20 10:15:00 | 2753.80 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2024-08-13 10:45:00 | 2730.80 | 2024-08-20 10:15:00 | 2753.80 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2024-08-13 11:15:00 | 2734.95 | 2024-08-20 10:15:00 | 2753.80 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2024-08-14 09:15:00 | 2748.40 | 2024-08-20 10:15:00 | 2753.80 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-09-10 09:15:00 | 2690.40 | 2024-09-12 13:15:00 | 2724.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-09-10 13:30:00 | 2704.75 | 2024-09-12 13:15:00 | 2724.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-09-20 09:15:00 | 2833.00 | 2024-09-25 09:15:00 | 3116.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-16 09:15:00 | 3103.10 | 2024-10-18 09:15:00 | 2947.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 09:15:00 | 3103.10 | 2024-10-21 13:15:00 | 2988.60 | STOP_HIT | 0.50 | 3.69% |
| BUY | retest2 | 2024-11-07 12:15:00 | 2935.00 | 2024-11-12 12:15:00 | 2904.95 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-11-07 12:45:00 | 2930.00 | 2024-11-12 12:15:00 | 2904.95 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-11-08 10:45:00 | 2921.20 | 2024-11-12 12:15:00 | 2904.95 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-11-08 11:15:00 | 2918.00 | 2024-11-12 12:15:00 | 2904.95 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-11-11 14:30:00 | 2936.00 | 2024-11-12 13:15:00 | 2902.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-11-11 15:00:00 | 2935.00 | 2024-11-12 13:15:00 | 2902.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-11-12 09:30:00 | 2936.00 | 2024-11-12 13:15:00 | 2902.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-11-12 10:15:00 | 2938.75 | 2024-11-12 13:15:00 | 2902.90 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest1 | 2024-11-22 09:15:00 | 2938.75 | 2024-11-25 09:15:00 | 3085.69 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-11-22 09:15:00 | 2938.75 | 2024-11-26 09:15:00 | 3002.00 | STOP_HIT | 0.50 | 2.15% |
| BUY | retest2 | 2025-01-01 10:30:00 | 3026.00 | 2025-01-06 14:15:00 | 3103.40 | STOP_HIT | 1.00 | 2.56% |
| SELL | retest2 | 2025-01-08 12:00:00 | 3100.80 | 2025-01-09 12:15:00 | 3151.60 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-01-09 09:30:00 | 3096.00 | 2025-01-09 12:15:00 | 3151.60 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-01-09 11:45:00 | 3100.95 | 2025-01-09 12:15:00 | 3151.60 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-01-14 14:15:00 | 3052.35 | 2025-01-20 09:15:00 | 2899.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-14 14:15:00 | 3052.35 | 2025-01-23 09:15:00 | 2891.95 | STOP_HIT | 0.50 | 5.25% |
| SELL | retest2 | 2025-01-28 09:15:00 | 2796.55 | 2025-01-28 10:15:00 | 2854.50 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-01-28 10:45:00 | 2824.75 | 2025-01-28 11:15:00 | 2894.95 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-02-07 09:15:00 | 3177.20 | 2025-02-07 13:15:00 | 3124.35 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-02-07 09:45:00 | 3174.85 | 2025-02-07 13:15:00 | 3124.35 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-02-07 10:45:00 | 3187.25 | 2025-02-07 13:15:00 | 3124.35 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-02-07 13:15:00 | 3200.00 | 2025-02-07 13:15:00 | 3124.35 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-02-10 09:30:00 | 3209.20 | 2025-02-10 15:15:00 | 3130.75 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest1 | 2025-02-13 13:30:00 | 2989.50 | 2025-02-17 09:15:00 | 2840.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 13:30:00 | 2989.50 | 2025-02-20 09:15:00 | 2690.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-03 11:45:00 | 2597.00 | 2025-03-05 09:15:00 | 2704.55 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2025-03-03 13:45:00 | 2610.90 | 2025-03-05 09:15:00 | 2704.55 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-03-03 15:00:00 | 2611.75 | 2025-03-05 09:15:00 | 2704.55 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-03-04 09:15:00 | 2593.65 | 2025-03-05 09:15:00 | 2704.55 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-03-10 11:15:00 | 2727.30 | 2025-03-11 09:15:00 | 2648.75 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-04-01 11:45:00 | 2638.20 | 2025-04-07 09:15:00 | 2506.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 13:15:00 | 2640.75 | 2025-04-07 09:15:00 | 2508.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 13:45:00 | 2638.75 | 2025-04-07 09:15:00 | 2506.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-02 11:45:00 | 2637.00 | 2025-04-07 09:15:00 | 2505.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:45:00 | 2638.20 | 2025-04-08 11:15:00 | 2512.25 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2025-04-01 13:15:00 | 2640.75 | 2025-04-08 11:15:00 | 2512.25 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2025-04-01 13:45:00 | 2638.75 | 2025-04-08 11:15:00 | 2512.25 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2025-04-02 11:45:00 | 2637.00 | 2025-04-08 11:15:00 | 2512.25 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2025-04-09 14:30:00 | 2522.75 | 2025-04-11 09:15:00 | 2597.45 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-04-17 10:15:00 | 2635.00 | 2025-04-23 09:15:00 | 2898.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-14 09:15:00 | 3078.80 | 2025-05-20 10:15:00 | 3098.50 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-05-21 12:15:00 | 3082.20 | 2025-05-26 12:15:00 | 3073.20 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-05-22 09:15:00 | 3042.70 | 2025-05-26 12:15:00 | 3073.20 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-05-26 10:15:00 | 3076.50 | 2025-05-26 12:15:00 | 3073.20 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-05-26 12:00:00 | 3076.00 | 2025-05-26 12:15:00 | 3073.20 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-05-30 09:15:00 | 2998.20 | 2025-06-02 12:15:00 | 3028.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-06-06 10:30:00 | 3071.90 | 2025-06-12 09:15:00 | 3058.40 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-06 11:30:00 | 3075.50 | 2025-06-12 09:15:00 | 3058.40 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-06-10 15:15:00 | 3075.20 | 2025-06-12 09:15:00 | 3058.40 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-06-17 09:15:00 | 3006.50 | 2025-06-18 09:15:00 | 3067.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-06-17 11:45:00 | 3013.40 | 2025-06-18 09:15:00 | 3067.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-06-24 15:15:00 | 3156.20 | 2025-06-30 13:15:00 | 3178.00 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-07-08 09:15:00 | 3145.20 | 2025-07-09 09:15:00 | 3183.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-07-21 14:15:00 | 3235.20 | 2025-07-25 11:15:00 | 3231.30 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-07-22 12:00:00 | 3237.20 | 2025-07-25 11:15:00 | 3231.30 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-07-25 10:30:00 | 3235.00 | 2025-07-25 11:15:00 | 3231.30 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-08-06 15:15:00 | 3240.10 | 2025-08-07 10:15:00 | 3188.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-08-20 10:15:00 | 3383.60 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-08-20 13:30:00 | 3387.30 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-08-21 13:30:00 | 3379.90 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-08-21 15:15:00 | 3377.50 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-08-22 09:15:00 | 3390.40 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-08-26 09:45:00 | 3387.00 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-08-26 12:15:00 | 3382.00 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-09-12 11:15:00 | 3597.00 | 2025-09-16 15:15:00 | 3613.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-09-12 11:45:00 | 3597.10 | 2025-09-16 15:15:00 | 3613.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-09-12 12:15:00 | 3594.30 | 2025-09-16 15:15:00 | 3613.60 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-10-01 12:30:00 | 3455.30 | 2025-10-01 15:15:00 | 3468.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-10-24 09:15:00 | 3650.10 | 2025-10-27 12:15:00 | 3598.10 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-10-24 10:30:00 | 3636.50 | 2025-10-27 12:15:00 | 3598.10 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-11-07 11:30:00 | 3647.90 | 2025-11-14 10:15:00 | 3710.00 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2025-11-21 09:15:00 | 3764.00 | 2025-11-24 11:15:00 | 3702.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-24 10:15:00 | 3719.50 | 2025-11-24 11:15:00 | 3702.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-27 10:45:00 | 3675.50 | 2025-11-28 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-11-27 13:00:00 | 3670.90 | 2025-11-28 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-11-27 14:00:00 | 3676.00 | 2025-11-28 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-11-27 14:30:00 | 3677.40 | 2025-11-28 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-02 12:30:00 | 3727.00 | 2025-12-02 14:15:00 | 3716.40 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-12-02 13:45:00 | 3728.90 | 2025-12-02 14:15:00 | 3716.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-12-10 14:15:00 | 3628.30 | 2025-12-12 09:15:00 | 3679.60 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-11 10:15:00 | 3631.40 | 2025-12-12 09:15:00 | 3679.60 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-17 10:15:00 | 3612.80 | 2025-12-22 09:15:00 | 3610.30 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-12-17 15:15:00 | 3612.50 | 2025-12-22 09:15:00 | 3610.30 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-12-19 13:45:00 | 3610.60 | 2025-12-22 09:15:00 | 3610.30 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-12-24 11:15:00 | 3640.20 | 2025-12-29 13:15:00 | 3593.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-24 12:00:00 | 3641.00 | 2025-12-29 13:15:00 | 3593.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-09 11:30:00 | 3706.40 | 2026-01-16 10:15:00 | 3715.80 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-01-20 11:00:00 | 3626.30 | 2026-01-27 09:15:00 | 3444.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:00:00 | 3632.60 | 2026-01-27 09:15:00 | 3450.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 11:00:00 | 3626.30 | 2026-01-28 10:15:00 | 3441.60 | STOP_HIT | 0.50 | 5.09% |
| SELL | retest2 | 2026-01-20 12:00:00 | 3632.60 | 2026-01-28 10:15:00 | 3441.60 | STOP_HIT | 0.50 | 5.26% |
| BUY | retest2 | 2026-02-06 14:30:00 | 3575.00 | 2026-02-12 12:15:00 | 3607.70 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2026-02-24 09:15:00 | 3426.00 | 2026-02-25 09:15:00 | 3463.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-26 14:30:00 | 3487.00 | 2026-02-27 09:15:00 | 3446.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-02-26 15:00:00 | 3485.40 | 2026-02-27 09:15:00 | 3446.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-03-11 10:30:00 | 3206.30 | 2026-03-12 10:15:00 | 3045.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:30:00 | 3206.30 | 2026-03-16 11:15:00 | 2965.30 | STOP_HIT | 0.50 | 7.52% |
| SELL | retest2 | 2026-03-24 10:30:00 | 2981.00 | 2026-03-24 13:15:00 | 3046.70 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-03-24 11:45:00 | 2985.40 | 2026-03-24 13:15:00 | 3046.70 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-01 10:15:00 | 3031.50 | 2026-04-08 09:15:00 | 3221.60 | STOP_HIT | 1.00 | -6.27% |
| SELL | retest2 | 2026-04-01 11:00:00 | 3033.70 | 2026-04-08 09:15:00 | 3221.60 | STOP_HIT | 1.00 | -6.19% |
| SELL | retest2 | 2026-04-01 14:30:00 | 3030.50 | 2026-04-08 09:15:00 | 3221.60 | STOP_HIT | 1.00 | -6.31% |
| BUY | retest2 | 2026-04-13 11:15:00 | 3228.40 | 2026-04-16 15:15:00 | 3220.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2026-04-15 09:15:00 | 3243.70 | 2026-04-16 15:15:00 | 3220.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-04-16 14:15:00 | 3228.00 | 2026-04-16 15:15:00 | 3220.00 | STOP_HIT | 1.00 | -0.25% |
