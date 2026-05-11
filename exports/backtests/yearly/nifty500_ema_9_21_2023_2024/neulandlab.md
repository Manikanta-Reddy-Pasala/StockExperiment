# Neuland Laboratories Ltd. (NEULANDLAB)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 17713.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 220 |
| ALERT1 | 150 |
| ALERT2 | 149 |
| ALERT2_SKIP | 103 |
| ALERT3 | 291 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 115 |
| PARTIAL | 15 |
| TARGET_HIT | 19 |
| STOP_HIT | 97 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 131 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 51 / 80
- **Target hits / Stop hits / Partials:** 19 / 97 / 15
- **Avg / median % per leg:** 0.70% / -1.40%
- **Sum % (uncompounded):** 91.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 21 | 39.6% | 14 | 39 | 0 | 1.00% | 53.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 53 | 21 | 39.6% | 14 | 39 | 0 | 1.00% | 53.0% |
| SELL (all) | 78 | 30 | 38.5% | 5 | 58 | 15 | 0.49% | 38.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -7.88% | -7.9% |
| SELL @ 3rd Alert (retest2) | 77 | 30 | 39.0% | 5 | 57 | 15 | 0.60% | 46.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -7.88% | -7.9% |
| retest2 (combined) | 130 | 51 | 39.2% | 19 | 96 | 15 | 0.76% | 99.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 10:15:00 | 2801.75 | 2831.94 | 2834.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 11:15:00 | 2798.00 | 2825.15 | 2831.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 09:15:00 | 2812.55 | 2762.15 | 2780.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 2812.55 | 2762.15 | 2780.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 2812.55 | 2762.15 | 2780.51 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 12:15:00 | 2835.85 | 2795.82 | 2793.04 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 09:15:00 | 2698.25 | 2789.90 | 2799.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 13:15:00 | 2665.45 | 2729.28 | 2764.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-30 12:15:00 | 2749.30 | 2697.87 | 2726.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 12:15:00 | 2749.30 | 2697.87 | 2726.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 2749.30 | 2697.87 | 2726.64 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 11:15:00 | 2739.55 | 2737.54 | 2737.39 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 12:15:00 | 2724.00 | 2734.83 | 2736.17 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 14:15:00 | 2744.30 | 2737.36 | 2737.13 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 15:15:00 | 2731.00 | 2736.09 | 2736.57 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 2782.25 | 2745.32 | 2740.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 2837.35 | 2768.10 | 2756.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 13:15:00 | 2812.00 | 2817.63 | 2787.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 15:15:00 | 2800.00 | 2812.43 | 2790.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 15:15:00 | 2800.00 | 2812.43 | 2790.31 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 15:15:00 | 2870.45 | 2894.89 | 2895.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 10:15:00 | 2846.25 | 2881.84 | 2889.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 10:15:00 | 2841.00 | 2839.19 | 2859.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 11:15:00 | 2920.00 | 2855.35 | 2865.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 2920.00 | 2855.35 | 2865.23 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 13:15:00 | 2960.55 | 2882.41 | 2876.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 15:15:00 | 2980.00 | 2914.47 | 2892.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 11:15:00 | 3015.75 | 3028.41 | 2982.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 09:15:00 | 2975.60 | 3020.75 | 2997.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 2975.60 | 3020.75 | 2997.00 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 11:15:00 | 2932.10 | 2980.49 | 2985.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-16 14:15:00 | 2912.00 | 2958.65 | 2973.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-19 14:15:00 | 2942.40 | 2927.04 | 2946.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 14:15:00 | 2942.40 | 2927.04 | 2946.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 14:15:00 | 2942.40 | 2927.04 | 2946.71 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 11:15:00 | 2970.10 | 2943.38 | 2942.70 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 14:15:00 | 2891.60 | 2932.80 | 2938.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 15:15:00 | 2836.00 | 2913.44 | 2928.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 11:15:00 | 2815.00 | 2812.66 | 2851.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 14:15:00 | 2870.05 | 2821.48 | 2829.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 2870.05 | 2821.48 | 2829.43 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 2874.60 | 2838.64 | 2836.26 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 10:15:00 | 2847.25 | 2862.33 | 2864.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 12:15:00 | 2828.00 | 2851.81 | 2858.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 11:15:00 | 2915.00 | 2849.20 | 2852.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 11:15:00 | 2915.00 | 2849.20 | 2852.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 11:15:00 | 2915.00 | 2849.20 | 2852.53 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 12:15:00 | 2945.00 | 2868.36 | 2860.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-04 15:15:00 | 2977.00 | 2911.15 | 2883.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 14:15:00 | 3112.80 | 3116.75 | 3076.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 15:15:00 | 3093.00 | 3112.00 | 3077.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 3093.00 | 3112.00 | 3077.54 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 13:15:00 | 3012.00 | 3064.86 | 3065.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 14:15:00 | 2920.00 | 3035.89 | 3052.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 11:15:00 | 2994.55 | 2991.90 | 3022.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 09:15:00 | 2929.65 | 2961.62 | 2983.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 2929.65 | 2961.62 | 2983.51 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 12:15:00 | 2997.90 | 2967.14 | 2966.70 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 09:15:00 | 2917.00 | 2959.19 | 2963.82 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 09:15:00 | 3229.70 | 3009.84 | 2985.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 11:15:00 | 3379.80 | 3252.27 | 3198.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 13:15:00 | 3323.75 | 3328.20 | 3279.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 14:15:00 | 3228.80 | 3308.32 | 3274.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 14:15:00 | 3228.80 | 3308.32 | 3274.63 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 14:15:00 | 3256.00 | 3264.33 | 3264.44 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 15:15:00 | 3284.00 | 3268.26 | 3266.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 09:15:00 | 3318.00 | 3278.21 | 3270.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 14:15:00 | 3292.75 | 3313.65 | 3294.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 14:15:00 | 3292.75 | 3313.65 | 3294.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 3292.75 | 3313.65 | 3294.63 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 14:15:00 | 3443.95 | 3455.96 | 3456.39 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 09:15:00 | 3526.00 | 3469.66 | 3462.52 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 13:15:00 | 3400.00 | 3459.11 | 3460.61 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 14:15:00 | 3540.00 | 3475.29 | 3467.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 11:15:00 | 3716.10 | 3563.06 | 3512.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 3780.00 | 3788.73 | 3714.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 11:15:00 | 3743.75 | 3779.73 | 3716.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 3743.75 | 3779.73 | 3716.89 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 10:15:00 | 3985.05 | 4114.36 | 4118.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 3877.55 | 4051.09 | 4088.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 3925.60 | 3905.92 | 3987.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 09:15:00 | 3944.35 | 3925.50 | 3964.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 3944.35 | 3925.50 | 3964.02 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 14:15:00 | 3971.95 | 3930.62 | 3930.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 09:15:00 | 4104.25 | 3970.04 | 3948.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 15:15:00 | 4005.00 | 4032.68 | 3997.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 10:15:00 | 4025.00 | 4027.51 | 4000.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 4025.00 | 4027.51 | 4000.93 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 15:15:00 | 3870.80 | 3988.44 | 3993.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 11:15:00 | 3866.50 | 3933.78 | 3964.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 3870.00 | 3842.16 | 3876.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 09:15:00 | 3870.00 | 3842.16 | 3876.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 3870.00 | 3842.16 | 3876.89 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 11:15:00 | 3937.50 | 3866.97 | 3864.71 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 15:15:00 | 3771.00 | 3865.57 | 3867.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 09:15:00 | 3691.00 | 3777.58 | 3813.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 09:15:00 | 3746.55 | 3699.02 | 3745.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 09:15:00 | 3746.55 | 3699.02 | 3745.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 3746.55 | 3699.02 | 3745.78 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 13:15:00 | 3617.15 | 3584.22 | 3582.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 14:15:00 | 3684.75 | 3604.32 | 3591.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 10:15:00 | 3630.40 | 3630.52 | 3608.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 12:15:00 | 3638.00 | 3631.03 | 3612.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 12:15:00 | 3638.00 | 3631.03 | 3612.87 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-09-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 12:15:00 | 3566.40 | 3637.86 | 3640.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 3530.05 | 3596.36 | 3618.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 3627.90 | 3587.53 | 3607.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 12:15:00 | 3627.90 | 3587.53 | 3607.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 3627.90 | 3587.53 | 3607.34 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 09:15:00 | 3719.95 | 3635.38 | 3625.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 11:15:00 | 3747.00 | 3671.79 | 3644.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 13:15:00 | 3791.00 | 3794.31 | 3740.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 12:15:00 | 3767.10 | 3775.64 | 3752.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 12:15:00 | 3767.10 | 3775.64 | 3752.88 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 09:15:00 | 3685.00 | 3763.26 | 3764.05 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 09:15:00 | 3849.10 | 3755.18 | 3745.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 09:15:00 | 3940.00 | 3843.45 | 3803.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 09:15:00 | 3893.00 | 3910.48 | 3866.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 3915.35 | 3911.11 | 3886.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 3915.35 | 3911.11 | 3886.98 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 14:15:00 | 3855.55 | 3905.93 | 3909.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-11 12:15:00 | 3800.25 | 3879.78 | 3895.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 3869.90 | 3860.25 | 3879.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 3869.90 | 3860.25 | 3879.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 3869.90 | 3860.25 | 3879.92 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 09:15:00 | 4051.70 | 3883.61 | 3869.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 09:15:00 | 4114.95 | 4005.80 | 3947.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 3984.20 | 4001.48 | 3951.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 13:15:00 | 3998.45 | 3996.23 | 3960.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 13:15:00 | 3998.45 | 3996.23 | 3960.90 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 3950.00 | 4056.32 | 4057.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 10:15:00 | 3874.00 | 4019.85 | 4040.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 14:15:00 | 3800.00 | 3798.54 | 3881.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 11:15:00 | 3846.00 | 3765.45 | 3793.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 3846.00 | 3765.45 | 3793.47 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 10:15:00 | 3846.05 | 3804.26 | 3802.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 10:15:00 | 3872.70 | 3843.12 | 3826.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 09:15:00 | 3822.15 | 3858.13 | 3844.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 3822.15 | 3858.13 | 3844.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 3822.15 | 3858.13 | 3844.63 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 09:15:00 | 5300.00 | 5329.27 | 5333.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 10:15:00 | 5184.55 | 5300.33 | 5319.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 5278.90 | 5253.95 | 5281.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 5278.90 | 5253.95 | 5281.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 5278.90 | 5253.95 | 5281.07 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 5195.70 | 5186.37 | 5186.24 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 09:15:00 | 5141.60 | 5180.75 | 5184.57 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 11:15:00 | 5301.30 | 5207.23 | 5196.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 5494.45 | 5295.43 | 5241.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 15:15:00 | 5354.00 | 5377.42 | 5327.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 15:15:00 | 5354.00 | 5377.42 | 5327.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 5354.00 | 5377.42 | 5327.27 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 09:15:00 | 5185.00 | 5321.58 | 5328.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 13:15:00 | 5131.10 | 5223.13 | 5274.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 10:15:00 | 5096.40 | 5095.10 | 5149.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 09:15:00 | 5210.00 | 5113.71 | 5134.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 5210.00 | 5113.71 | 5134.01 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 12:15:00 | 5230.00 | 5162.21 | 5153.38 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 09:15:00 | 5071.75 | 5151.69 | 5153.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 13:15:00 | 5055.00 | 5104.18 | 5127.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 09:15:00 | 5113.30 | 5069.15 | 5087.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 5113.30 | 5069.15 | 5087.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 5113.30 | 5069.15 | 5087.67 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 12:15:00 | 5160.00 | 5103.80 | 5100.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 13:15:00 | 5228.10 | 5128.66 | 5112.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 12:15:00 | 5152.05 | 5179.00 | 5151.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 12:15:00 | 5152.05 | 5179.00 | 5151.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 12:15:00 | 5152.05 | 5179.00 | 5151.13 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 10:15:00 | 5090.05 | 5142.46 | 5142.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 11:15:00 | 5073.00 | 5128.57 | 5136.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 14:15:00 | 5160.05 | 5129.92 | 5134.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 14:15:00 | 5160.05 | 5129.92 | 5134.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 5160.05 | 5129.92 | 5134.54 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 14:15:00 | 5200.00 | 5136.09 | 5132.22 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 4989.55 | 5136.26 | 5151.83 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 5129.75 | 5116.91 | 5115.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 5200.10 | 5136.05 | 5126.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 14:15:00 | 5286.85 | 5308.26 | 5250.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 09:15:00 | 5243.95 | 5291.05 | 5252.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 5243.95 | 5291.05 | 5252.82 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 15:15:00 | 5250.00 | 5284.98 | 5286.57 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 11:15:00 | 5383.95 | 5303.63 | 5294.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 09:15:00 | 5582.55 | 5439.99 | 5372.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 09:15:00 | 5632.00 | 5634.09 | 5568.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 10:15:00 | 5551.90 | 5617.65 | 5566.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 5551.90 | 5617.65 | 5566.73 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 5327.60 | 5533.00 | 5545.49 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 12:15:00 | 5515.45 | 5462.86 | 5458.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 14:15:00 | 5530.75 | 5482.40 | 5468.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 13:15:00 | 5571.40 | 5661.80 | 5606.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 13:15:00 | 5571.40 | 5661.80 | 5606.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 13:15:00 | 5571.40 | 5661.80 | 5606.80 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 10:15:00 | 5553.60 | 5615.12 | 5616.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 5519.65 | 5596.02 | 5607.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 5641.90 | 5560.55 | 5576.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 11:15:00 | 5641.90 | 5560.55 | 5576.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 5641.90 | 5560.55 | 5576.26 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-01-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 13:15:00 | 5659.50 | 5592.69 | 5588.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 14:15:00 | 5671.05 | 5608.36 | 5596.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 5798.00 | 5834.78 | 5750.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 5782.05 | 5852.86 | 5806.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 5782.05 | 5852.86 | 5806.50 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 5692.85 | 5766.48 | 5773.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 09:15:00 | 5544.55 | 5677.14 | 5725.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 5670.00 | 5663.80 | 5706.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 09:15:00 | 5741.65 | 5662.43 | 5690.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 5741.65 | 5662.43 | 5690.26 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 11:15:00 | 5821.00 | 5717.76 | 5712.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 12:15:00 | 5933.25 | 5760.85 | 5732.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 11:15:00 | 6374.50 | 6391.64 | 6205.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 13:15:00 | 6231.20 | 6349.21 | 6217.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 13:15:00 | 6231.20 | 6349.21 | 6217.48 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 15:15:00 | 6222.00 | 6288.71 | 6290.02 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 6317.90 | 6294.55 | 6292.55 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 12:15:00 | 6214.55 | 6280.73 | 6287.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 13:15:00 | 6174.95 | 6259.58 | 6276.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 11:15:00 | 6226.40 | 6208.71 | 6239.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 15:15:00 | 6181.60 | 6203.56 | 6227.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 6181.60 | 6203.56 | 6227.56 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 12:15:00 | 6329.90 | 6246.56 | 6241.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 13:15:00 | 6392.90 | 6275.82 | 6255.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 09:15:00 | 6501.20 | 6508.11 | 6417.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 6575.00 | 6711.47 | 6627.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 6575.00 | 6711.47 | 6627.72 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 12:15:00 | 6386.95 | 6550.92 | 6566.70 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 13:15:00 | 6587.30 | 6516.44 | 6508.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 6701.35 | 6553.42 | 6526.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 10:15:00 | 7025.50 | 7066.75 | 6939.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 7025.10 | 7059.69 | 6991.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 7025.10 | 7059.69 | 6991.99 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 10:15:00 | 6957.30 | 7105.88 | 7114.44 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 14:15:00 | 7109.75 | 7023.59 | 7018.90 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 10:15:00 | 6899.80 | 7002.06 | 7011.07 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 7149.90 | 7028.92 | 7018.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 15:15:00 | 7200.00 | 7063.14 | 7035.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-02 11:15:00 | 7200.10 | 7209.88 | 7148.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 7069.05 | 7184.61 | 7148.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 7069.05 | 7184.61 | 7148.18 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 15:15:00 | 7084.75 | 7126.29 | 7129.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 6992.40 | 7099.51 | 7117.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 11:15:00 | 7139.30 | 7099.55 | 7113.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 11:15:00 | 7139.30 | 7099.55 | 7113.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 7139.30 | 7099.55 | 7113.77 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 15:15:00 | 6127.00 | 5870.50 | 5870.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 12:15:00 | 6284.55 | 6045.23 | 5960.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 12:15:00 | 6165.85 | 6172.33 | 6084.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 13:15:00 | 6112.80 | 6160.43 | 6086.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 13:15:00 | 6112.80 | 6160.43 | 6086.86 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 11:15:00 | 5947.15 | 6044.54 | 6053.83 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 13:15:00 | 6068.05 | 6043.86 | 6042.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 6136.00 | 6063.93 | 6052.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 6359.15 | 6401.69 | 6312.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 09:15:00 | 6359.15 | 6401.69 | 6312.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 6359.15 | 6401.69 | 6312.51 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-01 12:15:00 | 6262.85 | 6326.45 | 6329.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-01 14:15:00 | 6255.65 | 6308.87 | 6321.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 6398.25 | 6318.92 | 6323.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 09:15:00 | 6398.25 | 6318.92 | 6323.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 6398.25 | 6318.92 | 6323.12 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 10:15:00 | 6523.05 | 6359.75 | 6341.30 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-04-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 11:15:00 | 6357.40 | 6431.03 | 6435.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 09:15:00 | 6255.30 | 6364.89 | 6398.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 10:15:00 | 6339.90 | 6186.52 | 6257.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 10:15:00 | 6339.90 | 6186.52 | 6257.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 10:15:00 | 6339.90 | 6186.52 | 6257.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 6567.25 | 6199.68 | 6225.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 09:15:00 | 6556.00 | 6270.94 | 6255.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 10:15:00 | 6673.00 | 6351.36 | 6293.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 7398.65 | 7536.33 | 7345.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 7398.65 | 7536.33 | 7345.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 7398.65 | 7536.33 | 7345.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 12:00:00 | 7604.95 | 7464.01 | 7391.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 13:00:00 | 7670.20 | 7505.25 | 7416.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 10:00:00 | 7644.15 | 7582.90 | 7485.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 10:15:00 | 7310.00 | 7477.14 | 7483.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 10:15:00 | 7310.00 | 7477.14 | 7483.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 13:15:00 | 7252.00 | 7388.64 | 7438.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 10:15:00 | 7100.00 | 7033.02 | 7114.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 11:00:00 | 7100.00 | 7033.02 | 7114.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 7154.50 | 7057.32 | 7118.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 11:45:00 | 7152.70 | 7057.32 | 7118.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 12:15:00 | 7254.00 | 7096.65 | 7130.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 13:00:00 | 7254.00 | 7096.65 | 7130.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 7333.70 | 7185.43 | 7165.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 10:15:00 | 7424.00 | 7233.14 | 7189.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 09:15:00 | 7540.25 | 7574.43 | 7469.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 09:15:00 | 7403.40 | 7531.32 | 7498.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 7403.40 | 7531.32 | 7498.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:45:00 | 7350.00 | 7531.32 | 7498.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 7391.15 | 7503.28 | 7488.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:30:00 | 7400.00 | 7503.28 | 7488.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 12:15:00 | 7381.85 | 7460.39 | 7470.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 7290.00 | 7417.83 | 7445.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 12:15:00 | 7425.00 | 7379.06 | 7417.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 12:15:00 | 7425.00 | 7379.06 | 7417.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 7425.00 | 7379.06 | 7417.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 13:00:00 | 7425.00 | 7379.06 | 7417.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 7400.80 | 7383.41 | 7415.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 14:30:00 | 7360.20 | 7383.78 | 7413.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 15:15:00 | 7312.50 | 7383.78 | 7413.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-08 09:15:00 | 7463.00 | 7388.22 | 7409.40 | SL hit (close>static) qty=1.00 sl=7428.40 alert=retest2 |

### Cycle 82 — BUY (started 2024-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 13:15:00 | 7460.00 | 7424.25 | 7421.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 14:15:00 | 7473.35 | 7434.07 | 7426.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 10:15:00 | 7448.60 | 7449.28 | 7436.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-09 11:00:00 | 7448.60 | 7449.28 | 7436.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 7446.00 | 7451.71 | 7439.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 13:00:00 | 7446.00 | 7451.71 | 7439.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 13:15:00 | 7235.00 | 7408.37 | 7421.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 7158.90 | 7358.48 | 7397.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 12:15:00 | 6229.50 | 6217.62 | 6394.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:00:00 | 6229.50 | 6217.62 | 6394.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 6241.30 | 6225.67 | 6285.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 10:15:00 | 6219.20 | 6225.67 | 6285.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-18 09:30:00 | 6206.40 | 6231.84 | 6260.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 6221.75 | 6232.16 | 6256.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 11:15:00 | 6280.15 | 6243.71 | 6242.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 11:15:00 | 6280.15 | 6243.71 | 6242.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 14:15:00 | 6290.75 | 6263.57 | 6253.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 6210.60 | 6260.41 | 6253.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 6210.60 | 6260.41 | 6253.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 6210.60 | 6260.41 | 6253.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:45:00 | 6209.85 | 6260.41 | 6253.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 6200.00 | 6248.33 | 6249.01 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 6364.95 | 6249.64 | 6245.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 10:15:00 | 6427.85 | 6285.28 | 6261.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 15:15:00 | 6325.00 | 6325.90 | 6294.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 15:15:00 | 6325.00 | 6325.90 | 6294.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 6325.00 | 6325.90 | 6294.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 6399.70 | 6325.90 | 6294.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 6225.10 | 6324.70 | 6318.99 | SL hit (close<static) qty=1.00 sl=6290.90 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 10:15:00 | 6216.90 | 6303.14 | 6309.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 6196.10 | 6281.73 | 6299.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 13:15:00 | 6202.40 | 6102.36 | 6142.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 13:15:00 | 6202.40 | 6102.36 | 6142.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 6202.40 | 6102.36 | 6142.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:00:00 | 6202.40 | 6102.36 | 6142.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 6210.05 | 6123.90 | 6148.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 15:15:00 | 6185.25 | 6123.90 | 6148.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 5875.99 | 6022.81 | 6066.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 5566.73 | 5940.06 | 6018.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 88 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 6257.00 | 6049.90 | 6044.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 6446.35 | 6238.31 | 6152.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 13:15:00 | 6272.55 | 6279.53 | 6203.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 14:15:00 | 6273.35 | 6279.53 | 6203.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 6177.05 | 6259.04 | 6201.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 15:00:00 | 6177.05 | 6259.04 | 6201.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 6145.00 | 6236.23 | 6196.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 6280.00 | 6236.23 | 6196.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 13:15:00 | 6450.00 | 6464.67 | 6465.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 13:15:00 | 6450.00 | 6464.67 | 6465.49 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 6499.00 | 6471.54 | 6468.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 11:15:00 | 6528.20 | 6498.06 | 6486.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 09:15:00 | 6500.00 | 6526.84 | 6508.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 6500.00 | 6526.84 | 6508.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 6500.00 | 6526.84 | 6508.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 6520.20 | 6526.84 | 6508.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 6500.00 | 6521.47 | 6507.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:00:00 | 6500.00 | 6521.47 | 6507.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 6530.60 | 6523.30 | 6509.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 6560.95 | 6523.18 | 6512.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-25 14:15:00 | 7217.05 | 6990.74 | 6818.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 7678.70 | 7716.59 | 7716.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 15:15:00 | 7676.00 | 7708.05 | 7712.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 13:15:00 | 7734.55 | 7698.80 | 7704.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 7734.55 | 7698.80 | 7704.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 7734.55 | 7698.80 | 7704.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:45:00 | 7748.75 | 7698.80 | 7704.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 14:15:00 | 7840.55 | 7727.15 | 7717.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 11:15:00 | 8015.80 | 7821.30 | 7767.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 14:15:00 | 8482.85 | 8487.86 | 8400.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 14:30:00 | 8485.00 | 8487.86 | 8400.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 8050.00 | 8399.35 | 8375.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 8050.00 | 8399.35 | 8375.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 8078.60 | 8335.20 | 8348.17 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 8281.40 | 8173.30 | 8173.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 12:15:00 | 8327.00 | 8204.04 | 8187.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 8370.05 | 8374.55 | 8307.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 14:30:00 | 8370.00 | 8374.55 | 8307.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 8364.60 | 8366.63 | 8320.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:30:00 | 8379.70 | 8372.95 | 8327.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 13:45:00 | 8385.00 | 8365.37 | 8350.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-01 15:15:00 | 9217.67 | 9064.19 | 8903.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 15:15:00 | 11500.00 | 11576.49 | 11586.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 15:15:00 | 11386.00 | 11516.77 | 11552.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 10:15:00 | 11499.30 | 11481.79 | 11528.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-21 11:00:00 | 11499.30 | 11481.79 | 11528.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 11429.90 | 11471.41 | 11519.62 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 11949.00 | 11586.21 | 11555.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 10:15:00 | 12040.00 | 11676.97 | 11599.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 12:15:00 | 12060.05 | 12169.39 | 12034.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 12:15:00 | 12060.05 | 12169.39 | 12034.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 12060.05 | 12169.39 | 12034.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 12060.05 | 12169.39 | 12034.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 11910.40 | 12099.11 | 12024.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:45:00 | 11883.30 | 12099.11 | 12024.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 11898.35 | 12058.95 | 12013.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 12184.00 | 12058.95 | 12013.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:30:00 | 12005.90 | 12024.73 | 12004.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 12:00:00 | 11950.30 | 12009.84 | 11999.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-06 09:15:00 | 13206.49 | 12822.81 | 12748.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 12650.05 | 12777.46 | 12784.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 14:15:00 | 12573.15 | 12705.77 | 12747.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 09:15:00 | 12718.90 | 12568.26 | 12627.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 12718.90 | 12568.26 | 12627.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 12718.90 | 12568.26 | 12627.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:45:00 | 12463.90 | 12553.47 | 12610.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 10:15:00 | 12742.80 | 12629.59 | 12625.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 10:15:00 | 12742.80 | 12629.59 | 12625.94 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 11:15:00 | 12573.70 | 12652.40 | 12654.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 14:15:00 | 12485.95 | 12592.34 | 12623.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 09:15:00 | 12640.85 | 12593.67 | 12618.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 12640.85 | 12593.67 | 12618.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 12640.85 | 12593.67 | 12618.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:45:00 | 12629.40 | 12593.67 | 12618.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 12658.85 | 12606.71 | 12622.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:30:00 | 12658.00 | 12606.71 | 12622.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 11:15:00 | 12807.80 | 12646.92 | 12638.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 13:15:00 | 12855.00 | 12712.23 | 12671.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 15:15:00 | 12817.80 | 12866.32 | 12796.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 09:15:00 | 12783.95 | 12866.32 | 12796.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 12702.45 | 12833.54 | 12788.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 12702.45 | 12833.54 | 12788.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 12716.20 | 12810.07 | 12781.56 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 12725.75 | 12763.18 | 12764.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 12601.95 | 12713.76 | 12740.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 12716.05 | 12657.18 | 12689.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 12716.05 | 12657.18 | 12689.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 12716.05 | 12657.18 | 12689.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 12716.05 | 12657.18 | 12689.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 12701.05 | 12665.95 | 12690.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:30:00 | 12724.05 | 12665.95 | 12690.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 12720.00 | 12676.76 | 12692.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 12723.00 | 12676.76 | 12692.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 12783.15 | 12705.52 | 12702.66 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 15:15:00 | 12499.80 | 12666.96 | 12688.17 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 13:15:00 | 13044.00 | 12684.20 | 12674.82 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 12509.85 | 12700.61 | 12713.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 11875.90 | 12499.56 | 12617.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 13:15:00 | 12200.10 | 12177.86 | 12398.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 14:00:00 | 12200.10 | 12177.86 | 12398.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 12800.00 | 12310.61 | 12420.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 13024.65 | 12310.61 | 12420.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 13061.50 | 12460.79 | 12479.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 13218.35 | 12460.79 | 12479.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 13044.45 | 12577.52 | 12530.60 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 12379.15 | 12549.26 | 12572.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 12239.55 | 12448.92 | 12505.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 14:15:00 | 12381.85 | 12377.23 | 12447.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 14:45:00 | 12399.00 | 12377.23 | 12447.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 12649.80 | 12431.74 | 12465.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 10:30:00 | 12350.00 | 12406.00 | 12448.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 11:30:00 | 12352.55 | 12229.29 | 12302.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:00:00 | 12351.00 | 12229.29 | 12302.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:30:00 | 12341.80 | 12253.73 | 12306.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 12154.80 | 12234.95 | 12289.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:30:00 | 12277.55 | 12234.95 | 12289.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 11928.80 | 12149.89 | 12239.52 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 11732.50 | 12057.15 | 12189.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 11734.92 | 12057.15 | 12189.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 11733.45 | 12057.15 | 12189.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 11724.71 | 12057.15 | 12189.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:45:00 | 11619.40 | 12057.15 | 12189.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:30:00 | 11700.55 | 11961.72 | 12133.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-07 12:15:00 | 11117.30 | 11818.24 | 12052.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 108 — BUY (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 10:15:00 | 12500.00 | 11945.23 | 11874.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 09:15:00 | 13733.00 | 12595.74 | 12255.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 15531.00 | 15776.65 | 15192.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 14934.40 | 15431.13 | 15286.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 14934.40 | 15431.13 | 15286.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:30:00 | 14845.80 | 15431.13 | 15286.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 14916.95 | 15328.30 | 15252.48 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 14509.25 | 15091.12 | 15153.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 14:15:00 | 14046.05 | 14778.04 | 14993.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 09:15:00 | 13990.30 | 13790.49 | 14193.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 13990.30 | 13790.49 | 14193.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 13990.30 | 13790.49 | 14193.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:45:00 | 13992.05 | 13790.49 | 14193.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 13910.00 | 13664.92 | 13886.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 13910.00 | 13664.92 | 13886.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 13920.00 | 13715.94 | 13889.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:30:00 | 13848.30 | 13715.94 | 13889.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 14149.00 | 13802.55 | 13913.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:30:00 | 14140.30 | 13802.55 | 13913.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 14232.55 | 13888.55 | 13942.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:45:00 | 14423.85 | 13888.55 | 13942.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 13900.00 | 13879.10 | 13927.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:15:00 | 14285.20 | 13879.10 | 13927.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 14102.40 | 13923.76 | 13943.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:45:00 | 13790.00 | 13947.37 | 13951.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:15:00 | 13937.35 | 13947.37 | 13951.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:45:00 | 13810.15 | 13918.10 | 13937.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 14:15:00 | 13900.20 | 13842.64 | 13836.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 13900.20 | 13842.64 | 13836.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 15:15:00 | 14100.00 | 13894.11 | 13860.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 13810.35 | 13877.36 | 13856.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 09:15:00 | 13810.35 | 13877.36 | 13856.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 13810.35 | 13877.36 | 13856.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 13810.35 | 13877.36 | 13856.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 13850.05 | 13871.90 | 13855.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:45:00 | 13941.70 | 13871.90 | 13855.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 13872.35 | 13871.99 | 13857.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 09:45:00 | 14085.00 | 13914.43 | 13883.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 10:15:00 | 14078.55 | 13914.43 | 13883.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 14:15:00 | 13806.30 | 13926.66 | 13908.14 | SL hit (close<static) qty=1.00 sl=13826.70 alert=retest2 |

### Cycle 111 — SELL (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 13:15:00 | 13770.40 | 14968.33 | 14971.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 12950.05 | 13661.56 | 13974.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 11:15:00 | 13825.30 | 13615.82 | 13894.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 11:15:00 | 13825.30 | 13615.82 | 13894.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 13825.30 | 13615.82 | 13894.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:00:00 | 13825.30 | 13615.82 | 13894.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 13816.60 | 13656.81 | 13864.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:00:00 | 13816.60 | 13656.81 | 13864.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 13796.90 | 13684.83 | 13858.66 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 15820.70 | 14131.46 | 14032.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 12:15:00 | 15900.00 | 15312.06 | 15017.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 09:15:00 | 15184.70 | 15405.40 | 15171.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 15184.70 | 15405.40 | 15171.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 15184.70 | 15405.40 | 15171.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:30:00 | 15000.00 | 15405.40 | 15171.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 15104.10 | 15345.14 | 15165.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:30:00 | 15139.10 | 15345.14 | 15165.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 15041.60 | 15284.43 | 15153.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 15041.60 | 15284.43 | 15153.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 15030.00 | 15233.55 | 15142.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:45:00 | 15035.00 | 15233.55 | 15142.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 15023.20 | 15191.48 | 15131.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:30:00 | 15025.85 | 15191.48 | 15131.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 15127.50 | 15113.15 | 15105.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:15:00 | 15085.60 | 15113.15 | 15105.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 11:15:00 | 15040.95 | 15098.71 | 15099.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 14:15:00 | 14825.05 | 15030.37 | 15066.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 09:15:00 | 15151.80 | 15034.59 | 15061.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 15151.80 | 15034.59 | 15061.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 15151.80 | 15034.59 | 15061.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 15151.80 | 15034.59 | 15061.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 14891.70 | 15006.02 | 15045.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 10:00:00 | 14720.00 | 14818.83 | 14872.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 10:30:00 | 14721.00 | 14795.55 | 14857.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 14:15:00 | 14924.85 | 14895.81 | 14892.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 14:15:00 | 14924.85 | 14895.81 | 14892.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 12:15:00 | 15801.00 | 15149.62 | 15020.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 11:15:00 | 16628.55 | 16698.78 | 16384.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 12:00:00 | 16628.55 | 16698.78 | 16384.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 16795.25 | 16703.86 | 16503.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 11:00:00 | 16910.05 | 16745.10 | 16540.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 15977.25 | 17323.56 | 17438.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 15977.25 | 17323.56 | 17438.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 14:15:00 | 15376.45 | 15718.38 | 16129.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 13:15:00 | 14901.00 | 14886.14 | 15212.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-17 14:00:00 | 14901.00 | 14886.14 | 15212.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 14857.15 | 14841.15 | 15107.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:45:00 | 14988.50 | 14841.15 | 15107.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 14958.05 | 14864.53 | 15094.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:45:00 | 14958.10 | 14864.53 | 15094.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 14009.20 | 13917.39 | 14082.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:15:00 | 13877.65 | 13917.39 | 14082.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 12:15:00 | 13910.00 | 13769.91 | 13753.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 12:15:00 | 13910.00 | 13769.91 | 13753.66 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 13:15:00 | 13738.20 | 13757.50 | 13757.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 14:15:00 | 13723.15 | 13750.63 | 13754.50 | Break + close below crossover candle low |

### Cycle 118 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 13850.00 | 13766.41 | 13760.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 14139.05 | 13865.87 | 13809.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 14245.30 | 14256.84 | 14123.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 09:30:00 | 14251.90 | 14256.84 | 14123.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 14181.00 | 14241.67 | 14128.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:45:00 | 14126.70 | 14241.67 | 14128.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 14029.90 | 14199.32 | 14119.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:00:00 | 14029.90 | 14199.32 | 14119.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 14000.80 | 14159.61 | 14108.63 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 13772.40 | 14042.23 | 14061.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 13720.00 | 13836.86 | 13934.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 13808.05 | 13768.97 | 13872.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 13808.05 | 13768.97 | 13872.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 13808.05 | 13768.97 | 13872.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 13864.20 | 13768.97 | 13872.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 14055.00 | 13838.32 | 13887.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:00:00 | 14055.00 | 13838.32 | 13887.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 14000.00 | 13870.66 | 13897.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:15:00 | 14200.00 | 13870.66 | 13897.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 14250.00 | 13946.53 | 13929.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 09:15:00 | 14748.00 | 14202.22 | 14058.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 14:15:00 | 14617.95 | 14758.89 | 14570.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 15:00:00 | 14617.95 | 14758.89 | 14570.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 14406.00 | 14658.25 | 14555.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:45:00 | 14270.40 | 14658.25 | 14555.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 14403.05 | 14607.21 | 14541.79 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 14276.15 | 14498.87 | 14501.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 14115.80 | 14422.25 | 14466.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 13449.35 | 13446.98 | 13809.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 09:30:00 | 13505.00 | 13446.98 | 13809.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 13600.00 | 13429.45 | 13508.70 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 13:15:00 | 13818.25 | 13570.50 | 13554.56 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 13463.00 | 13551.84 | 13553.35 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 12:15:00 | 13700.10 | 13567.86 | 13559.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 13:15:00 | 13750.30 | 13604.35 | 13576.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 15:15:00 | 13600.00 | 13611.25 | 13585.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 15:15:00 | 13600.00 | 13611.25 | 13585.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 13600.00 | 13611.25 | 13585.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 13674.50 | 13611.25 | 13585.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 13641.90 | 13617.38 | 13590.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:15:00 | 13780.45 | 13638.65 | 13602.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:30:00 | 13783.60 | 13692.80 | 13634.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:30:00 | 13863.85 | 13735.39 | 13665.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 13842.85 | 13738.54 | 13672.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 13605.70 | 13722.52 | 13677.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 13605.70 | 13722.52 | 13677.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 13630.00 | 13704.02 | 13673.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:15:00 | 13612.00 | 13704.02 | 13673.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-21 12:15:00 | 13345.75 | 13632.37 | 13643.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 13345.75 | 13632.37 | 13643.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 13198.65 | 13507.93 | 13582.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 09:15:00 | 13650.85 | 13480.05 | 13553.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 09:15:00 | 13650.85 | 13480.05 | 13553.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 13650.85 | 13480.05 | 13553.59 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 12:15:00 | 13698.20 | 13613.18 | 13604.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 13:15:00 | 13829.55 | 13656.45 | 13625.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 12:15:00 | 13802.10 | 13831.58 | 13744.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 12:15:00 | 13802.10 | 13831.58 | 13744.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 13802.10 | 13831.58 | 13744.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:45:00 | 13823.25 | 13831.58 | 13744.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 13619.00 | 13789.06 | 13733.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:30:00 | 13815.00 | 13789.06 | 13733.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 13855.65 | 13802.38 | 13744.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 14104.85 | 13821.89 | 13758.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 12:00:00 | 14000.05 | 13891.32 | 13810.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 13:15:00 | 13494.00 | 13757.66 | 13788.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 13:15:00 | 13494.00 | 13757.66 | 13788.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 13452.00 | 13696.53 | 13757.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 13441.90 | 13309.38 | 13447.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 13441.90 | 13309.38 | 13447.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 13441.90 | 13309.38 | 13447.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 13441.90 | 13309.38 | 13447.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 13535.00 | 13354.51 | 13455.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 13599.95 | 13354.51 | 13455.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 13678.95 | 13419.40 | 13475.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 13678.95 | 13419.40 | 13475.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 13900.00 | 13579.73 | 13542.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 14082.50 | 13774.33 | 13649.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 13720.00 | 13832.83 | 13725.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 13720.00 | 13832.83 | 13725.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 13720.00 | 13832.83 | 13725.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 13720.00 | 13832.83 | 13725.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 13711.65 | 13808.59 | 13724.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 13564.60 | 13808.59 | 13724.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 13811.20 | 13809.12 | 13731.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:30:00 | 14016.60 | 13833.56 | 13750.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:00:00 | 13907.50 | 13848.35 | 13764.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 13:15:00 | 13999.90 | 13849.89 | 13780.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:00:00 | 14032.50 | 14004.74 | 13906.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 13940.00 | 14074.55 | 13987.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:45:00 | 13844.25 | 14074.55 | 13987.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 13916.40 | 14042.92 | 13980.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:30:00 | 13866.05 | 14042.92 | 13980.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 14088.35 | 14049.05 | 13994.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:45:00 | 14029.80 | 14049.05 | 13994.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 13967.50 | 14032.74 | 13991.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:45:00 | 13983.95 | 14032.74 | 13991.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 14178.05 | 14061.80 | 14008.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:30:00 | 14609.95 | 14182.63 | 14074.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 14162.50 | 14525.41 | 14526.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 14162.50 | 14525.41 | 14526.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 13900.05 | 14251.42 | 14373.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 13:15:00 | 14207.60 | 14188.12 | 14299.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 14:00:00 | 14207.60 | 14188.12 | 14299.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 14484.95 | 14247.48 | 14316.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 14484.95 | 14247.48 | 14316.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 14800.00 | 14357.99 | 14360.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 13853.00 | 14357.99 | 14360.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 13160.35 | 13615.24 | 13903.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 12:15:00 | 13712.85 | 13526.90 | 13781.70 | SL hit (close>ema200) qty=0.50 sl=13526.90 alert=retest2 |

### Cycle 130 — BUY (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 11:15:00 | 12164.95 | 11390.42 | 11318.60 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 11040.70 | 11403.50 | 11431.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 10935.00 | 11309.80 | 11386.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 10680.00 | 10615.02 | 10861.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 09:15:00 | 10332.60 | 10615.02 | 10861.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 11149.85 | 10617.75 | 10702.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 11149.85 | 10617.75 | 10702.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 11360.00 | 10766.20 | 10761.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 11410.15 | 11048.05 | 10905.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 13:15:00 | 11775.00 | 11783.58 | 11543.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 14:00:00 | 11775.00 | 11783.58 | 11543.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 12001.00 | 11985.36 | 11816.35 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 11370.00 | 11718.27 | 11744.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 11120.75 | 11544.64 | 11657.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 12:15:00 | 10900.00 | 10897.40 | 11067.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 13:30:00 | 10752.55 | 10897.12 | 11051.67 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 11599.80 | 11054.07 | 11097.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-13 15:15:00 | 11599.80 | 11054.07 | 11097.20 | SL hit (close>ema400) qty=1.00 sl=11097.20 alert=retest1 |

### Cycle 134 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 11219.00 | 11080.60 | 11065.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 11313.60 | 11155.90 | 11103.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 13:15:00 | 11943.95 | 11949.32 | 11805.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 13:45:00 | 11916.70 | 11949.32 | 11805.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 11800.00 | 11910.18 | 11822.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:30:00 | 11801.10 | 11910.18 | 11822.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 11800.05 | 11888.16 | 11820.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:15:00 | 11700.20 | 11888.16 | 11820.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 11680.00 | 11846.52 | 11807.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 11733.50 | 11846.52 | 11807.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 11710.10 | 11819.24 | 11798.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:45:00 | 11761.90 | 11804.83 | 11794.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:15:00 | 11782.20 | 11804.83 | 11794.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 11426.15 | 11742.09 | 11772.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 11426.15 | 11742.09 | 11772.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 11340.00 | 11661.67 | 11733.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 11587.20 | 11424.33 | 11529.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 11587.20 | 11424.33 | 11529.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 11587.20 | 11424.33 | 11529.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 11587.20 | 11424.33 | 11529.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 11471.95 | 11433.85 | 11524.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:45:00 | 11605.00 | 11433.85 | 11524.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 12081.65 | 11563.41 | 11575.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 12081.65 | 11563.41 | 11575.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 12037.85 | 11658.30 | 11617.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 12534.25 | 11833.49 | 11700.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 14:15:00 | 12105.20 | 12128.29 | 11928.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 15:00:00 | 12105.20 | 12128.29 | 11928.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 11999.95 | 12102.62 | 11934.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 11750.00 | 12019.69 | 11912.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 11512.00 | 11918.16 | 11875.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 11512.00 | 11918.16 | 11875.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 11459.30 | 11826.38 | 11838.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 13:15:00 | 11340.00 | 11494.05 | 11619.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 11580.15 | 11511.27 | 11615.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 14:15:00 | 11580.15 | 11511.27 | 11615.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 11580.15 | 11511.27 | 11615.54 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 11857.50 | 11660.17 | 11658.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 12:15:00 | 12197.65 | 11767.67 | 11707.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 11459.85 | 11816.40 | 11764.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 11459.85 | 11816.40 | 11764.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 11459.85 | 11816.40 | 11764.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 11459.85 | 11816.40 | 11764.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 11664.90 | 11786.10 | 11755.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 11:15:00 | 11800.00 | 11786.10 | 11755.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 11498.00 | 11696.62 | 11718.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 11498.00 | 11696.62 | 11718.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 11452.85 | 11647.87 | 11693.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 11223.85 | 11054.67 | 11249.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 11223.85 | 11054.67 | 11249.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 11223.85 | 11054.67 | 11249.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 10953.20 | 11155.09 | 11228.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:45:00 | 11070.05 | 11097.27 | 11179.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 11070.70 | 11097.27 | 11179.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:45:00 | 10916.35 | 11049.30 | 11150.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 10915.00 | 10982.48 | 11089.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:15:00 | 11350.00 | 10982.48 | 11089.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 11260.45 | 11038.07 | 11105.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 11610.00 | 11182.72 | 11161.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 11610.00 | 11182.72 | 11161.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 11703.75 | 11286.92 | 11210.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 13122.00 | 13135.27 | 12974.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:45:00 | 13148.00 | 13135.27 | 12974.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 13000.00 | 13108.22 | 12976.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:15:00 | 13133.00 | 13108.22 | 12976.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 12887.00 | 13063.97 | 12968.29 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 14:15:00 | 12785.00 | 12907.93 | 12917.75 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 09:15:00 | 13131.00 | 12937.84 | 12928.73 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 12493.00 | 12931.64 | 12946.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 11:15:00 | 12344.00 | 12747.45 | 12855.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 12624.00 | 12551.14 | 12701.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 09:30:00 | 12565.00 | 12551.14 | 12701.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 12699.00 | 12580.71 | 12701.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 12699.00 | 12580.71 | 12701.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 12669.00 | 12598.37 | 12698.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 12680.00 | 12598.37 | 12698.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 12702.00 | 12619.10 | 12698.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 12725.00 | 12619.10 | 12698.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 12673.00 | 12629.88 | 12696.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 15:00:00 | 12563.00 | 12616.50 | 12684.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 09:15:00 | 12704.00 | 12628.16 | 12677.30 | SL hit (close>static) qty=1.00 sl=12702.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 12332.00 | 12205.75 | 12195.47 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 12055.00 | 12184.15 | 12190.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 12052.00 | 12157.72 | 12177.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 12102.00 | 12039.52 | 12091.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 14:15:00 | 12102.00 | 12039.52 | 12091.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 12102.00 | 12039.52 | 12091.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 12102.00 | 12039.52 | 12091.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 12140.00 | 12059.61 | 12095.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 12122.00 | 12059.61 | 12095.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 12295.00 | 12106.69 | 12113.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 12295.00 | 12106.69 | 12113.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 12436.00 | 12172.55 | 12143.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 11:15:00 | 12715.00 | 12281.04 | 12195.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 12334.00 | 12401.73 | 12283.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 12334.00 | 12401.73 | 12283.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 12334.00 | 12401.73 | 12283.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 12334.00 | 12401.73 | 12283.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 12400.00 | 12401.39 | 12293.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 11933.00 | 12401.39 | 12293.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 11970.00 | 12315.11 | 12264.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 11970.00 | 12315.11 | 12264.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 11893.00 | 12230.69 | 12230.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:30:00 | 11878.00 | 12230.69 | 12230.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 12033.00 | 12191.15 | 12212.65 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 14:15:00 | 12426.00 | 12234.27 | 12225.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 12596.00 | 12327.21 | 12270.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 12630.00 | 12638.70 | 12523.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 12630.00 | 12638.70 | 12523.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 12623.00 | 12629.37 | 12538.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:15:00 | 12665.00 | 12628.50 | 12546.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:00:00 | 12672.00 | 12629.92 | 12561.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 14:15:00 | 12480.00 | 12578.20 | 12552.45 | SL hit (close<static) qty=1.00 sl=12532.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 12071.00 | 12453.12 | 12500.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 11:15:00 | 11902.00 | 12342.89 | 12445.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 12:15:00 | 11255.00 | 11246.32 | 11546.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:30:00 | 11228.00 | 11246.32 | 11546.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 11161.00 | 11186.87 | 11316.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 11180.00 | 11186.87 | 11316.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 11194.00 | 11185.92 | 11253.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 15:15:00 | 11110.00 | 11199.83 | 11219.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 11315.00 | 11241.04 | 11232.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 11315.00 | 11241.04 | 11232.63 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 11173.00 | 11246.22 | 11247.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 13:15:00 | 11103.00 | 11217.58 | 11234.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 10:15:00 | 11356.00 | 11150.11 | 11164.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 10:15:00 | 11356.00 | 11150.11 | 11164.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 11356.00 | 11150.11 | 11164.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 11356.00 | 11150.11 | 11164.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 11206.00 | 11161.29 | 11168.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 11190.00 | 11163.43 | 11168.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 11368.00 | 11204.35 | 11186.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 11368.00 | 11204.35 | 11186.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 11485.00 | 11325.46 | 11255.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 11565.00 | 11622.33 | 11509.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 09:45:00 | 11557.00 | 11622.33 | 11509.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 11640.00 | 11714.94 | 11633.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 11640.00 | 11714.94 | 11633.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 11598.00 | 11691.55 | 11630.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:00:00 | 11598.00 | 11691.55 | 11630.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 11666.00 | 11686.44 | 11633.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 13:15:00 | 11728.00 | 11686.44 | 11633.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 15:00:00 | 11740.00 | 11776.98 | 11728.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 09:15:00 | 12900.80 | 11993.39 | 11835.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 10:15:00 | 12967.00 | 13147.61 | 13154.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 13:15:00 | 12891.00 | 13041.30 | 13099.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 15:15:00 | 12220.00 | 12197.23 | 12382.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 12228.00 | 12197.23 | 12382.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 12610.00 | 12279.78 | 12403.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 12610.00 | 12279.78 | 12403.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 12496.00 | 12323.03 | 12411.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 12315.00 | 12347.42 | 12414.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:00:00 | 12315.00 | 12337.11 | 12365.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:15:00 | 11699.25 | 11867.18 | 11914.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:15:00 | 11699.25 | 11867.18 | 11914.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 11918.00 | 11767.18 | 11816.37 | SL hit (close>ema200) qty=0.50 sl=11767.18 alert=retest2 |

### Cycle 154 — BUY (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 13:15:00 | 11980.00 | 11846.04 | 11845.92 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 11760.00 | 11840.60 | 11848.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 11685.00 | 11772.61 | 11808.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 12037.00 | 11788.24 | 11793.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 12037.00 | 11788.24 | 11793.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 12037.00 | 11788.24 | 11793.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 12365.00 | 11788.24 | 11793.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 11965.00 | 11823.59 | 11809.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 12:15:00 | 12090.00 | 12003.76 | 11946.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 14535.00 | 14647.08 | 14044.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 11:00:00 | 14535.00 | 14647.08 | 14044.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 14340.00 | 14491.87 | 14189.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 14459.00 | 14491.87 | 14189.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:45:00 | 14530.00 | 14460.40 | 14224.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 14496.00 | 14481.92 | 14339.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 14160.00 | 14313.42 | 14318.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 14160.00 | 14313.42 | 14318.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 14087.00 | 14244.55 | 14284.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 13897.00 | 13843.29 | 13948.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:00:00 | 13897.00 | 13843.29 | 13948.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 13952.00 | 13865.03 | 13949.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 13969.00 | 13865.03 | 13949.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 14090.00 | 13910.02 | 13961.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 14090.00 | 13910.02 | 13961.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 14120.00 | 13952.02 | 13976.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:15:00 | 13970.00 | 13952.02 | 13976.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 14020.00 | 13889.62 | 13888.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 14020.00 | 13889.62 | 13888.48 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 13811.00 | 13887.36 | 13890.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 09:15:00 | 13725.00 | 13854.89 | 13875.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 13906.00 | 13865.11 | 13878.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 13906.00 | 13865.11 | 13878.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 13906.00 | 13865.11 | 13878.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 13906.00 | 13865.11 | 13878.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 13856.00 | 13863.29 | 13876.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:45:00 | 13801.00 | 13852.17 | 13864.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 13922.00 | 13873.19 | 13871.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 13922.00 | 13873.19 | 13871.97 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 13768.00 | 13871.48 | 13874.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 11:15:00 | 13658.00 | 13828.79 | 13854.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 13023.00 | 13016.23 | 13230.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 13023.00 | 13016.23 | 13230.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 13024.00 | 13036.69 | 13173.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 13003.00 | 13036.69 | 13173.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:45:00 | 13000.00 | 13007.41 | 13124.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 13418.00 | 12889.54 | 12871.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 13418.00 | 12889.54 | 12871.35 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 13050.00 | 13099.40 | 13103.25 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 13188.00 | 13104.62 | 13095.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 14:15:00 | 13276.00 | 13138.89 | 13111.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 13239.00 | 13329.26 | 13247.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 13239.00 | 13329.26 | 13247.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 13239.00 | 13329.26 | 13247.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 13239.00 | 13329.26 | 13247.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 13170.00 | 13297.40 | 13240.67 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 13124.00 | 13208.91 | 13209.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 14:15:00 | 13055.00 | 13178.13 | 13195.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 13232.00 | 13156.64 | 13178.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 10:15:00 | 13232.00 | 13156.64 | 13178.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 13232.00 | 13156.64 | 13178.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:00:00 | 13232.00 | 13156.64 | 13178.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 11:15:00 | 13849.00 | 13295.11 | 13239.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 13937.00 | 13638.63 | 13454.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 13740.00 | 13768.74 | 13616.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 13649.00 | 13768.74 | 13616.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 13640.00 | 13742.99 | 13618.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 13622.00 | 13742.99 | 13618.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 13727.00 | 13739.79 | 13628.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:15:00 | 13870.00 | 13758.19 | 13656.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 14:30:00 | 13849.00 | 13784.08 | 13686.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 13850.00 | 13784.08 | 13686.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 13918.00 | 13807.81 | 13714.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 13749.00 | 13795.02 | 13732.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 13749.00 | 13795.02 | 13732.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 13714.00 | 13778.81 | 13730.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 13714.00 | 13778.81 | 13730.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 13561.00 | 13735.25 | 13715.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 13561.00 | 13735.25 | 13715.06 | SL hit (close<static) qty=1.00 sl=13622.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 13506.00 | 13689.40 | 13696.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 13356.00 | 13622.72 | 13665.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 13190.00 | 13183.24 | 13321.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:15:00 | 13286.00 | 13183.24 | 13321.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 13239.00 | 13194.39 | 13313.88 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 13492.00 | 13366.14 | 13361.99 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 13342.00 | 13364.42 | 13366.21 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 13452.00 | 13381.94 | 13374.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 13468.00 | 13400.61 | 13384.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 14:15:00 | 14435.00 | 14636.96 | 14419.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 14:15:00 | 14435.00 | 14636.96 | 14419.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 14435.00 | 14636.96 | 14419.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 14435.00 | 14636.96 | 14419.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 14549.00 | 14619.37 | 14431.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 14348.00 | 14619.37 | 14431.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 14262.00 | 14547.89 | 14416.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:00:00 | 14479.00 | 14534.12 | 14421.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:30:00 | 14471.00 | 14500.29 | 14416.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:30:00 | 14485.00 | 14444.39 | 14407.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:30:00 | 14448.00 | 14436.97 | 14413.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 14387.00 | 14426.97 | 14411.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 14387.00 | 14426.97 | 14411.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 14378.00 | 14417.18 | 14408.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 14395.00 | 14417.18 | 14408.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 14418.00 | 14415.50 | 14409.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 14564.00 | 14415.50 | 14409.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 14543.00 | 14441.00 | 14421.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:00:00 | 14749.00 | 14558.41 | 14488.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-12 09:15:00 | 15926.90 | 15423.26 | 15050.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 14:15:00 | 15822.00 | 15943.92 | 15952.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 15449.00 | 15830.52 | 15892.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 15023.00 | 14955.06 | 15159.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 13:45:00 | 15080.00 | 14955.06 | 15159.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 14412.00 | 14247.34 | 14534.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 14490.00 | 14247.34 | 14534.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 14483.00 | 14294.48 | 14529.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 14483.00 | 14294.48 | 14529.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 14405.00 | 14316.58 | 14518.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 14372.00 | 14406.18 | 14501.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 14618.00 | 14457.84 | 14480.89 | SL hit (close>static) qty=1.00 sl=14533.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 14923.00 | 14573.62 | 14530.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 10:15:00 | 15110.00 | 14680.89 | 14583.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 15328.00 | 15371.99 | 15114.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 13:45:00 | 15284.00 | 15371.99 | 15114.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 15195.00 | 15315.47 | 15131.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 15049.00 | 15315.47 | 15131.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 15183.00 | 15288.98 | 15136.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:15:00 | 15466.00 | 15191.58 | 15149.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 16147.00 | 16178.17 | 16179.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 16147.00 | 16178.17 | 16179.57 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 12:15:00 | 16310.00 | 16196.33 | 16186.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 16393.00 | 16292.75 | 16242.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 16209.00 | 16350.43 | 16309.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 16209.00 | 16350.43 | 16309.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 16209.00 | 16350.43 | 16309.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 16209.00 | 16350.43 | 16309.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 16240.00 | 16328.35 | 16302.78 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 16151.00 | 16272.98 | 16280.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 14:15:00 | 16143.00 | 16194.90 | 16224.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 15914.00 | 15882.90 | 16003.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 15914.00 | 15882.90 | 16003.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 15914.00 | 15882.90 | 16003.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 15713.00 | 15843.03 | 15955.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 15757.00 | 15803.73 | 15906.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 15740.00 | 15636.59 | 15724.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 16056.00 | 15802.03 | 15784.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 13:15:00 | 16056.00 | 15802.03 | 15784.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 16177.00 | 15877.03 | 15820.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 11:15:00 | 17408.00 | 17488.29 | 17184.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 11:45:00 | 17406.00 | 17488.29 | 17184.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 17570.00 | 17510.31 | 17308.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 18721.00 | 17697.87 | 17502.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:30:00 | 18160.00 | 17872.24 | 17620.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 12:00:00 | 18190.00 | 17935.79 | 17672.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 14:30:00 | 18102.00 | 18013.50 | 17777.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 17740.00 | 17960.80 | 17795.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 17740.00 | 17960.80 | 17795.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 17341.00 | 17836.84 | 17753.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 17341.00 | 17836.84 | 17753.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 17295.00 | 17728.47 | 17712.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 17297.00 | 17642.18 | 17674.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 12:15:00 | 17297.00 | 17642.18 | 17674.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 14:15:00 | 17201.00 | 17497.59 | 17599.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 10:15:00 | 17522.00 | 17398.08 | 17519.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 10:15:00 | 17522.00 | 17398.08 | 17519.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 17522.00 | 17398.08 | 17519.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 17522.00 | 17398.08 | 17519.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 17421.00 | 17402.66 | 17510.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 13:00:00 | 17335.00 | 17439.57 | 17485.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:00:00 | 17297.00 | 17411.06 | 17468.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:30:00 | 17333.00 | 17351.86 | 17422.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 17315.00 | 17388.09 | 17432.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 17867.00 | 17483.87 | 17471.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 17867.00 | 17483.87 | 17471.84 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 17360.00 | 17490.58 | 17496.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 17125.00 | 17417.46 | 17462.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 09:15:00 | 16931.00 | 16844.92 | 16991.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 16931.00 | 16844.92 | 16991.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 16931.00 | 16844.92 | 16991.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 17025.00 | 16844.92 | 16991.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 16609.00 | 16391.62 | 16516.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 16609.00 | 16391.62 | 16516.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 16584.00 | 16430.09 | 16522.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:45:00 | 16627.00 | 16430.09 | 16522.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 16909.00 | 16602.02 | 16589.46 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 16489.00 | 16610.29 | 16623.29 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 16723.00 | 16643.51 | 16636.79 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 16564.00 | 16632.80 | 16633.41 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 16716.00 | 16639.32 | 16635.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 11:15:00 | 16823.00 | 16676.06 | 16652.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 16883.00 | 16956.49 | 16822.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:00:00 | 16883.00 | 16956.49 | 16822.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 16866.00 | 16934.47 | 16835.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:30:00 | 16889.00 | 16934.47 | 16835.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 16853.00 | 16918.18 | 16836.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 16828.00 | 16918.18 | 16836.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 16772.00 | 16888.94 | 16831.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:45:00 | 16766.00 | 16888.94 | 16831.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 16667.00 | 16844.55 | 16816.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 16667.00 | 16844.55 | 16816.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 16530.00 | 16781.64 | 16790.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 16325.00 | 16690.31 | 16747.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 16656.00 | 16482.70 | 16579.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 16656.00 | 16482.70 | 16579.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 16656.00 | 16482.70 | 16579.37 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 17411.00 | 16713.21 | 16670.15 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 16611.00 | 16824.20 | 16850.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 16500.00 | 16759.36 | 16819.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 16431.00 | 16374.38 | 16564.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 16431.00 | 16374.38 | 16564.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 16431.00 | 16374.38 | 16564.47 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 16900.00 | 16670.97 | 16660.06 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 16550.00 | 16638.14 | 16646.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 16460.00 | 16575.69 | 16613.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 15:15:00 | 16285.00 | 16237.27 | 16358.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:15:00 | 16314.00 | 16237.27 | 16358.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 16320.00 | 16253.82 | 16355.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:15:00 | 16171.00 | 16267.08 | 16345.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 13:15:00 | 16426.00 | 16291.38 | 16294.37 | SL hit (close>static) qty=1.00 sl=16410.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 16450.00 | 16323.11 | 16308.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 15:15:00 | 16525.00 | 16363.49 | 16328.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 16177.00 | 16326.19 | 16314.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 16177.00 | 16326.19 | 16314.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 16177.00 | 16326.19 | 16314.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:15:00 | 16150.00 | 16326.19 | 16314.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 16080.00 | 16276.95 | 16293.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 15990.00 | 16125.85 | 16197.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 15767.00 | 15747.98 | 15911.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:30:00 | 15645.00 | 15747.98 | 15911.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 16262.00 | 15872.09 | 15918.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 15890.00 | 15933.83 | 15939.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 15996.00 | 15946.26 | 15944.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 15996.00 | 15946.26 | 15944.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 16089.00 | 15974.81 | 15957.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 15:15:00 | 15970.00 | 15973.85 | 15958.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 15:15:00 | 15970.00 | 15973.85 | 15958.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 15970.00 | 15973.85 | 15958.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 15873.00 | 15973.85 | 15958.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 15862.00 | 15951.48 | 15949.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:15:00 | 15835.00 | 15951.48 | 15949.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 10:15:00 | 15850.00 | 15931.18 | 15940.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 15770.00 | 15877.96 | 15913.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 15272.00 | 15186.08 | 15326.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 15272.00 | 15186.08 | 15326.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 15200.00 | 15201.81 | 15310.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:30:00 | 15295.00 | 15201.81 | 15310.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 14901.00 | 14967.02 | 15044.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 14865.00 | 14952.01 | 15030.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:45:00 | 14867.00 | 14936.21 | 15016.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 14:15:00 | 15210.00 | 15001.69 | 15026.87 | SL hit (close>static) qty=1.00 sl=15098.00 alert=retest2 |

### Cycle 194 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 15207.00 | 15076.08 | 15058.38 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 14646.00 | 15027.84 | 15078.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 14556.00 | 14933.47 | 15030.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 14958.00 | 14742.93 | 14870.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 14958.00 | 14742.93 | 14870.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 14958.00 | 14742.93 | 14870.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 14958.00 | 14742.93 | 14870.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 14928.00 | 14779.95 | 14875.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 14928.00 | 14779.95 | 14875.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 14998.00 | 14823.56 | 14886.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 14959.00 | 14823.56 | 14886.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 15039.00 | 14866.65 | 14900.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 15039.00 | 14866.65 | 14900.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 15149.00 | 14957.41 | 14937.95 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 14797.00 | 14942.19 | 14953.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 14711.00 | 14857.48 | 14906.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 14656.00 | 14579.34 | 14711.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 14656.00 | 14579.34 | 14711.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 14796.00 | 14622.67 | 14718.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 14790.00 | 14622.67 | 14718.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 14620.00 | 14622.14 | 14709.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 14467.00 | 14622.14 | 14709.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 14370.00 | 14571.71 | 14679.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 14350.00 | 14520.57 | 14646.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:30:00 | 14333.00 | 14417.75 | 14562.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 14345.00 | 14422.36 | 14496.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 14351.00 | 14343.26 | 14429.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 13632.50 | 13824.04 | 14017.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 13616.35 | 13824.04 | 14017.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 13627.75 | 13824.04 | 14017.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 13633.45 | 13824.04 | 14017.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 13579.00 | 13340.02 | 13527.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 13579.00 | 13340.02 | 13527.60 | SL hit (close>ema200) qty=0.50 sl=13340.02 alert=retest2 |

### Cycle 198 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 13275.00 | 12946.21 | 12912.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 13549.00 | 13145.52 | 13018.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 13450.00 | 13537.90 | 13320.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 13172.00 | 13537.90 | 13320.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 13022.00 | 13434.72 | 13293.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 13022.00 | 13434.72 | 13293.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 12931.00 | 13333.98 | 13260.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 12931.00 | 13333.98 | 13260.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 13:15:00 | 13060.00 | 13187.19 | 13203.16 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 13526.00 | 13244.11 | 13223.86 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 13120.00 | 13295.13 | 13314.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 13040.00 | 13167.74 | 13234.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 13322.00 | 13126.70 | 13174.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 13322.00 | 13126.70 | 13174.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 13322.00 | 13126.70 | 13174.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 13284.00 | 13126.70 | 13174.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 13556.00 | 13212.56 | 13209.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 14123.00 | 13535.92 | 13373.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 13519.00 | 13635.47 | 13468.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 10:15:00 | 13519.00 | 13635.47 | 13468.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 13519.00 | 13635.47 | 13468.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 13519.00 | 13635.47 | 13468.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 13361.00 | 13580.58 | 13459.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:00:00 | 13361.00 | 13580.58 | 13459.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 13532.00 | 13570.86 | 13465.79 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 13228.00 | 13396.66 | 13419.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 13109.00 | 13339.13 | 13391.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 13148.00 | 13055.41 | 13137.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 13:15:00 | 13148.00 | 13055.41 | 13137.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 13148.00 | 13055.41 | 13137.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:00:00 | 13148.00 | 13055.41 | 13137.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 13063.00 | 13056.93 | 13131.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 13045.00 | 13056.93 | 13131.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 13329.00 | 13109.44 | 13142.03 | SL hit (close>static) qty=1.00 sl=13159.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 13267.00 | 13178.64 | 13170.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 13953.00 | 13376.59 | 13269.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 13328.00 | 13557.86 | 13447.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 13328.00 | 13557.86 | 13447.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 13328.00 | 13557.86 | 13447.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 13328.00 | 13557.86 | 13447.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 13529.00 | 13552.09 | 13455.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:30:00 | 13325.00 | 13552.09 | 13455.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 13461.00 | 13582.50 | 13520.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 13520.00 | 13582.50 | 13520.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 13409.00 | 13547.80 | 13510.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 13428.00 | 13547.80 | 13510.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 13391.00 | 13486.71 | 13487.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 13171.00 | 13423.57 | 13458.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 13115.00 | 12890.84 | 13068.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 13115.00 | 12890.84 | 13068.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 13115.00 | 12890.84 | 13068.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 13199.00 | 12890.84 | 13068.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 12896.00 | 12891.87 | 13052.55 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 13519.00 | 13141.72 | 13138.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 13670.00 | 13247.37 | 13186.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 13321.00 | 13369.09 | 13321.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 13321.00 | 13369.09 | 13321.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 13321.00 | 13369.09 | 13321.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 13321.00 | 13369.09 | 13321.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 13241.00 | 13343.47 | 13314.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 13241.00 | 13343.47 | 13314.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 13342.00 | 13343.18 | 13316.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 13407.00 | 13347.74 | 13321.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:45:00 | 13555.00 | 13373.39 | 13335.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 12:15:00 | 13211.00 | 13320.90 | 13319.26 | SL hit (close<static) qty=1.00 sl=13230.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 13217.00 | 13300.12 | 13309.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 12870.00 | 13186.80 | 13253.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 09:15:00 | 12933.00 | 12921.52 | 13051.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 09:45:00 | 12955.00 | 12921.52 | 13051.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 13017.00 | 12940.61 | 13048.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:30:00 | 13062.00 | 12940.61 | 13048.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 13070.00 | 12982.86 | 13035.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 13070.00 | 12982.86 | 13035.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 13109.00 | 13008.09 | 13042.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 12833.00 | 13008.09 | 13042.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 10:15:00 | 13067.00 | 12968.76 | 12957.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 13067.00 | 12968.76 | 12957.95 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 12637.00 | 12938.25 | 12957.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 12592.00 | 12869.00 | 12924.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 12710.00 | 12692.03 | 12792.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 12710.00 | 12692.03 | 12792.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 12710.00 | 12692.03 | 12792.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:30:00 | 12665.00 | 12686.54 | 12772.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:00:00 | 12659.00 | 12686.54 | 12772.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 13017.00 | 12716.54 | 12748.75 | SL hit (close>static) qty=1.00 sl=12895.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 13024.00 | 12778.03 | 12773.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 13091.00 | 12840.62 | 12802.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 12795.00 | 12856.50 | 12821.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 12795.00 | 12856.50 | 12821.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 12795.00 | 12856.50 | 12821.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 12795.00 | 12856.50 | 12821.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 12838.00 | 12852.80 | 12823.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 12560.00 | 12852.80 | 12823.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 12635.00 | 12809.24 | 12806.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 12506.00 | 12809.24 | 12806.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 12550.00 | 12757.39 | 12782.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 12295.00 | 12586.09 | 12680.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 12192.00 | 12191.27 | 12333.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 12192.00 | 12191.27 | 12333.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 12330.00 | 12225.21 | 12324.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 12340.00 | 12225.21 | 12324.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 12284.00 | 12236.97 | 12320.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 12368.00 | 12236.97 | 12320.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 12351.00 | 12259.78 | 12323.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 12366.00 | 12259.78 | 12323.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 12309.00 | 12269.62 | 12322.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:15:00 | 12333.00 | 12269.62 | 12322.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 12293.00 | 12274.30 | 12319.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 12247.00 | 12265.44 | 12311.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 12466.00 | 12322.04 | 12324.07 | SL hit (close>static) qty=1.00 sl=12390.00 alert=retest2 |

### Cycle 212 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 12375.00 | 12332.64 | 12328.70 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 12125.00 | 12303.21 | 12325.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 11961.00 | 12178.13 | 12257.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 11810.00 | 11677.02 | 11809.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 11810.00 | 11677.02 | 11809.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 11810.00 | 11677.02 | 11809.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 11810.00 | 11677.02 | 11809.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 11985.00 | 11738.62 | 11825.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 12015.00 | 11738.62 | 11825.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 12038.00 | 11798.49 | 11845.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 12038.00 | 11798.49 | 11845.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 12002.00 | 11882.16 | 11877.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 12336.00 | 11972.92 | 11919.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 12090.00 | 12137.80 | 12039.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 12090.00 | 12137.80 | 12039.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 11890.00 | 12082.04 | 12030.60 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 11900.00 | 12031.64 | 12033.04 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 15:15:00 | 12060.00 | 12032.36 | 12031.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 12722.00 | 12170.28 | 12094.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 12178.00 | 12443.88 | 12314.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 12178.00 | 12443.88 | 12314.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 12178.00 | 12443.88 | 12314.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 12178.00 | 12443.88 | 12314.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 12266.00 | 12408.31 | 12310.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 12273.00 | 12408.31 | 12310.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-07 09:15:00 | 13500.30 | 13034.83 | 12753.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 14900.00 | 15119.10 | 15126.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 14776.00 | 15050.48 | 15094.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 11:15:00 | 15043.00 | 15007.31 | 15064.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 12:00:00 | 15043.00 | 15007.31 | 15064.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 15266.00 | 15059.04 | 15083.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 15266.00 | 15059.04 | 15083.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 15115.00 | 15070.24 | 15086.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:30:00 | 15318.00 | 15070.24 | 15086.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 15042.00 | 15066.15 | 15081.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 15120.00 | 15066.15 | 15081.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 15368.00 | 15126.52 | 15107.62 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 14950.00 | 15092.35 | 15106.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 14870.00 | 15047.88 | 15085.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 14748.00 | 14655.67 | 14825.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 14748.00 | 14655.67 | 14825.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 14748.00 | 14655.67 | 14825.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 14782.00 | 14655.67 | 14825.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 14809.00 | 14686.34 | 14823.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 14943.00 | 14686.34 | 14823.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 14871.00 | 14723.27 | 14827.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 14871.00 | 14723.27 | 14827.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 14887.00 | 14756.02 | 14833.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 14887.00 | 14756.02 | 14833.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 14888.00 | 14782.41 | 14838.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:15:00 | 14840.00 | 14782.41 | 14838.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:45:00 | 14836.00 | 14770.93 | 14827.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 14847.00 | 14709.64 | 14707.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 14847.00 | 14709.64 | 14707.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 14980.00 | 14799.61 | 14751.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 17153.00 | 17182.90 | 16810.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:15:00 | 17563.00 | 17182.90 | 16810.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-22 12:00:00 | 7604.95 | 2024-04-24 10:15:00 | 7310.00 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2024-04-22 13:00:00 | 7670.20 | 2024-04-24 10:15:00 | 7310.00 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest2 | 2024-04-23 10:00:00 | 7644.15 | 2024-04-24 10:15:00 | 7310.00 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2024-05-07 14:30:00 | 7360.20 | 2024-05-08 09:15:00 | 7463.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-05-07 15:15:00 | 7312.50 | 2024-05-08 09:15:00 | 7463.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-05-17 10:15:00 | 6219.20 | 2024-05-22 11:15:00 | 6280.15 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-05-18 09:30:00 | 6206.40 | 2024-05-22 11:15:00 | 6280.15 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-05-21 09:15:00 | 6221.75 | 2024-05-22 11:15:00 | 6280.15 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-05-27 09:15:00 | 6399.70 | 2024-05-28 09:15:00 | 6225.10 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2024-05-30 15:15:00 | 6185.25 | 2024-06-04 10:15:00 | 5875.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 15:15:00 | 6185.25 | 2024-06-04 12:15:00 | 5566.73 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-07 09:15:00 | 6280.00 | 2024-06-18 13:15:00 | 6450.00 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2024-06-21 15:00:00 | 6560.95 | 2024-06-25 14:15:00 | 7217.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-26 11:30:00 | 8379.70 | 2024-08-01 15:15:00 | 9217.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-29 13:45:00 | 8385.00 | 2024-08-01 15:15:00 | 9223.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-27 09:15:00 | 12184.00 | 2024-09-06 09:15:00 | 13206.49 | TARGET_HIT | 1.00 | 8.39% |
| BUY | retest2 | 2024-08-27 10:30:00 | 12005.90 | 2024-09-06 09:15:00 | 13145.33 | TARGET_HIT | 1.00 | 9.49% |
| BUY | retest2 | 2024-08-27 12:00:00 | 11950.30 | 2024-09-09 11:15:00 | 12650.05 | STOP_HIT | 1.00 | 5.86% |
| SELL | retest2 | 2024-09-11 11:45:00 | 12463.90 | 2024-09-12 10:15:00 | 12742.80 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-10-03 10:30:00 | 12350.00 | 2024-10-07 10:15:00 | 11732.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 11:30:00 | 12352.55 | 2024-10-07 10:15:00 | 11734.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 12:00:00 | 12351.00 | 2024-10-07 10:15:00 | 11733.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 12:30:00 | 12341.80 | 2024-10-07 10:15:00 | 11724.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 10:30:00 | 12350.00 | 2024-10-07 12:15:00 | 11117.30 | TARGET_HIT | 0.50 | 9.98% |
| SELL | retest2 | 2024-10-04 11:30:00 | 12352.55 | 2024-10-08 09:15:00 | 11115.00 | TARGET_HIT | 0.50 | 10.02% |
| SELL | retest2 | 2024-10-04 12:00:00 | 12351.00 | 2024-10-08 09:15:00 | 11115.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-04 12:30:00 | 12341.80 | 2024-10-08 09:15:00 | 11107.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-07 10:45:00 | 11619.40 | 2024-10-08 09:15:00 | 11115.52 | PARTIAL | 0.50 | 4.34% |
| SELL | retest2 | 2024-10-07 10:45:00 | 11619.40 | 2024-10-08 10:15:00 | 11849.80 | STOP_HIT | 0.50 | -1.98% |
| SELL | retest2 | 2024-10-07 11:30:00 | 11700.55 | 2024-10-10 10:15:00 | 12500.00 | STOP_HIT | 1.00 | -6.83% |
| SELL | retest2 | 2024-10-08 12:00:00 | 11770.00 | 2024-10-10 10:15:00 | 12500.00 | STOP_HIT | 1.00 | -6.20% |
| SELL | retest2 | 2024-10-08 14:00:00 | 11770.00 | 2024-10-10 10:15:00 | 12500.00 | STOP_HIT | 1.00 | -6.20% |
| SELL | retest2 | 2024-10-10 09:45:00 | 11745.55 | 2024-10-10 10:15:00 | 12500.00 | STOP_HIT | 1.00 | -6.42% |
| SELL | retest2 | 2024-10-24 11:45:00 | 13790.00 | 2024-10-28 14:15:00 | 13900.20 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-10-24 12:15:00 | 13937.35 | 2024-10-28 14:15:00 | 13900.20 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-10-24 13:45:00 | 13810.15 | 2024-10-28 14:15:00 | 13900.20 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-10-30 09:45:00 | 14085.00 | 2024-10-30 14:15:00 | 13806.30 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-10-30 10:15:00 | 14078.55 | 2024-10-30 14:15:00 | 13806.30 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-10-31 10:15:00 | 14072.55 | 2024-11-05 09:15:00 | 15479.81 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-26 10:00:00 | 14720.00 | 2024-11-26 14:15:00 | 14924.85 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-11-26 10:30:00 | 14721.00 | 2024-11-26 14:15:00 | 14924.85 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-12-03 11:00:00 | 16910.05 | 2024-12-09 09:15:00 | 15977.25 | STOP_HIT | 1.00 | -5.52% |
| SELL | retest2 | 2024-12-24 12:15:00 | 13877.65 | 2024-12-30 12:15:00 | 13910.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-01-20 11:15:00 | 13780.45 | 2025-01-21 12:15:00 | 13345.75 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-01-20 12:30:00 | 13783.60 | 2025-01-21 12:15:00 | 13345.75 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2025-01-20 14:30:00 | 13863.85 | 2025-01-21 12:15:00 | 13345.75 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-01-21 09:15:00 | 13842.85 | 2025-01-21 12:15:00 | 13345.75 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-01-24 09:15:00 | 14104.85 | 2025-01-27 13:15:00 | 13494.00 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-01-24 12:00:00 | 14000.05 | 2025-01-27 13:15:00 | 13494.00 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2025-01-31 09:30:00 | 14016.60 | 2025-02-07 09:15:00 | 14162.50 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2025-01-31 11:00:00 | 13907.50 | 2025-02-07 09:15:00 | 14162.50 | STOP_HIT | 1.00 | 1.83% |
| BUY | retest2 | 2025-01-31 13:15:00 | 13999.90 | 2025-02-07 09:15:00 | 14162.50 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2025-02-01 12:00:00 | 14032.50 | 2025-02-07 09:15:00 | 14162.50 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-02-04 09:30:00 | 14609.95 | 2025-02-07 09:15:00 | 14162.50 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-02-11 09:15:00 | 13853.00 | 2025-02-12 09:15:00 | 13160.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 13853.00 | 2025-02-12 12:15:00 | 13712.85 | STOP_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2025-03-13 13:30:00 | 10752.55 | 2025-03-13 15:15:00 | 11599.80 | STOP_HIT | 1.00 | -7.88% |
| SELL | retest2 | 2025-03-17 12:30:00 | 10954.85 | 2025-03-18 12:15:00 | 11219.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-03-25 13:45:00 | 11761.90 | 2025-03-26 10:15:00 | 11426.15 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-03-25 14:15:00 | 11782.20 | 2025-03-26 10:15:00 | 11426.15 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2025-04-04 11:15:00 | 11800.00 | 2025-04-04 12:15:00 | 11498.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-04-09 09:15:00 | 10953.20 | 2025-04-11 11:15:00 | 11610.00 | STOP_HIT | 1.00 | -6.00% |
| SELL | retest2 | 2025-04-09 11:45:00 | 11070.05 | 2025-04-11 11:15:00 | 11610.00 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2025-04-09 12:15:00 | 11070.70 | 2025-04-11 11:15:00 | 11610.00 | STOP_HIT | 1.00 | -4.87% |
| SELL | retest2 | 2025-04-09 12:45:00 | 10916.35 | 2025-04-11 11:15:00 | 11610.00 | STOP_HIT | 1.00 | -6.35% |
| SELL | retest2 | 2025-04-28 15:00:00 | 12563.00 | 2025-04-29 09:15:00 | 12704.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-04-29 11:45:00 | 12550.00 | 2025-04-30 15:15:00 | 11922.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 11:45:00 | 12550.00 | 2025-05-05 09:15:00 | 12081.00 | STOP_HIT | 0.50 | 3.74% |
| BUY | retest2 | 2025-05-14 10:15:00 | 12665.00 | 2025-05-14 14:15:00 | 12480.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-14 12:00:00 | 12672.00 | 2025-05-14 14:15:00 | 12480.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-05-23 15:15:00 | 11110.00 | 2025-05-26 11:15:00 | 11315.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-05-29 12:30:00 | 11190.00 | 2025-05-29 13:15:00 | 11368.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-06-04 13:15:00 | 11728.00 | 2025-06-06 09:15:00 | 12900.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-05 15:00:00 | 11740.00 | 2025-06-06 09:15:00 | 12914.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-20 12:15:00 | 12315.00 | 2025-07-03 10:15:00 | 11699.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-23 13:00:00 | 12315.00 | 2025-07-03 10:15:00 | 11699.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-20 12:15:00 | 12315.00 | 2025-07-04 11:15:00 | 11918.00 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2025-06-23 13:00:00 | 12315.00 | 2025-07-04 11:15:00 | 11918.00 | STOP_HIT | 0.50 | 3.22% |
| BUY | retest2 | 2025-07-17 09:15:00 | 14459.00 | 2025-07-21 09:15:00 | 14160.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-07-17 10:45:00 | 14530.00 | 2025-07-21 09:15:00 | 14160.00 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-07-18 10:00:00 | 14496.00 | 2025-07-21 09:15:00 | 14160.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-07-24 13:15:00 | 13970.00 | 2025-07-28 11:15:00 | 14020.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-07-30 10:45:00 | 13801.00 | 2025-07-30 13:15:00 | 13922.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-08-05 10:15:00 | 13003.00 | 2025-08-08 09:15:00 | 13418.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-08-05 12:45:00 | 13000.00 | 2025-08-08 09:15:00 | 13418.00 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-08-22 13:15:00 | 13870.00 | 2025-08-25 14:15:00 | 13561.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-08-22 14:30:00 | 13849.00 | 2025-08-25 14:15:00 | 13561.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-08-22 15:15:00 | 13850.00 | 2025-08-25 14:15:00 | 13561.00 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-08-25 09:30:00 | 13918.00 | 2025-08-25 14:15:00 | 13561.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-09-08 11:00:00 | 14479.00 | 2025-09-12 09:15:00 | 15926.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-08 11:30:00 | 14471.00 | 2025-09-12 09:15:00 | 15918.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-08 14:30:00 | 14485.00 | 2025-09-12 09:15:00 | 15933.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-09 10:30:00 | 14448.00 | 2025-09-12 09:15:00 | 15892.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-10 14:00:00 | 14749.00 | 2025-09-16 09:15:00 | 16223.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-30 09:15:00 | 14372.00 | 2025-09-30 14:15:00 | 14618.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-10-07 10:15:00 | 15466.00 | 2025-10-17 09:15:00 | 16147.00 | STOP_HIT | 1.00 | 4.40% |
| SELL | retest2 | 2025-10-28 12:45:00 | 15713.00 | 2025-10-30 13:15:00 | 16056.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-10-29 09:15:00 | 15757.00 | 2025-10-30 13:15:00 | 16056.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-10-30 11:15:00 | 15740.00 | 2025-10-30 13:15:00 | 16056.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-11-10 09:15:00 | 18721.00 | 2025-11-11 12:15:00 | 17297.00 | STOP_HIT | 1.00 | -7.61% |
| BUY | retest2 | 2025-11-10 10:30:00 | 18160.00 | 2025-11-11 12:15:00 | 17297.00 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2025-11-10 12:00:00 | 18190.00 | 2025-11-11 12:15:00 | 17297.00 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2025-11-10 14:30:00 | 18102.00 | 2025-11-11 12:15:00 | 17297.00 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2025-11-13 13:00:00 | 17335.00 | 2025-11-14 11:15:00 | 17867.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-11-13 14:00:00 | 17297.00 | 2025-11-14 11:15:00 | 17867.00 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-11-14 09:30:00 | 17333.00 | 2025-11-14 11:15:00 | 17867.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-14 10:30:00 | 17315.00 | 2025-11-14 11:15:00 | 17867.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-12-12 12:15:00 | 16171.00 | 2025-12-15 13:15:00 | 16426.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-12-19 13:15:00 | 15890.00 | 2025-12-19 13:15:00 | 15996.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-31 11:15:00 | 14865.00 | 2025-12-31 14:15:00 | 15210.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-12-31 11:45:00 | 14867.00 | 2025-12-31 14:15:00 | 15210.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-01-13 10:30:00 | 14350.00 | 2026-01-20 10:15:00 | 13632.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:30:00 | 14333.00 | 2026-01-20 10:15:00 | 13616.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:30:00 | 14345.00 | 2026-01-20 10:15:00 | 13627.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:00:00 | 14351.00 | 2026-01-20 10:15:00 | 13633.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:30:00 | 14350.00 | 2026-01-22 09:15:00 | 13579.00 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2026-01-13 13:30:00 | 14333.00 | 2026-01-22 09:15:00 | 13579.00 | STOP_HIT | 0.50 | 5.26% |
| SELL | retest2 | 2026-01-14 13:30:00 | 14345.00 | 2026-01-22 09:15:00 | 13579.00 | STOP_HIT | 0.50 | 5.34% |
| SELL | retest2 | 2026-01-16 11:00:00 | 14351.00 | 2026-01-22 09:15:00 | 13579.00 | STOP_HIT | 0.50 | 5.38% |
| SELL | retest2 | 2026-01-22 15:15:00 | 13521.00 | 2026-01-27 11:15:00 | 12844.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 15:15:00 | 13521.00 | 2026-01-30 09:15:00 | 12761.00 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2026-02-13 15:15:00 | 13045.00 | 2026-02-16 09:15:00 | 13329.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-02-26 09:15:00 | 13407.00 | 2026-02-26 12:15:00 | 13211.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-02-26 09:45:00 | 13555.00 | 2026-02-26 12:15:00 | 13211.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-03-04 09:15:00 | 12833.00 | 2026-03-06 10:15:00 | 13067.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-03-10 11:30:00 | 12665.00 | 2026-03-11 09:15:00 | 13017.00 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2026-03-10 12:00:00 | 12659.00 | 2026-03-11 09:15:00 | 13017.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2026-03-17 13:30:00 | 12247.00 | 2026-03-18 10:15:00 | 12466.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-04-02 11:15:00 | 12273.00 | 2026-04-07 09:15:00 | 13500.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 14:15:00 | 14840.00 | 2026-04-30 10:15:00 | 14847.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-04-27 14:45:00 | 14836.00 | 2026-04-30 10:15:00 | 14847.00 | STOP_HIT | 1.00 | -0.07% |
