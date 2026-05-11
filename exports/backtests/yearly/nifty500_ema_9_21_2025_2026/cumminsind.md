# Cummins India Ltd. (CUMMINSIND)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 5391.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 53 |
| ALERT2 | 53 |
| ALERT2_SKIP | 34 |
| ALERT3 | 125 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 65 |
| PARTIAL | 4 |
| TARGET_HIT | 7 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 47
- **Target hits / Stop hits / Partials:** 7 / 58 / 4
- **Avg / median % per leg:** 0.72% / -0.85%
- **Sum % (uncompounded):** 49.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 13 | 38.2% | 7 | 27 | 0 | 1.61% | 54.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 34 | 13 | 38.2% | 7 | 27 | 0 | 1.61% | 54.8% |
| SELL (all) | 35 | 9 | 25.7% | 0 | 31 | 4 | -0.15% | -5.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 9 | 25.7% | 0 | 31 | 4 | -0.15% | -5.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 69 | 22 | 31.9% | 7 | 58 | 4 | 0.72% | 49.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 2844.90 | 2800.87 | 2797.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 2878.50 | 2816.40 | 2804.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 2872.30 | 2878.68 | 2859.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 12:15:00 | 2872.30 | 2878.68 | 2859.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 2872.30 | 2878.68 | 2859.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 2860.60 | 2878.68 | 2859.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 2871.40 | 2874.77 | 2860.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 2871.10 | 2874.77 | 2860.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 2988.70 | 3014.68 | 2984.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 2988.70 | 3014.68 | 2984.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2951.80 | 3000.49 | 2982.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:15:00 | 2946.00 | 3000.49 | 2982.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2981.70 | 2996.73 | 2982.70 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 2950.90 | 2974.19 | 2975.54 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 13:15:00 | 2989.80 | 2976.91 | 2975.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 3006.80 | 2984.39 | 2979.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 2975.00 | 2982.51 | 2979.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 2975.00 | 2982.51 | 2979.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2975.00 | 2982.51 | 2979.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 2975.00 | 2982.51 | 2979.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 2972.60 | 2980.53 | 2978.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 2970.50 | 2980.53 | 2978.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 2971.00 | 2978.62 | 2977.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 2970.90 | 2978.62 | 2977.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 2970.50 | 2977.00 | 2977.12 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 2980.20 | 2977.67 | 2977.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 3000.00 | 2982.14 | 2979.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 10:15:00 | 2981.00 | 2981.91 | 2979.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 2981.00 | 2981.91 | 2979.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2981.00 | 2981.91 | 2979.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 2981.00 | 2981.91 | 2979.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 11:15:00 | 2962.00 | 2977.93 | 2978.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 12:15:00 | 2946.10 | 2971.56 | 2975.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 2962.80 | 2956.59 | 2965.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 2962.80 | 2956.59 | 2965.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2962.80 | 2956.59 | 2965.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 2968.40 | 2956.59 | 2965.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 2937.90 | 2952.85 | 2963.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:45:00 | 2924.00 | 2946.98 | 2959.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:45:00 | 2931.60 | 2942.08 | 2954.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 2927.90 | 2942.63 | 2952.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:45:00 | 2930.90 | 2942.82 | 2952.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 2966.30 | 2947.52 | 2953.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 2966.30 | 2947.52 | 2953.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 2953.80 | 2948.77 | 2953.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 2958.20 | 2948.77 | 2953.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 2967.40 | 2952.50 | 2954.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 2967.40 | 2952.50 | 2954.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 2954.50 | 2952.90 | 2954.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:30:00 | 2962.30 | 2952.90 | 2954.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-27 14:15:00 | 2973.70 | 2957.06 | 2956.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 2973.70 | 2957.06 | 2956.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 15:15:00 | 2979.00 | 2961.45 | 2958.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 2985.50 | 2991.92 | 2978.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 2985.50 | 2991.92 | 2978.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 2960.00 | 2985.53 | 2976.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 3061.80 | 2985.53 | 2976.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-04 09:15:00 | 3367.98 | 3317.44 | 3280.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 3359.90 | 3381.43 | 3382.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 3347.30 | 3370.84 | 3377.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 3321.00 | 3314.95 | 3333.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 3318.90 | 3314.95 | 3333.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 3292.10 | 3310.38 | 3330.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 3268.70 | 3297.81 | 3309.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 3274.10 | 3290.14 | 3304.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 3273.50 | 3287.45 | 3301.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 3273.00 | 3281.96 | 3295.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 3295.90 | 3263.84 | 3275.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 3295.90 | 3263.84 | 3275.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 3286.00 | 3268.28 | 3276.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 3277.10 | 3268.28 | 3276.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:00:00 | 3280.00 | 3273.49 | 3277.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 3267.30 | 3272.25 | 3276.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 3322.60 | 3269.68 | 3268.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 3322.60 | 3269.68 | 3268.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 10:15:00 | 3339.10 | 3314.29 | 3301.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 3373.50 | 3377.32 | 3352.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 14:15:00 | 3376.00 | 3377.32 | 3352.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 3330.20 | 3367.89 | 3350.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 3330.20 | 3367.89 | 3350.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 3358.40 | 3365.99 | 3351.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 3384.00 | 3365.99 | 3351.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 15:15:00 | 3352.00 | 3369.45 | 3371.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 3352.00 | 3369.45 | 3371.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 3329.70 | 3361.50 | 3367.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3362.80 | 3346.71 | 3354.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 3362.80 | 3346.71 | 3354.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 3362.80 | 3346.71 | 3354.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 3357.80 | 3346.71 | 3354.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 3372.50 | 3351.87 | 3356.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 3372.50 | 3351.87 | 3356.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 3367.30 | 3359.49 | 3359.17 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 3330.80 | 3353.75 | 3356.59 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 3377.90 | 3353.39 | 3352.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 12:15:00 | 3403.90 | 3369.24 | 3359.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 3509.60 | 3513.26 | 3483.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 3546.70 | 3523.26 | 3499.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 3546.70 | 3523.26 | 3499.70 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 3547.90 | 3561.08 | 3562.79 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 3586.70 | 3561.73 | 3559.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 13:15:00 | 3626.10 | 3574.61 | 3565.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 3563.90 | 3584.21 | 3573.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 3563.90 | 3584.21 | 3573.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3563.90 | 3584.21 | 3573.31 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 3567.00 | 3588.11 | 3589.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 3555.80 | 3581.65 | 3586.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 3529.70 | 3525.44 | 3544.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 15:15:00 | 3540.00 | 3528.35 | 3543.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 3540.00 | 3528.35 | 3543.85 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 3582.20 | 3542.37 | 3541.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 10:15:00 | 3583.10 | 3563.52 | 3555.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 3563.10 | 3578.30 | 3566.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 3563.10 | 3578.30 | 3566.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 3563.10 | 3578.30 | 3566.31 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 3499.70 | 3552.58 | 3558.20 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 3569.70 | 3561.89 | 3561.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 15:15:00 | 3600.00 | 3573.89 | 3567.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 11:15:00 | 3588.40 | 3612.12 | 3598.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 11:15:00 | 3588.40 | 3612.12 | 3598.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 3588.40 | 3612.12 | 3598.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 3797.10 | 3778.92 | 3714.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 3749.50 | 3783.21 | 3786.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 3749.50 | 3783.21 | 3786.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 11:15:00 | 3746.80 | 3775.93 | 3782.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 3776.30 | 3762.73 | 3772.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 3776.30 | 3762.73 | 3772.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 3776.30 | 3762.73 | 3772.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 3776.30 | 3762.73 | 3772.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 3789.90 | 3768.17 | 3773.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 3797.80 | 3768.17 | 3773.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 3823.70 | 3785.22 | 3780.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 3858.90 | 3810.08 | 3793.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 3884.90 | 3898.00 | 3870.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:00:00 | 3884.90 | 3898.00 | 3870.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 3901.60 | 3897.02 | 3874.46 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 3844.90 | 3869.24 | 3870.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 3841.10 | 3861.80 | 3866.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 3836.70 | 3836.22 | 3846.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 15:15:00 | 3836.70 | 3836.22 | 3846.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 3836.70 | 3836.22 | 3846.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 3851.00 | 3836.22 | 3846.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 3860.10 | 3841.00 | 3847.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 3858.70 | 3841.00 | 3847.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 3872.70 | 3847.34 | 3849.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 3872.70 | 3847.34 | 3849.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 3898.00 | 3857.47 | 3854.15 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 3823.20 | 3852.28 | 3852.93 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 3878.10 | 3854.51 | 3853.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 3899.30 | 3863.47 | 3857.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 3858.60 | 3872.45 | 3865.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 3858.60 | 3872.45 | 3865.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 3858.60 | 3872.45 | 3865.69 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 3837.60 | 3859.76 | 3861.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 14:15:00 | 3827.60 | 3853.33 | 3858.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 12:15:00 | 3860.00 | 3841.37 | 3849.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 12:15:00 | 3860.00 | 3841.37 | 3849.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 3860.00 | 3841.37 | 3849.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 3860.00 | 3841.37 | 3849.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 3867.10 | 3846.52 | 3850.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 3867.10 | 3846.52 | 3850.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 3886.40 | 3854.50 | 3853.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 15:15:00 | 3892.50 | 3862.10 | 3857.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 11:15:00 | 3915.50 | 3926.21 | 3905.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 12:00:00 | 3915.50 | 3926.21 | 3905.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 3919.10 | 3924.79 | 3906.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 3907.40 | 3924.79 | 3906.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 4020.00 | 4022.09 | 4002.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:30:00 | 4060.40 | 4040.67 | 4030.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 11:30:00 | 4059.30 | 4047.70 | 4035.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:15:00 | 4059.90 | 4047.70 | 4035.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 4060.00 | 4052.85 | 4042.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 4055.40 | 4053.36 | 4043.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 4078.80 | 4057.49 | 4047.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 4080.50 | 4098.18 | 4098.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 4080.50 | 4098.18 | 4098.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 4057.40 | 4087.67 | 4093.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 4016.60 | 4009.64 | 4032.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 4016.60 | 4009.64 | 4032.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 4040.00 | 4007.62 | 4019.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 4040.00 | 4007.62 | 4019.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 4013.50 | 4008.79 | 4018.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 4001.00 | 4004.91 | 4016.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 3954.20 | 3925.19 | 3922.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 3954.20 | 3925.19 | 3922.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 3965.00 | 3944.75 | 3937.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 3913.00 | 3940.41 | 3936.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 3913.00 | 3940.41 | 3936.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 3913.00 | 3940.41 | 3936.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 3913.00 | 3940.41 | 3936.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 3888.00 | 3929.92 | 3932.26 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 3950.90 | 3930.25 | 3929.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 4010.70 | 3960.42 | 3946.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 3969.80 | 3972.50 | 3959.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 3969.80 | 3972.50 | 3959.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 3969.80 | 3972.50 | 3959.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 3960.00 | 3972.50 | 3959.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3947.60 | 3965.81 | 3958.36 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 3922.10 | 3953.13 | 3953.65 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 3980.00 | 3950.57 | 3949.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 3984.80 | 3960.99 | 3954.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 14:15:00 | 3958.60 | 3962.69 | 3957.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 3958.60 | 3962.69 | 3957.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 3958.60 | 3962.69 | 3957.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 3958.60 | 3962.69 | 3957.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 3958.00 | 3961.75 | 3957.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 3975.40 | 3961.75 | 3957.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 3946.50 | 3959.99 | 3958.02 | SL hit (close<static) qty=1.00 sl=3948.70 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 3941.90 | 3954.28 | 3955.78 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 3984.60 | 3958.08 | 3957.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 4003.40 | 3967.14 | 3961.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 13:15:00 | 3971.10 | 3974.77 | 3967.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 13:15:00 | 3971.10 | 3974.77 | 3967.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 3971.10 | 3974.77 | 3967.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 3971.10 | 3974.77 | 3967.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 3978.00 | 3996.84 | 3987.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 3978.00 | 3996.84 | 3987.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 4357.20 | 4366.45 | 4335.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 4333.60 | 4366.45 | 4335.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 4345.00 | 4357.93 | 4337.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 4345.00 | 4357.93 | 4337.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 4351.00 | 4356.54 | 4338.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:45:00 | 4347.10 | 4356.54 | 4338.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 4351.00 | 4352.79 | 4339.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 4373.30 | 4352.79 | 4339.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 4327.60 | 4363.03 | 4355.61 | SL hit (close<static) qty=1.00 sl=4335.50 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 4307.30 | 4342.44 | 4346.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 4261.80 | 4314.98 | 4330.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 4317.40 | 4309.71 | 4324.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 14:00:00 | 4317.40 | 4309.71 | 4324.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 4321.30 | 4312.03 | 4324.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:45:00 | 4315.70 | 4312.03 | 4324.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 4401.40 | 4330.31 | 4330.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:15:00 | 4486.50 | 4330.31 | 4330.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 4292.80 | 4322.81 | 4327.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 4256.00 | 4322.81 | 4327.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:45:00 | 4268.80 | 4288.32 | 4303.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 4344.00 | 4311.55 | 4311.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 4344.00 | 4311.55 | 4311.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 4382.80 | 4345.73 | 4330.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 4385.00 | 4388.94 | 4369.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:15:00 | 4434.90 | 4388.94 | 4369.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 4420.60 | 4395.28 | 4373.78 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 4313.20 | 4369.73 | 4372.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 10:15:00 | 4296.00 | 4354.99 | 4365.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 4331.40 | 4316.57 | 4337.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 4331.40 | 4316.57 | 4337.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 4331.40 | 4316.57 | 4337.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 4331.40 | 4316.57 | 4337.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 4298.60 | 4312.98 | 4333.83 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 14:15:00 | 4387.20 | 4348.69 | 4345.80 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 4299.20 | 4343.64 | 4344.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 4284.30 | 4331.77 | 4338.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 4261.90 | 4257.13 | 4287.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:30:00 | 4258.90 | 4257.13 | 4287.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 4348.20 | 4275.46 | 4285.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 4352.80 | 4275.46 | 4285.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 4335.00 | 4287.37 | 4290.14 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 4332.00 | 4296.30 | 4293.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 12:15:00 | 4370.50 | 4311.14 | 4300.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 4344.30 | 4345.50 | 4323.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:15:00 | 4330.10 | 4345.50 | 4323.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 4337.80 | 4343.96 | 4324.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:45:00 | 4353.00 | 4342.39 | 4327.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:45:00 | 4355.00 | 4346.84 | 4333.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 4298.00 | 4336.52 | 4334.70 | SL hit (close<static) qty=1.00 sl=4315.60 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 15:15:00 | 4295.70 | 4328.36 | 4331.15 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 4401.60 | 4333.47 | 4330.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 4418.40 | 4350.46 | 4338.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 12:15:00 | 4525.60 | 4527.48 | 4496.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:00:00 | 4525.60 | 4527.48 | 4496.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 4461.70 | 4508.22 | 4497.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 13:30:00 | 4521.60 | 4497.90 | 4494.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:30:00 | 4545.90 | 4508.00 | 4500.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 4468.00 | 4498.56 | 4498.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 4468.00 | 4498.56 | 4498.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 4453.90 | 4485.08 | 4492.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 11:15:00 | 4474.90 | 4474.73 | 4484.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:45:00 | 4471.70 | 4474.73 | 4484.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 4464.30 | 4472.64 | 4482.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:30:00 | 4480.30 | 4472.64 | 4482.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 4481.50 | 4472.18 | 4480.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 4481.50 | 4472.18 | 4480.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 4457.70 | 4469.29 | 4478.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 4510.80 | 4469.29 | 4478.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 4493.80 | 4474.19 | 4480.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 4520.90 | 4474.19 | 4480.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 4494.00 | 4478.15 | 4481.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:45:00 | 4494.80 | 4478.15 | 4481.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 4471.00 | 4476.70 | 4480.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:30:00 | 4471.10 | 4476.70 | 4480.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 4466.10 | 4472.21 | 4477.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 4419.30 | 4469.99 | 4475.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 4479.80 | 4471.83 | 4473.45 | SL hit (close>static) qty=1.00 sl=4479.10 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 4525.10 | 4482.19 | 4477.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 4528.20 | 4491.40 | 4482.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 12:15:00 | 4525.40 | 4536.75 | 4518.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 12:15:00 | 4525.40 | 4536.75 | 4518.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 4525.40 | 4536.75 | 4518.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 4525.40 | 4536.75 | 4518.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 4520.00 | 4532.35 | 4520.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 4540.40 | 4532.35 | 4520.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 4552.50 | 4537.68 | 4524.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 11:00:00 | 4543.60 | 4538.86 | 4525.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 4515.30 | 4557.54 | 4545.92 | SL hit (close<static) qty=1.00 sl=4516.10 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 4500.20 | 4537.97 | 4540.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 4490.00 | 4515.33 | 4528.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 4503.80 | 4503.23 | 4517.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 4503.80 | 4503.23 | 4517.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 4503.80 | 4503.23 | 4517.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 4518.10 | 4503.23 | 4517.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 4527.50 | 4508.09 | 4518.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 4527.50 | 4508.09 | 4518.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 4514.90 | 4509.45 | 4518.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:45:00 | 4503.10 | 4508.40 | 4516.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 4495.90 | 4509.43 | 4515.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 4523.90 | 4445.72 | 4437.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 4523.90 | 4445.72 | 4437.28 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 4416.50 | 4472.56 | 4476.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 4399.30 | 4457.90 | 4469.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 4376.10 | 4374.12 | 4404.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 12:00:00 | 4376.10 | 4374.12 | 4404.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 4441.20 | 4389.06 | 4406.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 4441.20 | 4389.06 | 4406.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 4401.00 | 4391.44 | 4405.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 4369.00 | 4393.22 | 4405.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 4386.80 | 4391.96 | 4400.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 4390.10 | 4392.86 | 4399.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 4455.80 | 4401.63 | 4401.95 | SL hit (close>static) qty=1.00 sl=4443.20 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 4443.10 | 4409.92 | 4405.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 11:15:00 | 4478.90 | 4441.21 | 4426.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 4449.30 | 4463.09 | 4446.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 4449.30 | 4463.09 | 4446.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 4449.30 | 4463.09 | 4446.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 4449.30 | 4463.09 | 4446.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 4448.00 | 4460.07 | 4446.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 4448.00 | 4460.07 | 4446.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 4450.00 | 4458.06 | 4447.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:45:00 | 4465.40 | 4463.55 | 4451.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 4425.20 | 4467.00 | 4458.38 | SL hit (close<static) qty=1.00 sl=4443.50 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 4376.60 | 4448.92 | 4450.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 4313.00 | 4421.74 | 4438.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 4214.40 | 4173.14 | 4227.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 4214.40 | 4173.14 | 4227.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 4214.40 | 4173.14 | 4227.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:30:00 | 4137.30 | 4158.70 | 4207.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:30:00 | 4140.40 | 4141.17 | 4182.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:00:00 | 4145.20 | 4141.17 | 4182.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:30:00 | 4147.80 | 4141.14 | 4178.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:15:00 | 3937.94 | 4070.30 | 4124.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:15:00 | 3940.41 | 4070.30 | 4124.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 3930.43 | 4037.66 | 4105.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 3933.38 | 4037.66 | 4105.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 3986.50 | 3968.06 | 4015.73 | SL hit (close>ema200) qty=0.50 sl=3968.06 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 4064.60 | 4014.82 | 4012.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 4067.90 | 4025.44 | 4017.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 4031.50 | 4043.55 | 4031.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 4031.50 | 4043.55 | 4031.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 4031.50 | 4043.55 | 4031.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 4035.30 | 4043.55 | 4031.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 4030.70 | 4040.98 | 4031.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 4030.70 | 4040.98 | 4031.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 4019.30 | 4036.64 | 4030.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 4010.40 | 4036.64 | 4030.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 4017.70 | 4032.86 | 4029.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:45:00 | 4026.10 | 4030.88 | 4028.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 14:45:00 | 4037.90 | 4030.65 | 4028.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 4030.20 | 4027.74 | 4027.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 4025.20 | 4027.23 | 4027.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 4025.20 | 4027.23 | 4027.46 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 12:15:00 | 4057.80 | 4033.23 | 4030.10 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 4002.10 | 4023.31 | 4026.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 3995.80 | 4017.81 | 4023.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 4026.60 | 4009.30 | 4017.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 4026.60 | 4009.30 | 4017.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 4026.60 | 4009.30 | 4017.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 4026.60 | 4009.30 | 4017.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 3987.70 | 4004.98 | 4014.35 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 4035.70 | 4018.28 | 4017.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 4066.10 | 4027.84 | 4021.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 4026.10 | 4034.72 | 4026.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 4026.10 | 4034.72 | 4026.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 4026.10 | 4034.72 | 4026.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 4026.10 | 4034.72 | 4026.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 4021.00 | 4031.98 | 4025.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:00:00 | 4021.00 | 4031.98 | 4025.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 3973.90 | 4020.36 | 4021.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 3949.50 | 4001.71 | 4012.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 3994.80 | 3947.32 | 3966.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 3994.80 | 3947.32 | 3966.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3994.80 | 3947.32 | 3966.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 4010.90 | 3947.32 | 3966.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 4025.10 | 3962.88 | 3971.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 4025.10 | 3962.88 | 3971.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 4008.90 | 3979.17 | 3977.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 4020.90 | 3990.63 | 3983.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 3996.50 | 3997.63 | 3989.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 3996.50 | 3997.63 | 3989.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 3996.50 | 3997.63 | 3989.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 4020.00 | 3997.63 | 3989.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 12:45:00 | 4031.00 | 4004.84 | 3994.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:45:00 | 4018.30 | 4072.41 | 4062.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 4038.10 | 4072.41 | 4062.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 3996.60 | 4051.70 | 4054.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 3996.60 | 4051.70 | 4054.66 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 4083.00 | 4057.41 | 4055.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 4205.10 | 4086.94 | 4068.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 4161.70 | 4169.39 | 4123.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 15:00:00 | 4161.70 | 4169.39 | 4123.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 4174.90 | 4170.77 | 4132.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 11:45:00 | 4196.10 | 4177.23 | 4142.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:00:00 | 4199.00 | 4181.58 | 4147.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 14:00:00 | 4202.10 | 4185.69 | 4152.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 4211.20 | 4190.79 | 4157.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 4192.00 | 4192.51 | 4164.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 10:15:00 | 4244.60 | 4192.51 | 4164.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-18 09:15:00 | 4615.71 | 4584.20 | 4532.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 4888.50 | 4906.35 | 4907.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 4830.00 | 4891.08 | 4900.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 4720.60 | 4680.63 | 4748.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 4720.60 | 4680.63 | 4748.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 4748.40 | 4700.96 | 4746.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 4748.40 | 4700.96 | 4746.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 4739.20 | 4708.61 | 4745.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:45:00 | 4742.10 | 4708.61 | 4745.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 4779.10 | 4722.71 | 4748.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 4779.10 | 4722.71 | 4748.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 4804.10 | 4738.99 | 4753.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 4825.10 | 4738.99 | 4753.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 4862.00 | 4777.10 | 4769.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 4910.90 | 4817.99 | 4790.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 4801.80 | 4818.93 | 4795.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 4801.80 | 4818.93 | 4795.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 4801.80 | 4818.93 | 4795.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 4801.80 | 4818.93 | 4795.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 4824.00 | 4819.94 | 4798.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 4653.90 | 4819.94 | 4798.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 4666.00 | 4789.15 | 4786.14 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 4664.20 | 4764.16 | 4775.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 4590.20 | 4651.21 | 4684.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 4659.20 | 4652.80 | 4682.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 11:00:00 | 4659.20 | 4652.80 | 4682.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 4718.70 | 4665.98 | 4685.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 4718.70 | 4665.98 | 4685.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 4712.80 | 4675.35 | 4688.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:30:00 | 4720.30 | 4675.35 | 4688.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 4755.30 | 4702.80 | 4699.09 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 4666.40 | 4697.29 | 4698.13 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 11:15:00 | 4708.10 | 4699.45 | 4699.04 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 4651.00 | 4689.76 | 4694.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 4628.70 | 4673.59 | 4686.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 4683.90 | 4675.65 | 4686.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 09:15:00 | 4636.70 | 4675.65 | 4686.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 4624.00 | 4665.32 | 4680.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 4659.20 | 4665.32 | 4680.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 4626.00 | 4593.11 | 4625.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 4635.30 | 4593.11 | 4625.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 4602.20 | 4594.93 | 4623.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 4593.80 | 4594.93 | 4623.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:45:00 | 4600.00 | 4596.68 | 4621.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 4597.70 | 4596.68 | 4621.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 4597.00 | 4603.74 | 4620.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 4644.20 | 4611.37 | 4619.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 4649.10 | 4611.37 | 4619.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 4645.30 | 4618.16 | 4622.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 4695.20 | 4633.57 | 4628.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 4695.20 | 4633.57 | 4628.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 4744.50 | 4655.75 | 4639.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4592.10 | 4668.93 | 4654.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4592.10 | 4668.93 | 4654.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4592.10 | 4668.93 | 4654.01 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 4556.30 | 4629.06 | 4637.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 4532.80 | 4595.73 | 4619.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 4630.00 | 4575.98 | 4602.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 4630.00 | 4575.98 | 4602.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 4630.00 | 4575.98 | 4602.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 4644.20 | 4575.98 | 4602.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 4626.50 | 4586.08 | 4604.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 4620.10 | 4586.08 | 4604.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 4597.40 | 4588.34 | 4603.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 4579.50 | 4588.34 | 4603.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 4512.80 | 4607.78 | 4609.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 4628.50 | 4573.07 | 4567.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 4628.50 | 4573.07 | 4567.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 4752.60 | 4629.63 | 4597.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 4601.00 | 4690.79 | 4654.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 4601.00 | 4690.79 | 4654.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 4601.00 | 4690.79 | 4654.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 4605.00 | 4690.79 | 4654.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 4601.30 | 4672.89 | 4650.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:30:00 | 4618.60 | 4666.30 | 4649.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 4582.20 | 4638.80 | 4641.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4582.20 | 4638.80 | 4641.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 4527.20 | 4616.48 | 4630.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4615.00 | 4559.61 | 4590.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 4615.00 | 4559.61 | 4590.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4615.00 | 4559.61 | 4590.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 4625.40 | 4559.61 | 4590.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4560.00 | 4559.69 | 4587.35 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 4618.00 | 4600.83 | 4600.44 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 4503.40 | 4581.35 | 4591.62 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 4654.90 | 4594.15 | 4590.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 4681.50 | 4620.62 | 4604.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 4639.40 | 4661.26 | 4636.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 4639.40 | 4661.26 | 4636.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 4639.40 | 4661.26 | 4636.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 4781.30 | 4640.39 | 4634.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 09:15:00 | 5259.43 | 5191.00 | 5138.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 5144.60 | 5175.13 | 5178.45 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 14:15:00 | 5223.80 | 5180.41 | 5179.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 15:15:00 | 5241.80 | 5192.69 | 5185.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 5237.00 | 5238.19 | 5218.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 10:15:00 | 5224.80 | 5238.19 | 5218.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 5224.00 | 5235.35 | 5218.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:30:00 | 5219.10 | 5235.35 | 5218.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 5231.30 | 5234.54 | 5219.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 5226.40 | 5234.54 | 5219.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 5208.00 | 5229.23 | 5218.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 5208.00 | 5229.23 | 5218.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 5226.60 | 5228.71 | 5219.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 14:15:00 | 5231.20 | 5228.71 | 5219.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 5244.00 | 5266.08 | 5267.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 5244.00 | 5266.08 | 5267.88 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 5293.50 | 5268.50 | 5267.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 5362.50 | 5320.76 | 5299.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 5391.00 | 5408.85 | 5367.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 5391.00 | 5408.85 | 5367.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-26 11:45:00 | 2924.00 | 2025-05-27 14:15:00 | 2973.70 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-26 13:45:00 | 2931.60 | 2025-05-27 14:15:00 | 2973.70 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-05-27 09:15:00 | 2927.90 | 2025-05-27 14:15:00 | 2973.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-05-27 09:45:00 | 2930.90 | 2025-05-27 14:15:00 | 2973.70 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-29 09:15:00 | 3061.80 | 2025-06-04 09:15:00 | 3367.98 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-18 11:30:00 | 3268.70 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-06-18 13:45:00 | 3274.10 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-06-18 15:15:00 | 3273.50 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-06-19 10:30:00 | 3273.00 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-06-20 12:15:00 | 3277.10 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-06-20 14:00:00 | 3280.00 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-20 15:00:00 | 3267.30 | 2025-06-24 09:15:00 | 3322.60 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-30 09:15:00 | 3384.00 | 2025-07-01 15:15:00 | 3352.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-11 09:15:00 | 3797.10 | 2025-08-19 10:15:00 | 3749.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-09-15 09:30:00 | 4060.40 | 2025-09-19 14:15:00 | 4080.50 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-09-15 11:30:00 | 4059.30 | 2025-09-19 14:15:00 | 4080.50 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-09-15 12:15:00 | 4059.90 | 2025-09-19 14:15:00 | 4080.50 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-09-16 09:15:00 | 4060.00 | 2025-09-19 14:15:00 | 4080.50 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-09-16 13:00:00 | 4078.80 | 2025-09-19 14:15:00 | 4080.50 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-09-25 11:30:00 | 4001.00 | 2025-10-06 09:15:00 | 3954.20 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2025-10-16 09:15:00 | 3975.40 | 2025-10-16 11:15:00 | 3946.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-11-03 09:15:00 | 4373.30 | 2025-11-04 09:15:00 | 4327.60 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-11-07 11:15:00 | 4256.00 | 2025-11-10 14:15:00 | 4344.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-11-10 11:45:00 | 4268.80 | 2025-11-10 14:15:00 | 4344.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-11-21 12:45:00 | 4353.00 | 2025-11-24 14:15:00 | 4298.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-24 09:45:00 | 4355.00 | 2025-11-24 14:15:00 | 4298.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-12-03 13:30:00 | 4521.60 | 2025-12-04 13:15:00 | 4468.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-04 09:30:00 | 4545.90 | 2025-12-04 13:15:00 | 4468.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-09 09:15:00 | 4419.30 | 2025-12-09 14:15:00 | 4479.80 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-12 09:15:00 | 4540.40 | 2025-12-15 10:15:00 | 4515.30 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-12-12 09:45:00 | 4552.50 | 2025-12-15 10:15:00 | 4515.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-12-12 11:00:00 | 4543.60 | 2025-12-15 10:15:00 | 4515.30 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-15 12:00:00 | 4541.20 | 2025-12-16 09:15:00 | 4500.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-12-17 12:45:00 | 4503.10 | 2025-12-22 11:15:00 | 4523.90 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-12-18 09:15:00 | 4495.90 | 2025-12-22 11:15:00 | 4523.90 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-30 09:15:00 | 4369.00 | 2025-12-31 09:15:00 | 4455.80 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-12-30 13:15:00 | 4386.80 | 2025-12-31 09:15:00 | 4455.80 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-12-30 15:00:00 | 4390.10 | 2025-12-31 09:15:00 | 4455.80 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-01-02 14:45:00 | 4465.40 | 2026-01-05 11:15:00 | 4425.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-08 12:30:00 | 4137.30 | 2026-01-12 10:15:00 | 3937.94 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2026-01-09 09:30:00 | 4140.40 | 2026-01-12 10:15:00 | 3940.41 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2026-01-09 10:00:00 | 4145.20 | 2026-01-12 11:15:00 | 3930.43 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2026-01-09 10:30:00 | 4147.80 | 2026-01-12 11:15:00 | 3933.38 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-01-08 12:30:00 | 4137.30 | 2026-01-13 14:15:00 | 3986.50 | STOP_HIT | 0.50 | 3.64% |
| SELL | retest2 | 2026-01-09 09:30:00 | 4140.40 | 2026-01-13 14:15:00 | 3986.50 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2026-01-09 10:00:00 | 4145.20 | 2026-01-13 14:15:00 | 3986.50 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2026-01-09 10:30:00 | 4147.80 | 2026-01-13 14:15:00 | 3986.50 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2026-01-16 09:15:00 | 4015.60 | 2026-01-16 10:15:00 | 4064.60 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-19 13:45:00 | 4026.10 | 2026-01-20 09:15:00 | 4025.20 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2026-01-19 14:45:00 | 4037.90 | 2026-01-20 09:15:00 | 4025.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-01-20 09:15:00 | 4030.20 | 2026-01-20 09:15:00 | 4025.20 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2026-01-29 11:15:00 | 4020.00 | 2026-02-02 10:15:00 | 3996.60 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-01-29 12:45:00 | 4031.00 | 2026-02-02 10:15:00 | 3996.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-01 14:45:00 | 4018.30 | 2026-02-02 10:15:00 | 3996.60 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-02-01 15:15:00 | 4038.10 | 2026-02-02 10:15:00 | 3996.60 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-04 11:45:00 | 4196.10 | 2026-02-18 09:15:00 | 4615.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 13:00:00 | 4199.00 | 2026-02-18 09:15:00 | 4618.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 14:00:00 | 4202.10 | 2026-02-18 09:15:00 | 4622.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 15:00:00 | 4211.20 | 2026-02-18 09:15:00 | 4632.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-05 10:15:00 | 4244.60 | 2026-02-18 09:15:00 | 4669.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-17 11:15:00 | 4593.80 | 2026-03-18 11:15:00 | 4695.20 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-03-17 11:45:00 | 4600.00 | 2026-03-18 11:15:00 | 4695.20 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-03-17 12:15:00 | 4597.70 | 2026-03-18 11:15:00 | 4695.20 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-03-17 14:15:00 | 4597.00 | 2026-03-18 11:15:00 | 4695.20 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-03-20 12:15:00 | 4579.50 | 2026-03-24 12:15:00 | 4628.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-03-23 09:15:00 | 4512.80 | 2026-03-24 12:15:00 | 4628.50 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-03-27 11:30:00 | 4618.60 | 2026-03-30 09:15:00 | 4582.20 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-04-08 09:15:00 | 4781.30 | 2026-04-21 09:15:00 | 5259.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-28 14:15:00 | 5231.20 | 2026-05-05 11:15:00 | 5244.00 | STOP_HIT | 1.00 | 0.24% |
