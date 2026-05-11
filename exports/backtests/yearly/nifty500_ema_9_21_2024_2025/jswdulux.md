# JSW Dulux Ltd. (JSWDULUX)

## Backtest Summary

- **Window:** 2026-03-16 09:15:00 → 2026-05-08 15:15:00 (245 bars)
- **Last close:** 2950.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 9 |
| ALERT2 | 7 |
| ALERT2_SKIP | 5 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 9
- **Target hits / Stop hits / Partials:** 0 / 10 / 0
- **Avg / median % per leg:** -1.57% / -1.57%
- **Sum % (uncompounded):** -15.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.55% | -9.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.51% | -7.5% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.59% | -6.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.59% | -6.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| retest2 (combined) | 9 | 1 | 11.1% | 0 | 9 | 0 | -1.54% | -13.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 2796.90 | 2860.04 | 2868.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 2794.30 | 2846.89 | 2861.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 2835.40 | 2829.93 | 2845.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 13:00:00 | 2835.40 | 2829.93 | 2845.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 2823.40 | 2828.63 | 2843.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 2810.00 | 2828.63 | 2843.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 14:15:00 | 2895.80 | 2842.06 | 2848.20 | SL hit (close>static) qty=1.00 sl=2854.20 alert=retest2 |

### Cycle 2 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 2856.40 | 2832.91 | 2832.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 2914.40 | 2856.63 | 2844.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2859.70 | 2893.55 | 2873.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2859.70 | 2893.55 | 2873.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2859.70 | 2893.55 | 2873.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 2859.70 | 2893.55 | 2873.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 2885.70 | 2891.98 | 2874.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:30:00 | 2915.00 | 2901.20 | 2880.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 11:00:00 | 2898.40 | 2916.53 | 2900.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 2838.00 | 2896.38 | 2895.47 | SL hit (close<static) qty=1.00 sl=2859.70 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 2818.50 | 2880.80 | 2888.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 2799.00 | 2852.28 | 2869.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 12:15:00 | 2955.50 | 2862.32 | 2868.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 12:15:00 | 2955.50 | 2862.32 | 2868.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 2955.50 | 2862.32 | 2868.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:00:00 | 2955.50 | 2862.32 | 2868.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 3044.30 | 2898.71 | 2884.88 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 2936.40 | 2969.66 | 2971.94 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 15:15:00 | 2976.50 | 2971.45 | 2971.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 3015.00 | 2985.07 | 2977.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 15:15:00 | 2994.00 | 2995.21 | 2986.55 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:15:00 | 3019.30 | 2995.21 | 2986.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 2966.10 | 2989.39 | 2984.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 2966.10 | 2989.39 | 2984.69 | SL hit (close<ema400) qty=1.00 sl=2984.69 alert=retest1 |

### Cycle 7 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 2945.80 | 2975.36 | 2978.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 12:15:00 | 2943.90 | 2969.07 | 2975.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 2979.60 | 2964.46 | 2970.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 2979.60 | 2964.46 | 2970.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 2979.60 | 2964.46 | 2970.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 10:15:00 | 2951.40 | 2964.46 | 2970.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 2947.70 | 2914.07 | 2912.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 14:15:00 | 2947.70 | 2914.07 | 2912.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 11:15:00 | 2959.00 | 2934.82 | 2923.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 2946.30 | 2947.98 | 2936.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 10:15:00 | 2946.30 | 2947.98 | 2936.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 2946.30 | 2947.98 | 2936.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:45:00 | 2946.00 | 2947.98 | 2936.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 2958.00 | 2983.54 | 2975.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 2955.50 | 2983.54 | 2975.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 2981.60 | 2983.15 | 2975.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:30:00 | 2961.00 | 2983.15 | 2975.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 2975.00 | 2981.52 | 2975.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 2982.80 | 2981.78 | 2976.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 11:00:00 | 2997.10 | 2984.84 | 2978.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:00:00 | 2987.00 | 2986.50 | 2981.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 15:15:00 | 2961.00 | 2981.40 | 2979.69 | SL hit (close<static) qty=1.00 sl=2963.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 2938.50 | 2972.82 | 2975.95 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 2969.80 | 2964.36 | 2963.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 15:15:00 | 2979.00 | 2969.34 | 2966.31 | Break + close above crossover candle high |

### Cycle 11 — SELL (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 09:15:00 | 2933.50 | 2962.18 | 2963.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 10:15:00 | 2925.40 | 2954.82 | 2959.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 2973.60 | 2948.46 | 2954.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 2973.60 | 2948.46 | 2954.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 2973.60 | 2948.46 | 2954.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 2973.60 | 2948.46 | 2954.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 2989.20 | 2956.61 | 2957.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 2984.20 | 2956.61 | 2957.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 2969.30 | 2959.15 | 2958.44 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 13:15:00 | 2954.00 | 2957.43 | 2957.81 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 2969.00 | 2959.74 | 2958.83 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 2950.00 | 2957.15 | 2957.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 2941.10 | 2951.44 | 2954.85 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-03-20 14:15:00 | 2810.00 | 2026-03-20 14:15:00 | 2895.80 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2026-03-23 09:15:00 | 2802.40 | 2026-03-23 15:15:00 | 2855.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-03-24 09:45:00 | 2812.30 | 2026-03-24 12:15:00 | 2856.40 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-03-27 11:30:00 | 2915.00 | 2026-03-30 14:15:00 | 2838.00 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-03-30 11:00:00 | 2898.40 | 2026-03-30 14:15:00 | 2838.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest1 | 2026-04-16 09:15:00 | 3019.30 | 2026-04-16 09:15:00 | 2966.10 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-04-17 10:15:00 | 2951.40 | 2026-04-22 14:15:00 | 2947.70 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2026-04-29 10:00:00 | 2982.80 | 2026-04-29 15:15:00 | 2961.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-04-29 11:00:00 | 2997.10 | 2026-04-29 15:15:00 | 2961.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-04-29 15:00:00 | 2987.00 | 2026-04-29 15:15:00 | 2961.00 | STOP_HIT | 1.00 | -0.87% |
