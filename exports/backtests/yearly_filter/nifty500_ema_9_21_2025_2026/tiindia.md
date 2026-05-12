# Tube Investments of India Ltd. (TIINDIA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 3032.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 83 |
| ALERT1 | 55 |
| ALERT2 | 54 |
| ALERT2_SKIP | 25 |
| ALERT3 | 147 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 60 |
| PARTIAL | 16 |
| TARGET_HIT | 9 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 79 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 36
- **Target hits / Stop hits / Partials:** 9 / 54 / 16
- **Avg / median % per leg:** 2.46% / 3.20%
- **Sum % (uncompounded):** 194.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 16 | 59.3% | 9 | 15 | 3 | 3.92% | 105.9% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.91% | 29.5% |
| BUY @ 3rd Alert (retest2) | 21 | 10 | 47.6% | 9 | 12 | 0 | 3.64% | 76.4% |
| SELL (all) | 52 | 27 | 51.9% | 0 | 39 | 13 | 1.70% | 88.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 52 | 27 | 51.9% | 0 | 39 | 13 | 1.70% | 88.2% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.91% | 29.5% |
| retest2 (combined) | 73 | 37 | 50.7% | 9 | 51 | 13 | 2.25% | 164.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 2953.40 | 2901.85 | 2900.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 2968.90 | 2930.75 | 2915.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 3028.00 | 3029.87 | 3001.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 3028.00 | 3029.87 | 3001.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 2990.00 | 3019.81 | 3001.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:00:00 | 2990.00 | 3019.81 | 3001.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 2960.90 | 3008.03 | 2998.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 2960.90 | 3008.03 | 2998.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 14:15:00 | 2902.00 | 2986.82 | 2989.40 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 3034.50 | 2988.97 | 2987.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 3056.40 | 3002.46 | 2993.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 3079.90 | 3115.47 | 3079.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 3079.90 | 3115.47 | 3079.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 3079.90 | 3115.47 | 3079.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 3079.90 | 3115.47 | 3079.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 3052.30 | 3102.84 | 3076.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 3052.30 | 3102.84 | 3076.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 3041.00 | 3090.47 | 3073.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:30:00 | 3051.40 | 3090.47 | 3073.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 2994.50 | 3061.76 | 3062.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 2985.60 | 3031.54 | 3047.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 3033.20 | 3014.80 | 3028.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 3033.20 | 3014.80 | 3028.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 3033.20 | 3014.80 | 3028.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 3033.20 | 3014.80 | 3028.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 3005.20 | 3012.88 | 3026.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 2990.00 | 3012.88 | 3026.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:45:00 | 2990.00 | 3003.08 | 3019.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 3046.00 | 3011.24 | 3013.31 | SL hit (close>static) qty=1.00 sl=3036.30 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 3044.40 | 3017.87 | 3016.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 12:15:00 | 3059.00 | 3035.57 | 3026.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 3010.00 | 3037.74 | 3031.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 3010.00 | 3037.74 | 3031.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 3010.00 | 3037.74 | 3031.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 3010.00 | 3037.74 | 3031.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 3005.20 | 3031.23 | 3028.73 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 3010.00 | 3026.98 | 3027.03 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 3034.40 | 3028.47 | 3027.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 3046.90 | 3032.13 | 3029.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 3029.50 | 3039.05 | 3034.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 3029.50 | 3039.05 | 3034.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 3029.50 | 3039.05 | 3034.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 3029.50 | 3039.05 | 3034.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 3032.40 | 3037.72 | 3034.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:30:00 | 3029.60 | 3037.72 | 3034.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 3016.70 | 3033.51 | 3032.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:15:00 | 3016.80 | 3033.51 | 3032.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 3016.80 | 3030.17 | 3031.33 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 3041.80 | 3032.76 | 3032.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 3056.20 | 3037.45 | 3034.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 3061.30 | 3063.89 | 3053.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 15:00:00 | 3061.30 | 3063.89 | 3053.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 3035.10 | 3058.13 | 3051.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 3055.50 | 3058.13 | 3051.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 3061.10 | 3058.73 | 3052.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 3077.40 | 3054.44 | 3052.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 3040.00 | 3051.35 | 3051.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 3040.00 | 3051.35 | 3051.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 3036.50 | 3048.38 | 3050.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 3055.30 | 3043.96 | 3047.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 3055.30 | 3043.96 | 3047.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 3055.30 | 3043.96 | 3047.02 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 3080.90 | 3051.35 | 3050.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 3104.00 | 3075.91 | 3064.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 14:15:00 | 3082.30 | 3082.31 | 3072.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 3061.50 | 3077.91 | 3072.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 3061.50 | 3077.91 | 3072.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 3061.50 | 3077.91 | 3072.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 3047.80 | 3071.89 | 3070.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 3047.80 | 3071.89 | 3070.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 3030.50 | 3063.61 | 3066.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 12:15:00 | 3025.80 | 3050.81 | 3058.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 3083.40 | 3052.54 | 3056.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 3083.40 | 3052.54 | 3056.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 3083.40 | 3052.54 | 3056.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 3083.40 | 3052.54 | 3056.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 3073.20 | 3056.67 | 3057.84 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 3080.40 | 3061.42 | 3059.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 13:15:00 | 3087.60 | 3069.05 | 3063.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 3059.90 | 3077.75 | 3070.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 10:15:00 | 3059.90 | 3077.75 | 3070.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 3059.90 | 3077.75 | 3070.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 3059.90 | 3077.75 | 3070.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3078.40 | 3077.88 | 3071.24 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 3035.40 | 3061.51 | 3064.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 3012.00 | 3045.44 | 3056.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 3001.80 | 2960.26 | 2981.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 3001.80 | 2960.26 | 2981.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 3001.80 | 2960.26 | 2981.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 3001.80 | 2960.26 | 2981.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 3000.70 | 2968.35 | 2983.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:30:00 | 3010.80 | 2968.35 | 2983.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 2980.10 | 2981.18 | 2985.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 2993.00 | 2981.18 | 2985.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 2950.20 | 2974.98 | 2982.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 2944.90 | 2969.02 | 2979.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 2922.50 | 2906.09 | 2905.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 2922.50 | 2906.09 | 2905.54 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 15:15:00 | 2901.20 | 2905.11 | 2905.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 09:15:00 | 2900.80 | 2904.25 | 2904.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 2900.20 | 2882.25 | 2890.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 2900.20 | 2882.25 | 2890.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2900.20 | 2882.25 | 2890.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 2900.20 | 2882.25 | 2890.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 2906.00 | 2887.00 | 2891.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 2906.00 | 2887.00 | 2891.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 2906.30 | 2890.86 | 2892.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 2906.30 | 2890.86 | 2892.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 2866.10 | 2872.40 | 2882.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:30:00 | 2885.00 | 2872.40 | 2882.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 2885.60 | 2875.04 | 2882.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 2885.60 | 2875.04 | 2882.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 2840.60 | 2868.15 | 2878.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:15:00 | 2836.30 | 2868.15 | 2878.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 13:45:00 | 2835.50 | 2857.52 | 2871.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 2917.00 | 2869.31 | 2873.44 | SL hit (close>static) qty=1.00 sl=2888.40 alert=retest2 |

### Cycle 17 — BUY (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 10:15:00 | 2916.00 | 2878.65 | 2877.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 2951.90 | 2893.30 | 2884.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 12:15:00 | 3081.60 | 3095.66 | 3064.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 13:00:00 | 3081.60 | 3095.66 | 3064.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 3063.80 | 3085.87 | 3069.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 3063.80 | 3085.87 | 3069.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 3051.10 | 3078.92 | 3068.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 3051.10 | 3078.92 | 3068.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 3005.00 | 3056.87 | 3059.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 2986.00 | 3042.69 | 3052.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 13:15:00 | 2981.00 | 2964.49 | 2984.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 14:00:00 | 2981.00 | 2964.49 | 2984.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2993.90 | 2970.37 | 2985.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 2993.90 | 2970.37 | 2985.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 2979.10 | 2972.12 | 2984.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 2999.00 | 2972.12 | 2984.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2972.00 | 2972.10 | 2983.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 2994.80 | 2972.10 | 2983.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2910.00 | 2947.12 | 2964.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 2904.40 | 2936.16 | 2958.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 12:45:00 | 2906.80 | 2922.04 | 2947.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 2904.60 | 2922.49 | 2941.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 10:45:00 | 2904.90 | 2917.75 | 2935.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 2915.60 | 2903.82 | 2918.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 2914.60 | 2903.82 | 2918.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 2911.20 | 2905.30 | 2918.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:15:00 | 2900.90 | 2905.30 | 2918.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 2920.10 | 2890.97 | 2887.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 2920.10 | 2890.97 | 2887.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 2931.80 | 2899.13 | 2891.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 2937.10 | 2937.64 | 2919.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 15:00:00 | 2937.10 | 2937.64 | 2919.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 2925.00 | 2939.85 | 2931.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 2935.90 | 2939.85 | 2931.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 2937.00 | 2939.28 | 2932.02 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 2920.00 | 2928.67 | 2929.56 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 2952.00 | 2933.33 | 2931.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 2981.80 | 2946.33 | 2939.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 2946.90 | 2956.19 | 2948.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 14:15:00 | 2946.90 | 2956.19 | 2948.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 2946.90 | 2956.19 | 2948.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 2946.90 | 2956.19 | 2948.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 2952.00 | 2955.35 | 2948.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 2953.50 | 2955.35 | 2948.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2950.40 | 2954.36 | 2948.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 2975.60 | 2956.47 | 2950.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 2919.90 | 2948.91 | 2949.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 2919.90 | 2948.91 | 2949.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 2901.80 | 2933.20 | 2940.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2902.90 | 2902.26 | 2918.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 2902.90 | 2902.26 | 2918.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2870.00 | 2843.34 | 2867.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 2870.00 | 2843.34 | 2867.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2853.30 | 2845.33 | 2866.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 2864.20 | 2845.33 | 2866.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 2870.70 | 2850.41 | 2866.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 2873.00 | 2850.41 | 2866.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 2873.20 | 2854.97 | 2867.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 2884.00 | 2854.97 | 2867.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 2915.30 | 2878.40 | 2876.70 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 2841.00 | 2872.98 | 2875.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 2818.50 | 2857.34 | 2866.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 14:15:00 | 2843.80 | 2841.05 | 2853.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 2843.80 | 2841.05 | 2853.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 2840.00 | 2840.84 | 2852.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 2820.70 | 2840.84 | 2852.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 2913.20 | 2835.55 | 2839.93 | SL hit (close>static) qty=1.00 sl=2860.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 2904.00 | 2849.24 | 2845.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 13:15:00 | 2959.00 | 2916.99 | 2897.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 13:15:00 | 2938.00 | 2938.64 | 2921.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 13:30:00 | 2935.90 | 2938.64 | 2921.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 2912.00 | 2936.19 | 2926.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 2912.00 | 2936.19 | 2926.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 2914.70 | 2931.89 | 2925.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:30:00 | 2906.20 | 2931.89 | 2925.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 2906.80 | 2920.38 | 2920.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 2883.10 | 2912.92 | 2917.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 2919.00 | 2909.61 | 2914.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 2919.00 | 2909.61 | 2914.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2919.00 | 2909.61 | 2914.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 2911.40 | 2909.61 | 2914.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 2926.30 | 2912.95 | 2915.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 2926.30 | 2912.95 | 2915.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2925.00 | 2915.36 | 2916.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:15:00 | 2930.20 | 2915.36 | 2916.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 2936.90 | 2919.67 | 2918.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 2981.00 | 2931.93 | 2924.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 3055.00 | 3063.85 | 3047.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 3055.00 | 3063.85 | 3047.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 3049.20 | 3065.86 | 3052.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 3049.20 | 3065.86 | 3052.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 3051.10 | 3062.91 | 3052.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 3092.70 | 3062.91 | 3052.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 15:15:00 | 3099.50 | 3126.04 | 3129.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 3099.50 | 3126.04 | 3129.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 3089.90 | 3118.68 | 3125.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 3093.10 | 3090.48 | 3105.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 3093.10 | 3090.48 | 3105.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 3100.00 | 3092.39 | 3104.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 3097.40 | 3092.39 | 3104.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 3006.40 | 2974.15 | 2992.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 3009.60 | 2974.15 | 2992.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 3036.80 | 2986.68 | 2996.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 3038.00 | 2986.68 | 2996.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 3072.10 | 3003.76 | 3003.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 3113.00 | 3025.61 | 3013.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 11:15:00 | 3052.50 | 3087.29 | 3057.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 11:15:00 | 3052.50 | 3087.29 | 3057.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 3052.50 | 3087.29 | 3057.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:00:00 | 3052.50 | 3087.29 | 3057.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 3095.10 | 3088.85 | 3060.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:30:00 | 3068.60 | 3088.85 | 3060.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3065.00 | 3084.08 | 3061.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:30:00 | 3059.20 | 3084.08 | 3061.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 3048.80 | 3077.02 | 3060.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 3048.80 | 3077.02 | 3060.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 3050.00 | 3071.62 | 3059.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 3064.00 | 3071.62 | 3059.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 3022.80 | 3058.40 | 3055.20 | SL hit (close<static) qty=1.00 sl=3043.30 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 3028.80 | 3052.48 | 3052.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 12:15:00 | 3006.30 | 3043.24 | 3048.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 3044.80 | 3037.64 | 3044.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 3044.80 | 3037.64 | 3044.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 3044.80 | 3037.64 | 3044.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 3044.80 | 3037.64 | 3044.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 3050.00 | 3040.11 | 3045.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 3053.10 | 3040.11 | 3045.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 3062.70 | 3044.63 | 3046.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 3062.70 | 3044.63 | 3046.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 11:15:00 | 3050.70 | 3048.30 | 3048.25 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 3025.90 | 3043.82 | 3046.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 3003.40 | 3035.74 | 3042.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 2991.80 | 2991.36 | 3010.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:30:00 | 2990.40 | 2991.36 | 3010.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2980.00 | 2988.50 | 3006.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 2968.50 | 2985.64 | 3003.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 3017.90 | 2992.09 | 3004.50 | SL hit (close>static) qty=1.00 sl=3006.90 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 14:15:00 | 3070.80 | 3013.69 | 3011.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 15:15:00 | 3086.00 | 3028.15 | 3018.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 15:15:00 | 3207.00 | 3209.21 | 3178.53 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:15:00 | 3221.00 | 3209.21 | 3178.53 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:15:00 | 3218.00 | 3209.51 | 3181.46 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 15:00:00 | 3224.80 | 3212.29 | 3193.41 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:15:00 | 3382.05 | 3336.19 | 3282.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:15:00 | 3378.90 | 3336.19 | 3282.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:15:00 | 3386.04 | 3336.19 | 3282.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 3376.50 | 3378.99 | 3331.55 | SL hit (close<ema200) qty=0.50 sl=3378.99 alert=retest1 |

### Cycle 34 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 3328.90 | 3355.50 | 3358.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 3323.20 | 3349.04 | 3355.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 3377.10 | 3353.34 | 3355.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 11:15:00 | 3377.10 | 3353.34 | 3355.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 3377.10 | 3353.34 | 3355.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 3377.10 | 3353.34 | 3355.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 3354.60 | 3353.59 | 3355.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:15:00 | 3345.50 | 3353.59 | 3355.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 15:15:00 | 3340.00 | 3353.10 | 3354.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:45:00 | 3350.50 | 3352.12 | 3353.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:45:00 | 3347.30 | 3349.22 | 3352.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 3351.20 | 3349.61 | 3352.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:00:00 | 3351.20 | 3349.61 | 3352.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 3351.80 | 3350.05 | 3352.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 3363.00 | 3350.05 | 3352.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 3321.40 | 3344.32 | 3349.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:15:00 | 3316.90 | 3344.32 | 3349.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:45:00 | 3310.80 | 3315.01 | 3330.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:00:00 | 3317.30 | 3315.47 | 3329.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3178.22 | 3221.58 | 3261.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3173.00 | 3221.58 | 3261.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3182.97 | 3221.58 | 3261.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3179.93 | 3221.58 | 3261.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:15:00 | 3151.05 | 3206.86 | 3250.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:15:00 | 3151.43 | 3206.86 | 3250.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 3145.26 | 3191.99 | 3240.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 3110.80 | 3103.26 | 3130.74 | SL hit (close>ema200) qty=0.50 sl=3103.26 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 3130.00 | 3111.58 | 3110.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 3166.00 | 3125.81 | 3117.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 3182.70 | 3196.21 | 3172.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 3182.70 | 3196.21 | 3172.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 3183.20 | 3193.60 | 3173.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:15:00 | 3173.80 | 3193.60 | 3173.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 3164.00 | 3187.68 | 3172.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 3164.00 | 3187.68 | 3172.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 3165.60 | 3183.27 | 3171.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:30:00 | 3157.20 | 3183.27 | 3171.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 3136.90 | 3163.52 | 3164.81 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 3172.20 | 3166.53 | 3165.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 3183.30 | 3169.89 | 3167.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 15:15:00 | 3176.10 | 3176.67 | 3171.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 3171.00 | 3176.67 | 3171.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3169.40 | 3175.22 | 3171.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 3169.40 | 3175.22 | 3171.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 3161.00 | 3172.37 | 3170.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 3146.60 | 3172.37 | 3170.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 3156.20 | 3169.14 | 3169.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 3124.00 | 3155.18 | 3162.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 3119.00 | 3106.11 | 3119.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 3119.00 | 3106.11 | 3119.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 3119.00 | 3106.11 | 3119.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 3119.00 | 3106.11 | 3119.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 3157.90 | 3116.47 | 3122.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 3157.90 | 3116.47 | 3122.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 3155.40 | 3124.25 | 3125.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 3152.20 | 3124.25 | 3125.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 3142.70 | 3127.94 | 3127.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 3161.10 | 3140.07 | 3133.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 3168.30 | 3174.20 | 3161.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:45:00 | 3166.90 | 3174.20 | 3161.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 3152.00 | 3169.76 | 3160.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 3152.00 | 3169.76 | 3160.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 3122.80 | 3160.37 | 3157.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 3122.80 | 3160.37 | 3157.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 3130.00 | 3154.29 | 3154.69 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3192.90 | 3156.44 | 3155.11 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 3139.60 | 3153.07 | 3153.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 11:15:00 | 3122.90 | 3147.04 | 3150.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 3136.80 | 3134.79 | 3142.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 3136.80 | 3134.79 | 3142.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 3136.80 | 3134.79 | 3142.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 3141.00 | 3134.79 | 3142.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 3184.80 | 3144.10 | 3145.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 3184.80 | 3144.10 | 3145.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 3184.30 | 3152.14 | 3148.83 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 3152.10 | 3165.39 | 3166.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 3123.00 | 3148.79 | 3157.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 3159.30 | 3146.92 | 3154.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 3159.30 | 3146.92 | 3154.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 3159.30 | 3146.92 | 3154.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 3161.10 | 3146.92 | 3154.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 3150.00 | 3147.54 | 3153.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:15:00 | 3141.80 | 3147.54 | 3153.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:00:00 | 3145.10 | 3147.16 | 3152.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 3140.90 | 3135.03 | 3139.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 14:15:00 | 2984.71 | 3021.72 | 3041.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 14:15:00 | 2987.84 | 3021.72 | 3041.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 15:15:00 | 2983.86 | 3014.38 | 3036.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 3041.30 | 3019.76 | 3036.58 | SL hit (close>ema200) qty=0.50 sl=3019.76 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 2995.00 | 2982.79 | 2982.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 3027.90 | 2991.81 | 2986.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 3032.20 | 3042.85 | 3026.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 3032.20 | 3042.85 | 3026.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 3049.00 | 3044.08 | 3028.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 3067.40 | 3044.08 | 3028.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 3052.80 | 3045.94 | 3031.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:45:00 | 3052.30 | 3047.98 | 3035.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:15:00 | 3057.00 | 3047.98 | 3035.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 3051.70 | 3048.72 | 3036.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:00:00 | 3051.70 | 3048.72 | 3036.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 3030.50 | 3075.54 | 3064.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 3030.50 | 3075.54 | 3064.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 3035.50 | 3067.53 | 3061.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 3032.20 | 3067.53 | 3061.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 3043.20 | 3059.41 | 3058.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 3044.20 | 3059.41 | 3058.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-18 13:15:00 | 3047.10 | 3056.95 | 3057.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 13:15:00 | 3047.10 | 3056.95 | 3057.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 3024.10 | 3044.39 | 3051.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 3036.90 | 3036.02 | 3043.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:30:00 | 3038.90 | 3036.02 | 3043.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 3039.50 | 3034.96 | 3041.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:30:00 | 3031.80 | 3032.15 | 3039.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 15:15:00 | 2880.21 | 2927.53 | 2971.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 2913.10 | 2898.32 | 2933.79 | SL hit (close>ema200) qty=0.50 sl=2898.32 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 2671.00 | 2638.83 | 2637.93 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 2613.30 | 2637.25 | 2640.02 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 2646.20 | 2639.46 | 2639.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 2654.60 | 2642.49 | 2640.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 11:15:00 | 2639.30 | 2643.40 | 2641.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 11:15:00 | 2639.30 | 2643.40 | 2641.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 2639.30 | 2643.40 | 2641.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 2639.30 | 2643.40 | 2641.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 2645.20 | 2643.76 | 2641.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:30:00 | 2633.50 | 2643.76 | 2641.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 2651.90 | 2645.39 | 2642.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:30:00 | 2644.30 | 2645.39 | 2642.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 2660.10 | 2648.33 | 2644.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 2643.80 | 2648.33 | 2644.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2627.90 | 2643.71 | 2642.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 2620.60 | 2643.71 | 2642.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 2635.10 | 2641.99 | 2642.10 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 2654.50 | 2642.26 | 2641.59 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 2631.00 | 2640.01 | 2640.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 2616.70 | 2633.27 | 2637.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 2628.60 | 2626.85 | 2632.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 2628.60 | 2626.85 | 2632.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 2628.60 | 2626.85 | 2632.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 2651.50 | 2626.85 | 2632.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 2589.80 | 2611.32 | 2621.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:30:00 | 2582.00 | 2598.29 | 2611.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:00:00 | 2583.20 | 2589.59 | 2603.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 2578.30 | 2586.25 | 2600.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 2630.30 | 2600.76 | 2603.44 | SL hit (close>static) qty=1.00 sl=2629.10 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 2631.50 | 2606.91 | 2605.99 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 15:15:00 | 2593.60 | 2607.85 | 2608.55 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 2617.50 | 2603.11 | 2603.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 11:15:00 | 2621.00 | 2608.95 | 2605.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 2602.10 | 2611.52 | 2608.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 14:15:00 | 2602.10 | 2611.52 | 2608.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 2602.10 | 2611.52 | 2608.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 2613.30 | 2611.52 | 2608.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 2597.00 | 2608.62 | 2607.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 2597.30 | 2608.62 | 2607.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 2595.20 | 2604.80 | 2605.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 2583.00 | 2600.44 | 2603.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 2609.90 | 2598.63 | 2601.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 2609.90 | 2598.63 | 2601.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2609.90 | 2598.63 | 2601.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 2609.90 | 2598.63 | 2601.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2605.00 | 2599.90 | 2602.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 2605.00 | 2599.90 | 2602.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 2600.30 | 2599.98 | 2602.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 2578.00 | 2599.98 | 2602.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2570.10 | 2594.01 | 2599.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:15:00 | 2563.80 | 2585.36 | 2594.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:45:00 | 2564.00 | 2578.30 | 2589.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 13:15:00 | 2609.30 | 2590.73 | 2589.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 2609.30 | 2590.73 | 2589.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 2616.50 | 2595.89 | 2592.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 13:15:00 | 2609.80 | 2613.26 | 2604.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 13:30:00 | 2609.50 | 2613.26 | 2604.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 2608.00 | 2615.09 | 2607.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 2607.70 | 2615.09 | 2607.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 2593.40 | 2610.76 | 2606.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 2592.10 | 2610.76 | 2606.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 2591.30 | 2606.86 | 2604.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 2591.30 | 2606.86 | 2604.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 13:15:00 | 2590.00 | 2601.10 | 2602.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 2553.00 | 2592.10 | 2598.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 13:15:00 | 2553.30 | 2542.93 | 2558.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 2553.30 | 2542.93 | 2558.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 2545.40 | 2543.42 | 2557.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:45:00 | 2559.90 | 2543.42 | 2557.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 2558.20 | 2547.43 | 2556.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 2558.20 | 2547.43 | 2556.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 2553.20 | 2548.58 | 2556.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 2540.40 | 2548.58 | 2556.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 2413.38 | 2453.53 | 2487.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 2374.40 | 2362.87 | 2384.21 | SL hit (close>ema200) qty=0.50 sl=2362.87 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 15:15:00 | 2380.00 | 2378.08 | 2378.03 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 2361.80 | 2374.82 | 2376.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 2342.40 | 2362.39 | 2369.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2299.50 | 2291.64 | 2317.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:45:00 | 2301.70 | 2291.64 | 2317.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2303.10 | 2284.58 | 2303.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 2303.10 | 2284.58 | 2303.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 2305.00 | 2288.66 | 2303.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 2275.90 | 2288.66 | 2303.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 2298.00 | 2243.87 | 2237.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 2298.00 | 2243.87 | 2237.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 2304.50 | 2277.42 | 2258.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 2311.90 | 2314.79 | 2288.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:30:00 | 2315.80 | 2314.79 | 2288.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 2336.40 | 2350.38 | 2329.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 2313.60 | 2350.38 | 2329.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2306.80 | 2341.67 | 2327.44 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 2286.50 | 2313.81 | 2316.69 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2471.60 | 2343.87 | 2328.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 2515.20 | 2378.13 | 2345.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 2438.70 | 2530.31 | 2479.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 2438.70 | 2530.31 | 2479.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 2438.70 | 2530.31 | 2479.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 2438.70 | 2530.31 | 2479.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 2426.10 | 2509.47 | 2474.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 2434.10 | 2509.47 | 2474.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 2375.10 | 2449.40 | 2452.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2292.60 | 2397.57 | 2426.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2372.00 | 2337.08 | 2372.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 2372.00 | 2337.08 | 2372.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2372.00 | 2337.08 | 2372.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 2372.00 | 2337.08 | 2372.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 2359.80 | 2341.63 | 2371.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:15:00 | 2348.10 | 2341.63 | 2371.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 2382.00 | 2362.16 | 2370.04 | SL hit (close>static) qty=1.00 sl=2378.40 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 2438.50 | 2377.43 | 2376.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 2469.50 | 2430.11 | 2407.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 2431.40 | 2433.15 | 2413.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 13:00:00 | 2431.40 | 2433.15 | 2413.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2440.80 | 2439.52 | 2423.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 2422.10 | 2439.52 | 2423.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 2475.00 | 2478.34 | 2457.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 2491.10 | 2485.53 | 2462.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 2515.10 | 2490.12 | 2472.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 2458.40 | 2480.42 | 2482.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 2458.40 | 2480.42 | 2482.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 09:15:00 | 2439.30 | 2468.01 | 2475.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 2462.70 | 2455.07 | 2465.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 2462.70 | 2455.07 | 2465.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2473.20 | 2458.70 | 2466.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 2473.20 | 2458.70 | 2466.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 2469.90 | 2460.94 | 2466.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 2500.70 | 2460.94 | 2466.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2502.00 | 2469.15 | 2469.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 2502.10 | 2469.15 | 2469.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 2514.70 | 2478.26 | 2473.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 09:15:00 | 2517.10 | 2489.75 | 2482.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 11:15:00 | 2543.80 | 2551.81 | 2527.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 11:45:00 | 2532.90 | 2551.81 | 2527.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 2533.80 | 2548.21 | 2528.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:00:00 | 2533.80 | 2548.21 | 2528.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 2541.20 | 2544.56 | 2530.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 2556.90 | 2543.41 | 2530.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:30:00 | 2552.30 | 2546.69 | 2538.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-02 14:15:00 | 2812.59 | 2772.87 | 2728.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 13:15:00 | 2734.00 | 2755.07 | 2755.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 2627.10 | 2720.58 | 2738.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 2630.90 | 2626.61 | 2667.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 10:45:00 | 2623.50 | 2626.61 | 2667.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 2652.50 | 2630.88 | 2653.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 2666.20 | 2630.88 | 2653.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 2656.60 | 2636.03 | 2653.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 2640.70 | 2636.03 | 2653.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 2508.66 | 2574.26 | 2611.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 2547.40 | 2547.04 | 2584.56 | SL hit (close>ema200) qty=0.50 sl=2547.04 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 2525.10 | 2463.15 | 2462.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 2533.60 | 2477.24 | 2469.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2505.50 | 2523.19 | 2499.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 2505.50 | 2523.19 | 2499.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 2505.50 | 2523.19 | 2499.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 2522.90 | 2523.89 | 2502.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 2449.20 | 2524.72 | 2523.23 | SL hit (close<static) qty=1.00 sl=2476.40 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 2439.30 | 2507.63 | 2515.60 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 2567.00 | 2516.94 | 2512.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 2587.50 | 2531.05 | 2519.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2613.70 | 2642.94 | 2604.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 2613.70 | 2642.94 | 2604.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 2583.50 | 2631.05 | 2602.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 2583.50 | 2631.05 | 2602.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 2587.50 | 2622.34 | 2601.51 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 2533.90 | 2582.88 | 2588.51 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 2611.60 | 2574.81 | 2573.57 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 2490.90 | 2560.86 | 2568.83 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 2573.10 | 2561.25 | 2561.24 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 2541.20 | 2560.45 | 2562.10 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 12:15:00 | 2577.80 | 2565.30 | 2564.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 2588.20 | 2571.72 | 2567.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 13:15:00 | 2740.10 | 2742.04 | 2713.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 14:00:00 | 2740.10 | 2742.04 | 2713.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2716.50 | 2736.17 | 2717.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2726.80 | 2732.39 | 2717.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 2727.90 | 2729.58 | 2718.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 2726.30 | 2729.58 | 2718.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 2751.00 | 2723.75 | 2718.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 2724.90 | 2728.25 | 2723.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 13:15:00 | 2725.00 | 2728.25 | 2723.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 2718.50 | 2726.30 | 2722.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 2718.50 | 2726.30 | 2722.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 2760.00 | 2733.04 | 2726.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 2723.50 | 2733.04 | 2726.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 2733.50 | 2751.82 | 2740.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 2733.50 | 2751.82 | 2740.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 2754.20 | 2752.29 | 2741.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:45:00 | 2718.80 | 2752.29 | 2741.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 2754.20 | 2752.68 | 2742.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 2779.00 | 2753.90 | 2743.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 10:00:00 | 2770.00 | 2757.12 | 2746.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 2769.70 | 2782.67 | 2767.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 12:15:00 | 2999.48 | 2943.96 | 2892.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 2905.40 | 2979.25 | 2981.78 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 2993.00 | 2981.97 | 2981.55 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 2961.80 | 2982.32 | 2982.37 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 3019.00 | 2989.66 | 2985.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 3045.80 | 3010.37 | 2998.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 11:15:00 | 3015.00 | 3016.11 | 3003.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 12:00:00 | 3015.00 | 3016.11 | 3003.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 2997.70 | 3012.43 | 3002.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 2997.70 | 3012.43 | 3002.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 2969.80 | 3003.91 | 2999.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 2969.80 | 3003.91 | 2999.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 2969.70 | 2997.06 | 2997.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:45:00 | 2965.60 | 2997.06 | 2997.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 2974.80 | 2992.61 | 2995.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 2913.10 | 2976.71 | 2987.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 2963.40 | 2946.95 | 2963.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 2963.30 | 2946.95 | 2963.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2930.30 | 2943.62 | 2960.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:15:00 | 2913.60 | 2940.81 | 2956.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 2914.50 | 2923.35 | 2941.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 12:45:00 | 2919.40 | 2920.58 | 2933.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 2914.70 | 2927.79 | 2934.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 2905.10 | 2923.25 | 2931.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:45:00 | 2882.70 | 2914.17 | 2925.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 2879.70 | 2914.17 | 2925.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 2886.30 | 2908.59 | 2922.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 2980.30 | 2924.85 | 2925.21 | SL hit (close>static) qty=1.00 sl=2978.60 alert=retest2 |

### Cycle 83 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 2973.30 | 2934.54 | 2929.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 3019.10 | 2966.58 | 2946.71 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 11:15:00 | 2990.00 | 2025-05-23 13:15:00 | 3046.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-05-22 12:45:00 | 2990.00 | 2025-05-23 13:15:00 | 3046.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-06-03 09:15:00 | 3077.40 | 2025-06-03 11:15:00 | 3040.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-06-17 10:30:00 | 2944.90 | 2025-06-20 14:15:00 | 2922.50 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2025-06-25 12:15:00 | 2836.30 | 2025-06-26 09:15:00 | 2917.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-06-25 13:45:00 | 2835.50 | 2025-06-26 09:15:00 | 2917.00 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2904.40 | 2025-07-15 12:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-07-08 12:45:00 | 2906.80 | 2025-07-15 12:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-09 09:15:00 | 2904.60 | 2025-07-15 12:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-09 10:45:00 | 2904.90 | 2025-07-15 12:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-07-10 11:15:00 | 2900.90 | 2025-07-15 12:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-23 10:30:00 | 2975.60 | 2025-07-24 09:15:00 | 2919.90 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-08-01 09:15:00 | 2820.70 | 2025-08-04 09:15:00 | 2913.20 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-08-19 09:15:00 | 3092.70 | 2025-08-21 15:15:00 | 3099.50 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-09-03 09:15:00 | 3064.00 | 2025-09-03 10:15:00 | 3022.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-08 11:15:00 | 2968.50 | 2025-09-08 11:15:00 | 3017.90 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2025-09-12 09:15:00 | 3221.00 | 2025-09-16 09:15:00 | 3382.05 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-12 10:15:00 | 3218.00 | 2025-09-16 09:15:00 | 3378.90 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-12 15:00:00 | 3224.80 | 2025-09-16 09:15:00 | 3386.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-12 09:15:00 | 3221.00 | 2025-09-16 15:15:00 | 3376.50 | STOP_HIT | 0.50 | 4.83% |
| BUY | retest1 | 2025-09-12 10:15:00 | 3218.00 | 2025-09-16 15:15:00 | 3376.50 | STOP_HIT | 0.50 | 4.93% |
| BUY | retest1 | 2025-09-12 15:00:00 | 3224.80 | 2025-09-16 15:15:00 | 3376.50 | STOP_HIT | 0.50 | 4.70% |
| BUY | retest2 | 2025-09-18 14:15:00 | 3393.00 | 2025-09-19 11:15:00 | 3341.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-22 13:15:00 | 3345.50 | 2025-09-26 09:15:00 | 3178.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 15:15:00 | 3340.00 | 2025-09-26 09:15:00 | 3173.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:45:00 | 3350.50 | 2025-09-26 09:15:00 | 3182.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 10:45:00 | 3347.30 | 2025-09-26 09:15:00 | 3179.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:15:00 | 3316.90 | 2025-09-26 10:15:00 | 3151.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 11:45:00 | 3310.80 | 2025-09-26 10:15:00 | 3151.43 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2025-09-24 13:00:00 | 3317.30 | 2025-09-26 11:15:00 | 3145.26 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2025-09-22 13:15:00 | 3345.50 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 7.02% |
| SELL | retest2 | 2025-09-22 15:15:00 | 3340.00 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 6.86% |
| SELL | retest2 | 2025-09-23 09:45:00 | 3350.50 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 7.15% |
| SELL | retest2 | 2025-09-23 10:45:00 | 3347.30 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 7.07% |
| SELL | retest2 | 2025-09-23 14:15:00 | 3316.90 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 6.21% |
| SELL | retest2 | 2025-09-24 11:45:00 | 3310.80 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 6.04% |
| SELL | retest2 | 2025-09-24 13:00:00 | 3317.30 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 6.22% |
| SELL | retest2 | 2025-10-28 11:15:00 | 3141.80 | 2025-11-04 14:15:00 | 2984.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 13:00:00 | 3145.10 | 2025-11-04 14:15:00 | 2987.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 09:15:00 | 3140.90 | 2025-11-04 15:15:00 | 2983.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 11:15:00 | 3141.80 | 2025-11-06 09:15:00 | 3041.30 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-10-28 13:00:00 | 3145.10 | 2025-11-06 09:15:00 | 3041.30 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2025-10-30 09:15:00 | 3140.90 | 2025-11-06 09:15:00 | 3041.30 | STOP_HIT | 0.50 | 3.17% |
| BUY | retest2 | 2025-11-14 09:15:00 | 3067.40 | 2025-11-18 13:15:00 | 3047.10 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-11-14 11:00:00 | 3052.80 | 2025-11-18 13:15:00 | 3047.10 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-11-14 12:45:00 | 3052.30 | 2025-11-18 13:15:00 | 3047.10 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-11-14 13:15:00 | 3057.00 | 2025-11-18 13:15:00 | 3047.10 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-11-20 11:30:00 | 3031.80 | 2025-11-21 15:15:00 | 2880.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 11:30:00 | 3031.80 | 2025-11-24 14:15:00 | 2913.10 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-12-18 13:30:00 | 2582.00 | 2025-12-19 14:15:00 | 2630.30 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-19 10:00:00 | 2583.20 | 2025-12-19 14:15:00 | 2630.30 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-12-19 10:45:00 | 2578.30 | 2025-12-19 14:15:00 | 2630.30 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-12-30 12:15:00 | 2563.80 | 2025-12-31 13:15:00 | 2609.30 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-12-30 13:45:00 | 2564.00 | 2025-12-31 13:15:00 | 2609.30 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-01-07 11:15:00 | 2540.40 | 2026-01-12 09:15:00 | 2413.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:15:00 | 2540.40 | 2026-01-16 09:15:00 | 2374.40 | STOP_HIT | 0.50 | 6.53% |
| SELL | retest2 | 2026-01-23 09:15:00 | 2275.90 | 2026-01-28 15:15:00 | 2298.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-09 11:15:00 | 2348.10 | 2026-02-10 09:15:00 | 2382.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-02-13 11:45:00 | 2491.10 | 2026-02-17 13:15:00 | 2458.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-02-16 09:15:00 | 2515.10 | 2026-02-17 13:15:00 | 2458.40 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-02-24 09:15:00 | 2556.90 | 2026-03-02 14:15:00 | 2812.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-24 14:30:00 | 2552.30 | 2026-03-02 14:15:00 | 2807.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-11 10:15:00 | 2640.70 | 2026-03-12 09:15:00 | 2508.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:15:00 | 2640.70 | 2026-03-12 13:15:00 | 2547.40 | STOP_HIT | 0.50 | 3.53% |
| BUY | retest2 | 2026-03-19 10:30:00 | 2522.90 | 2026-03-23 09:15:00 | 2449.20 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2726.80 | 2026-04-22 12:15:00 | 2999.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 12:30:00 | 2727.90 | 2026-04-22 12:15:00 | 3000.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:00:00 | 2726.30 | 2026-04-22 12:15:00 | 2998.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 2751.00 | 2026-04-22 12:15:00 | 3026.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-17 09:15:00 | 2779.00 | 2026-04-22 12:15:00 | 3047.00 | TARGET_HIT | 1.00 | 9.64% |
| BUY | retest2 | 2026-04-17 10:00:00 | 2770.00 | 2026-04-22 12:15:00 | 3046.67 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2026-04-20 09:45:00 | 2769.70 | 2026-04-23 09:15:00 | 3056.90 | TARGET_HIT | 1.00 | 10.37% |
| SELL | retest2 | 2026-05-04 12:15:00 | 2913.60 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-05-05 09:15:00 | 2914.50 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-05-05 12:45:00 | 2919.40 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-05-06 09:15:00 | 2914.70 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-05-06 11:45:00 | 2882.70 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2026-05-06 12:15:00 | 2879.70 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2026-05-06 13:00:00 | 2886.30 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -3.26% |
