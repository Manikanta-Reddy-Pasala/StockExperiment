# SRF Ltd. (SRF)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2778.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 10 |
| TARGET_HIT | 6 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 11
- **Target hits / Stop hits / Partials:** 5 / 16 / 10
- **Avg / median % per leg:** 2.61% / 1.73%
- **Sum % (uncompounded):** 80.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.26% | -11.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.26% | -11.3% |
| SELL (all) | 26 | 20 | 76.9% | 5 | 11 | 10 | 3.55% | 92.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 20 | 76.9% | 5 | 11 | 10 | 3.55% | 92.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 31 | 20 | 64.5% | 5 | 16 | 10 | 2.61% | 81.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 2843.00 | 3058.51 | 3058.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 2831.00 | 2980.92 | 3012.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2961.50 | 2938.82 | 2979.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 2961.50 | 2938.82 | 2979.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2961.50 | 2938.82 | 2979.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 2938.40 | 2946.84 | 2978.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 2929.80 | 2946.68 | 2978.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 2929.50 | 2946.62 | 2978.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 2934.50 | 2946.62 | 2978.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 2982.00 | 2946.94 | 2977.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 2982.00 | 2946.94 | 2977.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 2980.00 | 2947.27 | 2977.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:15:00 | 2983.10 | 2947.27 | 2977.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 2971.10 | 2947.51 | 2977.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 2966.10 | 2947.74 | 2977.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 2817.79 | 2931.51 | 2962.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 2791.48 | 2924.79 | 2957.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 2783.31 | 2912.66 | 2949.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 2783.03 | 2912.66 | 2949.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 2787.78 | 2912.66 | 2949.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 2914.90 | 2910.45 | 2946.98 | SL hit (close>ema200) qty=0.50 sl=2910.45 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 3200.50 | 2969.90 | 2969.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3204.10 | 2972.23 | 2970.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 3009.50 | 3013.44 | 2994.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:45:00 | 3014.00 | 3013.44 | 2994.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2988.80 | 3014.95 | 2996.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 2988.80 | 3014.95 | 2996.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 2992.90 | 3014.73 | 2996.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 2973.60 | 3014.73 | 2996.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 2964.10 | 3014.23 | 2995.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 2964.10 | 3014.23 | 2995.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 2978.00 | 3013.87 | 2995.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 2981.90 | 3013.11 | 2995.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 2955.10 | 3012.21 | 2995.31 | SL hit (close<static) qty=1.00 sl=2962.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 2902.50 | 2981.90 | 2981.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 2885.80 | 2979.54 | 2980.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 2919.80 | 2895.86 | 2930.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 2919.80 | 2895.86 | 2930.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2919.80 | 2895.86 | 2930.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:45:00 | 2922.60 | 2895.86 | 2930.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 2921.80 | 2896.11 | 2930.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 2922.40 | 2896.11 | 2930.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 2945.50 | 2896.61 | 2930.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:00:00 | 2945.50 | 2896.61 | 2930.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 2918.40 | 2896.82 | 2930.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:30:00 | 2945.80 | 2896.82 | 2930.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 2911.10 | 2896.96 | 2930.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:30:00 | 2920.00 | 2896.96 | 2930.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 2933.00 | 2897.61 | 2930.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 2913.60 | 2897.61 | 2930.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2897.70 | 2897.61 | 2930.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:30:00 | 2890.20 | 2898.26 | 2929.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 13:45:00 | 2895.00 | 2883.61 | 2915.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:45:00 | 2894.20 | 2883.70 | 2915.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 11:15:00 | 2946.00 | 2885.32 | 2915.76 | SL hit (close>static) qty=1.00 sl=2939.90 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 3083.00 | 2938.21 | 2937.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 3126.90 | 2940.09 | 2938.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 2976.30 | 3009.61 | 2980.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 2976.30 | 3009.61 | 2980.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2976.30 | 3009.61 | 2980.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 2976.30 | 3009.61 | 2980.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 3015.80 | 3009.67 | 2980.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 3019.90 | 3009.67 | 2980.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 3036.20 | 3018.54 | 2988.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 09:45:00 | 3032.00 | 3019.00 | 2989.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 11:45:00 | 3020.00 | 3019.08 | 2989.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2994.00 | 3023.52 | 2995.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 2994.00 | 3023.52 | 2995.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 2982.60 | 3023.11 | 2995.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:45:00 | 2982.00 | 3023.11 | 2995.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 2968.30 | 3022.56 | 2995.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 2969.40 | 3022.56 | 2995.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-19 12:15:00 | 2948.30 | 3021.82 | 2995.18 | SL hit (close<static) qty=1.00 sl=2955.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2706.60 | 2971.95 | 2972.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 2693.80 | 2954.55 | 2963.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2929.60 | 2895.15 | 2928.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 2929.60 | 2895.15 | 2928.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2929.60 | 2895.15 | 2928.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:15:00 | 2912.00 | 2896.37 | 2928.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 2916.80 | 2897.26 | 2928.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 13:45:00 | 2921.80 | 2898.48 | 2928.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 14:45:00 | 2921.30 | 2898.71 | 2928.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 2902.20 | 2899.02 | 2928.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:30:00 | 2866.90 | 2899.11 | 2927.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 2963.30 | 2899.68 | 2926.55 | SL hit (close>static) qty=1.00 sl=2937.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-16 09:45:00 | 2938.40 | 2025-09-26 13:15:00 | 2817.79 | PARTIAL | 0.50 | 4.10% |
| SELL | retest2 | 2025-09-16 10:45:00 | 2929.80 | 2025-09-29 12:15:00 | 2791.48 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2025-09-16 11:45:00 | 2929.50 | 2025-10-01 09:15:00 | 2783.31 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-09-16 12:15:00 | 2934.50 | 2025-10-01 09:15:00 | 2783.03 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-09-17 13:15:00 | 2966.10 | 2025-10-01 09:15:00 | 2787.78 | PARTIAL | 0.50 | 6.01% |
| SELL | retest2 | 2025-09-16 09:45:00 | 2938.40 | 2025-10-03 12:15:00 | 2914.90 | STOP_HIT | 0.50 | 0.80% |
| SELL | retest2 | 2025-09-16 10:45:00 | 2929.80 | 2025-10-03 12:15:00 | 2914.90 | STOP_HIT | 0.50 | 0.51% |
| SELL | retest2 | 2025-09-16 11:45:00 | 2929.50 | 2025-10-03 12:15:00 | 2914.90 | STOP_HIT | 0.50 | 0.50% |
| SELL | retest2 | 2025-09-16 12:15:00 | 2934.50 | 2025-10-03 12:15:00 | 2914.90 | STOP_HIT | 0.50 | 0.67% |
| SELL | retest2 | 2025-09-17 13:15:00 | 2966.10 | 2025-10-03 12:15:00 | 2914.90 | STOP_HIT | 0.50 | 1.73% |
| SELL | retest2 | 2025-10-08 10:15:00 | 2968.80 | 2025-10-08 11:15:00 | 2995.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-08 15:15:00 | 2968.20 | 2025-10-09 09:15:00 | 2993.80 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-30 15:00:00 | 2981.90 | 2025-10-31 09:15:00 | 2955.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-12-02 09:30:00 | 2890.20 | 2025-12-10 11:15:00 | 2946.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-12-09 13:45:00 | 2895.00 | 2025-12-10 11:15:00 | 2946.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-09 14:45:00 | 2894.20 | 2025-12-10 11:15:00 | 2946.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-01-06 12:15:00 | 3019.90 | 2026-01-19 12:15:00 | 2948.30 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-01-09 09:45:00 | 3036.20 | 2026-01-19 12:15:00 | 2948.30 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2026-01-12 09:45:00 | 3032.00 | 2026-01-19 12:15:00 | 2948.30 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2026-01-12 11:45:00 | 3020.00 | 2026-01-19 12:15:00 | 2948.30 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-03 13:15:00 | 2912.00 | 2026-02-09 10:15:00 | 2963.30 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-02-04 10:15:00 | 2916.80 | 2026-02-13 09:15:00 | 2766.40 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2026-02-04 13:45:00 | 2921.80 | 2026-02-13 09:15:00 | 2770.96 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2026-02-04 14:45:00 | 2921.30 | 2026-02-13 09:15:00 | 2775.71 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2026-02-06 09:30:00 | 2866.90 | 2026-02-13 09:15:00 | 2775.24 | PARTIAL | 0.50 | 3.20% |
| SELL | retest2 | 2026-02-12 11:45:00 | 2862.50 | 2026-02-18 09:15:00 | 2719.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 10:15:00 | 2916.80 | 2026-02-23 15:15:00 | 2625.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 13:45:00 | 2921.80 | 2026-02-23 15:15:00 | 2629.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 14:45:00 | 2921.30 | 2026-02-23 15:15:00 | 2629.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-06 09:30:00 | 2866.90 | 2026-02-24 09:15:00 | 2620.80 | TARGET_HIT | 0.50 | 8.58% |
| SELL | retest2 | 2026-02-12 11:45:00 | 2862.50 | 2026-02-24 11:15:00 | 2576.25 | TARGET_HIT | 0.50 | 10.00% |
