# Ceat Ltd. (CEATLTD)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3326.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 58 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 52 |
| PARTIAL | 1 |
| TARGET_HIT | 6 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 46
- **Target hits / Stop hits / Partials:** 6 / 46 / 1
- **Avg / median % per leg:** -1.27% / -2.14%
- **Sum % (uncompounded):** -67.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 5 | 16.7% | 5 | 25 | 0 | -0.80% | -24.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 5 | 16.7% | 5 | 25 | 0 | -0.80% | -24.0% |
| SELL (all) | 23 | 2 | 8.7% | 1 | 21 | 1 | -1.88% | -43.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 2 | 8.7% | 1 | 21 | 1 | -1.88% | -43.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 7 | 13.2% | 6 | 46 | 1 | -1.27% | -67.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 2852.00 | 2500.73 | 2499.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 2870.00 | 2507.94 | 2503.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 09:15:00 | 2625.25 | 2629.31 | 2579.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 2572.15 | 2627.90 | 2580.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 2572.15 | 2627.90 | 2580.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:00:00 | 2572.15 | 2627.90 | 2580.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 2572.25 | 2627.35 | 2580.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:45:00 | 2565.60 | 2627.35 | 2580.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2646.75 | 2624.55 | 2580.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 10:30:00 | 2661.65 | 2624.20 | 2585.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 14:45:00 | 2652.85 | 2638.86 | 2597.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 11:00:00 | 2665.00 | 2638.99 | 2598.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 2558.75 | 2639.08 | 2599.62 | SL hit (close<static) qty=1.00 sl=2575.25 alert=retest2 |

### Cycle 2 — SELL (started 2024-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 15:15:00 | 2702.85 | 2848.97 | 2849.47 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 13:15:00 | 2975.00 | 2847.04 | 2846.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 11:15:00 | 2980.00 | 2852.82 | 2849.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 3044.45 | 3053.29 | 2979.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 14:00:00 | 3044.45 | 3053.29 | 2979.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 2955.05 | 3051.77 | 2979.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:15:00 | 3028.40 | 3051.49 | 2979.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:45:00 | 3028.95 | 3051.28 | 2979.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 09:30:00 | 3028.00 | 3050.42 | 2981.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 2882.00 | 3094.92 | 3042.46 | SL hit (close<static) qty=1.00 sl=2955.05 alert=retest2 |

### Cycle 4 — SELL (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 10:15:00 | 2937.90 | 3007.99 | 3008.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 11:15:00 | 2916.45 | 3007.08 | 3007.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 3042.35 | 2989.71 | 2998.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 10:15:00 | 3042.35 | 2989.71 | 2998.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 3042.35 | 2989.71 | 2998.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:00:00 | 3042.35 | 2989.71 | 2998.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 3011.60 | 2989.93 | 2998.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 12:45:00 | 2998.10 | 2990.07 | 2998.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 14:30:00 | 3007.20 | 2990.35 | 2998.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 14:15:00 | 3062.35 | 2993.81 | 3000.19 | SL hit (close>static) qty=1.00 sl=3052.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 15:15:00 | 3023.00 | 2815.07 | 2814.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 3036.10 | 2817.26 | 2815.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 3648.90 | 3664.51 | 3457.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:00:00 | 3648.90 | 3664.51 | 3457.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 3480.20 | 3641.70 | 3479.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 3480.00 | 3641.70 | 3479.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 3479.60 | 3640.09 | 3479.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:30:00 | 3481.70 | 3640.09 | 3479.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 3473.70 | 3638.43 | 3479.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:30:00 | 3475.30 | 3638.43 | 3479.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 3460.50 | 3636.66 | 3478.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 3460.50 | 3636.66 | 3478.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 3473.80 | 3635.04 | 3478.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 15:00:00 | 3508.40 | 3633.78 | 3479.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-08 14:15:00 | 3859.24 | 3644.39 | 3533.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 14:15:00 | 3245.30 | 3535.01 | 3535.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 3235.10 | 3529.21 | 3532.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 3293.50 | 3273.04 | 3363.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 3365.70 | 3274.80 | 3363.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3365.70 | 3274.80 | 3363.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 3365.70 | 3274.80 | 3363.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 3379.00 | 3275.83 | 3363.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 3389.00 | 3275.83 | 3363.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 3403.10 | 3280.33 | 3361.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 3367.10 | 3281.29 | 3361.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:00:00 | 3376.70 | 3281.29 | 3361.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:45:00 | 3362.00 | 3283.83 | 3361.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 3378.30 | 3290.73 | 3361.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 3375.00 | 3293.19 | 3361.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 3375.40 | 3293.19 | 3361.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 3353.40 | 3297.09 | 3361.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 3364.90 | 3297.09 | 3361.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 3368.60 | 3298.33 | 3361.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:45:00 | 3367.90 | 3298.33 | 3361.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 3379.20 | 3299.13 | 3361.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 3379.20 | 3299.13 | 3361.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 3366.10 | 3299.80 | 3361.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 3369.90 | 3299.80 | 3361.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 3378.90 | 3301.22 | 3361.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 3378.90 | 3301.22 | 3361.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 3362.00 | 3301.82 | 3361.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 3335.00 | 3304.69 | 3362.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 3353.00 | 3305.26 | 3362.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 3355.00 | 3306.23 | 3362.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 3407.90 | 3307.87 | 3357.86 | SL hit (close>static) qty=1.00 sl=3382.10 alert=retest2 |

### Cycle 7 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 3457.40 | 3388.68 | 3388.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 3474.10 | 3391.54 | 3390.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 3874.00 | 3902.36 | 3742.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 15:00:00 | 3874.00 | 3902.36 | 3742.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 3765.00 | 3890.89 | 3798.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 3733.20 | 3890.89 | 3798.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 3798.90 | 3889.97 | 3798.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:15:00 | 3819.00 | 3889.97 | 3798.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 12:45:00 | 3810.00 | 3884.96 | 3799.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 14:30:00 | 3805.20 | 3883.41 | 3799.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 15:00:00 | 3806.30 | 3883.41 | 3799.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 3803.80 | 3882.62 | 3799.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 3791.40 | 3882.62 | 3799.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 3767.70 | 3881.47 | 3799.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 3754.40 | 3881.47 | 3799.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 3771.80 | 3880.38 | 3799.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 3771.80 | 3880.38 | 3799.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 3785.00 | 3877.68 | 3799.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:30:00 | 3791.00 | 3877.68 | 3799.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 3768.60 | 3873.63 | 3798.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 3760.80 | 3873.63 | 3798.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 3736.30 | 3871.18 | 3798.31 | SL hit (close<static) qty=1.00 sl=3753.00 alert=retest2 |

### Cycle 8 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 3710.00 | 3790.93 | 3791.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 3681.00 | 3787.10 | 3789.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 3780.90 | 3768.29 | 3779.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 3780.90 | 3768.29 | 3779.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 3780.90 | 3768.29 | 3779.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 3780.90 | 3768.29 | 3779.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 3791.10 | 3768.52 | 3779.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 3791.10 | 3768.52 | 3779.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 3755.00 | 3768.39 | 3778.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 3729.40 | 3768.39 | 3778.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:00:00 | 3742.60 | 3768.13 | 3778.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:00:00 | 3742.00 | 3767.82 | 3778.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 3830.20 | 3764.21 | 3776.14 | SL hit (close>static) qty=1.00 sl=3808.00 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 3878.10 | 3787.59 | 3787.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 12:15:00 | 3920.40 | 3791.20 | 3789.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 3851.30 | 3861.66 | 3828.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 11:45:00 | 3852.70 | 3861.66 | 3828.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 3820.40 | 3865.04 | 3834.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 3820.40 | 3865.04 | 3834.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 3804.80 | 3864.44 | 3833.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 3825.90 | 3858.83 | 3832.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 3767.90 | 3857.92 | 3831.94 | SL hit (close<static) qty=1.00 sl=3770.50 alert=retest2 |

### Cycle 10 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 3460.00 | 3808.15 | 3809.54 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-21 12:30:00 | 2474.50 | 2024-06-25 14:15:00 | 2487.65 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-06-24 09:30:00 | 2477.50 | 2024-06-26 13:15:00 | 2484.95 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-06-24 11:45:00 | 2473.95 | 2024-06-27 10:15:00 | 2618.90 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2024-06-24 13:45:00 | 2473.45 | 2024-06-27 10:15:00 | 2618.90 | STOP_HIT | 1.00 | -5.88% |
| SELL | retest2 | 2024-06-25 10:15:00 | 2465.55 | 2024-06-27 10:15:00 | 2618.90 | STOP_HIT | 1.00 | -6.22% |
| SELL | retest2 | 2024-06-25 15:15:00 | 2466.00 | 2024-06-27 10:15:00 | 2618.90 | STOP_HIT | 1.00 | -6.20% |
| BUY | retest2 | 2024-07-29 10:30:00 | 2661.65 | 2024-08-05 10:15:00 | 2558.75 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2024-08-01 14:45:00 | 2652.85 | 2024-08-05 10:15:00 | 2558.75 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-08-02 11:00:00 | 2665.00 | 2024-08-05 10:15:00 | 2558.75 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-08-07 10:30:00 | 2657.80 | 2024-08-14 09:15:00 | 2603.70 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-08-14 09:15:00 | 2628.15 | 2024-08-22 11:15:00 | 2893.00 | TARGET_HIT | 1.00 | 10.08% |
| BUY | retest2 | 2024-08-16 09:30:00 | 2630.00 | 2024-08-22 11:15:00 | 2891.24 | TARGET_HIT | 1.00 | 9.93% |
| BUY | retest2 | 2024-08-16 10:00:00 | 2628.40 | 2024-08-22 11:15:00 | 2890.58 | TARGET_HIT | 1.00 | 9.97% |
| BUY | retest2 | 2024-08-16 11:00:00 | 2627.80 | 2024-08-28 09:15:00 | 2923.58 | TARGET_HIT | 1.00 | 11.26% |
| BUY | retest2 | 2024-10-18 11:45:00 | 2955.00 | 2024-11-13 15:15:00 | 2702.85 | STOP_HIT | 1.00 | -8.53% |
| BUY | retest2 | 2024-10-21 10:15:00 | 2935.90 | 2024-11-13 15:15:00 | 2702.85 | STOP_HIT | 1.00 | -7.94% |
| BUY | retest2 | 2024-12-23 10:15:00 | 3028.40 | 2025-01-16 09:15:00 | 2882.00 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest2 | 2024-12-23 10:45:00 | 3028.95 | 2025-01-16 09:15:00 | 2882.00 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest2 | 2024-12-24 09:30:00 | 3028.00 | 2025-01-16 09:15:00 | 2882.00 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest2 | 2025-01-16 13:15:00 | 3038.00 | 2025-01-21 15:15:00 | 2952.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-02-03 12:45:00 | 2998.10 | 2025-02-04 14:15:00 | 3062.35 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-02-03 14:30:00 | 3007.20 | 2025-02-04 14:15:00 | 3062.35 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-02-05 09:30:00 | 3000.00 | 2025-02-10 09:15:00 | 2850.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 09:30:00 | 3000.00 | 2025-02-12 09:15:00 | 2700.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-17 09:15:00 | 3007.90 | 2025-04-17 11:15:00 | 3056.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-06-23 15:00:00 | 3508.40 | 2025-07-08 14:15:00 | 3859.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-23 09:15:00 | 3485.50 | 2025-07-24 09:15:00 | 3424.40 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-07-23 09:45:00 | 3510.80 | 2025-07-24 09:15:00 | 3424.40 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-07-23 10:45:00 | 3486.50 | 2025-07-24 09:15:00 | 3424.40 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-09-04 10:30:00 | 3367.10 | 2025-09-16 09:15:00 | 3407.90 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-04 11:00:00 | 3376.70 | 2025-09-16 09:15:00 | 3407.90 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-09-04 13:45:00 | 3362.00 | 2025-09-16 09:15:00 | 3407.90 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-09-08 11:15:00 | 3378.30 | 2025-09-17 09:15:00 | 3461.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-09-11 09:15:00 | 3335.00 | 2025-09-17 09:15:00 | 3461.00 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-09-11 10:15:00 | 3353.00 | 2025-09-17 09:15:00 | 3461.00 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-09-11 11:30:00 | 3355.00 | 2025-09-17 09:15:00 | 3461.00 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-09-29 11:15:00 | 3354.60 | 2025-09-30 09:15:00 | 3433.80 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-09-30 13:00:00 | 3406.30 | 2025-10-01 09:15:00 | 3465.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-12-09 11:15:00 | 3819.00 | 2025-12-12 12:15:00 | 3736.30 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-12-10 12:45:00 | 3810.00 | 2025-12-12 12:15:00 | 3736.30 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-12-10 14:30:00 | 3805.20 | 2025-12-12 12:15:00 | 3736.30 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-12-10 15:00:00 | 3806.30 | 2025-12-12 12:15:00 | 3736.30 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-01-02 10:30:00 | 3837.90 | 2026-01-06 09:15:00 | 3744.70 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2026-01-08 09:30:00 | 3846.00 | 2026-01-09 10:15:00 | 3800.70 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-01-09 09:45:00 | 3845.00 | 2026-01-09 10:15:00 | 3800.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-01-13 14:45:00 | 3838.30 | 2026-01-16 12:15:00 | 3796.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-01-19 09:15:00 | 3879.20 | 2026-01-20 12:15:00 | 3773.80 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2026-01-20 10:00:00 | 3875.00 | 2026-01-20 12:15:00 | 3773.80 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-01-20 10:30:00 | 3834.10 | 2026-01-20 12:15:00 | 3773.80 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-01 12:15:00 | 3729.40 | 2026-02-03 10:15:00 | 3830.20 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-02-01 13:00:00 | 3742.60 | 2026-02-03 10:15:00 | 3830.20 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2026-02-01 15:00:00 | 3742.00 | 2026-02-03 10:15:00 | 3830.20 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2026-02-23 09:30:00 | 3825.90 | 2026-02-23 10:15:00 | 3767.90 | STOP_HIT | 1.00 | -1.52% |
