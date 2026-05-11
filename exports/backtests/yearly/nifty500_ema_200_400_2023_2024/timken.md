# Timken India Ltd. (TIMKEN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3600.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 6 |
| ALERT3 | 66 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 57 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 54
- **Target hits / Stop hits / Partials:** 4 / 56 / 3
- **Avg / median % per leg:** -1.44% / -1.57%
- **Sum % (uncompounded):** -90.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 3 | 17.6% | 3 | 14 | 0 | -1.02% | -17.3% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.64% | -13.9% |
| BUY @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 3 | 11 | 0 | -0.24% | -3.3% |
| SELL (all) | 46 | 6 | 13.0% | 1 | 42 | 3 | -1.59% | -73.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 6 | 13.0% | 1 | 42 | 3 | -1.59% | -73.3% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.64% | -13.9% |
| retest2 (combined) | 60 | 9 | 15.0% | 4 | 53 | 3 | -1.28% | -76.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 14:15:00 | 3197.05 | 3270.86 | 3270.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 10:15:00 | 3190.10 | 3268.81 | 3269.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 11:15:00 | 3254.00 | 3248.17 | 3258.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 11:15:00 | 3254.00 | 3248.17 | 3258.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 11:15:00 | 3254.00 | 3248.17 | 3258.39 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-12-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 15:15:00 | 3208.70 | 3058.25 | 3058.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 3213.80 | 3059.80 | 3058.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 11:15:00 | 3137.10 | 3146.43 | 3111.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 12:15:00 | 3223.65 | 3291.44 | 3228.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 12:15:00 | 3223.65 | 3291.44 | 3228.02 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 09:15:00 | 2840.25 | 3176.32 | 3177.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 10:15:00 | 2770.00 | 3026.67 | 3092.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-18 10:15:00 | 2795.75 | 2792.06 | 2921.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 11:15:00 | 2901.05 | 2797.18 | 2906.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 11:15:00 | 2901.05 | 2797.18 | 2906.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:30:00 | 2898.20 | 2855.96 | 2904.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 2940.30 | 2856.80 | 2904.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:30:00 | 2938.70 | 2856.80 | 2904.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 2938.90 | 2862.19 | 2905.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 11:00:00 | 2938.90 | 2862.19 | 2905.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 2931.55 | 2862.88 | 2905.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 12:30:00 | 2915.05 | 2863.39 | 2906.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-16 10:15:00 | 2989.00 | 2866.98 | 2906.79 | SL hit (close>static) qty=1.00 sl=2959.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 14:15:00 | 3169.40 | 2939.71 | 2939.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 09:15:00 | 3179.40 | 2944.34 | 2941.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 4244.35 | 4250.97 | 3977.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 10:00:00 | 4244.35 | 4250.97 | 3977.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 3950.00 | 4226.15 | 3993.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:00:00 | 3950.00 | 4226.15 | 3993.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 3935.05 | 4223.25 | 3993.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 14:45:00 | 3975.00 | 4214.93 | 3992.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 4003.05 | 4212.11 | 3991.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 12:15:00 | 3905.25 | 4202.25 | 3991.35 | SL hit (close<static) qty=1.00 sl=3926.05 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 09:15:00 | 3707.80 | 3967.32 | 3967.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 3682.20 | 3903.39 | 3932.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 3920.20 | 3879.91 | 3916.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 3920.20 | 3879.91 | 3916.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 3920.20 | 3879.91 | 3916.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 3920.20 | 3879.91 | 3916.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 3880.00 | 3879.92 | 3915.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:30:00 | 3929.00 | 3879.92 | 3915.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 3866.40 | 3879.78 | 3915.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 11:30:00 | 3909.95 | 3879.78 | 3915.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 3876.00 | 3815.74 | 3864.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 3889.85 | 3815.74 | 3864.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 3863.40 | 3816.21 | 3864.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 3838.45 | 3816.39 | 3864.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:15:00 | 3841.15 | 3817.31 | 3864.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:45:00 | 3840.05 | 3817.57 | 3864.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 11:15:00 | 3829.65 | 3817.94 | 3864.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 3867.75 | 3817.38 | 3862.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:45:00 | 3865.00 | 3817.38 | 3862.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 3863.00 | 3817.83 | 3862.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:45:00 | 3873.25 | 3817.83 | 3862.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 3844.15 | 3818.09 | 3862.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 3932.00 | 3819.23 | 3862.39 | SL hit (close>static) qty=1.00 sl=3879.70 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 3126.90 | 2692.85 | 2691.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 3142.70 | 2697.33 | 2693.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 15:15:00 | 3365.00 | 3366.19 | 3235.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:15:00 | 3422.50 | 3366.19 | 3235.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:15:00 | 3402.10 | 3370.62 | 3247.16 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 14:00:00 | 3400.00 | 3371.35 | 3249.97 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.82 | SL hit (close<ema400) qty=1.00 sl=3253.82 alert=retest1 |

### Cycle 7 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 2881.70 | 3173.78 | 3174.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 14:15:00 | 2852.10 | 3170.58 | 3172.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 3092.20 | 3089.02 | 3126.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:00:00 | 3092.20 | 3089.02 | 3126.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 3036.70 | 3002.07 | 3058.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 3037.30 | 3002.07 | 3058.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 3061.30 | 3005.94 | 3056.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 3061.30 | 3005.94 | 3056.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 3050.40 | 3006.38 | 3056.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 3042.70 | 3006.81 | 3056.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 3042.40 | 3007.17 | 3056.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 3033.20 | 3008.73 | 3056.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 3041.70 | 3008.98 | 3056.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 3052.20 | 3010.06 | 3056.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:15:00 | 3050.00 | 3010.06 | 3056.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 3056.00 | 3010.52 | 3056.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:15:00 | 3058.00 | 3010.52 | 3056.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 3058.00 | 3010.99 | 3056.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 3042.40 | 3010.99 | 3056.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 3038.00 | 3011.26 | 3056.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:15:00 | 3023.00 | 3011.26 | 3056.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:45:00 | 3028.50 | 3013.50 | 3055.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 3030.70 | 3013.50 | 3055.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:45:00 | 3030.90 | 3015.20 | 3054.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 3050.00 | 3016.26 | 3054.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:15:00 | 3029.70 | 3016.26 | 3054.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 15:00:00 | 3049.10 | 3016.59 | 3054.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3063.30 alert=retest2 |

### Cycle 8 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 3100.00 | 3029.03 | 3028.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 3153.40 | 3041.41 | 3035.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 3069.50 | 3079.19 | 3059.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:00:00 | 3069.50 | 3079.19 | 3059.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 3035.00 | 3078.75 | 3059.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 3035.00 | 3078.75 | 3059.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 3024.10 | 3078.20 | 3059.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 3023.40 | 3078.20 | 3059.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 3012.50 | 3073.32 | 3057.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 3031.40 | 3073.32 | 3057.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 3050.00 | 3072.92 | 3057.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:45:00 | 3050.00 | 3072.92 | 3057.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 3050.50 | 3066.48 | 3055.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:30:00 | 3050.00 | 3066.48 | 3055.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 3059.80 | 3066.08 | 3055.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 3080.80 | 3066.08 | 3055.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 3083.60 | 3066.26 | 3055.41 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 2977.30 | 3050.30 | 3050.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 2948.00 | 3048.72 | 3049.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 3050.20 | 3042.74 | 3046.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 3050.20 | 3042.74 | 3046.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 3050.20 | 3042.74 | 3046.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 3050.20 | 3042.74 | 3046.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 3047.60 | 3042.79 | 3046.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 3054.70 | 3042.79 | 3046.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 3050.00 | 3042.86 | 3046.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 3032.90 | 3043.10 | 3046.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 3060.30 | 3043.27 | 3046.50 | SL hit (close>static) qty=1.00 sl=3052.20 alert=retest2 |

### Cycle 10 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 3252.90 | 3037.02 | 3036.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 3293.50 | 3062.81 | 3050.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 3089.70 | 3098.62 | 3071.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 3089.70 | 3098.62 | 3071.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 3089.70 | 3098.62 | 3071.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 3086.30 | 3098.62 | 3071.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 3057.00 | 3098.35 | 3071.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 3057.00 | 3098.35 | 3071.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 3120.00 | 3098.56 | 3072.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 3038.90 | 3098.56 | 3072.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 3057.10 | 3098.15 | 3071.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 09:45:00 | 3087.70 | 3094.13 | 3070.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-27 09:15:00 | 3396.47 | 3155.08 | 3109.68 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-15 12:30:00 | 2915.05 | 2024-04-16 10:15:00 | 2989.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-04-16 11:45:00 | 2925.70 | 2024-04-18 09:15:00 | 3068.40 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2024-04-16 14:45:00 | 2924.45 | 2024-04-18 09:15:00 | 3068.40 | STOP_HIT | 1.00 | -4.92% |
| BUY | retest2 | 2024-07-15 14:45:00 | 3975.00 | 2024-07-16 12:15:00 | 3905.25 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-07-16 09:15:00 | 4003.05 | 2024-07-16 12:15:00 | 3905.25 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-07-18 09:15:00 | 4007.25 | 2024-07-18 14:15:00 | 3923.05 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-07-24 09:15:00 | 3969.00 | 2024-08-01 09:15:00 | 4365.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-26 09:15:00 | 4155.25 | 2024-08-09 09:15:00 | 4028.15 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2024-07-29 09:45:00 | 4132.75 | 2024-08-09 15:15:00 | 3952.95 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2024-07-29 10:15:00 | 4138.00 | 2024-08-09 15:15:00 | 3952.95 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2024-07-29 11:00:00 | 4131.75 | 2024-08-09 15:15:00 | 3952.95 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2024-08-08 12:45:00 | 4179.00 | 2024-08-09 15:15:00 | 3952.95 | STOP_HIT | 1.00 | -5.41% |
| SELL | retest2 | 2024-09-20 13:30:00 | 3838.45 | 2024-09-24 14:15:00 | 3932.00 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2024-09-23 09:15:00 | 3841.15 | 2024-09-24 14:15:00 | 3932.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-09-23 09:45:00 | 3840.05 | 2024-09-24 14:15:00 | 3932.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-09-23 11:15:00 | 3829.65 | 2024-09-24 14:15:00 | 3932.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-09-25 12:15:00 | 3839.60 | 2024-10-04 09:15:00 | 3647.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 13:15:00 | 3839.15 | 2024-10-04 09:15:00 | 3647.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 12:15:00 | 3839.60 | 2024-10-10 10:15:00 | 3773.90 | STOP_HIT | 0.50 | 1.71% |
| SELL | retest2 | 2024-09-25 13:15:00 | 3839.15 | 2024-10-10 10:15:00 | 3773.90 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2024-10-15 10:00:00 | 3832.00 | 2024-10-18 09:15:00 | 3640.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 10:00:00 | 3832.00 | 2024-10-23 09:15:00 | 3448.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-07-28 09:15:00 | 3422.50 | 2025-08-01 09:15:00 | 3250.10 | STOP_HIT | 1.00 | -5.04% |
| BUY | retest1 | 2025-07-30 10:15:00 | 3402.10 | 2025-08-01 09:15:00 | 3250.10 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest1 | 2025-07-30 14:00:00 | 3400.00 | 2025-08-01 09:15:00 | 3250.10 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-08-01 11:15:00 | 3252.10 | 2025-08-01 13:15:00 | 3218.70 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-08-01 13:00:00 | 3253.30 | 2025-08-01 13:15:00 | 3218.70 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-08-01 13:45:00 | 3251.60 | 2025-08-01 14:15:00 | 3143.00 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2025-09-16 11:30:00 | 3042.70 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-16 13:00:00 | 3042.40 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-09-17 09:45:00 | 3033.20 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-09-17 10:30:00 | 3041.70 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-18 10:15:00 | 3023.00 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-09-19 09:45:00 | 3028.50 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-09-19 10:15:00 | 3030.70 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-09-22 10:45:00 | 3030.90 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-09-22 14:15:00 | 3029.70 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-22 15:00:00 | 3049.10 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-03 09:15:00 | 3044.80 | 2025-10-03 14:15:00 | 3070.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-03 14:15:00 | 3047.30 | 2025-10-03 14:15:00 | 3070.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-10-06 11:15:00 | 3023.50 | 2025-10-30 11:15:00 | 3055.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-06 12:00:00 | 3021.00 | 2025-10-30 11:15:00 | 3055.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-10-07 09:30:00 | 3022.80 | 2025-10-30 11:15:00 | 3055.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-08 09:15:00 | 2997.50 | 2025-10-30 11:15:00 | 3055.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-10-30 10:30:00 | 3005.00 | 2025-10-30 11:15:00 | 3055.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-11-04 09:30:00 | 2983.20 | 2025-11-04 11:15:00 | 3042.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-11-06 09:30:00 | 2996.40 | 2025-11-06 11:15:00 | 3041.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-11-07 09:15:00 | 3005.80 | 2025-11-07 13:15:00 | 3047.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-11-10 11:00:00 | 3009.60 | 2025-11-11 15:15:00 | 3057.90 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-01-05 09:15:00 | 3032.90 | 2026-01-05 09:15:00 | 3060.30 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-07 12:15:00 | 3030.60 | 2026-01-14 12:15:00 | 3052.30 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-08 10:30:00 | 3042.40 | 2026-01-14 12:15:00 | 3052.30 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-01-14 10:00:00 | 3033.60 | 2026-01-14 12:15:00 | 3052.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-01-22 13:15:00 | 3008.80 | 2026-01-23 15:15:00 | 3045.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-23 09:30:00 | 3004.00 | 2026-01-23 15:15:00 | 3045.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-01-23 10:15:00 | 3012.60 | 2026-01-23 15:15:00 | 3045.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-23 14:00:00 | 3013.00 | 2026-01-23 15:15:00 | 3045.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-27 14:45:00 | 2971.00 | 2026-02-03 09:15:00 | 3281.80 | STOP_HIT | 1.00 | -10.46% |
| SELL | retest2 | 2026-01-28 09:15:00 | 2949.80 | 2026-02-03 09:15:00 | 3281.80 | STOP_HIT | 1.00 | -11.26% |
| SELL | retest2 | 2026-01-28 13:15:00 | 2962.00 | 2026-02-03 09:15:00 | 3281.80 | STOP_HIT | 1.00 | -10.80% |
| SELL | retest2 | 2026-01-29 11:15:00 | 2966.50 | 2026-02-03 09:15:00 | 3281.80 | STOP_HIT | 1.00 | -10.63% |
| BUY | retest2 | 2026-02-17 09:45:00 | 3087.70 | 2026-02-27 09:15:00 | 3396.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-23 15:00:00 | 3107.50 | 2026-04-01 10:15:00 | 3418.25 | TARGET_HIT | 1.00 | 10.00% |
