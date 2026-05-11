# Kaynes Technology India Ltd. (KAYNES)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 4497.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 130 |
| ALERT1 | 96 |
| ALERT2 | 94 |
| ALERT2_SKIP | 47 |
| ALERT3 | 238 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 103 |
| PARTIAL | 24 |
| TARGET_HIT | 18 |
| STOP_HIT | 85 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 64
- **Target hits / Stop hits / Partials:** 18 / 85 / 24
- **Avg / median % per leg:** 1.50% / -0.16%
- **Sum % (uncompounded):** 190.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 14 | 38.9% | 11 | 25 | 0 | 1.66% | 59.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 36 | 14 | 38.9% | 11 | 25 | 0 | 1.66% | 59.9% |
| SELL (all) | 91 | 49 | 53.8% | 7 | 60 | 24 | 1.44% | 130.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 91 | 49 | 53.8% | 7 | 60 | 24 | 1.44% | 130.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 127 | 63 | 49.6% | 18 | 85 | 24 | 1.50% | 190.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 12:15:00 | 2458.20 | 2482.42 | 2484.63 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 2510.90 | 2485.06 | 2484.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 2539.00 | 2495.85 | 2489.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 2599.15 | 2605.98 | 2574.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 2599.15 | 2605.98 | 2574.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 2568.95 | 2595.00 | 2575.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 2568.95 | 2595.00 | 2575.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 2586.95 | 2593.39 | 2576.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 2836.60 | 2593.39 | 2576.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-18 09:15:00 | 3120.26 | 2999.30 | 2830.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 3292.60 | 3323.56 | 3326.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 10:15:00 | 3279.50 | 3304.72 | 3315.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 3321.30 | 3301.70 | 3311.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 12:15:00 | 3321.30 | 3301.70 | 3311.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 3321.30 | 3301.70 | 3311.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:45:00 | 3325.10 | 3301.70 | 3311.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 3301.10 | 3301.58 | 3310.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 14:45:00 | 3294.40 | 3301.27 | 3309.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:45:00 | 3287.45 | 3296.45 | 3306.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 12:15:00 | 3241.00 | 3294.02 | 3303.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 15:15:00 | 3295.00 | 3287.36 | 3297.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 3295.00 | 3288.89 | 3296.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 3300.20 | 3288.89 | 3296.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 3238.90 | 3278.89 | 3291.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-31 12:15:00 | 3351.35 | 3286.31 | 3291.14 | SL hit (close>static) qty=1.00 sl=3334.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 3420.00 | 3313.05 | 3302.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 3521.60 | 3374.61 | 3335.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 3403.00 | 3460.79 | 3410.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 3403.00 | 3460.79 | 3410.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 3403.00 | 3460.79 | 3410.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 3409.55 | 3460.79 | 3410.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3290.10 | 3426.66 | 3399.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 3290.10 | 3426.66 | 3399.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 2949.85 | 3331.29 | 3358.58 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 3205.05 | 3121.74 | 3113.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 3337.85 | 3236.32 | 3178.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 3872.85 | 3875.11 | 3806.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 10:15:00 | 3850.25 | 3875.11 | 3806.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 3852.00 | 3857.22 | 3825.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 3838.35 | 3857.22 | 3825.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 3795.30 | 3844.83 | 3823.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 3795.30 | 3844.83 | 3823.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 3795.20 | 3834.91 | 3820.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 3795.20 | 3834.91 | 3820.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 3799.75 | 3827.87 | 3818.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 3799.75 | 3827.87 | 3818.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 13:15:00 | 3782.30 | 3808.38 | 3810.75 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 3843.00 | 3815.65 | 3812.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 11:15:00 | 3885.05 | 3829.53 | 3819.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 13:15:00 | 3907.60 | 3916.42 | 3881.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-24 14:00:00 | 3907.60 | 3916.42 | 3881.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 3884.05 | 3909.95 | 3881.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 3884.05 | 3909.95 | 3881.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 3897.00 | 3907.36 | 3882.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 3945.15 | 3907.36 | 3882.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 3870.00 | 3898.96 | 3886.97 | SL hit (close<static) qty=1.00 sl=3875.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 14:15:00 | 3880.10 | 3943.69 | 3947.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 09:15:00 | 3853.05 | 3918.57 | 3934.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 3965.15 | 3882.28 | 3899.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 3965.15 | 3882.28 | 3899.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 3965.15 | 3882.28 | 3899.85 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 3980.00 | 3918.74 | 3914.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 4089.85 | 3989.54 | 3963.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 12:15:00 | 3991.00 | 4006.70 | 3979.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 13:00:00 | 3991.00 | 4006.70 | 3979.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 3977.95 | 4000.95 | 3979.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:00:00 | 3977.95 | 4000.95 | 3979.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 3982.55 | 3997.27 | 3979.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:45:00 | 3980.85 | 3997.27 | 3979.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 4005.25 | 3998.86 | 3981.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 4003.95 | 3998.86 | 3981.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 4007.70 | 4000.63 | 3984.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:15:00 | 4044.00 | 4001.20 | 3987.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 4093.90 | 4020.39 | 4002.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-08 10:15:00 | 4448.40 | 4264.75 | 4166.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 15:15:00 | 4158.00 | 4205.14 | 4208.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 4122.30 | 4177.75 | 4195.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 4172.70 | 4152.18 | 4172.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 4172.70 | 4152.18 | 4172.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 4172.70 | 4152.18 | 4172.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:15:00 | 4125.00 | 4150.67 | 4170.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 14:00:00 | 4129.95 | 4144.19 | 4150.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 4088.00 | 4143.73 | 4149.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 11:15:00 | 4253.00 | 4145.69 | 4136.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 4253.00 | 4145.69 | 4136.93 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 4084.65 | 4130.74 | 4135.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 3935.00 | 4067.45 | 4101.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 3986.55 | 3970.62 | 4025.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 3986.55 | 3970.62 | 4025.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 3986.55 | 3970.62 | 4025.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 4000.15 | 3970.62 | 4025.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 3989.00 | 3950.18 | 3984.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:00:00 | 3989.00 | 3950.18 | 3984.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 4005.00 | 3961.14 | 3986.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:30:00 | 4003.00 | 3961.14 | 3986.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 3979.95 | 3964.90 | 3985.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 3812.25 | 3964.90 | 3985.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 4062.30 | 3986.95 | 3990.64 | SL hit (close>static) qty=1.00 sl=4027.50 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 15:15:00 | 4036.00 | 3996.76 | 3994.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 4118.25 | 4021.06 | 4005.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 10:15:00 | 4502.00 | 4509.13 | 4391.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:45:00 | 4494.00 | 4509.13 | 4391.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 4430.00 | 4493.77 | 4441.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 4430.00 | 4493.77 | 4441.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 4445.30 | 4484.08 | 4442.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:30:00 | 4431.30 | 4484.08 | 4442.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 4433.65 | 4473.99 | 4441.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:00:00 | 4433.65 | 4473.99 | 4441.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 4425.85 | 4464.36 | 4439.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 4425.85 | 4464.36 | 4439.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 4419.10 | 4455.31 | 4438.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:30:00 | 4412.60 | 4455.31 | 4438.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 4461.60 | 4456.04 | 4441.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:15:00 | 4467.75 | 4456.04 | 4441.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 4460.30 | 4456.89 | 4443.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 4460.30 | 4456.89 | 4443.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 4396.30 | 4444.77 | 4438.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 4396.30 | 4444.77 | 4438.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 4405.55 | 4436.93 | 4435.79 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 4396.00 | 4428.74 | 4432.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 4388.95 | 4420.78 | 4428.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 14:15:00 | 4378.80 | 4348.95 | 4379.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 14:15:00 | 4378.80 | 4348.95 | 4379.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 4378.80 | 4348.95 | 4379.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 15:00:00 | 4378.80 | 4348.95 | 4379.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 4379.00 | 4354.96 | 4379.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 4132.50 | 4354.96 | 4379.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 15:15:00 | 4279.80 | 4246.97 | 4245.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 4279.80 | 4246.97 | 4245.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 15:15:00 | 4281.00 | 4271.58 | 4261.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 4283.00 | 4285.13 | 4272.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 12:45:00 | 4280.00 | 4285.13 | 4272.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 4283.65 | 4284.89 | 4274.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:15:00 | 4265.00 | 4284.89 | 4274.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 4265.00 | 4280.91 | 4273.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 4257.05 | 4280.91 | 4273.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 4247.10 | 4274.15 | 4271.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 13:00:00 | 4309.70 | 4279.68 | 4274.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-14 11:15:00 | 4740.67 | 4637.40 | 4516.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 11:15:00 | 5037.05 | 5108.23 | 5114.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 13:15:00 | 5004.40 | 5073.53 | 5096.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 10:15:00 | 5119.75 | 5067.30 | 5084.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 10:15:00 | 5119.75 | 5067.30 | 5084.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 5119.75 | 5067.30 | 5084.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 5119.75 | 5067.30 | 5084.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 5051.95 | 5064.23 | 5081.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 5149.65 | 5064.23 | 5081.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 5053.40 | 5052.07 | 5070.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:30:00 | 5050.05 | 5052.07 | 5070.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 5029.85 | 5048.83 | 5066.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 14:15:00 | 5007.05 | 5039.27 | 5056.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 14:45:00 | 4999.80 | 5031.41 | 5051.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:30:00 | 5000.10 | 5029.25 | 5041.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:15:00 | 4756.70 | 4868.37 | 4932.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:15:00 | 4749.81 | 4868.37 | 4932.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:15:00 | 4750.10 | 4868.37 | 4932.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-30 11:15:00 | 4797.25 | 4776.63 | 4837.29 | SL hit (close>ema200) qty=0.50 sl=4776.63 alert=retest2 |

### Cycle 18 — BUY (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 12:15:00 | 4798.00 | 4781.46 | 4780.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 4875.40 | 4822.90 | 4802.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 10:15:00 | 4845.15 | 4875.46 | 4848.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 10:15:00 | 4845.15 | 4875.46 | 4848.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 4845.15 | 4875.46 | 4848.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:00:00 | 4845.15 | 4875.46 | 4848.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 4849.95 | 4870.36 | 4848.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:30:00 | 4848.15 | 4870.36 | 4848.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 4826.00 | 4861.49 | 4846.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:45:00 | 4832.55 | 4861.49 | 4846.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 4766.10 | 4831.73 | 4834.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 4730.90 | 4801.35 | 4819.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 4675.00 | 4651.64 | 4697.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 4675.00 | 4651.64 | 4697.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 4675.00 | 4651.64 | 4697.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 4682.05 | 4651.64 | 4697.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 4659.75 | 4648.33 | 4672.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:15:00 | 4709.55 | 4648.33 | 4672.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 4669.00 | 4652.46 | 4672.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:45:00 | 4651.40 | 4654.56 | 4671.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:15:00 | 4655.70 | 4654.56 | 4671.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:45:00 | 4657.00 | 4653.71 | 4669.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 15:15:00 | 4650.00 | 4656.71 | 4668.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 4650.00 | 4655.37 | 4666.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 4875.20 | 4655.37 | 4666.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 4977.00 | 4719.70 | 4694.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 4977.00 | 4719.70 | 4694.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 11:15:00 | 5073.85 | 4829.94 | 4751.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 5256.50 | 5268.67 | 5119.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 10:00:00 | 5256.50 | 5268.67 | 5119.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 5444.55 | 5494.88 | 5434.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 09:15:00 | 5790.70 | 5552.23 | 5509.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 5458.85 | 5552.28 | 5563.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 5458.85 | 5552.28 | 5563.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 11:15:00 | 5443.00 | 5517.18 | 5544.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 5524.70 | 5493.61 | 5519.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 5524.70 | 5493.61 | 5519.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 5524.70 | 5493.61 | 5519.06 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 5550.05 | 5529.94 | 5527.92 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 5501.00 | 5524.15 | 5525.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 10:15:00 | 5462.90 | 5511.90 | 5519.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 10:15:00 | 5465.00 | 5456.00 | 5479.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 10:15:00 | 5465.00 | 5456.00 | 5479.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 5465.00 | 5456.00 | 5479.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:00:00 | 5465.00 | 5456.00 | 5479.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 5514.00 | 5467.60 | 5483.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:00:00 | 5514.00 | 5467.60 | 5483.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 5494.60 | 5473.00 | 5484.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:30:00 | 5507.10 | 5473.00 | 5484.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 5450.00 | 5468.40 | 5480.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:30:00 | 5483.50 | 5468.40 | 5480.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 5461.80 | 5467.08 | 5479.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 5461.80 | 5467.08 | 5479.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 5074.30 | 4956.08 | 5041.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:45:00 | 5066.00 | 4956.08 | 5041.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 5138.85 | 4992.64 | 5050.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 5138.85 | 4992.64 | 5050.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 5229.00 | 5039.91 | 5066.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 5229.00 | 5039.91 | 5066.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 5264.95 | 5084.92 | 5084.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 5398.30 | 5174.15 | 5127.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 5488.70 | 5509.90 | 5416.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:00:00 | 5488.70 | 5509.90 | 5416.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 5779.20 | 5727.86 | 5665.80 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 5648.00 | 5673.06 | 5674.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 5575.00 | 5653.45 | 5665.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 5604.00 | 5559.57 | 5601.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 5604.00 | 5559.57 | 5601.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 5604.00 | 5559.57 | 5601.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:45:00 | 5610.00 | 5559.57 | 5601.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 5620.00 | 5571.65 | 5602.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:45:00 | 5592.50 | 5571.65 | 5602.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 5621.00 | 5581.52 | 5604.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 5635.85 | 5581.52 | 5604.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 5675.00 | 5610.93 | 5614.47 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 11:15:00 | 5661.00 | 5620.95 | 5618.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 12:15:00 | 5730.00 | 5642.76 | 5628.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 5610.00 | 5639.49 | 5632.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 5610.00 | 5639.49 | 5632.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 5610.00 | 5639.49 | 5632.25 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 5571.05 | 5625.80 | 5626.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 5520.15 | 5604.67 | 5617.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 13:15:00 | 5518.00 | 5471.12 | 5518.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 13:15:00 | 5518.00 | 5471.12 | 5518.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 5518.00 | 5471.12 | 5518.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 5518.00 | 5471.12 | 5518.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 5470.05 | 5470.91 | 5514.12 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 11:15:00 | 5625.00 | 5538.23 | 5534.35 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 5304.80 | 5491.92 | 5515.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 5218.95 | 5437.33 | 5488.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 10:15:00 | 5362.05 | 5332.79 | 5395.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:45:00 | 5340.55 | 5332.79 | 5395.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 5359.00 | 5344.64 | 5381.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:30:00 | 5376.55 | 5344.64 | 5381.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 5303.00 | 5336.37 | 5371.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:15:00 | 5279.15 | 5336.37 | 5371.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 10:00:00 | 5280.00 | 5268.04 | 5309.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 11:00:00 | 5296.70 | 5273.77 | 5308.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 14:00:00 | 5295.90 | 5297.40 | 5312.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 5274.95 | 5292.91 | 5309.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 09:15:00 | 5242.30 | 5291.13 | 5306.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 09:15:00 | 5460.00 | 5324.90 | 5320.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 09:15:00 | 5460.00 | 5324.90 | 5320.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 10:15:00 | 5489.15 | 5357.75 | 5336.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 13:15:00 | 5367.00 | 5379.70 | 5353.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 14:00:00 | 5367.00 | 5379.70 | 5353.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 5436.40 | 5391.04 | 5360.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:15:00 | 5461.00 | 5391.04 | 5360.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 09:45:00 | 5450.80 | 5438.03 | 5396.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 5459.00 | 5439.09 | 5407.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 5301.35 | 5400.27 | 5398.36 | SL hit (close<static) qty=1.00 sl=5350.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 5259.70 | 5372.16 | 5385.75 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 5595.40 | 5381.14 | 5376.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 5697.00 | 5444.32 | 5405.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 5782.30 | 5790.99 | 5671.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 14:45:00 | 5782.60 | 5790.99 | 5671.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 5785.00 | 5791.21 | 5692.06 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 5544.45 | 5638.73 | 5650.23 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 12:15:00 | 5677.45 | 5657.65 | 5655.86 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 5638.15 | 5653.75 | 5654.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 15:15:00 | 5615.00 | 5642.63 | 5648.90 | Break + close below crossover candle low |

### Cycle 36 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 5817.05 | 5677.52 | 5664.19 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 5529.90 | 5681.59 | 5682.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 12:15:00 | 5497.00 | 5597.83 | 5639.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 5559.00 | 5494.20 | 5553.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 12:15:00 | 5559.00 | 5494.20 | 5553.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 5559.00 | 5494.20 | 5553.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 5559.00 | 5494.20 | 5553.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 5497.70 | 5494.90 | 5547.99 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 5639.20 | 5561.93 | 5559.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 5725.00 | 5608.83 | 5582.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 5750.00 | 5758.32 | 5687.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 09:15:00 | 5809.25 | 5758.32 | 5687.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 5805.05 | 5796.64 | 5743.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 5875.50 | 5796.64 | 5743.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 10:15:00 | 5847.45 | 5804.31 | 5752.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 5907.45 | 5845.85 | 5799.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 5886.45 | 5910.24 | 5912.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 5886.45 | 5910.24 | 5912.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 14:15:00 | 5832.00 | 5879.21 | 5896.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 5916.05 | 5881.09 | 5893.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 5916.05 | 5881.09 | 5893.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 5916.05 | 5881.09 | 5893.63 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 5940.40 | 5908.20 | 5904.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 6154.75 | 5983.15 | 5943.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 10:15:00 | 6169.20 | 6201.80 | 6104.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 11:00:00 | 6169.20 | 6201.80 | 6104.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 6181.05 | 6203.64 | 6157.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 6156.90 | 6203.64 | 6157.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 6266.80 | 6245.93 | 6215.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:30:00 | 6199.80 | 6245.93 | 6215.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 6206.80 | 6242.38 | 6219.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:00:00 | 6206.80 | 6242.38 | 6219.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 6207.15 | 6235.34 | 6218.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:30:00 | 6201.05 | 6235.34 | 6218.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 6231.25 | 6234.52 | 6219.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 6271.90 | 6235.27 | 6222.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 6243.85 | 6227.23 | 6219.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:45:00 | 6270.25 | 6236.82 | 6224.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:00:00 | 6244.50 | 6247.24 | 6233.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 6303.75 | 6258.54 | 6239.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:30:00 | 6236.95 | 6258.54 | 6239.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 6238.15 | 6261.10 | 6244.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 6238.15 | 6261.10 | 6244.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 6260.85 | 6261.05 | 6246.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 11:30:00 | 6338.00 | 6287.04 | 6259.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-16 09:15:00 | 6899.09 | 6762.61 | 6641.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 7103.30 | 7228.62 | 7245.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 10:15:00 | 7084.45 | 7199.79 | 7230.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 11:15:00 | 7099.00 | 7097.69 | 7146.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 12:00:00 | 7099.00 | 7097.69 | 7146.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 7096.70 | 7097.49 | 7142.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:45:00 | 7100.05 | 7097.49 | 7142.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 7223.85 | 7008.13 | 7054.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 7223.85 | 7008.13 | 7054.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 7190.00 | 7044.51 | 7066.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 7146.00 | 7044.51 | 7066.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 7214.70 | 7101.42 | 7090.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 7214.70 | 7101.42 | 7090.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 7346.00 | 7186.95 | 7135.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 7471.35 | 7536.73 | 7411.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 10:00:00 | 7471.35 | 7536.73 | 7411.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 7375.00 | 7510.97 | 7487.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 7375.00 | 7510.97 | 7487.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 7291.90 | 7467.16 | 7469.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 7227.00 | 7375.35 | 7423.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 7415.10 | 7327.62 | 7384.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 7415.10 | 7327.62 | 7384.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 7415.10 | 7327.62 | 7384.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 7400.00 | 7327.62 | 7384.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 7311.70 | 7324.43 | 7377.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 7185.20 | 7292.45 | 7341.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 09:15:00 | 6825.94 | 7004.91 | 7133.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 09:15:00 | 6466.68 | 6636.59 | 6790.17 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 44 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 6577.65 | 6459.08 | 6444.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 6650.00 | 6560.90 | 6535.57 | Break + close above crossover candle high |

### Cycle 45 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 6286.25 | 6519.73 | 6526.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 6222.75 | 6460.34 | 6499.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 5788.95 | 5607.26 | 5879.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 5788.95 | 5607.26 | 5879.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 5768.15 | 5639.44 | 5869.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:15:00 | 5692.45 | 5639.44 | 5869.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 11:00:00 | 5699.95 | 5649.34 | 5761.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 5407.83 | 5568.52 | 5677.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 5414.95 | 5568.52 | 5677.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 5123.20 | 5141.29 | 5393.01 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 46 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 4183.80 | 4132.18 | 4125.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 4244.20 | 4163.52 | 4141.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 11:15:00 | 4169.00 | 4173.92 | 4150.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-13 12:00:00 | 4169.00 | 4173.92 | 4150.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 4139.60 | 4163.89 | 4149.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 13:30:00 | 4128.05 | 4163.89 | 4149.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 4141.00 | 4159.32 | 4148.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:30:00 | 4124.40 | 4159.32 | 4148.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 15:15:00 | 4136.30 | 4154.71 | 4147.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:15:00 | 4070.05 | 4154.71 | 4147.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 4080.60 | 4139.89 | 4141.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 4043.30 | 4120.57 | 4132.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 4006.10 | 3992.15 | 4042.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 12:15:00 | 4006.10 | 3992.15 | 4042.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 4006.10 | 3992.15 | 4042.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 4006.10 | 3992.15 | 4042.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 4038.45 | 4001.41 | 4041.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 4087.45 | 4001.41 | 4041.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 4058.80 | 4012.89 | 4043.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 4058.80 | 4012.89 | 4043.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 4041.00 | 4018.51 | 4043.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 3972.75 | 4018.51 | 4043.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 4050.00 | 4024.81 | 4043.89 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 4126.00 | 4036.85 | 4030.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 4140.40 | 4057.56 | 4040.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 4300.05 | 4307.24 | 4220.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 14:00:00 | 4300.05 | 4307.24 | 4220.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 4256.00 | 4314.75 | 4247.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 4341.20 | 4313.69 | 4252.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:45:00 | 4346.40 | 4341.04 | 4288.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 15:15:00 | 4370.00 | 4341.04 | 4288.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 4140.15 | 4278.52 | 4289.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 4140.15 | 4278.52 | 4289.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 4112.80 | 4245.38 | 4273.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 12:15:00 | 4098.50 | 4080.04 | 4147.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 13:00:00 | 4098.50 | 4080.04 | 4147.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 4165.50 | 4106.62 | 4143.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:45:00 | 4100.00 | 4090.82 | 4133.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 15:15:00 | 4112.00 | 4097.50 | 4117.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 10:15:00 | 4087.60 | 4113.52 | 4121.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 13:15:00 | 4169.45 | 4104.19 | 4101.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 4169.45 | 4104.19 | 4101.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 4265.35 | 4155.44 | 4127.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 4365.05 | 4379.66 | 4304.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 10:00:00 | 4365.05 | 4379.66 | 4304.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 4346.80 | 4358.87 | 4317.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:15:00 | 4312.65 | 4358.87 | 4317.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 4309.70 | 4349.04 | 4316.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 4322.55 | 4349.04 | 4316.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 4260.00 | 4331.23 | 4311.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 4184.60 | 4331.23 | 4311.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 4240.00 | 4312.99 | 4305.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 4173.00 | 4312.99 | 4305.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 4254.95 | 4293.22 | 4297.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 4219.00 | 4278.38 | 4289.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 4284.30 | 4273.58 | 4285.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 14:15:00 | 4284.30 | 4273.58 | 4285.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 4284.30 | 4273.58 | 4285.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 4284.30 | 4273.58 | 4285.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 4321.15 | 4283.09 | 4288.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 4017.35 | 4283.09 | 4288.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 14:15:00 | 4337.85 | 4294.11 | 4288.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 4337.85 | 4294.11 | 4288.74 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 15:15:00 | 4248.00 | 4284.89 | 4285.03 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 4293.95 | 4286.70 | 4285.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 10:15:00 | 4374.15 | 4304.19 | 4293.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 4240.15 | 4308.44 | 4301.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 14:15:00 | 4240.15 | 4308.44 | 4301.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 4240.15 | 4308.44 | 4301.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 4240.15 | 4308.44 | 4301.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 4225.05 | 4291.76 | 4294.36 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 4313.40 | 4296.09 | 4296.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 4426.75 | 4341.53 | 4321.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 4516.40 | 4528.62 | 4470.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 4516.40 | 4528.62 | 4470.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 4902.30 | 4947.48 | 4865.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 4859.45 | 4947.48 | 4865.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 4834.00 | 4910.75 | 4873.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 4828.25 | 4910.75 | 4873.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 4954.25 | 4919.45 | 4880.68 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 13:15:00 | 4838.80 | 4881.98 | 4883.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 14:15:00 | 4746.50 | 4836.68 | 4857.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 14:15:00 | 4798.05 | 4795.69 | 4821.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 14:15:00 | 4798.05 | 4795.69 | 4821.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 4798.05 | 4795.69 | 4821.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 4798.05 | 4795.69 | 4821.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 4783.00 | 4789.20 | 4813.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 4797.40 | 4789.20 | 4813.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 4824.65 | 4796.29 | 4814.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 4824.65 | 4796.29 | 4814.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 4820.45 | 4801.12 | 4814.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 4820.45 | 4801.12 | 4814.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 4935.00 | 4827.90 | 4825.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 5030.00 | 4882.89 | 4852.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 13:15:00 | 4962.45 | 4966.64 | 4916.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 14:00:00 | 4962.45 | 4966.64 | 4916.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 4766.20 | 4923.08 | 4908.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 4766.20 | 4923.08 | 4908.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 4765.00 | 4891.47 | 4895.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 4708.55 | 4812.46 | 4854.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 4441.20 | 4434.77 | 4596.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 4441.20 | 4434.77 | 4596.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 4591.10 | 4478.88 | 4589.54 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 4825.00 | 4656.57 | 4644.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 5030.00 | 4817.34 | 4743.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 5564.70 | 5605.73 | 5435.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 5564.70 | 5605.73 | 5435.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 5923.80 | 5936.06 | 5851.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 5835.30 | 5936.06 | 5851.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 5876.30 | 5912.27 | 5874.14 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 5800.10 | 5857.99 | 5860.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 5604.90 | 5807.37 | 5837.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 5697.20 | 5643.97 | 5716.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 5697.20 | 5643.97 | 5716.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 5697.20 | 5643.97 | 5716.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 5697.20 | 5643.97 | 5716.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 5751.90 | 5665.56 | 5720.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 5751.90 | 5665.56 | 5720.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 5764.00 | 5685.25 | 5724.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 5775.00 | 5685.25 | 5724.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 5850.50 | 5743.81 | 5743.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 15:15:00 | 5938.00 | 5782.65 | 5760.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 10:15:00 | 5833.00 | 5872.97 | 5837.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 10:15:00 | 5833.00 | 5872.97 | 5837.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 5833.00 | 5872.97 | 5837.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:00:00 | 5833.00 | 5872.97 | 5837.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 5796.40 | 5857.66 | 5833.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 5802.50 | 5857.66 | 5833.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 5765.20 | 5839.17 | 5827.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:00:00 | 5765.20 | 5839.17 | 5827.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 5760.90 | 5813.49 | 5817.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 5709.00 | 5792.59 | 5807.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 5806.00 | 5763.82 | 5778.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 5806.00 | 5763.82 | 5778.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 5806.00 | 5763.82 | 5778.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 5760.00 | 5779.77 | 5781.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:45:00 | 5732.50 | 5773.82 | 5778.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 13:15:00 | 5816.00 | 5739.12 | 5737.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 5816.00 | 5739.12 | 5737.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 15:15:00 | 5840.00 | 5774.88 | 5755.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 5731.50 | 5812.56 | 5787.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 5731.50 | 5812.56 | 5787.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 5731.50 | 5812.56 | 5787.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 5731.50 | 5812.56 | 5787.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 5670.50 | 5784.15 | 5777.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 5670.50 | 5784.15 | 5777.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 5626.00 | 5752.52 | 5763.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 5580.00 | 5718.02 | 5746.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 5662.00 | 5650.15 | 5695.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 15:00:00 | 5662.00 | 5650.15 | 5695.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 5923.00 | 5704.70 | 5712.29 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 5986.00 | 5760.96 | 5737.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 6052.00 | 5936.28 | 5848.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 6240.50 | 6273.72 | 6168.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 10:00:00 | 6240.50 | 6273.72 | 6168.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 6277.00 | 6352.26 | 6294.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 6277.00 | 6352.26 | 6294.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 6260.00 | 6333.81 | 6290.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 6112.00 | 6333.81 | 6290.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 6186.00 | 6278.84 | 6271.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:30:00 | 6173.50 | 6278.84 | 6271.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 6156.00 | 6254.27 | 6261.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 13:15:00 | 6085.00 | 6199.57 | 6233.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 5952.00 | 5949.78 | 6024.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 5952.00 | 5949.78 | 6024.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 6065.00 | 5973.50 | 6022.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 6038.00 | 5973.50 | 6022.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 6011.50 | 5981.10 | 6021.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 5993.50 | 5981.10 | 6021.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 15:15:00 | 6017.00 | 6008.65 | 6007.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 6017.00 | 6008.65 | 6007.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 6075.50 | 6022.02 | 6013.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 12:15:00 | 5996.50 | 6025.72 | 6018.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 12:15:00 | 5996.50 | 6025.72 | 6018.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 5996.50 | 6025.72 | 6018.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:00:00 | 5996.50 | 6025.72 | 6018.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 5998.00 | 6020.18 | 6016.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:45:00 | 5996.50 | 6020.18 | 6016.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 5974.00 | 6010.94 | 6012.69 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 6038.00 | 6002.41 | 6002.19 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 5971.00 | 6005.71 | 6008.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 5965.50 | 5986.16 | 5996.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 5776.00 | 5768.81 | 5826.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 14:00:00 | 5776.00 | 5768.81 | 5826.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 5802.50 | 5765.41 | 5809.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 5861.50 | 5765.41 | 5809.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 5750.00 | 5762.33 | 5804.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 11:30:00 | 5740.00 | 5757.56 | 5798.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:00:00 | 5738.50 | 5757.56 | 5798.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 09:30:00 | 5735.00 | 5707.69 | 5725.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 5453.00 | 5493.31 | 5544.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 5451.57 | 5493.31 | 5544.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 5448.25 | 5493.31 | 5544.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 13:15:00 | 5505.00 | 5445.62 | 5500.05 | SL hit (close>ema200) qty=0.50 sl=5445.62 alert=retest2 |

### Cycle 72 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 5602.50 | 5488.59 | 5486.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 5610.00 | 5512.88 | 5497.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 5626.00 | 5674.80 | 5643.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 5626.00 | 5674.80 | 5643.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 5626.00 | 5674.80 | 5643.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 5626.00 | 5674.80 | 5643.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 5650.00 | 5669.84 | 5644.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 5619.00 | 5669.84 | 5644.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 5621.00 | 5660.07 | 5641.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 5621.00 | 5660.07 | 5641.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 5600.00 | 5648.06 | 5638.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 5672.50 | 5648.06 | 5638.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 5753.00 | 5797.96 | 5799.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 5753.00 | 5797.96 | 5799.08 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 5869.50 | 5793.77 | 5792.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 13:15:00 | 5915.00 | 5846.33 | 5820.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 10:15:00 | 6098.50 | 6104.44 | 6046.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:30:00 | 6072.50 | 6104.44 | 6046.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 6155.00 | 6204.49 | 6167.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 6214.00 | 6204.49 | 6167.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 6206.50 | 6204.89 | 6170.65 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 6105.50 | 6169.68 | 6173.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 6082.00 | 6152.14 | 6164.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 6204.00 | 6090.02 | 6118.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 6204.00 | 6090.02 | 6118.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 6204.00 | 6090.02 | 6118.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 6221.50 | 6090.02 | 6118.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 6146.50 | 6101.32 | 6120.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:15:00 | 6127.00 | 6124.60 | 6128.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 6175.00 | 6134.81 | 6131.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 6175.00 | 6134.81 | 6131.96 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 6038.50 | 6123.70 | 6134.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 6020.00 | 6102.96 | 6123.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 6026.00 | 6018.85 | 6052.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 6026.00 | 6018.85 | 6052.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 6049.00 | 6025.07 | 6049.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 6074.50 | 6025.07 | 6049.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 6053.00 | 6030.65 | 6049.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 6053.00 | 6030.65 | 6049.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 6002.00 | 6024.92 | 6045.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 12:15:00 | 5985.00 | 6024.92 | 6045.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:15:00 | 5963.00 | 6020.34 | 6041.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:45:00 | 5997.00 | 5998.48 | 6020.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 10:00:00 | 5991.00 | 5981.61 | 5999.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 5982.50 | 5981.79 | 5998.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:15:00 | 5972.50 | 5981.79 | 5998.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 12:15:00 | 5685.75 | 5740.30 | 5779.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 12:15:00 | 5697.15 | 5740.30 | 5779.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 12:15:00 | 5691.45 | 5740.30 | 5779.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 14:15:00 | 5664.85 | 5715.49 | 5760.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 14:15:00 | 5673.88 | 5715.49 | 5760.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 5741.00 | 5714.60 | 5752.11 | SL hit (close>ema200) qty=0.50 sl=5714.60 alert=retest2 |

### Cycle 78 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 5612.00 | 5602.36 | 5601.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 5662.50 | 5614.39 | 5607.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 6306.50 | 6313.40 | 6137.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:00:00 | 6306.50 | 6313.40 | 6137.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 6080.50 | 6293.20 | 6266.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:45:00 | 6089.00 | 6293.20 | 6266.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 6051.00 | 6244.76 | 6246.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 6003.00 | 6076.74 | 6144.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 6085.00 | 6055.44 | 6110.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 6085.00 | 6055.44 | 6110.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 5888.00 | 6028.28 | 6088.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:30:00 | 5874.50 | 5943.38 | 6021.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 6115.50 | 5929.62 | 5914.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 6115.50 | 5929.62 | 5914.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 6294.00 | 6135.52 | 6062.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 6249.00 | 6274.49 | 6210.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:00:00 | 6249.00 | 6274.49 | 6210.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 6263.00 | 6271.54 | 6225.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 6301.00 | 6271.54 | 6225.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 13:15:00 | 6182.50 | 6214.05 | 6217.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 6182.50 | 6214.05 | 6217.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 6139.50 | 6199.14 | 6209.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 6194.00 | 6169.94 | 6188.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 12:15:00 | 6194.00 | 6169.94 | 6188.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 6194.00 | 6169.94 | 6188.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 6202.00 | 6169.94 | 6188.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 6227.00 | 6181.35 | 6191.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 6227.00 | 6181.35 | 6191.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 6247.50 | 6203.76 | 6200.16 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 6180.00 | 6217.40 | 6218.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 6124.00 | 6198.72 | 6209.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 12:15:00 | 6207.00 | 6196.33 | 6205.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 12:15:00 | 6207.00 | 6196.33 | 6205.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 6207.00 | 6196.33 | 6205.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:00:00 | 6207.00 | 6196.33 | 6205.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 6212.00 | 6199.46 | 6205.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 6212.00 | 6199.46 | 6205.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 6192.00 | 6197.97 | 6204.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 6208.00 | 6197.97 | 6204.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 6155.00 | 6189.38 | 6200.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 6205.50 | 6189.38 | 6200.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 6146.50 | 6180.80 | 6195.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:15:00 | 6123.50 | 6164.33 | 6182.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 6355.00 | 6189.86 | 6188.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 6355.00 | 6189.86 | 6188.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 6441.50 | 6240.19 | 6211.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 7077.50 | 7079.99 | 7010.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:45:00 | 7076.00 | 7079.99 | 7010.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 7227.00 | 7147.23 | 7103.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:45:00 | 7253.50 | 7174.83 | 7158.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:00:00 | 7260.00 | 7198.90 | 7175.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 7305.00 | 7231.94 | 7211.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 13:15:00 | 7141.00 | 7199.74 | 7205.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 7141.00 | 7199.74 | 7205.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 7075.00 | 7148.92 | 7173.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 7151.50 | 7144.73 | 7166.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 7151.50 | 7144.73 | 7166.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 7151.50 | 7144.73 | 7166.89 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 7250.00 | 7193.12 | 7185.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 7450.00 | 7264.60 | 7223.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 7499.50 | 7511.12 | 7422.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 7499.50 | 7511.12 | 7422.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 7463.00 | 7498.91 | 7431.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 7415.50 | 7498.91 | 7431.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 7423.00 | 7483.73 | 7431.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 7423.00 | 7483.73 | 7431.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 7381.50 | 7463.29 | 7426.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 7378.00 | 7463.29 | 7426.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 7386.00 | 7447.83 | 7422.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:15:00 | 7355.50 | 7447.83 | 7422.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 7330.50 | 7424.36 | 7414.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:45:00 | 7305.00 | 7424.36 | 7414.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 15:15:00 | 7340.00 | 7394.71 | 7401.97 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 7461.50 | 7408.07 | 7407.38 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 10:15:00 | 7212.00 | 7368.86 | 7389.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 7165.50 | 7328.18 | 7369.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 7143.50 | 7093.39 | 7167.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 7143.50 | 7093.39 | 7167.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 7143.50 | 7093.39 | 7167.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 7153.00 | 7093.39 | 7167.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 7174.50 | 7116.03 | 7165.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 7207.00 | 7116.03 | 7165.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 7157.50 | 7124.32 | 7164.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:30:00 | 7174.50 | 7124.32 | 7164.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 7158.00 | 7131.06 | 7163.95 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 7215.00 | 7181.41 | 7177.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 7277.50 | 7206.88 | 7190.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 7205.50 | 7221.82 | 7202.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 7205.50 | 7221.82 | 7202.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 7205.50 | 7221.82 | 7202.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 7208.50 | 7221.82 | 7202.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 7233.00 | 7224.06 | 7205.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:15:00 | 7259.00 | 7224.06 | 7205.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 13:15:00 | 7247.50 | 7376.20 | 7382.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 7247.50 | 7376.20 | 7382.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 7184.00 | 7337.76 | 7364.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 6918.50 | 6886.46 | 6989.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 6918.50 | 6886.46 | 6989.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 6918.50 | 6886.46 | 6989.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 6950.50 | 6886.46 | 6989.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 6932.00 | 6897.50 | 6944.82 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 7035.00 | 6961.51 | 6959.03 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 6960.00 | 6990.57 | 6991.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 6916.00 | 6975.65 | 6984.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 6713.00 | 6708.83 | 6767.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:15:00 | 6708.00 | 6708.83 | 6767.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 6735.00 | 6714.47 | 6748.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 6735.00 | 6714.47 | 6748.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 6735.00 | 6718.57 | 6747.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 6925.00 | 6718.57 | 6747.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 6885.00 | 6751.86 | 6759.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 6973.50 | 6751.86 | 6759.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 6951.00 | 6791.69 | 6776.97 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 6807.00 | 6823.96 | 6825.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 6763.50 | 6811.87 | 6820.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 10:15:00 | 6698.00 | 6678.09 | 6713.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 6698.00 | 6678.09 | 6713.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 6698.00 | 6678.09 | 6713.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 6698.00 | 6678.09 | 6713.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 6686.50 | 6652.25 | 6688.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 6686.50 | 6652.25 | 6688.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 6678.00 | 6657.40 | 6687.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 6599.00 | 6657.40 | 6687.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 6534.50 | 6632.82 | 6673.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:00:00 | 6473.50 | 6600.95 | 6655.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 6433.00 | 6406.50 | 6403.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 6433.00 | 6406.50 | 6403.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 6466.50 | 6427.14 | 6414.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 12:15:00 | 6437.00 | 6439.27 | 6426.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 13:00:00 | 6437.00 | 6439.27 | 6426.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 6437.50 | 6437.83 | 6427.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:45:00 | 6431.50 | 6437.83 | 6427.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 6423.00 | 6434.87 | 6427.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 6457.50 | 6434.87 | 6427.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 6434.00 | 6434.69 | 6427.90 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 6402.50 | 6421.78 | 6423.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 6387.50 | 6414.92 | 6420.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 5978.00 | 5947.59 | 6035.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 5978.00 | 5947.59 | 6035.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 5970.00 | 5953.82 | 6023.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:15:00 | 5957.00 | 5953.82 | 6023.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:45:00 | 5958.00 | 5974.86 | 5999.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:15:00 | 5920.00 | 5974.86 | 5999.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 12:30:00 | 5955.50 | 5962.27 | 5988.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 5845.00 | 5813.67 | 5854.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:00:00 | 5822.50 | 5815.44 | 5851.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:30:00 | 5806.00 | 5811.68 | 5843.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:15:00 | 5659.15 | 5764.53 | 5811.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:15:00 | 5660.10 | 5764.53 | 5811.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:15:00 | 5624.00 | 5764.53 | 5811.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:15:00 | 5657.72 | 5764.53 | 5811.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 10:15:00 | 5531.38 | 5589.65 | 5678.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 10:15:00 | 5515.70 | 5589.65 | 5678.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-01 11:15:00 | 5361.30 | 5466.43 | 5558.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 98 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 4241.50 | 4142.50 | 4134.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 4267.50 | 4182.78 | 4155.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 11:15:00 | 4212.00 | 4233.76 | 4192.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 11:15:00 | 4212.00 | 4233.76 | 4192.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 4212.00 | 4233.76 | 4192.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 4198.50 | 4233.76 | 4192.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 4203.50 | 4229.15 | 4200.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 4203.50 | 4229.15 | 4200.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 4195.50 | 4222.42 | 4200.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 4181.00 | 4222.42 | 4200.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 4144.50 | 4206.84 | 4195.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 4144.50 | 4206.84 | 4195.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 4173.00 | 4200.07 | 4193.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:30:00 | 4181.50 | 4204.76 | 4196.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 15:00:00 | 4186.50 | 4197.32 | 4194.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 10:45:00 | 4198.50 | 4194.51 | 4193.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:30:00 | 4192.50 | 4194.01 | 4193.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 4174.00 | 4190.00 | 4191.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 4174.00 | 4190.00 | 4191.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 4159.50 | 4183.90 | 4188.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 4095.00 | 4073.99 | 4104.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 11:15:00 | 4095.00 | 4073.99 | 4104.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 4095.00 | 4073.99 | 4104.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 4095.00 | 4073.99 | 4104.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 4121.00 | 4083.39 | 4105.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 4140.00 | 4083.39 | 4105.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 4144.00 | 4095.51 | 4109.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 4144.00 | 4095.51 | 4109.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 4181.00 | 4126.53 | 4121.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 4206.00 | 4142.42 | 4129.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 4188.50 | 4193.43 | 4167.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 4161.50 | 4193.43 | 4167.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 4168.00 | 4188.34 | 4167.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:45:00 | 4159.00 | 4188.34 | 4167.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 4159.50 | 4182.58 | 4166.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 4157.00 | 4182.58 | 4166.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 4155.00 | 4177.06 | 4165.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 4156.50 | 4177.06 | 4165.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 4152.00 | 4168.28 | 4163.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 4152.00 | 4168.28 | 4163.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 4138.00 | 4157.86 | 4159.11 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 4171.00 | 4161.47 | 4160.60 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 4153.00 | 4159.05 | 4159.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 4117.50 | 4150.74 | 4155.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 3977.00 | 3961.29 | 4001.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 3977.00 | 3961.29 | 4001.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 3977.00 | 3961.29 | 4001.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 3944.80 | 3977.03 | 3993.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:30:00 | 3935.80 | 3965.08 | 3984.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:45:00 | 3944.50 | 3950.54 | 3959.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 4069.90 | 3984.58 | 3973.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 4069.90 | 3984.58 | 3973.72 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 3942.00 | 3975.11 | 3975.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 3743.00 | 3923.85 | 3951.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 3688.20 | 3674.32 | 3724.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 3682.40 | 3674.32 | 3724.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 3690.00 | 3685.24 | 3714.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 3675.00 | 3684.19 | 3711.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 3669.70 | 3681.29 | 3707.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 3667.60 | 3676.40 | 3702.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 3740.80 | 3692.88 | 3701.92 | SL hit (close>static) qty=1.00 sl=3738.80 alert=retest2 |

### Cycle 106 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 3486.60 | 3435.23 | 3428.88 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 3368.50 | 3415.37 | 3420.55 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 3547.70 | 3426.41 | 3419.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 3619.10 | 3502.08 | 3463.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 3478.90 | 3533.63 | 3501.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 3478.90 | 3533.63 | 3501.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 3478.90 | 3533.63 | 3501.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:30:00 | 3476.60 | 3533.63 | 3501.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 3382.80 | 3503.46 | 3490.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 3382.80 | 3503.46 | 3490.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 3389.20 | 3469.18 | 3476.66 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 3625.70 | 3495.51 | 3484.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 12:15:00 | 3747.50 | 3657.21 | 3593.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 3602.00 | 3692.97 | 3635.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 3602.00 | 3692.97 | 3635.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3602.00 | 3692.97 | 3635.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 3620.10 | 3692.97 | 3635.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 3593.90 | 3673.16 | 3631.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:15:00 | 3627.10 | 3662.17 | 3630.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 3630.70 | 3649.07 | 3629.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:45:00 | 3626.40 | 3643.25 | 3629.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 3493.00 | 3606.12 | 3614.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 3493.00 | 3606.12 | 3614.20 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 3707.00 | 3610.55 | 3608.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 3836.00 | 3672.03 | 3638.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 4083.90 | 4091.72 | 3999.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 10:00:00 | 4083.90 | 4091.72 | 3999.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 3943.00 | 4060.83 | 4030.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 3952.80 | 4060.83 | 4030.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 3916.40 | 4031.95 | 4019.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 3916.40 | 4031.95 | 4019.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 3969.50 | 4005.32 | 4009.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 10:15:00 | 3936.00 | 3972.39 | 3985.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 3951.00 | 3950.15 | 3969.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 3951.00 | 3950.15 | 3969.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 3968.00 | 3953.72 | 3968.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 3994.30 | 3953.72 | 3968.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 3927.30 | 3948.43 | 3965.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 3917.40 | 3948.43 | 3965.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 11:00:00 | 3919.00 | 3942.55 | 3960.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 4020.50 | 3967.32 | 3968.44 | SL hit (close>static) qty=1.00 sl=4019.80 alert=retest2 |

### Cycle 114 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 4058.60 | 3985.58 | 3976.64 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 3922.30 | 3972.45 | 3974.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 3874.00 | 3939.85 | 3957.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 3904.30 | 3904.17 | 3932.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:00:00 | 3904.30 | 3904.17 | 3932.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 3833.10 | 3803.16 | 3833.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 3833.10 | 3803.16 | 3833.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 3860.00 | 3814.53 | 3836.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 3893.50 | 3814.53 | 3836.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 3890.60 | 3829.74 | 3841.00 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 3961.00 | 3855.99 | 3851.91 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 3860.00 | 3887.34 | 3889.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 3846.20 | 3879.11 | 3885.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 3699.20 | 3692.17 | 3746.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 15:00:00 | 3699.20 | 3692.17 | 3746.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3661.80 | 3687.51 | 3735.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:30:00 | 3651.00 | 3679.55 | 3727.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 3803.00 | 3712.21 | 3728.11 | SL hit (close>static) qty=1.00 sl=3754.90 alert=retest2 |

### Cycle 118 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 3796.00 | 3743.97 | 3740.72 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 3623.80 | 3731.79 | 3739.91 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 3759.00 | 3730.51 | 3727.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 3837.90 | 3764.50 | 3744.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 3796.30 | 3797.39 | 3769.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:45:00 | 3797.70 | 3797.39 | 3769.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 3768.10 | 3791.53 | 3769.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 3768.10 | 3791.53 | 3769.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 3744.70 | 3782.17 | 3767.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 3744.70 | 3782.17 | 3767.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 3719.30 | 3769.59 | 3762.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 3719.30 | 3769.59 | 3762.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 3715.10 | 3758.70 | 3758.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 3613.60 | 3758.70 | 3758.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 3594.20 | 3725.80 | 3743.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 3550.10 | 3674.41 | 3706.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 3560.00 | 3524.48 | 3573.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 3560.00 | 3524.48 | 3573.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 3560.00 | 3524.48 | 3573.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 3562.30 | 3524.48 | 3573.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3628.20 | 3547.70 | 3575.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 3628.20 | 3547.70 | 3575.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3618.80 | 3561.92 | 3579.70 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 3633.60 | 3592.45 | 3590.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 3677.00 | 3622.67 | 3606.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3619.00 | 3672.14 | 3647.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3619.00 | 3672.14 | 3647.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3619.00 | 3672.14 | 3647.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 3626.60 | 3672.14 | 3647.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 3624.30 | 3662.57 | 3645.21 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 3580.00 | 3632.31 | 3634.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 3565.30 | 3618.91 | 3628.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 3665.00 | 3621.74 | 3627.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 3665.00 | 3621.74 | 3627.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 3665.00 | 3621.74 | 3627.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 3665.00 | 3621.74 | 3627.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 3687.30 | 3634.85 | 3632.97 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 3623.70 | 3634.35 | 3634.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 3472.70 | 3602.02 | 3619.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 3486.40 | 3455.68 | 3506.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 3486.40 | 3455.68 | 3506.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 3486.40 | 3455.68 | 3506.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 3486.40 | 3455.68 | 3506.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 3542.70 | 3473.08 | 3509.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 3545.50 | 3473.08 | 3509.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 3512.00 | 3480.86 | 3509.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 3537.50 | 3480.86 | 3509.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 3513.70 | 3487.43 | 3510.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 3517.00 | 3487.43 | 3510.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 3518.00 | 3493.55 | 3511.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 3632.00 | 3493.55 | 3511.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 3729.80 | 3540.80 | 3530.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 3756.60 | 3583.96 | 3551.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 3618.90 | 3664.87 | 3616.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 3618.90 | 3664.87 | 3616.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 3618.90 | 3664.87 | 3616.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 3618.90 | 3664.87 | 3616.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3609.90 | 3653.88 | 3615.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 3609.90 | 3653.88 | 3615.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 3647.80 | 3652.66 | 3618.84 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 3498.00 | 3593.73 | 3603.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 3478.60 | 3544.76 | 3576.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3643.80 | 3530.17 | 3559.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3643.80 | 3530.17 | 3559.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3643.80 | 3530.17 | 3559.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 3570.00 | 3537.14 | 3559.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 3538.00 | 3557.20 | 3563.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 3391.50 | 3522.99 | 3547.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 3497.70 | 3487.41 | 3519.21 | SL hit (close>ema200) qty=0.50 sl=3487.41 alert=retest2 |

### Cycle 128 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 3610.70 | 3548.74 | 3540.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 3645.00 | 3567.99 | 3549.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 3834.40 | 3847.73 | 3771.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 3828.40 | 3847.73 | 3771.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3816.90 | 3870.43 | 3845.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 3860.00 | 3866.60 | 3845.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 3855.00 | 3866.60 | 3845.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 4246.00 | 4151.84 | 4064.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 4263.00 | 4334.89 | 4335.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 09:15:00 | 4233.00 | 4288.02 | 4310.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 10:15:00 | 4166.00 | 4147.08 | 4192.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 10:45:00 | 4160.00 | 4147.08 | 4192.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 4191.90 | 4159.76 | 4190.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 4191.90 | 4159.76 | 4190.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 4121.80 | 4152.17 | 4184.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 4039.90 | 4138.95 | 4172.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 4185.00 | 4099.65 | 4098.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 4185.00 | 4099.65 | 4098.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 4226.60 | 4168.78 | 4139.97 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 2836.60 | 2024-05-18 09:15:00 | 3120.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-29 14:45:00 | 3294.40 | 2024-05-31 12:15:00 | 3351.35 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-05-30 09:45:00 | 3287.45 | 2024-05-31 12:15:00 | 3351.35 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-05-30 12:15:00 | 3241.00 | 2024-05-31 12:15:00 | 3351.35 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2024-05-30 15:15:00 | 3295.00 | 2024-05-31 12:15:00 | 3351.35 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-06-25 09:15:00 | 3945.15 | 2024-06-25 12:15:00 | 3870.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-06-25 14:15:00 | 3921.65 | 2024-06-27 14:15:00 | 3880.10 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-07-04 12:15:00 | 4044.00 | 2024-07-08 10:15:00 | 4448.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-05 09:15:00 | 4093.90 | 2024-07-08 11:15:00 | 4503.29 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-11 11:15:00 | 4125.00 | 2024-07-16 11:15:00 | 4253.00 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2024-07-12 14:00:00 | 4129.95 | 2024-07-16 11:15:00 | 4253.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2024-07-15 09:15:00 | 4088.00 | 2024-07-16 11:15:00 | 4253.00 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2024-07-23 12:15:00 | 3812.25 | 2024-07-23 14:15:00 | 4062.30 | STOP_HIT | 1.00 | -6.56% |
| SELL | retest2 | 2024-08-05 09:15:00 | 4132.50 | 2024-08-07 15:15:00 | 4279.80 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-08-12 13:00:00 | 4309.70 | 2024-08-14 11:15:00 | 4740.67 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-26 14:15:00 | 5007.05 | 2024-08-29 10:15:00 | 4756.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-26 14:45:00 | 4999.80 | 2024-08-29 10:15:00 | 4749.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-27 14:30:00 | 5000.10 | 2024-08-29 10:15:00 | 4750.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-26 14:15:00 | 5007.05 | 2024-08-30 11:15:00 | 4797.25 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2024-08-26 14:45:00 | 4999.80 | 2024-08-30 11:15:00 | 4797.25 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2024-08-27 14:30:00 | 5000.10 | 2024-08-30 11:15:00 | 4797.25 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2024-09-11 11:45:00 | 4651.40 | 2024-09-12 09:15:00 | 4977.00 | STOP_HIT | 1.00 | -7.00% |
| SELL | retest2 | 2024-09-11 12:15:00 | 4655.70 | 2024-09-12 09:15:00 | 4977.00 | STOP_HIT | 1.00 | -6.90% |
| SELL | retest2 | 2024-09-11 12:45:00 | 4657.00 | 2024-09-12 09:15:00 | 4977.00 | STOP_HIT | 1.00 | -6.87% |
| SELL | retest2 | 2024-09-11 15:15:00 | 4650.00 | 2024-09-12 09:15:00 | 4977.00 | STOP_HIT | 1.00 | -7.03% |
| BUY | retest2 | 2024-09-24 09:15:00 | 5790.70 | 2024-09-26 09:15:00 | 5458.85 | STOP_HIT | 1.00 | -5.73% |
| SELL | retest2 | 2024-10-29 10:15:00 | 5279.15 | 2024-10-31 09:15:00 | 5460.00 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-10-30 10:00:00 | 5280.00 | 2024-10-31 09:15:00 | 5460.00 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2024-10-30 11:00:00 | 5296.70 | 2024-10-31 09:15:00 | 5460.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-10-30 14:00:00 | 5295.90 | 2024-10-31 09:15:00 | 5460.00 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2024-10-31 09:15:00 | 5242.30 | 2024-10-31 09:15:00 | 5460.00 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2024-10-31 15:15:00 | 5461.00 | 2024-11-05 09:15:00 | 5301.35 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2024-11-04 09:45:00 | 5450.80 | 2024-11-05 09:15:00 | 5301.35 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-11-04 13:00:00 | 5459.00 | 2024-11-05 09:15:00 | 5301.35 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-11-22 09:15:00 | 5875.50 | 2024-11-28 10:15:00 | 5886.45 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-11-22 10:15:00 | 5847.45 | 2024-11-28 10:15:00 | 5886.45 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2024-11-25 09:15:00 | 5907.45 | 2024-11-28 10:15:00 | 5886.45 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-12-09 09:15:00 | 6271.90 | 2024-12-16 09:15:00 | 6899.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-09 10:15:00 | 6243.85 | 2024-12-16 09:15:00 | 6868.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-09 10:45:00 | 6270.25 | 2024-12-16 09:15:00 | 6897.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-09 14:00:00 | 6244.50 | 2024-12-16 09:15:00 | 6868.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-10 11:30:00 | 6338.00 | 2024-12-16 09:15:00 | 6971.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-31 09:15:00 | 7146.00 | 2024-12-31 10:15:00 | 7214.70 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-01-08 09:15:00 | 7185.20 | 2025-01-09 09:15:00 | 6825.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 7185.20 | 2025-01-13 09:15:00 | 6466.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 11:15:00 | 5692.45 | 2025-01-27 09:15:00 | 5407.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 11:00:00 | 5699.95 | 2025-01-27 09:15:00 | 5414.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:15:00 | 5692.45 | 2025-01-28 09:15:00 | 5123.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 11:00:00 | 5699.95 | 2025-01-28 09:15:00 | 5129.95 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-24 11:15:00 | 4341.20 | 2025-02-27 09:15:00 | 4140.15 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2025-02-24 14:45:00 | 4346.40 | 2025-02-27 09:15:00 | 4140.15 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2025-02-24 15:15:00 | 4370.00 | 2025-02-27 09:15:00 | 4140.15 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2025-03-03 09:45:00 | 4100.00 | 2025-03-05 13:15:00 | 4169.45 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-03-03 15:15:00 | 4112.00 | 2025-03-05 13:15:00 | 4169.45 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-03-04 10:15:00 | 4087.60 | 2025-03-05 13:15:00 | 4169.45 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-03-12 09:15:00 | 4017.35 | 2025-03-12 14:15:00 | 4337.85 | STOP_HIT | 1.00 | -7.98% |
| SELL | retest2 | 2025-05-06 09:15:00 | 5760.00 | 2025-05-07 13:15:00 | 5816.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-05-06 09:45:00 | 5732.50 | 2025-05-07 13:15:00 | 5816.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-05-22 11:15:00 | 5993.50 | 2025-05-23 15:15:00 | 6017.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-06-05 11:30:00 | 5740.00 | 2025-06-13 09:15:00 | 5453.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 12:00:00 | 5738.50 | 2025-06-13 09:15:00 | 5451.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 09:30:00 | 5735.00 | 2025-06-13 09:15:00 | 5448.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 11:30:00 | 5740.00 | 2025-06-13 13:15:00 | 5505.00 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-06-05 12:00:00 | 5738.50 | 2025-06-13 13:15:00 | 5505.00 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-06-09 09:30:00 | 5735.00 | 2025-06-13 13:15:00 | 5505.00 | STOP_HIT | 0.50 | 4.01% |
| BUY | retest2 | 2025-06-20 09:15:00 | 5672.50 | 2025-06-26 10:15:00 | 5753.00 | STOP_HIT | 1.00 | 1.42% |
| SELL | retest2 | 2025-07-09 13:15:00 | 6127.00 | 2025-07-10 09:15:00 | 6175.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-15 12:15:00 | 5985.00 | 2025-07-24 12:15:00 | 5685.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 13:15:00 | 5963.00 | 2025-07-24 12:15:00 | 5697.15 | PARTIAL | 0.50 | 4.46% |
| SELL | retest2 | 2025-07-16 10:45:00 | 5997.00 | 2025-07-24 12:15:00 | 5691.45 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-07-17 10:00:00 | 5991.00 | 2025-07-24 14:15:00 | 5664.85 | PARTIAL | 0.50 | 5.44% |
| SELL | retest2 | 2025-07-17 11:15:00 | 5972.50 | 2025-07-24 14:15:00 | 5673.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 12:15:00 | 5985.00 | 2025-07-25 09:15:00 | 5741.00 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2025-07-15 13:15:00 | 5963.00 | 2025-07-25 09:15:00 | 5741.00 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2025-07-16 10:45:00 | 5997.00 | 2025-07-25 09:15:00 | 5741.00 | STOP_HIT | 0.50 | 4.27% |
| SELL | retest2 | 2025-07-17 10:00:00 | 5991.00 | 2025-07-25 09:15:00 | 5741.00 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2025-07-17 11:15:00 | 5972.50 | 2025-07-25 09:15:00 | 5741.00 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2025-08-08 14:30:00 | 5874.50 | 2025-08-13 11:15:00 | 6115.50 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-08-20 10:15:00 | 6301.00 | 2025-08-21 13:15:00 | 6182.50 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-08-29 14:15:00 | 6123.50 | 2025-09-01 09:15:00 | 6355.00 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-09-17 10:45:00 | 7253.50 | 2025-09-19 13:15:00 | 7141.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-09-17 15:00:00 | 7260.00 | 2025-09-19 13:15:00 | 7141.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-19 09:15:00 | 7305.00 | 2025-09-19 13:15:00 | 7141.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-06 12:15:00 | 7259.00 | 2025-10-08 13:15:00 | 7247.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-11-06 11:00:00 | 6473.50 | 2025-11-11 10:15:00 | 6433.00 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-11-20 10:15:00 | 5957.00 | 2025-11-27 09:15:00 | 5659.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 10:45:00 | 5958.00 | 2025-11-27 09:15:00 | 5660.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 11:15:00 | 5920.00 | 2025-11-27 09:15:00 | 5624.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 12:30:00 | 5955.50 | 2025-11-27 09:15:00 | 5657.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 11:00:00 | 5822.50 | 2025-11-28 10:15:00 | 5531.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 12:30:00 | 5806.00 | 2025-11-28 10:15:00 | 5515.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 10:15:00 | 5957.00 | 2025-12-01 11:15:00 | 5361.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-21 10:45:00 | 5958.00 | 2025-12-01 11:15:00 | 5362.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-21 11:15:00 | 5920.00 | 2025-12-01 11:15:00 | 5359.95 | TARGET_HIT | 0.50 | 9.46% |
| SELL | retest2 | 2025-11-21 12:30:00 | 5955.50 | 2025-12-01 12:15:00 | 5328.00 | TARGET_HIT | 0.50 | 10.54% |
| SELL | retest2 | 2025-11-26 11:00:00 | 5822.50 | 2025-12-02 09:15:00 | 5428.00 | STOP_HIT | 0.50 | 6.78% |
| SELL | retest2 | 2025-11-26 12:30:00 | 5806.00 | 2025-12-02 09:15:00 | 5428.00 | STOP_HIT | 0.50 | 6.51% |
| BUY | retest2 | 2025-12-16 11:30:00 | 4181.50 | 2025-12-17 12:15:00 | 4174.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-12-16 15:00:00 | 4186.50 | 2025-12-17 12:15:00 | 4174.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-12-17 10:45:00 | 4198.50 | 2025-12-17 12:15:00 | 4174.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-12-17 11:30:00 | 4192.50 | 2025-12-17 12:15:00 | 4174.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-01-01 12:00:00 | 3944.80 | 2026-01-05 12:15:00 | 4069.90 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2026-01-01 13:30:00 | 3935.80 | 2026-01-05 12:15:00 | 4069.90 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2026-01-05 09:45:00 | 3944.50 | 2026-01-05 12:15:00 | 4069.90 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2026-01-13 10:45:00 | 3675.00 | 2026-01-14 09:15:00 | 3740.80 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-01-13 12:00:00 | 3669.70 | 2026-01-14 09:15:00 | 3740.80 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-01-13 12:45:00 | 3667.60 | 2026-01-14 09:15:00 | 3740.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 3673.00 | 2026-01-20 13:15:00 | 3489.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:45:00 | 3675.00 | 2026-01-20 13:15:00 | 3491.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:15:00 | 3667.00 | 2026-01-20 13:15:00 | 3483.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 3673.00 | 2026-01-21 12:15:00 | 3515.00 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2026-01-16 11:45:00 | 3675.00 | 2026-01-21 12:15:00 | 3515.00 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2026-01-16 12:15:00 | 3667.00 | 2026-01-21 12:15:00 | 3515.00 | STOP_HIT | 0.50 | 4.15% |
| BUY | retest2 | 2026-02-05 12:15:00 | 3627.10 | 2026-02-06 09:15:00 | 3493.00 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2026-02-05 13:30:00 | 3630.70 | 2026-02-06 09:15:00 | 3493.00 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2026-02-05 14:45:00 | 3626.40 | 2026-02-06 09:15:00 | 3493.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2026-02-18 10:15:00 | 3917.40 | 2026-02-18 13:15:00 | 4020.50 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-02-18 11:00:00 | 3919.00 | 2026-02-18 13:15:00 | 4020.50 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-03-05 10:30:00 | 3651.00 | 2026-03-05 14:15:00 | 3803.00 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2026-04-01 10:45:00 | 3570.00 | 2026-04-02 09:15:00 | 3391.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:45:00 | 3570.00 | 2026-04-02 13:15:00 | 3497.70 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2026-04-01 14:30:00 | 3538.00 | 2026-04-06 11:15:00 | 3610.70 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-06 10:15:00 | 3568.00 | 2026-04-06 11:15:00 | 3610.70 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-04-06 11:00:00 | 3562.80 | 2026-04-06 11:15:00 | 3610.70 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-04-13 10:45:00 | 3860.00 | 2026-04-17 09:15:00 | 4246.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 11:15:00 | 3855.00 | 2026-04-17 09:15:00 | 4240.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 09:15:00 | 4039.90 | 2026-05-05 09:15:00 | 4185.00 | STOP_HIT | 1.00 | -3.59% |
