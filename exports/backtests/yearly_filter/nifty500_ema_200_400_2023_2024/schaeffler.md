# Schaeffler India Ltd. (SCHAEFFLER)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4226.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 4 |
| ALERT3 | 63 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 40 |
| PARTIAL | 19 |
| TARGET_HIT | 14 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 28
- **Target hits / Stop hits / Partials:** 14 / 30 / 19
- **Avg / median % per leg:** 2.71% / 4.32%
- **Sum % (uncompounded):** 170.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 9 | 39.1% | 5 | 14 | 4 | 1.61% | 37.0% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 15 | 1 | 6.7% | 1 | 14 | 0 | -1.53% | -23.0% |
| SELL (all) | 40 | 26 | 65.0% | 9 | 16 | 15 | 3.34% | 133.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 40 | 26 | 65.0% | 9 | 16 | 15 | 3.34% | 133.5% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 55 | 27 | 49.1% | 10 | 30 | 15 | 2.01% | 110.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 2968.40 | 3138.11 | 3138.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 09:15:00 | 2927.50 | 3117.97 | 3128.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 09:15:00 | 2859.85 | 2853.74 | 2939.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-01 09:45:00 | 2863.55 | 2853.74 | 2939.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 12:15:00 | 2935.00 | 2856.53 | 2937.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-04 12:45:00 | 2930.00 | 2856.53 | 2937.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 13:15:00 | 2927.00 | 2857.23 | 2936.98 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 3150.00 | 2982.76 | 2982.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 12:15:00 | 3156.75 | 2984.49 | 2983.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 12:15:00 | 3184.15 | 3190.02 | 3118.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-23 12:45:00 | 3183.70 | 3190.02 | 3118.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 3115.00 | 3189.02 | 3118.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 14:45:00 | 3144.15 | 3189.02 | 3118.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 15:15:00 | 3126.00 | 3188.39 | 3118.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:15:00 | 3140.70 | 3188.39 | 3118.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 3114.35 | 3187.66 | 3118.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-05 09:30:00 | 3170.60 | 3161.82 | 3119.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 13:15:00 | 3033.05 | 3141.76 | 3113.56 | SL hit (close<static) qty=1.00 sl=3041.90 alert=retest2 |

### Cycle 3 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 2984.95 | 3090.55 | 3090.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 11:15:00 | 2962.30 | 3089.28 | 3090.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 10:15:00 | 2930.00 | 2926.23 | 2979.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-21 10:45:00 | 2933.00 | 2926.23 | 2979.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 13:15:00 | 2964.15 | 2926.69 | 2979.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 13:45:00 | 2959.30 | 2926.69 | 2979.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 2952.00 | 2927.69 | 2978.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 12:30:00 | 2946.00 | 2928.53 | 2978.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 09:15:00 | 2798.70 | 2928.74 | 2977.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-26 10:15:00 | 2939.65 | 2928.85 | 2977.58 | SL hit (close>ema200) qty=0.50 sl=2928.85 alert=retest2 |

### Cycle 4 — BUY (started 2024-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 12:15:00 | 3258.35 | 3007.79 | 3006.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 09:15:00 | 3296.60 | 3043.76 | 3025.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 12:15:00 | 3928.25 | 4077.10 | 3742.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-04 13:30:00 | 4220.95 | 4079.25 | 3745.17 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 12:45:00 | 4178.40 | 4081.76 | 3756.32 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 14:45:00 | 4179.95 | 4082.95 | 3760.16 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:15:00 | 4214.80 | 4083.47 | 3762.03 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:15:00 | 4387.32 | 4116.87 | 3816.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:15:00 | 4388.95 | 4116.87 | 3816.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 12:15:00 | 4432.00 | 4119.82 | 3819.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 12:15:00 | 4425.54 | 4119.82 | 3819.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-06-14 09:15:00 | 4643.05 | 4176.00 | 3874.24 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 5 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 3977.10 | 4108.96 | 4109.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 3930.05 | 4103.34 | 4106.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 13:15:00 | 4073.85 | 4061.41 | 4082.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-21 13:30:00 | 4073.45 | 4061.41 | 4082.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 4119.80 | 4061.99 | 4082.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:45:00 | 4119.65 | 4061.99 | 4082.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 4130.00 | 4062.66 | 4083.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:15:00 | 4123.10 | 4062.66 | 4083.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 4064.85 | 4063.10 | 4082.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 12:00:00 | 4064.85 | 4063.10 | 4082.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 4074.90 | 4063.27 | 4082.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:00:00 | 4074.90 | 4063.27 | 4082.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 4064.00 | 4063.17 | 4082.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 4030.10 | 4064.89 | 4082.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:15:00 | 3828.59 | 4005.61 | 4046.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-13 13:15:00 | 3954.55 | 3946.46 | 4003.32 | SL hit (close>ema200) qty=0.50 sl=3946.46 alert=retest2 |

### Cycle 6 — BUY (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 14:15:00 | 3492.70 | 3315.39 | 3315.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 3529.00 | 3319.23 | 3316.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 3297.75 | 3343.68 | 3330.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 3297.75 | 3343.68 | 3330.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 3297.75 | 3343.68 | 3330.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 3297.75 | 3343.68 | 3330.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 3285.40 | 3343.10 | 3330.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:45:00 | 3283.60 | 3343.10 | 3330.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 2968.55 | 3317.67 | 3318.35 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 3594.60 | 3290.96 | 3290.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 3613.00 | 3322.27 | 3306.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 4000.00 | 4010.40 | 3808.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 10:00:00 | 3995.60 | 4010.26 | 3809.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 3999.00 | 4106.26 | 3993.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 4000.00 | 4106.26 | 3993.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 4000.00 | 4105.20 | 3993.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:15:00 | 4041.00 | 4102.13 | 3993.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:45:00 | 4007.70 | 4095.34 | 4014.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 3955.20 | 4090.12 | 4013.60 | SL hit (close<static) qty=1.00 sl=3975.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 3867.00 | 3970.51 | 3970.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 13:15:00 | 3858.00 | 3968.31 | 3969.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 3951.60 | 3935.94 | 3951.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 3951.60 | 3935.94 | 3951.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3951.60 | 3935.94 | 3951.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 3959.00 | 3935.94 | 3951.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 3981.00 | 3936.39 | 3952.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 3981.40 | 3936.39 | 3952.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 4001.40 | 3937.03 | 3952.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 4001.40 | 3937.03 | 3952.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 3960.60 | 3937.27 | 3952.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:30:00 | 3952.90 | 3937.70 | 3952.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 15:00:00 | 3952.30 | 3937.70 | 3952.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 4015.20 | 3940.57 | 3953.52 | SL hit (close>static) qty=1.00 sl=4008.70 alert=retest2 |

### Cycle 10 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 4129.00 | 3964.19 | 3964.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 4146.40 | 3966.00 | 3964.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 4012.80 | 4016.96 | 3993.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 4012.80 | 4016.96 | 3993.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 4012.80 | 4016.96 | 3993.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 4012.80 | 4016.96 | 3993.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 3992.70 | 4016.72 | 3993.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 3992.70 | 4016.72 | 3993.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 3995.30 | 4016.51 | 3993.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 3974.60 | 4016.08 | 3993.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 3966.90 | 4015.60 | 3993.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 3961.10 | 4015.60 | 3993.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 4049.90 | 4094.07 | 4044.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 4050.20 | 4094.07 | 4044.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 4028.70 | 4093.42 | 4044.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 4028.70 | 4093.42 | 4044.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 4025.60 | 4092.75 | 4044.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 4025.60 | 4092.75 | 4044.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 3869.90 | 4010.42 | 4010.91 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 4294.00 | 4011.71 | 4010.91 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 3860.90 | 4028.20 | 4029.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 3800.80 | 3983.08 | 4003.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 3903.60 | 3901.73 | 3949.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 3903.60 | 3901.73 | 3949.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3807.70 | 3859.49 | 3907.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 3755.80 | 3856.29 | 3904.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 3747.10 | 3845.99 | 3897.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 3774.00 | 3843.63 | 3895.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 3783.90 | 3841.65 | 3893.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 3568.01 | 3822.31 | 3876.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 3585.30 | 3822.31 | 3876.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 3594.70 | 3822.31 | 3876.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 3559.74 | 3819.93 | 3875.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 3835.90 | 3724.32 | 3804.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 3835.90 | 3724.32 | 3804.29 | SL hit (close>ema200) qty=0.50 sl=3724.32 alert=retest2 |

### Cycle 14 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 4248.30 | 3838.74 | 3838.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 4289.00 | 3847.23 | 3842.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 3982.30 | 4028.11 | 3950.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 3927.90 | 4025.45 | 3952.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 3927.90 | 4025.45 | 3952.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 3927.90 | 4025.45 | 3952.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 3877.00 | 4023.98 | 3951.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 3877.00 | 4023.98 | 3951.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 3989.00 | 3990.81 | 3941.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 12:45:00 | 4000.00 | 3990.43 | 3941.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 09:45:00 | 3997.40 | 3992.30 | 3943.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 4079.10 | 3990.83 | 3944.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 12:00:00 | 4017.50 | 3999.81 | 3951.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 3954.90 | 3999.02 | 3951.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 3954.90 | 3999.02 | 3951.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 3937.00 | 3998.40 | 3951.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 3990.80 | 3998.40 | 3951.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 3919.40 | 3997.61 | 3951.36 | SL hit (close<static) qty=1.00 sl=3922.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-02-05 09:30:00 | 3170.60 | 2024-02-08 13:15:00 | 3033.05 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest2 | 2024-03-22 12:30:00 | 2946.00 | 2024-03-26 09:15:00 | 2798.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-22 12:30:00 | 2946.00 | 2024-03-26 10:15:00 | 2939.65 | STOP_HIT | 0.50 | 0.22% |
| SELL | retest2 | 2024-04-01 12:15:00 | 2949.00 | 2024-04-02 09:15:00 | 3040.65 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-04-01 12:45:00 | 2948.05 | 2024-04-02 09:15:00 | 3040.65 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest1 | 2024-06-04 13:30:00 | 4220.95 | 2024-06-11 11:15:00 | 4387.32 | PARTIAL | 0.50 | 3.94% |
| BUY | retest1 | 2024-06-05 12:45:00 | 4178.40 | 2024-06-11 11:15:00 | 4388.95 | PARTIAL | 0.50 | 5.04% |
| BUY | retest1 | 2024-06-05 14:45:00 | 4179.95 | 2024-06-11 12:15:00 | 4432.00 | PARTIAL | 0.50 | 6.03% |
| BUY | retest1 | 2024-06-06 09:15:00 | 4214.80 | 2024-06-11 12:15:00 | 4425.54 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-04 13:30:00 | 4220.95 | 2024-06-14 09:15:00 | 4643.05 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-06-05 12:45:00 | 4178.40 | 2024-06-14 09:15:00 | 4596.24 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-06-05 14:45:00 | 4179.95 | 2024-06-14 09:15:00 | 4597.95 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-06-06 09:15:00 | 4214.80 | 2024-06-14 09:15:00 | 4636.28 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-07-31 13:30:00 | 4253.65 | 2024-08-01 14:15:00 | 4167.05 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-08-26 10:15:00 | 4030.10 | 2024-09-04 09:15:00 | 3828.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-26 10:15:00 | 4030.10 | 2024-09-13 13:15:00 | 3954.55 | STOP_HIT | 0.50 | 1.87% |
| SELL | retest2 | 2024-10-15 09:30:00 | 4008.40 | 2024-10-22 10:15:00 | 3825.79 | PARTIAL | 0.50 | 4.56% |
| SELL | retest2 | 2024-10-15 12:15:00 | 4027.15 | 2024-10-22 10:15:00 | 3831.64 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2024-10-16 13:45:00 | 4033.30 | 2024-10-23 09:15:00 | 3807.98 | PARTIAL | 0.50 | 5.59% |
| SELL | retest2 | 2024-10-21 11:45:00 | 3957.35 | 2024-10-23 09:15:00 | 3759.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 3955.95 | 2024-10-23 09:15:00 | 3758.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:45:00 | 3953.00 | 2024-10-23 09:15:00 | 3755.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 3937.50 | 2024-10-24 10:15:00 | 3740.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 09:30:00 | 4008.40 | 2024-10-24 12:15:00 | 3607.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-15 12:15:00 | 4027.15 | 2024-10-24 12:15:00 | 3624.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-16 13:45:00 | 4033.30 | 2024-10-24 12:15:00 | 3629.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-22 10:15:00 | 3843.45 | 2024-10-24 12:15:00 | 3651.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 09:15:00 | 3829.25 | 2024-10-24 12:15:00 | 3637.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:45:00 | 3957.35 | 2024-10-25 10:15:00 | 3561.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 3955.95 | 2024-10-25 10:15:00 | 3560.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 14:45:00 | 3953.00 | 2024-10-25 10:15:00 | 3557.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 3937.50 | 2024-10-25 10:15:00 | 3543.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-22 10:15:00 | 3843.45 | 2024-10-28 14:15:00 | 3459.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-23 09:15:00 | 3829.25 | 2024-10-28 14:15:00 | 3446.33 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-29 14:15:00 | 4041.00 | 2025-08-08 12:15:00 | 3955.20 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-08-07 14:45:00 | 4007.70 | 2025-08-08 12:15:00 | 3955.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-08-21 09:15:00 | 4028.00 | 2025-08-22 09:15:00 | 3966.70 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-08-21 14:00:00 | 4008.00 | 2025-08-22 09:15:00 | 3966.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-10 14:30:00 | 3952.90 | 2025-09-11 12:15:00 | 4015.20 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-09-10 15:00:00 | 3952.30 | 2025-09-11 12:15:00 | 4015.20 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-09-12 09:45:00 | 3952.10 | 2025-09-16 09:15:00 | 4071.50 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-09-12 11:15:00 | 3954.60 | 2025-09-16 09:15:00 | 4071.50 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-01-09 15:00:00 | 3755.80 | 2026-01-20 15:15:00 | 3568.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 09:15:00 | 3747.10 | 2026-01-20 15:15:00 | 3585.30 | PARTIAL | 0.50 | 4.32% |
| SELL | retest2 | 2026-01-13 11:30:00 | 3774.00 | 2026-01-20 15:15:00 | 3594.70 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-13 15:15:00 | 3783.90 | 2026-01-21 09:15:00 | 3559.74 | PARTIAL | 0.50 | 5.92% |
| SELL | retest2 | 2026-01-09 15:00:00 | 3755.80 | 2026-02-03 09:15:00 | 3835.90 | STOP_HIT | 0.50 | -2.13% |
| SELL | retest2 | 2026-01-13 09:15:00 | 3747.10 | 2026-02-03 09:15:00 | 3835.90 | STOP_HIT | 0.50 | -2.37% |
| SELL | retest2 | 2026-01-13 11:30:00 | 3774.00 | 2026-02-03 09:15:00 | 3835.90 | STOP_HIT | 0.50 | -1.64% |
| SELL | retest2 | 2026-01-13 15:15:00 | 3783.90 | 2026-02-03 09:15:00 | 3835.90 | STOP_HIT | 0.50 | -1.37% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3730.70 | 2026-02-16 10:15:00 | 3829.20 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-02-13 15:00:00 | 3767.60 | 2026-02-16 10:15:00 | 3829.20 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-02-16 15:15:00 | 3767.00 | 2026-02-18 14:15:00 | 3885.30 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2026-02-18 10:00:00 | 3762.30 | 2026-02-18 14:15:00 | 3885.30 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2026-03-18 12:45:00 | 4000.00 | 2026-03-24 09:15:00 | 3919.40 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-03-19 09:45:00 | 3997.40 | 2026-03-24 10:15:00 | 3904.10 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-03-20 09:15:00 | 4079.10 | 2026-03-24 10:15:00 | 3904.10 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2026-03-23 12:00:00 | 4017.50 | 2026-03-24 10:15:00 | 3904.10 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2026-03-24 09:15:00 | 3990.80 | 2026-03-24 10:15:00 | 3904.10 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-03-24 14:00:00 | 3978.30 | 2026-03-30 09:15:00 | 3865.00 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2026-04-01 10:00:00 | 3978.20 | 2026-04-01 10:15:00 | 3882.20 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-04-08 12:00:00 | 3970.50 | 2026-04-08 13:15:00 | 3902.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-04-16 14:30:00 | 3972.50 | 2026-04-27 10:15:00 | 4369.75 | TARGET_HIT | 1.00 | 10.00% |
