# Gujarat Fluorochemicals Ltd. (FLUOROCHEM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3777.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT2_SKIP | 4 |
| ALERT3 | 88 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 90 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 88 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 101 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 79
- **Target hits / Stop hits / Partials:** 2 / 88 / 11
- **Avg / median % per leg:** -1.03% / -1.78%
- **Sum % (uncompounded):** -104.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 0 | 0.0% | 0 | 30 | 0 | -2.56% | -76.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 0 | 0.0% | 0 | 30 | 0 | -2.56% | -76.7% |
| SELL (all) | 71 | 22 | 31.0% | 2 | 58 | 11 | -0.39% | -27.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 71 | 22 | 31.0% | 2 | 58 | 11 | -0.39% | -27.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 101 | 22 | 21.8% | 2 | 88 | 11 | -1.03% | -104.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 11:15:00 | 3206.90 | 3421.34 | 3422.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 3195.45 | 3416.96 | 3420.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-11 14:15:00 | 3175.05 | 3162.99 | 3259.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-11 14:45:00 | 3205.15 | 3162.99 | 3259.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 3283.00 | 3163.90 | 3258.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:00:00 | 3283.00 | 3163.90 | 3258.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 3255.70 | 3164.82 | 3258.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 12:15:00 | 3228.75 | 3164.82 | 3258.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 10:45:00 | 3246.75 | 3168.12 | 3257.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 15:15:00 | 3250.00 | 3178.68 | 3258.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 15:15:00 | 3245.40 | 3180.73 | 3254.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 3245.40 | 3181.37 | 3254.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 3410.90 | 3183.66 | 3254.82 | SL hit (close>static) qty=1.00 sl=3368.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 3384.80 | 3268.86 | 3268.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 12:15:00 | 3416.35 | 3271.43 | 3269.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 11:15:00 | 3250.95 | 3298.53 | 3284.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 11:15:00 | 3250.95 | 3298.53 | 3284.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 3250.95 | 3298.53 | 3284.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:00:00 | 3250.95 | 3298.53 | 3284.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 3264.50 | 3298.19 | 3284.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:15:00 | 3226.00 | 3298.19 | 3284.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 3296.00 | 3297.46 | 3284.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 3410.40 | 3297.46 | 3284.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 3243.45 | 3308.30 | 3291.41 | SL hit (close<static) qty=1.00 sl=3271.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 13:15:00 | 3162.30 | 3277.94 | 3278.16 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 11:15:00 | 3761.95 | 3281.27 | 3278.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 12:15:00 | 3799.00 | 3286.42 | 3281.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 3969.55 | 3994.56 | 3753.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-07 11:00:00 | 3969.55 | 3994.56 | 3753.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 4073.60 | 4309.04 | 4056.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:00:00 | 4073.60 | 4309.04 | 4056.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 4079.00 | 4304.69 | 4056.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 4079.00 | 4304.69 | 4056.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 4060.00 | 4302.26 | 4056.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:45:00 | 4035.00 | 4302.26 | 4056.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 4086.40 | 4300.11 | 4056.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:30:00 | 4074.90 | 4300.11 | 4056.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 4152.15 | 4277.71 | 4102.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:15:00 | 4177.20 | 4277.71 | 4102.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 11:00:00 | 4180.00 | 4276.73 | 4103.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 11:30:00 | 4180.60 | 4276.08 | 4103.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 4010.00 | 4263.33 | 4107.36 | SL hit (close<static) qty=1.00 sl=4058.05 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 3659.00 | 4129.78 | 4130.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 3631.05 | 4124.81 | 4128.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 3782.60 | 3742.50 | 3864.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 14:45:00 | 3787.45 | 3742.50 | 3864.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 3805.00 | 3741.00 | 3859.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:30:00 | 3798.00 | 3741.55 | 3858.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:00:00 | 3798.75 | 3742.12 | 3858.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 3608.10 | 3742.16 | 3854.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 3608.81 | 3742.16 | 3854.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-24 14:15:00 | 3789.20 | 3713.91 | 3815.73 | SL hit (close>ema200) qty=0.50 sl=3713.91 alert=retest2 |

### Cycle 6 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 4017.20 | 3816.78 | 3816.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 4021.85 | 3818.83 | 3817.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 3715.00 | 3844.08 | 3830.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 3715.00 | 3844.08 | 3830.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 3715.00 | 3844.08 | 3830.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 3713.90 | 3844.08 | 3830.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 3740.75 | 3843.06 | 3830.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 14:45:00 | 3788.65 | 3839.68 | 3829.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 3504.30 | 3836.35 | 3827.49 | SL hit (close<static) qty=1.00 sl=3680.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-08 11:15:00 | 3702.60 | 3818.39 | 3818.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 09:15:00 | 3655.00 | 3812.14 | 3815.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 11:15:00 | 3846.85 | 3803.41 | 3810.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 11:15:00 | 3846.85 | 3803.41 | 3810.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 3846.85 | 3803.41 | 3810.86 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 13:15:00 | 3930.00 | 3818.49 | 3818.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 13:15:00 | 3956.60 | 3825.65 | 3821.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 10:15:00 | 3850.20 | 3874.97 | 3850.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 10:15:00 | 3850.20 | 3874.97 | 3850.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 3850.20 | 3874.97 | 3850.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:45:00 | 3840.30 | 3874.97 | 3850.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 3882.00 | 3875.04 | 3851.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 12:45:00 | 3887.00 | 3875.12 | 3851.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 14:00:00 | 3890.50 | 3875.28 | 3851.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 09:15:00 | 3830.80 | 3874.65 | 3851.44 | SL hit (close<static) qty=1.00 sl=3833.30 alert=retest2 |

### Cycle 9 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 3640.10 | 3860.23 | 3860.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 3632.70 | 3857.96 | 3859.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 3635.70 | 3633.56 | 3716.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 3635.70 | 3633.56 | 3716.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3619.70 | 3526.51 | 3614.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 3619.70 | 3526.51 | 3614.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 3574.40 | 3526.98 | 3614.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:30:00 | 3547.00 | 3528.85 | 3613.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 3563.40 | 3533.39 | 3612.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 3569.20 | 3535.06 | 3611.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 15:15:00 | 3560.00 | 3535.06 | 3611.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 3596.50 | 3531.83 | 3599.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:45:00 | 3598.00 | 3531.83 | 3599.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 3592.80 | 3533.95 | 3598.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 3596.70 | 3533.95 | 3598.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 3616.60 | 3534.77 | 3598.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:30:00 | 3611.00 | 3534.77 | 3598.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 3591.10 | 3535.33 | 3598.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:30:00 | 3618.40 | 3535.33 | 3598.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 3646.90 | 3532.10 | 3590.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 3646.90 | 3532.10 | 3590.23 | SL hit (close>static) qty=1.00 sl=3624.10 alert=retest2 |

### Cycle 10 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 3805.60 | 3543.14 | 3542.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 10:15:00 | 3837.00 | 3546.07 | 3544.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 3617.40 | 3646.61 | 3606.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:00:00 | 3617.40 | 3646.61 | 3606.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 3645.00 | 3646.29 | 3606.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:30:00 | 3611.40 | 3646.29 | 3606.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 3619.40 | 3660.09 | 3623.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 3619.40 | 3660.09 | 3623.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 3624.00 | 3659.73 | 3623.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 3664.10 | 3659.73 | 3623.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 10:15:00 | 3590.00 | 3677.44 | 3639.46 | SL hit (close<static) qty=1.00 sl=3603.80 alert=retest2 |

### Cycle 11 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 3506.90 | 3624.67 | 3624.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 09:15:00 | 3491.90 | 3620.10 | 3622.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 3485.60 | 3478.19 | 3536.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 3485.60 | 3478.19 | 3536.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 3476.10 | 3478.17 | 3535.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 3454.30 | 3478.17 | 3535.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 3460.40 | 3477.36 | 3533.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 3468.00 | 3477.36 | 3533.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:45:00 | 3467.60 | 3477.62 | 3530.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 3521.90 | 3476.65 | 3527.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:45:00 | 3531.50 | 3476.65 | 3527.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 3533.00 | 3477.22 | 3527.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 3478.50 | 3477.22 | 3527.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 3479.90 | 3477.80 | 3526.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 3500.00 | 3478.72 | 3524.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:45:00 | 3500.20 | 3479.26 | 3524.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 3529.20 | 3479.76 | 3524.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 3529.20 | 3479.76 | 3524.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 3559.70 | 3480.56 | 3524.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 3559.70 | 3480.56 | 3524.52 | SL hit (close>static) qty=1.00 sl=3540.50 alert=retest2 |

### Cycle 12 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 3658.10 | 3554.53 | 3554.29 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 3493.70 | 3554.31 | 3554.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 3475.00 | 3552.15 | 3553.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 3291.30 | 3283.33 | 3387.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 12:45:00 | 3294.10 | 3283.33 | 3387.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 3391.40 | 3289.90 | 3386.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 3391.40 | 3289.90 | 3386.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 3379.90 | 3290.80 | 3386.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 3342.40 | 3290.80 | 3386.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:00:00 | 3358.90 | 3293.12 | 3386.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 14:15:00 | 3428.90 | 3295.58 | 3386.04 | SL hit (close>static) qty=1.00 sl=3395.00 alert=retest2 |

### Cycle 14 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 3585.40 | 3309.52 | 3308.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 3750.00 | 3319.47 | 3313.95 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-12 12:15:00 | 3228.75 | 2024-06-20 09:15:00 | 3410.90 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2024-06-13 10:45:00 | 3246.75 | 2024-06-20 09:15:00 | 3410.90 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2024-06-14 15:15:00 | 3250.00 | 2024-06-20 09:15:00 | 3410.90 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2024-06-19 15:15:00 | 3245.40 | 2024-06-20 09:15:00 | 3410.90 | STOP_HIT | 1.00 | -5.10% |
| SELL | retest2 | 2024-06-20 12:45:00 | 3387.25 | 2024-06-27 13:15:00 | 3217.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-20 13:30:00 | 3382.05 | 2024-06-27 13:15:00 | 3233.56 | PARTIAL | 0.50 | 4.39% |
| SELL | retest2 | 2024-06-21 09:15:00 | 3403.75 | 2024-06-27 13:15:00 | 3220.07 | PARTIAL | 0.50 | 5.40% |
| SELL | retest2 | 2024-06-21 10:45:00 | 3389.55 | 2024-06-27 14:15:00 | 3212.95 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2024-06-20 12:45:00 | 3387.25 | 2024-07-01 11:15:00 | 3244.05 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2024-06-20 13:30:00 | 3382.05 | 2024-07-01 11:15:00 | 3244.05 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2024-06-21 09:15:00 | 3403.75 | 2024-07-01 11:15:00 | 3244.05 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2024-06-21 10:45:00 | 3389.55 | 2024-07-01 11:15:00 | 3244.05 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2024-07-04 14:00:00 | 3251.95 | 2024-07-15 13:15:00 | 3302.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-07-04 14:30:00 | 3248.75 | 2024-07-15 13:15:00 | 3302.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-07-04 15:00:00 | 3245.40 | 2024-07-15 13:15:00 | 3302.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-07-05 13:30:00 | 3250.00 | 2024-07-15 13:15:00 | 3302.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-07-08 11:00:00 | 3244.80 | 2024-07-15 13:15:00 | 3302.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-07-08 11:45:00 | 3244.10 | 2024-07-15 13:15:00 | 3302.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-07-09 10:00:00 | 3248.35 | 2024-07-15 13:15:00 | 3302.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-07-10 10:00:00 | 3225.95 | 2024-07-15 13:15:00 | 3302.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-07-19 10:15:00 | 3208.90 | 2024-07-22 15:15:00 | 3270.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-07-19 12:45:00 | 3214.85 | 2024-07-22 15:15:00 | 3270.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-07-22 09:15:00 | 3202.95 | 2024-07-22 15:15:00 | 3270.00 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-07-23 11:30:00 | 3212.80 | 2024-07-26 13:15:00 | 3270.10 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-08-05 12:30:00 | 3206.60 | 2024-08-06 09:15:00 | 3310.00 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2024-08-05 13:15:00 | 3204.45 | 2024-08-06 09:15:00 | 3310.00 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2024-08-16 09:15:00 | 3410.40 | 2024-08-21 09:15:00 | 3243.45 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2024-08-27 10:00:00 | 3318.50 | 2024-08-27 13:15:00 | 3267.20 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-08-27 12:45:00 | 3306.25 | 2024-08-27 13:15:00 | 3267.20 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-28 09:45:00 | 3304.00 | 2024-08-28 12:15:00 | 3265.15 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-11-11 10:15:00 | 4177.20 | 2024-11-13 09:15:00 | 4010.00 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2024-11-11 11:00:00 | 4180.00 | 2024-11-13 09:15:00 | 4010.00 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2024-11-11 11:30:00 | 4180.60 | 2024-11-13 09:15:00 | 4010.00 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2024-12-05 09:15:00 | 4195.55 | 2024-12-31 10:15:00 | 4043.80 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2025-02-13 09:30:00 | 3798.00 | 2025-02-14 10:15:00 | 3608.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:00:00 | 3798.75 | 2025-02-14 10:15:00 | 3608.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 09:30:00 | 3798.00 | 2025-02-24 14:15:00 | 3789.20 | STOP_HIT | 0.50 | 0.23% |
| SELL | retest2 | 2025-02-13 11:00:00 | 3798.75 | 2025-02-24 14:15:00 | 3789.20 | STOP_HIT | 0.50 | 0.25% |
| SELL | retest2 | 2025-03-10 10:30:00 | 3799.30 | 2025-03-20 13:15:00 | 3873.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-03-10 13:30:00 | 3795.00 | 2025-03-20 13:15:00 | 3873.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-03-12 09:15:00 | 3814.80 | 2025-03-20 14:15:00 | 3999.45 | STOP_HIT | 1.00 | -4.84% |
| SELL | retest2 | 2025-03-13 09:15:00 | 3785.00 | 2025-03-20 14:15:00 | 3999.45 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2025-03-13 10:45:00 | 3808.65 | 2025-03-20 14:15:00 | 3999.45 | STOP_HIT | 1.00 | -5.01% |
| SELL | retest2 | 2025-03-13 13:15:00 | 3809.00 | 2025-03-20 14:15:00 | 3999.45 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2025-04-04 14:45:00 | 3788.65 | 2025-04-07 09:15:00 | 3504.30 | STOP_HIT | 1.00 | -7.51% |
| BUY | retest2 | 2025-04-30 12:45:00 | 3887.00 | 2025-05-02 09:15:00 | 3830.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-04-30 14:00:00 | 3890.50 | 2025-05-02 09:15:00 | 3830.80 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-05-05 09:45:00 | 3887.40 | 2025-05-06 09:15:00 | 3830.50 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-06 10:45:00 | 3900.00 | 2025-05-07 09:15:00 | 3807.00 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-05-07 11:15:00 | 3887.00 | 2025-05-09 14:15:00 | 3785.10 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-05-07 12:45:00 | 3882.90 | 2025-05-09 14:15:00 | 3785.10 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-07 14:30:00 | 3885.50 | 2025-05-09 14:15:00 | 3785.10 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-05-09 10:00:00 | 3895.00 | 2025-05-09 14:15:00 | 3785.10 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-05-12 09:15:00 | 3968.10 | 2025-05-29 09:15:00 | 3751.00 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest2 | 2025-05-28 11:45:00 | 3804.00 | 2025-05-29 09:15:00 | 3751.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-05-28 14:00:00 | 3798.50 | 2025-05-29 09:15:00 | 3751.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-05-28 14:45:00 | 3833.00 | 2025-05-29 09:15:00 | 3751.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-07-22 14:30:00 | 3547.00 | 2025-08-06 09:15:00 | 3646.90 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2025-07-24 10:15:00 | 3563.40 | 2025-08-06 09:15:00 | 3646.90 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-07-24 14:30:00 | 3569.20 | 2025-08-06 09:15:00 | 3646.90 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-07-24 15:15:00 | 3560.00 | 2025-08-06 09:15:00 | 3646.90 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-08-07 10:00:00 | 3583.40 | 2025-08-11 15:15:00 | 3404.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 11:00:00 | 3565.80 | 2025-08-13 15:15:00 | 3387.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 09:15:00 | 3563.20 | 2025-08-14 09:15:00 | 3385.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 10:00:00 | 3583.40 | 2025-09-09 09:15:00 | 3449.00 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2025-08-07 11:00:00 | 3565.80 | 2025-09-09 09:15:00 | 3449.00 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-08-08 09:15:00 | 3563.20 | 2025-09-09 09:15:00 | 3449.00 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-09-11 14:15:00 | 3580.70 | 2025-09-12 09:15:00 | 3738.00 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2025-10-16 09:15:00 | 3664.10 | 2025-10-27 10:15:00 | 3590.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-10-28 10:00:00 | 3625.00 | 2025-10-28 14:15:00 | 3597.50 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-28 10:30:00 | 3651.70 | 2025-10-28 14:15:00 | 3597.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-28 11:30:00 | 3625.70 | 2025-10-28 14:15:00 | 3597.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-10-30 15:00:00 | 3654.10 | 2025-11-07 09:15:00 | 3581.80 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-10-31 09:15:00 | 3762.00 | 2025-11-07 09:15:00 | 3581.80 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2025-11-06 11:45:00 | 3647.90 | 2025-11-07 09:15:00 | 3581.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-11-06 12:30:00 | 3645.10 | 2025-11-07 09:15:00 | 3581.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-11-12 14:00:00 | 3670.90 | 2025-11-13 11:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-12-10 09:15:00 | 3454.30 | 2025-12-18 15:15:00 | 3559.70 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-12-11 09:30:00 | 3460.40 | 2025-12-18 15:15:00 | 3559.70 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-12-11 10:15:00 | 3468.00 | 2025-12-18 15:15:00 | 3559.70 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-12-12 12:45:00 | 3467.60 | 2025-12-18 15:15:00 | 3559.70 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-12-16 09:15:00 | 3478.50 | 2025-12-18 15:15:00 | 3559.70 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-12-17 09:15:00 | 3479.90 | 2025-12-18 15:15:00 | 3559.70 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-12-18 09:30:00 | 3500.00 | 2025-12-18 15:15:00 | 3559.70 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-18 13:45:00 | 3500.20 | 2025-12-18 15:15:00 | 3559.70 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-12-19 09:30:00 | 3520.30 | 2025-12-19 14:15:00 | 3624.70 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-12-19 10:30:00 | 3522.60 | 2025-12-19 14:15:00 | 3624.70 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2026-02-06 09:15:00 | 3342.40 | 2026-02-06 14:15:00 | 3428.90 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-02-06 12:00:00 | 3358.90 | 2026-02-06 14:15:00 | 3428.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-02-12 10:15:00 | 3369.90 | 2026-02-12 14:15:00 | 3396.80 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-12 14:45:00 | 3355.70 | 2026-02-13 14:15:00 | 3396.50 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3277.90 | 2026-02-13 14:15:00 | 3396.50 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2026-02-16 09:15:00 | 3340.30 | 2026-02-17 13:15:00 | 3403.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-03-02 13:30:00 | 3339.00 | 2026-03-09 09:15:00 | 3172.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 14:45:00 | 3344.70 | 2026-03-09 09:15:00 | 3177.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 13:30:00 | 3339.00 | 2026-03-13 13:15:00 | 3005.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 14:45:00 | 3344.70 | 2026-03-13 13:15:00 | 3010.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-07 13:30:00 | 3243.00 | 2026-04-08 09:15:00 | 3326.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-04-20 15:15:00 | 3257.00 | 2026-04-21 10:15:00 | 3296.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-04-21 14:30:00 | 3261.30 | 2026-04-22 11:15:00 | 3295.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-04-21 15:15:00 | 3264.70 | 2026-04-22 11:15:00 | 3295.00 | STOP_HIT | 1.00 | -0.93% |
