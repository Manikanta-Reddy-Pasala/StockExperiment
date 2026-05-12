# Avenue Supermarts Ltd. (DMART)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 4396.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 4 |
| ALERT3 | 81 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 55 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 60 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 50
- **Target hits / Stop hits / Partials:** 5 / 51 / 4
- **Avg / median % per leg:** 0.12% / -1.11%
- **Sum % (uncompounded):** 7.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 5 | 15.2% | 5 | 28 | 0 | 0.43% | 14.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.74% | -2.7% |
| BUY @ 3rd Alert (retest2) | 32 | 5 | 15.6% | 5 | 27 | 0 | 0.53% | 17.0% |
| SELL (all) | 27 | 5 | 18.5% | 0 | 23 | 4 | -0.26% | -7.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 5 | 18.5% | 0 | 23 | 4 | -0.26% | -7.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.74% | -2.7% |
| retest2 (combined) | 59 | 10 | 16.9% | 5 | 50 | 4 | 0.17% | 10.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 3480.00 | 3539.41 | 3539.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 09:15:00 | 3460.00 | 3538.62 | 3539.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 14:15:00 | 3504.75 | 3493.87 | 3513.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-26 15:00:00 | 3504.75 | 3493.87 | 3513.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 3500.00 | 3493.93 | 3513.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 10:45:00 | 3493.65 | 3493.99 | 3513.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 11:15:00 | 3496.50 | 3493.99 | 3513.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 14:00:00 | 3499.40 | 3494.00 | 3512.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 09:45:00 | 3490.05 | 3495.09 | 3512.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 3510.00 | 3494.44 | 3511.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-02 13:15:00 | 3551.95 | 3497.06 | 3511.88 | SL hit (close>static) qty=1.00 sl=3545.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 15:15:00 | 3638.00 | 3523.94 | 3523.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 3669.80 | 3525.40 | 3524.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 15:15:00 | 3769.50 | 3769.97 | 3687.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:15:00 | 3809.50 | 3769.97 | 3687.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 3705.00 | 3777.95 | 3705.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-17 15:15:00 | 3705.00 | 3777.95 | 3705.35 | SL hit (close<ema400) qty=1.00 sl=3705.35 alert=retest1 |

### Cycle 3 — SELL (started 2023-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 09:15:00 | 3514.00 | 3682.22 | 3682.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 11:15:00 | 3498.00 | 3678.77 | 3680.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 3623.45 | 3622.40 | 3647.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 3623.45 | 3622.40 | 3647.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 3623.45 | 3622.40 | 3647.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:30:00 | 3634.90 | 3622.40 | 3647.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 12:15:00 | 3647.95 | 3622.69 | 3647.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 13:00:00 | 3647.95 | 3622.69 | 3647.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 13:15:00 | 3653.15 | 3622.99 | 3647.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 13:45:00 | 3650.00 | 3622.99 | 3647.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 14:15:00 | 3660.05 | 3623.36 | 3647.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 15:00:00 | 3660.05 | 3623.36 | 3647.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 14:15:00 | 3798.95 | 3666.27 | 3666.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 09:15:00 | 3853.15 | 3669.37 | 3667.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 09:15:00 | 3717.15 | 3722.68 | 3699.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-21 10:00:00 | 3717.15 | 3722.68 | 3699.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 3707.15 | 3722.52 | 3699.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 10:45:00 | 3699.90 | 3722.52 | 3699.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 3703.50 | 3722.33 | 3699.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:45:00 | 3700.00 | 3722.33 | 3699.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 3699.95 | 3722.11 | 3699.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 12:45:00 | 3700.00 | 3722.11 | 3699.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 3688.60 | 3721.78 | 3699.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:00:00 | 3688.60 | 3721.78 | 3699.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 3688.00 | 3721.44 | 3699.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 15:15:00 | 3690.00 | 3721.44 | 3699.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 3671.65 | 3720.10 | 3698.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 15:00:00 | 3693.85 | 3718.28 | 3698.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-25 13:15:00 | 3677.85 | 3716.56 | 3697.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-26 10:15:00 | 3650.00 | 3714.07 | 3697.14 | SL hit (close<static) qty=1.00 sl=3660.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 15:15:00 | 3649.25 | 3720.41 | 3720.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 09:15:00 | 3644.50 | 3719.66 | 3720.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 14:15:00 | 3717.55 | 3714.13 | 3717.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 14:15:00 | 3717.55 | 3714.13 | 3717.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 3717.55 | 3714.13 | 3717.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 15:00:00 | 3717.55 | 3714.13 | 3717.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 15:15:00 | 3726.80 | 3714.26 | 3717.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:15:00 | 3753.00 | 3714.26 | 3717.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 3761.00 | 3714.72 | 3717.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:00:00 | 3761.00 | 3714.72 | 3717.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 14:15:00 | 3799.45 | 3720.21 | 3720.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-12 18:15:00 | 3802.00 | 3721.80 | 3720.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-21 12:15:00 | 3926.60 | 3946.25 | 3870.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-21 13:00:00 | 3926.60 | 3946.25 | 3870.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 3910.00 | 3979.64 | 3910.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 10:45:00 | 3910.00 | 3979.64 | 3910.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 3906.75 | 3978.91 | 3910.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 12:00:00 | 3906.75 | 3978.91 | 3910.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 12:15:00 | 3898.00 | 3978.11 | 3909.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 13:00:00 | 3898.00 | 3978.11 | 3909.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 11:15:00 | 3846.45 | 3973.29 | 3909.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 12:00:00 | 3846.45 | 3973.29 | 3909.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 12:15:00 | 3850.00 | 3972.07 | 3909.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-05 15:00:00 | 3859.50 | 3969.74 | 3908.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 10:00:00 | 3860.00 | 3967.77 | 3908.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 14:15:00 | 3802.25 | 3960.76 | 3906.24 | SL hit (close<static) qty=1.00 sl=3810.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 09:15:00 | 3658.00 | 3868.96 | 3870.01 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 14:15:00 | 3930.25 | 3819.01 | 3818.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 09:15:00 | 3986.90 | 3821.71 | 3820.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 4487.05 | 4497.44 | 4284.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 10:00:00 | 4487.05 | 4497.44 | 4284.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 4479.95 | 4630.63 | 4479.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 4479.95 | 4630.63 | 4479.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 4477.05 | 4629.11 | 4479.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:30:00 | 4450.00 | 4629.11 | 4479.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 4472.00 | 4627.54 | 4479.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 4477.75 | 4627.54 | 4479.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 4482.65 | 4624.43 | 4479.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:30:00 | 4463.00 | 4624.43 | 4479.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 4505.00 | 4623.24 | 4479.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 12:15:00 | 4507.70 | 4623.24 | 4479.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 12:45:00 | 4507.35 | 4622.11 | 4480.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:45:00 | 4537.45 | 4621.37 | 4480.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 15:15:00 | 4470.00 | 4618.54 | 4480.35 | SL hit (close<static) qty=1.00 sl=4474.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 4607.55 | 4994.55 | 4996.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 4565.25 | 4986.51 | 4992.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 4134.10 | 3662.66 | 3882.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 4134.10 | 3662.66 | 3882.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 4134.10 | 3662.66 | 3882.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 4134.10 | 3662.66 | 3882.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 4097.40 | 3666.98 | 3883.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 11:15:00 | 4042.85 | 3666.98 | 3883.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 14:15:00 | 3840.71 | 3714.68 | 3890.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 3824.25 | 3714.68 | 3890.00 | SL hit (close>static) qty=0.50 sl=3714.68 alert=retest2 |

### Cycle 10 — BUY (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 15:15:00 | 3988.00 | 3722.70 | 3721.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 09:15:00 | 4018.10 | 3741.04 | 3731.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 10:15:00 | 4137.30 | 4147.52 | 3999.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 11:00:00 | 4137.30 | 4147.52 | 3999.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 3959.90 | 4141.00 | 4000.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 12:00:00 | 4042.10 | 4138.30 | 4000.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:00:00 | 4040.50 | 4132.64 | 4001.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 11:45:00 | 4048.00 | 4114.39 | 4001.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 09:15:00 | 4064.30 | 4094.07 | 4000.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 4051.10 | 4100.39 | 4036.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 4053.90 | 4100.39 | 4036.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 4038.00 | 4099.77 | 4036.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 4038.00 | 4099.77 | 4036.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 4040.00 | 4099.18 | 4037.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 4040.00 | 4099.18 | 4037.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 4022.80 | 4098.42 | 4036.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 4024.10 | 4098.42 | 4036.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 4046.90 | 4097.90 | 4036.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 14:45:00 | 4047.40 | 4097.39 | 4037.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 4050.00 | 4097.39 | 4037.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 4015.50 | 4094.02 | 4036.82 | SL hit (close<static) qty=1.00 sl=4017.90 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 3950.00 | 4115.07 | 4115.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 3941.00 | 4108.57 | 4111.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 11:15:00 | 4206.10 | 4102.00 | 4108.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 11:15:00 | 4206.10 | 4102.00 | 4108.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 4206.10 | 4102.00 | 4108.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:00:00 | 4206.10 | 4102.00 | 4108.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 4247.70 | 4103.45 | 4109.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 4247.70 | 4103.45 | 4109.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 4293.20 | 4116.12 | 4115.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 4316.10 | 4170.86 | 4147.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 4616.60 | 4620.19 | 4480.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:45:00 | 4593.00 | 4620.19 | 4480.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 4548.60 | 4615.09 | 4495.83 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 4305.00 | 4429.38 | 4429.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 4294.80 | 4426.75 | 4428.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 3847.40 | 3838.18 | 3965.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 3847.40 | 3838.18 | 3965.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3854.50 | 3757.11 | 3851.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:30:00 | 3837.50 | 3757.11 | 3851.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 3876.90 | 3758.30 | 3851.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 3868.80 | 3758.30 | 3851.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 3867.50 | 3759.39 | 3851.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 3881.60 | 3759.39 | 3851.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 3860.10 | 3768.79 | 3852.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 3871.50 | 3768.79 | 3852.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 3875.80 | 3770.75 | 3853.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 3875.80 | 3770.75 | 3853.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 3889.10 | 3771.93 | 3853.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 3889.10 | 3771.93 | 3853.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 3880.30 | 3826.93 | 3869.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:00:00 | 3873.70 | 3827.39 | 3869.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 3916.00 | 3829.88 | 3869.58 | SL hit (close>static) qty=1.00 sl=3897.10 alert=retest2 |

### Cycle 14 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 4263.90 | 3871.74 | 3870.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 10:15:00 | 4302.20 | 3883.53 | 3876.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 4318.00 | 4347.05 | 4186.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 4318.00 | 4347.05 | 4186.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-29 10:45:00 | 3493.65 | 2023-06-02 13:15:00 | 3551.95 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2023-05-29 11:15:00 | 3496.50 | 2023-06-02 13:15:00 | 3551.95 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2023-05-29 14:00:00 | 3499.40 | 2023-06-02 13:15:00 | 3551.95 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2023-05-31 09:45:00 | 3490.05 | 2023-06-02 13:15:00 | 3551.95 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest1 | 2023-07-11 09:15:00 | 3809.50 | 2023-07-17 15:15:00 | 3705.00 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2023-07-19 11:00:00 | 3714.40 | 2023-07-20 11:15:00 | 3670.85 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2023-07-26 13:45:00 | 3714.40 | 2023-07-27 13:15:00 | 3673.20 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2023-07-26 15:00:00 | 3713.40 | 2023-07-27 13:15:00 | 3673.20 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-07-28 15:00:00 | 3712.50 | 2023-08-04 10:15:00 | 3675.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-09-22 15:00:00 | 3693.85 | 2023-09-26 10:15:00 | 3650.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2023-09-25 13:15:00 | 3677.85 | 2023-09-26 10:15:00 | 3650.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2023-09-28 09:15:00 | 3677.95 | 2023-09-29 09:15:00 | 3659.95 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-09-28 13:00:00 | 3677.00 | 2023-09-29 09:15:00 | 3659.95 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-10-27 09:15:00 | 3750.75 | 2023-10-27 09:15:00 | 3714.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-01-05 15:00:00 | 3859.50 | 2024-01-08 14:15:00 | 3802.25 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-01-08 10:00:00 | 3860.00 | 2024-01-08 14:15:00 | 3802.25 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-01-15 09:15:00 | 3881.00 | 2024-01-17 09:15:00 | 3807.95 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-05-29 12:15:00 | 4507.70 | 2024-05-29 15:15:00 | 4470.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-05-29 12:45:00 | 4507.35 | 2024-05-29 15:15:00 | 4470.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-05-29 13:45:00 | 4537.45 | 2024-05-29 15:15:00 | 4470.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-06-04 10:00:00 | 4539.00 | 2024-06-04 10:15:00 | 4455.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-06-04 13:15:00 | 4503.75 | 2024-06-18 12:15:00 | 4954.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-09 09:15:00 | 4628.15 | 2024-10-10 14:15:00 | 4607.55 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-01-03 11:15:00 | 4042.85 | 2025-01-07 14:15:00 | 3840.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 11:15:00 | 4042.85 | 2025-01-07 14:15:00 | 3824.25 | STOP_HIT | 0.50 | 5.41% |
| BUY | retest2 | 2025-05-05 12:00:00 | 4042.10 | 2025-05-30 12:15:00 | 4015.50 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-06 10:00:00 | 4040.50 | 2025-05-30 12:15:00 | 4015.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-05-08 11:45:00 | 4048.00 | 2025-06-16 09:15:00 | 4002.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-13 09:15:00 | 4064.30 | 2025-06-16 09:15:00 | 4002.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-05-29 14:45:00 | 4047.40 | 2025-06-16 09:15:00 | 4002.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-05-29 15:15:00 | 4050.00 | 2025-06-16 09:15:00 | 4002.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-06-03 12:00:00 | 4052.90 | 2025-06-16 09:15:00 | 4002.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-03 13:15:00 | 4049.00 | 2025-06-16 09:15:00 | 4002.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-06-04 14:30:00 | 4052.10 | 2025-07-01 09:15:00 | 4446.31 | TARGET_HIT | 1.00 | 9.73% |
| BUY | retest2 | 2025-06-10 12:30:00 | 4057.10 | 2025-07-01 09:15:00 | 4444.55 | TARGET_HIT | 1.00 | 9.55% |
| BUY | retest2 | 2025-06-13 11:15:00 | 4064.10 | 2025-07-01 09:15:00 | 4452.80 | TARGET_HIT | 1.00 | 9.56% |
| BUY | retest2 | 2025-06-13 12:00:00 | 4050.40 | 2025-07-01 09:15:00 | 4470.73 | TARGET_HIT | 1.00 | 10.38% |
| BUY | retest2 | 2025-06-18 09:15:00 | 4152.40 | 2025-07-14 10:15:00 | 4012.70 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-07-16 14:15:00 | 4115.00 | 2025-07-18 09:15:00 | 4045.70 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-02-16 12:00:00 | 3873.70 | 2026-02-17 09:15:00 | 3916.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-02-19 11:15:00 | 3865.50 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-20 11:45:00 | 3872.30 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-02-20 14:30:00 | 3871.10 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-23 12:30:00 | 3843.40 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-02-23 14:00:00 | 3839.70 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-23 15:15:00 | 3840.00 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-24 14:30:00 | 3838.00 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-02-27 09:45:00 | 3814.70 | 2026-03-09 11:15:00 | 3897.00 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-03-02 09:15:00 | 3797.10 | 2026-03-09 11:15:00 | 3897.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-03-02 11:45:00 | 3807.60 | 2026-03-09 11:15:00 | 3897.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-03-06 12:45:00 | 3827.50 | 2026-03-09 11:15:00 | 3897.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-03-09 09:15:00 | 3860.90 | 2026-03-09 12:15:00 | 3925.20 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-03-09 09:45:00 | 3858.00 | 2026-03-09 12:15:00 | 3925.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-03-09 10:30:00 | 3858.00 | 2026-03-09 12:15:00 | 3925.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-03-13 13:45:00 | 3843.70 | 2026-03-23 10:15:00 | 3667.19 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2026-03-18 12:30:00 | 3860.20 | 2026-03-23 12:15:00 | 3651.51 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2026-03-19 11:15:00 | 3843.00 | 2026-03-23 13:15:00 | 3650.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 13:45:00 | 3843.70 | 2026-03-25 09:15:00 | 3869.40 | STOP_HIT | 0.50 | -0.67% |
| SELL | retest2 | 2026-03-18 12:30:00 | 3860.20 | 2026-03-25 09:15:00 | 3869.40 | STOP_HIT | 0.50 | -0.24% |
| SELL | retest2 | 2026-03-19 11:15:00 | 3843.00 | 2026-03-25 09:15:00 | 3869.40 | STOP_HIT | 0.50 | -0.69% |
