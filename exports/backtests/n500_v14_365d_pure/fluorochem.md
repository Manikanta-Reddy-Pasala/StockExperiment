# Gujarat Fluorochemicals Ltd. (FLUOROCHEM)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 3777.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 38
- **Target hits / Stop hits / Partials:** 2 / 41 / 5
- **Avg / median % per leg:** -0.61% / -1.74%
- **Sum % (uncompounded):** -29.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 0 | 0.0% | 0 | 13 | 0 | -2.10% | -27.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -2.10% | -27.3% |
| SELL (all) | 35 | 10 | 28.6% | 2 | 28 | 5 | -0.06% | -2.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 10 | 28.6% | 2 | 28 | 5 | -0.06% | -2.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 10 | 20.8% | 2 | 41 | 5 | -0.61% | -29.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 3640.10 | 3860.23 | 3860.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 3632.70 | 3857.96 | 3859.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 3635.70 | 3633.56 | 3716.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 3635.70 | 3633.56 | 3716.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3619.70 | 3526.51 | 3614.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 3619.70 | 3526.51 | 3614.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 3574.40 | 3526.98 | 3614.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:30:00 | 3547.00 | 3528.85 | 3613.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 3563.40 | 3533.39 | 3612.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 3569.20 | 3535.06 | 3611.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 15:15:00 | 3560.00 | 3535.06 | 3611.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
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
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 3646.90 | 3532.10 | 3590.23 | SL hit (close>static) qty=1.00 sl=3624.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 3646.90 | 3532.10 | 3590.23 | SL hit (close>static) qty=1.00 sl=3624.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 3646.90 | 3532.10 | 3590.23 | SL hit (close>static) qty=1.00 sl=3624.10 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 3646.90 | 3532.10 | 3590.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 3662.40 | 3533.40 | 3590.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 10:00:00 | 3583.40 | 3539.92 | 3592.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:00:00 | 3565.80 | 3540.18 | 3592.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 3563.20 | 3541.39 | 3591.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 15:15:00 | 3404.23 | 3531.36 | 3582.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 15:15:00 | 3387.51 | 3516.10 | 3571.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 09:15:00 | 3385.04 | 3514.80 | 3570.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 3449.00 | 3438.03 | 3498.00 | SL hit (close>ema200) qty=0.50 sl=3438.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 3449.00 | 3438.03 | 3498.00 | SL hit (close>ema200) qty=0.50 sl=3438.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 3449.00 | 3438.03 | 3498.00 | SL hit (close>ema200) qty=0.50 sl=3438.03 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:15:00 | 3580.70 | 3473.73 | 3511.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 3738.00 | 3479.37 | 3513.78 | SL hit (close>static) qty=1.00 sl=3682.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 3805.60 | 3543.14 | 3542.92 | EMA200 above EMA400 |
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
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:00:00 | 3625.00 | 3672.78 | 3638.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:30:00 | 3651.70 | 3672.31 | 3638.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 3625.70 | 3671.87 | 3638.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 3623.80 | 3670.93 | 3637.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 3625.70 | 3670.93 | 3637.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 3597.50 | 3670.20 | 3637.76 | SL hit (close<static) qty=1.00 sl=3603.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 3597.50 | 3670.20 | 3637.76 | SL hit (close<static) qty=1.00 sl=3603.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 3597.50 | 3670.20 | 3637.76 | SL hit (close<static) qty=1.00 sl=3603.80 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 3633.00 | 3666.22 | 3636.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 3633.00 | 3666.22 | 3636.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 3636.00 | 3665.92 | 3636.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 3635.40 | 3665.92 | 3636.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 3629.00 | 3665.55 | 3636.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 3654.10 | 3663.48 | 3636.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 3762.00 | 3663.07 | 3636.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:45:00 | 3647.90 | 3672.64 | 3644.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:30:00 | 3645.10 | 3672.46 | 3644.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 3643.20 | 3672.17 | 3644.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:45:00 | 3640.30 | 3672.17 | 3644.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 3659.80 | 3672.04 | 3644.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:15:00 | 3623.00 | 3672.04 | 3644.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 3623.00 | 3671.56 | 3644.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 3603.20 | 3671.56 | 3644.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 3581.80 | 3670.66 | 3644.26 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 3581.80 | 3670.66 | 3644.26 | SL hit (close<static) qty=1.00 sl=3615.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 3581.80 | 3670.66 | 3644.26 | SL hit (close<static) qty=1.00 sl=3615.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 3581.80 | 3670.66 | 3644.26 | SL hit (close<static) qty=1.00 sl=3615.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 3581.80 | 3670.66 | 3644.26 | SL hit (close<static) qty=1.00 sl=3615.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-07 10:00:00 | 3581.80 | 3670.66 | 3644.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 3635.40 | 3647.46 | 3634.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:30:00 | 3659.00 | 3647.46 | 3634.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 3665.00 | 3647.63 | 3635.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:00:00 | 3670.90 | 3647.99 | 3635.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 11:15:00 | 3609.00 | 3648.04 | 3635.65 | SL hit (close<static) qty=1.00 sl=3619.30 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-19 12:15:00)

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
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 3559.70 | 3480.56 | 3524.52 | SL hit (close>static) qty=1.00 sl=3540.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 3559.70 | 3480.56 | 3524.52 | SL hit (close>static) qty=1.00 sl=3540.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 3559.70 | 3480.56 | 3524.52 | SL hit (close>static) qty=1.00 sl=3540.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 3559.70 | 3480.56 | 3524.52 | SL hit (close>static) qty=1.00 sl=3540.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 3559.70 | 3480.56 | 3524.52 | SL hit (close>static) qty=1.00 sl=3540.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 3559.70 | 3480.56 | 3524.52 | SL hit (close>static) qty=1.00 sl=3540.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 3559.70 | 3480.56 | 3524.52 | SL hit (close>static) qty=1.00 sl=3540.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:30:00 | 3520.30 | 3480.99 | 3524.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 3522.60 | 3481.46 | 3524.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 3624.70 | 3484.32 | 3525.12 | SL hit (close>static) qty=1.00 sl=3569.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 3624.70 | 3484.32 | 3525.12 | SL hit (close>static) qty=1.00 sl=3569.70 alert=retest2 |

### Cycle 4 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 3658.10 | 3554.53 | 3554.29 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-07 12:15:00)

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
| Stop hit — per-position SL triggered | 2026-02-06 14:15:00 | 3428.90 | 3295.58 | 3386.04 | SL hit (close>static) qty=1.00 sl=3395.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 3369.90 | 3333.43 | 3396.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 3396.80 | 3335.42 | 3395.68 | SL hit (close>static) qty=1.00 sl=3395.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 3355.70 | 3335.42 | 3395.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 3360.00 | 3335.66 | 3395.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 3277.90 | 3335.66 | 3395.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 3396.50 | 3335.59 | 3393.69 | SL hit (close>static) qty=1.00 sl=3395.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 3396.50 | 3335.59 | 3393.69 | SL hit (close>static) qty=1.00 sl=3395.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 3340.30 | 3336.22 | 3393.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 3403.50 | 3337.30 | 3390.90 | SL hit (close>static) qty=1.00 sl=3395.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 13:30:00 | 3339.00 | 3380.09 | 3400.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 14:45:00 | 3344.70 | 3379.58 | 3400.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 3172.05 | 3357.05 | 3386.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 3177.46 | 3357.05 | 3386.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-13 13:15:00 | 3005.10 | 3315.71 | 3360.10 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-13 13:15:00 | 3010.23 | 3315.71 | 3360.10 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 3270.00 | 3240.27 | 3304.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 14:45:00 | 3278.00 | 3240.27 | 3304.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 3282.90 | 3209.44 | 3275.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:00:00 | 3282.90 | 3209.44 | 3275.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 3278.20 | 3210.13 | 3275.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 13:30:00 | 3243.00 | 3210.39 | 3275.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 3326.00 | 3212.12 | 3275.67 | SL hit (close>static) qty=1.00 sl=3283.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 15:15:00 | 3257.00 | 3268.79 | 3292.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 3296.00 | 3268.97 | 3292.41 | SL hit (close>static) qty=1.00 sl=3283.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 14:30:00 | 3261.30 | 3269.39 | 3292.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 15:15:00 | 3264.70 | 3269.39 | 3292.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 3270.00 | 3269.35 | 3291.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 3295.00 | 3269.65 | 3291.84 | SL hit (close>static) qty=1.00 sl=3283.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 3295.00 | 3269.65 | 3291.84 | SL hit (close>static) qty=1.00 sl=3283.80 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 3585.40 | 3309.52 | 3308.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 3750.00 | 3319.47 | 3313.95 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
