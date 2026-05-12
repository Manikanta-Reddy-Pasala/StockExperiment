# Titan Company Ltd. (TITAN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 4517.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 74 |
| ALERT1 | 46 |
| ALERT2 | 46 |
| ALERT2_SKIP | 25 |
| ALERT3 | 123 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 70 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 71 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 49
- **Target hits / Stop hits / Partials:** 0 / 70 / 1
- **Avg / median % per leg:** 0.10% / -0.71%
- **Sum % (uncompounded):** 7.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 19 | 42.2% | 0 | 45 | 0 | 0.95% | 42.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 45 | 19 | 42.2% | 0 | 45 | 0 | 0.95% | 42.8% |
| SELL (all) | 26 | 3 | 11.5% | 0 | 25 | 1 | -1.38% | -35.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 3 | 11.5% | 0 | 25 | 1 | -1.38% | -35.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 71 | 22 | 31.0% | 0 | 70 | 1 | 0.10% | 7.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 3597.90 | 3611.60 | 3612.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 3581.20 | 3605.52 | 3609.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 3608.60 | 3601.96 | 3606.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 3608.60 | 3601.96 | 3606.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 3608.60 | 3601.96 | 3606.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 3608.60 | 3601.96 | 3606.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 3613.00 | 3604.16 | 3607.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 3616.00 | 3604.16 | 3607.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 3580.00 | 3599.33 | 3604.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 3577.70 | 3599.33 | 3604.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:30:00 | 3578.30 | 3594.66 | 3601.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 14:30:00 | 3576.80 | 3591.73 | 3599.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 3537.10 | 3590.18 | 3598.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 3573.10 | 3558.09 | 3572.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 3573.10 | 3558.09 | 3572.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 3597.40 | 3565.95 | 3574.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 3597.40 | 3565.95 | 3574.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 3585.00 | 3569.76 | 3575.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:30:00 | 3579.40 | 3576.85 | 3577.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 3628.90 | 3587.76 | 3582.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 3628.90 | 3587.76 | 3582.60 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 3563.00 | 3585.82 | 3588.63 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 3600.00 | 3585.41 | 3583.75 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 3566.00 | 3580.52 | 3581.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 3549.10 | 3571.88 | 3577.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 15:15:00 | 3534.90 | 3530.50 | 3547.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 09:15:00 | 3519.80 | 3530.50 | 3547.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 3520.00 | 3528.40 | 3544.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 3498.00 | 3522.62 | 3534.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:30:00 | 3505.00 | 3508.10 | 3516.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:00:00 | 3503.60 | 3510.83 | 3516.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 12:15:00 | 3537.40 | 3517.08 | 3516.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 3537.40 | 3517.08 | 3516.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 3554.30 | 3524.53 | 3519.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 3531.90 | 3538.42 | 3529.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 10:15:00 | 3531.90 | 3538.42 | 3529.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 3531.90 | 3538.42 | 3529.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:30:00 | 3535.30 | 3538.42 | 3529.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 3530.90 | 3536.92 | 3529.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:45:00 | 3526.90 | 3536.92 | 3529.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 3536.10 | 3536.75 | 3529.93 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 3520.70 | 3528.50 | 3528.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 3512.50 | 3523.17 | 3525.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 11:15:00 | 3532.80 | 3522.86 | 3525.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 11:15:00 | 3532.80 | 3522.86 | 3525.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3532.80 | 3522.86 | 3525.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 3532.80 | 3522.86 | 3525.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 3527.10 | 3523.71 | 3525.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:15:00 | 3520.30 | 3523.71 | 3525.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 14:15:00 | 3539.70 | 3527.00 | 3526.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 14:15:00 | 3539.70 | 3527.00 | 3526.59 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 3502.10 | 3523.79 | 3525.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 3499.00 | 3518.83 | 3523.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 3433.70 | 3431.21 | 3452.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:30:00 | 3434.60 | 3431.21 | 3452.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 3436.90 | 3434.22 | 3450.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 3442.80 | 3434.22 | 3450.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 3428.60 | 3417.00 | 3429.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 3431.50 | 3417.00 | 3429.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 3418.40 | 3417.28 | 3428.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 3411.20 | 3417.28 | 3428.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 14:15:00 | 3476.90 | 3432.82 | 3432.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 3476.90 | 3432.82 | 3432.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 15:15:00 | 3478.80 | 3442.02 | 3436.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 15:15:00 | 3510.40 | 3513.97 | 3494.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:15:00 | 3493.50 | 3513.97 | 3494.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 3475.80 | 3506.34 | 3492.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 3475.80 | 3506.34 | 3492.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 3487.70 | 3502.61 | 3491.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 3489.00 | 3499.89 | 3491.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 14:15:00 | 3680.90 | 3692.89 | 3693.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 3680.90 | 3692.89 | 3693.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 09:15:00 | 3663.80 | 3685.17 | 3689.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 12:15:00 | 3677.50 | 3675.32 | 3683.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 13:00:00 | 3677.50 | 3675.32 | 3683.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 3687.80 | 3677.57 | 3682.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 3687.80 | 3677.57 | 3682.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 3684.00 | 3678.86 | 3682.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 3691.20 | 3678.19 | 3682.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 3404.90 | 3388.43 | 3407.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 3410.00 | 3388.43 | 3407.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 3402.00 | 3391.14 | 3406.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 3413.00 | 3391.14 | 3406.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3395.00 | 3391.91 | 3405.63 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 3415.40 | 3409.60 | 3408.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 10:15:00 | 3436.00 | 3417.33 | 3412.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 3429.90 | 3429.96 | 3421.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:15:00 | 3424.50 | 3429.96 | 3421.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 3414.40 | 3426.85 | 3421.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 3413.80 | 3426.85 | 3421.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 3406.30 | 3422.74 | 3419.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 3403.70 | 3422.74 | 3419.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 3407.60 | 3417.16 | 3417.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 3404.40 | 3413.11 | 3415.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 3428.00 | 3408.19 | 3411.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 3428.00 | 3408.19 | 3411.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 3428.00 | 3408.19 | 3411.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 3428.00 | 3408.19 | 3411.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 3433.80 | 3413.31 | 3413.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 3470.20 | 3430.19 | 3421.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 3460.00 | 3461.97 | 3446.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 3460.00 | 3461.97 | 3446.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 3460.00 | 3461.97 | 3446.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 3450.20 | 3461.97 | 3446.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 3472.80 | 3470.11 | 3458.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 3481.10 | 3470.11 | 3458.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 3477.90 | 3479.12 | 3469.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 3456.50 | 3471.69 | 3468.06 | SL hit (close<static) qty=1.00 sl=3458.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 3451.20 | 3466.69 | 3466.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 3433.60 | 3460.07 | 3463.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 14:15:00 | 3377.60 | 3369.46 | 3387.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 3377.60 | 3369.46 | 3387.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 3338.00 | 3330.53 | 3345.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 3333.70 | 3330.53 | 3345.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 3351.00 | 3335.86 | 3344.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 3351.00 | 3335.86 | 3344.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 3359.00 | 3340.48 | 3345.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 3359.00 | 3340.48 | 3345.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 3358.70 | 3344.13 | 3346.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 3355.90 | 3344.13 | 3346.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 3358.00 | 3349.18 | 3348.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 3374.00 | 3354.15 | 3350.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 14:15:00 | 3418.90 | 3422.24 | 3399.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 15:00:00 | 3418.90 | 3422.24 | 3399.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 3426.50 | 3419.53 | 3402.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 3442.90 | 3408.56 | 3403.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:45:00 | 3448.30 | 3420.73 | 3409.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:30:00 | 3437.40 | 3447.22 | 3434.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 3591.60 | 3621.11 | 3621.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 3591.60 | 3621.11 | 3621.12 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 11:15:00 | 3641.00 | 3623.07 | 3621.42 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 3613.00 | 3620.54 | 3621.55 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 3634.50 | 3623.98 | 3622.68 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 15:15:00 | 3618.00 | 3621.71 | 3622.10 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 3637.00 | 3624.77 | 3623.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 3642.50 | 3628.32 | 3625.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 3617.60 | 3629.49 | 3626.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 3617.60 | 3629.49 | 3626.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3617.60 | 3629.49 | 3626.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 3617.60 | 3629.49 | 3626.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 3623.00 | 3628.19 | 3626.44 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 3612.20 | 3624.99 | 3625.15 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 3630.80 | 3626.15 | 3625.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 3648.30 | 3630.58 | 3627.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 15:15:00 | 3678.00 | 3683.97 | 3669.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 09:15:00 | 3681.00 | 3683.97 | 3669.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 3672.00 | 3683.04 | 3672.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 3672.00 | 3683.04 | 3672.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 3673.40 | 3681.11 | 3672.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 3673.40 | 3681.11 | 3672.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 3678.80 | 3680.65 | 3673.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 3673.00 | 3680.65 | 3673.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 3666.80 | 3677.88 | 3672.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 3666.80 | 3677.88 | 3672.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 3660.00 | 3674.30 | 3671.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 3655.10 | 3674.30 | 3671.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 3669.70 | 3673.38 | 3671.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 3677.30 | 3673.71 | 3671.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 3675.80 | 3673.71 | 3671.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 12:15:00 | 3659.60 | 3669.54 | 3670.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 12:15:00 | 3659.60 | 3669.54 | 3670.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 3625.00 | 3659.29 | 3665.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 12:15:00 | 3538.90 | 3533.51 | 3553.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 3538.90 | 3533.51 | 3553.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 3557.00 | 3538.21 | 3553.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 3557.00 | 3538.21 | 3553.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 3555.90 | 3541.74 | 3554.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 3540.00 | 3544.26 | 3553.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3363.00 | 3390.38 | 3409.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 3393.50 | 3391.00 | 3407.77 | SL hit (close>ema200) qty=0.50 sl=3391.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 3398.10 | 3389.26 | 3388.55 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 12:15:00 | 3367.70 | 3387.58 | 3388.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 09:15:00 | 3361.00 | 3375.96 | 3381.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 3384.30 | 3376.12 | 3380.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 11:15:00 | 3384.30 | 3376.12 | 3380.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 3384.30 | 3376.12 | 3380.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:45:00 | 3385.20 | 3376.12 | 3380.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 3385.40 | 3377.97 | 3381.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 3385.40 | 3377.97 | 3381.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 3387.60 | 3379.90 | 3381.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 3387.60 | 3379.90 | 3381.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 3406.20 | 3385.16 | 3384.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 3416.00 | 3396.98 | 3390.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 3420.40 | 3421.17 | 3406.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:30:00 | 3427.00 | 3421.17 | 3406.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 3407.00 | 3418.33 | 3406.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 3407.00 | 3418.33 | 3406.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 3425.00 | 3419.67 | 3408.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:15:00 | 3432.00 | 3419.67 | 3408.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:45:00 | 3430.10 | 3428.24 | 3418.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 3551.40 | 3427.88 | 3422.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 3517.00 | 3527.67 | 3528.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 3517.00 | 3527.67 | 3528.60 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 3546.00 | 3529.75 | 3528.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 3636.30 | 3557.27 | 3542.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 3742.90 | 3756.06 | 3727.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 09:45:00 | 3748.90 | 3756.06 | 3727.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 3724.00 | 3749.65 | 3727.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 3724.00 | 3749.65 | 3727.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 3732.00 | 3746.12 | 3727.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 3723.70 | 3746.12 | 3727.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 3720.00 | 3740.90 | 3727.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 3720.00 | 3740.90 | 3727.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 3714.40 | 3735.60 | 3726.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:00:00 | 3714.40 | 3735.60 | 3726.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 3724.00 | 3728.76 | 3725.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 3723.50 | 3728.76 | 3725.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 3738.60 | 3730.72 | 3726.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 3746.20 | 3730.72 | 3726.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:45:00 | 3749.50 | 3733.20 | 3727.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 14:15:00 | 3752.30 | 3734.56 | 3729.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 3760.00 | 3735.01 | 3730.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 3729.90 | 3733.98 | 3730.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 3725.00 | 3733.98 | 3730.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 3692.10 | 3725.61 | 3726.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 3692.10 | 3725.61 | 3726.75 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 3738.50 | 3723.08 | 3722.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 3750.70 | 3732.26 | 3727.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 3730.50 | 3737.34 | 3731.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 3730.50 | 3737.34 | 3731.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 3730.50 | 3737.34 | 3731.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 3725.30 | 3737.34 | 3731.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 3772.00 | 3744.27 | 3735.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:30:00 | 3777.30 | 3753.50 | 3744.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 3675.30 | 3737.81 | 3742.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 3675.30 | 3737.81 | 3742.30 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 3807.00 | 3746.37 | 3742.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 3815.40 | 3770.12 | 3754.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 3791.60 | 3795.38 | 3775.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 3786.80 | 3793.66 | 3776.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 3786.80 | 3793.66 | 3776.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 3780.00 | 3793.66 | 3776.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 3780.90 | 3791.11 | 3776.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:45:00 | 3778.70 | 3791.11 | 3776.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 3791.80 | 3791.25 | 3778.03 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 12:15:00 | 3771.00 | 3774.00 | 3774.15 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 3797.00 | 3776.88 | 3775.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 3831.20 | 3802.97 | 3793.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 3834.00 | 3842.70 | 3821.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:15:00 | 3842.50 | 3842.70 | 3821.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 3820.00 | 3839.12 | 3828.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 3820.00 | 3839.12 | 3828.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 3835.00 | 3838.29 | 3829.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:30:00 | 3850.00 | 3835.20 | 3829.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 3813.20 | 3829.89 | 3827.90 | SL hit (close<static) qty=1.00 sl=3817.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 3809.50 | 3825.81 | 3826.23 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 3841.70 | 3826.33 | 3825.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 3864.40 | 3836.90 | 3831.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 3852.80 | 3854.47 | 3843.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 3852.80 | 3854.47 | 3843.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 3852.80 | 3854.47 | 3843.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:30:00 | 3848.00 | 3854.47 | 3843.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 3859.00 | 3854.88 | 3846.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 3846.30 | 3854.88 | 3846.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 3902.00 | 3917.84 | 3901.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 3902.00 | 3917.84 | 3901.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 3900.00 | 3914.27 | 3901.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 3905.50 | 3914.27 | 3901.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 3910.60 | 3913.54 | 3902.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:45:00 | 3925.30 | 3912.21 | 3906.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 12:15:00 | 3887.40 | 3902.78 | 3903.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 3887.40 | 3902.78 | 3903.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 3872.20 | 3894.08 | 3898.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 3906.70 | 3896.30 | 3899.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 3906.70 | 3896.30 | 3899.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 3906.70 | 3896.30 | 3899.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 3906.70 | 3896.30 | 3899.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 3896.80 | 3896.40 | 3898.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:15:00 | 3890.70 | 3896.40 | 3898.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 3910.60 | 3882.72 | 3887.38 | SL hit (close>static) qty=1.00 sl=3908.40 alert=retest2 |

### Cycle 40 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 3902.70 | 3892.16 | 3890.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 3915.90 | 3898.04 | 3893.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 3898.90 | 3903.78 | 3898.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 12:15:00 | 3898.90 | 3903.78 | 3898.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 3898.90 | 3903.78 | 3898.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 3898.90 | 3903.78 | 3898.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 3906.80 | 3904.39 | 3898.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 3915.90 | 3904.74 | 3899.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 3912.10 | 3910.49 | 3904.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 3884.50 | 3903.51 | 3903.29 | SL hit (close<static) qty=1.00 sl=3892.60 alert=retest2 |

### Cycle 41 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 3866.60 | 3896.13 | 3899.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 3858.00 | 3880.70 | 3886.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 3814.10 | 3811.63 | 3831.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:00:00 | 3814.10 | 3811.63 | 3831.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 3808.30 | 3809.98 | 3820.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:30:00 | 3820.00 | 3809.98 | 3820.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 3820.00 | 3795.44 | 3806.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 3820.00 | 3795.44 | 3806.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 3841.70 | 3804.69 | 3809.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 3841.70 | 3804.69 | 3809.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 3867.60 | 3817.27 | 3814.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 10:15:00 | 3885.30 | 3851.53 | 3840.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 3864.10 | 3869.95 | 3856.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 3864.10 | 3869.95 | 3856.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 3864.10 | 3869.95 | 3856.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 3854.00 | 3869.95 | 3856.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 3863.20 | 3868.60 | 3857.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 3863.20 | 3868.60 | 3857.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 3861.00 | 3867.08 | 3857.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 3857.10 | 3867.08 | 3857.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 3855.20 | 3863.97 | 3859.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 3866.80 | 3863.97 | 3859.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 3887.50 | 3868.68 | 3861.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 3925.30 | 3899.24 | 3880.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:30:00 | 3921.00 | 3907.11 | 3898.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:15:00 | 3918.50 | 3907.11 | 3898.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 13:00:00 | 3919.20 | 3909.53 | 3900.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 3922.70 | 3911.84 | 3903.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 3924.40 | 3912.19 | 3904.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 3924.40 | 3930.54 | 3929.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 3916.70 | 3927.77 | 3928.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 3916.70 | 3927.77 | 3928.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 10:15:00 | 3908.00 | 3923.82 | 3926.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 15:15:00 | 3920.00 | 3916.76 | 3921.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 09:15:00 | 3928.90 | 3916.76 | 3921.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 44 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 3964.30 | 3926.27 | 3925.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 11:15:00 | 3969.00 | 3941.16 | 3932.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 14:15:00 | 3981.90 | 3993.00 | 3974.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:00:00 | 3981.90 | 3993.00 | 3974.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 3974.80 | 3988.08 | 3975.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 3985.00 | 3987.30 | 3976.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 3985.10 | 3986.86 | 3976.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:00:00 | 3983.70 | 3986.23 | 3977.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 4008.90 | 3981.20 | 3977.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 4037.50 | 3992.46 | 3982.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:15:00 | 4057.50 | 3992.46 | 3982.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:45:00 | 4056.50 | 4005.89 | 3989.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 4055.00 | 4031.94 | 4008.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 4053.50 | 4035.15 | 4012.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 4025.90 | 4044.69 | 4031.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:30:00 | 4060.60 | 4048.94 | 4040.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 4073.70 | 4048.94 | 4040.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 4183.20 | 4213.78 | 4215.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 4183.20 | 4213.78 | 4215.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 4164.80 | 4203.98 | 4210.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 4215.00 | 4196.16 | 4204.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 13:15:00 | 4215.00 | 4196.16 | 4204.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 4215.00 | 4196.16 | 4204.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 4215.00 | 4196.16 | 4204.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 4228.00 | 4202.53 | 4206.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 4238.90 | 4202.53 | 4206.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 4243.00 | 4214.79 | 4211.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 4259.00 | 4236.59 | 4226.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 13:15:00 | 4231.90 | 4240.55 | 4231.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 13:15:00 | 4231.90 | 4240.55 | 4231.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 4231.90 | 4240.55 | 4231.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 4231.90 | 4240.55 | 4231.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 4222.00 | 4236.84 | 4230.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 4220.30 | 4236.84 | 4230.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 4217.60 | 4232.99 | 4229.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 4220.00 | 4232.99 | 4229.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 4217.00 | 4226.42 | 4226.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 4194.40 | 4217.57 | 4222.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 4089.00 | 4086.24 | 4111.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 4116.70 | 4086.24 | 4111.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 4105.00 | 4089.99 | 4111.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 4115.70 | 4089.99 | 4111.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 4055.90 | 4042.86 | 4063.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:45:00 | 4050.40 | 4042.86 | 4063.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3974.90 | 3997.60 | 4021.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 3916.20 | 3981.01 | 4001.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 14:00:00 | 3964.80 | 3948.91 | 3956.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 3967.50 | 3959.08 | 3960.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 4044.90 | 3958.28 | 3957.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 4044.90 | 3958.28 | 3957.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 13:15:00 | 4087.30 | 3984.09 | 3969.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 3944.00 | 3978.41 | 3969.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 3944.00 | 3978.41 | 3969.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 3944.00 | 3978.41 | 3969.47 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 3905.00 | 3955.21 | 3959.95 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 4069.00 | 3972.71 | 3964.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 4151.80 | 4080.62 | 4034.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 4092.30 | 4114.92 | 4076.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 4092.00 | 4114.92 | 4076.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 4078.80 | 4107.70 | 4076.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 4078.80 | 4107.70 | 4076.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 4066.60 | 4099.48 | 4075.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 4066.60 | 4099.48 | 4075.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 4082.80 | 4096.14 | 4076.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 4091.00 | 4095.21 | 4077.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 4087.40 | 4090.89 | 4081.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 4209.90 | 4233.84 | 4236.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 4209.90 | 4233.84 | 4236.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 4180.00 | 4223.07 | 4231.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 4202.80 | 4186.90 | 4201.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 4202.80 | 4186.90 | 4201.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 4202.80 | 4186.90 | 4201.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 4202.80 | 4186.90 | 4201.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 4208.10 | 4191.14 | 4202.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:15:00 | 4221.50 | 4191.14 | 4202.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 4228.20 | 4198.55 | 4204.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 4236.90 | 4198.55 | 4204.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 4224.10 | 4209.97 | 4209.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 4255.00 | 4226.75 | 4218.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 4240.90 | 4241.37 | 4230.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 4240.90 | 4241.37 | 4230.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 4240.90 | 4241.37 | 4230.62 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 4200.90 | 4227.13 | 4227.54 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 4241.80 | 4229.65 | 4228.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 4256.00 | 4238.57 | 4233.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 4248.50 | 4258.52 | 4248.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 4248.50 | 4258.52 | 4248.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 4248.50 | 4258.52 | 4248.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 4241.20 | 4258.52 | 4248.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 4271.00 | 4261.02 | 4250.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 11:15:00 | 4275.50 | 4261.02 | 4250.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:00:00 | 4277.90 | 4265.90 | 4255.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 4277.00 | 4315.28 | 4315.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 4277.00 | 4315.28 | 4315.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 4239.00 | 4300.02 | 4308.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 4278.30 | 4274.99 | 4293.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 4278.30 | 4274.99 | 4293.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 4194.20 | 4201.91 | 4232.55 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 4268.20 | 4243.75 | 4241.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 10:15:00 | 4281.50 | 4251.30 | 4244.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 4253.20 | 4260.12 | 4252.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 4253.20 | 4260.12 | 4252.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 4253.20 | 4260.12 | 4252.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:45:00 | 4253.90 | 4260.12 | 4252.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 4235.00 | 4255.10 | 4250.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 4132.50 | 4255.10 | 4250.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 4152.50 | 4234.58 | 4241.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 4127.10 | 4178.42 | 4209.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 4196.10 | 4178.38 | 4201.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 4196.10 | 4178.38 | 4201.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 4196.10 | 4178.38 | 4201.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:15:00 | 4172.50 | 4200.44 | 4203.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 4101.50 | 4090.87 | 4090.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 4101.50 | 4090.87 | 4090.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 4128.20 | 4098.34 | 4093.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4067.00 | 4112.04 | 4103.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4067.00 | 4112.04 | 4103.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4067.00 | 4112.04 | 4103.85 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 4077.00 | 4097.87 | 4098.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 4062.00 | 4087.39 | 4093.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 4130.50 | 4087.07 | 4090.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 4130.50 | 4087.07 | 4090.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 4130.50 | 4087.07 | 4090.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 4130.50 | 4087.07 | 4090.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 4138.60 | 4097.37 | 4095.08 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 3946.40 | 4079.37 | 4090.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 3930.10 | 4049.52 | 4075.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 3921.30 | 3907.06 | 3963.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 3918.50 | 3907.06 | 3963.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 4013.00 | 3929.61 | 3956.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 4013.00 | 3929.61 | 3956.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 4059.90 | 3955.67 | 3965.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 4059.90 | 3955.67 | 3965.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 4091.00 | 3982.73 | 3977.01 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 3961.30 | 3994.95 | 3998.54 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 4083.20 | 4003.49 | 3994.26 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 3954.00 | 4008.45 | 4009.94 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 4071.10 | 4018.14 | 4013.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 4100.40 | 4034.59 | 4021.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 4435.80 | 4438.96 | 4361.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 4435.80 | 4438.96 | 4361.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 4444.30 | 4473.75 | 4433.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:30:00 | 4500.40 | 4473.35 | 4453.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 11:00:00 | 4512.10 | 4473.35 | 4453.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 12:45:00 | 4499.50 | 4484.09 | 4462.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 13:30:00 | 4498.80 | 4490.37 | 4467.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 4450.50 | 4492.04 | 4478.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 4450.50 | 4492.04 | 4478.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 4449.20 | 4483.47 | 4476.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:45:00 | 4449.90 | 4483.47 | 4476.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 4462.00 | 4473.33 | 4472.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:15:00 | 4466.00 | 4473.33 | 4472.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 4425.70 | 4468.01 | 4470.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 10:15:00 | 4425.70 | 4468.01 | 4470.71 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 4528.50 | 4474.51 | 4471.49 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 4461.40 | 4484.53 | 4486.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 4436.10 | 4462.34 | 4473.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 11:15:00 | 4471.90 | 4461.96 | 4471.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 11:15:00 | 4471.90 | 4461.96 | 4471.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 4471.90 | 4461.96 | 4471.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:00:00 | 4471.90 | 4461.96 | 4471.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 4446.60 | 4458.89 | 4469.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 4427.20 | 4456.68 | 4465.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:15:00 | 4430.40 | 4454.39 | 4463.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 09:30:00 | 4433.10 | 4410.60 | 4431.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:45:00 | 4436.00 | 4424.90 | 4431.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 4443.30 | 4428.58 | 4432.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:45:00 | 4448.00 | 4428.58 | 4432.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 4436.30 | 4430.12 | 4433.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 4440.00 | 4430.12 | 4433.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 4468.90 | 4437.88 | 4436.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 4468.90 | 4437.88 | 4436.45 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 4418.80 | 4437.01 | 4437.42 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 4461.00 | 4439.09 | 4438.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 4467.10 | 4444.69 | 4440.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 12:15:00 | 4436.80 | 4443.67 | 4441.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 12:15:00 | 4436.80 | 4443.67 | 4441.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 4436.80 | 4443.67 | 4441.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 4436.80 | 4443.67 | 4441.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 4436.20 | 4442.18 | 4440.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:30:00 | 4423.90 | 4442.18 | 4440.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 4443.70 | 4442.48 | 4440.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:45:00 | 4428.80 | 4442.48 | 4440.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 4439.90 | 4441.97 | 4440.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 4389.80 | 4441.97 | 4440.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 4370.10 | 4427.59 | 4434.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 15:15:00 | 4360.00 | 4380.06 | 4396.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 13:15:00 | 4372.50 | 4370.87 | 4384.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 14:00:00 | 4372.50 | 4370.87 | 4384.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 4390.00 | 4374.16 | 4383.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 4394.10 | 4374.16 | 4383.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 4323.70 | 4364.07 | 4378.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 4301.40 | 4351.53 | 4371.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:45:00 | 4308.10 | 4340.09 | 4356.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:30:00 | 4310.50 | 4311.55 | 4333.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 15:00:00 | 4305.20 | 4311.55 | 4333.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 4301.00 | 4309.19 | 4328.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-08 13:15:00 | 4537.20 | 4348.29 | 4339.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 13:15:00 | 4537.20 | 4348.29 | 4339.53 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-21 12:15:00 | 3577.70 | 2025-05-26 09:15:00 | 3628.90 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-05-21 13:30:00 | 3578.30 | 2025-05-26 09:15:00 | 3628.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-05-21 14:30:00 | 3576.80 | 2025-05-26 09:15:00 | 3628.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-05-22 09:15:00 | 3537.10 | 2025-05-26 09:15:00 | 3628.90 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-05-23 14:30:00 | 3579.40 | 2025-05-26 09:15:00 | 3628.90 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-06-04 09:15:00 | 3498.00 | 2025-06-06 12:15:00 | 3537.40 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-06-05 10:30:00 | 3505.00 | 2025-06-06 12:15:00 | 3537.40 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-06-05 13:00:00 | 3503.60 | 2025-06-06 12:15:00 | 3537.40 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-11 13:15:00 | 3520.30 | 2025-06-11 14:15:00 | 3539.70 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-06-18 11:15:00 | 3411.20 | 2025-06-18 14:15:00 | 3476.90 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-06-23 12:00:00 | 3489.00 | 2025-07-03 14:15:00 | 3680.90 | STOP_HIT | 1.00 | 5.50% |
| BUY | retest2 | 2025-07-24 10:15:00 | 3481.10 | 2025-07-25 11:15:00 | 3456.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-25 09:15:00 | 3477.90 | 2025-07-25 11:15:00 | 3456.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-07-25 13:45:00 | 3476.80 | 2025-07-25 15:15:00 | 3451.20 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-08-08 09:15:00 | 3442.90 | 2025-08-26 14:15:00 | 3591.60 | STOP_HIT | 1.00 | 4.32% |
| BUY | retest2 | 2025-08-08 09:45:00 | 3448.30 | 2025-08-26 14:15:00 | 3591.60 | STOP_HIT | 1.00 | 4.16% |
| BUY | retest2 | 2025-08-11 10:30:00 | 3437.40 | 2025-08-26 14:15:00 | 3591.60 | STOP_HIT | 1.00 | 4.49% |
| BUY | retest2 | 2025-09-08 10:30:00 | 3677.30 | 2025-09-08 12:15:00 | 3659.60 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-09-08 11:15:00 | 3675.80 | 2025-09-08 12:15:00 | 3659.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-17 09:45:00 | 3540.00 | 2025-09-26 09:15:00 | 3363.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 09:45:00 | 3540.00 | 2025-09-26 10:15:00 | 3393.50 | STOP_HIT | 0.50 | 4.14% |
| BUY | retest2 | 2025-10-06 12:15:00 | 3432.00 | 2025-10-14 11:15:00 | 3517.00 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2025-10-07 10:45:00 | 3430.10 | 2025-10-14 11:15:00 | 3517.00 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2025-10-08 09:15:00 | 3551.40 | 2025-10-14 11:15:00 | 3517.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-10-27 12:15:00 | 3746.20 | 2025-10-28 10:15:00 | 3692.10 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-10-27 12:45:00 | 3749.50 | 2025-10-28 10:15:00 | 3692.10 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-10-27 14:15:00 | 3752.30 | 2025-10-28 10:15:00 | 3692.10 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-28 09:15:00 | 3760.00 | 2025-10-28 10:15:00 | 3692.10 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-10-31 09:30:00 | 3777.30 | 2025-11-03 09:15:00 | 3675.30 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-11-14 09:30:00 | 3850.00 | 2025-11-14 11:15:00 | 3813.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-11-24 09:45:00 | 3925.30 | 2025-11-24 12:15:00 | 3887.40 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-25 11:15:00 | 3890.70 | 2025-11-26 10:15:00 | 3910.60 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-26 13:45:00 | 3891.40 | 2025-11-26 14:15:00 | 3902.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-11-28 09:15:00 | 3915.90 | 2025-12-01 09:15:00 | 3884.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-28 13:15:00 | 3912.10 | 2025-12-01 09:15:00 | 3884.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-12-16 14:45:00 | 3925.30 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-12-18 11:30:00 | 3921.00 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-12-18 12:15:00 | 3918.50 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-12-18 13:00:00 | 3919.20 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-12-19 09:15:00 | 3924.40 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-12-24 09:15:00 | 3924.40 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-12-30 10:45:00 | 3985.00 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 4.97% |
| BUY | retest2 | 2025-12-30 12:00:00 | 3985.10 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 4.97% |
| BUY | retest2 | 2025-12-30 13:00:00 | 3983.70 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 5.01% |
| BUY | retest2 | 2025-12-31 09:15:00 | 4008.90 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 4.35% |
| BUY | retest2 | 2025-12-31 10:15:00 | 4057.50 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2025-12-31 10:45:00 | 4056.50 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 3.12% |
| BUY | retest2 | 2025-12-31 15:00:00 | 4055.00 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2026-01-01 09:15:00 | 4053.50 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 3.20% |
| BUY | retest2 | 2026-01-05 09:30:00 | 4060.60 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 3.02% |
| BUY | retest2 | 2026-01-05 10:15:00 | 4073.70 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 2.69% |
| SELL | retest2 | 2026-01-29 09:15:00 | 3916.20 | 2026-02-01 12:15:00 | 4044.90 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-01-30 14:00:00 | 3964.80 | 2026-02-01 12:15:00 | 4044.90 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-02-01 09:15:00 | 3967.50 | 2026-02-01 12:15:00 | 4044.90 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-02-05 13:30:00 | 4091.00 | 2026-02-13 13:15:00 | 4209.90 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2026-02-06 11:00:00 | 4087.40 | 2026-02-13 13:15:00 | 4209.90 | STOP_HIT | 1.00 | 3.00% |
| BUY | retest2 | 2026-02-24 11:15:00 | 4275.50 | 2026-03-02 10:15:00 | 4277.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2026-02-24 14:00:00 | 4277.90 | 2026-03-02 10:15:00 | 4277.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2026-03-11 11:15:00 | 4172.50 | 2026-03-18 11:15:00 | 4101.50 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2026-04-15 10:30:00 | 4500.40 | 2026-04-17 10:15:00 | 4425.70 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-04-15 11:00:00 | 4512.10 | 2026-04-17 10:15:00 | 4425.70 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-04-15 12:45:00 | 4499.50 | 2026-04-17 10:15:00 | 4425.70 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-04-15 13:30:00 | 4498.80 | 2026-04-17 10:15:00 | 4425.70 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-04-24 09:15:00 | 4427.20 | 2026-04-28 09:15:00 | 4468.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-04-24 10:15:00 | 4430.40 | 2026-04-28 09:15:00 | 4468.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-04-27 09:30:00 | 4433.10 | 2026-04-28 09:15:00 | 4468.90 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-04-27 13:45:00 | 4436.00 | 2026-04-28 09:15:00 | 4468.90 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-05-06 11:00:00 | 4301.40 | 2026-05-08 13:15:00 | 4537.20 | STOP_HIT | 1.00 | -5.48% |
| SELL | retest2 | 2026-05-07 09:45:00 | 4308.10 | 2026-05-08 13:15:00 | 4537.20 | STOP_HIT | 1.00 | -5.32% |
| SELL | retest2 | 2026-05-07 14:30:00 | 4310.50 | 2026-05-08 13:15:00 | 4537.20 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2026-05-07 15:00:00 | 4305.20 | 2026-05-08 13:15:00 | 4537.20 | STOP_HIT | 1.00 | -5.39% |
