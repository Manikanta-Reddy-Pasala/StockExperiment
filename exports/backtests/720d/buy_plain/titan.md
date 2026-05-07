# TITAN (TITAN)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 4310.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 6 |
| PENDING | 25 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 2 |
| ENTRY2 | 16 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 12
- **Target hits / Stop hits / Partials:** 0 / 17 / 0
- **Avg / median % per leg:** -1.24% / -1.76%
- **Sum % (uncompounded):** -21.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 5 | 29.4% | 0 | 17 | 0 | -1.24% | -21.0% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.92% | 1.8% |
| BUY @ 3rd Alert (retest2) | 15 | 3 | 20.0% | 0 | 15 | 0 | -1.52% | -22.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.92% | 1.8% |
| retest2 (combined) | 15 | 3 | 20.0% | 0 | 15 | 0 | -1.52% | -22.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 11:15:00 | 3463.65 | 3387.58 | 3387.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 3485.40 | 3373.12 | 3378.28 | Break + close above crossover candle high |

### Cycle 2 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 3575.60 | 3384.84 | 3384.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 09:15:00 | 3623.15 | 3392.76 | 3388.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 3408.15 | 3420.62 | 3403.64 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 14:15:00 | 3406.10 | 3420.47 | 3403.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 3406.10 | 3420.47 | 3403.65 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-06 15:15:00 | 3413.00 | 3420.40 | 3403.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 3438.10 | 3420.57 | 3403.87 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 3377.75 | 3420.91 | 3404.62 | SL hit (close<static) qty=1.00 sl=3397.85 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 3379.70 | 3223.88 | 3223.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 11:15:00 | 3402.60 | 3237.64 | 3230.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 3499.00 | 3499.42 | 3425.69 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-12 12:15:00 | 3512.00 | 3499.55 | 3426.12 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-12 13:15:00 | 3461.40 | 3499.17 | 3426.29 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 3408.90 | 3497.31 | 3426.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 3408.90 | 3497.31 | 3426.44 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-16 13:15:00 | 3442.90 | 3490.01 | 3426.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:15:00 | 3436.90 | 3489.48 | 3426.53 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-18 14:15:00 | 3476.90 | 3480.68 | 3426.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 3478.80 | 3480.66 | 3426.48 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-11 13:15:00 | 3374.00 | 3544.40 | 3494.52 | SL hit (close<static) qty=1.00 sl=3383.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 13:15:00 | 3374.00 | 3544.40 | 3494.52 | SL hit (close<static) qty=1.00 sl=3383.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-17 10:15:00 | 3436.00 | 3512.84 | 3483.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:15:00 | 3439.20 | 3512.10 | 3483.32 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-22 09:15:00 | 3470.20 | 3496.16 | 3477.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 3483.30 | 3496.03 | 3477.61 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 3473.80 | 3495.81 | 3477.59 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-24 10:15:00 | 3485.40 | 3493.05 | 3477.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-24 11:15:00 | 3481.10 | 3492.93 | 3477.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-24 14:15:00 | 3487.50 | 3492.69 | 3477.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-24 15:15:00 | 3476.30 | 3492.53 | 3477.43 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 3380.00 | 3483.89 | 3474.09 | SL hit (close<static) qty=1.00 sl=3383.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 3380.00 | 3483.89 | 3474.09 | SL hit (close<static) qty=1.00 sl=3383.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-12 09:15:00 | 3526.50 | 3442.75 | 3451.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 3497.40 | 3443.30 | 3451.82 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-13 09:15:00 | 3491.90 | 3446.22 | 3453.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-13 10:15:00 | 3479.50 | 3446.55 | 3453.18 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-14 12:15:00 | 3504.80 | 3449.22 | 3454.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 13:15:00 | 3492.70 | 3449.66 | 3454.45 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-18 09:15:00 | 3507.90 | 3450.98 | 3455.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:15:00 | 3525.30 | 3451.72 | 3455.39 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 3559.00 | 3459.85 | 3459.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 3559.00 | 3459.85 | 3459.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 3559.00 | 3459.85 | 3459.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 3559.00 | 3459.85 | 3459.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 3569.70 | 3462.87 | 3460.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 3576.80 | 3579.54 | 3536.54 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 10:15:00 | 3526.10 | 3578.50 | 3537.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 3526.10 | 3578.50 | 3537.08 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-08 10:15:00 | 3560.00 | 3480.14 | 3494.22 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-08 12:15:00 | 3573.90 | 3481.86 | 3494.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-08 13:15:00 | 3562.30 | 3482.66 | 3495.28 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-08 14:15:00 | 3566.10 | 3483.49 | 3495.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 3567.00 | 3484.33 | 3495.99 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-10 11:15:00 | 3569.20 | 3490.58 | 3498.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-10 12:15:00 | 3556.50 | 3491.24 | 3498.91 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 3511.00 | 3496.08 | 3500.88 | SL hit (close<static) qty=1.00 sl=3515.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-16 09:15:00 | 3636.30 | 3501.02 | 3503.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 3643.80 | 3502.44 | 3503.86 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 3641.40 | 3506.57 | 3505.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 3641.40 | 3506.57 | 3505.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3694.00 | 3511.02 | 3508.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 13:15:00 | 3810.00 | 3813.27 | 3724.93 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-09 11:15:00 | 3841.70 | 3811.72 | 3734.90 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 12:15:00 | 3867.60 | 3812.28 | 3735.56 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-11 13:15:00 | 3839.40 | 3815.82 | 3742.96 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 14:15:00 | 3845.10 | 3816.11 | 3743.47 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 3999.00 | 4056.22 | 3958.24 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 3891.70 | 4048.75 | 3958.75 | SL hit (close<ema400) qty=1.00 sl=3958.75 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 3891.70 | 4048.75 | 3958.75 | SL hit (close<ema400) qty=1.00 sl=3958.75 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-01 12:15:00 | 4044.90 | 4031.74 | 3957.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 13:15:00 | 4087.30 | 4032.30 | 3957.81 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 3944.00 | 4031.09 | 3957.95 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 4069.00 | 4024.98 | 3957.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 4085.10 | 4025.58 | 3958.32 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 3946.40 | 4154.57 | 4112.10 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 4013.00 | 4121.34 | 4097.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 4059.90 | 4120.73 | 4097.56 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 4024.50 | 4116.17 | 4096.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 4024.80 | 4115.26 | 4095.69 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 4008.70 | 4114.20 | 4095.26 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 3936.70 | 4107.20 | 4092.18 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 3936.70 | 4107.20 | 4092.18 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 4073.80 | 4099.25 | 4088.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 4083.20 | 4099.09 | 4088.56 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 3984.70 | 4096.73 | 4087.68 | SL hit (close<static) qty=1.00 sl=4000.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-02 13:15:00 | 4071.10 | 4092.79 | 4085.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:15:00 | 4100.40 | 4092.87 | 4085.94 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-02-07 09:15:00 | 3438.10 | 2025-02-10 09:15:00 | 3377.75 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-06-16 14:15:00 | 3436.90 | 2025-07-11 13:15:00 | 3374.00 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-06-18 15:15:00 | 3478.80 | 2025-07-11 13:15:00 | 3374.00 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2025-07-17 11:15:00 | 3439.20 | 2025-07-29 09:15:00 | 3380.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-07-22 10:15:00 | 3483.30 | 2025-07-29 09:15:00 | 3380.00 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-08-12 10:15:00 | 3497.40 | 2025-08-19 11:15:00 | 3559.00 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2025-08-14 13:15:00 | 3492.70 | 2025-08-19 11:15:00 | 3559.00 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2025-08-18 10:15:00 | 3525.30 | 2025-08-19 11:15:00 | 3559.00 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-10-08 15:15:00 | 3567.00 | 2025-10-14 13:15:00 | 3511.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-10-16 10:15:00 | 3643.80 | 2025-10-16 13:15:00 | 3641.40 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest1 | 2025-12-09 12:15:00 | 3867.60 | 2026-01-29 09:15:00 | 3891.70 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest1 | 2025-12-11 14:15:00 | 3845.10 | 2026-01-29 09:15:00 | 3891.70 | STOP_HIT | 1.00 | 1.21% |
| BUY | retest2 | 2026-02-01 13:15:00 | 4087.30 | 2026-02-01 15:15:00 | 3944.00 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2026-02-03 10:15:00 | 4085.10 | 2026-03-23 09:15:00 | 3946.40 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-03-25 10:15:00 | 4059.90 | 2026-03-30 10:15:00 | 3936.70 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2026-03-27 11:15:00 | 4024.80 | 2026-03-30 10:15:00 | 3936.70 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-04-01 10:15:00 | 4083.20 | 2026-04-02 09:15:00 | 3984.70 | STOP_HIT | 1.00 | -2.41% |
