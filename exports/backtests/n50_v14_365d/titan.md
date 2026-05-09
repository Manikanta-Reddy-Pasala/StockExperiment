# TITAN (TITAN)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 4517.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 22
- **Target hits / Stop hits / Partials:** 2 / 22 / 0
- **Avg / median % per leg:** -0.66% / -1.30%
- **Sum % (uncompounded):** -15.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 2 | 11.1% | 2 | 16 | 0 | -0.21% | -3.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 2 | 11.1% | 2 | 16 | 0 | -0.21% | -3.8% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.01% | -12.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.01% | -12.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 2 | 8.3% | 2 | 22 | 0 | -0.66% | -15.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 3355.10 | 3464.89 | 3465.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 3328.50 | 3460.14 | 3462.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 3449.00 | 3441.68 | 3452.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 12:00:00 | 3449.00 | 3441.68 | 3452.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 3453.00 | 3441.79 | 3452.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 13:30:00 | 3438.60 | 3441.67 | 3452.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 3469.40 | 3439.12 | 3450.49 | SL hit (close>static) qty=1.00 sl=3462.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 3420.60 | 3440.74 | 3450.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 3465.00 | 3440.86 | 3450.89 | SL hit (close>static) qty=1.00 sl=3462.40 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 3559.00 | 3459.85 | 3459.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 3569.70 | 3462.87 | 3460.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 3576.80 | 3579.54 | 3536.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:45:00 | 3577.50 | 3579.54 | 3536.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 3526.10 | 3578.50 | 3537.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 3526.10 | 3578.50 | 3537.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 3535.30 | 3578.07 | 3537.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:30:00 | 3526.50 | 3578.07 | 3537.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 3536.60 | 3577.66 | 3537.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 3530.80 | 3577.66 | 3537.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 3540.70 | 3577.29 | 3537.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:30:00 | 3545.00 | 3576.85 | 3537.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 3532.00 | 3576.41 | 3537.05 | SL hit (close<static) qty=1.00 sl=3532.40 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:45:00 | 3544.20 | 3574.35 | 3536.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 3557.00 | 3574.18 | 3536.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 3527.00 | 3573.06 | 3537.06 | SL hit (close<static) qty=1.00 sl=3532.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 3527.00 | 3573.06 | 3537.06 | SL hit (close<static) qty=1.00 sl=3532.40 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 3544.70 | 3570.55 | 3536.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 3533.80 | 3570.19 | 3536.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:15:00 | 3530.20 | 3570.19 | 3536.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 3528.50 | 3569.77 | 3536.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-18 11:15:00 | 3528.50 | 3569.77 | 3536.80 | SL hit (close<static) qty=1.00 sl=3532.40 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-18 11:30:00 | 3523.40 | 3569.77 | 3536.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 11:15:00 | 3380.00 | 3511.17 | 3511.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 09:15:00 | 3361.00 | 3496.94 | 3504.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 3571.00 | 3479.34 | 3493.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 3560.00 | 3480.14 | 3494.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 3539.00 | 3480.94 | 3494.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 3547.90 | 3484.33 | 3495.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 3555.40 | 3489.09 | 3497.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:45:00 | 3553.50 | 3491.24 | 3498.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 3636.30 | 3501.02 | 3503.16 | SL hit (close>static) qty=1.00 sl=3578.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 3636.30 | 3501.02 | 3503.16 | SL hit (close>static) qty=1.00 sl=3578.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 3636.30 | 3501.02 | 3503.16 | SL hit (close>static) qty=1.00 sl=3578.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 3636.30 | 3501.02 | 3503.16 | SL hit (close>static) qty=1.00 sl=3578.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 3641.40 | 3506.57 | 3505.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3694.00 | 3511.02 | 3508.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 13:15:00 | 3810.00 | 3813.27 | 3724.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 14:00:00 | 3810.00 | 3813.27 | 3724.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 3999.00 | 4056.22 | 3958.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 4011.70 | 4055.69 | 3958.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 3891.70 | 4048.75 | 3958.75 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 13:00:00 | 4044.90 | 4031.74 | 3957.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 3944.00 | 4031.09 | 3957.95 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 4070.00 | 4024.54 | 3957.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 3946.40 | 4154.57 | 4112.10 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:00:00 | 4013.00 | 4121.34 | 4097.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 3936.70 | 4107.20 | 4092.18 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4083.20 | 4099.09 | 4088.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 15:00:00 | 4100.40 | 4092.87 | 4085.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 4152.60 | 4092.74 | 4085.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 12:15:00 | 4510.44 | 4167.45 | 4126.55 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-05-08 14:15:00 | 4567.86 | 4342.53 | 4265.02 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 13:30:00 | 3438.40 | 2025-07-11 09:15:00 | 3400.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-16 13:15:00 | 3438.40 | 2025-07-11 09:15:00 | 3400.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-16 14:00:00 | 3442.90 | 2025-07-11 09:15:00 | 3400.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-16 14:45:00 | 3442.80 | 2025-07-11 09:15:00 | 3400.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-18 09:30:00 | 3448.00 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-06-18 14:45:00 | 3464.80 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-07-09 12:00:00 | 3438.30 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-17 11:30:00 | 3440.90 | 2025-07-21 09:15:00 | 3395.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-08-06 13:30:00 | 3438.60 | 2025-08-08 09:15:00 | 3469.40 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-08-11 09:15:00 | 3420.60 | 2025-08-11 11:15:00 | 3465.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-15 14:30:00 | 3545.00 | 2025-09-15 15:15:00 | 3532.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-09-16 12:45:00 | 3544.20 | 2025-09-17 10:15:00 | 3527.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-09-16 14:00:00 | 3557.00 | 2025-09-17 10:15:00 | 3527.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-18 10:00:00 | 3544.70 | 2025-09-18 11:15:00 | 3528.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-10-08 12:15:00 | 3539.00 | 2025-10-16 09:15:00 | 3636.30 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-10-09 09:15:00 | 3547.90 | 2025-10-16 09:15:00 | 3636.30 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-10-10 10:15:00 | 3555.40 | 2025-10-16 09:15:00 | 3636.30 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-10-10 12:45:00 | 3553.50 | 2025-10-16 09:15:00 | 3636.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-01-28 09:15:00 | 4011.70 | 2026-01-29 09:15:00 | 3891.70 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2026-02-01 13:00:00 | 4044.90 | 2026-02-01 15:15:00 | 3944.00 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-02-03 09:15:00 | 4070.00 | 2026-03-23 09:15:00 | 3946.40 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2026-03-25 10:00:00 | 4013.00 | 2026-03-30 10:15:00 | 3936.70 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-02 15:00:00 | 4100.40 | 2026-04-10 12:15:00 | 4510.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 09:15:00 | 4152.60 | 2026-05-08 14:15:00 | 4567.86 | TARGET_HIT | 1.00 | 10.00% |
