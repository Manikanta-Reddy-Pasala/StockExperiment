# Thermax Ltd. (THERMAX)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 4707.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 91 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 30 |
| ALERT3 | 139 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 66 |
| PARTIAL | 1 |
| TARGET_HIT | 4 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 70 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 56
- **Target hits / Stop hits / Partials:** 4 / 65 / 1
- **Avg / median % per leg:** -0.21% / -0.82%
- **Sum % (uncompounded):** -14.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 11 | 50.0% | 4 | 18 | 0 | 1.29% | 28.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 11 | 50.0% | 4 | 18 | 0 | 1.29% | 28.3% |
| SELL (all) | 48 | 3 | 6.2% | 0 | 47 | 1 | -0.89% | -42.8% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.14% | -3.4% |
| SELL @ 3rd Alert (retest2) | 45 | 3 | 6.7% | 0 | 44 | 1 | -0.88% | -39.4% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.14% | -3.4% |
| retest2 (combined) | 67 | 14 | 20.9% | 4 | 62 | 1 | -0.17% | -11.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3251.00 | 3209.18 | 3207.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 3416.50 | 3343.31 | 3303.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 12:15:00 | 3420.50 | 3428.47 | 3402.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:00:00 | 3420.50 | 3428.47 | 3402.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 3556.80 | 3574.59 | 3553.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 3556.80 | 3574.59 | 3553.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 3560.70 | 3571.81 | 3554.14 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 3514.80 | 3545.67 | 3547.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 3455.20 | 3491.71 | 3514.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 15:15:00 | 3493.00 | 3475.25 | 3494.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 3489.00 | 3476.91 | 3491.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 3489.00 | 3476.91 | 3491.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 3440.90 | 3477.37 | 3486.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 3449.30 | 3436.04 | 3435.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 3449.30 | 3436.04 | 3435.92 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 3420.90 | 3433.01 | 3434.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 3394.50 | 3423.48 | 3429.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 3439.00 | 3412.06 | 3421.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 3439.00 | 3412.06 | 3421.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 3439.00 | 3412.06 | 3421.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 3420.30 | 3412.06 | 3421.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 3474.90 | 3424.62 | 3426.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 3474.90 | 3424.62 | 3426.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 3501.50 | 3440.00 | 3432.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 12:15:00 | 3515.50 | 3455.10 | 3440.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 14:15:00 | 3524.80 | 3526.87 | 3495.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 15:00:00 | 3524.80 | 3526.87 | 3495.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 3552.00 | 3558.16 | 3545.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 3552.00 | 3558.16 | 3545.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 3549.80 | 3558.86 | 3548.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 3546.90 | 3558.86 | 3548.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3545.80 | 3556.25 | 3548.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 3545.80 | 3556.25 | 3548.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 3542.30 | 3553.46 | 3548.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:45:00 | 3544.00 | 3553.46 | 3548.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 3540.10 | 3550.79 | 3547.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 3540.10 | 3550.79 | 3547.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 3563.40 | 3552.87 | 3549.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 3545.00 | 3552.87 | 3549.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 3555.00 | 3564.58 | 3558.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 3508.60 | 3564.58 | 3558.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 3555.40 | 3562.75 | 3558.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 3573.90 | 3562.75 | 3558.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 13:15:00 | 3528.40 | 3554.83 | 3556.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 13:15:00 | 3528.40 | 3554.83 | 3556.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 3483.30 | 3538.87 | 3548.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 3534.10 | 3521.84 | 3534.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 14:15:00 | 3534.10 | 3521.84 | 3534.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 3534.10 | 3521.84 | 3534.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 3534.10 | 3521.84 | 3534.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 3531.50 | 3523.77 | 3534.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 3534.50 | 3523.77 | 3534.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 3522.10 | 3523.43 | 3532.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 3491.50 | 3516.15 | 3527.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 3505.00 | 3498.18 | 3513.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 3505.20 | 3498.18 | 3513.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:45:00 | 3504.70 | 3499.34 | 3512.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 3503.40 | 3501.40 | 3510.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:30:00 | 3501.00 | 3501.40 | 3510.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 3513.70 | 3503.86 | 3510.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 3513.70 | 3503.86 | 3510.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 3517.00 | 3506.49 | 3511.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 3558.90 | 3506.49 | 3511.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 3592.80 | 3523.75 | 3518.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 3592.80 | 3523.75 | 3518.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 3592.80 | 3523.75 | 3518.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 3592.80 | 3523.75 | 3518.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 09:15:00 | 3592.80 | 3523.75 | 3518.86 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 3475.20 | 3521.50 | 3522.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 3456.70 | 3493.07 | 3508.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 3484.00 | 3462.26 | 3486.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:45:00 | 3476.80 | 3462.26 | 3486.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 3490.00 | 3467.81 | 3486.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:30:00 | 3488.00 | 3467.81 | 3486.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 3503.40 | 3474.93 | 3488.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 3503.40 | 3474.93 | 3488.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 3465.00 | 3475.08 | 3484.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 3456.50 | 3475.08 | 3484.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 3457.30 | 3453.31 | 3465.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 10:00:00 | 3462.10 | 3442.13 | 3450.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 12:00:00 | 3464.90 | 3449.86 | 3453.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 3462.20 | 3455.49 | 3455.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 3462.20 | 3455.49 | 3455.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 3462.20 | 3455.49 | 3455.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 3462.20 | 3455.49 | 3455.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 13:15:00 | 3462.20 | 3455.49 | 3455.25 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 10:15:00 | 3450.00 | 3454.85 | 3455.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 11:15:00 | 3443.20 | 3452.52 | 3454.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 3410.30 | 3407.27 | 3428.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:15:00 | 3429.40 | 3407.27 | 3428.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 3441.30 | 3414.08 | 3429.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 3433.10 | 3414.08 | 3429.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 3443.00 | 3419.86 | 3430.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:30:00 | 3447.60 | 3419.86 | 3430.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 3407.90 | 3417.47 | 3428.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:30:00 | 3403.70 | 3416.06 | 3427.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:00:00 | 3401.40 | 3412.37 | 3422.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 3404.30 | 3377.23 | 3391.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 3404.60 | 3384.23 | 3392.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 3434.80 | 3394.34 | 3396.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:00:00 | 3434.80 | 3394.34 | 3396.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 3414.70 | 3401.33 | 3399.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 3414.70 | 3401.33 | 3399.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 3414.70 | 3401.33 | 3399.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 3414.70 | 3401.33 | 3399.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 15:15:00 | 3414.70 | 3401.33 | 3399.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 3489.30 | 3418.93 | 3407.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 3460.10 | 3474.41 | 3453.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 3460.10 | 3474.41 | 3453.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 3455.00 | 3470.53 | 3454.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 3451.80 | 3470.53 | 3454.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 3453.00 | 3467.03 | 3453.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:15:00 | 3442.40 | 3467.03 | 3453.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 3459.50 | 3465.52 | 3454.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:30:00 | 3449.50 | 3465.52 | 3454.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 3450.20 | 3462.46 | 3454.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 3455.00 | 3462.46 | 3454.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 3490.80 | 3468.12 | 3457.37 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 3438.60 | 3452.17 | 3453.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 3429.70 | 3447.68 | 3451.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 3452.90 | 3435.14 | 3441.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 3452.90 | 3435.14 | 3441.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 3452.90 | 3435.14 | 3441.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 3452.90 | 3435.14 | 3441.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 3440.00 | 3436.11 | 3441.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:00:00 | 3435.80 | 3438.16 | 3441.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 3435.00 | 3438.57 | 3441.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 3434.00 | 3429.18 | 3435.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 3462.50 | 3435.73 | 3434.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 3462.50 | 3435.73 | 3434.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 3462.50 | 3435.73 | 3434.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 3462.50 | 3435.73 | 3434.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 12:15:00 | 3473.10 | 3443.20 | 3438.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 15:15:00 | 3444.00 | 3450.74 | 3443.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 15:15:00 | 3444.00 | 3450.74 | 3443.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 3444.00 | 3450.74 | 3443.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:00:00 | 3470.70 | 3447.60 | 3444.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:30:00 | 3471.50 | 3451.40 | 3446.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 3469.80 | 3451.40 | 3446.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 14:15:00 | 3471.30 | 3461.18 | 3452.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-17 09:15:00 | 3817.77 | 3647.74 | 3569.22 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-17 09:15:00 | 3818.65 | 3647.74 | 3569.22 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-17 09:15:00 | 3816.78 | 3647.74 | 3569.22 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-17 09:15:00 | 3818.43 | 3647.74 | 3569.22 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 3870.00 | 3908.33 | 3886.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 3870.00 | 3908.33 | 3886.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 3842.90 | 3895.24 | 3882.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 3842.90 | 3895.24 | 3882.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 3869.10 | 3877.50 | 3876.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 3889.00 | 3877.50 | 3876.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 3872.00 | 3876.37 | 3876.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 3872.00 | 3876.37 | 3876.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 13:15:00 | 3852.60 | 3869.00 | 3872.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 3771.70 | 3741.29 | 3768.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 3771.70 | 3741.29 | 3768.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 3771.70 | 3741.29 | 3768.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 3766.50 | 3741.29 | 3768.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 3769.80 | 3746.99 | 3768.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:15:00 | 3763.00 | 3746.99 | 3768.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 3794.00 | 3769.61 | 3773.69 | SL hit (close>static) qty=1.00 sl=3792.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 3817.30 | 3779.15 | 3777.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 14:15:00 | 3847.80 | 3816.28 | 3798.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 3773.80 | 3810.17 | 3799.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 3773.80 | 3810.17 | 3799.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 3773.80 | 3810.17 | 3799.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 3821.00 | 3818.06 | 3804.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:45:00 | 3836.00 | 3867.41 | 3847.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 15:15:00 | 3767.00 | 3823.23 | 3830.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 15:15:00 | 3767.00 | 3823.23 | 3830.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 3767.00 | 3823.23 | 3830.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 3697.00 | 3797.99 | 3818.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 14:15:00 | 3566.00 | 3560.74 | 3635.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 15:00:00 | 3566.00 | 3560.74 | 3635.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 3298.20 | 3283.92 | 3319.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 3300.00 | 3283.92 | 3319.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 3216.70 | 3278.03 | 3301.26 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 3298.10 | 3287.88 | 3286.74 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 14:15:00 | 3266.70 | 3286.80 | 3286.85 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 3303.00 | 3286.74 | 3285.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 15:15:00 | 3314.50 | 3292.29 | 3288.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 3287.70 | 3291.37 | 3288.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 3287.70 | 3291.37 | 3288.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 3287.70 | 3291.37 | 3288.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 3266.00 | 3291.37 | 3288.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 3280.90 | 3289.28 | 3287.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 3278.70 | 3289.28 | 3287.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 3278.10 | 3287.04 | 3286.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 3278.90 | 3287.04 | 3286.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 3261.60 | 3281.95 | 3284.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 3234.20 | 3269.29 | 3278.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 3241.30 | 3230.51 | 3250.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:45:00 | 3238.70 | 3230.51 | 3250.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 3230.20 | 3232.13 | 3246.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:00:00 | 3225.10 | 3230.40 | 3242.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 13:15:00 | 3271.00 | 3243.96 | 3247.26 | SL hit (close>static) qty=1.00 sl=3268.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 3279.30 | 3254.24 | 3251.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 10:15:00 | 3301.00 | 3275.42 | 3265.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 3259.40 | 3283.31 | 3273.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 3259.40 | 3283.31 | 3273.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 3259.40 | 3283.31 | 3273.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 3259.40 | 3283.31 | 3273.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 3299.00 | 3286.45 | 3275.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 3359.80 | 3286.45 | 3275.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 12:15:00 | 3232.40 | 3270.59 | 3271.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 3232.40 | 3270.59 | 3271.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 3177.30 | 3251.93 | 3263.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 13:15:00 | 3207.90 | 3207.80 | 3229.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:45:00 | 3211.40 | 3207.80 | 3229.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 3183.60 | 3204.48 | 3222.70 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 3255.90 | 3225.93 | 3224.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 3268.90 | 3234.53 | 3228.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 3260.00 | 3264.17 | 3248.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 10:15:00 | 3260.00 | 3264.17 | 3248.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 3260.00 | 3264.17 | 3248.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:45:00 | 3269.40 | 3264.17 | 3248.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 3295.90 | 3312.80 | 3293.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 3295.90 | 3312.80 | 3293.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 3267.80 | 3303.80 | 3290.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:45:00 | 3269.60 | 3303.80 | 3290.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 3262.10 | 3295.46 | 3288.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 3266.60 | 3295.46 | 3288.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 3262.30 | 3288.34 | 3286.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 3262.30 | 3288.34 | 3286.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 3315.10 | 3293.69 | 3288.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 3320.80 | 3305.05 | 3294.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 3437.00 | 3329.88 | 3323.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 3313.20 | 3341.61 | 3344.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 3313.20 | 3341.61 | 3344.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 3313.20 | 3341.61 | 3344.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 3306.00 | 3318.80 | 3329.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 15:15:00 | 3321.00 | 3312.45 | 3320.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 15:15:00 | 3321.00 | 3312.45 | 3320.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 3321.00 | 3312.45 | 3320.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 3329.00 | 3312.45 | 3320.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 3322.80 | 3314.52 | 3321.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 3306.60 | 3314.56 | 3319.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 3345.70 | 3325.27 | 3322.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 3345.70 | 3325.27 | 3322.99 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 3303.60 | 3320.91 | 3321.62 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 3329.10 | 3322.02 | 3321.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 12:15:00 | 3351.30 | 3327.87 | 3324.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 14:15:00 | 3330.00 | 3330.02 | 3325.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 3330.00 | 3330.02 | 3325.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 3330.00 | 3330.02 | 3325.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 3330.00 | 3330.02 | 3325.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 3324.00 | 3328.81 | 3325.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 3336.30 | 3328.81 | 3325.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 3359.20 | 3334.89 | 3328.73 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 3299.20 | 3325.17 | 3325.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 3291.20 | 3310.14 | 3316.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 3350.20 | 3306.53 | 3312.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 3350.20 | 3306.53 | 3312.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 3350.20 | 3306.53 | 3312.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 3350.20 | 3306.53 | 3312.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 3329.80 | 3311.18 | 3313.96 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 3359.00 | 3324.43 | 3319.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 13:15:00 | 3367.10 | 3332.96 | 3324.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 12:15:00 | 3338.80 | 3345.05 | 3335.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 3338.80 | 3345.05 | 3335.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 3338.80 | 3345.05 | 3335.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:45:00 | 3335.70 | 3345.05 | 3335.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 3323.70 | 3340.78 | 3334.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:00:00 | 3323.70 | 3340.78 | 3334.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 3313.60 | 3335.34 | 3332.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 3313.60 | 3335.34 | 3332.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 3282.50 | 3322.14 | 3326.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 3267.10 | 3311.13 | 3321.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 3172.10 | 3162.16 | 3195.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:30:00 | 3184.00 | 3162.16 | 3195.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 3178.10 | 3170.90 | 3186.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:30:00 | 3163.00 | 3168.65 | 3182.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:45:00 | 3159.90 | 3166.15 | 3177.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 15:00:00 | 3163.00 | 3165.52 | 3176.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:45:00 | 3151.90 | 3167.17 | 3174.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 3173.60 | 3168.46 | 3174.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 3173.60 | 3168.46 | 3174.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 3189.10 | 3171.37 | 3174.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 3189.10 | 3171.37 | 3174.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 3182.00 | 3173.50 | 3175.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 3170.00 | 3173.50 | 3175.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 14:30:00 | 3172.20 | 3166.86 | 3169.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 3162.50 | 3166.86 | 3169.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 3192.30 | 3171.21 | 3169.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 3192.30 | 3171.21 | 3169.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 3192.30 | 3171.21 | 3169.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 3192.30 | 3171.21 | 3169.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 3192.30 | 3171.21 | 3169.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 3192.30 | 3171.21 | 3169.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 3192.30 | 3171.21 | 3169.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 3192.30 | 3171.21 | 3169.57 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 3134.20 | 3163.65 | 3166.66 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 3180.30 | 3166.05 | 3164.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 3198.70 | 3174.61 | 3168.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 3181.50 | 3200.72 | 3187.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 3181.50 | 3200.72 | 3187.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3181.50 | 3200.72 | 3187.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 3184.60 | 3200.72 | 3187.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 3192.20 | 3199.01 | 3188.34 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 3155.00 | 3181.90 | 3184.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 3121.70 | 3165.57 | 3176.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 3147.00 | 3127.08 | 3140.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 3147.00 | 3127.08 | 3140.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 3147.00 | 3127.08 | 3140.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 3165.00 | 3127.08 | 3140.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 3165.30 | 3134.72 | 3143.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 3167.90 | 3134.72 | 3143.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 3152.00 | 3138.18 | 3143.82 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 3168.30 | 3149.29 | 3148.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3208.00 | 3166.08 | 3156.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 3157.00 | 3167.51 | 3159.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 11:15:00 | 3157.00 | 3167.51 | 3159.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 3157.00 | 3167.51 | 3159.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 3157.00 | 3167.51 | 3159.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 3132.60 | 3160.53 | 3156.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 3135.10 | 3160.53 | 3156.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 3121.00 | 3152.62 | 3153.43 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3191.70 | 3158.24 | 3155.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 3231.60 | 3205.05 | 3186.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 3184.70 | 3209.54 | 3195.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 3184.70 | 3209.54 | 3195.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 3184.70 | 3209.54 | 3195.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 3184.70 | 3209.54 | 3195.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 3184.80 | 3204.59 | 3194.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 3212.00 | 3204.59 | 3194.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 3253.30 | 3258.75 | 3258.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 3253.30 | 3258.75 | 3258.78 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 3268.70 | 3260.74 | 3259.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 3277.90 | 3264.17 | 3261.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 15:15:00 | 3242.00 | 3260.81 | 3260.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 15:15:00 | 3242.00 | 3260.81 | 3260.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 3242.00 | 3260.81 | 3260.37 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 3248.40 | 3260.97 | 3261.08 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 3264.20 | 3261.62 | 3261.37 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 3250.50 | 3259.39 | 3260.38 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 3271.30 | 3261.77 | 3261.37 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 3254.00 | 3260.22 | 3260.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 3241.10 | 3256.40 | 3258.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 3287.00 | 3225.81 | 3231.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 3287.00 | 3225.81 | 3231.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 3287.00 | 3225.81 | 3231.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 3287.00 | 3225.81 | 3231.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 3262.80 | 3233.21 | 3234.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 3250.10 | 3233.21 | 3234.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 3258.20 | 3238.20 | 3236.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 3258.20 | 3238.20 | 3236.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 12:15:00 | 3267.10 | 3243.98 | 3239.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 3247.00 | 3252.90 | 3245.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 3247.00 | 3252.90 | 3245.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 3247.00 | 3252.90 | 3245.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:30:00 | 3239.20 | 3252.90 | 3245.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 3257.30 | 3253.78 | 3246.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 3245.00 | 3253.78 | 3246.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 3251.00 | 3253.22 | 3247.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 3251.00 | 3253.22 | 3247.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 3235.50 | 3249.68 | 3245.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 3236.80 | 3249.68 | 3245.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 3237.30 | 3247.20 | 3245.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:30:00 | 3248.70 | 3248.90 | 3246.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 3178.70 | 3233.20 | 3239.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 3178.70 | 3233.20 | 3239.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 10:15:00 | 3163.00 | 3219.16 | 3232.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 10:15:00 | 3176.70 | 3157.08 | 3177.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 10:15:00 | 3176.70 | 3157.08 | 3177.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 3176.70 | 3157.08 | 3177.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 3176.70 | 3157.08 | 3177.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 3158.80 | 3157.42 | 3175.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 3055.70 | 3170.97 | 3177.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:15:00 | 2902.91 | 2954.41 | 2985.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 2975.00 | 2940.24 | 2958.67 | SL hit (close>ema200) qty=0.50 sl=2940.24 alert=retest2 |

### Cycle 47 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 2921.00 | 2897.20 | 2894.32 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 2903.40 | 2916.48 | 2916.68 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 2919.20 | 2917.03 | 2916.91 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 2915.20 | 2916.66 | 2916.75 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 2920.00 | 2917.30 | 2917.03 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 2915.00 | 2916.89 | 2916.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 2902.20 | 2913.95 | 2915.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 2914.10 | 2911.10 | 2913.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 13:15:00 | 2914.10 | 2911.10 | 2913.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 2914.10 | 2911.10 | 2913.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 2914.10 | 2911.10 | 2913.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 2926.30 | 2914.14 | 2914.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 2926.30 | 2914.14 | 2914.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 2922.50 | 2915.81 | 2915.32 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 2907.60 | 2914.17 | 2914.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 2877.30 | 2894.14 | 2902.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 2811.50 | 2799.48 | 2828.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 15:00:00 | 2811.50 | 2799.48 | 2828.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 2824.10 | 2801.10 | 2813.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 2824.10 | 2801.10 | 2813.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 2828.60 | 2806.60 | 2814.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 2815.00 | 2812.61 | 2816.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 2916.00 | 2812.35 | 2803.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 2916.00 | 2812.35 | 2803.63 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 2836.50 | 2864.93 | 2868.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 12:15:00 | 2831.90 | 2853.52 | 2862.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 15:15:00 | 2854.00 | 2853.33 | 2859.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:15:00 | 2865.90 | 2853.33 | 2859.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2852.80 | 2853.22 | 2859.14 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 2870.40 | 2862.66 | 2862.09 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 15:15:00 | 2853.00 | 2860.73 | 2861.27 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 2883.50 | 2865.28 | 2863.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 2894.20 | 2878.87 | 2872.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 3001.00 | 3004.39 | 2978.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 10:45:00 | 2995.90 | 3004.39 | 2978.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2971.90 | 2999.41 | 2987.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 11:45:00 | 3010.50 | 3000.43 | 2990.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 3006.60 | 3006.32 | 2999.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 3011.80 | 3010.26 | 3004.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 3014.10 | 3034.54 | 3036.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 3014.10 | 3034.54 | 3036.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 3014.10 | 3034.54 | 3036.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 3014.10 | 3034.54 | 3036.39 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 3083.80 | 3042.75 | 3039.70 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 3033.40 | 3051.57 | 3052.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 3007.00 | 3042.65 | 3048.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 2966.60 | 2964.90 | 2989.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 14:15:00 | 2946.40 | 2963.60 | 2983.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:15:00 | 2947.70 | 2962.92 | 2979.34 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2955.00 | 2961.34 | 2977.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 2975.70 | 2969.62 | 2974.76 | SL hit (close>ema400) qty=1.00 sl=2974.76 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 2975.70 | 2969.62 | 2974.76 | SL hit (close>ema400) qty=1.00 sl=2974.76 alert=retest1 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 2926.30 | 2969.62 | 2974.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:30:00 | 2935.10 | 2961.25 | 2965.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 2931.80 | 2961.25 | 2965.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:15:00 | 2940.00 | 2957.98 | 2963.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 2922.00 | 2925.05 | 2937.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 2860.40 | 2925.05 | 2937.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 2916.50 | 2923.06 | 2927.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:00:00 | 2913.00 | 2921.05 | 2926.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:00:00 | 2912.10 | 2923.78 | 2926.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 2923.30 | 2923.69 | 2926.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 2930.30 | 2923.69 | 2926.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2945.80 | 2928.11 | 2928.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2945.80 | 2928.11 | 2928.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2945.80 | 2928.11 | 2928.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2945.80 | 2928.11 | 2928.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2945.80 | 2928.11 | 2928.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2945.80 | 2928.11 | 2928.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2945.80 | 2928.11 | 2928.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 2945.80 | 2928.11 | 2928.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 2945.80 | 2928.11 | 2928.08 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 2923.20 | 2928.34 | 2928.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 2859.50 | 2911.90 | 2920.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 2857.60 | 2856.05 | 2881.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:45:00 | 2860.70 | 2856.05 | 2881.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 2880.00 | 2863.07 | 2873.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 2851.10 | 2863.07 | 2873.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 2893.00 | 2855.05 | 2850.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 2893.00 | 2855.05 | 2850.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 2906.40 | 2872.50 | 2860.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 2869.90 | 2884.23 | 2869.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 2869.90 | 2884.23 | 2869.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2869.90 | 2884.23 | 2869.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2868.00 | 2884.23 | 2869.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 2877.70 | 2882.92 | 2870.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 2871.70 | 2882.92 | 2870.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2897.50 | 2886.61 | 2875.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:15:00 | 2853.00 | 2886.61 | 2875.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 2825.00 | 2874.29 | 2870.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 2825.00 | 2874.29 | 2870.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 2822.20 | 2863.87 | 2866.42 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2945.00 | 2862.41 | 2862.08 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 2854.50 | 2912.55 | 2917.13 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 2939.10 | 2900.74 | 2899.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 2949.90 | 2918.07 | 2907.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 11:15:00 | 2938.00 | 2946.63 | 2930.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 11:15:00 | 2938.00 | 2946.63 | 2930.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 2938.00 | 2946.63 | 2930.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 2939.40 | 2946.63 | 2930.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 2937.80 | 2944.86 | 2931.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:45:00 | 2936.20 | 2944.86 | 2931.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 2935.80 | 2943.05 | 2931.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 2935.80 | 2943.05 | 2931.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 2931.70 | 2940.78 | 2931.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 2931.70 | 2940.78 | 2931.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 2925.00 | 2937.62 | 2931.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 2903.70 | 2937.62 | 2931.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2896.40 | 2929.38 | 2928.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 2896.40 | 2929.38 | 2928.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 2892.90 | 2922.08 | 2924.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 2824.00 | 2897.35 | 2912.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 2857.50 | 2830.18 | 2863.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 13:00:00 | 2857.50 | 2830.18 | 2863.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 2877.90 | 2839.72 | 2864.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:30:00 | 2882.00 | 2839.72 | 2864.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 2890.00 | 2849.78 | 2867.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 2864.50 | 2855.82 | 2868.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:00:00 | 2867.70 | 2858.20 | 2868.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 12:15:00 | 2908.00 | 2872.56 | 2872.60 | SL hit (close>static) qty=1.00 sl=2895.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 12:15:00 | 2908.00 | 2872.56 | 2872.60 | SL hit (close>static) qty=1.00 sl=2895.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 2892.90 | 2876.62 | 2874.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 14:15:00 | 2925.90 | 2886.48 | 2879.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 3004.90 | 3059.93 | 3035.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 3004.90 | 3059.93 | 3035.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 3004.90 | 3059.93 | 3035.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 3004.90 | 3059.93 | 3035.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 3023.50 | 3052.64 | 3034.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 3027.80 | 3037.09 | 3029.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:30:00 | 3026.20 | 3042.27 | 3032.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 3100.60 | 3151.23 | 3156.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 3100.60 | 3151.23 | 3156.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 3100.60 | 3151.23 | 3156.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 3054.50 | 3108.94 | 3130.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 3100.90 | 3080.54 | 3105.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 3100.90 | 3080.54 | 3105.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 3100.90 | 3080.54 | 3105.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 3100.90 | 3080.54 | 3105.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3060.40 | 3033.38 | 3058.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 3067.90 | 3033.38 | 3058.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 3052.10 | 3037.12 | 3058.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 3057.30 | 3037.12 | 3058.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 3059.20 | 3041.54 | 3058.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 3059.20 | 3041.54 | 3058.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 3045.90 | 3042.41 | 3057.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 3045.80 | 3042.53 | 3055.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 3070.80 | 3048.18 | 3057.33 | SL hit (close>static) qty=1.00 sl=3062.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 3137.40 | 3066.80 | 3064.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 10:15:00 | 3154.50 | 3084.34 | 3072.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 3051.30 | 3121.24 | 3102.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 3051.30 | 3121.24 | 3102.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 3051.30 | 3121.24 | 3102.39 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 3070.10 | 3089.82 | 3091.32 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 3199.00 | 3111.07 | 3099.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 3208.20 | 3161.05 | 3133.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 3185.00 | 3185.50 | 3160.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:15:00 | 3118.10 | 3185.50 | 3160.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 3117.30 | 3171.86 | 3156.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:45:00 | 3117.10 | 3171.86 | 3156.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 3112.50 | 3159.99 | 3152.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:45:00 | 3108.80 | 3159.99 | 3152.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 3097.80 | 3147.55 | 3147.72 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 15:15:00 | 3165.60 | 3149.13 | 3147.73 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 3137.20 | 3146.74 | 3146.77 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 11:15:00 | 3202.80 | 3157.34 | 3151.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 14:15:00 | 3220.60 | 3183.97 | 3166.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 10:15:00 | 3222.10 | 3247.99 | 3223.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 10:15:00 | 3222.10 | 3247.99 | 3223.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3222.10 | 3247.99 | 3223.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:30:00 | 3188.50 | 3247.99 | 3223.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 3237.90 | 3245.97 | 3224.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:45:00 | 3260.20 | 3246.96 | 3228.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:00:00 | 3263.10 | 3245.19 | 3232.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 15:15:00 | 3210.00 | 3267.62 | 3265.01 | SL hit (close<static) qty=1.00 sl=3212.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-19 15:15:00 | 3210.00 | 3267.62 | 3265.01 | SL hit (close<static) qty=1.00 sl=3212.50 alert=retest2 |

### Cycle 80 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 3236.00 | 3261.30 | 3262.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 3069.10 | 3200.87 | 3229.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 3152.50 | 3124.74 | 3160.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 3152.50 | 3124.74 | 3160.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 3202.90 | 3140.37 | 3163.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 3202.90 | 3140.37 | 3163.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 3235.40 | 3159.37 | 3170.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 3232.80 | 3159.37 | 3170.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 3275.80 | 3191.20 | 3183.06 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 3158.10 | 3201.45 | 3205.23 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 3223.70 | 3208.41 | 3207.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 14:15:00 | 3250.80 | 3216.89 | 3211.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 3308.80 | 3313.92 | 3277.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 3308.80 | 3313.92 | 3277.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3308.80 | 3313.92 | 3277.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 3222.00 | 3313.92 | 3277.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 3210.00 | 3293.14 | 3271.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:00:00 | 3210.00 | 3293.14 | 3271.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 3255.00 | 3285.51 | 3270.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 3273.30 | 3284.43 | 3271.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 3258.80 | 3279.88 | 3274.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 3259.00 | 3273.42 | 3274.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 3259.00 | 3273.42 | 3274.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 3259.00 | 3273.42 | 3274.00 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 14:15:00 | 3290.00 | 3263.72 | 3260.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 3404.70 | 3294.68 | 3275.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 15:15:00 | 4080.10 | 4096.26 | 3988.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 10:00:00 | 4094.10 | 4095.83 | 3998.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 4284.80 | 4150.22 | 4073.06 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 11:15:00 | 4060.00 | 4111.07 | 4114.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 15:15:00 | 4049.80 | 4079.68 | 4097.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 4083.20 | 4080.38 | 4095.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 10:00:00 | 4083.20 | 4080.38 | 4095.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 4126.10 | 4089.53 | 4098.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:30:00 | 4130.60 | 4089.53 | 4098.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 4083.10 | 4088.24 | 4097.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:15:00 | 4066.80 | 4088.24 | 4097.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 10:00:00 | 4081.90 | 4036.14 | 4054.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 4081.60 | 4047.05 | 4057.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 4152.40 | 4075.95 | 4069.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 4152.40 | 4075.95 | 4069.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 4152.40 | 4075.95 | 4069.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 4152.40 | 4075.95 | 4069.66 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 4029.30 | 4072.35 | 4077.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 14:15:00 | 4023.60 | 4062.60 | 4072.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 15:15:00 | 4004.90 | 4001.39 | 4028.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:00:00 | 3962.30 | 3993.57 | 4022.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 4020.20 | 3992.61 | 4016.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-30 11:15:00 | 4020.20 | 3992.61 | 4016.49 | SL hit (close>ema400) qty=1.00 sl=4016.49 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 4020.20 | 3992.61 | 4016.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 4050.10 | 4004.11 | 4019.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:00:00 | 4050.10 | 4004.11 | 4019.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 4100.00 | 4036.87 | 4032.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 11:15:00 | 4136.00 | 4070.83 | 4050.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 4142.40 | 4146.92 | 4110.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 13:00:00 | 4142.40 | 4146.92 | 4110.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 4117.40 | 4141.01 | 4111.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:30:00 | 4121.50 | 4141.01 | 4111.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 4112.80 | 4135.37 | 4111.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:45:00 | 4109.80 | 4135.37 | 4111.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 4125.00 | 4133.30 | 4112.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 4106.00 | 4133.30 | 4112.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 4108.20 | 4128.28 | 4112.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 4108.20 | 4128.28 | 4112.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 4121.40 | 4126.90 | 4113.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:15:00 | 4096.80 | 4126.90 | 4113.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 4089.90 | 4119.50 | 4110.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:30:00 | 4098.40 | 4119.50 | 4110.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 4089.80 | 4113.56 | 4109.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:30:00 | 4088.10 | 4113.56 | 4109.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 13:15:00 | 4068.00 | 4104.45 | 4105.28 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 4277.70 | 4129.67 | 4115.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 4530.00 | 4262.62 | 4195.58 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-29 09:15:00 | 3440.90 | 2025-06-03 09:15:00 | 3449.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-06-13 10:15:00 | 3573.90 | 2025-06-13 13:15:00 | 3528.40 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-06-17 12:30:00 | 3491.50 | 2025-06-19 09:15:00 | 3592.80 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-06-18 09:45:00 | 3505.00 | 2025-06-19 09:15:00 | 3592.80 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-06-18 10:15:00 | 3505.20 | 2025-06-19 09:15:00 | 3592.80 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-06-18 10:45:00 | 3504.70 | 2025-06-19 09:15:00 | 3592.80 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-06-24 09:15:00 | 3456.50 | 2025-06-26 13:15:00 | 3462.20 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-06-25 09:30:00 | 3457.30 | 2025-06-26 13:15:00 | 3462.20 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-06-26 10:00:00 | 3462.10 | 2025-06-26 13:15:00 | 3462.20 | STOP_HIT | 1.00 | -0.00% |
| SELL | retest2 | 2025-06-26 12:00:00 | 3464.90 | 2025-06-26 13:15:00 | 3462.20 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-06-30 13:30:00 | 3403.70 | 2025-07-02 15:15:00 | 3414.70 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-07-01 10:00:00 | 3401.40 | 2025-07-02 15:15:00 | 3414.70 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-07-02 12:15:00 | 3404.30 | 2025-07-02 15:15:00 | 3414.70 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-07-02 13:15:00 | 3404.60 | 2025-07-02 15:15:00 | 3414.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-09 11:00:00 | 3435.80 | 2025-07-11 11:15:00 | 3462.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-09 12:15:00 | 3435.00 | 2025-07-11 11:15:00 | 3462.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-07-10 09:30:00 | 3434.00 | 2025-07-11 11:15:00 | 3462.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-15 10:00:00 | 3470.70 | 2025-07-17 09:15:00 | 3817.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-15 10:30:00 | 3471.50 | 2025-07-17 09:15:00 | 3818.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-15 11:15:00 | 3469.80 | 2025-07-17 09:15:00 | 3816.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-15 14:15:00 | 3471.30 | 2025-07-17 09:15:00 | 3818.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-24 09:15:00 | 3889.00 | 2025-07-24 10:15:00 | 3872.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-29 12:15:00 | 3763.00 | 2025-07-29 15:15:00 | 3794.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-31 11:30:00 | 3821.00 | 2025-08-01 15:15:00 | 3767.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-08-01 12:45:00 | 3836.00 | 2025-08-01 15:15:00 | 3767.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-08-22 12:00:00 | 3225.10 | 2025-08-22 13:15:00 | 3271.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-08-28 09:15:00 | 3359.80 | 2025-08-28 12:15:00 | 3232.40 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2025-09-08 09:30:00 | 3320.80 | 2025-09-12 09:15:00 | 3313.20 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-09-10 09:15:00 | 3437.00 | 2025-09-12 09:15:00 | 3313.20 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-09-16 13:00:00 | 3306.60 | 2025-09-17 10:15:00 | 3345.70 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-01 10:30:00 | 3163.00 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-01 13:45:00 | 3159.90 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-10-01 15:00:00 | 3163.00 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-03 10:45:00 | 3151.90 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-06 09:15:00 | 3170.00 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-10-06 14:30:00 | 3172.20 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-10-06 15:15:00 | 3162.50 | 2025-10-07 14:15:00 | 3192.30 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-10-24 09:15:00 | 3212.00 | 2025-10-29 11:15:00 | 3253.30 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2025-11-04 11:15:00 | 3250.10 | 2025-11-04 11:15:00 | 3258.20 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-11-06 14:30:00 | 3248.70 | 2025-11-07 09:15:00 | 3178.70 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-11-12 09:15:00 | 3055.70 | 2025-11-19 09:15:00 | 2902.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 09:15:00 | 3055.70 | 2025-11-20 10:15:00 | 2975.00 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-12-10 10:45:00 | 2815.00 | 2025-12-12 09:15:00 | 2916.00 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-12-30 11:45:00 | 3010.50 | 2026-01-06 12:15:00 | 3014.10 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-12-31 12:00:00 | 3006.60 | 2026-01-06 12:15:00 | 3014.10 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2026-01-01 09:30:00 | 3011.80 | 2026-01-06 12:15:00 | 3014.10 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest1 | 2026-01-13 14:15:00 | 2946.40 | 2026-01-14 15:15:00 | 2975.70 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest1 | 2026-01-14 09:15:00 | 2947.70 | 2026-01-14 15:15:00 | 2975.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-16 09:15:00 | 2926.30 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-01-19 09:30:00 | 2935.10 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-01-19 10:15:00 | 2931.80 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-01-19 11:15:00 | 2940.00 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-01-21 09:15:00 | 2860.40 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2026-01-22 10:15:00 | 2916.50 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-22 11:00:00 | 2913.00 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-01-22 15:00:00 | 2912.10 | 2026-01-23 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-29 09:15:00 | 2851.10 | 2026-01-30 13:15:00 | 2893.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-16 09:15:00 | 2864.50 | 2026-02-16 12:15:00 | 2908.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-16 10:00:00 | 2867.70 | 2026-02-16 12:15:00 | 2908.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-02-20 10:30:00 | 3027.80 | 2026-02-27 10:15:00 | 3100.60 | STOP_HIT | 1.00 | 2.40% |
| BUY | retest2 | 2026-02-20 11:30:00 | 3026.20 | 2026-02-27 10:15:00 | 3100.60 | STOP_HIT | 1.00 | 2.46% |
| SELL | retest2 | 2026-03-05 13:45:00 | 3045.80 | 2026-03-05 14:15:00 | 3070.80 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-03-17 13:45:00 | 3260.20 | 2026-03-19 15:15:00 | 3210.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-18 10:00:00 | 3263.10 | 2026-03-19 15:15:00 | 3210.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-04-02 12:30:00 | 3273.30 | 2026-04-07 09:15:00 | 3259.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-04-06 10:30:00 | 3258.80 | 2026-04-07 09:15:00 | 3259.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2026-04-23 12:15:00 | 4066.80 | 2026-04-27 12:15:00 | 4152.40 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-04-27 10:00:00 | 4081.90 | 2026-04-27 12:15:00 | 4152.40 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-04-27 11:15:00 | 4081.60 | 2026-04-27 12:15:00 | 4152.40 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest1 | 2026-04-30 10:00:00 | 3962.30 | 2026-04-30 11:15:00 | 4020.20 | STOP_HIT | 1.00 | -1.46% |
