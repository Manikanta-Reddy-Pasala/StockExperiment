# PI Industries Ltd. (PIIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3103.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 19 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 20
- **Target hits / Stop hits / Partials:** 1 / 20 / 2
- **Avg / median % per leg:** -1.36% / -1.84%
- **Sum % (uncompounded):** -31.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.58% | -36.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.58% | -36.1% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.53% | 4.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.38% | -6.8% |
| SELL @ 3rd Alert (retest2) | 7 | 3 | 42.9% | 1 | 4 | 2 | 1.65% | 11.5% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.38% | -6.8% |
| retest2 (combined) | 21 | 3 | 14.3% | 1 | 18 | 2 | -1.17% | -24.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 14:15:00 | 3416.60 | 3630.66 | 3631.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 15:15:00 | 3412.00 | 3628.48 | 3630.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 11:15:00 | 3517.00 | 3515.39 | 3561.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 14:15:00 | 3488.90 | 3447.32 | 3500.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 3488.90 | 3447.32 | 3500.65 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 11:15:00 | 3739.25 | 3535.99 | 3535.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 11:15:00 | 3760.95 | 3574.16 | 3556.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 13:15:00 | 3685.15 | 3715.44 | 3646.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 14:15:00 | 3457.35 | 3712.88 | 3645.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 3457.35 | 3712.88 | 3645.85 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-12-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 14:15:00 | 3425.80 | 3593.16 | 3593.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 3405.05 | 3521.76 | 3550.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 15:15:00 | 3412.00 | 3411.63 | 3470.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 10:15:00 | 3449.45 | 3394.25 | 3453.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 3449.45 | 3394.25 | 3453.17 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-02-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 14:15:00 | 3679.40 | 3487.33 | 3486.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 11:15:00 | 3687.60 | 3494.85 | 3490.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 3566.30 | 3586.47 | 3548.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 14:15:00 | 3581.00 | 3586.12 | 3548.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 3581.00 | 3586.12 | 3548.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 3907.45 | 3757.26 | 3671.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 12:15:00 | 3606.00 | 3737.16 | 3693.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 15:15:00 | 3611.65 | 3732.83 | 3691.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 11:30:00 | 3611.15 | 3727.92 | 3690.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 10:15:00 | 3522.50 | 3718.92 | 3686.66 | SL hit (close<static) qty=1.00 sl=3545.15 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 11:15:00 | 3539.85 | 3660.55 | 3660.56 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 12:15:00 | 3758.80 | 3649.37 | 3649.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 13:15:00 | 3769.00 | 3650.56 | 3649.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 4564.10 | 4585.85 | 4425.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-07 10:00:00 | 4564.10 | 4585.85 | 4425.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 4461.00 | 4566.88 | 4456.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:30:00 | 4509.05 | 4563.46 | 4457.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 13:00:00 | 4489.95 | 4562.87 | 4461.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 13:15:00 | 4407.55 | 4561.33 | 4460.92 | SL hit (close<static) qty=1.00 sl=4447.55 alert=retest2 |

### Cycle 7 — SELL (started 2024-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 15:15:00 | 4145.00 | 4429.33 | 4429.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 10:15:00 | 4095.70 | 4423.17 | 4426.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 09:15:00 | 3689.35 | 3593.04 | 3782.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 10:00:00 | 3689.35 | 3593.04 | 3782.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 3418.40 | 3281.24 | 3440.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 3429.85 | 3281.24 | 3440.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 3392.35 | 3289.19 | 3438.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 10:15:00 | 3387.05 | 3289.19 | 3438.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 13:45:00 | 3385.50 | 3293.72 | 3438.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 10:15:00 | 3453.00 | 3306.40 | 3436.91 | SL hit (close>static) qty=1.00 sl=3440.45 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 3636.00 | 3472.85 | 3472.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 14:15:00 | 3659.20 | 3474.70 | 3473.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 3620.00 | 3622.94 | 3569.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:45:00 | 3613.60 | 3622.94 | 3569.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4045.00 | 4116.97 | 4024.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:45:00 | 4051.70 | 4116.97 | 4024.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 4020.00 | 4115.15 | 4024.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:30:00 | 4014.10 | 4115.15 | 4024.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 3973.70 | 4113.74 | 4024.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 3973.70 | 4113.74 | 4024.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 4028.40 | 4110.70 | 4024.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 3995.00 | 4110.70 | 4024.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 3970.60 | 4109.31 | 4023.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 3970.60 | 4109.31 | 4023.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 3965.10 | 4107.87 | 4023.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 3965.10 | 4107.87 | 4023.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 3865.30 | 3963.08 | 3963.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 3826.90 | 3956.79 | 3960.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 3612.40 | 3610.69 | 3696.05 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 14:30:00 | 3559.30 | 3610.08 | 3690.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 12:45:00 | 3560.70 | 3606.88 | 3681.82 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3680.40 | 3606.59 | 3680.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 3680.40 | 3606.59 | 3680.19 | SL hit (close>ema400) qty=1.00 sl=3680.19 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 3907.45 | 2024-05-07 10:15:00 | 3522.50 | STOP_HIT | 1.00 | -9.85% |
| BUY | retest2 | 2024-05-03 12:15:00 | 3606.00 | 2024-05-07 10:15:00 | 3522.50 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-05-03 15:15:00 | 3611.65 | 2024-05-07 10:15:00 | 3522.50 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-05-06 11:30:00 | 3611.15 | 2024-05-07 10:15:00 | 3522.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-05-15 13:45:00 | 3658.00 | 2024-05-16 09:15:00 | 3608.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-05-15 14:30:00 | 3661.25 | 2024-05-16 09:15:00 | 3608.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-10-18 11:30:00 | 4509.05 | 2024-10-21 13:15:00 | 4407.55 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-10-21 13:00:00 | 4489.95 | 2024-10-21 13:15:00 | 4407.55 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-10-30 10:00:00 | 4483.55 | 2024-11-13 09:15:00 | 4436.55 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-10-31 09:30:00 | 4489.75 | 2024-11-13 09:15:00 | 4436.55 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-11-05 13:30:00 | 4502.25 | 2024-11-13 09:15:00 | 4436.55 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-11-05 14:00:00 | 4514.50 | 2024-11-13 09:15:00 | 4436.55 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-11-11 09:45:00 | 4492.85 | 2024-11-13 09:15:00 | 4436.55 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-11-13 15:15:00 | 4499.15 | 2024-11-14 09:15:00 | 4253.00 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest2 | 2025-03-19 10:15:00 | 3387.05 | 2025-03-21 10:15:00 | 3453.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-03-19 13:45:00 | 3385.50 | 2025-03-21 10:15:00 | 3453.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-04-04 09:45:00 | 3369.70 | 2025-04-07 09:15:00 | 3201.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:45:00 | 3369.70 | 2025-04-11 09:15:00 | 3478.00 | STOP_HIT | 0.50 | -3.21% |
| SELL | retest1 | 2025-10-28 14:30:00 | 3559.30 | 2025-11-03 09:15:00 | 3680.40 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest1 | 2025-10-31 12:45:00 | 3560.70 | 2025-11-03 09:15:00 | 3680.40 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-11-04 12:30:00 | 3636.90 | 2025-11-04 14:15:00 | 3684.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-11-12 11:45:00 | 3631.90 | 2025-11-19 09:15:00 | 3450.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 11:45:00 | 3631.90 | 2025-12-09 09:15:00 | 3268.71 | TARGET_HIT | 0.50 | 10.00% |
