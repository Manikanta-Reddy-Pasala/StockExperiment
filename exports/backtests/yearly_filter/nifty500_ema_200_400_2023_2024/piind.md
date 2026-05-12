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
| ALERT2_SKIP | 0 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 24 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 21
- **Target hits / Stop hits / Partials:** 5 / 25 / 6
- **Avg / median % per leg:** 0.85% / -1.25%
- **Sum % (uncompounded):** 30.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 4 | 26.7% | 4 | 11 | 0 | 1.06% | 15.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 4 | 26.7% | 4 | 11 | 0 | 1.06% | 15.9% |
| SELL (all) | 21 | 11 | 52.4% | 1 | 14 | 6 | 0.70% | 14.8% |
| SELL @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 0 | 6 | 4 | 1.88% | 18.8% |
| SELL @ 3rd Alert (retest2) | 11 | 3 | 27.3% | 1 | 8 | 2 | -0.37% | -4.1% |
| retest1 (combined) | 10 | 8 | 80.0% | 0 | 6 | 4 | 1.88% | 18.8% |
| retest2 (combined) | 26 | 7 | 26.9% | 5 | 19 | 2 | 0.46% | 11.9% |

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
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-11 13:15:00 | 3500.00 | 3515.29 | 3561.29 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 12:30:00 | 3502.75 | 3510.84 | 3554.35 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 14:15:00 | 3504.00 | 3510.81 | 3554.12 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 15:00:00 | 3500.05 | 3510.70 | 3553.85 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 15:15:00 | 3325.00 | 3492.02 | 3535.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 15:15:00 | 3327.61 | 3492.02 | 3535.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 15:15:00 | 3328.80 | 3492.02 | 3535.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 15:15:00 | 3325.05 | 3492.02 | 3535.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-11-03 09:15:00 | 3452.90 | 3445.97 | 3501.34 | SL hit (close>ema200) qty=0.50 sl=3445.97 alert=retest1 |

### Cycle 2 — BUY (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 11:15:00 | 3739.25 | 3535.99 | 3535.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 11:15:00 | 3760.95 | 3574.16 | 3556.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 13:15:00 | 3685.15 | 3715.44 | 3646.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-12 14:00:00 | 3685.15 | 3715.44 | 3646.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 3457.35 | 3712.88 | 3645.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 15:00:00 | 3457.35 | 3712.88 | 3645.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 15:15:00 | 3485.00 | 3710.61 | 3645.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 09:30:00 | 3465.80 | 3707.43 | 3643.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-12-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 14:15:00 | 3425.80 | 3593.16 | 3593.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 3405.05 | 3521.76 | 3550.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 15:15:00 | 3412.00 | 3411.63 | 3470.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-02 09:15:00 | 3442.90 | 3411.63 | 3470.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 3449.45 | 3394.25 | 3453.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 3449.45 | 3394.25 | 3453.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 3448.65 | 3394.80 | 3453.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:45:00 | 3441.15 | 3394.80 | 3453.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 12:15:00 | 3451.45 | 3395.36 | 3453.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 13:00:00 | 3451.45 | 3395.36 | 3453.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 3448.75 | 3395.89 | 3453.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 14:15:00 | 3459.45 | 3395.89 | 3453.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 3454.20 | 3396.47 | 3453.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 14:45:00 | 3447.00 | 3396.47 | 3453.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 3447.25 | 3396.98 | 3453.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:15:00 | 3475.15 | 3396.98 | 3453.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 3470.80 | 3397.71 | 3453.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:15:00 | 3471.70 | 3397.71 | 3453.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 3436.00 | 3398.09 | 3453.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 09:45:00 | 3402.25 | 3401.45 | 3453.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 09:30:00 | 3421.50 | 3401.06 | 3451.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 11:30:00 | 3397.45 | 3401.30 | 3450.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 10:00:00 | 3415.00 | 3401.74 | 3449.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 3402.00 | 3402.02 | 3449.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-15 09:15:00 | 3541.85 | 3404.01 | 3449.30 | SL hit (close>static) qty=1.00 sl=3504.15 alert=retest2 |

### Cycle 4 — BUY (started 2024-02-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 14:15:00 | 3679.40 | 3487.33 | 3486.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 11:15:00 | 3687.60 | 3494.85 | 3490.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 3566.30 | 3586.47 | 3548.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-13 12:00:00 | 3566.30 | 3586.47 | 3548.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 3581.00 | 3586.12 | 3548.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 14:30:00 | 3561.60 | 3586.12 | 3548.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 3604.45 | 3586.19 | 3549.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 10:30:00 | 3615.90 | 3586.58 | 3549.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 11:30:00 | 3628.60 | 3586.91 | 3549.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 13:00:00 | 3612.95 | 3587.17 | 3550.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 14:30:00 | 3614.90 | 3587.97 | 3550.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-08 14:15:00 | 3974.25 | 3734.56 | 3653.50 | Target hit (10%) qty=1.00 alert=retest2 |

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
| SELL | retest1 | 2023-10-11 13:15:00 | 3500.00 | 2023-10-25 15:15:00 | 3325.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2023-10-16 12:30:00 | 3502.75 | 2023-10-25 15:15:00 | 3327.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2023-10-16 14:15:00 | 3504.00 | 2023-10-25 15:15:00 | 3328.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2023-10-16 15:00:00 | 3500.05 | 2023-10-25 15:15:00 | 3325.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2023-10-11 13:15:00 | 3500.00 | 2023-11-03 09:15:00 | 3452.90 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2023-10-16 12:30:00 | 3502.75 | 2023-11-03 09:15:00 | 3452.90 | STOP_HIT | 0.50 | 1.42% |
| SELL | retest1 | 2023-10-16 14:15:00 | 3504.00 | 2023-11-03 09:15:00 | 3452.90 | STOP_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2023-10-16 15:00:00 | 3500.05 | 2023-11-03 09:15:00 | 3452.90 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2024-02-12 09:45:00 | 3402.25 | 2024-02-15 09:15:00 | 3541.85 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2024-02-13 09:30:00 | 3421.50 | 2024-02-15 09:15:00 | 3541.85 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2024-02-13 11:30:00 | 3397.45 | 2024-02-15 09:15:00 | 3541.85 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2024-02-14 10:00:00 | 3415.00 | 2024-02-15 09:15:00 | 3541.85 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2024-03-14 10:30:00 | 3615.90 | 2024-04-08 14:15:00 | 3974.25 | TARGET_HIT | 1.00 | 9.91% |
| BUY | retest2 | 2024-03-14 11:30:00 | 3628.60 | 2024-04-09 09:15:00 | 3977.49 | TARGET_HIT | 1.00 | 9.62% |
| BUY | retest2 | 2024-03-14 13:00:00 | 3612.95 | 2024-04-09 09:15:00 | 3991.46 | TARGET_HIT | 1.00 | 10.48% |
| BUY | retest2 | 2024-03-14 14:30:00 | 3614.90 | 2024-04-09 09:15:00 | 3976.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-19 12:00:00 | 3722.05 | 2024-04-30 10:15:00 | 3626.55 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-04-22 09:15:00 | 3729.90 | 2024-04-30 10:15:00 | 3626.55 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-04-24 10:30:00 | 3720.00 | 2024-04-30 10:15:00 | 3626.55 | STOP_HIT | 1.00 | -2.51% |
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
