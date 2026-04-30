# PI Industries Ltd. (PIIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 3054.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 7 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** 71.04
- **Avg P&L per closed trade:** 7.89

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 10:15:00 | 3587.95 | 3678.77 | 3679.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 10:15:00 | 3554.45 | 3670.58 | 3674.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 11:15:00 | 3517.00 | 3515.56 | 3576.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-11 13:15:00 | 3498.95 | 3515.29 | 3575.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 3463.80 | 3447.93 | 3508.65 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-07 09:15:00 | 3507.50 | 3449.52 | 3507.36 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 12:15:00 | 3695.90 | 3548.44 | 3548.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 10:15:00 | 3720.00 | 3571.30 | 3560.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 13:15:00 | 3685.15 | 3715.14 | 3649.99 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2023-12-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 11:15:00 | 3417.00 | 3598.10 | 3598.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 3355.00 | 3494.56 | 3530.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 3432.00 | 3415.75 | 3475.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-02 11:15:00 | 3415.25 | 3415.94 | 3474.54 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 3448.65 | 3397.88 | 3457.17 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-02-09 09:15:00 | 3468.85 | 3400.61 | 3457.09 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 3680.85 | 3490.58 | 3490.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 11:15:00 | 3687.60 | 3496.19 | 3493.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 3565.00 | 3584.60 | 3548.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-14 10:15:00 | 3624.60 | 3584.87 | 3549.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 3692.90 | 3765.01 | 3687.34 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-19 11:15:00 | 3722.20 | 3763.89 | 3687.56 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 3696.60 | 3758.41 | 3689.54 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-23 11:15:00 | 3689.30 | 3757.72 | 3689.53 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 12:15:00 | 3548.75 | 3659.65 | 3660.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 3526.60 | 3639.06 | 3648.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 3633.20 | 3632.83 | 3645.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-10 09:15:00 | 3567.35 | 3635.29 | 3644.57 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-10 13:15:00 | 3653.50 | 3634.62 | 3644.05 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 12:15:00 | 3758.80 | 3649.81 | 3649.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 13:15:00 | 3769.00 | 3650.99 | 3650.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 15:15:00 | 4581.20 | 4585.97 | 4424.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 14:15:00 | 4603.05 | 4579.14 | 4436.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 4462.35 | 4567.25 | 4457.03 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-18 09:15:00 | 4453.65 | 4565.16 | 4457.08 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 09:15:00 | 4140.25 | 4426.83 | 4428.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 10:15:00 | 4095.70 | 4423.53 | 4426.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 09:15:00 | 3690.30 | 3600.43 | 3793.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 11:15:00 | 3495.30 | 3596.69 | 3758.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 3419.75 | 3282.04 | 3443.85 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 10:15:00 | 3453.00 | 3307.08 | 3440.16 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 15:15:00 | 3658.40 | 3476.60 | 3476.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 11:15:00 | 3683.00 | 3517.88 | 3498.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 3620.00 | 3622.89 | 3570.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-22 13:15:00 | 3650.10 | 3623.16 | 3570.64 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4045.00 | 4117.30 | 4024.85 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-07 11:15:00 | 4020.00 | 4115.47 | 4024.85 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 3865.30 | 3963.23 | 3963.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 3826.90 | 3957.00 | 3960.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 3612.40 | 3610.72 | 3696.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-28 11:15:00 | 3575.70 | 3611.17 | 3692.59 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-03 09:15:00 | 3680.40 | 3606.70 | 3680.29 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-11 13:15:00 | 3498.95 | 2023-11-07 09:15:00 | 3507.50 | EXIT_EMA400 | -8.55 |
| SELL | 2024-02-02 11:15:00 | 3415.25 | 2024-02-09 09:15:00 | 3468.85 | EXIT_EMA400 | -53.60 |
| BUY | 2024-03-14 10:15:00 | 3624.60 | 2024-03-27 11:15:00 | 3850.26 | TARGET | 225.66 |
| BUY | 2024-04-19 11:15:00 | 3722.20 | 2024-04-23 11:15:00 | 3689.30 | EXIT_EMA400 | -32.90 |
| SELL | 2024-06-10 09:15:00 | 3567.35 | 2024-06-10 13:15:00 | 3653.50 | EXIT_EMA400 | -86.15 |
| BUY | 2024-10-09 14:15:00 | 4603.05 | 2024-10-18 09:15:00 | 4453.65 | EXIT_EMA400 | -149.40 |
| SELL | 2025-02-10 11:15:00 | 3495.30 | 2025-03-21 10:15:00 | 3453.00 | EXIT_EMA400 | 42.30 |
| BUY | 2025-05-22 13:15:00 | 3650.10 | 2025-05-29 09:15:00 | 3888.48 | TARGET | 238.38 |
| SELL | 2025-10-28 11:15:00 | 3575.70 | 2025-11-03 09:15:00 | 3680.40 | EXIT_EMA400 | -104.70 |
