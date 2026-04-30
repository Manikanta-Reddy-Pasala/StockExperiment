# Larsen & Toubro Ltd. (LT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 4014.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -162.16
- **Avg P&L per closed trade:** -32.43

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 11:15:00 | 3292.55 | 3542.51 | 3543.12 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 3647.55 | 3533.29 | 3533.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 09:15:00 | 3705.20 | 3543.38 | 3538.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3526.00 | 3576.53 | 3555.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-29 10:15:00 | 3773.50 | 3611.03 | 3592.97 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-05 09:15:00 | 3577.95 | 3653.15 | 3618.92 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 3483.10 | 3629.81 | 3630.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 3478.70 | 3622.31 | 3626.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 3601.00 | 3593.35 | 3609.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-22 09:15:00 | 3551.75 | 3592.86 | 3608.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-31 09:15:00 | 3627.15 | 3522.22 | 3565.96 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 3706.60 | 3586.11 | 3585.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 3738.25 | 3593.78 | 3589.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 3722.05 | 3745.13 | 3684.15 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 3520.00 | 3655.63 | 3655.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 3475.05 | 3653.84 | 3655.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 3561.95 | 3553.29 | 3594.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 3313.75 | 3551.58 | 3591.84 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 3379.00 | 3279.06 | 3362.71 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 3593.70 | 3351.79 | 3351.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 3640.00 | 3374.03 | 3362.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 3553.50 | 3578.56 | 3503.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 12:15:00 | 3567.30 | 3578.12 | 3504.53 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-11 10:15:00 | 3555.50 | 3605.05 | 3560.79 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 3422.00 | 3532.71 | 3533.04 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 3678.80 | 3533.33 | 3533.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 3715.00 | 3578.11 | 3558.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 3589.00 | 3602.03 | 3575.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-21 09:15:00 | 3633.50 | 3602.19 | 3576.19 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-26 09:15:00 | 3573.60 | 3602.61 | 3579.04 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 3770.80 | 3951.04 | 3951.84 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 09:15:00 | 4045.30 | 3950.38 | 3949.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 4107.70 | 3965.56 | 3957.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4057.70 | 4162.83 | 4081.72 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3485.60 | 4023.65 | 4024.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 3474.90 | 4018.19 | 4021.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 3732.90 | 3727.26 | 3839.68 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 4079.50 | 3903.18 | 3903.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 4099.40 | 3919.20 | 3911.31 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-07-29 10:15:00 | 3773.50 | 2024-08-05 09:15:00 | 3577.95 | EXIT_EMA400 | -195.55 |
| SELL | 2024-10-22 09:15:00 | 3551.75 | 2024-10-25 09:15:00 | 3381.41 | TARGET | 170.34 |
| SELL | 2025-02-03 09:15:00 | 3313.75 | 2025-03-21 09:15:00 | 3379.00 | EXIT_EMA400 | -65.25 |
| BUY | 2025-06-13 12:15:00 | 3567.30 | 2025-07-11 10:15:00 | 3555.50 | EXIT_EMA400 | -11.80 |
| BUY | 2025-08-21 09:15:00 | 3633.50 | 2025-08-26 09:15:00 | 3573.60 | EXIT_EMA400 | -59.90 |
