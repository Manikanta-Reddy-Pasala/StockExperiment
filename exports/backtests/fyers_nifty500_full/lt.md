# Larsen & Toubro Ltd. (LT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4021.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -120.75
- **Avg P&L per closed trade:** -30.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 3488.80 | 3614.95 | 3615.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 12:15:00 | 3476.10 | 3584.98 | 3597.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 09:15:00 | 3627.15 | 3522.41 | 3561.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-19 14:15:00 | 3497.55 | 3560.07 | 3571.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 3560.95 | 3553.68 | 3567.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-22 13:15:00 | 3580.05 | 3553.94 | 3567.70 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 11:15:00 | 3695.05 | 3580.88 | 3580.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 12:15:00 | 3710.00 | 3582.16 | 3581.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 3722.05 | 3745.35 | 3682.83 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 12:15:00 | 3475.45 | 3653.97 | 3654.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 3468.95 | 3652.13 | 3653.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 3561.95 | 3553.70 | 3593.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 12:15:00 | 3414.05 | 3553.73 | 3591.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 3379.00 | 3278.80 | 3361.60 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 3593.10 | 3351.74 | 3350.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 3639.00 | 3373.98 | 3362.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 3553.50 | 3578.39 | 3503.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 12:15:00 | 3568.50 | 3577.96 | 3504.27 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-11 10:15:00 | 3555.50 | 3604.99 | 3560.64 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 3422.00 | 3532.73 | 3532.97 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 3678.50 | 3533.26 | 3533.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 3715.00 | 3578.00 | 3558.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 3589.90 | 3601.96 | 3575.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-21 09:15:00 | 3633.50 | 3602.12 | 3576.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-26 09:15:00 | 3573.20 | 3602.53 | 3578.93 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 3770.80 | 3950.99 | 3951.81 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 09:15:00 | 4044.00 | 3948.93 | 3948.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 4107.80 | 3964.27 | 3956.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4057.70 | 4162.32 | 4081.08 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3485.60 | 4023.27 | 4024.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 3474.90 | 4017.82 | 4021.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 3732.90 | 3727.11 | 3839.35 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 4071.00 | 3903.71 | 3903.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 4082.30 | 3922.45 | 3912.91 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-19 14:15:00 | 3497.55 | 2024-11-22 13:15:00 | 3580.05 | EXIT_EMA400 | -82.50 |
| SELL | 2025-02-01 12:15:00 | 3414.05 | 2025-03-21 09:15:00 | 3379.00 | EXIT_EMA400 | 35.05 |
| BUY | 2025-06-13 12:15:00 | 3568.50 | 2025-07-11 10:15:00 | 3555.50 | EXIT_EMA400 | -13.00 |
| BUY | 2025-08-21 09:15:00 | 3633.50 | 2025-08-26 09:15:00 | 3573.20 | EXIT_EMA400 | -60.30 |
