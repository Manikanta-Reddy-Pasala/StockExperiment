# Escorts Kubota Ltd. (ESCORTS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3250.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 138.90
- **Avg P&L per closed trade:** 23.15

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 3694.90 | 3935.83 | 3935.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 3655.45 | 3933.04 | 3934.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 13:15:00 | 3899.95 | 3869.21 | 3897.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-28 15:15:00 | 3830.85 | 3869.98 | 3895.27 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-16 11:15:00 | 3855.00 | 3812.96 | 3852.05 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 13:15:00 | 4336.75 | 3880.62 | 3879.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 4352.35 | 3885.32 | 3882.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 4019.50 | 4023.57 | 3961.77 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 3703.90 | 3929.32 | 3929.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 09:15:00 | 3498.25 | 3912.23 | 3920.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 3403.25 | 3390.59 | 3516.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-03 13:15:00 | 3364.90 | 3390.18 | 3514.26 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 3472.00 | 3380.57 | 3480.01 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-15 10:15:00 | 3482.40 | 3381.59 | 3480.02 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 13:15:00 | 3345.00 | 3222.27 | 3222.09 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 15:15:00 | 3131.00 | 3225.57 | 3225.91 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3358.80 | 3226.89 | 3226.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 3388.30 | 3232.14 | 3229.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 3378.20 | 3389.83 | 3326.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 12:15:00 | 3415.70 | 3381.30 | 3327.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-03 11:15:00 | 3318.10 | 3379.95 | 3328.48 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 3135.80 | 3296.50 | 3296.71 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 3363.30 | 3295.45 | 3295.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 11:15:00 | 3383.90 | 3314.47 | 3306.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 3312.50 | 3322.60 | 3311.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 09:15:00 | 3329.80 | 3318.11 | 3309.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3329.80 | 3318.11 | 3309.63 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-15 10:15:00 | 3355.70 | 3318.49 | 3309.86 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 3352.60 | 3381.90 | 3350.79 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-01 10:15:00 | 3331.40 | 3381.07 | 3350.83 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 3452.20 | 3680.58 | 3681.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 3422.60 | 3674.03 | 3678.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 3600.10 | 3598.29 | 3635.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-13 14:15:00 | 3583.70 | 3644.17 | 3652.30 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-25 10:15:00 | 3621.00 | 3590.07 | 3619.84 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-28 15:15:00 | 3830.85 | 2024-09-09 09:15:00 | 3637.58 | TARGET | 193.27 |
| SELL | 2025-01-03 13:15:00 | 3364.90 | 2025-01-15 10:15:00 | 3482.40 | EXIT_EMA400 | -117.50 |
| BUY | 2025-06-02 12:15:00 | 3415.70 | 2025-06-03 11:15:00 | 3318.10 | EXIT_EMA400 | -97.60 |
| BUY | 2025-07-15 09:15:00 | 3329.80 | 2025-07-16 12:15:00 | 3390.31 | TARGET | 60.51 |
| BUY | 2025-07-15 10:15:00 | 3355.70 | 2025-07-24 09:15:00 | 3493.22 | TARGET | 137.52 |
| SELL | 2026-02-13 14:15:00 | 3583.70 | 2026-02-25 10:15:00 | 3621.00 | EXIT_EMA400 | -37.30 |
