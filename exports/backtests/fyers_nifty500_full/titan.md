# Titan Company Ltd. (TITAN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4388.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -182.45
- **Avg P&L per closed trade:** -26.06

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 12:15:00 | 3321.80 | 3543.98 | 3544.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 3287.00 | 3539.13 | 3542.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 14:15:00 | 3313.80 | 3297.75 | 3383.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-28 10:15:00 | 3251.80 | 3299.57 | 3374.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 10:15:00 | 3368.10 | 3295.16 | 3362.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 10:15:00 | 3463.55 | 3386.82 | 3386.76 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 3299.75 | 3386.38 | 3386.67 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 11:15:00 | 3478.40 | 3382.92 | 3382.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 12:15:00 | 3575.60 | 3384.84 | 3383.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 3408.15 | 3420.62 | 3403.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-07 09:15:00 | 3438.10 | 3420.57 | 3403.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 3438.10 | 3420.57 | 3403.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-10 09:15:00 | 3377.75 | 3420.91 | 3404.46 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 14:15:00 | 3229.85 | 3390.32 | 3390.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 3207.70 | 3385.38 | 3387.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 3167.65 | 3154.29 | 3234.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-24 09:15:00 | 3092.10 | 3156.34 | 3230.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 3133.45 | 3110.42 | 3182.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-08 14:15:00 | 3120.60 | 3111.76 | 3180.98 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 12:15:00 | 3175.80 | 3113.44 | 3180.12 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-09 13:15:00 | 3183.20 | 3114.13 | 3180.13 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 3379.70 | 3223.88 | 3223.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 11:15:00 | 3402.60 | 3237.64 | 3230.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 3499.00 | 3499.42 | 3425.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-12 12:15:00 | 3512.00 | 3499.55 | 3426.11 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-13 09:15:00 | 3408.90 | 3497.31 | 3426.43 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 3355.10 | 3464.89 | 3465.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 3328.50 | 3460.14 | 3462.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 3449.00 | 3441.68 | 3452.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-06 15:15:00 | 3400.00 | 3441.03 | 3451.89 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-08 09:15:00 | 3469.40 | 3439.12 | 3450.49 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 3559.00 | 3459.85 | 3459.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 3569.70 | 3462.87 | 3460.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 3576.80 | 3579.54 | 3536.54 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 11:15:00 | 3380.00 | 3511.17 | 3511.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 09:15:00 | 3361.00 | 3496.94 | 3504.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 3641.40 | 3506.57 | 3505.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3694.00 | 3511.02 | 3508.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 13:15:00 | 3810.00 | 3813.27 | 3724.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-09 11:15:00 | 3841.70 | 3811.72 | 3734.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 3999.00 | 4056.22 | 3958.24 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-29 09:15:00 | 3891.70 | 4048.75 | 3958.75 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-28 10:15:00 | 3251.80 | 2024-12-04 10:15:00 | 3368.10 | EXIT_EMA400 | -116.30 |
| BUY | 2025-02-07 09:15:00 | 3438.10 | 2025-02-10 09:15:00 | 3377.75 | EXIT_EMA400 | -60.35 |
| SELL | 2025-03-24 09:15:00 | 3092.10 | 2025-04-09 13:15:00 | 3183.20 | EXIT_EMA400 | -91.10 |
| SELL | 2025-04-08 14:15:00 | 3120.60 | 2025-04-09 13:15:00 | 3183.20 | EXIT_EMA400 | -62.60 |
| BUY | 2025-06-12 12:15:00 | 3512.00 | 2025-06-13 09:15:00 | 3408.90 | EXIT_EMA400 | -103.10 |
| SELL | 2025-08-06 15:15:00 | 3400.00 | 2025-08-08 09:15:00 | 3469.40 | EXIT_EMA400 | -69.40 |
| BUY | 2025-12-09 11:15:00 | 3841.70 | 2026-01-07 09:15:00 | 4162.10 | TARGET | 320.40 |
