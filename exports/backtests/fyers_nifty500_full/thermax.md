# Thermax Ltd. (THERMAX.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4065.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 7 |
| ENTRY1 | 5 |
| ENTRY2 | 5 |
| EXIT | 4 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / EMA400 exits:** 2 / 8
- **Total realized P&L (per unit):** -88.79
- **Avg P&L per closed trade:** -8.88

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 15:15:00 | 4435.00 | 5049.78 | 5052.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 14:15:00 | 4378.85 | 5015.05 | 5034.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 4534.65 | 4500.11 | 4662.85 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 15:15:00 | 5300.00 | 4774.07 | 4773.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-25 11:15:00 | 5372.00 | 5041.15 | 4954.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 12:15:00 | 5080.40 | 5087.81 | 4988.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-07 09:15:00 | 5112.80 | 5063.54 | 4990.98 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 5083.00 | 5089.78 | 5014.38 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-11-13 11:15:00 | 4986.40 | 5088.46 | 5014.47 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 09:15:00 | 4379.70 | 4957.38 | 4958.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 09:15:00 | 4293.25 | 4686.72 | 4775.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 09:15:00 | 3420.00 | 3390.00 | 3675.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 13:15:00 | 3339.10 | 3389.41 | 3669.57 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 3575.80 | 3365.52 | 3594.56 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-20 14:15:00 | 3524.00 | 3367.10 | 3594.21 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-25 09:15:00 | 3596.80 | 3388.22 | 3587.75 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 12:15:00 | 3592.10 | 3466.84 | 3466.61 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 3407.90 | 3469.93 | 3470.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 3401.40 | 3467.57 | 3469.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3489.30 | 3457.55 | 3463.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-08 10:15:00 | 3406.60 | 3459.10 | 3463.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 3452.90 | 3457.92 | 3463.25 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-11 09:15:00 | 3416.10 | 3453.82 | 3460.72 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-11 11:15:00 | 3462.50 | 3453.73 | 3460.60 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 3649.80 | 3466.47 | 3466.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 3811.20 | 3469.90 | 3467.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 11:15:00 | 3621.10 | 3677.00 | 3595.82 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 3271.10 | 3535.76 | 3537.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 12:15:00 | 3261.60 | 3486.36 | 3510.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 3361.00 | 3360.79 | 3425.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-08 13:15:00 | 3331.50 | 3360.49 | 3425.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3403.00 | 3358.85 | 3421.20 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-10 12:15:00 | 3374.00 | 3359.50 | 3420.61 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 3350.20 | 3342.43 | 3395.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-25 09:15:00 | 3282.50 | 3341.84 | 3391.31 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3271.70 | 3223.65 | 3287.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-27 12:15:00 | 3261.40 | 3224.96 | 3287.46 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 3285.00 | 3225.56 | 3287.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-27 14:15:00 | 3315.40 | 3226.45 | 3287.59 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 3185.60 | 2983.29 | 2982.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 14:15:00 | 3220.60 | 3061.77 | 3028.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 3069.10 | 3120.70 | 3066.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-24 12:15:00 | 3202.90 | 3120.27 | 3069.10 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-11-07 09:15:00 | 5112.80 | 2024-11-13 11:15:00 | 4986.40 | EXIT_EMA400 | -126.40 |
| SELL | 2025-03-10 13:15:00 | 3339.10 | 2025-03-25 09:15:00 | 3596.80 | EXIT_EMA400 | -257.70 |
| SELL | 2025-03-20 14:15:00 | 3524.00 | 2025-03-25 09:15:00 | 3596.80 | EXIT_EMA400 | -72.80 |
| SELL | 2025-07-08 10:15:00 | 3406.60 | 2025-07-11 11:15:00 | 3462.50 | EXIT_EMA400 | -55.90 |
| SELL | 2025-07-11 09:15:00 | 3416.10 | 2025-07-11 11:15:00 | 3462.50 | EXIT_EMA400 | -46.40 |
| SELL | 2025-09-10 12:15:00 | 3374.00 | 2025-09-25 14:15:00 | 3234.18 | TARGET | 139.82 |
| SELL | 2025-09-08 13:15:00 | 3331.50 | 2025-10-27 14:15:00 | 3315.40 | EXIT_EMA400 | 16.10 |
| SELL | 2025-09-25 09:15:00 | 3282.50 | 2025-10-27 14:15:00 | 3315.40 | EXIT_EMA400 | -32.90 |
| SELL | 2025-10-27 12:15:00 | 3261.40 | 2025-10-27 14:15:00 | 3315.40 | EXIT_EMA400 | -54.00 |
| BUY | 2026-03-24 12:15:00 | 3202.90 | 2026-04-10 09:15:00 | 3604.29 | TARGET | 401.39 |
