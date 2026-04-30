# Bajaj Finance Ltd. (BAJFINANCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 937.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 7 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -520.32
- **Avg P&L per closed trade:** -74.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 12:15:00 | 3659.07 | 3606.80 | 3606.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 14:15:00 | 3692.98 | 3608.20 | 3607.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 10:15:00 | 3886.45 | 3893.65 | 3798.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-19 11:15:00 | 3910.15 | 3893.81 | 3799.17 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-10-26 09:15:00 | 3756.68 | 3891.73 | 3809.72 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 14:15:00 | 3608.70 | 3767.45 | 3767.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 09:15:00 | 3583.90 | 3764.08 | 3766.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 09:15:00 | 3673.60 | 3669.38 | 3710.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-08 11:15:00 | 3646.38 | 3675.76 | 3708.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-12-14 09:15:00 | 3739.20 | 3669.68 | 3701.07 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 15:15:00 | 3868.00 | 3714.34 | 3713.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 09:15:00 | 3879.62 | 3715.98 | 3714.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 13:15:00 | 3745.50 | 3748.73 | 3732.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-16 09:15:00 | 3780.15 | 3748.79 | 3732.91 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 3734.95 | 3748.82 | 3733.17 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-16 14:15:00 | 3724.70 | 3748.56 | 3733.19 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 09:15:00 | 3526.07 | 3719.39 | 3719.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 09:15:00 | 3439.05 | 3686.41 | 3702.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 09:15:00 | 3320.00 | 3306.03 | 3406.88 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 12:15:00 | 3602.77 | 3460.86 | 3460.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 14:15:00 | 3610.12 | 3463.75 | 3462.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 3480.98 | 3481.04 | 3471.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-16 13:15:00 | 3513.18 | 3481.21 | 3471.55 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 3513.18 | 3481.21 | 3471.55 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-18 09:15:00 | 3453.82 | 3480.92 | 3471.55 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 15:15:00 | 3302.20 | 3474.09 | 3474.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 3269.00 | 3416.10 | 3435.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 3414.18 | 3408.60 | 3431.11 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 3631.12 | 3449.95 | 3449.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 3639.00 | 3456.50 | 3453.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 10:15:00 | 3503.43 | 3513.98 | 3486.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-25 12:15:00 | 3540.85 | 3514.17 | 3487.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 3531.25 | 3542.56 | 3513.54 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-11 09:15:00 | 3512.75 | 3542.11 | 3513.60 | Close below EMA400 |

### Cycle 8 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 11:15:00 | 3317.23 | 3495.76 | 3496.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 3304.98 | 3455.57 | 3473.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 13:15:00 | 3379.95 | 3370.66 | 3417.98 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 10:15:00 | 3656.00 | 3443.21 | 3442.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 11:15:00 | 3676.38 | 3470.47 | 3456.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 3648.32 | 3698.71 | 3608.62 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 3455.55 | 3570.43 | 3570.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 3422.00 | 3567.83 | 3569.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 3396.00 | 3384.01 | 3444.95 | EMA200 retest candle locked |

### Cycle 11 — BUY (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 13:15:00 | 3697.88 | 3473.44 | 3472.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 09:15:00 | 3779.55 | 3480.88 | 3476.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 4247.50 | 4298.26 | 4133.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-04 10:15:00 | 4380.88 | 4298.79 | 4139.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 4306.75 | 4448.53 | 4293.99 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-30 11:15:00 | 4292.50 | 4445.53 | 4294.02 | Close below EMA400 |

### Cycle 12 — SELL (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 14:15:00 | 938.00 | 4380.08 | 4386.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 09:15:00 | 927.00 | 4311.82 | 4351.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 961.00 | 960.24 | 1342.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-12 09:15:00 | 945.05 | 996.77 | 1025.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-12 10:15:00 | 987.80 | 962.69 | 986.44 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-19 11:15:00 | 3910.15 | 2023-10-26 09:15:00 | 3756.68 | EXIT_EMA400 | -153.47 |
| SELL | 2023-12-08 11:15:00 | 3646.38 | 2023-12-14 09:15:00 | 3739.20 | EXIT_EMA400 | -92.82 |
| BUY | 2024-01-16 09:15:00 | 3780.15 | 2024-01-16 14:15:00 | 3724.70 | EXIT_EMA400 | -55.45 |
| BUY | 2024-04-16 13:15:00 | 3513.18 | 2024-04-18 09:15:00 | 3453.82 | EXIT_EMA400 | -59.35 |
| BUY | 2024-06-25 12:15:00 | 3540.85 | 2024-07-11 09:15:00 | 3512.75 | EXIT_EMA400 | -28.10 |
| BUY | 2025-04-04 10:15:00 | 4380.88 | 2025-04-30 11:15:00 | 4292.50 | EXIT_EMA400 | -88.38 |
| SELL | 2026-01-12 09:15:00 | 945.05 | 2026-02-12 10:15:00 | 987.80 | EXIT_EMA400 | -42.75 |
