# Gujarat Fluorochemicals Ltd. (FLUOROCHEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3590.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 57.87
- **Avg P&L per closed trade:** 8.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 3659.00 | 4129.77 | 4129.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 3631.05 | 4124.81 | 4127.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 3782.60 | 3742.50 | 3864.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-12 09:15:00 | 3615.30 | 3741.19 | 3862.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 3805.00 | 3741.00 | 3859.01 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-13 09:15:00 | 3795.60 | 3741.55 | 3858.69 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-07 12:15:00 | 3773.55 | 3673.87 | 3768.33 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 4017.20 | 3816.78 | 3816.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 4021.85 | 3818.83 | 3817.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 3715.00 | 3844.08 | 3830.88 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-08 11:15:00 | 3702.60 | 3818.39 | 3818.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 09:15:00 | 3655.00 | 3812.14 | 3815.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 11:15:00 | 3846.85 | 3803.41 | 3810.78 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 13:15:00 | 3930.00 | 3818.49 | 3818.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 13:15:00 | 3956.60 | 3825.65 | 3821.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 10:15:00 | 3850.20 | 3874.97 | 3850.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-06 10:15:00 | 3907.00 | 3872.84 | 3852.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 3907.00 | 3872.84 | 3852.11 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-07 09:15:00 | 3807.00 | 3872.21 | 3852.41 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 3640.10 | 3860.23 | 3860.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 3632.70 | 3857.96 | 3859.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 3635.70 | 3633.56 | 3716.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-01 09:15:00 | 3575.00 | 3632.23 | 3713.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-22 09:15:00 | 3619.70 | 3526.51 | 3614.59 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 3805.60 | 3543.14 | 3542.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 10:15:00 | 3837.00 | 3546.07 | 3544.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 3617.40 | 3646.61 | 3606.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-07 09:15:00 | 3685.30 | 3646.58 | 3606.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-15 14:15:00 | 3619.40 | 3660.09 | 3623.09 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 3506.90 | 3624.67 | 3624.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 09:15:00 | 3491.90 | 3620.10 | 3622.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 3485.60 | 3478.19 | 3536.24 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 3658.10 | 3554.53 | 3554.29 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 3493.70 | 3554.31 | 3554.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 3475.00 | 3552.15 | 3553.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 3291.30 | 3283.33 | 3387.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-04 09:15:00 | 3240.10 | 3377.46 | 3399.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 3270.00 | 3240.27 | 3304.78 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-27 09:15:00 | 3210.60 | 3240.02 | 3304.01 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-07 11:15:00 | 3282.90 | 3209.44 | 3275.96 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 3585.40 | 3309.52 | 3308.94 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-13 09:15:00 | 3795.60 | 2025-02-14 10:15:00 | 3606.32 | TARGET | 189.28 |
| SELL | 2025-02-12 09:15:00 | 3615.30 | 2025-03-07 12:15:00 | 3773.55 | EXIT_EMA400 | -158.25 |
| BUY | 2025-05-06 10:15:00 | 3907.00 | 2025-05-07 09:15:00 | 3807.00 | EXIT_EMA400 | -100.00 |
| SELL | 2025-07-01 09:15:00 | 3575.00 | 2025-07-22 09:15:00 | 3619.70 | EXIT_EMA400 | -44.70 |
| BUY | 2025-10-07 09:15:00 | 3685.30 | 2025-10-15 14:15:00 | 3619.40 | EXIT_EMA400 | -65.90 |
| SELL | 2026-03-27 09:15:00 | 3210.60 | 2026-03-30 13:15:00 | 2930.36 | TARGET | 280.24 |
| SELL | 2026-03-04 09:15:00 | 3240.10 | 2026-04-07 11:15:00 | 3282.90 | EXIT_EMA400 | -42.80 |
