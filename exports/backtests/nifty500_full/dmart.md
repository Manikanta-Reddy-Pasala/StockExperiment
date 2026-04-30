# Avenue Supermarts Ltd. (DMART.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 4585.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 7 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -634.91
- **Avg P&L per closed trade:** -90.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 10:15:00 | 3792.70 | 3660.58 | 3659.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 11:15:00 | 3799.85 | 3661.97 | 3660.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 09:15:00 | 3717.15 | 3722.51 | 3696.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-04 09:15:00 | 3848.90 | 3701.39 | 3690.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-10-23 09:15:00 | 3729.95 | 3774.80 | 3740.90 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 11:15:00 | 3640.80 | 3718.09 | 3718.47 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 13:15:00 | 3793.00 | 3719.25 | 3718.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 14:15:00 | 3804.00 | 3723.70 | 3721.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-21 12:15:00 | 3926.70 | 3945.96 | 3870.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-22 09:15:00 | 3983.75 | 3946.43 | 3872.09 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 3910.00 | 3979.58 | 3909.67 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-04 11:15:00 | 3906.75 | 3978.85 | 3909.66 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 3672.00 | 3870.33 | 3870.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 09:15:00 | 3660.10 | 3868.23 | 3869.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 14:15:00 | 3763.50 | 3759.28 | 3798.15 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 09:15:00 | 3986.90 | 3821.63 | 3821.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-12 10:15:00 | 4025.95 | 3834.72 | 3828.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 4488.55 | 4497.29 | 4284.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-30 09:15:00 | 4600.00 | 4496.98 | 4291.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 4479.35 | 4628.00 | 4477.28 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-28 15:15:00 | 4472.00 | 4626.44 | 4477.25 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 4607.30 | 4994.63 | 4996.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 4565.25 | 4986.60 | 4992.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 4133.00 | 3663.48 | 3884.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-07 14:15:00 | 3824.25 | 3715.50 | 3891.60 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-03 09:15:00 | 4032.45 | 3635.20 | 3757.83 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 15:15:00 | 3998.80 | 3721.49 | 3721.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 09:15:00 | 4018.10 | 3740.04 | 3730.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 10:15:00 | 4137.30 | 4147.57 | 3999.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-19 09:15:00 | 4230.50 | 4094.17 | 4013.07 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 4038.00 | 4099.97 | 4037.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-29 12:15:00 | 4022.80 | 4098.61 | 4036.96 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 3950.00 | 4115.21 | 4115.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 3941.00 | 4108.71 | 4112.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 11:15:00 | 4205.30 | 4102.11 | 4108.59 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 4293.20 | 4116.32 | 4115.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 4316.10 | 4171.06 | 4147.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 4616.60 | 4620.00 | 4479.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-24 12:15:00 | 4729.40 | 4622.40 | 4488.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 4548.60 | 4614.96 | 4495.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 4490.20 | 4612.89 | 4495.93 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 4305.00 | 4429.39 | 4429.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 4294.80 | 4426.75 | 4428.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 3847.40 | 3838.14 | 3965.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 3810.80 | 3837.92 | 3963.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3853.10 | 3762.33 | 3857.20 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-05 10:15:00 | 3876.90 | 3763.47 | 3857.29 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 4263.00 | 3875.73 | 3873.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 10:15:00 | 4302.20 | 3883.62 | 3877.77 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-04 09:15:00 | 3848.90 | 2023-10-23 09:15:00 | 3729.95 | EXIT_EMA400 | -118.95 |
| BUY | 2023-12-22 09:15:00 | 3983.75 | 2024-01-04 11:15:00 | 3906.75 | EXIT_EMA400 | -77.00 |
| BUY | 2024-04-30 09:15:00 | 4600.00 | 2024-05-28 15:15:00 | 4472.00 | EXIT_EMA400 | -128.00 |
| SELL | 2025-01-07 14:15:00 | 3824.25 | 2025-01-13 09:15:00 | 3622.21 | TARGET | 202.04 |
| BUY | 2025-05-19 09:15:00 | 4230.50 | 2025-05-29 12:15:00 | 4022.80 | EXIT_EMA400 | -207.70 |
| BUY | 2025-09-24 12:15:00 | 4729.40 | 2025-09-29 11:15:00 | 4490.20 | EXIT_EMA400 | -239.20 |
| SELL | 2026-01-08 10:15:00 | 3810.80 | 2026-02-05 10:15:00 | 3876.90 | EXIT_EMA400 | -66.10 |
