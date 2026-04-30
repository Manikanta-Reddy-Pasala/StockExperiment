# Navin Fluorine International Ltd. (NAVINFLUOR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 6821.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / EMA400 exits:** 3 / 7
- **Total realized P&L (per unit):** 371.86
- **Avg P&L per closed trade:** 37.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 11:15:00 | 4609.70 | 4510.81 | 4510.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 10:15:00 | 4687.40 | 4517.43 | 4513.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 13:15:00 | 4534.00 | 4555.95 | 4535.14 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2023-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 12:15:00 | 4445.05 | 4522.02 | 4522.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 4359.25 | 4518.19 | 4520.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 3724.00 | 3674.80 | 3863.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-08 14:15:00 | 3610.35 | 3791.46 | 3823.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-10 11:15:00 | 3235.15 | 3111.43 | 3205.43 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 13:15:00 | 3417.60 | 3253.63 | 3253.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 15:15:00 | 3451.95 | 3257.31 | 3255.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 13:15:00 | 3242.80 | 3286.56 | 3271.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-14 10:15:00 | 3332.35 | 3278.26 | 3268.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 3332.35 | 3278.26 | 3268.29 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-14 15:15:00 | 3336.25 | 3280.32 | 3269.57 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-05-30 09:15:00 | 3256.20 | 3315.61 | 3293.85 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 15:15:00 | 3291.90 | 3497.73 | 3498.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 15:15:00 | 3280.00 | 3442.00 | 3467.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 3398.00 | 3383.12 | 3426.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-10 12:15:00 | 3353.60 | 3382.78 | 3425.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-24 09:15:00 | 3416.00 | 3335.20 | 3385.53 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 13:15:00 | 3474.85 | 3388.14 | 3387.99 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 3258.95 | 3387.75 | 3387.84 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 14:15:00 | 3475.35 | 3383.01 | 3383.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 3501.35 | 3385.03 | 3384.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 12:15:00 | 3485.00 | 3487.76 | 3446.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-16 13:15:00 | 3512.10 | 3488.01 | 3446.41 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-19 09:15:00 | 3407.25 | 3486.43 | 3449.01 | Close below EMA400 |

### Cycle 8 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 11:15:00 | 3200.00 | 3420.06 | 3420.77 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 12:15:00 | 3862.00 | 3417.62 | 3416.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 14:15:00 | 3914.35 | 3605.88 | 3535.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 13:15:00 | 3913.65 | 3943.13 | 3789.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-05 09:15:00 | 3976.90 | 3895.23 | 3789.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 3991.35 | 4093.25 | 3970.11 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-03 13:15:00 | 4091.85 | 4093.23 | 3970.72 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 3774.65 | 4091.40 | 3975.82 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 4644.00 | 4807.92 | 4807.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 4617.60 | 4804.32 | 4806.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 10:15:00 | 4725.00 | 4711.24 | 4751.80 | EMA200 retest candle locked |

### Cycle 11 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 5110.00 | 4783.76 | 4783.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 5151.60 | 4798.44 | 4790.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 5647.00 | 5695.59 | 5452.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-10 09:15:00 | 5807.00 | 5686.43 | 5463.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 5792.50 | 5845.25 | 5694.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-09 10:15:00 | 5835.00 | 5845.15 | 5694.89 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 5703.00 | 5842.24 | 5696.40 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-12 09:15:00 | 5789.50 | 5840.45 | 5696.95 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 5761.00 | 5896.94 | 5754.69 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-21 10:15:00 | 5685.00 | 5894.83 | 5754.34 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-08 14:15:00 | 3610.35 | 2024-03-13 09:15:00 | 2971.56 | TARGET | 638.79 |
| BUY | 2024-05-14 10:15:00 | 3332.35 | 2024-05-30 09:15:00 | 3256.20 | EXIT_EMA400 | -76.15 |
| BUY | 2024-05-14 15:15:00 | 3336.25 | 2024-05-30 09:15:00 | 3256.20 | EXIT_EMA400 | -80.05 |
| SELL | 2024-09-10 12:15:00 | 3353.60 | 2024-09-24 09:15:00 | 3416.00 | EXIT_EMA400 | -62.40 |
| BUY | 2024-12-16 13:15:00 | 3512.10 | 2024-12-19 09:15:00 | 3407.25 | EXIT_EMA400 | -104.85 |
| BUY | 2025-03-05 09:15:00 | 3976.90 | 2025-04-07 09:15:00 | 3774.65 | EXIT_EMA400 | -202.25 |
| BUY | 2025-04-03 13:15:00 | 4091.85 | 2025-04-07 09:15:00 | 3774.65 | EXIT_EMA400 | -317.20 |
| BUY | 2026-01-12 09:15:00 | 5789.50 | 2026-01-16 09:15:00 | 6067.14 | TARGET | 277.64 |
| BUY | 2026-01-09 10:15:00 | 5835.00 | 2026-01-19 10:15:00 | 6255.33 | TARGET | 420.33 |
| BUY | 2025-12-10 09:15:00 | 5807.00 | 2026-01-21 10:15:00 | 5685.00 | EXIT_EMA400 | -122.00 |
