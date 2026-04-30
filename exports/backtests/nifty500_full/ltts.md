# L&T Technology Services Ltd. (LTTS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 3626.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -952.85
- **Avg P&L per closed trade:** -136.12

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 12:15:00 | 4240.00 | 4402.13 | 4402.49 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 10:15:00 | 4599.00 | 4392.23 | 4391.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 4675.00 | 4433.96 | 4414.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 09:15:00 | 5379.45 | 5414.54 | 5207.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-28 11:15:00 | 5542.50 | 5341.94 | 5286.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 5433.55 | 5512.53 | 5405.92 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-16 11:15:00 | 5404.25 | 5510.74 | 5406.08 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 13:15:00 | 4702.50 | 5330.79 | 5333.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 10:15:00 | 4691.70 | 5306.21 | 5320.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 13:15:00 | 4685.00 | 4671.88 | 4859.36 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 12:15:00 | 5018.55 | 4909.86 | 4909.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 14:15:00 | 5043.85 | 4912.12 | 4910.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 4926.95 | 4935.95 | 4923.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-19 12:15:00 | 4954.60 | 4920.99 | 4916.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 4954.60 | 4920.99 | 4916.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-19 13:15:00 | 4891.40 | 4920.70 | 4916.37 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 15:15:00 | 5150.10 | 5295.59 | 5296.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 5135.40 | 5292.56 | 5294.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 5218.00 | 5208.41 | 5246.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 14:15:00 | 5097.35 | 5206.06 | 5244.85 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-12 09:15:00 | 5307.15 | 5200.21 | 5240.13 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 14:15:00 | 5295.85 | 5257.41 | 5257.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 5358.50 | 5258.89 | 5258.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 11:15:00 | 5268.65 | 5303.75 | 5283.89 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 4887.05 | 5263.38 | 5265.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 4822.60 | 5258.99 | 5262.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 5193.20 | 4946.95 | 5054.84 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 5340.10 | 5135.66 | 5134.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 5366.40 | 5142.07 | 5138.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 14:15:00 | 5276.55 | 5298.03 | 5228.27 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 10:15:00 | 4912.60 | 5174.30 | 5174.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 4850.65 | 5155.11 | 5164.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 09:15:00 | 4519.50 | 4453.78 | 4646.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-23 11:15:00 | 4457.50 | 4454.21 | 4644.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-15 09:15:00 | 4492.40 | 4324.50 | 4485.80 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 4432.90 | 4211.35 | 4210.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 13:15:00 | 4446.40 | 4215.79 | 4212.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 4487.80 | 4488.22 | 4393.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-29 14:15:00 | 4550.00 | 4488.93 | 4395.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-01 13:15:00 | 4361.00 | 4484.08 | 4402.05 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 3852.00 | 4354.77 | 4356.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 3823.70 | 4349.48 | 4354.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 3466.00 | 3461.01 | 3709.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-16 09:15:00 | 3274.70 | 3459.15 | 3707.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 3456.00 | 3358.62 | 3514.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 10:15:00 | 3447.60 | 3359.51 | 3514.50 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 3500.00 | 3365.15 | 3512.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 11:15:00 | 3520.00 | 3367.93 | 3512.70 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-03-28 11:15:00 | 5542.50 | 2024-04-16 11:15:00 | 5404.25 | EXIT_EMA400 | -138.25 |
| BUY | 2024-07-19 12:15:00 | 4954.60 | 2024-07-19 13:15:00 | 4891.40 | EXIT_EMA400 | -63.20 |
| SELL | 2024-11-08 14:15:00 | 5097.35 | 2024-11-12 09:15:00 | 5307.15 | EXIT_EMA400 | -209.80 |
| SELL | 2025-04-23 11:15:00 | 4457.50 | 2025-05-15 09:15:00 | 4492.40 | EXIT_EMA400 | -34.90 |
| BUY | 2025-12-29 14:15:00 | 4550.00 | 2026-01-01 13:15:00 | 4361.00 | EXIT_EMA400 | -189.00 |
| SELL | 2026-03-16 09:15:00 | 3274.70 | 2026-04-17 11:15:00 | 3520.00 | EXIT_EMA400 | -245.30 |
| SELL | 2026-04-16 10:15:00 | 3447.60 | 2026-04-17 11:15:00 | 3520.00 | EXIT_EMA400 | -72.40 |
