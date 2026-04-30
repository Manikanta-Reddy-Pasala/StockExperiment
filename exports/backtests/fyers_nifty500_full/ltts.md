# L&T Technology Services Ltd. (LTTS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3635.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -761.10
- **Avg P&L per closed trade:** -152.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 4957.15 | 5269.94 | 5270.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 4884.50 | 5247.39 | 5259.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 5218.00 | 5203.88 | 5233.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 14:15:00 | 5097.35 | 5201.78 | 5232.16 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-12 09:15:00 | 5307.15 | 5196.15 | 5227.92 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 15:15:00 | 5432.30 | 5244.42 | 5243.90 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 4830.00 | 5253.96 | 5255.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 4796.00 | 5249.40 | 5253.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 5193.20 | 4946.91 | 5052.27 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 5281.00 | 5131.68 | 5130.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 5319.00 | 5133.54 | 5131.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 14:15:00 | 5276.55 | 5308.56 | 5234.27 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 12:15:00 | 4904.65 | 5175.78 | 5176.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 4850.65 | 5160.92 | 5169.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 09:15:00 | 4515.00 | 4453.50 | 4647.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-23 12:15:00 | 4448.00 | 4453.92 | 4644.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-15 09:15:00 | 4492.60 | 4324.60 | 4486.33 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 4432.90 | 4211.01 | 4210.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 13:15:00 | 4447.00 | 4215.46 | 4212.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 4487.80 | 4488.39 | 4393.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-29 14:15:00 | 4550.00 | 4489.09 | 4395.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-01 13:15:00 | 4361.00 | 4484.31 | 4402.16 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 3852.00 | 4354.86 | 4356.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 3823.70 | 4349.57 | 4354.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 3470.00 | 3457.53 | 3703.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-16 09:15:00 | 3274.70 | 3455.71 | 3701.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 3456.00 | 3357.36 | 3511.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 10:15:00 | 3447.60 | 3358.26 | 3511.23 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 3500.00 | 3363.98 | 3509.61 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 11:15:00 | 3520.00 | 3366.78 | 3509.56 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 14:15:00 | 5097.35 | 2024-11-12 09:15:00 | 5307.15 | EXIT_EMA400 | -209.80 |
| SELL | 2025-04-23 12:15:00 | 4448.00 | 2025-05-15 09:15:00 | 4492.60 | EXIT_EMA400 | -44.60 |
| BUY | 2025-12-29 14:15:00 | 4550.00 | 2026-01-01 13:15:00 | 4361.00 | EXIT_EMA400 | -189.00 |
| SELL | 2026-03-16 09:15:00 | 3274.70 | 2026-04-17 11:15:00 | 3520.00 | EXIT_EMA400 | -245.30 |
| SELL | 2026-04-16 10:15:00 | 3447.60 | 2026-04-17 11:15:00 | 3520.00 | EXIT_EMA400 | -72.40 |
