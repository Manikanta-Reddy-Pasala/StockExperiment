# J.K. Cement Ltd. (JKCEMENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 5298.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -132.97
- **Avg P&L per closed trade:** -19.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 4204.45 | 4438.42 | 4439.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 4185.05 | 4435.90 | 4438.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 4284.45 | 4178.17 | 4272.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-27 10:15:00 | 4115.00 | 4179.32 | 4265.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 4221.80 | 4177.21 | 4258.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-29 14:15:00 | 4261.95 | 4179.15 | 4258.35 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 4583.00 | 4323.19 | 4322.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 4716.60 | 4339.33 | 4330.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 4570.00 | 4593.62 | 4503.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-23 14:15:00 | 4789.10 | 4565.03 | 4510.81 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-14 11:15:00 | 4642.90 | 4735.67 | 4644.64 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 09:15:00 | 4448.85 | 4596.16 | 4596.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 4431.40 | 4593.14 | 4594.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 4530.90 | 4512.39 | 4549.22 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 10:15:00 | 4750.10 | 4580.58 | 4579.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 10:15:00 | 4784.45 | 4590.98 | 4585.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 11:15:00 | 4690.25 | 4708.75 | 4651.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 13:15:00 | 4791.10 | 4709.79 | 4652.54 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 6627.50 | 6839.41 | 6557.88 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-08 10:15:00 | 6706.00 | 6836.15 | 6559.04 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 6574.50 | 6811.53 | 6571.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-11 09:15:00 | 6610.50 | 6809.53 | 6571.60 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-12 13:15:00 | 6561.50 | 6788.90 | 6573.75 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 6361.00 | 6538.86 | 6539.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 6279.50 | 6512.70 | 6525.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 14:15:00 | 5925.00 | 5888.49 | 6116.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-27 09:15:00 | 5824.00 | 5888.06 | 6114.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 5790.00 | 5638.46 | 5803.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-05 13:15:00 | 5756.00 | 5639.63 | 5803.36 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-07 09:15:00 | 5821.00 | 5648.33 | 5799.83 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-27 10:15:00 | 4115.00 | 2024-11-29 14:15:00 | 4261.95 | EXIT_EMA400 | -146.95 |
| BUY | 2025-01-23 14:15:00 | 4789.10 | 2025-02-14 11:15:00 | 4642.90 | EXIT_EMA400 | -146.20 |
| BUY | 2025-04-07 13:15:00 | 4791.10 | 2025-04-22 10:15:00 | 5206.78 | TARGET | 415.68 |
| BUY | 2025-09-08 10:15:00 | 6706.00 | 2025-09-12 13:15:00 | 6561.50 | EXIT_EMA400 | -144.50 |
| BUY | 2025-09-11 09:15:00 | 6610.50 | 2025-09-12 13:15:00 | 6561.50 | EXIT_EMA400 | -49.00 |
| SELL | 2025-11-27 09:15:00 | 5824.00 | 2026-01-07 09:15:00 | 5821.00 | EXIT_EMA400 | 3.00 |
| SELL | 2026-01-05 13:15:00 | 5756.00 | 2026-01-07 09:15:00 | 5821.00 | EXIT_EMA400 | -65.00 |
