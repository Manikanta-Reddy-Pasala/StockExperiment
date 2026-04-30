# Craftsman Automation Ltd. (CRAFTSMAN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 7667.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / EMA400 exits:** 0 / 2
- **Total realized P&L (per unit):** -395.50
- **Avg P&L per closed trade:** -197.75

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 12:15:00 | 5046.80 | 5816.61 | 5817.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 4970.45 | 5808.19 | 5813.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 5131.60 | 5093.78 | 5283.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-16 09:15:00 | 5076.35 | 5093.70 | 5278.69 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-17 12:15:00 | 5311.35 | 5094.34 | 5269.95 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 4750.00 | 4678.46 | 4678.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 4780.15 | 4679.47 | 4678.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 4565.00 | 4682.52 | 4680.42 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 4563.00 | 4677.60 | 4677.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 4306.80 | 4673.91 | 4676.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 4623.80 | 4597.78 | 4635.09 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 15:15:00 | 4769.00 | 4661.63 | 4661.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 4888.20 | 4669.93 | 4665.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 13:15:00 | 4658.30 | 4684.50 | 4673.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 14:15:00 | 4802.60 | 4681.95 | 4672.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-07 09:15:00 | 4642.10 | 4687.94 | 4676.07 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 09:15:00 | 6936.00 | 7448.31 | 7448.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 11:15:00 | 6855.00 | 7437.44 | 7442.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 12:15:00 | 7269.50 | 7095.35 | 7233.94 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 7708.00 | 7326.60 | 7326.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 13:15:00 | 7763.00 | 7411.62 | 7372.30 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-16 09:15:00 | 5076.35 | 2024-12-17 12:15:00 | 5311.35 | EXIT_EMA400 | -235.00 |
| BUY | 2025-05-05 14:15:00 | 4802.60 | 2025-05-07 09:15:00 | 4642.10 | EXIT_EMA400 | -160.50 |
