# Hero MotoCorp Ltd. (HEROMOTOCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 5099.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 478.64
- **Avg P&L per closed trade:** 119.66

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 5216.70 | 5528.42 | 5529.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 5152.35 | 5513.17 | 5521.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 4225.90 | 4190.34 | 4409.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 4143.95 | 4217.04 | 4377.52 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-21 11:15:00 | 3859.90 | 3710.46 | 3842.42 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 4314.90 | 3887.26 | 3887.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 4352.10 | 3891.88 | 3889.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 4257.00 | 4257.63 | 4149.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 4299.00 | 4257.94 | 4150.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 4218.70 | 4280.12 | 4205.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-14 12:15:00 | 4234.00 | 4278.05 | 4205.69 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 4201.00 | 4320.01 | 4249.25 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 5379.00 | 5703.51 | 5705.06 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 5796.50 | 5701.05 | 5700.98 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 5593.50 | 5700.19 | 5700.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 5569.50 | 5698.89 | 5700.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 5653.50 | 5621.66 | 5656.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-04 09:15:00 | 5476.00 | 5639.46 | 5661.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-10 10:15:00 | 5685.00 | 5605.95 | 5640.93 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-11 09:15:00 | 4143.95 | 2025-04-07 09:15:00 | 3443.24 | TARGET | 700.71 |
| BUY | 2025-07-14 12:15:00 | 4234.00 | 2025-07-15 09:15:00 | 4318.93 | TARGET | 84.93 |
| BUY | 2025-06-24 09:15:00 | 4299.00 | 2025-07-25 09:15:00 | 4201.00 | EXIT_EMA400 | -98.00 |
| SELL | 2026-03-04 09:15:00 | 5476.00 | 2026-03-10 10:15:00 | 5685.00 | EXIT_EMA400 | -209.00 |
