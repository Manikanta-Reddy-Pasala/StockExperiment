# Supreme Industries Ltd. (SUPREMEIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3628.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -594.30
- **Avg P&L per closed trade:** -198.10

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 4130.00 | 3633.09 | 3632.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 12:15:00 | 4168.60 | 3643.36 | 3637.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 4253.30 | 4269.74 | 4089.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 15:15:00 | 4283.00 | 4269.88 | 4090.01 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-11 09:15:00 | 4107.00 | 4252.93 | 4110.75 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 4177.70 | 4321.74 | 4322.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 4163.70 | 4301.39 | 4311.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 3445.00 | 3410.99 | 3603.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 15:15:00 | 3358.60 | 3470.97 | 3572.20 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3526.00 | 3463.38 | 3554.24 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-02 13:15:00 | 3563.20 | 3474.75 | 3546.89 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 13:15:00 | 3792.50 | 3597.33 | 3596.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 3840.30 | 3609.16 | 3602.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 11:15:00 | 3787.50 | 3818.44 | 3734.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-10 09:15:00 | 3907.70 | 3818.86 | 3736.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 3694.00 | 3866.78 | 3787.76 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 3658.30 | 3755.46 | 3755.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 3648.80 | 3748.56 | 3752.11 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-07-03 15:15:00 | 4283.00 | 2025-07-11 09:15:00 | 4107.00 | EXIT_EMA400 | -176.00 |
| SELL | 2026-01-20 15:15:00 | 3358.60 | 2026-02-02 13:15:00 | 3563.20 | EXIT_EMA400 | -204.60 |
| BUY | 2026-03-10 09:15:00 | 3907.70 | 2026-03-23 09:15:00 | 3694.00 | EXIT_EMA400 | -213.70 |
