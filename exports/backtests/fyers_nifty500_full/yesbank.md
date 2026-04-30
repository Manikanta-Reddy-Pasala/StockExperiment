# Yes Bank Ltd. (YESBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 19.99
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -0.76
- **Avg P&L per closed trade:** -0.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 23.16 | 24.13 | 24.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 22.73 | 24.12 | 24.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 10:15:00 | 23.91 | 23.88 | 23.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-16 14:15:00 | 23.45 | 23.87 | 23.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 20.82 | 20.31 | 21.09 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-04 11:15:00 | 21.24 | 20.36 | 21.08 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 20.45 | 17.95 | 17.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 21.78 | 18.70 | 18.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 20.44 | 20.51 | 19.73 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 18.61 | 19.82 | 19.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 18.53 | 19.79 | 19.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 19.36 | 19.29 | 19.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 10:15:00 | 19.14 | 19.33 | 19.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 19.43 | 19.28 | 19.46 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-29 11:15:00 | 19.46 | 19.28 | 19.46 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 20.39 | 19.58 | 19.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 20.75 | 19.61 | 19.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 22.43 | 22.47 | 21.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-17 09:15:00 | 22.97 | 22.47 | 21.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-08 11:15:00 | 22.16 | 22.61 | 22.20 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 21.39 | 21.98 | 21.98 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 22.79 | 21.98 | 21.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 23.15 | 21.99 | 21.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 22.17 | 22.45 | 22.25 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 21.16 | 22.09 | 22.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 20.92 | 21.94 | 22.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 14:15:00 | 19.01 | 19.01 | 19.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 18.74 | 19.01 | 19.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 19.73 | 19.03 | 19.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-16 10:15:00 | 19.98 | 19.04 | 19.78 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-16 14:15:00 | 23.45 | 2024-10-03 15:15:00 | 21.84 | TARGET | 1.61 |
| SELL | 2025-08-26 10:15:00 | 19.14 | 2025-08-29 11:15:00 | 19.46 | EXIT_EMA400 | -0.32 |
| BUY | 2025-11-17 09:15:00 | 22.97 | 2025-12-08 11:15:00 | 22.16 | EXIT_EMA400 | -0.81 |
| SELL | 2026-04-13 09:15:00 | 18.74 | 2026-04-16 10:15:00 | 19.98 | EXIT_EMA400 | -1.24 |
