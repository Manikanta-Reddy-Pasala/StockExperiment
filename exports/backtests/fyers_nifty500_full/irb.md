# IRB Infrastructure Developers Ltd. (IRB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 21.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 3.76
- **Avg P&L per closed trade:** 0.54

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 15:15:00 | 30.38 | 33.07 | 33.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 30.00 | 31.94 | 32.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 14:15:00 | 31.73 | 31.25 | 31.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-04 12:15:00 | 29.61 | 30.99 | 31.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-02 13:15:00 | 27.73 | 26.17 | 27.55 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 13:15:00 | 30.31 | 28.16 | 28.16 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 25.62 | 28.17 | 28.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 25.37 | 28.11 | 28.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 27.62 | 27.56 | 27.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-21 09:15:00 | 27.05 | 27.56 | 27.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-31 09:15:00 | 27.48 | 26.89 | 27.38 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 25.74 | 23.93 | 23.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 26.04 | 24.32 | 24.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 24.99 | 25.19 | 24.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 14:15:00 | 25.08 | 25.18 | 24.73 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 24.54 | 25.17 | 24.73 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 24.29 | 24.60 | 24.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 24.19 | 24.60 | 24.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 10:15:00 | 22.07 | 21.95 | 22.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-17 13:15:00 | 21.90 | 21.95 | 22.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 21.87 | 21.36 | 21.97 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-13 09:15:00 | 21.41 | 21.37 | 21.96 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 21.70 | 21.35 | 21.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-24 10:15:00 | 21.58 | 21.38 | 21.83 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 22.60 | 21.41 | 21.83 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 15:15:00 | 21.95 | 21.12 | 21.12 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 20.61 | 21.12 | 21.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 20.35 | 21.09 | 21.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 20.89 | 20.88 | 20.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-02 09:15:00 | 20.05 | 20.85 | 20.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 20.76 | 20.62 | 20.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-10 14:15:00 | 20.99 | 20.63 | 20.82 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 21.53 | 20.89 | 20.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 14:15:00 | 21.68 | 20.89 | 20.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 21.39 | 21.48 | 21.24 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-04 12:15:00 | 29.61 | 2024-11-18 09:15:00 | 23.79 | TARGET | 5.82 |
| SELL | 2025-01-21 09:15:00 | 27.05 | 2025-01-27 14:15:00 | 24.72 | TARGET | 2.33 |
| BUY | 2025-06-13 14:15:00 | 25.08 | 2025-06-16 09:15:00 | 24.54 | EXIT_EMA400 | -0.54 |
| SELL | 2025-09-17 13:15:00 | 21.90 | 2025-10-27 09:15:00 | 22.60 | EXIT_EMA400 | -0.70 |
| SELL | 2025-10-13 09:15:00 | 21.41 | 2025-10-27 09:15:00 | 22.60 | EXIT_EMA400 | -1.19 |
| SELL | 2025-10-24 10:15:00 | 21.58 | 2025-10-27 09:15:00 | 22.60 | EXIT_EMA400 | -1.02 |
| SELL | 2026-03-02 09:15:00 | 20.05 | 2026-03-10 14:15:00 | 20.99 | EXIT_EMA400 | -0.94 |
