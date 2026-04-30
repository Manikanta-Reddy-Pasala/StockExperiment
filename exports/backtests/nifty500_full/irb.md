# IRB Infrastructure Developers Ltd. (IRB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 21.54
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 4 |
| EXIT | 5 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** 0.37
- **Avg P&L per closed trade:** 0.04

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 11:15:00 | 14.25 | 13.40 | 13.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 09:15:00 | 14.43 | 13.44 | 13.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 12:15:00 | 14.85 | 14.90 | 14.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-25 09:15:00 | 15.40 | 14.91 | 14.36 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-12 10:15:00 | 28.05 | 30.83 | 28.24 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 11:15:00 | 30.35 | 33.22 | 33.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 12:15:00 | 30.22 | 33.19 | 33.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 15:15:00 | 32.44 | 32.44 | 32.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-20 15:15:00 | 32.29 | 32.44 | 32.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 32.60 | 32.44 | 32.74 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-08-22 15:15:00 | 32.51 | 32.45 | 32.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 32.46 | 32.45 | 32.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-08-23 11:15:00 | 32.42 | 32.45 | 32.73 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-08-26 09:15:00 | 32.97 | 32.45 | 32.73 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 14:15:00 | 30.17 | 28.19 | 28.18 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 26.00 | 28.19 | 28.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 10:15:00 | 25.61 | 28.17 | 28.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 27.61 | 27.56 | 27.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-21 09:15:00 | 27.05 | 27.56 | 27.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-31 09:15:00 | 27.48 | 26.89 | 27.38 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 25.74 | 23.93 | 23.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 26.04 | 24.32 | 24.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 24.99 | 25.19 | 24.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 10:15:00 | 25.00 | 25.19 | 24.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 24.53 | 25.17 | 24.72 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 24.19 | 24.60 | 24.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 15:15:00 | 24.15 | 24.58 | 24.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 21.95 | 21.95 | 22.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 14:15:00 | 21.83 | 21.95 | 22.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 21.86 | 21.35 | 21.97 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-13 09:15:00 | 21.39 | 21.37 | 21.96 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 21.70 | 21.35 | 21.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-24 09:15:00 | 21.63 | 21.37 | 21.83 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 22.59 | 21.40 | 21.83 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 14:15:00 | 21.68 | 20.89 | 20.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 22.00 | 20.94 | 20.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 21.39 | 21.46 | 21.23 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-25 09:15:00 | 15.40 | 2023-11-17 09:15:00 | 18.52 | TARGET | 3.12 |
| SELL | 2024-08-20 15:15:00 | 32.29 | 2024-08-26 09:15:00 | 32.97 | EXIT_EMA400 | -0.68 |
| SELL | 2024-08-22 15:15:00 | 32.51 | 2024-08-26 09:15:00 | 32.97 | EXIT_EMA400 | -0.47 |
| SELL | 2024-08-23 11:15:00 | 32.42 | 2024-08-26 09:15:00 | 32.97 | EXIT_EMA400 | -0.55 |
| SELL | 2025-01-21 09:15:00 | 27.05 | 2025-01-27 14:15:00 | 24.68 | TARGET | 2.36 |
| BUY | 2025-06-13 10:15:00 | 25.00 | 2025-06-16 09:15:00 | 24.53 | EXIT_EMA400 | -0.47 |
| SELL | 2025-09-19 14:15:00 | 21.83 | 2025-10-27 09:15:00 | 22.59 | EXIT_EMA400 | -0.77 |
| SELL | 2025-10-13 09:15:00 | 21.39 | 2025-10-27 09:15:00 | 22.59 | EXIT_EMA400 | -1.21 |
| SELL | 2025-10-24 09:15:00 | 21.63 | 2025-10-27 09:15:00 | 22.59 | EXIT_EMA400 | -0.97 |
