# IFCI Ltd. (IFCI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 58.69
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -7.45
- **Avg P&L per closed trade:** -1.06

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 61.87 | 69.28 | 69.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 15:15:00 | 61.45 | 69.13 | 69.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 63.12 | 59.63 | 62.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-14 13:15:00 | 58.80 | 60.78 | 62.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 64.69 | 60.52 | 62.50 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 58.45 | 46.25 | 46.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 60.08 | 46.73 | 46.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 58.73 | 59.07 | 54.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 59.67 | 59.03 | 54.85 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 14:15:00 | 58.64 | 61.31 | 59.06 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 53.44 | 57.86 | 57.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 52.83 | 57.76 | 57.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 56.32 | 54.40 | 55.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 12:15:00 | 53.62 | 55.75 | 56.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 55.72 | 55.39 | 55.80 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-03 11:15:00 | 57.06 | 55.41 | 55.81 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 58.41 | 56.14 | 56.14 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 13:15:00 | 55.13 | 56.14 | 56.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 54.65 | 56.04 | 56.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 15:15:00 | 56.31 | 55.99 | 56.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-24 10:15:00 | 55.90 | 56.03 | 56.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 55.90 | 56.03 | 56.08 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-24 12:15:00 | 55.80 | 56.03 | 56.08 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 56.09 | 56.02 | 56.07 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 58.65 | 56.13 | 56.12 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 54.50 | 56.14 | 56.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 53.84 | 56.04 | 56.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 55.80 | 55.76 | 55.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-18 09:15:00 | 53.93 | 55.71 | 55.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-23 12:15:00 | 52.76 | 50.31 | 52.17 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 61.98 | 52.88 | 52.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 64.10 | 55.98 | 54.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 58.90 | 59.26 | 57.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-23 14:15:00 | 59.70 | 59.26 | 57.15 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-02 09:15:00 | 57.15 | 59.25 | 57.43 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 51.08 | 56.28 | 56.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 50.54 | 56.23 | 56.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 55.14 | 54.27 | 55.12 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 62.45 | 55.76 | 55.74 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-14 13:15:00 | 58.80 | 2024-11-25 09:15:00 | 64.69 | EXIT_EMA400 | -5.89 |
| BUY | 2025-06-20 09:15:00 | 59.67 | 2025-07-25 14:15:00 | 58.64 | EXIT_EMA400 | -1.03 |
| SELL | 2025-09-26 12:15:00 | 53.62 | 2025-10-03 11:15:00 | 57.06 | EXIT_EMA400 | -3.44 |
| SELL | 2025-10-24 10:15:00 | 55.90 | 2025-10-27 09:15:00 | 56.09 | EXIT_EMA400 | -0.19 |
| SELL | 2025-10-24 12:15:00 | 55.80 | 2025-10-27 09:15:00 | 56.09 | EXIT_EMA400 | -0.29 |
| SELL | 2025-11-18 09:15:00 | 53.93 | 2025-12-08 11:15:00 | 47.99 | TARGET | 5.94 |
| BUY | 2026-02-23 14:15:00 | 59.70 | 2026-03-02 09:15:00 | 57.15 | EXIT_EMA400 | -2.55 |
