# IFCI Ltd. (IFCI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 58.55
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
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -7.32
- **Avg P&L per closed trade:** -1.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 61.44 | 69.57 | 69.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 11:15:00 | 59.12 | 66.39 | 67.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 63.12 | 59.61 | 62.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-14 13:15:00 | 58.80 | 60.77 | 62.98 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 64.69 | 60.51 | 62.52 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 58.45 | 46.24 | 46.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 60.08 | 46.73 | 46.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 58.73 | 59.07 | 54.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 59.67 | 59.02 | 54.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 14:15:00 | 58.64 | 61.31 | 59.05 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 53.44 | 57.86 | 57.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 52.83 | 57.76 | 57.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 56.32 | 54.40 | 55.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 13:15:00 | 53.29 | 55.72 | 56.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 55.72 | 55.39 | 55.80 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-03 11:15:00 | 57.06 | 55.41 | 55.81 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 58.41 | 56.14 | 56.14 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 55.34 | 56.14 | 56.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 55.13 | 56.13 | 56.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 15:15:00 | 56.31 | 55.99 | 56.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-24 11:15:00 | 55.93 | 56.03 | 56.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 55.80 | 56.03 | 56.08 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-24 13:15:00 | 55.63 | 56.02 | 56.08 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 56.09 | 56.02 | 56.07 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 58.65 | 56.12 | 56.12 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 54.48 | 56.14 | 56.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 53.84 | 56.04 | 56.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 55.80 | 55.75 | 55.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-18 09:15:00 | 53.93 | 55.71 | 55.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-23 12:15:00 | 52.76 | 50.32 | 52.17 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 62.00 | 52.87 | 52.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 64.10 | 56.11 | 54.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 58.90 | 59.33 | 57.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-23 14:15:00 | 59.70 | 59.33 | 57.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-02 09:15:00 | 57.15 | 59.30 | 57.50 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 51.18 | 56.35 | 56.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 50.51 | 56.24 | 56.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 55.14 | 54.28 | 55.15 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 62.61 | 55.83 | 55.80 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-14 13:15:00 | 58.80 | 2024-11-25 09:15:00 | 64.69 | EXIT_EMA400 | -5.89 |
| BUY | 2025-06-20 09:15:00 | 59.67 | 2025-07-25 14:15:00 | 58.64 | EXIT_EMA400 | -1.03 |
| SELL | 2025-09-26 13:15:00 | 53.29 | 2025-10-03 11:15:00 | 57.06 | EXIT_EMA400 | -3.77 |
| SELL | 2025-10-24 11:15:00 | 55.93 | 2025-10-24 13:15:00 | 55.48 | TARGET | 0.45 |
| SELL | 2025-10-24 13:15:00 | 55.63 | 2025-10-27 09:15:00 | 56.09 | EXIT_EMA400 | -0.46 |
| SELL | 2025-11-18 09:15:00 | 53.93 | 2025-12-08 10:15:00 | 48.00 | TARGET | 5.93 |
| BUY | 2026-02-23 14:15:00 | 59.70 | 2026-03-02 09:15:00 | 57.15 | EXIT_EMA400 | -2.55 |
