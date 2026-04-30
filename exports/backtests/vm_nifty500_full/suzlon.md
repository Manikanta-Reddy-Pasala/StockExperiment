# Suzlon Energy Ltd. (SUZLON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 55.58
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -2.15
- **Avg P&L per closed trade:** -0.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 12:15:00 | 36.60 | 41.28 | 41.30 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 14:15:00 | 42.15 | 41.09 | 41.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 10:15:00 | 42.55 | 41.12 | 41.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 40.75 | 41.18 | 41.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-15 11:15:00 | 41.05 | 41.17 | 41.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 41.05 | 41.17 | 41.13 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-15 12:15:00 | 41.05 | 41.17 | 41.13 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 14:15:00 | 39.10 | 41.08 | 41.09 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 11:15:00 | 42.15 | 41.10 | 41.10 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 11:15:00 | 39.90 | 41.13 | 41.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 37.95 | 40.88 | 41.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 10:15:00 | 40.95 | 40.70 | 40.90 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 14:15:00 | 44.00 | 41.09 | 41.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 46.20 | 41.17 | 41.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 77.60 | 78.88 | 73.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 09:15:00 | 79.50 | 77.66 | 73.38 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-15 09:15:00 | 73.12 | 77.09 | 73.64 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 15:15:00 | 67.27 | 71.92 | 71.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 14:15:00 | 66.82 | 71.46 | 71.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 66.12 | 65.76 | 67.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-20 14:15:00 | 64.11 | 66.50 | 67.54 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 54.91 | 53.54 | 56.40 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-07 13:15:00 | 54.49 | 53.57 | 56.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 55.87 | 53.62 | 56.37 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-10 10:15:00 | 55.18 | 53.63 | 56.36 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 55.98 | 53.71 | 56.06 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-17 14:15:00 | 54.48 | 53.79 | 56.04 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-19 09:15:00 | 56.07 | 53.90 | 55.99 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 61.05 | 56.36 | 56.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 61.25 | 56.41 | 56.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 64.03 | 64.15 | 61.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 11:15:00 | 64.24 | 64.06 | 61.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 64.12 | 65.52 | 64.09 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 10:15:00 | 63.86 | 65.50 | 64.09 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 58.30 | 63.44 | 63.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 13:15:00 | 57.76 | 62.21 | 62.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 59.29 | 59.23 | 60.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-17 13:15:00 | 58.97 | 59.22 | 60.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-19 15:15:00 | 60.60 | 59.26 | 60.52 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 54.34 | 45.94 | 45.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 55.71 | 46.83 | 46.37 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-15 11:15:00 | 41.05 | 2024-04-15 12:15:00 | 41.05 | EXIT_EMA400 | 0.00 |
| BUY | 2024-10-09 09:15:00 | 79.50 | 2024-10-15 09:15:00 | 73.12 | EXIT_EMA400 | -6.38 |
| SELL | 2024-12-20 14:15:00 | 64.11 | 2025-01-24 09:15:00 | 53.81 | TARGET | 10.30 |
| SELL | 2025-03-07 13:15:00 | 54.49 | 2025-03-19 09:15:00 | 56.07 | EXIT_EMA400 | -1.58 |
| SELL | 2025-03-10 10:15:00 | 55.18 | 2025-03-19 09:15:00 | 56.07 | EXIT_EMA400 | -0.89 |
| SELL | 2025-03-17 14:15:00 | 54.48 | 2025-03-19 09:15:00 | 56.07 | EXIT_EMA400 | -1.59 |
| BUY | 2025-06-20 11:15:00 | 64.24 | 2025-07-25 10:15:00 | 63.86 | EXIT_EMA400 | -0.38 |
| SELL | 2025-09-17 13:15:00 | 58.97 | 2025-09-19 15:15:00 | 60.60 | EXIT_EMA400 | -1.63 |
