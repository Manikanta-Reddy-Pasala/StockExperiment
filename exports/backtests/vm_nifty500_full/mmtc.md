# MMTC Ltd. (MMTC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 64.78
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 15 |
| ALERT2 | 14 |
| ALERT3 | 9 |
| ENTRY1 | 8 |
| ENTRY2 | 4 |
| EXIT | 8 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / EMA400 exits:** 2 / 10
- **Total realized P&L (per unit):** -11.66
- **Avg P&L per closed trade:** -0.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 09:15:00 | 51.60 | 53.82 | 53.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 10:15:00 | 51.15 | 53.79 | 53.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 09:15:00 | 55.15 | 53.49 | 53.65 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 12:15:00 | 58.40 | 53.84 | 53.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 11:15:00 | 61.15 | 54.35 | 54.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 56.45 | 56.55 | 55.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-26 10:15:00 | 59.75 | 56.80 | 55.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-11 10:15:00 | 72.70 | 78.60 | 74.26 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 12:15:00 | 67.45 | 71.53 | 71.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 14:15:00 | 66.55 | 71.44 | 71.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 73.20 | 71.27 | 71.40 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 11:15:00 | 76.85 | 71.55 | 71.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 14:15:00 | 77.70 | 72.23 | 71.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 72.75 | 73.24 | 72.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-16 09:15:00 | 73.35 | 73.19 | 72.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 73.35 | 73.19 | 72.48 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-16 10:15:00 | 73.65 | 73.19 | 72.49 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 72.80 | 73.19 | 72.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-16 15:15:00 | 72.30 | 73.17 | 72.49 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 13:15:00 | 71.05 | 72.49 | 72.49 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 12:15:00 | 72.70 | 72.50 | 72.50 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 11:15:00 | 71.85 | 72.50 | 72.50 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 13:15:00 | 75.10 | 72.52 | 72.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 09:15:00 | 75.75 | 72.59 | 72.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 09:15:00 | 71.40 | 73.26 | 72.92 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 12:15:00 | 68.35 | 72.62 | 72.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 64.75 | 72.42 | 72.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 12:15:00 | 72.06 | 71.76 | 72.17 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 15:15:00 | 77.14 | 72.55 | 72.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 13:15:00 | 83.41 | 73.08 | 72.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 99.98 | 100.16 | 94.49 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 83.52 | 93.25 | 93.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 82.37 | 93.14 | 93.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 84.42 | 81.67 | 85.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 09:15:00 | 77.21 | 81.57 | 85.14 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 80.96 | 79.08 | 82.44 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-06 09:15:00 | 84.10 | 79.53 | 82.27 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 64.14 | 58.07 | 58.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 70.14 | 58.67 | 58.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 67.38 | 68.17 | 64.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-17 09:15:00 | 72.09 | 68.23 | 64.57 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 68.29 | 69.47 | 67.20 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-15 09:15:00 | 68.82 | 69.37 | 67.22 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 67.66 | 69.34 | 67.73 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 62.90 | 66.78 | 66.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 62.02 | 66.56 | 66.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 65.85 | 65.75 | 66.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 14:15:00 | 65.08 | 65.73 | 66.18 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 64.62 | 64.52 | 65.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-08 09:15:00 | 66.36 | 64.48 | 65.30 | Close above EMA400 |

### Cycle 14 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 69.39 | 65.72 | 65.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 70.15 | 65.81 | 65.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 66.25 | 66.34 | 66.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-03 10:15:00 | 68.81 | 66.09 | 65.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 68.81 | 66.09 | 65.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-03 13:15:00 | 69.40 | 66.18 | 66.00 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 67.31 | 67.71 | 66.99 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-20 11:15:00 | 67.63 | 67.71 | 66.99 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-24 10:15:00 | 66.96 | 67.71 | 67.04 | Close below EMA400 |

### Cycle 15 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 64.16 | 66.87 | 66.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 15:15:00 | 63.83 | 66.59 | 66.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 58.60 | 58.57 | 61.33 | EMA200 retest candle locked |

### Cycle 16 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 69.20 | 63.08 | 63.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 70.79 | 63.21 | 63.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 63.46 | 64.40 | 63.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-27 09:15:00 | 64.66 | 64.37 | 63.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 64.66 | 64.37 | 63.82 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-27 10:15:00 | 63.76 | 64.37 | 63.82 | Close below EMA400 |

### Cycle 17 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 61.79 | 63.94 | 63.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 59.61 | 63.47 | 63.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 60.49 | 58.97 | 60.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-30 09:15:00 | 53.27 | 58.45 | 60.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 58.90 | 57.63 | 59.48 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 59.95 | 57.73 | 59.46 | Close above EMA400 |

### Cycle 18 — BUY (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 15:15:00 | 66.83 | 60.63 | 60.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 67.34 | 61.40 | 61.02 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-26 10:15:00 | 59.75 | 2024-01-23 09:15:00 | 72.22 | TARGET | 12.47 |
| BUY | 2024-04-16 09:15:00 | 73.35 | 2024-04-16 15:15:00 | 72.30 | EXIT_EMA400 | -1.05 |
| BUY | 2024-04-16 10:15:00 | 73.65 | 2024-04-16 15:15:00 | 72.30 | EXIT_EMA400 | -1.35 |
| SELL | 2024-11-13 09:15:00 | 77.21 | 2024-12-06 09:15:00 | 84.10 | EXIT_EMA400 | -6.89 |
| BUY | 2025-06-17 09:15:00 | 72.09 | 2025-07-25 09:15:00 | 67.66 | EXIT_EMA400 | -4.43 |
| BUY | 2025-07-15 09:15:00 | 68.82 | 2025-07-25 09:15:00 | 67.66 | EXIT_EMA400 | -1.16 |
| SELL | 2025-08-21 14:15:00 | 65.08 | 2025-08-28 09:15:00 | 61.79 | TARGET | 3.29 |
| BUY | 2025-10-03 10:15:00 | 68.81 | 2025-10-24 10:15:00 | 66.96 | EXIT_EMA400 | -1.85 |
| BUY | 2025-10-03 13:15:00 | 69.40 | 2025-10-24 10:15:00 | 66.96 | EXIT_EMA400 | -2.44 |
| BUY | 2025-10-20 11:15:00 | 67.63 | 2025-10-24 10:15:00 | 66.96 | EXIT_EMA400 | -0.67 |
| BUY | 2026-01-27 09:15:00 | 64.66 | 2026-01-27 10:15:00 | 63.76 | EXIT_EMA400 | -0.90 |
| SELL | 2026-03-30 09:15:00 | 53.27 | 2026-04-10 09:15:00 | 59.95 | EXIT_EMA400 | -6.68 |
