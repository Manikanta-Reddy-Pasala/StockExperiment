# IDFC First Bank Ltd. (IDFCFIRSTB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 69.64
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -4.61
- **Avg P&L per closed trade:** -0.58

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 14:15:00 | 81.10 | 89.10 | 89.11 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 90.60 | 87.50 | 87.50 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 15:15:00 | 86.75 | 87.57 | 87.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 09:15:00 | 86.15 | 87.56 | 87.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 10:15:00 | 87.20 | 86.97 | 87.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-15 09:15:00 | 86.60 | 86.99 | 87.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 86.60 | 86.99 | 87.25 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-01-16 09:15:00 | 87.80 | 86.99 | 87.24 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 15:15:00 | 82.90 | 79.79 | 79.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 83.31 | 79.82 | 79.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 79.93 | 80.25 | 80.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-03 14:15:00 | 80.86 | 80.19 | 80.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 80.86 | 80.19 | 80.01 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-04 14:15:00 | 81.16 | 80.22 | 80.03 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-08 11:15:00 | 79.95 | 80.31 | 80.09 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 78.29 | 79.89 | 79.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 77.92 | 79.87 | 79.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 74.63 | 74.47 | 76.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-29 10:15:00 | 73.49 | 74.52 | 76.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 74.03 | 73.69 | 74.72 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-27 11:15:00 | 74.72 | 73.72 | 74.71 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 68.47 | 60.13 | 60.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 69.00 | 63.75 | 62.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 72.89 | 73.41 | 70.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-16 11:15:00 | 73.78 | 73.40 | 70.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 10:15:00 | 70.92 | 73.25 | 71.14 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 15:15:00 | 67.99 | 70.23 | 70.24 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 72.32 | 70.23 | 70.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 72.61 | 70.25 | 70.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 71.08 | 71.13 | 70.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-16 13:15:00 | 71.37 | 71.13 | 70.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-22 14:15:00 | 70.89 | 71.30 | 70.89 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 12:15:00 | 69.07 | 70.59 | 70.60 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 71.80 | 70.60 | 70.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 72.01 | 70.62 | 70.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 15:15:00 | 78.21 | 78.26 | 75.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-24 09:15:00 | 78.52 | 78.26 | 75.95 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-21 09:15:00 | 81.27 | 83.47 | 81.65 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 70.83 | 81.68 | 81.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 66.62 | 77.45 | 79.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 15:15:00 | 66.19 | 66.07 | 70.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 09:15:00 | 65.69 | 66.07 | 70.90 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-27 13:15:00 | 70.12 | 66.83 | 69.78 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-15 09:15:00 | 86.60 | 2024-01-16 09:15:00 | 87.80 | EXIT_EMA400 | -1.20 |
| BUY | 2024-07-03 14:15:00 | 80.86 | 2024-07-08 11:15:00 | 79.95 | EXIT_EMA400 | -0.91 |
| BUY | 2024-07-04 14:15:00 | 81.16 | 2024-07-08 11:15:00 | 79.95 | EXIT_EMA400 | -1.21 |
| SELL | 2024-08-29 10:15:00 | 73.49 | 2024-09-27 11:15:00 | 74.72 | EXIT_EMA400 | -1.23 |
| BUY | 2025-07-16 11:15:00 | 73.78 | 2025-07-25 10:15:00 | 70.92 | EXIT_EMA400 | -2.86 |
| BUY | 2025-09-16 13:15:00 | 71.37 | 2025-09-22 14:15:00 | 70.89 | EXIT_EMA400 | -0.48 |
| BUY | 2025-11-24 09:15:00 | 78.52 | 2026-01-02 09:15:00 | 86.23 | TARGET | 7.71 |
| SELL | 2026-04-09 09:15:00 | 65.69 | 2026-04-27 13:15:00 | 70.12 | EXIT_EMA400 | -4.43 |
