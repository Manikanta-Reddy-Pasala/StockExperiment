# IDBI Bank Ltd. (IDBI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 75.88
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -13.82
- **Avg P&L per closed trade:** -1.73

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 13:15:00 | 62.90 | 64.94 | 64.95 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 13:15:00 | 66.05 | 64.95 | 64.95 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 14:15:00 | 62.60 | 64.93 | 64.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 13:15:00 | 61.90 | 64.80 | 64.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 10:15:00 | 63.50 | 63.43 | 64.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-04 13:15:00 | 63.15 | 63.43 | 64.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-12-05 09:15:00 | 64.35 | 63.44 | 64.04 | Close above EMA400 |

### Cycle 4 — BUY (started 2023-12-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 12:15:00 | 66.45 | 64.50 | 64.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 66.95 | 64.57 | 64.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 65.35 | 65.53 | 65.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-22 09:15:00 | 66.70 | 65.54 | 65.07 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 81.60 | 85.87 | 81.34 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-12 13:15:00 | 82.60 | 85.79 | 81.35 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-03-13 09:15:00 | 80.10 | 85.66 | 81.35 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 88.65 | 91.76 | 91.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 87.43 | 91.37 | 91.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 09:15:00 | 84.34 | 84.16 | 86.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-04 09:15:00 | 81.89 | 84.09 | 86.52 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 85.94 | 83.91 | 86.19 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-07 09:15:00 | 86.65 | 83.94 | 86.19 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 80.76 | 76.32 | 76.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 81.85 | 76.51 | 76.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 79.25 | 79.54 | 78.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-02 09:15:00 | 80.52 | 79.55 | 78.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-06 14:15:00 | 77.79 | 79.61 | 78.40 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 88.54 | 92.87 | 92.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 88.10 | 92.82 | 92.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 92.80 | 92.18 | 92.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 12:15:00 | 90.02 | 92.58 | 92.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-08 09:15:00 | 92.02 | 91.26 | 91.93 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 93.34 | 92.40 | 92.39 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 89.77 | 92.37 | 92.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 88.88 | 92.14 | 92.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 92.30 | 91.97 | 92.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-08 10:15:00 | 91.60 | 92.12 | 92.22 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 09:15:00 | 93.91 | 92.05 | 92.18 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 94.14 | 92.30 | 92.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 95.28 | 92.33 | 92.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 98.30 | 99.56 | 97.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-12 14:15:00 | 100.03 | 98.22 | 97.20 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-18 09:15:00 | 96.81 | 98.30 | 97.36 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 10:15:00 | 75.09 | 103.64 | 103.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 14:15:00 | 73.90 | 102.50 | 103.18 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-12-04 13:15:00 | 63.15 | 2023-12-05 09:15:00 | 64.35 | EXIT_EMA400 | -1.20 |
| BUY | 2023-12-22 09:15:00 | 66.70 | 2024-01-16 09:15:00 | 71.60 | TARGET | 4.90 |
| BUY | 2024-03-12 13:15:00 | 82.60 | 2024-03-13 09:15:00 | 80.10 | EXIT_EMA400 | -2.50 |
| SELL | 2024-11-04 09:15:00 | 81.89 | 2024-11-07 09:15:00 | 86.65 | EXIT_EMA400 | -4.76 |
| BUY | 2025-05-02 09:15:00 | 80.52 | 2025-05-06 14:15:00 | 77.79 | EXIT_EMA400 | -2.73 |
| SELL | 2025-08-28 12:15:00 | 90.02 | 2025-09-08 09:15:00 | 92.02 | EXIT_EMA400 | -2.00 |
| SELL | 2025-10-08 10:15:00 | 91.60 | 2025-10-10 09:15:00 | 93.91 | EXIT_EMA400 | -2.31 |
| BUY | 2025-12-12 14:15:00 | 100.03 | 2025-12-18 09:15:00 | 96.81 | EXIT_EMA400 | -3.22 |
