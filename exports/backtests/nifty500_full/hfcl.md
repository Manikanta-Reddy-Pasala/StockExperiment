# HFCL Ltd. (HFCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 116.03
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -10.15
- **Avg P&L per closed trade:** -1.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 10:15:00 | 65.25 | 71.16 | 71.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 14:15:00 | 65.20 | 70.93 | 71.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 68.75 | 68.35 | 69.48 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 10:15:00 | 79.35 | 69.27 | 69.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 14:15:00 | 80.95 | 69.67 | 69.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 15:15:00 | 94.05 | 94.54 | 87.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-13 10:15:00 | 95.60 | 94.54 | 87.52 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-11 14:15:00 | 95.50 | 103.43 | 96.74 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 11:15:00 | 119.70 | 138.00 | 138.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 12:15:00 | 119.10 | 137.82 | 137.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 130.55 | 129.80 | 133.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 09:15:00 | 123.09 | 129.45 | 132.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 129.32 | 128.78 | 131.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-19 15:15:00 | 128.03 | 128.77 | 131.82 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 130.41 | 128.51 | 131.27 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-27 10:15:00 | 131.48 | 128.54 | 131.27 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 92.96 | 86.44 | 86.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 93.34 | 86.83 | 86.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 86.80 | 87.31 | 86.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 10:15:00 | 88.40 | 87.32 | 86.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 88.40 | 87.32 | 86.89 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 15:15:00 | 86.41 | 87.31 | 86.89 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 81.22 | 86.48 | 86.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 80.56 | 86.42 | 86.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 86.24 | 85.46 | 85.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-07 11:15:00 | 83.44 | 85.60 | 85.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 75.91 | 73.14 | 75.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-16 14:15:00 | 76.50 | 73.20 | 75.98 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 73.02 | 68.81 | 68.81 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 64.96 | 68.86 | 68.87 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 12:15:00 | 71.62 | 68.86 | 68.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 09:15:00 | 73.15 | 68.99 | 68.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 69.75 | 70.06 | 69.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-20 09:15:00 | 71.70 | 70.08 | 69.55 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 68.03 | 70.14 | 69.60 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-13 10:15:00 | 95.60 | 2024-03-11 14:15:00 | 95.50 | EXIT_EMA400 | -0.10 |
| SELL | 2024-11-13 09:15:00 | 123.09 | 2024-11-27 10:15:00 | 131.48 | EXIT_EMA400 | -8.39 |
| SELL | 2024-11-19 15:15:00 | 128.03 | 2024-11-27 10:15:00 | 131.48 | EXIT_EMA400 | -3.45 |
| BUY | 2025-06-13 10:15:00 | 88.40 | 2025-06-13 15:15:00 | 86.41 | EXIT_EMA400 | -1.99 |
| SELL | 2025-07-07 11:15:00 | 83.44 | 2025-07-25 12:15:00 | 75.99 | TARGET | 7.45 |
| BUY | 2026-03-20 09:15:00 | 71.70 | 2026-03-23 09:15:00 | 68.03 | EXIT_EMA400 | -3.67 |
