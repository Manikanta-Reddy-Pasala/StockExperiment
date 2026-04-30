# SBFC Finance Ltd. (SBFC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 92.16
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 8 |
| ENTRY1 | 7 |
| ENTRY2 | 3 |
| EXIT | 7 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 1 / 9
- **Target hits / EMA400 exits:** 1 / 9
- **Total realized P&L (per unit):** -20.30
- **Avg P&L per closed trade:** -2.03

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 80.96 | 82.66 | 82.67 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 14:15:00 | 84.68 | 82.58 | 82.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 89.09 | 82.66 | 82.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 83.97 | 84.01 | 83.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-04 09:15:00 | 85.38 | 84.01 | 83.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 85.38 | 84.01 | 83.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-06 09:15:00 | 82.98 | 84.02 | 83.49 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 11:15:00 | 84.75 | 86.27 | 86.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 83.70 | 86.21 | 86.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 86.88 | 86.15 | 86.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-04 09:15:00 | 84.04 | 86.16 | 86.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 84.04 | 86.16 | 86.22 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-04 10:15:00 | 83.80 | 86.14 | 86.21 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 85.45 | 85.93 | 86.09 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-07 11:15:00 | 86.25 | 85.94 | 86.09 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 87.08 | 85.88 | 85.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 89.82 | 86.00 | 85.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 87.57 | 88.14 | 87.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-27 11:15:00 | 89.65 | 87.48 | 86.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 88.18 | 88.44 | 87.61 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-06 13:15:00 | 86.70 | 88.42 | 87.61 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 84.82 | 87.25 | 87.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 10:15:00 | 84.10 | 86.53 | 86.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 09:15:00 | 84.68 | 84.61 | 85.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-17 15:15:00 | 82.31 | 85.27 | 85.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 84.53 | 85.15 | 85.61 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 85.92 | 85.04 | 85.51 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 15:15:00 | 88.33 | 85.93 | 85.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 09:15:00 | 89.82 | 85.97 | 85.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 87.35 | 87.77 | 86.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 15:15:00 | 90.00 | 87.81 | 86.97 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-09 10:15:00 | 86.79 | 87.91 | 87.06 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 105.79 | 107.54 | 107.55 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 109.30 | 107.55 | 107.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 15:15:00 | 109.99 | 107.58 | 107.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 108.22 | 108.24 | 107.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-06 09:15:00 | 110.95 | 108.06 | 107.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 108.34 | 108.41 | 108.07 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-10 09:15:00 | 108.75 | 108.41 | 108.08 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 108.31 | 108.42 | 108.09 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-10 14:15:00 | 108.73 | 108.42 | 108.09 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-13 09:15:00 | 106.87 | 108.41 | 108.09 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 106.89 | 109.68 | 109.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 106.60 | 109.64 | 109.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 106.91 | 106.49 | 107.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 14:15:00 | 106.22 | 106.50 | 107.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 105.39 | 104.02 | 105.69 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-16 13:15:00 | 108.34 | 104.06 | 105.70 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-04 09:15:00 | 85.38 | 2024-09-06 09:15:00 | 82.98 | EXIT_EMA400 | -2.40 |
| SELL | 2024-11-04 09:15:00 | 84.04 | 2024-11-07 11:15:00 | 86.25 | EXIT_EMA400 | -2.21 |
| SELL | 2024-11-04 10:15:00 | 83.80 | 2024-11-07 11:15:00 | 86.25 | EXIT_EMA400 | -2.45 |
| BUY | 2024-12-27 11:15:00 | 89.65 | 2025-01-06 13:15:00 | 86.70 | EXIT_EMA400 | -2.95 |
| SELL | 2025-03-17 15:15:00 | 82.31 | 2025-03-21 09:15:00 | 85.92 | EXIT_EMA400 | -3.61 |
| BUY | 2025-04-07 15:15:00 | 90.00 | 2025-04-09 10:15:00 | 86.79 | EXIT_EMA400 | -3.21 |
| BUY | 2025-10-06 09:15:00 | 110.95 | 2025-10-13 09:15:00 | 106.87 | EXIT_EMA400 | -4.08 |
| BUY | 2025-10-10 09:15:00 | 108.75 | 2025-10-13 09:15:00 | 106.87 | EXIT_EMA400 | -1.88 |
| BUY | 2025-10-10 14:15:00 | 108.73 | 2025-10-13 09:15:00 | 106.87 | EXIT_EMA400 | -1.86 |
| SELL | 2025-12-24 14:15:00 | 106.22 | 2026-01-05 13:15:00 | 101.87 | TARGET | 4.35 |
