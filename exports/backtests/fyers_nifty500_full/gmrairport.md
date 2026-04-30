# GMR Airports Ltd. (GMRAIRPORT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 95.98
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 2.86
- **Avg P&L per closed trade:** 0.71

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 86.27 | 93.78 | 93.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 83.99 | 90.79 | 92.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 12:15:00 | 81.76 | 81.41 | 84.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-19 12:15:00 | 80.63 | 82.99 | 84.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 74.22 | 71.87 | 74.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-05 15:15:00 | 74.36 | 71.89 | 74.35 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 82.67 | 75.25 | 75.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 09:15:00 | 82.88 | 75.39 | 75.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 83.74 | 84.13 | 81.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 87.39 | 84.13 | 81.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 83.86 | 86.11 | 83.67 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-02 09:15:00 | 85.48 | 86.02 | 83.68 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-12 09:15:00 | 83.97 | 85.74 | 84.09 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 93.32 | 98.93 | 98.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 92.72 | 98.26 | 98.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 10:15:00 | 97.74 | 97.49 | 98.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 11:15:00 | 95.92 | 97.42 | 98.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-16 09:15:00 | 99.25 | 97.18 | 97.84 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 102.10 | 98.37 | 98.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 102.77 | 98.41 | 98.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 97.97 | 99.00 | 98.70 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 96.73 | 98.42 | 98.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 94.95 | 98.37 | 98.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 94.67 | 92.02 | 94.29 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-19 12:15:00 | 80.63 | 2025-01-13 14:15:00 | 69.51 | TARGET | 11.12 |
| BUY | 2025-05-12 09:15:00 | 87.39 | 2025-06-12 09:15:00 | 83.97 | EXIT_EMA400 | -3.42 |
| BUY | 2025-06-02 09:15:00 | 85.48 | 2025-06-12 09:15:00 | 83.97 | EXIT_EMA400 | -1.51 |
| SELL | 2026-02-12 11:15:00 | 95.92 | 2026-02-16 09:15:00 | 99.25 | EXIT_EMA400 | -3.33 |
