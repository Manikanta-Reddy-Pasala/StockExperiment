# NHPC Ltd. (NHPC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 83.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 2.38
- **Avg P&L per closed trade:** 0.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 15:15:00 | 96.74 | 100.73 | 100.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 12:15:00 | 95.95 | 100.27 | 100.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 12:15:00 | 98.60 | 98.50 | 99.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-05 09:15:00 | 98.01 | 98.52 | 99.38 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-06 09:15:00 | 85.97 | 82.56 | 85.68 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 84.92 | 78.82 | 78.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 85.92 | 81.19 | 80.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 83.03 | 84.36 | 82.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-14 14:15:00 | 86.16 | 83.53 | 82.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 86.60 | 87.18 | 85.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-17 12:15:00 | 85.21 | 87.01 | 85.44 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 84.26 | 85.46 | 85.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 83.29 | 85.39 | 85.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 85.34 | 84.83 | 85.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 09:15:00 | 83.22 | 84.80 | 85.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 82.21 | 81.03 | 82.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-11 12:15:00 | 81.94 | 81.04 | 82.58 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-12 09:15:00 | 83.37 | 81.10 | 82.58 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 87.02 | 83.67 | 83.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 87.78 | 83.85 | 83.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 83.75 | 84.04 | 83.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 85.41 | 84.05 | 83.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 85.41 | 84.05 | 83.87 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-29 14:15:00 | 85.89 | 84.11 | 83.90 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-24 12:15:00 | 85.00 | 85.72 | 85.02 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 81.79 | 84.66 | 84.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 80.82 | 84.20 | 84.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 78.25 | 78.01 | 79.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 10:15:00 | 77.28 | 78.06 | 79.81 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 79.59 | 78.04 | 79.67 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-31 13:15:00 | 79.10 | 78.07 | 79.67 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-01 09:15:00 | 80.03 | 78.11 | 79.66 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 83.56 | 77.08 | 77.08 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-05 09:15:00 | 98.01 | 2024-09-09 09:15:00 | 93.91 | TARGET | 4.10 |
| BUY | 2025-05-14 14:15:00 | 86.16 | 2025-06-17 12:15:00 | 85.21 | EXIT_EMA400 | -0.95 |
| SELL | 2025-08-14 09:15:00 | 83.22 | 2025-08-29 09:15:00 | 77.58 | TARGET | 5.64 |
| SELL | 2025-09-11 12:15:00 | 81.94 | 2025-09-12 09:15:00 | 83.37 | EXIT_EMA400 | -1.43 |
| BUY | 2025-09-29 09:15:00 | 85.41 | 2025-10-24 12:15:00 | 85.00 | EXIT_EMA400 | -0.41 |
| BUY | 2025-09-29 14:15:00 | 85.89 | 2025-10-24 12:15:00 | 85.00 | EXIT_EMA400 | -0.89 |
| SELL | 2025-12-29 10:15:00 | 77.28 | 2026-01-01 09:15:00 | 80.03 | EXIT_EMA400 | -2.75 |
| SELL | 2025-12-31 13:15:00 | 79.10 | 2026-01-01 09:15:00 | 80.03 | EXIT_EMA400 | -0.93 |
