# MMTC Ltd. (MMTC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 64.81
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -21.93
- **Avg P&L per closed trade:** -2.44

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 11:15:00 | 82.07 | 93.02 | 93.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 12:15:00 | 81.47 | 92.91 | 92.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 84.49 | 81.64 | 85.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 09:15:00 | 77.21 | 81.53 | 85.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 80.96 | 79.07 | 82.37 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-06 09:15:00 | 84.13 | 79.52 | 82.21 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 64.27 | 58.00 | 57.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 70.15 | 58.67 | 58.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 67.38 | 68.18 | 64.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-17 09:15:00 | 72.09 | 68.24 | 64.56 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 68.27 | 69.47 | 67.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-15 09:15:00 | 68.82 | 69.37 | 67.22 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 67.66 | 69.34 | 67.72 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 62.90 | 66.78 | 66.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 62.02 | 66.56 | 66.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 65.85 | 65.74 | 66.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 14:15:00 | 65.08 | 65.72 | 66.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 64.62 | 64.52 | 65.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-08 09:15:00 | 66.36 | 64.48 | 65.29 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 69.39 | 65.72 | 65.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 70.25 | 65.80 | 65.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 66.25 | 66.33 | 66.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-03 10:15:00 | 68.81 | 66.09 | 65.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 68.81 | 66.09 | 65.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-03 13:15:00 | 69.40 | 66.18 | 65.99 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 67.31 | 67.71 | 66.98 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-20 11:15:00 | 67.63 | 67.71 | 66.99 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-24 10:15:00 | 66.96 | 67.71 | 67.04 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 64.16 | 66.87 | 66.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 63.18 | 66.56 | 66.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 58.61 | 58.56 | 61.33 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 69.20 | 63.07 | 63.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 70.77 | 63.20 | 63.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 63.46 | 64.39 | 63.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-27 09:15:00 | 64.66 | 64.37 | 63.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 64.66 | 64.37 | 63.82 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-27 10:15:00 | 63.76 | 64.37 | 63.82 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 60.80 | 63.96 | 63.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 59.61 | 63.45 | 63.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 60.50 | 58.96 | 60.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-30 10:15:00 | 53.11 | 58.40 | 60.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 58.70 | 57.54 | 59.50 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 59.95 | 57.74 | 59.47 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 67.04 | 60.63 | 60.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 67.34 | 61.62 | 61.14 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-13 09:15:00 | 77.21 | 2024-12-06 09:15:00 | 84.13 | EXIT_EMA400 | -6.92 |
| BUY | 2025-06-17 09:15:00 | 72.09 | 2025-07-25 09:15:00 | 67.66 | EXIT_EMA400 | -4.43 |
| BUY | 2025-07-15 09:15:00 | 68.82 | 2025-07-25 09:15:00 | 67.66 | EXIT_EMA400 | -1.16 |
| SELL | 2025-08-21 14:15:00 | 65.08 | 2025-08-28 09:15:00 | 61.80 | TARGET | 3.28 |
| BUY | 2025-10-03 10:15:00 | 68.81 | 2025-10-24 10:15:00 | 66.96 | EXIT_EMA400 | -1.85 |
| BUY | 2025-10-03 13:15:00 | 69.40 | 2025-10-24 10:15:00 | 66.96 | EXIT_EMA400 | -2.44 |
| BUY | 2025-10-20 11:15:00 | 67.63 | 2025-10-24 10:15:00 | 66.96 | EXIT_EMA400 | -0.67 |
| BUY | 2026-01-27 09:15:00 | 64.66 | 2026-01-27 10:15:00 | 63.76 | EXIT_EMA400 | -0.90 |
| SELL | 2026-03-30 10:15:00 | 53.11 | 2026-04-10 09:15:00 | 59.95 | EXIT_EMA400 | -6.84 |
