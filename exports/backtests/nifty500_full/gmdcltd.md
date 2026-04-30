# Gujarat Mineral Development Corporation Ltd. (GMDCLTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 736.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -2.41
- **Avg P&L per closed trade:** -0.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 387.20 | 417.99 | 418.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 381.85 | 417.31 | 417.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 385.35 | 377.44 | 392.57 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 11:15:00 | 428.85 | 397.41 | 397.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 13:15:00 | 429.50 | 398.04 | 397.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 12:15:00 | 404.15 | 405.03 | 401.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-08 09:15:00 | 414.95 | 405.27 | 401.67 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-08 14:15:00 | 400.50 | 405.66 | 401.96 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 14:15:00 | 348.00 | 401.21 | 401.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 334.95 | 400.10 | 400.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 09:15:00 | 393.00 | 392.79 | 396.47 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 15:15:00 | 424.05 | 398.58 | 398.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 428.35 | 398.88 | 398.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 397.90 | 406.57 | 403.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-24 09:15:00 | 409.90 | 405.08 | 402.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 409.90 | 405.08 | 402.59 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-24 13:15:00 | 418.20 | 405.32 | 402.77 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-26 14:15:00 | 402.75 | 405.74 | 403.18 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 372.15 | 401.14 | 401.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 366.00 | 400.79 | 401.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 12:15:00 | 376.70 | 372.43 | 380.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-18 13:15:00 | 369.70 | 372.61 | 379.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 357.50 | 350.76 | 361.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-23 12:15:00 | 361.95 | 350.87 | 361.26 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 326.35 | 286.95 | 286.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 332.55 | 301.37 | 295.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 390.60 | 392.00 | 367.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-18 11:15:00 | 396.00 | 387.86 | 371.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 387.80 | 406.30 | 387.47 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-01 13:15:00 | 385.70 | 405.92 | 387.46 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 489.10 | 537.94 | 538.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 486.25 | 537.43 | 537.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 11:15:00 | 536.40 | 531.87 | 534.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-16 09:15:00 | 524.50 | 531.92 | 534.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-23 09:15:00 | 539.80 | 527.03 | 531.60 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 13:15:00 | 596.05 | 535.58 | 535.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 14:15:00 | 599.80 | 536.22 | 535.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 11:15:00 | 563.90 | 565.45 | 552.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-13 09:15:00 | 572.60 | 564.38 | 552.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 556.10 | 564.98 | 554.40 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 548.60 | 564.82 | 554.37 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 547.00 | 561.51 | 561.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 545.15 | 561.34 | 561.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 563.00 | 556.21 | 558.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 10:15:00 | 544.80 | 556.25 | 558.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-20 09:15:00 | 576.10 | 555.60 | 558.22 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 578.60 | 560.50 | 560.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 585.15 | 561.82 | 561.17 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-08 09:15:00 | 414.95 | 2024-05-08 14:15:00 | 400.50 | EXIT_EMA400 | -14.45 |
| BUY | 2024-07-24 09:15:00 | 409.90 | 2024-07-26 14:15:00 | 402.75 | EXIT_EMA400 | -7.15 |
| BUY | 2024-07-24 13:15:00 | 418.20 | 2024-07-26 14:15:00 | 402.75 | EXIT_EMA400 | -15.45 |
| SELL | 2024-09-18 13:15:00 | 369.70 | 2024-10-04 09:15:00 | 338.98 | TARGET | 30.72 |
| BUY | 2025-07-18 11:15:00 | 396.00 | 2025-07-23 09:15:00 | 470.51 | TARGET | 74.51 |
| SELL | 2025-12-16 09:15:00 | 524.50 | 2025-12-23 09:15:00 | 539.80 | EXIT_EMA400 | -15.30 |
| BUY | 2026-01-13 09:15:00 | 572.60 | 2026-01-19 09:15:00 | 548.60 | EXIT_EMA400 | -24.00 |
| SELL | 2026-03-19 10:15:00 | 544.80 | 2026-03-20 09:15:00 | 576.10 | EXIT_EMA400 | -31.30 |
