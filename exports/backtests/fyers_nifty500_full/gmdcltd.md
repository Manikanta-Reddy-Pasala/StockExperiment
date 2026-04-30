# Gujarat Mineral Development Corporation Ltd. (GMDCLTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 737.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 24.37
- **Avg P&L per closed trade:** 4.06

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 375.40 | 397.54 | 397.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 13:15:00 | 371.95 | 397.28 | 397.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 12:15:00 | 376.70 | 372.39 | 379.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-18 13:15:00 | 369.70 | 372.57 | 379.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 358.00 | 351.17 | 362.75 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-17 14:15:00 | 353.40 | 351.35 | 362.55 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 357.50 | 350.76 | 360.94 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-23 12:15:00 | 361.90 | 350.87 | 360.95 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 326.35 | 286.95 | 286.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 332.55 | 301.39 | 295.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 390.60 | 392.01 | 367.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-18 11:15:00 | 396.25 | 387.88 | 371.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 389.05 | 406.67 | 387.46 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-01 13:15:00 | 385.70 | 405.92 | 387.47 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 489.10 | 537.93 | 538.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 486.25 | 537.41 | 537.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 11:15:00 | 536.40 | 531.78 | 534.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-16 09:15:00 | 524.70 | 531.83 | 534.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-23 09:15:00 | 540.00 | 526.96 | 531.54 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 13:15:00 | 596.05 | 535.52 | 535.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 14:15:00 | 599.60 | 536.15 | 535.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 11:15:00 | 563.90 | 565.39 | 552.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-13 09:15:00 | 572.60 | 564.33 | 552.93 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 556.10 | 564.93 | 554.35 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 548.60 | 564.77 | 554.32 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 550.50 | 562.16 | 562.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 545.15 | 561.67 | 561.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 563.00 | 556.42 | 559.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 10:15:00 | 544.80 | 556.46 | 559.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-20 09:15:00 | 576.40 | 555.80 | 558.61 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 09:15:00 | 574.85 | 561.13 | 561.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 585.15 | 561.92 | 561.47 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-18 13:15:00 | 369.70 | 2024-10-04 09:15:00 | 341.17 | TARGET | 28.53 |
| SELL | 2024-10-17 14:15:00 | 353.40 | 2024-10-23 12:15:00 | 361.90 | EXIT_EMA400 | -8.50 |
| BUY | 2025-07-18 11:15:00 | 396.25 | 2025-07-23 09:15:00 | 471.49 | TARGET | 75.24 |
| SELL | 2025-12-16 09:15:00 | 524.70 | 2025-12-23 09:15:00 | 540.00 | EXIT_EMA400 | -15.30 |
| BUY | 2026-01-13 09:15:00 | 572.60 | 2026-01-19 09:15:00 | 548.60 | EXIT_EMA400 | -24.00 |
| SELL | 2026-03-19 10:15:00 | 544.80 | 2026-03-20 09:15:00 | 576.40 | EXIT_EMA400 | -31.60 |
