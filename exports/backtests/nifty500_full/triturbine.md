# Triveni Turbine Ltd. (TRITURBINE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 573.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 65.28
- **Avg P&L per closed trade:** 13.06

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 408.65 | 394.54 | 394.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 10:15:00 | 411.00 | 394.70 | 394.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 14:15:00 | 414.95 | 415.04 | 407.24 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 09:15:00 | 377.10 | 401.39 | 401.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 10:15:00 | 374.45 | 401.12 | 401.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 13:15:00 | 378.55 | 374.66 | 385.71 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 11:15:00 | 416.55 | 391.78 | 391.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 14:15:00 | 418.35 | 394.07 | 392.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 09:15:00 | 402.80 | 408.41 | 401.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-14 09:15:00 | 420.40 | 407.89 | 402.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 404.50 | 408.55 | 402.95 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-12-20 09:15:00 | 402.05 | 408.25 | 402.99 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 10:15:00 | 377.30 | 404.25 | 404.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 12:15:00 | 375.30 | 403.69 | 404.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 395.90 | 393.85 | 398.57 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-02-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 14:15:00 | 469.60 | 403.28 | 403.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-28 09:15:00 | 488.80 | 433.00 | 421.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 451.60 | 455.36 | 437.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-19 09:15:00 | 492.70 | 457.87 | 441.07 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 11:15:00 | 537.50 | 568.68 | 542.95 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 634.00 | 706.29 | 706.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 623.15 | 704.04 | 705.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 11:15:00 | 691.25 | 687.87 | 696.16 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 854.45 | 703.64 | 703.03 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 684.50 | 734.69 | 734.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 682.65 | 733.68 | 734.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 10:15:00 | 578.90 | 571.33 | 619.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 15:15:00 | 560.80 | 572.72 | 617.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 555.90 | 561.25 | 594.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-04 09:15:00 | 518.60 | 556.84 | 587.88 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-05-12 09:15:00 | 567.10 | 523.58 | 547.57 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 586.70 | 562.26 | 562.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 591.75 | 562.83 | 562.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 11:15:00 | 632.75 | 632.88 | 612.45 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 528.90 | 602.34 | 602.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 516.20 | 597.93 | 600.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 537.30 | 536.54 | 557.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-24 15:15:00 | 521.50 | 536.43 | 551.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 535.00 | 526.13 | 536.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-29 15:15:00 | 537.95 | 526.34 | 536.42 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 584.80 | 486.88 | 486.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 586.40 | 501.88 | 494.60 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-14 09:15:00 | 420.40 | 2023-12-20 09:15:00 | 402.05 | EXIT_EMA400 | -18.35 |
| BUY | 2024-03-19 09:15:00 | 492.70 | 2024-05-17 09:15:00 | 647.58 | TARGET | 154.88 |
| SELL | 2025-03-10 15:15:00 | 560.80 | 2025-05-12 09:15:00 | 567.10 | EXIT_EMA400 | -6.30 |
| SELL | 2025-04-04 09:15:00 | 518.60 | 2025-05-12 09:15:00 | 567.10 | EXIT_EMA400 | -48.50 |
| SELL | 2025-09-24 15:15:00 | 521.50 | 2025-10-29 15:15:00 | 537.95 | EXIT_EMA400 | -16.45 |
