# Patanjali Foods Ltd. (PATANJALI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 459.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 5.46
- **Avg P&L per closed trade:** 1.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 12:15:00 | 461.70 | 515.19 | 515.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 11:15:00 | 451.80 | 505.81 | 510.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 15:15:00 | 468.50 | 468.48 | 483.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-09 10:15:00 | 462.72 | 481.25 | 486.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 476.17 | 471.37 | 478.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-05-27 15:15:00 | 480.38 | 471.65 | 478.45 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 510.65 | 479.89 | 479.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 512.68 | 481.36 | 480.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 516.87 | 518.12 | 503.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-18 09:15:00 | 527.92 | 518.21 | 504.17 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-19 10:15:00 | 589.07 | 616.57 | 590.35 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 15:15:00 | 586.33 | 594.57 | 594.59 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 603.23 | 594.63 | 594.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 605.88 | 594.74 | 594.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 13:15:00 | 602.20 | 602.44 | 598.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-15 09:15:00 | 612.80 | 601.89 | 598.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 607.75 | 604.83 | 601.11 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-22 14:15:00 | 622.97 | 605.00 | 601.23 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-01-24 13:15:00 | 600.93 | 606.10 | 602.05 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-03-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 10:15:00 | 571.00 | 601.50 | 601.56 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 604.48 | 596.62 | 596.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 619.38 | 597.06 | 596.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 13:15:00 | 623.63 | 626.48 | 614.99 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 572.37 | 608.90 | 608.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 554.83 | 594.50 | 600.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 557.07 | 556.10 | 568.70 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 650.53 | 578.40 | 578.15 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 584.00 | 595.80 | 595.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 579.00 | 594.65 | 595.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 595.85 | 593.09 | 594.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-24 10:15:00 | 584.25 | 593.08 | 594.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 591.95 | 592.26 | 593.74 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-29 09:15:00 | 598.45 | 592.25 | 593.69 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-09 10:15:00 | 462.72 | 2024-05-27 15:15:00 | 480.38 | EXIT_EMA400 | -17.67 |
| BUY | 2024-07-18 09:15:00 | 527.92 | 2024-08-07 11:15:00 | 599.15 | TARGET | 71.23 |
| BUY | 2025-01-15 09:15:00 | 612.80 | 2025-01-24 13:15:00 | 600.93 | EXIT_EMA400 | -11.87 |
| BUY | 2025-01-22 14:15:00 | 622.97 | 2025-01-24 13:15:00 | 600.93 | EXIT_EMA400 | -22.03 |
| SELL | 2025-10-24 10:15:00 | 584.25 | 2025-10-29 09:15:00 | 598.45 | EXIT_EMA400 | -14.20 |
