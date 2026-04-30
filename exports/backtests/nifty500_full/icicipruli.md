# ICICI Prudential Life Insurance Company Ltd. (ICICIPRULI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 513.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 1
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 114.79
- **Avg P&L per closed trade:** 14.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 11:15:00 | 533.00 | 554.41 | 554.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 09:15:00 | 519.50 | 551.78 | 553.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 13:15:00 | 536.05 | 533.06 | 541.06 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 562.70 | 544.45 | 544.38 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 11:15:00 | 518.15 | 544.64 | 544.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 12:15:00 | 517.05 | 544.37 | 544.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 14:15:00 | 534.90 | 534.16 | 538.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-01 12:15:00 | 531.20 | 534.13 | 538.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 537.00 | 534.00 | 538.12 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-01-03 11:15:00 | 538.55 | 534.09 | 538.12 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 12:15:00 | 562.35 | 525.22 | 525.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-06 13:15:00 | 570.00 | 525.67 | 525.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 590.00 | 594.62 | 572.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-23 13:15:00 | 596.40 | 593.01 | 573.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 578.75 | 592.91 | 573.76 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-25 09:15:00 | 571.25 | 591.74 | 573.82 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-06 11:15:00 | 562.85 | 572.38 | 572.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-06 12:15:00 | 562.05 | 572.27 | 572.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 09:15:00 | 574.55 | 571.86 | 572.15 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 09:15:00 | 578.25 | 572.47 | 572.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 13:15:00 | 592.50 | 574.10 | 573.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 622.90 | 624.77 | 607.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-24 09:15:00 | 683.65 | 625.73 | 608.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 729.80 | 754.41 | 729.20 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-08 14:15:00 | 743.00 | 754.06 | 729.27 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 733.95 | 751.42 | 731.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-22 14:15:00 | 732.10 | 749.28 | 733.80 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 11:15:00 | 693.65 | 729.53 | 729.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 09:15:00 | 685.95 | 727.71 | 728.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 13:15:00 | 676.15 | 674.08 | 689.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 660.80 | 673.93 | 689.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 595.80 | 569.26 | 594.01 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 612.85 | 592.09 | 592.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 617.20 | 593.14 | 592.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 627.00 | 629.26 | 616.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 14:15:00 | 629.65 | 629.11 | 616.92 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-18 09:15:00 | 636.80 | 650.31 | 637.51 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 615.00 | 630.43 | 630.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 13:15:00 | 612.35 | 629.35 | 629.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 10:15:00 | 628.90 | 625.96 | 628.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-13 13:15:00 | 621.45 | 625.88 | 628.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-14 09:15:00 | 629.90 | 625.82 | 627.93 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 629.45 | 608.34 | 608.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 630.85 | 609.52 | 608.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 14:15:00 | 611.00 | 611.84 | 610.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-24 13:15:00 | 613.90 | 611.81 | 610.30 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-24 14:15:00 | 607.70 | 611.77 | 610.28 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 15:15:00 | 603.90 | 644.09 | 644.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 599.35 | 639.78 | 642.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 565.10 | 564.34 | 591.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-15 11:15:00 | 562.45 | 564.34 | 591.50 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-01 12:15:00 | 531.20 | 2024-01-03 11:15:00 | 538.55 | EXIT_EMA400 | -7.35 |
| BUY | 2024-04-23 13:15:00 | 596.40 | 2024-04-25 09:15:00 | 571.25 | EXIT_EMA400 | -25.15 |
| BUY | 2024-07-24 09:15:00 | 683.65 | 2024-10-22 14:15:00 | 732.10 | EXIT_EMA400 | 48.45 |
| BUY | 2024-10-08 14:15:00 | 743.00 | 2024-10-22 14:15:00 | 732.10 | EXIT_EMA400 | -10.90 |
| SELL | 2025-01-06 10:15:00 | 660.80 | 2025-01-22 09:15:00 | 574.59 | TARGET | 86.21 |
| BUY | 2025-06-13 14:15:00 | 629.65 | 2025-07-07 12:15:00 | 667.83 | TARGET | 38.18 |
| SELL | 2025-08-13 13:15:00 | 621.45 | 2025-08-14 09:15:00 | 629.90 | EXIT_EMA400 | -8.45 |
| BUY | 2025-11-24 13:15:00 | 613.90 | 2025-11-24 14:15:00 | 607.70 | EXIT_EMA400 | -6.20 |
