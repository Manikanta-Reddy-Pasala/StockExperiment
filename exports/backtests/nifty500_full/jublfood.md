# Jubilant Foodworks Ltd. (JUBLFOOD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 478.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 5 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / EMA400 exits:** 0 / 9
- **Total realized P&L (per unit):** -181.80
- **Avg P&L per closed trade:** -20.20

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 11:15:00 | 510.60 | 514.66 | 514.67 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 13:15:00 | 516.50 | 514.69 | 514.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 09:15:00 | 521.50 | 514.77 | 514.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 556.70 | 556.92 | 544.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-04 14:15:00 | 561.65 | 556.82 | 545.47 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-08 10:15:00 | 543.15 | 556.61 | 545.92 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 12:15:00 | 508.90 | 538.97 | 539.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 11:15:00 | 507.65 | 537.29 | 538.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 09:15:00 | 508.10 | 503.82 | 517.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-22 09:15:00 | 486.90 | 503.58 | 516.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-30 13:15:00 | 466.45 | 450.89 | 464.93 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 517.00 | 470.00 | 469.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 13:15:00 | 518.10 | 470.48 | 470.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 09:15:00 | 664.00 | 667.77 | 636.64 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 15:15:00 | 575.30 | 625.19 | 625.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 574.05 | 621.28 | 623.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 616.05 | 610.73 | 617.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-11 11:15:00 | 603.85 | 610.66 | 616.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 603.85 | 610.66 | 616.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-11 12:15:00 | 599.55 | 610.55 | 616.86 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-12 09:15:00 | 642.15 | 610.62 | 616.76 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 10:15:00 | 648.20 | 620.52 | 620.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 12:15:00 | 654.70 | 626.13 | 623.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 14:15:00 | 707.25 | 708.61 | 680.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-14 10:15:00 | 713.75 | 708.64 | 681.28 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 684.40 | 704.90 | 683.68 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-21 11:15:00 | 683.60 | 704.68 | 683.68 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 13:15:00 | 625.55 | 681.07 | 681.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 12:15:00 | 613.10 | 677.77 | 679.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 12:15:00 | 646.50 | 644.46 | 659.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-19 14:15:00 | 634.25 | 644.30 | 659.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-24 13:15:00 | 665.40 | 642.70 | 656.77 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 696.90 | 664.33 | 664.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 702.40 | 664.71 | 664.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 676.05 | 690.72 | 680.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 14:15:00 | 695.95 | 688.86 | 680.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 693.25 | 690.31 | 681.66 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-15 09:15:00 | 679.30 | 690.23 | 681.71 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 660.40 | 677.06 | 677.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 658.90 | 676.88 | 677.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 678.00 | 675.50 | 676.33 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 693.50 | 677.25 | 677.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 698.45 | 678.52 | 677.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 677.45 | 680.74 | 679.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 10:15:00 | 681.00 | 680.06 | 678.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 681.00 | 680.06 | 678.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 12:15:00 | 677.45 | 680.07 | 678.79 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 656.45 | 684.50 | 684.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 650.65 | 683.36 | 684.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 649.00 | 647.46 | 660.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 11:15:00 | 636.20 | 647.00 | 659.20 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 654.55 | 643.44 | 655.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-04 11:15:00 | 657.55 | 643.70 | 655.41 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-04 14:15:00 | 561.65 | 2024-01-08 10:15:00 | 543.15 | EXIT_EMA400 | -18.50 |
| SELL | 2024-02-22 09:15:00 | 486.90 | 2024-04-30 13:15:00 | 466.45 | EXIT_EMA400 | 20.45 |
| SELL | 2024-11-11 11:15:00 | 603.85 | 2024-11-12 09:15:00 | 642.15 | EXIT_EMA400 | -38.30 |
| SELL | 2024-11-11 12:15:00 | 599.55 | 2024-11-12 09:15:00 | 642.15 | EXIT_EMA400 | -42.60 |
| BUY | 2025-01-14 10:15:00 | 713.75 | 2025-01-21 11:15:00 | 683.60 | EXIT_EMA400 | -30.15 |
| SELL | 2025-03-19 14:15:00 | 634.25 | 2025-03-24 13:15:00 | 665.40 | EXIT_EMA400 | -31.15 |
| BUY | 2025-05-12 14:15:00 | 695.95 | 2025-05-15 09:15:00 | 679.30 | EXIT_EMA400 | -16.65 |
| BUY | 2025-06-13 10:15:00 | 681.00 | 2025-06-13 12:15:00 | 677.45 | EXIT_EMA400 | -3.55 |
| SELL | 2025-08-28 11:15:00 | 636.20 | 2025-09-04 11:15:00 | 657.55 | EXIT_EMA400 | -21.35 |
