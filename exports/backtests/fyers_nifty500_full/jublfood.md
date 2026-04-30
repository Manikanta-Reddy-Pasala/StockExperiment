# Jubilant Foodworks Ltd. (JUBLFOOD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 479.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -187.45
- **Avg P&L per closed trade:** -26.78

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 15:15:00 | 578.20 | 625.21 | 625.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 574.30 | 621.31 | 623.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 616.05 | 610.20 | 616.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-11 11:15:00 | 603.85 | 610.13 | 616.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 603.85 | 610.13 | 616.50 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-11 12:15:00 | 599.55 | 610.03 | 616.41 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-12 09:15:00 | 641.80 | 610.12 | 616.33 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 10:15:00 | 647.80 | 620.34 | 620.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 12:15:00 | 654.70 | 625.99 | 623.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 14:15:00 | 707.20 | 708.52 | 680.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-14 10:15:00 | 713.75 | 708.56 | 681.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 689.90 | 705.06 | 683.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-21 12:15:00 | 682.00 | 704.42 | 683.58 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 13:15:00 | 625.55 | 681.69 | 681.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 12:15:00 | 613.10 | 678.35 | 680.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 12:15:00 | 646.50 | 644.72 | 659.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-19 14:15:00 | 634.35 | 644.56 | 659.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-24 13:15:00 | 665.40 | 642.91 | 657.23 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 12:15:00 | 702.40 | 664.86 | 664.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 13:15:00 | 705.30 | 671.77 | 668.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 676.10 | 690.70 | 680.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 14:15:00 | 695.95 | 688.84 | 680.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 693.25 | 690.28 | 681.79 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-15 09:15:00 | 679.30 | 690.19 | 681.83 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 14:15:00 | 660.20 | 677.25 | 677.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 656.05 | 676.52 | 676.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 678.00 | 675.51 | 676.42 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 695.75 | 677.42 | 677.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 698.45 | 678.51 | 677.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 677.45 | 680.71 | 679.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 11:15:00 | 683.40 | 680.06 | 678.85 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-13 12:15:00 | 677.45 | 680.04 | 678.84 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 656.45 | 684.49 | 684.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 655.50 | 683.67 | 684.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 649.00 | 647.44 | 660.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 12:15:00 | 635.70 | 646.87 | 659.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 654.55 | 643.42 | 655.38 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-04 11:15:00 | 657.55 | 643.68 | 655.39 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-11 11:15:00 | 603.85 | 2024-11-12 09:15:00 | 641.80 | EXIT_EMA400 | -37.95 |
| SELL | 2024-11-11 12:15:00 | 599.55 | 2024-11-12 09:15:00 | 641.80 | EXIT_EMA400 | -42.25 |
| BUY | 2025-01-14 10:15:00 | 713.75 | 2025-01-21 12:15:00 | 682.00 | EXIT_EMA400 | -31.75 |
| SELL | 2025-03-19 14:15:00 | 634.35 | 2025-03-24 13:15:00 | 665.40 | EXIT_EMA400 | -31.05 |
| BUY | 2025-05-12 14:15:00 | 695.95 | 2025-05-15 09:15:00 | 679.30 | EXIT_EMA400 | -16.65 |
| BUY | 2025-06-13 11:15:00 | 683.40 | 2025-06-13 12:15:00 | 677.45 | EXIT_EMA400 | -5.95 |
| SELL | 2025-08-28 12:15:00 | 635.70 | 2025-09-04 11:15:00 | 657.55 | EXIT_EMA400 | -21.85 |
