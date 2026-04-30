# CCL Products (I) Ltd. (CCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1135.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 9 |
| ENTRY1 | 10 |
| ENTRY2 | 5 |
| EXIT | 10 |

## P&L

- **Trades closed:** 15
- **Trades open at end:** 0
- **Winners / losers:** 4 / 11
- **Target hits / EMA400 exits:** 3 / 12
- **Total realized P&L (per unit):** 5.30
- **Avg P&L per closed trade:** 0.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 671.80 | 632.66 | 632.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 15:15:00 | 682.00 | 637.42 | 635.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 10:15:00 | 630.00 | 639.09 | 636.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-26 09:15:00 | 644.10 | 636.04 | 634.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-09-27 14:15:00 | 633.60 | 636.72 | 635.21 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 14:15:00 | 618.90 | 637.46 | 637.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 09:15:00 | 610.15 | 637.00 | 637.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 11:15:00 | 631.10 | 630.54 | 633.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-07 09:15:00 | 623.65 | 630.60 | 633.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 623.65 | 630.60 | 633.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-07 10:15:00 | 622.05 | 630.51 | 633.60 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-08 11:15:00 | 634.20 | 630.09 | 633.26 | Close above EMA400 |

### Cycle 3 — BUY (started 2023-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 11:15:00 | 662.45 | 634.37 | 634.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 09:15:00 | 664.10 | 636.98 | 635.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 641.60 | 642.87 | 639.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-13 12:15:00 | 646.55 | 642.87 | 639.25 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-12-14 14:15:00 | 638.00 | 643.40 | 639.68 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 13:15:00 | 622.60 | 637.30 | 637.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 11:15:00 | 621.30 | 635.84 | 636.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 12:15:00 | 636.00 | 635.51 | 636.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-10 09:15:00 | 631.45 | 635.49 | 636.36 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-01-10 13:15:00 | 639.25 | 635.41 | 636.30 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 13:15:00 | 666.50 | 636.72 | 636.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 671.65 | 637.68 | 637.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 15:15:00 | 639.95 | 644.95 | 641.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-06 10:15:00 | 649.75 | 644.37 | 641.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 643.60 | 646.65 | 642.56 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-09 12:15:00 | 650.00 | 646.69 | 642.62 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 643.80 | 647.02 | 642.89 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-12 11:15:00 | 641.55 | 646.97 | 642.88 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 625.70 | 641.57 | 641.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 12:15:00 | 623.45 | 641.24 | 641.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 14:15:00 | 590.15 | 587.60 | 601.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-02 14:15:00 | 582.40 | 587.83 | 600.47 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 587.65 | 575.59 | 587.81 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-05-27 10:15:00 | 591.75 | 575.75 | 587.83 | Close above EMA400 |

### Cycle 7 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 11:15:00 | 604.70 | 591.79 | 591.78 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 15:15:00 | 587.65 | 591.93 | 591.95 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 595.85 | 591.97 | 591.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 14:15:00 | 602.60 | 592.24 | 592.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 589.00 | 592.31 | 592.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-10 14:15:00 | 594.65 | 592.34 | 592.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 594.65 | 592.34 | 592.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-19 14:15:00 | 589.30 | 595.58 | 594.01 | Close below EMA400 |

### Cycle 10 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 628.45 | 679.18 | 679.30 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 713.65 | 676.15 | 676.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 741.15 | 684.48 | 680.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 751.30 | 754.85 | 730.26 | EMA200 retest candle locked |

### Cycle 12 — SELL (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 09:15:00 | 634.80 | 720.28 | 720.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 10:15:00 | 627.00 | 703.82 | 711.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 659.05 | 658.83 | 682.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 12:15:00 | 647.80 | 661.13 | 681.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-11 10:15:00 | 608.20 | 575.50 | 599.83 | Close above EMA400 |

### Cycle 13 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 772.10 | 612.65 | 611.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 13:15:00 | 819.25 | 673.46 | 646.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 803.70 | 803.82 | 747.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 09:15:00 | 816.65 | 801.62 | 752.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 880.20 | 894.24 | 871.96 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-19 10:15:00 | 883.60 | 893.41 | 872.19 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 887.55 | 893.04 | 872.43 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-22 10:15:00 | 895.80 | 892.97 | 872.70 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 878.80 | 895.38 | 876.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-26 14:15:00 | 893.25 | 895.36 | 877.01 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 877.50 | 895.04 | 877.03 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-29 10:15:00 | 867.95 | 894.77 | 876.99 | Close below EMA400 |

### Cycle 14 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 841.45 | 865.52 | 865.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 832.40 | 864.49 | 865.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 858.85 | 856.71 | 860.51 | EMA200 retest candle locked |

### Cycle 15 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 988.15 | 864.31 | 863.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 1012.45 | 870.25 | 866.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 970.40 | 971.09 | 934.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-03 09:15:00 | 996.60 | 971.89 | 936.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-05 11:15:00 | 936.70 | 973.35 | 939.99 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-26 09:15:00 | 644.10 | 2023-09-27 14:15:00 | 633.60 | EXIT_EMA400 | -10.50 |
| SELL | 2023-11-07 09:15:00 | 623.65 | 2023-11-08 11:15:00 | 634.20 | EXIT_EMA400 | -10.55 |
| SELL | 2023-11-07 10:15:00 | 622.05 | 2023-11-08 11:15:00 | 634.20 | EXIT_EMA400 | -12.15 |
| BUY | 2023-12-13 12:15:00 | 646.55 | 2023-12-14 14:15:00 | 638.00 | EXIT_EMA400 | -8.55 |
| SELL | 2024-01-10 09:15:00 | 631.45 | 2024-01-10 13:15:00 | 639.25 | EXIT_EMA400 | -7.80 |
| BUY | 2024-02-06 10:15:00 | 649.75 | 2024-02-12 11:15:00 | 641.55 | EXIT_EMA400 | -8.20 |
| BUY | 2024-02-09 12:15:00 | 650.00 | 2024-02-12 11:15:00 | 641.55 | EXIT_EMA400 | -8.45 |
| SELL | 2024-05-02 14:15:00 | 582.40 | 2024-05-27 10:15:00 | 591.75 | EXIT_EMA400 | -9.35 |
| BUY | 2024-07-10 14:15:00 | 594.65 | 2024-07-11 09:15:00 | 602.12 | TARGET | 7.47 |
| SELL | 2025-02-06 12:15:00 | 647.80 | 2025-03-17 14:15:00 | 546.89 | TARGET | 100.91 |
| BUY | 2025-09-19 10:15:00 | 883.60 | 2025-09-24 09:15:00 | 917.82 | TARGET | 34.22 |
| BUY | 2025-06-23 09:15:00 | 816.65 | 2025-09-29 10:15:00 | 867.95 | EXIT_EMA400 | 51.30 |
| BUY | 2025-09-22 10:15:00 | 895.80 | 2025-09-29 10:15:00 | 867.95 | EXIT_EMA400 | -27.85 |
| BUY | 2025-09-26 14:15:00 | 893.25 | 2025-09-29 10:15:00 | 867.95 | EXIT_EMA400 | -25.30 |
| BUY | 2025-12-03 09:15:00 | 996.60 | 2025-12-05 11:15:00 | 936.70 | EXIT_EMA400 | -59.90 |
