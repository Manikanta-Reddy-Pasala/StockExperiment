# Can Fin Homes Ltd. (CANFINHOME.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 865.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 19 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT3 | 12 |
| ENTRY1 | 11 |
| ENTRY2 | 9 |
| EXIT | 11 |

## P&L

- **Trades closed:** 20
- **Trades open at end:** 0
- **Winners / losers:** 10 / 10
- **Target hits / EMA400 exits:** 10 / 10
- **Total realized P&L (per unit):** 59.49
- **Avg P&L per closed trade:** 2.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 694.85 | 756.41 | 756.65 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 10:15:00 | 776.25 | 756.42 | 756.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 13:15:00 | 780.50 | 757.04 | 756.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 14:15:00 | 756.40 | 757.64 | 756.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-08 10:15:00 | 758.65 | 757.64 | 756.98 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-11-09 12:15:00 | 754.60 | 758.01 | 757.20 | Close below EMA400 |

### Cycle 3 — SELL (started 2023-11-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 13:15:00 | 750.80 | 756.48 | 756.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 14:15:00 | 749.55 | 756.41 | 756.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 763.00 | 756.41 | 756.45 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 11:15:00 | 768.10 | 756.55 | 756.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 13:15:00 | 769.65 | 756.78 | 756.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 12:15:00 | 759.50 | 760.48 | 758.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-29 13:15:00 | 766.10 | 760.12 | 758.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 766.10 | 760.12 | 758.63 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-11-29 14:15:00 | 770.90 | 760.22 | 758.69 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-12-11 14:15:00 | 755.00 | 777.95 | 769.22 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 12:15:00 | 739.15 | 769.52 | 769.56 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 12:15:00 | 783.35 | 769.54 | 769.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-01 14:15:00 | 793.80 | 769.91 | 769.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 14:15:00 | 783.00 | 788.07 | 780.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-13 10:15:00 | 797.30 | 788.12 | 780.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 790.70 | 788.45 | 780.59 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-14 13:15:00 | 797.95 | 788.64 | 780.84 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 786.10 | 793.32 | 784.79 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-22 09:15:00 | 789.00 | 793.28 | 784.81 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 787.05 | 794.38 | 786.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-28 10:15:00 | 781.80 | 794.15 | 786.43 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 11:15:00 | 730.25 | 781.96 | 782.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 09:15:00 | 717.15 | 779.29 | 780.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 770.70 | 759.89 | 768.67 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2024-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 15:15:00 | 810.10 | 775.60 | 775.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 09:15:00 | 826.90 | 776.11 | 775.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 777.15 | 779.23 | 777.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-15 11:15:00 | 782.70 | 779.24 | 777.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 782.70 | 779.24 | 777.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-15 14:15:00 | 777.00 | 779.29 | 777.47 | Close below EMA400 |

### Cycle 9 — SELL (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 12:15:00 | 761.10 | 775.81 | 775.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-22 14:15:00 | 753.25 | 775.46 | 775.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 09:15:00 | 782.75 | 766.26 | 770.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-06 09:15:00 | 761.45 | 767.08 | 770.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 761.45 | 767.08 | 770.64 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-07 09:15:00 | 749.30 | 766.68 | 770.31 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 756.00 | 756.68 | 763.59 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-17 11:15:00 | 755.15 | 756.66 | 763.51 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-06-03 12:15:00 | 756.30 | 747.18 | 756.07 | Close above EMA400 |

### Cycle 10 — BUY (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 11:15:00 | 825.85 | 761.08 | 760.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 12:15:00 | 827.10 | 761.74 | 761.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 10:15:00 | 859.70 | 859.72 | 829.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-28 10:15:00 | 871.00 | 837.30 | 830.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 846.70 | 856.06 | 843.91 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-11 10:15:00 | 861.50 | 856.12 | 844.00 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-09-19 10:15:00 | 846.40 | 868.68 | 853.30 | Close below EMA400 |

### Cycle 11 — SELL (started 2024-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 13:15:00 | 819.10 | 858.09 | 858.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 814.10 | 857.65 | 857.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 863.05 | 849.76 | 853.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 12:15:00 | 849.10 | 849.94 | 853.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 849.10 | 849.94 | 853.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-25 13:15:00 | 844.55 | 849.89 | 853.55 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 841.65 | 841.05 | 847.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-05 12:15:00 | 839.70 | 841.03 | 847.55 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-19 13:15:00 | 660.00 | 623.36 | 657.69 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 740.00 | 670.01 | 669.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 12:15:00 | 746.10 | 670.76 | 670.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 09:15:00 | 766.00 | 769.97 | 743.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 11:15:00 | 782.50 | 769.73 | 745.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 777.10 | 796.20 | 776.42 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-24 11:15:00 | 773.05 | 795.97 | 776.40 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 12:15:00 | 759.00 | 765.58 | 765.59 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 11:15:00 | 778.10 | 765.59 | 765.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 13:15:00 | 778.45 | 765.82 | 765.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 14:15:00 | 766.30 | 768.02 | 766.90 | EMA200 retest candle locked |

### Cycle 15 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 729.75 | 765.58 | 765.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 726.25 | 765.19 | 765.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 15:15:00 | 759.00 | 757.41 | 761.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-05 10:15:00 | 749.20 | 758.18 | 761.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-16 09:15:00 | 774.90 | 753.42 | 758.05 | Close above EMA400 |

### Cycle 16 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 779.50 | 761.73 | 761.73 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 740.45 | 761.62 | 761.68 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 793.20 | 761.86 | 761.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 796.90 | 764.14 | 762.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 13:15:00 | 900.10 | 908.37 | 879.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-30 13:15:00 | 932.00 | 908.47 | 880.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 894.00 | 918.49 | 892.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-09 09:15:00 | 905.40 | 918.36 | 892.31 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-09 13:15:00 | 889.25 | 917.59 | 892.44 | Close below EMA400 |

### Cycle 19 — SELL (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 14:15:00 | 820.40 | 897.67 | 897.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 12:15:00 | 814.65 | 893.84 | 895.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 876.65 | 874.51 | 884.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-13 09:15:00 | 860.65 | 874.42 | 884.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-18 11:15:00 | 884.00 | 866.67 | 879.18 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-08 10:15:00 | 758.65 | 2023-11-08 12:15:00 | 763.67 | TARGET | 5.02 |
| BUY | 2023-11-29 13:15:00 | 766.10 | 2023-11-30 15:15:00 | 788.51 | TARGET | 22.41 |
| BUY | 2023-11-29 14:15:00 | 770.90 | 2023-12-01 10:15:00 | 807.53 | TARGET | 36.63 |
| BUY | 2024-02-22 09:15:00 | 789.00 | 2024-02-23 09:15:00 | 801.58 | TARGET | 12.58 |
| BUY | 2024-02-13 10:15:00 | 797.30 | 2024-02-28 10:15:00 | 781.80 | EXIT_EMA400 | -15.50 |
| BUY | 2024-02-14 13:15:00 | 797.95 | 2024-02-28 10:15:00 | 781.80 | EXIT_EMA400 | -16.15 |
| BUY | 2024-04-15 11:15:00 | 782.70 | 2024-04-15 14:15:00 | 777.00 | EXIT_EMA400 | -5.70 |
| SELL | 2024-05-06 09:15:00 | 761.45 | 2024-05-07 12:15:00 | 733.89 | TARGET | 27.56 |
| SELL | 2024-05-17 11:15:00 | 755.15 | 2024-05-29 10:15:00 | 730.06 | TARGET | 25.09 |
| SELL | 2024-05-07 09:15:00 | 749.30 | 2024-06-03 12:15:00 | 756.30 | EXIT_EMA400 | -7.00 |
| BUY | 2024-09-11 10:15:00 | 861.50 | 2024-09-13 09:15:00 | 914.01 | TARGET | 52.51 |
| BUY | 2024-08-28 10:15:00 | 871.00 | 2024-09-19 10:15:00 | 846.40 | EXIT_EMA400 | -24.60 |
| SELL | 2024-11-25 12:15:00 | 849.10 | 2024-11-26 09:15:00 | 835.61 | TARGET | 13.49 |
| SELL | 2024-11-25 13:15:00 | 844.55 | 2024-11-27 09:15:00 | 817.54 | TARGET | 27.01 |
| SELL | 2024-12-05 12:15:00 | 839.70 | 2024-12-06 09:15:00 | 816.16 | TARGET | 23.54 |
| BUY | 2025-06-24 11:15:00 | 782.50 | 2025-07-24 11:15:00 | 773.05 | EXIT_EMA400 | -9.45 |
| SELL | 2025-09-05 10:15:00 | 749.20 | 2025-09-16 09:15:00 | 774.90 | EXIT_EMA400 | -25.70 |
| BUY | 2025-12-30 13:15:00 | 932.00 | 2026-01-09 13:15:00 | 889.25 | EXIT_EMA400 | -42.75 |
| BUY | 2026-01-09 09:15:00 | 905.40 | 2026-01-09 13:15:00 | 889.25 | EXIT_EMA400 | -16.15 |
| SELL | 2026-03-13 09:15:00 | 860.65 | 2026-03-18 11:15:00 | 884.00 | EXIT_EMA400 | -23.35 |
