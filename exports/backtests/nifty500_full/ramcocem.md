# The Ramco Cements Ltd. (RAMCOCEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 935.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 8 |
| ENTRY1 | 11 |
| ENTRY2 | 4 |
| EXIT | 9 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 1
- **Winners / losers:** 6 / 8
- **Target hits / EMA400 exits:** 4 / 10
- **Total realized P&L (per unit):** 80.36
- **Avg P&L per closed trade:** 5.74

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 919.70 | 893.27 | 893.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-04 09:15:00 | 929.80 | 894.83 | 894.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 15:15:00 | 981.85 | 981.98 | 958.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-22 09:15:00 | 987.45 | 982.03 | 959.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 15:15:00 | 960.10 | 979.60 | 959.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-11-28 10:15:00 | 965.00 | 979.28 | 959.96 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-12-21 09:15:00 | 985.30 | 1008.96 | 987.09 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 10:15:00 | 977.15 | 986.11 | 986.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 12:15:00 | 973.20 | 985.89 | 986.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 09:15:00 | 990.85 | 985.82 | 985.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-30 14:15:00 | 984.15 | 985.94 | 986.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 984.15 | 985.94 | 986.03 | EMA400 retest candle locked |

### Cycle 3 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 1009.00 | 986.16 | 986.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 10:15:00 | 1013.90 | 986.43 | 986.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 14:15:00 | 985.60 | 988.62 | 987.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-02 11:15:00 | 995.05 | 988.69 | 987.47 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 988.60 | 988.85 | 987.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-05 14:15:00 | 981.00 | 988.91 | 987.65 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 901.85 | 986.70 | 986.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 886.65 | 980.76 | 983.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 12:15:00 | 839.30 | 837.80 | 878.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-15 09:15:00 | 826.65 | 841.05 | 869.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 789.75 | 773.86 | 800.12 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-07 09:15:00 | 821.15 | 774.87 | 799.99 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 15:15:00 | 846.10 | 817.78 | 817.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 10:15:00 | 853.40 | 818.39 | 818.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 820.50 | 831.39 | 825.64 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 790.00 | 820.92 | 821.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 784.75 | 816.33 | 818.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 15:15:00 | 810.00 | 806.13 | 812.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-08 14:15:00 | 793.20 | 813.00 | 814.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-19 09:15:00 | 817.30 | 807.59 | 811.56 | Close above EMA400 |

### Cycle 7 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 831.35 | 814.31 | 814.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 10:15:00 | 842.70 | 814.97 | 814.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 10:15:00 | 828.85 | 831.30 | 824.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-20 11:15:00 | 837.20 | 831.07 | 824.69 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 845.85 | 853.95 | 842.65 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-18 11:15:00 | 853.35 | 853.87 | 842.72 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-21 09:15:00 | 840.15 | 853.74 | 842.93 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 863.15 | 932.89 | 933.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 852.00 | 903.20 | 914.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 10:15:00 | 874.00 | 869.66 | 890.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 11:15:00 | 865.45 | 870.52 | 889.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 865.45 | 857.45 | 875.82 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 15:15:00 | 862.50 | 858.09 | 875.61 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-27 12:15:00 | 879.20 | 858.58 | 874.91 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 948.00 | 886.51 | 886.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 955.15 | 891.63 | 889.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 935.20 | 937.19 | 919.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 949.50 | 937.37 | 920.53 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-08 09:15:00 | 1070.60 | 1135.85 | 1090.65 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 1045.00 | 1075.09 | 1075.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1024.10 | 1063.98 | 1068.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 1030.40 | 1027.19 | 1044.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-17 09:15:00 | 1020.45 | 1027.30 | 1043.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-21 13:15:00 | 1043.25 | 1026.87 | 1042.31 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 1061.40 | 1030.46 | 1030.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 1074.70 | 1034.06 | 1032.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1057.50 | 1059.07 | 1048.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-20 11:15:00 | 1070.70 | 1059.58 | 1049.07 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1065.90 | 1059.86 | 1049.42 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-21 09:15:00 | 1074.60 | 1060.01 | 1049.54 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1055.00 | 1062.65 | 1051.94 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-27 09:15:00 | 1049.80 | 1062.52 | 1051.93 | Close below EMA400 |

### Cycle 12 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 978.80 | 1083.97 | 1083.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 946.70 | 1078.61 | 1081.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 987.00 | 985.80 | 1022.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-23 10:15:00 | 965.00 | 991.04 | 1015.38 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-28 10:15:00 | 965.00 | 2023-11-29 09:15:00 | 980.12 | TARGET | 15.12 |
| BUY | 2023-11-22 09:15:00 | 987.45 | 2023-12-21 09:15:00 | 985.30 | EXIT_EMA400 | -2.15 |
| SELL | 2024-01-30 14:15:00 | 984.15 | 2024-02-05 14:15:00 | 981.00 | EXIT_EMA400 | 3.15 |
| BUY | 2024-02-02 11:15:00 | 995.05 | 2024-02-05 14:15:00 | 981.00 | EXIT_EMA400 | -14.05 |
| SELL | 2024-04-15 09:15:00 | 826.65 | 2024-06-07 09:15:00 | 821.15 | EXIT_EMA400 | 5.50 |
| SELL | 2024-08-08 14:15:00 | 793.20 | 2024-08-19 09:15:00 | 817.30 | EXIT_EMA400 | -24.10 |
| BUY | 2024-09-20 11:15:00 | 837.20 | 2024-09-27 09:15:00 | 874.74 | TARGET | 37.54 |
| BUY | 2024-10-18 11:15:00 | 853.35 | 2024-10-21 09:15:00 | 840.15 | EXIT_EMA400 | -13.20 |
| SELL | 2025-03-10 11:15:00 | 865.45 | 2025-03-12 12:15:00 | 794.60 | TARGET | 70.85 |
| SELL | 2025-03-25 15:15:00 | 862.50 | 2025-03-27 12:15:00 | 879.20 | EXIT_EMA400 | -16.70 |
| BUY | 2025-05-12 09:15:00 | 949.50 | 2025-06-09 11:15:00 | 1036.40 | TARGET | 86.90 |
| SELL | 2025-10-17 09:15:00 | 1020.45 | 2025-10-21 13:15:00 | 1043.25 | EXIT_EMA400 | -22.80 |
| BUY | 2026-01-20 11:15:00 | 1070.70 | 2026-01-27 09:15:00 | 1049.80 | EXIT_EMA400 | -20.90 |
| BUY | 2026-01-21 09:15:00 | 1074.60 | 2026-01-27 09:15:00 | 1049.80 | EXIT_EMA400 | -24.80 |
