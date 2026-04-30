# K.P.R. Mill Ltd. (KPRMILL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 936.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -331.79
- **Avg P&L per closed trade:** -47.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 13:15:00 | 777.35 | 798.14 | 798.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-19 14:15:00 | 771.70 | 797.88 | 798.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 14:15:00 | 791.30 | 783.79 | 790.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-07 14:15:00 | 779.25 | 785.00 | 789.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-02-26 10:15:00 | 773.05 | 756.22 | 770.98 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 15:15:00 | 828.00 | 776.26 | 776.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 09:15:00 | 837.50 | 776.87 | 776.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 09:15:00 | 834.45 | 834.91 | 816.60 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 770.15 | 809.34 | 809.51 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 824.05 | 809.72 | 809.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 844.80 | 810.49 | 810.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 863.00 | 865.20 | 845.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-08 15:15:00 | 875.70 | 865.31 | 845.77 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-15 09:15:00 | 846.50 | 865.25 | 848.41 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 15:15:00 | 848.90 | 858.99 | 859.03 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 11:15:00 | 889.45 | 859.25 | 859.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 12:15:00 | 897.30 | 859.63 | 859.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 14:15:00 | 925.90 | 926.04 | 905.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-31 09:15:00 | 942.00 | 915.90 | 904.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-11-05 14:15:00 | 903.60 | 918.14 | 906.80 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 877.20 | 971.49 | 971.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 846.45 | 938.61 | 952.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 09:15:00 | 872.25 | 862.76 | 900.14 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 926.45 | 909.98 | 909.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 13:15:00 | 930.70 | 910.18 | 910.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 1110.00 | 1116.97 | 1057.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-03 09:15:00 | 1135.70 | 1117.48 | 1060.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1078.30 | 1119.10 | 1076.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-16 10:15:00 | 1076.20 | 1118.67 | 1076.88 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 987.50 | 1116.49 | 1117.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 975.60 | 1111.45 | 1114.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1023.20 | 1022.32 | 1051.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-09 10:15:00 | 1009.20 | 1055.89 | 1061.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1043.00 | 1041.23 | 1052.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-20 09:15:00 | 1021.00 | 1041.09 | 1051.88 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 1075.50 | 1040.29 | 1050.99 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 09:15:00 | 1080.30 | 1057.18 | 1057.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 1091.60 | 1059.49 | 1058.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 1063.90 | 1074.63 | 1067.64 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 984.60 | 1061.46 | 1061.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 978.10 | 1060.63 | 1061.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 12:15:00 | 911.50 | 897.75 | 945.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-29 11:15:00 | 885.00 | 899.35 | 943.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 1000.60 | 894.67 | 937.17 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 12:15:00 | 934.85 | 891.97 | 891.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 13:15:00 | 937.40 | 892.42 | 892.11 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-07 14:15:00 | 779.25 | 2024-02-12 10:15:00 | 747.54 | TARGET | 31.71 |
| BUY | 2024-07-08 15:15:00 | 875.70 | 2024-07-15 09:15:00 | 846.50 | EXIT_EMA400 | -29.20 |
| BUY | 2024-10-31 09:15:00 | 942.00 | 2024-11-05 14:15:00 | 903.60 | EXIT_EMA400 | -38.40 |
| BUY | 2025-06-03 09:15:00 | 1135.70 | 2025-06-16 10:15:00 | 1076.20 | EXIT_EMA400 | -59.50 |
| SELL | 2025-10-09 10:15:00 | 1009.20 | 2025-10-23 09:15:00 | 1075.50 | EXIT_EMA400 | -66.30 |
| SELL | 2025-10-20 09:15:00 | 1021.00 | 2025-10-23 09:15:00 | 1075.50 | EXIT_EMA400 | -54.50 |
| SELL | 2026-01-29 11:15:00 | 885.00 | 2026-02-03 09:15:00 | 1000.60 | EXIT_EMA400 | -115.60 |
