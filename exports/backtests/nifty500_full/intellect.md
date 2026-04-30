# Intellect Design Arena Ltd. (INTELLECT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 745.35
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
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -115.13
- **Avg P&L per closed trade:** -12.79

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 14:15:00 | 652.45 | 680.34 | 680.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 11:15:00 | 651.55 | 679.22 | 679.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 10:15:00 | 679.35 | 675.49 | 677.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-09 11:15:00 | 672.40 | 675.46 | 677.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 672.40 | 675.46 | 677.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-09 12:15:00 | 667.95 | 675.39 | 677.64 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-15 09:15:00 | 686.20 | 674.01 | 676.73 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 13:15:00 | 698.15 | 679.09 | 679.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 09:15:00 | 708.95 | 681.11 | 680.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 13:15:00 | 898.25 | 904.30 | 855.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-21 09:15:00 | 905.00 | 904.22 | 856.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-04-10 15:15:00 | 1007.00 | 1061.33 | 1007.55 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 880.55 | 999.98 | 1000.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 14:15:00 | 878.75 | 997.59 | 999.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 950.00 | 936.49 | 960.95 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 13:15:00 | 1053.30 | 977.61 | 977.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 11:15:00 | 1060.85 | 989.04 | 983.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 10:15:00 | 1052.00 | 1056.46 | 1030.17 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 14:15:00 | 901.55 | 1013.98 | 1014.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 898.30 | 983.98 | 988.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 764.45 | 763.28 | 821.82 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 1008.00 | 839.36 | 839.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 14:15:00 | 1019.35 | 841.16 | 840.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 890.90 | 895.58 | 875.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-20 10:15:00 | 920.70 | 895.70 | 876.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 882.85 | 897.23 | 878.54 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-22 10:15:00 | 876.15 | 897.02 | 878.52 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 15:15:00 | 810.50 | 867.79 | 867.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 810.00 | 861.72 | 864.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 12:15:00 | 706.85 | 706.57 | 753.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-28 13:15:00 | 698.85 | 710.70 | 748.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 719.35 | 687.07 | 725.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 13:15:00 | 727.55 | 687.81 | 725.02 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 13:15:00 | 789.50 | 747.26 | 747.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 09:15:00 | 799.60 | 748.46 | 747.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1078.00 | 1106.71 | 1005.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 14:15:00 | 1116.80 | 1101.85 | 1012.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 12:15:00 | 1090.90 | 1149.14 | 1091.17 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 927.20 | 1056.55 | 1056.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 921.70 | 1047.18 | 1051.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 994.50 | 980.35 | 1006.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-29 15:15:00 | 956.70 | 1011.71 | 1016.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 11:15:00 | 1009.10 | 998.44 | 1007.48 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1206.40 | 1007.69 | 1007.66 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 1007.50 | 1050.13 | 1050.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 1003.00 | 1049.22 | 1049.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 682.45 | 676.79 | 742.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 12:15:00 | 673.80 | 677.19 | 739.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 707.50 | 676.29 | 731.56 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 11:15:00 | 697.90 | 676.79 | 731.27 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-28 09:15:00 | 725.05 | 683.67 | 722.99 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-09 11:15:00 | 672.40 | 2023-11-15 09:15:00 | 686.20 | EXIT_EMA400 | -13.80 |
| SELL | 2023-11-09 12:15:00 | 667.95 | 2023-11-15 09:15:00 | 686.20 | EXIT_EMA400 | -18.25 |
| BUY | 2024-02-21 09:15:00 | 905.00 | 2024-02-27 10:15:00 | 1051.87 | TARGET | 146.87 |
| BUY | 2025-01-20 10:15:00 | 920.70 | 2025-01-22 10:15:00 | 876.15 | EXIT_EMA400 | -44.55 |
| SELL | 2025-03-28 13:15:00 | 698.85 | 2025-04-15 13:15:00 | 727.55 | EXIT_EMA400 | -28.70 |
| BUY | 2025-06-25 14:15:00 | 1116.80 | 2025-07-25 12:15:00 | 1090.90 | EXIT_EMA400 | -25.90 |
| SELL | 2025-09-29 15:15:00 | 956.70 | 2025-10-10 11:15:00 | 1009.10 | EXIT_EMA400 | -52.40 |
| SELL | 2026-04-09 12:15:00 | 673.80 | 2026-04-28 09:15:00 | 725.05 | EXIT_EMA400 | -51.25 |
| SELL | 2026-04-16 11:15:00 | 697.90 | 2026-04-28 09:15:00 | 725.05 | EXIT_EMA400 | -27.15 |
