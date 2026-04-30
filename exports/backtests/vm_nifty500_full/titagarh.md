# Titagarh Rail Systems Ltd. (TITAGARH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-19 09:15:00 → 2026-04-30 15:15:00 (4912 bars)
- **Last close:** 768.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 7 |
| ENTRY1 | 8 |
| ENTRY2 | 4 |
| EXIT | 8 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / EMA400 exits:** 2 / 10
- **Total realized P&L (per unit):** -219.31
- **Avg P&L per closed trade:** -18.28

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 887.50 | 983.25 | 983.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 10:15:00 | 881.00 | 982.23 | 983.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 10:15:00 | 927.50 | 916.88 | 942.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-15 09:15:00 | 911.00 | 932.96 | 944.11 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-16 10:15:00 | 950.90 | 932.07 | 943.21 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 10:15:00 | 1029.95 | 952.17 | 952.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 11:15:00 | 1044.00 | 953.09 | 952.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-05 09:15:00 | 1044.90 | 1192.19 | 1107.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-10 09:15:00 | 1311.95 | 1188.26 | 1114.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1585.10 | 1616.36 | 1469.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-24 09:15:00 | 1637.95 | 1615.85 | 1471.92 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-02 13:15:00 | 1500.80 | 1615.11 | 1505.42 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 13:15:00 | 1374.45 | 1457.67 | 1457.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 11:15:00 | 1370.00 | 1449.38 | 1453.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 12:15:00 | 1207.80 | 1193.55 | 1263.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 14:15:00 | 1152.00 | 1194.79 | 1250.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 1234.25 | 1163.81 | 1213.43 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 14:15:00 | 1318.55 | 1234.20 | 1233.79 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 13:15:00 | 1182.40 | 1233.49 | 1233.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 14:15:00 | 1177.00 | 1232.93 | 1233.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 792.50 | 776.38 | 873.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 722.30 | 789.99 | 853.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 795.90 | 757.63 | 797.51 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-14 14:15:00 | 805.35 | 758.11 | 797.55 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 938.80 | 825.87 | 825.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 942.75 | 864.00 | 847.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 882.00 | 884.82 | 861.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 906.40 | 881.01 | 864.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 907.95 | 925.39 | 904.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 890.50 | 924.65 | 904.37 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 834.75 | 890.39 | 890.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 11:15:00 | 826.55 | 889.22 | 890.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 858.90 | 857.58 | 871.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 10:15:00 | 845.50 | 857.36 | 870.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 869.20 | 857.60 | 870.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-25 12:15:00 | 861.85 | 857.64 | 870.53 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 868.20 | 858.06 | 870.30 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-26 13:15:00 | 865.95 | 858.14 | 870.28 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 869.30 | 858.36 | 870.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-28 09:15:00 | 852.95 | 858.31 | 870.18 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-09 10:15:00 | 874.70 | 852.51 | 863.87 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 934.65 | 873.45 | 873.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 937.70 | 874.09 | 873.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 888.85 | 895.49 | 886.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 10:15:00 | 896.15 | 894.96 | 886.09 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 885.75 | 894.87 | 886.09 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 11:15:00 | 846.65 | 885.88 | 885.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 846.00 | 885.48 | 885.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 886.95 | 882.70 | 884.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-13 12:15:00 | 866.10 | 882.23 | 884.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 863.70 | 881.53 | 883.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-17 09:15:00 | 893.70 | 880.78 | 883.18 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-15 09:15:00 | 911.00 | 2024-04-16 10:15:00 | 950.90 | EXIT_EMA400 | -39.90 |
| BUY | 2024-06-10 09:15:00 | 1311.95 | 2024-08-02 13:15:00 | 1500.80 | EXIT_EMA400 | 188.85 |
| BUY | 2024-07-24 09:15:00 | 1637.95 | 2024-08-02 13:15:00 | 1500.80 | EXIT_EMA400 | -137.15 |
| SELL | 2024-11-08 14:15:00 | 1152.00 | 2024-11-28 09:15:00 | 1234.25 | EXIT_EMA400 | -82.25 |
| SELL | 2025-04-07 09:15:00 | 722.30 | 2025-05-14 14:15:00 | 805.35 | EXIT_EMA400 | -83.05 |
| BUY | 2025-06-24 09:15:00 | 906.40 | 2025-07-25 09:15:00 | 890.50 | EXIT_EMA400 | -15.90 |
| SELL | 2025-08-26 13:15:00 | 865.95 | 2025-08-28 09:15:00 | 852.96 | TARGET | 12.99 |
| SELL | 2025-08-25 12:15:00 | 861.85 | 2025-08-29 12:15:00 | 835.80 | TARGET | 26.05 |
| SELL | 2025-08-22 10:15:00 | 845.50 | 2025-09-09 10:15:00 | 874.70 | EXIT_EMA400 | -29.20 |
| SELL | 2025-08-28 09:15:00 | 852.95 | 2025-09-09 10:15:00 | 874.70 | EXIT_EMA400 | -21.75 |
| BUY | 2025-09-29 10:15:00 | 896.15 | 2025-09-29 11:15:00 | 885.75 | EXIT_EMA400 | -10.40 |
| SELL | 2025-11-13 12:15:00 | 866.10 | 2025-11-17 09:15:00 | 893.70 | EXIT_EMA400 | -27.60 |
