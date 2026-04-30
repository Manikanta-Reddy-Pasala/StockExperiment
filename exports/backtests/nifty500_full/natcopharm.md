# NATCO Pharma Ltd. (NATCOPHARM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1095.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 3 |
| EXIT | 7 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 5 / 6
- **Target hits / EMA400 exits:** 5 / 6
- **Total realized P&L (per unit):** 329.70
- **Avg P&L per closed trade:** 29.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 10:15:00 | 739.70 | 826.00 | 826.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 14:15:00 | 732.00 | 804.57 | 814.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 805.90 | 794.12 | 806.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-24 12:15:00 | 778.05 | 793.95 | 805.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 791.05 | 791.29 | 802.14 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-12-05 10:15:00 | 784.10 | 791.39 | 801.77 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-12-22 09:15:00 | 803.40 | 781.72 | 792.16 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 11:15:00 | 843.95 | 798.90 | 798.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 12:15:00 | 851.00 | 799.42 | 799.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 14:15:00 | 818.45 | 818.75 | 810.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-18 10:15:00 | 821.60 | 818.80 | 810.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 832.25 | 845.24 | 830.48 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-13 10:15:00 | 843.90 | 844.59 | 830.66 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 972.75 | 997.41 | 972.60 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-09 14:15:00 | 970.00 | 996.89 | 972.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 1324.95 | 1398.99 | 1399.31 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 15:15:00 | 1448.00 | 1399.05 | 1398.96 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 13:15:00 | 1369.30 | 1399.43 | 1399.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 1357.70 | 1399.02 | 1399.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 1438.00 | 1383.32 | 1390.26 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 1442.50 | 1396.61 | 1396.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 1457.10 | 1397.21 | 1396.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 1412.25 | 1426.99 | 1414.69 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1322.40 | 1406.02 | 1406.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1313.75 | 1403.53 | 1404.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1266.00 | 1260.27 | 1310.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-12 09:15:00 | 1198.75 | 1267.44 | 1306.72 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-24 10:15:00 | 933.15 | 833.49 | 917.49 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 14:15:00 | 973.25 | 888.14 | 887.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 980.65 | 889.88 | 888.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 962.35 | 967.35 | 938.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-30 10:15:00 | 971.00 | 965.82 | 939.38 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-01 09:15:00 | 940.00 | 965.17 | 940.71 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 888.50 | 927.57 | 927.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 885.00 | 927.15 | 927.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 892.85 | 877.57 | 895.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-22 13:15:00 | 868.70 | 877.87 | 893.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 833.10 | 823.11 | 839.91 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-20 09:15:00 | 862.00 | 824.02 | 839.80 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 925.00 | 851.11 | 850.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 932.80 | 851.92 | 851.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 868.15 | 873.23 | 863.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-11 09:15:00 | 903.50 | 873.46 | 863.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 890.90 | 904.26 | 889.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-09 15:15:00 | 888.35 | 904.10 | 889.40 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 830.75 | 879.31 | 879.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 818.45 | 878.24 | 878.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 855.75 | 853.81 | 863.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 12:15:00 | 848.00 | 853.71 | 863.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 840.00 | 853.58 | 863.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-12 15:15:00 | 833.55 | 853.38 | 862.98 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-16 09:15:00 | 917.00 | 852.21 | 862.00 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 943.55 | 869.92 | 869.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 976.85 | 870.99 | 870.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 12:15:00 | 939.80 | 941.37 | 912.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-16 14:15:00 | 946.70 | 941.38 | 913.20 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-24 12:15:00 | 778.05 | 2023-12-22 09:15:00 | 803.40 | EXIT_EMA400 | -25.35 |
| SELL | 2023-12-05 10:15:00 | 784.10 | 2023-12-22 09:15:00 | 803.40 | EXIT_EMA400 | -19.30 |
| BUY | 2024-01-18 10:15:00 | 821.60 | 2024-01-24 09:15:00 | 854.85 | TARGET | 33.25 |
| BUY | 2024-02-13 10:15:00 | 843.90 | 2024-02-14 12:15:00 | 883.63 | TARGET | 39.73 |
| SELL | 2025-02-12 09:15:00 | 1198.75 | 2025-02-14 10:15:00 | 874.84 | TARGET | 323.91 |
| BUY | 2025-07-30 10:15:00 | 971.00 | 2025-08-01 09:15:00 | 940.00 | EXIT_EMA400 | -31.00 |
| SELL | 2025-09-22 13:15:00 | 868.70 | 2025-09-29 14:15:00 | 793.13 | TARGET | 75.57 |
| BUY | 2025-12-11 09:15:00 | 903.50 | 2026-01-09 15:15:00 | 888.35 | EXIT_EMA400 | -15.15 |
| SELL | 2026-02-12 12:15:00 | 848.00 | 2026-02-16 09:15:00 | 917.00 | EXIT_EMA400 | -69.00 |
| SELL | 2026-02-12 15:15:00 | 833.55 | 2026-02-16 09:15:00 | 917.00 | EXIT_EMA400 | -83.45 |
| BUY | 2026-03-16 14:15:00 | 946.70 | 2026-04-07 09:15:00 | 1047.19 | TARGET | 100.49 |
