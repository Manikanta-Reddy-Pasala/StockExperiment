# NATCO Pharma Ltd. (NATCOPHARM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1092.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 291.87
- **Avg P&L per closed trade:** 41.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 15:15:00 | 1320.90 | 1399.66 | 1399.87 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 15:15:00 | 1448.00 | 1399.48 | 1399.35 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 13:15:00 | 1369.30 | 1399.72 | 1399.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 1356.95 | 1399.30 | 1399.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 1438.00 | 1383.49 | 1390.50 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 1442.50 | 1396.78 | 1396.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 1457.70 | 1397.39 | 1396.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 1412.00 | 1427.10 | 1414.87 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1321.35 | 1406.10 | 1406.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1313.75 | 1403.62 | 1404.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1265.65 | 1254.98 | 1306.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-12 09:15:00 | 1198.75 | 1263.72 | 1303.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-24 10:15:00 | 933.15 | 833.36 | 916.75 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 970.55 | 887.30 | 887.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 14:15:00 | 973.25 | 888.16 | 887.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 962.35 | 967.41 | 938.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-30 10:15:00 | 971.00 | 965.87 | 939.35 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-01 09:15:00 | 940.00 | 965.23 | 940.69 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 888.70 | 927.61 | 927.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 885.00 | 927.19 | 927.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 892.65 | 877.56 | 895.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-22 13:15:00 | 868.70 | 877.87 | 893.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 832.60 | 823.12 | 839.92 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-20 09:15:00 | 862.00 | 824.03 | 839.80 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 925.00 | 851.13 | 850.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 932.80 | 851.94 | 851.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 868.15 | 873.33 | 863.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-11 09:15:00 | 903.50 | 873.54 | 863.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 891.20 | 904.29 | 889.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-09 15:15:00 | 888.35 | 904.13 | 889.43 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 830.75 | 879.29 | 879.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 818.45 | 878.22 | 878.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 855.75 | 852.30 | 862.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 12:15:00 | 848.00 | 852.34 | 862.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 840.00 | 852.24 | 862.03 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-12 15:15:00 | 833.55 | 852.06 | 861.88 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-16 09:15:00 | 917.00 | 850.99 | 860.95 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 943.55 | 869.18 | 868.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 976.85 | 870.25 | 869.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 12:15:00 | 939.80 | 941.05 | 912.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-16 14:15:00 | 946.70 | 941.06 | 912.67 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-12 09:15:00 | 1198.75 | 2025-02-14 09:15:00 | 885.92 | TARGET | 312.83 |
| BUY | 2025-07-30 10:15:00 | 971.00 | 2025-08-01 09:15:00 | 940.00 | EXIT_EMA400 | -31.00 |
| SELL | 2025-09-22 13:15:00 | 868.70 | 2025-09-29 14:15:00 | 793.16 | TARGET | 75.54 |
| BUY | 2025-12-11 09:15:00 | 903.50 | 2026-01-09 15:15:00 | 888.35 | EXIT_EMA400 | -15.15 |
| SELL | 2026-02-12 12:15:00 | 848.00 | 2026-02-16 09:15:00 | 917.00 | EXIT_EMA400 | -69.00 |
| SELL | 2026-02-12 15:15:00 | 833.55 | 2026-02-16 09:15:00 | 917.00 | EXIT_EMA400 | -83.45 |
| BUY | 2026-03-16 14:15:00 | 946.70 | 2026-04-07 09:15:00 | 1048.79 | TARGET | 102.09 |
