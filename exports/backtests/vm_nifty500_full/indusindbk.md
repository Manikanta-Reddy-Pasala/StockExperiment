# IndusInd Bank Ltd. (INDUSINDBK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 916.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** -2.86
- **Avg P&L per closed trade:** -0.32

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 14:15:00 | 1475.15 | 1532.12 | 1532.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-15 09:15:00 | 1469.70 | 1531.00 | 1531.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 14:15:00 | 1521.95 | 1520.52 | 1525.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-22 10:15:00 | 1486.15 | 1520.01 | 1525.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-03-01 13:15:00 | 1516.20 | 1504.32 | 1515.65 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 1553.75 | 1518.07 | 1518.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1558.50 | 1518.81 | 1518.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 1513.00 | 1529.35 | 1524.25 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 13:15:00 | 1471.05 | 1519.50 | 1519.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 09:15:00 | 1469.00 | 1515.48 | 1517.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 12:15:00 | 1507.90 | 1506.41 | 1512.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-03 11:15:00 | 1487.10 | 1506.51 | 1512.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1462.90 | 1454.51 | 1476.97 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-29 09:15:00 | 1441.30 | 1454.89 | 1476.40 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-06-03 09:15:00 | 1517.90 | 1455.73 | 1474.68 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 15:15:00 | 1527.15 | 1483.64 | 1483.53 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 15:15:00 | 1456.00 | 1483.60 | 1483.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 09:15:00 | 1440.55 | 1483.17 | 1483.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 14:15:00 | 1458.80 | 1457.60 | 1467.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-18 15:15:00 | 1452.00 | 1457.55 | 1467.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-28 11:15:00 | 1419.00 | 1391.63 | 1415.39 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 1491.00 | 1426.86 | 1426.54 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 1349.20 | 1428.84 | 1429.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 14:15:00 | 1342.05 | 1419.65 | 1424.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 15:15:00 | 996.00 | 992.78 | 1069.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 978.50 | 993.26 | 1066.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-03 10:15:00 | 1019.55 | 970.46 | 1014.48 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 866.20 | 822.24 | 822.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 882.00 | 828.18 | 825.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 845.10 | 850.14 | 839.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-21 13:15:00 | 852.95 | 850.07 | 839.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 835.75 | 849.73 | 840.86 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 796.10 | 833.76 | 833.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 780.55 | 824.54 | 828.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 09:15:00 | 752.70 | 750.31 | 770.36 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 799.00 | 771.99 | 771.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 819.85 | 773.01 | 772.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 830.35 | 833.02 | 813.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-11 09:15:00 | 838.25 | 833.07 | 813.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 886.55 | 888.23 | 862.27 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-27 14:15:00 | 894.60 | 888.21 | 862.90 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 882.85 | 924.00 | 902.74 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 823.90 | 887.83 | 887.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 817.35 | 886.51 | 887.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 838.75 | 835.30 | 856.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 12:15:00 | 831.40 | 835.28 | 856.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.28 | 852.47 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-22 10:15:00 | 1486.15 | 2024-03-01 13:15:00 | 1516.20 | EXIT_EMA400 | -30.05 |
| SELL | 2024-05-03 11:15:00 | 1487.10 | 2024-05-09 12:15:00 | 1412.23 | TARGET | 74.87 |
| SELL | 2024-05-29 09:15:00 | 1441.30 | 2024-06-03 09:15:00 | 1517.90 | EXIT_EMA400 | -76.60 |
| SELL | 2024-07-18 15:15:00 | 1452.00 | 2024-07-23 12:15:00 | 1404.58 | TARGET | 47.42 |
| SELL | 2025-01-06 10:15:00 | 978.50 | 2025-02-03 10:15:00 | 1019.55 | EXIT_EMA400 | -41.05 |
| BUY | 2025-07-21 13:15:00 | 852.95 | 2025-07-25 09:15:00 | 835.75 | EXIT_EMA400 | -17.20 |
| BUY | 2025-12-11 09:15:00 | 838.25 | 2026-01-06 09:15:00 | 911.49 | TARGET | 73.24 |
| BUY | 2026-01-27 14:15:00 | 894.60 | 2026-03-09 09:15:00 | 882.85 | EXIT_EMA400 | -11.75 |
| SELL | 2026-04-08 12:15:00 | 831.40 | 2026-04-16 09:15:00 | 853.15 | EXIT_EMA400 | -21.75 |
