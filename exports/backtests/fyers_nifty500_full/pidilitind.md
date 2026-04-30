# Pidilite Industries Ltd. (PIDILITIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1378.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 7 |
| ENTRY1 | 5 |
| ENTRY2 | 5 |
| EXIT | 5 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / EMA400 exits:** 5 / 5
- **Total realized P&L (per unit):** 99.67
- **Avg P&L per closed trade:** 9.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 1561.00 | 1586.16 | 1586.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 1556.70 | 1585.86 | 1586.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 1592.23 | 1583.37 | 1584.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 09:15:00 | 1576.23 | 1583.41 | 1584.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1576.23 | 1583.41 | 1584.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-08 10:15:00 | 1568.45 | 1582.80 | 1584.36 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-02 11:15:00 | 1557.60 | 1539.71 | 1557.12 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 14:15:00 | 1513.15 | 1425.94 | 1425.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 1521.30 | 1427.73 | 1426.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 10:15:00 | 1473.05 | 1476.78 | 1457.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-08 12:15:00 | 1482.20 | 1476.82 | 1457.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-13 09:15:00 | 1493.75 | 1520.24 | 1499.46 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 1452.20 | 1500.28 | 1500.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1442.70 | 1498.79 | 1499.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1482.95 | 1478.34 | 1488.21 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1540.65 | 1495.59 | 1495.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 1564.35 | 1514.12 | 1505.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 09:15:00 | 1513.90 | 1520.05 | 1509.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-01 09:15:00 | 1549.50 | 1520.57 | 1510.40 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1526.70 | 1536.07 | 1524.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-19 09:15:00 | 1533.35 | 1535.89 | 1524.41 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1524.95 | 1535.78 | 1524.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-19 11:15:00 | 1524.10 | 1535.67 | 1524.41 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1451.70 | 1516.54 | 1516.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 1445.00 | 1505.99 | 1509.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 1489.30 | 1487.56 | 1498.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-13 14:15:00 | 1481.00 | 1487.48 | 1497.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1491.70 | 1485.53 | 1495.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-18 14:15:00 | 1489.40 | 1485.57 | 1495.73 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1492.10 | 1485.72 | 1495.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-19 10:15:00 | 1481.60 | 1485.68 | 1495.63 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1472.50 | 1483.66 | 1493.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-08 13:15:00 | 1451.20 | 1478.04 | 1487.31 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-16 09:15:00 | 1484.30 | 1474.66 | 1483.77 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 1497.70 | 1471.73 | 1471.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 10:15:00 | 1508.30 | 1472.60 | 1472.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1464.50 | 1476.06 | 1473.91 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 10:15:00 | 1426.50 | 1471.86 | 1471.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1374.20 | 1467.48 | 1469.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1371.00 | 1364.17 | 1403.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 14:15:00 | 1355.40 | 1364.12 | 1402.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 1390.00 | 1356.65 | 1391.57 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 13:15:00 | 1392.60 | 1357.35 | 1391.57 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 09:15:00 | 1576.23 | 2024-11-11 09:15:00 | 1550.74 | TARGET | 25.49 |
| SELL | 2024-11-08 10:15:00 | 1568.45 | 2024-11-12 14:15:00 | 1520.71 | TARGET | 47.74 |
| BUY | 2025-05-08 12:15:00 | 1482.20 | 2025-05-13 09:15:00 | 1555.82 | TARGET | 73.62 |
| BUY | 2025-09-01 09:15:00 | 1549.50 | 2025-09-19 11:15:00 | 1524.10 | EXIT_EMA400 | -25.40 |
| BUY | 2025-09-19 09:15:00 | 1533.35 | 2025-09-19 11:15:00 | 1524.10 | EXIT_EMA400 | -9.25 |
| SELL | 2025-11-18 14:15:00 | 1489.40 | 2025-11-21 10:15:00 | 1470.42 | TARGET | 18.98 |
| SELL | 2025-11-19 10:15:00 | 1481.60 | 2025-12-09 09:15:00 | 1439.51 | TARGET | 42.09 |
| SELL | 2025-11-13 14:15:00 | 1481.00 | 2025-12-16 09:15:00 | 1484.30 | EXIT_EMA400 | -3.30 |
| SELL | 2025-12-08 13:15:00 | 1451.20 | 2025-12-16 09:15:00 | 1484.30 | EXIT_EMA400 | -33.10 |
| SELL | 2026-04-08 14:15:00 | 1355.40 | 2026-04-17 13:15:00 | 1392.60 | EXIT_EMA400 | -37.20 |
