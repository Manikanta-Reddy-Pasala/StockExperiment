# Havells India Ltd. (HAVELLS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1240.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 381.14
- **Avg P&L per closed trade:** 42.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 15:15:00 | 1338.90 | 1310.64 | 1310.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 1346.85 | 1311.00 | 1310.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 14:15:00 | 1375.00 | 1379.05 | 1355.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-29 09:15:00 | 1386.35 | 1379.09 | 1356.15 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-10-18 09:15:00 | 1355.60 | 1388.57 | 1369.74 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-27 13:15:00 | 1261.70 | 1354.75 | 1355.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-27 14:15:00 | 1259.50 | 1353.80 | 1354.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 10:15:00 | 1296.45 | 1294.80 | 1316.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-28 09:15:00 | 1287.70 | 1296.07 | 1314.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 1307.20 | 1295.18 | 1312.25 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-12-01 11:15:00 | 1312.35 | 1295.51 | 1312.25 | Close above EMA400 |

### Cycle 3 — BUY (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 12:15:00 | 1370.05 | 1321.85 | 1321.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 09:15:00 | 1389.20 | 1340.10 | 1331.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 13:15:00 | 1376.05 | 1382.48 | 1360.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-16 11:15:00 | 1393.55 | 1353.61 | 1349.93 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 1484.50 | 1498.32 | 1464.78 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-19 11:15:00 | 1493.95 | 1498.02 | 1464.96 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-19 10:15:00 | 1791.60 | 1863.35 | 1800.38 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 15:15:00 | 1723.30 | 1899.08 | 1899.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1694.25 | 1897.04 | 1898.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1725.40 | 1715.57 | 1780.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-20 12:15:00 | 1688.70 | 1725.66 | 1755.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-03 09:15:00 | 1686.15 | 1593.69 | 1647.14 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 1610.70 | 1549.68 | 1549.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 1619.90 | 1550.95 | 1550.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 12:15:00 | 1550.00 | 1563.94 | 1557.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 10:15:00 | 1576.20 | 1561.80 | 1556.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 1565.20 | 1570.26 | 1562.60 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-20 14:15:00 | 1559.90 | 1570.16 | 1562.59 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 1500.00 | 1558.73 | 1558.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 1497.80 | 1558.12 | 1558.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 1548.20 | 1547.45 | 1552.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-16 09:15:00 | 1531.00 | 1550.68 | 1553.78 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-16 12:15:00 | 1554.20 | 1550.67 | 1553.73 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 1585.60 | 1554.63 | 1554.60 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 1527.40 | 1554.61 | 1554.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 09:15:00 | 1521.10 | 1553.56 | 1554.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 10:15:00 | 1547.00 | 1541.19 | 1546.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-22 11:15:00 | 1526.40 | 1541.04 | 1546.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1526.40 | 1541.04 | 1546.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-22 12:15:00 | 1554.10 | 1541.17 | 1546.67 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 1572.00 | 1536.27 | 1536.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 1582.00 | 1536.72 | 1536.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1549.10 | 1567.86 | 1555.73 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 1473.00 | 1546.72 | 1546.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1468.50 | 1530.53 | 1537.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 1510.60 | 1508.88 | 1523.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-24 14:15:00 | 1494.40 | 1508.51 | 1523.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 1449.20 | 1427.66 | 1447.66 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-29 09:15:00 | 1386.35 | 2023-10-18 09:15:00 | 1355.60 | EXIT_EMA400 | -30.75 |
| SELL | 2023-11-28 09:15:00 | 1287.70 | 2023-12-01 11:15:00 | 1312.35 | EXIT_EMA400 | -24.65 |
| BUY | 2024-02-16 11:15:00 | 1393.55 | 2024-02-28 10:15:00 | 1524.41 | TARGET | 130.86 |
| BUY | 2024-04-19 11:15:00 | 1493.95 | 2024-04-26 09:15:00 | 1580.93 | TARGET | 86.98 |
| SELL | 2024-12-20 12:15:00 | 1688.70 | 2025-01-28 09:15:00 | 1489.60 | TARGET | 199.10 |
| BUY | 2025-05-12 10:15:00 | 1576.20 | 2025-05-20 14:15:00 | 1559.90 | EXIT_EMA400 | -16.30 |
| SELL | 2025-06-16 09:15:00 | 1531.00 | 2025-06-16 12:15:00 | 1554.20 | EXIT_EMA400 | -23.20 |
| SELL | 2025-07-22 11:15:00 | 1526.40 | 2025-07-22 12:15:00 | 1554.10 | EXIT_EMA400 | -27.70 |
| SELL | 2025-10-24 14:15:00 | 1494.40 | 2025-12-08 11:15:00 | 1407.60 | TARGET | 86.80 |
