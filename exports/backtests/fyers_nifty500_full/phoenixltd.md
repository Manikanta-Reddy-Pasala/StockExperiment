# Phoenix Mills Ltd. (PHOENIXLTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1768.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 330.22
- **Avg P&L per closed trade:** 41.28

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 11:15:00 | 1689.10 | 1752.23 | 1752.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 12:15:00 | 1670.40 | 1751.41 | 1751.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 1749.65 | 1748.54 | 1750.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-11 09:15:00 | 1647.10 | 1742.05 | 1746.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1602.00 | 1542.45 | 1605.30 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-22 14:15:00 | 1630.45 | 1543.33 | 1605.43 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 15:15:00 | 1791.00 | 1643.98 | 1643.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 10:15:00 | 1819.00 | 1647.27 | 1645.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 1706.80 | 1707.82 | 1680.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-19 11:15:00 | 1722.65 | 1707.98 | 1680.76 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 1682.05 | 1707.64 | 1680.86 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-19 14:15:00 | 1669.95 | 1707.26 | 1680.80 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 1550.00 | 1666.49 | 1666.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 1518.55 | 1654.76 | 1660.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 12:15:00 | 1657.60 | 1644.88 | 1654.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-20 09:15:00 | 1627.00 | 1646.87 | 1654.81 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1627.00 | 1646.87 | 1654.81 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-20 14:15:00 | 1609.60 | 1645.45 | 1653.89 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1600.00 | 1593.81 | 1622.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 11:15:00 | 1625.05 | 1594.41 | 1621.85 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 1649.30 | 1610.78 | 1610.69 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1544.75 | 1610.43 | 1610.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-11 13:15:00 | 1474.30 | 1592.44 | 1601.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 14:15:00 | 1593.70 | 1586.10 | 1597.16 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 10:15:00 | 1651.60 | 1606.32 | 1606.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 1672.70 | 1611.88 | 1609.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 09:15:00 | 1568.00 | 1614.59 | 1610.51 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 13:15:00 | 1535.00 | 1606.02 | 1606.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 1519.50 | 1602.37 | 1604.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 1588.40 | 1567.34 | 1583.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-30 15:15:00 | 1534.30 | 1579.08 | 1585.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1578.30 | 1578.70 | 1585.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-02 14:15:00 | 1585.80 | 1578.82 | 1585.57 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 1644.70 | 1590.34 | 1590.23 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 1538.00 | 1592.38 | 1592.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 11:15:00 | 1499.50 | 1591.45 | 1591.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 10:15:00 | 1528.00 | 1524.00 | 1550.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-28 09:15:00 | 1506.60 | 1524.21 | 1549.35 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-20 09:15:00 | 1532.30 | 1483.29 | 1512.98 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1602.60 | 1528.51 | 1528.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 1616.10 | 1532.31 | 1530.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 1558.40 | 1563.55 | 1548.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-01 13:15:00 | 1576.20 | 1562.34 | 1549.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1551.70 | 1562.21 | 1549.83 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-03 10:15:00 | 1542.20 | 1562.01 | 1549.79 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 10:15:00 | 1748.70 | 1758.36 | 1758.37 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 1769.20 | 1758.45 | 1758.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 1800.20 | 1758.97 | 1758.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1733.00 | 1761.35 | 1759.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-16 14:15:00 | 1774.50 | 1759.86 | 1759.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-17 09:15:00 | 1752.70 | 1759.94 | 1759.28 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1720.40 | 1758.39 | 1758.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 1720.10 | 1756.03 | 1757.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1599.30 | 1587.42 | 1641.71 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 1812.40 | 1679.02 | 1678.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 1834.10 | 1709.11 | 1694.94 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-11 09:15:00 | 1647.10 | 2024-10-28 09:15:00 | 1347.49 | TARGET | 299.61 |
| BUY | 2024-12-19 11:15:00 | 1722.65 | 2024-12-19 14:15:00 | 1669.95 | EXIT_EMA400 | -52.70 |
| SELL | 2025-01-20 09:15:00 | 1627.00 | 2025-01-21 12:15:00 | 1543.57 | TARGET | 83.43 |
| SELL | 2025-01-20 14:15:00 | 1609.60 | 2025-01-22 09:15:00 | 1476.72 | TARGET | 132.88 |
| SELL | 2025-05-30 15:15:00 | 1534.30 | 2025-06-02 14:15:00 | 1585.80 | EXIT_EMA400 | -51.50 |
| SELL | 2025-07-28 09:15:00 | 1506.60 | 2025-08-20 09:15:00 | 1532.30 | EXIT_EMA400 | -25.70 |
| BUY | 2025-10-01 13:15:00 | 1576.20 | 2025-10-03 10:15:00 | 1542.20 | EXIT_EMA400 | -34.00 |
| BUY | 2026-02-16 14:15:00 | 1774.50 | 2026-02-17 09:15:00 | 1752.70 | EXIT_EMA400 | -21.80 |
