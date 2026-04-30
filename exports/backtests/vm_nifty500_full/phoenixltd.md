# Phoenix Mills Ltd. (PHOENIXLTD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1765.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 345.08
- **Avg P&L per closed trade:** 49.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 12:15:00 | 1671.30 | 1751.20 | 1751.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 13:15:00 | 1652.40 | 1750.21 | 1750.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 1749.65 | 1748.34 | 1749.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-11 09:15:00 | 1647.10 | 1741.95 | 1746.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-22 14:15:00 | 1630.45 | 1543.89 | 1606.04 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 15:15:00 | 1791.05 | 1644.28 | 1644.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 10:15:00 | 1819.00 | 1647.57 | 1645.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 1706.80 | 1708.01 | 1680.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-19 11:15:00 | 1722.65 | 1708.17 | 1681.09 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 1682.00 | 1707.82 | 1681.18 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-19 14:15:00 | 1670.00 | 1707.44 | 1681.13 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 1547.60 | 1666.43 | 1666.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 1521.25 | 1654.80 | 1660.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 12:15:00 | 1657.60 | 1645.02 | 1654.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-20 09:15:00 | 1625.40 | 1647.09 | 1655.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1625.40 | 1647.09 | 1655.05 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-20 12:15:00 | 1614.75 | 1646.31 | 1654.53 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1600.00 | 1593.84 | 1622.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 11:15:00 | 1625.05 | 1594.56 | 1622.03 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 1641.75 | 1609.95 | 1609.82 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 1569.15 | 1609.77 | 1609.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 09:15:00 | 1503.30 | 1603.31 | 1606.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 14:15:00 | 1594.80 | 1585.95 | 1596.72 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 09:15:00 | 1662.30 | 1605.73 | 1605.48 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 13:15:00 | 1535.00 | 1606.01 | 1606.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 1519.50 | 1602.33 | 1604.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 1588.40 | 1567.46 | 1583.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-30 15:15:00 | 1537.10 | 1579.23 | 1585.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1578.30 | 1578.85 | 1585.54 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-02 14:15:00 | 1585.80 | 1578.96 | 1585.53 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 1644.70 | 1590.39 | 1590.17 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 1499.50 | 1591.54 | 1591.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 1489.80 | 1590.53 | 1591.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 10:15:00 | 1528.00 | 1524.00 | 1550.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-28 09:15:00 | 1506.00 | 1524.27 | 1549.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-20 09:15:00 | 1532.10 | 1483.29 | 1512.99 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1602.60 | 1528.48 | 1528.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 1616.10 | 1532.29 | 1530.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 1558.40 | 1563.55 | 1548.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-01 13:15:00 | 1576.20 | 1562.33 | 1549.71 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1551.70 | 1562.20 | 1549.83 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-03 10:15:00 | 1542.20 | 1562.00 | 1549.79 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 1730.60 | 1760.29 | 1760.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 1727.10 | 1759.96 | 1760.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1599.30 | 1588.31 | 1642.91 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 1787.30 | 1680.33 | 1679.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 11:15:00 | 1797.90 | 1681.50 | 1680.46 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-11 09:15:00 | 1647.10 | 2024-10-28 09:15:00 | 1348.86 | TARGET | 298.24 |
| BUY | 2024-12-19 11:15:00 | 1722.65 | 2024-12-19 14:15:00 | 1670.00 | EXIT_EMA400 | -52.65 |
| SELL | 2025-01-20 09:15:00 | 1625.40 | 2025-01-21 14:15:00 | 1536.46 | TARGET | 88.94 |
| SELL | 2025-01-20 12:15:00 | 1614.75 | 2025-01-22 09:15:00 | 1495.40 | TARGET | 119.35 |
| SELL | 2025-05-30 15:15:00 | 1537.10 | 2025-06-02 14:15:00 | 1585.80 | EXIT_EMA400 | -48.70 |
| SELL | 2025-07-28 09:15:00 | 1506.00 | 2025-08-20 09:15:00 | 1532.10 | EXIT_EMA400 | -26.10 |
| BUY | 2025-10-01 13:15:00 | 1576.20 | 2025-10-03 10:15:00 | 1542.20 | EXIT_EMA400 | -34.00 |
