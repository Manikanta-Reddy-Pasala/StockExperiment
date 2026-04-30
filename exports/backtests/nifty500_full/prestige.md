# Prestige Estates Projects Ltd. (PRESTIGE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1414.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 8 |
| ENTRY1 | 7 |
| ENTRY2 | 5 |
| EXIT | 7 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / EMA400 exits:** 5 / 7
- **Total realized P&L (per unit):** 850.86
- **Avg P&L per closed trade:** 70.90

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 15:15:00 | 1036.55 | 1136.91 | 1137.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-21 11:15:00 | 1007.95 | 1133.63 | 1135.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 1132.00 | 1128.23 | 1132.75 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 1230.65 | 1137.05 | 1136.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 1271.40 | 1141.58 | 1139.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 14:15:00 | 1188.85 | 1200.90 | 1174.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-15 09:15:00 | 1219.10 | 1201.06 | 1174.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 1219.10 | 1201.06 | 1174.45 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-18 11:15:00 | 1241.00 | 1203.02 | 1177.52 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1504.00 | 1498.84 | 1397.54 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-04 13:15:00 | 1557.20 | 1499.42 | 1398.34 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 1541.60 | 1499.78 | 1400.52 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-05 11:15:00 | 1556.60 | 1500.35 | 1401.30 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1735.30 | 1783.25 | 1690.87 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-24 09:15:00 | 1841.00 | 1783.38 | 1692.31 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-02 09:15:00 | 1702.50 | 1802.30 | 1723.20 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 1656.25 | 1779.07 | 1779.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 15:15:00 | 1650.50 | 1777.80 | 1778.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1685.10 | 1659.47 | 1703.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-27 09:15:00 | 1644.00 | 1665.49 | 1703.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1689.00 | 1662.73 | 1697.71 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-02 14:15:00 | 1730.75 | 1663.40 | 1697.87 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 15:15:00 | 1854.40 | 1718.47 | 1718.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 12:15:00 | 1884.60 | 1723.95 | 1721.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 11:15:00 | 1728.65 | 1742.89 | 1731.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-23 14:15:00 | 1758.60 | 1743.00 | 1731.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-26 09:15:00 | 1717.05 | 1743.90 | 1732.71 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 1658.90 | 1724.47 | 1724.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 1654.70 | 1723.12 | 1723.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 1212.45 | 1210.74 | 1318.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 14:15:00 | 1197.65 | 1214.66 | 1304.50 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-21 12:15:00 | 1249.00 | 1178.98 | 1247.68 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 1389.00 | 1285.32 | 1285.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 13:15:00 | 1391.50 | 1286.38 | 1285.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 10:15:00 | 1610.50 | 1613.15 | 1520.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 10:15:00 | 1628.80 | 1612.98 | 1523.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1642.00 | 1684.14 | 1606.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-28 12:15:00 | 1606.10 | 1682.22 | 1606.78 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 1564.70 | 1599.65 | 1599.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 11:15:00 | 1543.50 | 1594.39 | 1596.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 1593.60 | 1591.52 | 1595.41 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 12:15:00 | 1635.70 | 1599.06 | 1598.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 1640.00 | 1602.03 | 1600.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1601.00 | 1603.55 | 1601.35 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 1522.90 | 1598.69 | 1598.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 1521.10 | 1595.23 | 1597.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1569.40 | 1566.04 | 1580.17 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1710.00 | 1591.40 | 1591.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 11:15:00 | 1720.20 | 1593.91 | 1592.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 1695.80 | 1696.64 | 1658.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-13 09:15:00 | 1758.70 | 1697.26 | 1658.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1676.60 | 1707.34 | 1673.18 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-24 10:15:00 | 1670.70 | 1706.97 | 1673.16 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 1598.80 | 1659.18 | 1659.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 1588.90 | 1642.16 | 1649.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 10:15:00 | 1638.00 | 1630.95 | 1642.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-07 12:15:00 | 1613.80 | 1633.99 | 1643.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1540.30 | 1513.10 | 1564.79 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-04 09:15:00 | 1515.90 | 1514.98 | 1563.97 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-09 09:15:00 | 1591.90 | 1517.50 | 1560.37 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-15 09:15:00 | 1219.10 | 2024-04-26 12:15:00 | 1353.04 | TARGET | 133.94 |
| BUY | 2024-04-18 11:15:00 | 1241.00 | 2024-05-03 09:15:00 | 1431.45 | TARGET | 190.45 |
| BUY | 2024-06-04 13:15:00 | 1557.20 | 2024-06-20 15:15:00 | 2033.79 | TARGET | 476.59 |
| BUY | 2024-06-05 11:15:00 | 1556.60 | 2024-06-20 15:15:00 | 2022.50 | TARGET | 465.90 |
| BUY | 2024-07-24 09:15:00 | 1841.00 | 2024-08-02 09:15:00 | 1702.50 | EXIT_EMA400 | -138.50 |
| SELL | 2024-11-27 09:15:00 | 1644.00 | 2024-12-02 14:15:00 | 1730.75 | EXIT_EMA400 | -86.75 |
| BUY | 2024-12-23 14:15:00 | 1758.60 | 2024-12-26 09:15:00 | 1717.05 | EXIT_EMA400 | -41.55 |
| SELL | 2025-03-26 14:15:00 | 1197.65 | 2025-04-21 12:15:00 | 1249.00 | EXIT_EMA400 | -51.35 |
| BUY | 2025-07-03 10:15:00 | 1628.80 | 2025-07-28 12:15:00 | 1606.10 | EXIT_EMA400 | -22.70 |
| BUY | 2025-11-13 09:15:00 | 1758.70 | 2025-11-24 10:15:00 | 1670.70 | EXIT_EMA400 | -88.00 |
| SELL | 2026-01-07 12:15:00 | 1613.80 | 2026-01-12 09:15:00 | 1524.97 | TARGET | 88.83 |
| SELL | 2026-02-04 09:15:00 | 1515.90 | 2026-02-09 09:15:00 | 1591.90 | EXIT_EMA400 | -76.00 |
