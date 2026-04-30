# APL Apollo Tubes Ltd. (APLAPOLLO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1905.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 6 |
| ENTRY1 | 10 |
| ENTRY2 | 3 |
| EXIT | 10 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 2 / 11
- **Target hits / EMA400 exits:** 2 / 11
- **Total realized P&L (per unit):** -314.47
- **Avg P&L per closed trade:** -24.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 14:15:00 | 1543.10 | 1614.13 | 1614.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 09:15:00 | 1538.60 | 1612.68 | 1613.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 10:15:00 | 1570.00 | 1568.13 | 1587.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-15 13:15:00 | 1538.05 | 1567.53 | 1585.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-02-29 12:15:00 | 1497.95 | 1454.47 | 1491.94 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 15:15:00 | 1600.45 | 1512.99 | 1512.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 1627.30 | 1553.37 | 1542.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 1598.10 | 1606.90 | 1576.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-07 14:15:00 | 1621.95 | 1577.63 | 1566.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-13 10:15:00 | 1571.95 | 1586.34 | 1572.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 1530.00 | 1571.39 | 1571.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 13:15:00 | 1523.50 | 1570.92 | 1571.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 1445.00 | 1444.64 | 1485.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-06 14:15:00 | 1400.15 | 1448.66 | 1477.77 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-13 12:15:00 | 1478.95 | 1441.60 | 1469.41 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 13:15:00 | 1618.50 | 1479.76 | 1479.71 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 10:15:00 | 1422.85 | 1501.87 | 1502.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 12:15:00 | 1419.65 | 1500.28 | 1501.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 10:15:00 | 1497.35 | 1495.43 | 1498.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 14:15:00 | 1479.50 | 1495.37 | 1498.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 1479.50 | 1495.37 | 1498.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-27 09:15:00 | 1474.00 | 1495.03 | 1498.56 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 1483.75 | 1494.37 | 1498.14 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-28 09:15:00 | 1472.15 | 1494.03 | 1497.93 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1496.80 | 1492.43 | 1496.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-29 10:15:00 | 1518.55 | 1492.69 | 1497.09 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 1577.75 | 1501.51 | 1501.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 13:15:00 | 1580.70 | 1502.30 | 1501.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 09:15:00 | 1548.25 | 1552.00 | 1532.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-31 12:15:00 | 1572.70 | 1542.59 | 1530.46 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-08 11:15:00 | 1536.25 | 1557.76 | 1541.08 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 1443.85 | 1534.15 | 1534.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 12:15:00 | 1434.95 | 1533.16 | 1533.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 09:15:00 | 1444.70 | 1437.88 | 1477.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-13 15:15:00 | 1372.80 | 1439.50 | 1463.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 09:15:00 | 1462.85 | 1437.58 | 1460.52 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 1551.60 | 1476.10 | 1476.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 15:15:00 | 1562.20 | 1477.79 | 1476.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1431.00 | 1486.15 | 1481.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 12:15:00 | 1485.90 | 1481.96 | 1479.43 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1485.90 | 1481.96 | 1479.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-08 13:15:00 | 1475.40 | 1481.89 | 1479.41 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 1545.10 | 1717.63 | 1717.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 1497.50 | 1713.79 | 1715.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 1638.10 | 1630.45 | 1660.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 11:15:00 | 1608.60 | 1633.29 | 1657.01 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-02 10:15:00 | 1661.50 | 1631.28 | 1653.65 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 1669.70 | 1665.95 | 1665.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 1689.40 | 1667.84 | 1666.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 1667.90 | 1669.37 | 1667.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 1685.10 | 1669.38 | 1667.79 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-19 13:15:00 | 1730.00 | 1761.58 | 1735.76 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 13:15:00 | 1913.00 | 2016.66 | 2016.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 09:15:00 | 1880.20 | 2013.35 | 2015.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 12:15:00 | 2033.90 | 2005.84 | 2011.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 1999.40 | 2011.73 | 2013.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1999.40 | 2011.73 | 2013.77 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-13 14:15:00 | 1980.40 | 2011.26 | 2013.48 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 2004.30 | 2010.88 | 2013.27 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 10:15:00 | 2019.80 | 2010.97 | 2013.30 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 2085.90 | 2015.81 | 2015.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 13:15:00 | 2106.90 | 2017.37 | 2016.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 2023.10 | 2035.67 | 2026.34 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 1902.30 | 2018.17 | 2018.52 | EMA200 below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-15 13:15:00 | 1538.05 | 2024-02-06 10:15:00 | 1395.95 | TARGET | 142.10 |
| BUY | 2024-06-07 14:15:00 | 1621.95 | 2024-06-13 10:15:00 | 1571.95 | EXIT_EMA400 | -50.00 |
| SELL | 2024-09-06 14:15:00 | 1400.15 | 2024-09-13 12:15:00 | 1478.95 | EXIT_EMA400 | -78.80 |
| SELL | 2024-11-26 14:15:00 | 1479.50 | 2024-11-29 10:15:00 | 1518.55 | EXIT_EMA400 | -39.05 |
| SELL | 2024-11-27 09:15:00 | 1474.00 | 2024-11-29 10:15:00 | 1518.55 | EXIT_EMA400 | -44.55 |
| SELL | 2024-11-28 09:15:00 | 1472.15 | 2024-11-29 10:15:00 | 1518.55 | EXIT_EMA400 | -46.40 |
| BUY | 2024-12-31 12:15:00 | 1572.70 | 2025-01-08 11:15:00 | 1536.25 | EXIT_EMA400 | -36.45 |
| SELL | 2025-03-13 15:15:00 | 1372.80 | 2025-03-19 09:15:00 | 1462.85 | EXIT_EMA400 | -90.05 |
| BUY | 2025-04-08 12:15:00 | 1485.90 | 2025-04-08 13:15:00 | 1475.40 | EXIT_EMA400 | -10.50 |
| SELL | 2025-08-28 11:15:00 | 1608.60 | 2025-09-02 10:15:00 | 1661.50 | EXIT_EMA400 | -52.90 |
| BUY | 2025-09-29 09:15:00 | 1685.10 | 2025-10-01 15:15:00 | 1737.03 | TARGET | 51.93 |
| SELL | 2026-04-13 09:15:00 | 1999.40 | 2026-04-15 10:15:00 | 2019.80 | EXIT_EMA400 | -20.40 |
| SELL | 2026-04-13 14:15:00 | 1980.40 | 2026-04-15 10:15:00 | 2019.80 | EXIT_EMA400 | -39.40 |
