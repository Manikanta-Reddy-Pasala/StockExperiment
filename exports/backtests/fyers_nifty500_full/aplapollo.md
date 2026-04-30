# APL Apollo Tubes Ltd. (APLAPOLLO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1907.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 8 |
| ENTRY2 | 3 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / EMA400 exits:** 2 / 9
- **Total realized P&L (per unit):** -206.75
- **Avg P&L per closed trade:** -18.80

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 12:15:00 | 1615.80 | 1478.35 | 1478.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 13:15:00 | 1618.75 | 1479.75 | 1478.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 1522.55 | 1528.42 | 1508.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-07 09:15:00 | 1549.65 | 1514.75 | 1506.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-11-08 11:15:00 | 1490.05 | 1515.44 | 1507.29 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 10:15:00 | 1422.85 | 1502.02 | 1502.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 12:15:00 | 1419.65 | 1500.41 | 1501.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 10:15:00 | 1497.35 | 1495.81 | 1498.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 14:15:00 | 1479.50 | 1495.74 | 1498.81 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 1479.50 | 1495.74 | 1498.81 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-27 09:15:00 | 1474.70 | 1495.39 | 1498.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 1483.85 | 1494.71 | 1498.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-28 09:15:00 | 1472.75 | 1494.38 | 1497.97 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-29 09:15:00 | 1498.20 | 1492.77 | 1497.03 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 11:15:00 | 1578.55 | 1501.00 | 1500.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 1589.35 | 1512.93 | 1507.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 09:15:00 | 1548.25 | 1552.08 | 1532.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-31 12:15:00 | 1572.95 | 1542.86 | 1530.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-08 11:15:00 | 1536.25 | 1558.03 | 1541.22 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 1508.45 | 1534.23 | 1534.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1457.90 | 1533.15 | 1533.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 09:15:00 | 1444.80 | 1436.66 | 1476.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-13 15:15:00 | 1367.00 | 1438.85 | 1462.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 09:15:00 | 1462.85 | 1436.93 | 1459.84 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 1551.60 | 1475.57 | 1475.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 15:15:00 | 1562.50 | 1477.27 | 1476.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1431.00 | 1485.68 | 1480.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-11 09:15:00 | 1516.15 | 1480.16 | 1478.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-30 10:15:00 | 1732.60 | 1794.96 | 1737.56 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 1551.60 | 1716.09 | 1716.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 1497.50 | 1713.92 | 1715.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 1638.10 | 1630.49 | 1660.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 11:15:00 | 1608.60 | 1633.34 | 1657.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1648.70 | 1631.00 | 1653.65 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-02 10:15:00 | 1661.10 | 1631.30 | 1653.68 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 1669.70 | 1666.01 | 1666.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 1689.40 | 1667.86 | 1666.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 1667.90 | 1669.40 | 1667.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 1685.00 | 1669.38 | 1667.81 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-19 13:15:00 | 1730.00 | 1761.65 | 1735.80 | Close below EMA400 |

### Cycle 8 — SELL (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 12:15:00 | 1910.30 | 2018.31 | 2018.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 09:15:00 | 1880.00 | 2013.93 | 2016.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 12:15:00 | 2033.90 | 2006.43 | 2012.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 1999.40 | 2012.20 | 2015.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1999.40 | 2012.20 | 2015.08 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-13 14:15:00 | 1980.70 | 2011.71 | 2014.76 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 2004.30 | 2011.32 | 2014.53 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 10:15:00 | 2019.80 | 2011.41 | 2014.56 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 2106.90 | 2017.77 | 2017.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 2121.80 | 2021.48 | 2019.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 2023.10 | 2039.66 | 2029.47 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 1902.30 | 2021.13 | 2021.23 | EMA200 below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-11-07 09:15:00 | 1549.65 | 2024-11-08 11:15:00 | 1490.05 | EXIT_EMA400 | -59.60 |
| SELL | 2024-11-26 14:15:00 | 1479.50 | 2024-11-29 09:15:00 | 1498.20 | EXIT_EMA400 | -18.70 |
| SELL | 2024-11-27 09:15:00 | 1474.70 | 2024-11-29 09:15:00 | 1498.20 | EXIT_EMA400 | -23.50 |
| SELL | 2024-11-28 09:15:00 | 1472.75 | 2024-11-29 09:15:00 | 1498.20 | EXIT_EMA400 | -25.45 |
| BUY | 2024-12-31 12:15:00 | 1572.95 | 2025-01-08 11:15:00 | 1536.25 | EXIT_EMA400 | -36.70 |
| SELL | 2025-03-13 15:15:00 | 1367.00 | 2025-03-19 09:15:00 | 1462.85 | EXIT_EMA400 | -95.85 |
| BUY | 2025-04-11 09:15:00 | 1516.15 | 2025-04-22 11:15:00 | 1629.63 | TARGET | 113.48 |
| SELL | 2025-08-28 11:15:00 | 1608.60 | 2025-09-02 10:15:00 | 1661.10 | EXIT_EMA400 | -52.50 |
| BUY | 2025-09-29 09:15:00 | 1685.00 | 2025-10-01 15:15:00 | 1736.57 | TARGET | 51.57 |
| SELL | 2026-04-13 09:15:00 | 1999.40 | 2026-04-15 10:15:00 | 2019.80 | EXIT_EMA400 | -20.40 |
| SELL | 2026-04-13 14:15:00 | 1980.70 | 2026-04-15 10:15:00 | 2019.80 | EXIT_EMA400 | -39.10 |
