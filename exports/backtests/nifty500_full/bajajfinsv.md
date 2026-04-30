# Bajaj Finserv Ltd. (BAJAJFINSV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1747.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT3 | 9 |
| ENTRY1 | 11 |
| ENTRY2 | 3 |
| EXIT | 11 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 3 / 11
- **Target hits / EMA400 exits:** 3 / 11
- **Total realized P&L (per unit):** -245.68
- **Avg P&L per closed trade:** -17.55

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 10:15:00 | 1560.80 | 1523.33 | 1523.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 09:15:00 | 1582.45 | 1529.51 | 1526.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 1535.65 | 1538.16 | 1531.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-03 11:15:00 | 1549.65 | 1538.29 | 1532.02 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-10-04 09:15:00 | 1526.55 | 1538.91 | 1532.49 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 12:15:00 | 1590.00 | 1638.50 | 1638.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 13:15:00 | 1584.80 | 1637.96 | 1638.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 11:15:00 | 1613.20 | 1609.83 | 1622.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-21 11:15:00 | 1596.40 | 1609.89 | 1621.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 12:15:00 | 1616.00 | 1607.09 | 1619.00 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-02-26 09:15:00 | 1620.00 | 1607.49 | 1618.96 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 1675.65 | 1612.35 | 1612.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1692.05 | 1613.79 | 1612.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 1628.35 | 1636.34 | 1625.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-25 13:15:00 | 1668.90 | 1632.30 | 1625.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-04-26 09:15:00 | 1604.30 | 1632.47 | 1625.36 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 10:15:00 | 1570.90 | 1620.16 | 1620.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 1562.20 | 1617.23 | 1618.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 12:15:00 | 1607.75 | 1602.89 | 1610.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-29 10:15:00 | 1579.10 | 1602.35 | 1609.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-12 11:15:00 | 1590.85 | 1574.34 | 1590.28 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 13:15:00 | 1646.75 | 1592.28 | 1592.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 1665.00 | 1598.03 | 1595.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 1594.50 | 1606.18 | 1600.19 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 1561.10 | 1594.97 | 1595.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1545.50 | 1590.34 | 1592.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 10:15:00 | 1581.00 | 1580.28 | 1587.02 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 1673.30 | 1592.74 | 1592.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 1679.30 | 1593.60 | 1593.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 12:15:00 | 1850.75 | 1860.78 | 1782.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 10:15:00 | 1877.30 | 1860.36 | 1784.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1811.25 | 1859.53 | 1800.28 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-18 10:15:00 | 1821.10 | 1859.15 | 1800.38 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-21 09:15:00 | 1797.80 | 1856.79 | 1800.93 | Close below EMA400 |

### Cycle 8 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 1679.60 | 1770.94 | 1771.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 1675.60 | 1769.99 | 1770.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 1661.90 | 1661.43 | 1700.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-18 09:15:00 | 1626.05 | 1662.88 | 1693.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 1657.65 | 1620.89 | 1659.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-02 12:15:00 | 1711.80 | 1621.80 | 1660.02 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 1738.50 | 1679.22 | 1679.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 11:15:00 | 1749.05 | 1685.55 | 1682.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 10:15:00 | 1806.30 | 1815.29 | 1769.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-06 10:15:00 | 1831.65 | 1812.95 | 1771.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-07 12:15:00 | 1832.05 | 1879.88 | 1833.74 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 14:15:00 | 1940.00 | 1992.63 | 1992.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 1916.30 | 1990.75 | 1991.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2010.90 | 1963.66 | 1976.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 1939.90 | 1966.19 | 1975.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1966.10 | 1957.45 | 1968.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-04 09:15:00 | 2015.50 | 1958.03 | 1968.94 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 2036.30 | 1978.32 | 1978.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 2038.20 | 1978.92 | 1978.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 2015.00 | 2025.45 | 2006.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-07 09:15:00 | 2048.40 | 2019.75 | 2006.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 2016.50 | 2020.85 | 2007.84 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-09 11:15:00 | 2022.00 | 2020.48 | 2008.04 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 2010.00 | 2020.20 | 2008.38 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-10 13:15:00 | 2007.20 | 2020.07 | 2008.38 | Close below EMA400 |

### Cycle 12 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 2010.00 | 2053.54 | 2053.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1988.60 | 2051.64 | 2052.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 2073.20 | 2046.07 | 2049.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-06 14:15:00 | 2044.50 | 2046.27 | 2049.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2010.50 | 1990.83 | 2014.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 10:15:00 | 2015.60 | 1991.07 | 2014.40 | Close above EMA400 |

### Cycle 13 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 2050.20 | 2024.62 | 2024.52 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 1994.00 | 2024.15 | 2024.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1966.20 | 2023.58 | 2024.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.53 | 1866.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 12:15:00 | 1775.00 | 1783.40 | 1862.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1849.40 | 1796.77 | 1853.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-22 09:15:00 | 1840.80 | 1798.27 | 1853.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 1850.30 | 1799.23 | 1853.53 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-22 12:15:00 | 1857.20 | 1799.80 | 1853.55 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-03 11:15:00 | 1549.65 | 2023-10-04 09:15:00 | 1526.55 | EXIT_EMA400 | -23.10 |
| SELL | 2024-02-21 11:15:00 | 1596.40 | 2024-02-26 09:15:00 | 1620.00 | EXIT_EMA400 | -23.60 |
| BUY | 2024-04-25 13:15:00 | 1668.90 | 2024-04-26 09:15:00 | 1604.30 | EXIT_EMA400 | -64.60 |
| SELL | 2024-05-29 10:15:00 | 1579.10 | 2024-06-04 10:15:00 | 1489.00 | TARGET | 90.10 |
| BUY | 2024-10-09 10:15:00 | 1877.30 | 2024-10-21 09:15:00 | 1797.80 | EXIT_EMA400 | -79.50 |
| BUY | 2024-10-18 10:15:00 | 1821.10 | 2024-10-21 09:15:00 | 1797.80 | EXIT_EMA400 | -23.30 |
| SELL | 2024-12-18 09:15:00 | 1626.05 | 2025-01-02 12:15:00 | 1711.80 | EXIT_EMA400 | -85.75 |
| BUY | 2025-03-06 10:15:00 | 1831.65 | 2025-03-27 14:15:00 | 2011.25 | TARGET | 179.60 |
| SELL | 2025-08-26 09:15:00 | 1939.90 | 2025-09-04 09:15:00 | 2015.50 | EXIT_EMA400 | -75.60 |
| BUY | 2025-10-07 09:15:00 | 2048.40 | 2025-10-10 13:15:00 | 2007.20 | EXIT_EMA400 | -41.20 |
| BUY | 2025-10-09 11:15:00 | 2022.00 | 2025-10-10 13:15:00 | 2007.20 | EXIT_EMA400 | -14.80 |
| SELL | 2026-01-06 14:15:00 | 2044.50 | 2026-01-07 09:15:00 | 2029.84 | TARGET | 14.66 |
| SELL | 2026-04-09 12:15:00 | 1775.00 | 2026-04-22 12:15:00 | 1857.20 | EXIT_EMA400 | -82.20 |
| SELL | 2026-04-22 09:15:00 | 1840.80 | 2026-04-22 12:15:00 | 1857.20 | EXIT_EMA400 | -16.40 |
