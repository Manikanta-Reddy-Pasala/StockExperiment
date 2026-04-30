# Tata Communications Ltd. (TATACOMM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1580.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 7 |
| ENTRY1 | 10 |
| ENTRY2 | 2 |
| EXIT | 10 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 6 / 6
- **Target hits / EMA400 exits:** 6 / 6
- **Total realized P&L (per unit):** 685.33
- **Avg P&L per closed trade:** 57.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 13:15:00 | 1669.95 | 1753.64 | 1753.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 14:15:00 | 1665.50 | 1752.76 | 1753.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 13:15:00 | 1740.90 | 1740.67 | 1746.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-07 10:15:00 | 1725.95 | 1740.38 | 1746.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-24 10:15:00 | 1757.85 | 1723.65 | 1734.25 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 14:15:00 | 1798.95 | 1731.24 | 1731.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 15:15:00 | 1821.50 | 1732.14 | 1731.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 09:15:00 | 1737.25 | 1737.43 | 1734.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-03 11:15:00 | 1754.00 | 1737.65 | 1734.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 1737.55 | 1741.04 | 1736.66 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-08 13:15:00 | 1735.85 | 1740.91 | 1736.66 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 10:15:00 | 1713.50 | 1733.92 | 1733.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 12:15:00 | 1700.05 | 1729.55 | 1731.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 09:15:00 | 1727.55 | 1721.28 | 1726.87 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 12:15:00 | 1817.50 | 1731.65 | 1731.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 10:15:00 | 1828.30 | 1735.88 | 1733.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-20 09:15:00 | 1858.05 | 1883.34 | 1831.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-26 13:15:00 | 1955.25 | 1888.92 | 1840.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-04-16 09:15:00 | 1896.75 | 1957.05 | 1899.65 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 15:15:00 | 1729.85 | 1860.39 | 1860.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 1722.55 | 1848.75 | 1854.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 10:15:00 | 1803.70 | 1800.28 | 1824.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-30 09:15:00 | 1775.35 | 1806.24 | 1821.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1815.40 | 1802.22 | 1818.74 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-06-03 14:15:00 | 1773.05 | 1801.51 | 1817.98 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 1800.00 | 1787.81 | 1808.48 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-07 13:15:00 | 1812.25 | 1788.05 | 1808.50 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 1865.15 | 1824.39 | 1824.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 14:15:00 | 1890.10 | 1835.37 | 1830.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 10:15:00 | 1835.90 | 1846.49 | 1837.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-09 12:15:00 | 1860.20 | 1846.60 | 1837.30 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 1860.20 | 1846.60 | 1837.30 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-10 09:15:00 | 1834.35 | 1846.53 | 1837.45 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 13:15:00 | 1794.20 | 1945.22 | 1945.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 14:15:00 | 1784.50 | 1943.62 | 1945.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 1796.65 | 1794.90 | 1841.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-19 12:15:00 | 1763.00 | 1804.38 | 1830.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 09:15:00 | 1563.00 | 1482.62 | 1554.80 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 14:15:00 | 1644.00 | 1573.36 | 1573.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1655.50 | 1578.59 | 1575.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 1665.60 | 1672.31 | 1639.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 1686.00 | 1667.07 | 1642.46 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-01 10:15:00 | 1691.30 | 1725.79 | 1698.71 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 1589.00 | 1682.21 | 1682.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1570.60 | 1670.37 | 1676.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 1616.50 | 1611.13 | 1637.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-29 14:15:00 | 1601.20 | 1638.31 | 1646.05 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-06 11:15:00 | 1643.90 | 1633.30 | 1642.48 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1869.00 | 1651.27 | 1650.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 1888.20 | 1653.63 | 1651.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 1823.00 | 1824.73 | 1768.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-12 15:15:00 | 1854.00 | 1824.89 | 1770.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1814.90 | 1847.42 | 1811.73 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-08 14:15:00 | 1808.50 | 1847.03 | 1811.71 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 13:15:00 | 1725.90 | 1801.29 | 1801.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 15:15:00 | 1712.20 | 1783.53 | 1791.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 14:15:00 | 1678.50 | 1651.73 | 1707.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-16 09:15:00 | 1636.10 | 1657.09 | 1703.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1684.10 | 1655.77 | 1695.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-23 09:15:00 | 1663.00 | 1655.84 | 1695.57 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1531.00 | 1470.04 | 1534.37 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 10:15:00 | 1542.10 | 1470.76 | 1534.40 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-07 10:15:00 | 1725.95 | 2023-11-24 10:15:00 | 1757.85 | EXIT_EMA400 | -31.90 |
| BUY | 2024-01-03 11:15:00 | 1754.00 | 2024-01-08 13:15:00 | 1735.85 | EXIT_EMA400 | -18.15 |
| BUY | 2024-03-26 13:15:00 | 1955.25 | 2024-04-16 09:15:00 | 1896.75 | EXIT_EMA400 | -58.50 |
| SELL | 2024-05-30 09:15:00 | 1775.35 | 2024-06-04 11:15:00 | 1635.73 | TARGET | 139.62 |
| SELL | 2024-06-03 14:15:00 | 1773.05 | 2024-06-04 11:15:00 | 1638.27 | TARGET | 134.78 |
| BUY | 2024-07-09 12:15:00 | 1860.20 | 2024-07-10 09:15:00 | 1834.35 | EXIT_EMA400 | -25.85 |
| SELL | 2024-12-19 12:15:00 | 1763.00 | 2025-01-28 09:15:00 | 1561.22 | TARGET | 201.78 |
| BUY | 2025-06-27 09:15:00 | 1686.00 | 2025-07-02 12:15:00 | 1816.63 | TARGET | 130.63 |
| SELL | 2025-09-29 14:15:00 | 1601.20 | 2025-10-06 11:15:00 | 1643.90 | EXIT_EMA400 | -42.70 |
| BUY | 2025-11-12 15:15:00 | 1854.00 | 2025-12-08 14:15:00 | 1808.50 | EXIT_EMA400 | -45.50 |
| SELL | 2026-02-23 09:15:00 | 1663.00 | 2026-03-02 09:15:00 | 1565.29 | TARGET | 97.71 |
| SELL | 2026-02-16 09:15:00 | 1636.10 | 2026-03-09 09:15:00 | 1432.70 | TARGET | 203.40 |
