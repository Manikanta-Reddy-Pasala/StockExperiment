# Gland Pharma Ltd. (GLAND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1750.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 9 |
| ENTRY1 | 7 |
| ENTRY2 | 7 |
| EXIT | 7 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 3 / 11
- **Target hits / EMA400 exits:** 3 / 11
- **Total realized P&L (per unit):** -428.86
- **Avg P&L per closed trade:** -30.63

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 1750.05 | 1865.94 | 1866.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 1727.35 | 1860.16 | 1863.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 09:15:00 | 1812.00 | 1808.93 | 1831.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-04 13:15:00 | 1778.05 | 1808.72 | 1827.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-08 09:15:00 | 1846.05 | 1805.35 | 1825.09 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 09:15:00 | 1852.00 | 1793.45 | 1793.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 1860.05 | 1806.83 | 1800.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 1800.40 | 1828.57 | 1814.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-08 10:15:00 | 1875.05 | 1821.94 | 1815.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 2014.10 | 2002.16 | 1935.05 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-07 10:15:00 | 2065.10 | 2002.79 | 1935.70 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-16 09:15:00 | 1918.35 | 2006.43 | 1950.44 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 09:15:00 | 1879.25 | 1922.53 | 1922.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 10:15:00 | 1866.85 | 1921.97 | 1922.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 1787.00 | 1697.54 | 1764.92 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 1842.50 | 1776.19 | 1776.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 1900.05 | 1779.88 | 1777.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 1800.05 | 1804.85 | 1792.08 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 15:15:00 | 1674.45 | 1780.36 | 1780.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 1659.00 | 1755.99 | 1767.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 1551.60 | 1547.94 | 1618.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 1489.70 | 1582.31 | 1601.34 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-20 09:15:00 | 1498.60 | 1458.81 | 1498.19 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 1620.00 | 1521.59 | 1521.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 1627.60 | 1522.65 | 1521.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 1932.40 | 1936.03 | 1834.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-07 11:15:00 | 1968.50 | 1936.36 | 1835.26 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1876.00 | 1936.30 | 1875.09 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-29 12:15:00 | 1873.20 | 1935.67 | 1875.08 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 12:15:00 | 1849.00 | 1919.09 | 1919.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 14:15:00 | 1838.10 | 1917.60 | 1918.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 1721.50 | 1718.70 | 1776.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-01 09:15:00 | 1696.60 | 1718.51 | 1775.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1733.60 | 1713.29 | 1763.10 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-08 10:15:00 | 1710.20 | 1713.26 | 1762.84 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1752.10 | 1711.25 | 1754.31 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-16 09:15:00 | 1736.90 | 1711.85 | 1754.19 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-29 09:15:00 | 1778.00 | 1703.17 | 1738.94 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1879.70 | 1767.05 | 1766.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 1888.00 | 1769.36 | 1767.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1772.30 | 1786.53 | 1777.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-17 09:15:00 | 1812.10 | 1787.63 | 1778.40 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1800.00 | 1800.07 | 1786.76 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-24 10:15:00 | 1802.00 | 1800.09 | 1786.84 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-02 09:15:00 | 1783.60 | 1806.96 | 1792.14 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 1679.40 | 1779.81 | 1779.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 1671.30 | 1767.32 | 1773.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 1718.00 | 1700.62 | 1732.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 09:15:00 | 1681.60 | 1700.97 | 1731.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1707.00 | 1701.31 | 1731.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-30 09:15:00 | 1673.00 | 1701.03 | 1730.88 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1721.60 | 1701.31 | 1729.85 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-02 09:15:00 | 1679.30 | 1701.71 | 1729.21 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1723.80 | 1700.99 | 1727.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-07 09:15:00 | 1682.10 | 1700.82 | 1726.89 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1724.00 | 1701.50 | 1726.33 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 10:15:00 | 1731.20 | 1701.79 | 1726.36 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-04 13:15:00 | 1778.05 | 2024-04-08 09:15:00 | 1846.05 | EXIT_EMA400 | -68.00 |
| BUY | 2024-07-08 10:15:00 | 1875.05 | 2024-07-09 13:15:00 | 2054.28 | TARGET | 179.23 |
| BUY | 2024-08-07 10:15:00 | 2065.10 | 2024-08-16 09:15:00 | 1918.35 | EXIT_EMA400 | -146.75 |
| SELL | 2025-04-04 09:15:00 | 1489.70 | 2025-05-20 09:15:00 | 1498.60 | EXIT_EMA400 | -8.90 |
| BUY | 2025-08-07 11:15:00 | 1968.50 | 2025-08-29 12:15:00 | 1873.20 | EXIT_EMA400 | -95.30 |
| SELL | 2026-01-16 09:15:00 | 1736.90 | 2026-01-19 13:15:00 | 1685.03 | TARGET | 51.87 |
| SELL | 2026-01-01 09:15:00 | 1696.60 | 2026-01-29 09:15:00 | 1778.00 | EXIT_EMA400 | -81.40 |
| SELL | 2026-01-08 10:15:00 | 1710.20 | 2026-01-29 09:15:00 | 1778.00 | EXIT_EMA400 | -67.80 |
| BUY | 2026-02-24 10:15:00 | 1802.00 | 2026-02-25 14:15:00 | 1847.49 | TARGET | 45.49 |
| BUY | 2026-02-17 09:15:00 | 1812.10 | 2026-03-02 09:15:00 | 1783.60 | EXIT_EMA400 | -28.50 |
| SELL | 2026-03-27 09:15:00 | 1681.60 | 2026-04-08 10:15:00 | 1731.20 | EXIT_EMA400 | -49.60 |
| SELL | 2026-03-30 09:15:00 | 1673.00 | 2026-04-08 10:15:00 | 1731.20 | EXIT_EMA400 | -58.20 |
| SELL | 2026-04-02 09:15:00 | 1679.30 | 2026-04-08 10:15:00 | 1731.20 | EXIT_EMA400 | -51.90 |
| SELL | 2026-04-07 09:15:00 | 1682.10 | 2026-04-08 10:15:00 | 1731.20 | EXIT_EMA400 | -49.10 |
