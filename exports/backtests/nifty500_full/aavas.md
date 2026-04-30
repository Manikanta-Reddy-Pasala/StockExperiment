# Aavas Financiers Ltd. (AAVAS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1380.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 12 |
| ENTRY1 | 8 |
| ENTRY2 | 9 |
| EXIT | 8 |

## P&L

- **Trades closed:** 17
- **Trades open at end:** 0
- **Winners / losers:** 5 / 12
- **Target hits / EMA400 exits:** 5 / 12
- **Total realized P&L (per unit):** 150.83
- **Avg P&L per closed trade:** 8.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 09:15:00 | 1423.45 | 1617.26 | 1617.78 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 13:15:00 | 1595.40 | 1539.33 | 1539.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 09:15:00 | 1600.90 | 1541.03 | 1540.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 11:15:00 | 1538.45 | 1547.03 | 1543.32 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 15:15:00 | 1490.00 | 1539.86 | 1539.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 09:15:00 | 1479.90 | 1536.70 | 1538.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 09:15:00 | 1469.00 | 1464.87 | 1490.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-29 14:15:00 | 1454.25 | 1465.26 | 1487.13 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 15:15:00 | 1422.00 | 1390.81 | 1429.20 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-02 09:15:00 | 1404.00 | 1390.94 | 1429.07 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-04-04 09:15:00 | 1472.20 | 1394.25 | 1428.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 10:15:00 | 1535.35 | 1455.89 | 1455.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-15 11:15:00 | 1550.00 | 1456.82 | 1456.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 09:15:00 | 1575.45 | 1578.14 | 1543.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-31 14:15:00 | 1598.95 | 1577.31 | 1546.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1592.50 | 1581.08 | 1549.61 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 1520.20 | 1580.47 | 1549.46 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 1682.55 | 1705.88 | 1705.97 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 14:15:00 | 1711.00 | 1706.02 | 1706.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 10:15:00 | 1722.90 | 1706.27 | 1706.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 15:15:00 | 1707.00 | 1707.40 | 1706.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-03 09:15:00 | 1723.05 | 1707.56 | 1706.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 1778.20 | 1810.46 | 1775.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-04 09:15:00 | 1790.80 | 1809.65 | 1775.30 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1776.95 | 1809.15 | 1775.56 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-04 14:15:00 | 1770.00 | 1808.46 | 1775.55 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 09:15:00 | 1677.00 | 1755.90 | 1756.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 10:15:00 | 1671.05 | 1755.06 | 1755.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 1680.00 | 1679.80 | 1703.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-29 14:15:00 | 1674.90 | 1679.75 | 1703.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1691.00 | 1673.24 | 1693.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-12 11:15:00 | 1676.10 | 1673.42 | 1693.44 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1686.35 | 1671.84 | 1690.16 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-19 12:15:00 | 1660.00 | 1672.34 | 1689.53 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 1682.00 | 1672.46 | 1689.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-20 09:15:00 | 1677.15 | 1672.51 | 1689.28 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1677.15 | 1670.55 | 1684.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-02 10:15:00 | 1695.00 | 1670.80 | 1684.08 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 1710.30 | 1684.48 | 1684.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 10:15:00 | 1722.35 | 1687.07 | 1685.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 1673.35 | 1692.19 | 1688.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-10 09:15:00 | 1823.90 | 1692.22 | 1689.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1701.20 | 1697.66 | 1692.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-03-11 12:15:00 | 1749.00 | 1698.36 | 1693.04 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1904.05 | 1906.63 | 1823.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-07 15:15:00 | 2024.95 | 1910.28 | 1827.44 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-02 09:15:00 | 1904.80 | 1998.55 | 1915.59 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 1782.50 | 1867.37 | 1867.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 1765.10 | 1861.49 | 1864.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 1849.00 | 1843.83 | 1854.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-02 09:15:00 | 1799.70 | 1843.39 | 1853.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-06 12:15:00 | 1856.30 | 1834.62 | 1847.65 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 1967.40 | 1855.25 | 1854.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 1976.50 | 1856.46 | 1855.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 1901.90 | 1903.77 | 1882.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 09:15:00 | 1961.90 | 1903.56 | 1883.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1961.90 | 1903.56 | 1883.00 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-08 11:15:00 | 1982.30 | 1905.02 | 1883.93 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-18 11:15:00 | 1891.00 | 1922.97 | 1899.63 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1741.30 | 1884.94 | 1885.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 1715.20 | 1877.44 | 1881.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1650.00 | 1648.79 | 1716.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-25 11:15:00 | 1632.20 | 1654.30 | 1702.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1672.00 | 1646.36 | 1687.95 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-07 12:15:00 | 1662.50 | 1646.99 | 1687.65 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 1676.00 | 1632.82 | 1667.85 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-29 14:15:00 | 1454.25 | 2024-03-05 09:15:00 | 1355.62 | TARGET | 98.63 |
| SELL | 2024-04-02 09:15:00 | 1404.00 | 2024-04-04 09:15:00 | 1472.20 | EXIT_EMA400 | -68.20 |
| BUY | 2024-05-31 14:15:00 | 1598.95 | 2024-06-04 10:15:00 | 1520.20 | EXIT_EMA400 | -78.75 |
| BUY | 2024-09-03 09:15:00 | 1723.05 | 2024-09-05 11:15:00 | 1771.78 | TARGET | 48.73 |
| BUY | 2024-10-04 09:15:00 | 1790.80 | 2024-10-04 14:15:00 | 1770.00 | EXIT_EMA400 | -20.80 |
| SELL | 2024-11-29 14:15:00 | 1674.90 | 2025-01-02 10:15:00 | 1695.00 | EXIT_EMA400 | -20.10 |
| SELL | 2024-12-12 11:15:00 | 1676.10 | 2025-01-02 10:15:00 | 1695.00 | EXIT_EMA400 | -18.90 |
| SELL | 2024-12-19 12:15:00 | 1660.00 | 2025-01-02 10:15:00 | 1695.00 | EXIT_EMA400 | -35.00 |
| SELL | 2024-12-20 09:15:00 | 1677.15 | 2025-01-02 10:15:00 | 1695.00 | EXIT_EMA400 | -17.85 |
| BUY | 2025-03-11 12:15:00 | 1749.00 | 2025-03-17 15:15:00 | 1916.89 | TARGET | 167.89 |
| BUY | 2025-03-10 09:15:00 | 1823.90 | 2025-04-22 10:15:00 | 2226.37 | TARGET | 402.47 |
| BUY | 2025-04-07 15:15:00 | 2024.95 | 2025-05-02 09:15:00 | 1904.80 | EXIT_EMA400 | -120.15 |
| SELL | 2025-06-02 09:15:00 | 1799.70 | 2025-06-06 12:15:00 | 1856.30 | EXIT_EMA400 | -56.60 |
| BUY | 2025-07-08 09:15:00 | 1961.90 | 2025-07-18 11:15:00 | 1891.00 | EXIT_EMA400 | -70.90 |
| BUY | 2025-07-08 11:15:00 | 1982.30 | 2025-07-18 11:15:00 | 1891.00 | EXIT_EMA400 | -91.30 |
| SELL | 2025-10-07 12:15:00 | 1662.50 | 2025-10-15 12:15:00 | 1587.04 | TARGET | 75.46 |
| SELL | 2025-09-25 11:15:00 | 1632.20 | 2025-10-23 09:15:00 | 1676.00 | EXIT_EMA400 | -43.80 |
