# J.B. Chemicals & Pharmaceuticals Ltd. (JBCHEPHARM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2049.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / EMA400 exits:** 5 / 2
- **Total realized P&L (per unit):** 395.81
- **Avg P&L per closed trade:** 56.54

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 1717.05 | 1877.80 | 1878.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 1686.20 | 1875.89 | 1877.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 1867.65 | 1846.04 | 1860.35 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 1915.70 | 1870.20 | 1870.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 12:15:00 | 1918.95 | 1870.68 | 1870.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 11:15:00 | 1871.50 | 1874.55 | 1872.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-06 15:15:00 | 1890.00 | 1872.85 | 1871.67 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1880.00 | 1872.92 | 1871.71 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-11-07 10:15:00 | 1871.55 | 1872.91 | 1871.71 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 1840.00 | 1870.30 | 1870.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1770.65 | 1869.01 | 1869.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1810.80 | 1805.03 | 1833.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 12:15:00 | 1762.00 | 1804.05 | 1832.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1816.00 | 1785.14 | 1816.04 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-04 10:15:00 | 1818.95 | 1785.48 | 1816.06 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 1852.00 | 1825.87 | 1825.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 1885.00 | 1826.46 | 1826.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 13:15:00 | 1826.10 | 1830.30 | 1828.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-07 09:15:00 | 1837.00 | 1830.30 | 1828.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1845.10 | 1839.82 | 1833.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-10 11:15:00 | 1802.70 | 1839.41 | 1833.27 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 13:15:00 | 1766.00 | 1827.83 | 1827.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 1748.40 | 1821.67 | 1824.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 1841.70 | 1817.29 | 1822.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-21 15:15:00 | 1790.00 | 1816.38 | 1821.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 1673.80 | 1617.92 | 1669.76 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 1668.90 | 1628.96 | 1628.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 1690.50 | 1631.26 | 1630.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 1680.80 | 1681.34 | 1661.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 1704.70 | 1681.35 | 1661.69 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1685.00 | 1712.81 | 1682.74 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-30 10:15:00 | 1680.80 | 1712.49 | 1682.73 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 1653.10 | 1698.92 | 1698.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 13:15:00 | 1650.00 | 1695.44 | 1697.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 10:15:00 | 1685.00 | 1684.72 | 1690.67 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1803.20 | 1694.19 | 1694.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 15:15:00 | 1815.40 | 1700.73 | 1697.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1735.10 | 1743.68 | 1723.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-24 10:15:00 | 1749.00 | 1743.00 | 1723.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1854.50 | 1853.66 | 1820.60 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-30 12:15:00 | 1869.60 | 1853.93 | 1821.23 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1998.00 | 2052.50 | 1991.96 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-04-02 09:15:00 | 1908.60 | 2050.11 | 1991.66 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-11-06 15:15:00 | 1890.00 | 2024-11-07 10:15:00 | 1871.55 | EXIT_EMA400 | -18.45 |
| SELL | 2024-11-25 12:15:00 | 1762.00 | 2024-12-04 10:15:00 | 1818.95 | EXIT_EMA400 | -56.95 |
| BUY | 2025-01-07 09:15:00 | 1837.00 | 2025-01-07 12:15:00 | 1863.42 | TARGET | 26.42 |
| SELL | 2025-01-21 15:15:00 | 1790.00 | 2025-01-28 09:15:00 | 1695.00 | TARGET | 95.00 |
| BUY | 2025-06-20 09:15:00 | 1704.70 | 2025-06-25 09:15:00 | 1833.73 | TARGET | 129.03 |
| BUY | 2025-11-24 10:15:00 | 1749.00 | 2025-12-04 13:15:00 | 1824.65 | TARGET | 75.65 |
| BUY | 2026-01-30 12:15:00 | 1869.60 | 2026-02-19 09:15:00 | 2014.72 | TARGET | 145.12 |
