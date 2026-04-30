# Dalmia Bharat Ltd. (DALBHARAT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1905.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 68.83
- **Avg P&L per closed trade:** 9.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 1742.95 | 1804.84 | 1805.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 1732.20 | 1802.84 | 1804.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 11:15:00 | 1796.40 | 1786.64 | 1794.76 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 1894.00 | 1801.57 | 1801.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 1926.95 | 1804.53 | 1802.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 1842.00 | 1852.23 | 1832.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-23 09:15:00 | 1865.55 | 1847.31 | 1832.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-07 13:15:00 | 1853.85 | 1886.81 | 1859.17 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 1781.80 | 1847.88 | 1848.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 1771.25 | 1830.14 | 1838.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 10:15:00 | 1797.55 | 1795.99 | 1817.12 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 15:15:00 | 1911.35 | 1830.20 | 1829.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 11:15:00 | 1925.00 | 1832.82 | 1831.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 1857.05 | 1873.14 | 1854.85 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 1740.00 | 1839.93 | 1840.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 15:15:00 | 1735.05 | 1838.89 | 1839.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 13:15:00 | 1825.55 | 1824.56 | 1832.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 1776.60 | 1823.68 | 1831.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1776.60 | 1823.68 | 1831.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-06 12:15:00 | 1771.15 | 1822.70 | 1830.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1827.55 | 1819.02 | 1828.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-10 09:15:00 | 1763.50 | 1816.91 | 1826.57 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1798.50 | 1783.32 | 1804.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-23 10:15:00 | 1818.50 | 1784.00 | 1804.29 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 14:15:00 | 1866.25 | 1816.49 | 1816.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 10:15:00 | 1872.55 | 1818.00 | 1817.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 11:15:00 | 1815.35 | 1819.91 | 1818.10 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 15:15:00 | 1795.70 | 1816.39 | 1816.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 1776.60 | 1815.39 | 1815.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 1748.00 | 1724.13 | 1757.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-20 09:15:00 | 1703.20 | 1724.17 | 1757.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 1744.50 | 1722.48 | 1754.24 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-25 09:15:00 | 1764.90 | 1724.64 | 1753.94 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 1810.55 | 1772.27 | 1772.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1853.00 | 1773.46 | 1772.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 2048.40 | 2056.71 | 1991.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 11:15:00 | 2062.80 | 2055.14 | 1996.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-25 09:15:00 | 2293.10 | 2369.34 | 2304.37 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 2184.00 | 2270.17 | 2270.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 09:15:00 | 2150.00 | 2268.98 | 2269.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 2049.30 | 2030.56 | 2093.42 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 15:15:00 | 2179.70 | 2105.61 | 2105.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 09:15:00 | 2198.80 | 2106.54 | 2105.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 2113.10 | 2116.62 | 2111.17 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 2048.30 | 2106.44 | 2106.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2034.40 | 2105.47 | 2106.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 2098.80 | 2097.89 | 2102.17 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 2162.10 | 2106.32 | 2106.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 2176.80 | 2107.02 | 2106.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 2116.10 | 2125.00 | 2116.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-18 12:15:00 | 2131.90 | 2124.70 | 2116.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-19 11:15:00 | 2107.50 | 2124.76 | 2117.14 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 2046.80 | 2110.31 | 2110.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 2019.40 | 2106.15 | 2108.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1943.20 | 1889.82 | 1959.76 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-23 09:15:00 | 1865.55 | 2024-09-27 09:15:00 | 1966.07 | TARGET | 100.52 |
| SELL | 2025-01-06 10:15:00 | 1776.60 | 2025-01-23 10:15:00 | 1818.50 | EXIT_EMA400 | -41.90 |
| SELL | 2025-01-06 12:15:00 | 1771.15 | 2025-01-23 10:15:00 | 1818.50 | EXIT_EMA400 | -47.35 |
| SELL | 2025-01-10 09:15:00 | 1763.50 | 2025-01-23 10:15:00 | 1818.50 | EXIT_EMA400 | -55.00 |
| SELL | 2025-03-20 09:15:00 | 1703.20 | 2025-03-25 09:15:00 | 1764.90 | EXIT_EMA400 | -61.70 |
| BUY | 2025-06-23 11:15:00 | 2062.80 | 2025-07-21 09:15:00 | 2261.46 | TARGET | 198.66 |
| BUY | 2026-02-18 12:15:00 | 2131.90 | 2026-02-19 11:15:00 | 2107.50 | EXIT_EMA400 | -24.40 |
