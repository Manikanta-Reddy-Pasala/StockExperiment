# Dalmia Bharat Ltd. (DALBHARAT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 1906.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT3 | 7 |
| ENTRY1 | 9 |
| ENTRY2 | 3 |
| EXIT | 9 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / EMA400 exits:** 5 / 7
- **Total realized P&L (per unit):** 265.43
- **Avg P&L per closed trade:** 22.12

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 15:15:00 | 2308.00 | 2061.52 | 2061.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 11:15:00 | 2318.80 | 2083.89 | 2072.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-05 09:15:00 | 2243.30 | 2262.30 | 2192.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-11 09:15:00 | 2317.80 | 2255.91 | 2198.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-10-19 09:15:00 | 2197.00 | 2263.97 | 2213.86 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 10:15:00 | 2077.05 | 2178.98 | 2179.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 14:15:00 | 2070.95 | 2168.46 | 2173.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 12:15:00 | 2149.60 | 2145.02 | 2159.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-15 14:15:00 | 2124.40 | 2144.77 | 2159.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-16 14:15:00 | 2161.00 | 2144.43 | 2158.43 | Close above EMA400 |

### Cycle 3 — BUY (started 2023-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 15:15:00 | 2203.65 | 2169.04 | 2168.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 2221.55 | 2169.57 | 2169.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 2231.85 | 2277.47 | 2235.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-27 09:15:00 | 2295.85 | 2265.01 | 2233.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 2272.55 | 2298.50 | 2263.90 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-11 11:15:00 | 2290.95 | 2298.43 | 2264.03 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 2274.15 | 2298.18 | 2267.34 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-16 12:15:00 | 2244.00 | 2297.65 | 2267.23 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 2150.60 | 2247.14 | 2247.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 11:15:00 | 2135.05 | 2246.03 | 2246.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 11:15:00 | 1988.60 | 1986.27 | 2059.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-02 09:15:00 | 1979.45 | 1986.43 | 2057.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 2050.05 | 1990.57 | 2055.19 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-04 11:15:00 | 2023.90 | 1991.45 | 2054.99 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 1864.80 | 1808.81 | 1868.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-10 14:15:00 | 1877.90 | 1810.00 | 1868.42 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 14:15:00 | 1912.60 | 1830.96 | 1830.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 1932.65 | 1853.07 | 1844.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 12:15:00 | 1867.45 | 1887.32 | 1866.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 09:15:00 | 1897.25 | 1886.58 | 1866.10 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1897.25 | 1886.58 | 1866.10 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-08 10:15:00 | 1908.00 | 1886.80 | 1866.31 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-09 11:15:00 | 1859.90 | 1886.64 | 1867.03 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 1789.10 | 1856.50 | 1856.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 14:15:00 | 1771.95 | 1854.20 | 1855.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 12:15:00 | 1846.50 | 1845.00 | 1850.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-04 09:15:00 | 1809.00 | 1843.84 | 1849.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1816.65 | 1796.21 | 1819.47 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 1853.30 | 1797.35 | 1819.47 | Close above EMA400 |

### Cycle 7 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 1925.85 | 1833.79 | 1833.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 1950.05 | 1845.06 | 1839.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 1857.05 | 1873.16 | 1856.08 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 12:15:00 | 1734.95 | 1841.87 | 1842.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 1724.95 | 1814.22 | 1825.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 09:15:00 | 1791.80 | 1782.76 | 1805.04 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 13:15:00 | 1845.10 | 1817.04 | 1816.98 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 14:15:00 | 1786.05 | 1816.92 | 1816.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 1747.05 | 1815.96 | 1816.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1821.70 | 1814.82 | 1815.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-13 13:15:00 | 1797.95 | 1814.59 | 1815.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 1744.50 | 1722.25 | 1754.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-25 09:15:00 | 1765.15 | 1724.42 | 1753.74 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 1810.55 | 1772.10 | 1772.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1853.00 | 1773.33 | 1772.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 2048.40 | 2056.92 | 1991.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 11:15:00 | 2062.80 | 2055.32 | 1996.68 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-25 09:15:00 | 2293.10 | 2369.16 | 2304.32 | Close below EMA400 |

### Cycle 12 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 2185.40 | 2270.40 | 2270.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 09:15:00 | 2150.00 | 2269.20 | 2269.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 2050.00 | 2030.63 | 2093.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-19 11:15:00 | 2009.40 | 2039.67 | 2087.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-24 09:15:00 | 2104.40 | 2038.75 | 2083.04 | Close above EMA400 |

### Cycle 13 — BUY (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 15:15:00 | 2165.00 | 2105.83 | 2105.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 09:15:00 | 2198.80 | 2106.75 | 2106.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 2113.10 | 2116.82 | 2111.31 | EMA200 retest candle locked |

### Cycle 14 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 2018.60 | 2106.14 | 2106.55 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 13:15:00 | 2153.30 | 2106.59 | 2106.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 2163.10 | 2108.89 | 2107.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 12:15:00 | 2126.20 | 2127.30 | 2118.46 | EMA200 retest candle locked |

### Cycle 16 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 2063.50 | 2111.90 | 2111.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 2019.40 | 2107.63 | 2109.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1943.20 | 1889.95 | 1960.25 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-11 09:15:00 | 2317.80 | 2023-10-19 09:15:00 | 2197.00 | EXIT_EMA400 | -120.80 |
| SELL | 2023-11-15 14:15:00 | 2124.40 | 2023-11-16 14:15:00 | 2161.00 | EXIT_EMA400 | -36.60 |
| BUY | 2023-12-27 09:15:00 | 2295.85 | 2024-01-16 12:15:00 | 2244.00 | EXIT_EMA400 | -51.85 |
| BUY | 2024-01-11 11:15:00 | 2290.95 | 2024-01-16 12:15:00 | 2244.00 | EXIT_EMA400 | -46.95 |
| SELL | 2024-04-04 11:15:00 | 2023.90 | 2024-04-15 09:15:00 | 1930.64 | TARGET | 93.26 |
| SELL | 2024-04-02 09:15:00 | 1979.45 | 2024-05-08 09:15:00 | 1744.07 | TARGET | 235.38 |
| BUY | 2024-10-08 09:15:00 | 1897.25 | 2024-10-09 11:15:00 | 1859.90 | EXIT_EMA400 | -37.35 |
| BUY | 2024-10-08 10:15:00 | 1908.00 | 2024-10-09 11:15:00 | 1859.90 | EXIT_EMA400 | -48.10 |
| SELL | 2024-11-04 09:15:00 | 1809.00 | 2024-11-14 09:15:00 | 1687.26 | TARGET | 121.74 |
| SELL | 2025-02-13 13:15:00 | 1797.95 | 2025-02-21 09:15:00 | 1744.60 | TARGET | 53.35 |
| BUY | 2025-06-23 11:15:00 | 2062.80 | 2025-07-21 09:15:00 | 2261.15 | TARGET | 198.35 |
| SELL | 2025-12-19 11:15:00 | 2009.40 | 2025-12-24 09:15:00 | 2104.40 | EXIT_EMA400 | -95.00 |
