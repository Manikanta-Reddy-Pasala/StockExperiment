# ACC Ltd. (ACC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1422.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 659.91
- **Avg P&L per closed trade:** 94.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 1843.75 | 1961.15 | 1961.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 09:15:00 | 1827.60 | 1925.83 | 1942.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 11:15:00 | 1889.65 | 1879.44 | 1909.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-28 14:15:00 | 1870.85 | 1879.34 | 1909.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 1890.40 | 1877.94 | 1906.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-12-04 09:15:00 | 1980.00 | 1880.24 | 1906.60 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 12:15:00 | 2144.50 | 1929.95 | 1929.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 2153.60 | 1938.41 | 1934.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 2221.00 | 2221.46 | 2132.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-19 14:15:00 | 2278.00 | 2225.18 | 2139.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 2470.00 | 2586.52 | 2466.47 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-13 11:15:00 | 2435.75 | 2585.02 | 2466.31 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 11:15:00 | 2385.60 | 2577.23 | 2578.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 12:15:00 | 2378.60 | 2575.25 | 2577.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 2408.25 | 2395.72 | 2456.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-07 11:15:00 | 2354.55 | 2452.20 | 2462.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 11:15:00 | 2303.95 | 2237.55 | 2297.82 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 15:15:00 | 2080.00 | 1973.62 | 1973.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 2087.40 | 1980.21 | 1976.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 1961.00 | 1986.10 | 1980.07 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 1884.00 | 1974.03 | 1974.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 10:15:00 | 1876.10 | 1967.18 | 1970.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 11:15:00 | 1927.40 | 1912.07 | 1937.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-30 09:15:00 | 1902.80 | 1924.37 | 1937.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1926.50 | 1910.85 | 1926.53 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-11 13:15:00 | 1898.80 | 1911.18 | 1926.01 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-06-27 09:15:00 | 1917.70 | 1878.74 | 1901.77 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 1999.70 | 1917.26 | 1917.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 2002.70 | 1918.11 | 1917.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1922.00 | 1943.93 | 1932.55 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 1829.00 | 1923.28 | 1923.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 1814.20 | 1922.20 | 1922.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 1855.40 | 1851.52 | 1879.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 10:15:00 | 1833.90 | 1852.76 | 1877.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1851.00 | 1837.95 | 1862.68 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-04 12:15:00 | 1850.20 | 1838.39 | 1862.53 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 1860.00 | 1840.99 | 1858.95 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-28 14:15:00 | 1870.85 | 2023-12-04 09:15:00 | 1980.00 | EXIT_EMA400 | -109.15 |
| BUY | 2024-01-19 14:15:00 | 2278.00 | 2024-02-16 09:15:00 | 2692.23 | TARGET | 414.23 |
| SELL | 2024-10-07 11:15:00 | 2354.55 | 2024-11-21 09:15:00 | 2030.56 | TARGET | 323.99 |
| SELL | 2025-06-11 13:15:00 | 1898.80 | 2025-06-20 09:15:00 | 1817.16 | TARGET | 81.64 |
| SELL | 2025-05-30 09:15:00 | 1902.80 | 2025-06-27 09:15:00 | 1917.70 | EXIT_EMA400 | -14.90 |
| SELL | 2025-08-22 10:15:00 | 1833.90 | 2025-09-15 09:15:00 | 1860.00 | EXIT_EMA400 | -26.10 |
| SELL | 2025-09-04 12:15:00 | 1850.20 | 2025-09-15 09:15:00 | 1860.00 | EXIT_EMA400 | -9.80 |
