# ICICI Lombard General Insurance Company Ltd. (ICICIGI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1763.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 7 |
| ENTRY1 | 8 |
| ENTRY2 | 3 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / EMA400 exits:** 3 / 8
- **Total realized P&L (per unit):** -37.22
- **Avg P&L per closed trade:** -3.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 09:15:00 | 1315.70 | 1326.71 | 1326.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-11 10:15:00 | 1312.65 | 1326.57 | 1326.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 11:15:00 | 1326.75 | 1325.78 | 1326.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-12 13:15:00 | 1318.25 | 1325.70 | 1326.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 13:15:00 | 1318.25 | 1325.70 | 1326.22 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-10-12 15:15:00 | 1326.80 | 1325.70 | 1326.21 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 11:15:00 | 1346.35 | 1326.84 | 1326.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 13:15:00 | 1357.05 | 1327.35 | 1327.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 09:15:00 | 1342.55 | 1346.01 | 1337.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-31 09:15:00 | 1363.90 | 1346.38 | 1338.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 1347.10 | 1354.67 | 1345.05 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-11-13 11:15:00 | 1356.45 | 1354.53 | 1345.27 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 1406.35 | 1431.91 | 1405.24 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-12-21 12:15:00 | 1409.55 | 1431.17 | 1405.27 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-12-21 13:15:00 | 1405.10 | 1430.91 | 1405.27 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 09:15:00 | 1579.95 | 1640.50 | 1640.73 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 12:15:00 | 1676.00 | 1640.99 | 1640.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 13:15:00 | 1686.25 | 1641.44 | 1641.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 2096.35 | 2101.82 | 2005.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-17 14:15:00 | 2105.25 | 2101.34 | 2010.46 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-08 11:15:00 | 2063.05 | 2149.50 | 2076.17 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 1918.25 | 2046.16 | 2046.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 09:15:00 | 1905.70 | 2042.46 | 2044.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 1919.90 | 1906.13 | 1953.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-18 13:15:00 | 1882.20 | 1926.15 | 1950.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 1897.00 | 1867.82 | 1900.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-16 09:15:00 | 1912.20 | 1868.52 | 1900.29 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 09:15:00 | 1862.00 | 1790.83 | 1790.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 11:15:00 | 1909.30 | 1829.45 | 1814.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 1839.50 | 1841.28 | 1823.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-28 12:15:00 | 1854.20 | 1841.40 | 1824.43 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-18 09:15:00 | 1938.90 | 1982.36 | 1939.37 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 1847.80 | 1923.01 | 1923.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 10:15:00 | 1832.20 | 1917.60 | 1920.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 1898.10 | 1883.92 | 1899.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-17 13:15:00 | 1868.00 | 1886.70 | 1899.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1891.70 | 1884.52 | 1897.00 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-24 10:15:00 | 1871.60 | 1884.80 | 1896.71 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-26 09:15:00 | 1900.10 | 1884.15 | 1895.62 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 2005.30 | 1900.23 | 1900.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 2027.80 | 1908.36 | 1904.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 1995.70 | 1996.83 | 1966.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-25 09:15:00 | 2008.00 | 1996.90 | 1967.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1970.70 | 1996.41 | 1969.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-28 10:15:00 | 1959.10 | 1996.04 | 1969.73 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 1957.40 | 1960.11 | 1960.12 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1967.70 | 1960.16 | 1960.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1979.90 | 1960.35 | 1960.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 15:15:00 | 1962.90 | 1965.75 | 1963.11 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1895.40 | 1960.61 | 1960.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 1882.00 | 1956.60 | 1958.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 15:15:00 | 1877.00 | 1876.64 | 1907.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-06 09:15:00 | 1863.20 | 1876.50 | 1906.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1899.90 | 1875.67 | 1904.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-10 12:15:00 | 1905.90 | 1876.20 | 1904.13 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-12 13:15:00 | 1318.25 | 2023-10-12 15:15:00 | 1326.80 | EXIT_EMA400 | -8.55 |
| BUY | 2023-11-13 11:15:00 | 1356.45 | 2023-11-15 11:15:00 | 1390.00 | TARGET | 33.55 |
| BUY | 2023-10-31 09:15:00 | 1363.90 | 2023-11-17 09:15:00 | 1441.21 | TARGET | 77.31 |
| BUY | 2023-12-21 12:15:00 | 1409.55 | 2023-12-21 13:15:00 | 1405.10 | EXIT_EMA400 | -4.45 |
| BUY | 2024-09-17 14:15:00 | 2105.25 | 2024-10-08 11:15:00 | 2063.05 | EXIT_EMA400 | -42.20 |
| SELL | 2024-12-18 13:15:00 | 1882.20 | 2025-01-16 09:15:00 | 1912.20 | EXIT_EMA400 | -30.00 |
| BUY | 2025-05-28 12:15:00 | 1854.20 | 2025-06-06 13:15:00 | 1943.52 | TARGET | 89.32 |
| SELL | 2025-09-17 13:15:00 | 1868.00 | 2025-09-26 09:15:00 | 1900.10 | EXIT_EMA400 | -32.10 |
| SELL | 2025-09-24 10:15:00 | 1871.60 | 2025-09-26 09:15:00 | 1900.10 | EXIT_EMA400 | -28.50 |
| BUY | 2025-11-25 09:15:00 | 2008.00 | 2025-11-28 10:15:00 | 1959.10 | EXIT_EMA400 | -48.90 |
| SELL | 2026-02-06 09:15:00 | 1863.20 | 2026-02-10 12:15:00 | 1905.90 | EXIT_EMA400 | -42.70 |
