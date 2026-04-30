# Astral Ltd. (ASTRAL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1529.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT3 | 5 |
| ENTRY1 | 10 |
| ENTRY2 | 2 |
| EXIT | 10 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 1 / 11
- **Target hits / EMA400 exits:** 0 / 12
- **Total realized P&L (per unit):** -502.75
- **Avg P&L per closed trade:** -41.90

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 14:15:00 | 1841.30 | 1922.95 | 1923.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 09:15:00 | 1813.00 | 1921.08 | 1922.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 11:15:00 | 1916.95 | 1916.36 | 1919.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-09-28 12:15:00 | 1901.20 | 1916.74 | 1919.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-10-10 14:15:00 | 1912.75 | 1903.47 | 1911.68 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 15:15:00 | 1963.40 | 1918.13 | 1918.05 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 1851.00 | 1917.46 | 1917.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 14:15:00 | 1841.95 | 1914.09 | 1916.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 14:15:00 | 1872.70 | 1871.35 | 1889.04 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 15:15:00 | 1934.00 | 1897.84 | 1897.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 1946.10 | 1898.32 | 1897.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 13:15:00 | 1924.25 | 1927.27 | 1914.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-11 14:15:00 | 1951.30 | 1927.65 | 1915.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 1929.65 | 1928.67 | 1916.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-12-13 10:15:00 | 1901.20 | 1928.30 | 1916.04 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-01-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 12:15:00 | 1840.75 | 1912.40 | 1912.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 09:15:00 | 1830.35 | 1905.57 | 1908.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 11:15:00 | 1859.05 | 1846.49 | 1871.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-31 10:15:00 | 1810.50 | 1849.09 | 1871.17 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-02-01 09:15:00 | 1874.60 | 1848.22 | 1870.07 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 14:15:00 | 1968.95 | 1883.46 | 1883.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 10:15:00 | 1978.30 | 1891.46 | 1887.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 2005.45 | 2005.50 | 1958.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-14 11:15:00 | 2033.25 | 2005.77 | 1960.68 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 1977.05 | 2005.78 | 1961.80 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-15 13:15:00 | 2023.35 | 2005.65 | 1962.60 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 1966.65 | 2005.32 | 1964.13 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-19 09:15:00 | 1917.90 | 2004.11 | 1963.93 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 2031.35 | 2196.71 | 2196.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1928.25 | 2194.04 | 2195.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 1971.00 | 1958.73 | 2022.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-03 12:15:00 | 1930.30 | 1972.64 | 2015.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 13:15:00 | 1851.70 | 1797.66 | 1848.67 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 1513.80 | 1378.35 | 1377.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 12:15:00 | 1520.40 | 1379.76 | 1378.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 1486.30 | 1493.92 | 1459.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 14:15:00 | 1507.50 | 1490.83 | 1467.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1475.00 | 1493.69 | 1473.24 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-23 09:15:00 | 1459.90 | 1493.36 | 1473.17 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 1409.90 | 1459.99 | 1460.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 1401.50 | 1456.04 | 1458.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1439.40 | 1402.74 | 1426.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 10:15:00 | 1390.60 | 1406.13 | 1426.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-02 11:15:00 | 1426.40 | 1400.21 | 1420.18 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 1450.00 | 1431.59 | 1431.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 1455.00 | 1432.72 | 1432.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 11:15:00 | 1432.60 | 1432.99 | 1432.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-23 14:15:00 | 1437.70 | 1433.08 | 1432.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1436.40 | 1433.11 | 1432.31 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-24 12:15:00 | 1443.40 | 1433.26 | 1432.40 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-25 09:15:00 | 1402.80 | 1433.22 | 1432.40 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1404.90 | 1431.60 | 1431.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1400.50 | 1431.29 | 1431.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 13:15:00 | 1417.40 | 1411.72 | 1420.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-08 15:15:00 | 1400.00 | 1411.39 | 1419.89 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 11:15:00 | 1422.40 | 1410.92 | 1419.23 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 1444.40 | 1424.71 | 1424.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 1455.00 | 1425.69 | 1425.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 1468.50 | 1482.65 | 1458.41 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 1405.70 | 1449.25 | 1449.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 1395.90 | 1436.18 | 1442.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 13:15:00 | 1426.70 | 1420.68 | 1432.62 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 1478.30 | 1441.71 | 1441.68 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 1416.50 | 1441.60 | 1441.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 1404.00 | 1441.22 | 1441.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 1428.80 | 1426.61 | 1433.48 | EMA200 retest candle locked |

### Cycle 16 — BUY (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 10:15:00 | 1499.30 | 1439.03 | 1438.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 1501.20 | 1441.30 | 1440.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 15:15:00 | 1610.00 | 1610.70 | 1556.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-16 09:15:00 | 1617.00 | 1610.76 | 1556.36 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 1563.50 | 1618.80 | 1569.47 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-09-28 12:15:00 | 1901.20 | 2023-10-10 14:15:00 | 1912.75 | EXIT_EMA400 | -11.55 |
| BUY | 2023-12-11 14:15:00 | 1951.30 | 2023-12-13 10:15:00 | 1901.20 | EXIT_EMA400 | -50.10 |
| SELL | 2024-01-31 10:15:00 | 1810.50 | 2024-02-01 09:15:00 | 1874.60 | EXIT_EMA400 | -64.10 |
| BUY | 2024-03-14 11:15:00 | 2033.25 | 2024-03-19 09:15:00 | 1917.90 | EXIT_EMA400 | -115.35 |
| BUY | 2024-03-15 13:15:00 | 2023.35 | 2024-03-19 09:15:00 | 1917.90 | EXIT_EMA400 | -105.45 |
| SELL | 2024-10-03 12:15:00 | 1930.30 | 2024-12-03 13:15:00 | 1851.70 | EXIT_EMA400 | 78.60 |
| BUY | 2025-07-15 14:15:00 | 1507.50 | 2025-07-23 09:15:00 | 1459.90 | EXIT_EMA400 | -47.60 |
| SELL | 2025-08-26 10:15:00 | 1390.60 | 2025-09-02 11:15:00 | 1426.40 | EXIT_EMA400 | -35.80 |
| BUY | 2025-09-23 14:15:00 | 1437.70 | 2025-09-25 09:15:00 | 1402.80 | EXIT_EMA400 | -34.90 |
| BUY | 2025-09-24 12:15:00 | 1443.40 | 2025-09-25 09:15:00 | 1402.80 | EXIT_EMA400 | -40.60 |
| SELL | 2025-10-08 15:15:00 | 1400.00 | 2025-10-10 11:15:00 | 1422.40 | EXIT_EMA400 | -22.40 |
| BUY | 2026-03-16 09:15:00 | 1617.00 | 2026-03-23 09:15:00 | 1563.50 | EXIT_EMA400 | -53.50 |
