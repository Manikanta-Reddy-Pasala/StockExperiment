# Bombay Burmah Trading Corporation Ltd. (BBTC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5005 bars)
- **Last close:** 1501.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 8 |
| ENTRY1 | 8 |
| ENTRY2 | 5 |
| EXIT | 8 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / EMA400 exits:** 5 / 8
- **Total realized P&L (per unit):** 946.30
- **Avg P&L per closed trade:** 72.79

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 12:15:00 | 983.75 | 1030.59 | 1030.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 13:15:00 | 982.00 | 1030.11 | 1030.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 12:15:00 | 1022.70 | 1020.05 | 1024.96 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 13:15:00 | 1092.50 | 1029.55 | 1029.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 15:15:00 | 1096.80 | 1030.87 | 1030.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 14:15:00 | 1189.90 | 1192.21 | 1142.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-20 09:15:00 | 1366.00 | 1193.14 | 1149.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 1343.30 | 1394.58 | 1339.98 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-12-12 15:15:00 | 1339.05 | 1394.03 | 1339.97 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 1498.00 | 1608.37 | 1608.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 1483.90 | 1573.46 | 1587.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 12:15:00 | 1576.80 | 1568.29 | 1583.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-16 10:15:00 | 1548.45 | 1567.80 | 1582.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1568.80 | 1565.45 | 1580.72 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-21 15:15:00 | 1551.00 | 1565.39 | 1580.32 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1571.15 | 1565.40 | 1580.18 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-22 12:15:00 | 1548.05 | 1565.17 | 1579.91 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 1536.15 | 1527.35 | 1553.82 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-06-06 14:15:00 | 1525.60 | 1527.40 | 1553.58 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-06-07 10:15:00 | 1608.00 | 1528.35 | 1553.67 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 1637.80 | 1573.08 | 1572.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 1697.80 | 1577.69 | 1575.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 2127.50 | 2131.20 | 1971.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-06 09:15:00 | 2211.75 | 2131.72 | 1976.27 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-07 15:15:00 | 2482.85 | 2647.02 | 2484.84 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 14:15:00 | 2405.75 | 2569.89 | 2570.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 15:15:00 | 2398.00 | 2568.18 | 2569.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 2226.05 | 2139.14 | 2264.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-28 09:15:00 | 2026.35 | 2143.15 | 2253.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1883.75 | 1796.46 | 1905.33 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-03 09:15:00 | 1859.70 | 1798.94 | 1904.96 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-15 09:15:00 | 1894.40 | 1790.98 | 1880.44 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 13:15:00 | 1998.50 | 1895.91 | 1895.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 2019.00 | 1899.03 | 1896.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1927.90 | 1962.84 | 1937.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 12:15:00 | 1982.00 | 1939.85 | 1931.28 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1954.00 | 1958.74 | 1944.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-09 15:15:00 | 1940.10 | 1958.62 | 1944.53 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 1861.60 | 1945.75 | 1945.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 1857.00 | 1943.29 | 1944.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 1888.10 | 1856.85 | 1888.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-05 10:15:00 | 1851.40 | 1858.19 | 1888.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 1893.00 | 1858.08 | 1885.11 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 2046.30 | 1886.95 | 1886.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 2048.30 | 1888.56 | 1887.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 15:15:00 | 1945.40 | 1947.20 | 1923.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-18 10:15:00 | 1982.80 | 1920.96 | 1914.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 1920.00 | 1928.48 | 1918.98 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-20 14:15:00 | 1917.50 | 1928.37 | 1918.97 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 14:15:00 | 1839.80 | 1910.63 | 1910.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 1836.70 | 1904.18 | 1907.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 1914.30 | 1890.60 | 1899.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-03 14:15:00 | 1870.50 | 1890.53 | 1899.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1870.50 | 1890.53 | 1899.80 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-04 09:15:00 | 1866.70 | 1890.17 | 1899.53 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-09 14:15:00 | 1897.00 | 1879.61 | 1892.77 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-20 09:15:00 | 1366.00 | 2023-12-12 15:15:00 | 1339.05 | EXIT_EMA400 | -26.95 |
| SELL | 2024-05-21 15:15:00 | 1551.00 | 2024-05-31 11:15:00 | 1463.04 | TARGET | 87.96 |
| SELL | 2024-05-16 10:15:00 | 1548.45 | 2024-06-04 09:15:00 | 1444.90 | TARGET | 103.55 |
| SELL | 2024-05-22 12:15:00 | 1548.05 | 2024-06-04 09:15:00 | 1452.47 | TARGET | 95.58 |
| SELL | 2024-06-06 14:15:00 | 1525.60 | 2024-06-07 10:15:00 | 1608.00 | EXIT_EMA400 | -82.40 |
| BUY | 2024-08-06 09:15:00 | 2211.75 | 2024-10-01 09:15:00 | 2918.18 | TARGET | 706.43 |
| SELL | 2025-04-03 09:15:00 | 1859.70 | 2025-04-07 09:15:00 | 1723.92 | TARGET | 135.78 |
| SELL | 2025-01-28 09:15:00 | 2026.35 | 2025-04-15 09:15:00 | 1894.40 | EXIT_EMA400 | 131.95 |
| BUY | 2025-06-27 12:15:00 | 1982.00 | 2025-07-09 15:15:00 | 1940.10 | EXIT_EMA400 | -41.90 |
| SELL | 2025-09-05 10:15:00 | 1851.40 | 2025-09-10 09:15:00 | 1893.00 | EXIT_EMA400 | -41.60 |
| BUY | 2025-11-18 10:15:00 | 1982.80 | 2025-11-20 14:15:00 | 1917.50 | EXIT_EMA400 | -65.30 |
| SELL | 2025-12-03 14:15:00 | 1870.50 | 2025-12-09 14:15:00 | 1897.00 | EXIT_EMA400 | -26.50 |
| SELL | 2025-12-04 09:15:00 | 1866.70 | 2025-12-09 14:15:00 | 1897.00 | EXIT_EMA400 | -30.30 |
