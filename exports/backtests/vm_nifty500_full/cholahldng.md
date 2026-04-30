# Cholamandalam Financial Holdings Ltd. (CHOLAHLDNG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1556.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 6 |
| ENTRY1 | 9 |
| ENTRY2 | 1 |
| EXIT | 9 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 1 / 9
- **Target hits / EMA400 exits:** 0 / 10
- **Total realized P&L (per unit):** -282.65
- **Avg P&L per closed trade:** -28.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 09:15:00 | 1050.30 | 1072.18 | 1072.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 09:15:00 | 1035.00 | 1070.47 | 1071.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 09:15:00 | 1075.90 | 1057.96 | 1064.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-15 11:15:00 | 1061.40 | 1058.08 | 1064.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 11:15:00 | 1061.40 | 1058.08 | 1064.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-12-15 13:15:00 | 1069.20 | 1058.22 | 1064.32 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 14:15:00 | 1163.20 | 1050.03 | 1049.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 1175.00 | 1052.40 | 1050.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 12:15:00 | 1091.80 | 1094.09 | 1075.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-12 14:15:00 | 1127.95 | 1094.42 | 1075.83 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-02-15 15:15:00 | 1078.00 | 1096.35 | 1078.82 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 15:15:00 | 1031.15 | 1069.92 | 1070.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 10:15:00 | 1017.30 | 1066.60 | 1068.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 09:15:00 | 1068.55 | 1055.78 | 1062.30 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 15:15:00 | 1136.00 | 1068.16 | 1067.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 09:15:00 | 1172.40 | 1069.20 | 1068.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 1082.65 | 1094.20 | 1082.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-02 10:15:00 | 1127.55 | 1088.48 | 1083.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 1089.35 | 1097.71 | 1088.45 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-07 13:15:00 | 1099.35 | 1097.62 | 1088.54 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 1094.45 | 1097.55 | 1088.77 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-09 09:15:00 | 1078.95 | 1097.18 | 1088.76 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 1540.10 | 1762.71 | 1762.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 10:15:00 | 1511.65 | 1760.21 | 1761.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 14:15:00 | 1502.00 | 1501.79 | 1577.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-13 09:15:00 | 1477.25 | 1515.12 | 1569.96 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 1499.45 | 1470.52 | 1523.30 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-30 09:15:00 | 1528.05 | 1471.10 | 1523.33 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 12:15:00 | 1605.05 | 1527.38 | 1527.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 14:15:00 | 1618.00 | 1529.07 | 1528.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1553.55 | 1651.40 | 1605.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 12:15:00 | 1582.05 | 1649.05 | 1604.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 12:15:00 | 1582.05 | 1649.05 | 1604.79 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-07 13:15:00 | 1599.85 | 1648.57 | 1604.76 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1847.90 | 1943.60 | 1943.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 1826.20 | 1942.44 | 1943.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1938.60 | 1928.56 | 1935.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-19 12:15:00 | 1872.10 | 1928.11 | 1935.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-11 11:15:00 | 1883.60 | 1833.95 | 1873.93 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1942.00 | 1886.17 | 1885.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1962.30 | 1892.07 | 1889.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 13:15:00 | 1900.00 | 1913.39 | 1900.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-11 09:15:00 | 1929.50 | 1912.64 | 1901.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-19 09:15:00 | 1884.90 | 1924.65 | 1909.94 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 1803.10 | 1898.23 | 1898.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 1793.00 | 1896.31 | 1897.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 11:15:00 | 1907.40 | 1889.43 | 1893.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-02 09:15:00 | 1861.60 | 1889.76 | 1893.86 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-03 12:15:00 | 1897.30 | 1887.70 | 1892.61 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 1931.30 | 1896.21 | 1896.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 1946.80 | 1897.07 | 1896.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1894.30 | 1898.72 | 1897.43 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 1830.10 | 1896.07 | 1896.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 1820.50 | 1895.32 | 1895.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 1901.50 | 1876.92 | 1885.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 14:15:00 | 1867.90 | 1876.96 | 1885.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1884.00 | 1876.84 | 1885.48 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-24 13:15:00 | 1899.00 | 1877.10 | 1885.52 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-12-15 11:15:00 | 1061.40 | 2023-12-15 13:15:00 | 1069.20 | EXIT_EMA400 | -7.80 |
| BUY | 2024-02-12 14:15:00 | 1127.95 | 2024-02-15 15:15:00 | 1078.00 | EXIT_EMA400 | -49.95 |
| BUY | 2024-05-02 10:15:00 | 1127.55 | 2024-05-09 09:15:00 | 1078.95 | EXIT_EMA400 | -48.60 |
| BUY | 2024-05-07 13:15:00 | 1099.35 | 2024-05-09 09:15:00 | 1078.95 | EXIT_EMA400 | -20.40 |
| SELL | 2025-01-13 09:15:00 | 1477.25 | 2025-01-30 09:15:00 | 1528.05 | EXIT_EMA400 | -50.80 |
| BUY | 2025-04-07 12:15:00 | 1582.05 | 2025-04-07 13:15:00 | 1599.85 | EXIT_EMA400 | 17.80 |
| SELL | 2025-08-19 12:15:00 | 1872.10 | 2025-09-11 11:15:00 | 1883.60 | EXIT_EMA400 | -11.50 |
| BUY | 2025-11-11 09:15:00 | 1929.50 | 2025-11-19 09:15:00 | 1884.90 | EXIT_EMA400 | -44.60 |
| SELL | 2025-12-02 09:15:00 | 1861.60 | 2025-12-03 12:15:00 | 1897.30 | EXIT_EMA400 | -35.70 |
| SELL | 2025-12-23 14:15:00 | 1867.90 | 2025-12-24 13:15:00 | 1899.00 | EXIT_EMA400 | -31.10 |
