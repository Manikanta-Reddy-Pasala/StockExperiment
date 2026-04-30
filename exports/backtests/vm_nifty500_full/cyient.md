# Cyient Ltd. (CYIENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 871.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -347.95
- **Avg P&L per closed trade:** -57.99

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 14:15:00 | 1993.70 | 2058.21 | 2058.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 10:15:00 | 1951.85 | 2055.47 | 2057.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 13:15:00 | 2028.85 | 2026.14 | 2040.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-15 14:15:00 | 1991.40 | 2025.79 | 2040.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 1980.00 | 2017.98 | 2035.49 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-03-22 09:15:00 | 1965.05 | 2014.00 | 2032.26 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 1998.80 | 2002.20 | 2023.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-03 09:15:00 | 2035.35 | 2002.88 | 2022.39 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 14:15:00 | 2090.65 | 2038.53 | 2038.50 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-04-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 14:15:00 | 1969.75 | 2038.60 | 2038.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-22 13:15:00 | 1957.05 | 2035.30 | 2037.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 10:15:00 | 1800.65 | 1796.69 | 1863.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-11 11:15:00 | 1774.65 | 1832.03 | 1854.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 1848.90 | 1829.84 | 1852.89 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-07-15 09:15:00 | 1889.85 | 1830.69 | 1852.75 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 2000.90 | 1830.17 | 1830.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 11:15:00 | 2003.65 | 1831.89 | 1830.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 15:15:00 | 1990.00 | 1996.76 | 1940.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-25 10:15:00 | 2007.90 | 1996.88 | 1940.63 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-27 09:15:00 | 1933.00 | 1995.79 | 1943.63 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 1833.00 | 1913.58 | 1913.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 1824.70 | 1910.25 | 1912.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 1871.70 | 1853.80 | 1877.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 13:15:00 | 1829.70 | 1864.32 | 1879.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-26 09:15:00 | 1894.25 | 1840.90 | 1862.42 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 12:15:00 | 2023.45 | 1876.71 | 1876.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 2048.65 | 1882.22 | 1878.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 1946.10 | 1960.24 | 1925.09 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1753.15 | 1902.86 | 1903.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1743.95 | 1901.28 | 1902.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 11:15:00 | 1342.75 | 1333.23 | 1460.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-21 13:15:00 | 1310.95 | 1332.90 | 1459.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1288.00 | 1218.20 | 1288.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-14 11:15:00 | 1290.00 | 1219.56 | 1288.13 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-15 14:15:00 | 1991.40 | 2024-04-03 09:15:00 | 2035.35 | EXIT_EMA400 | -43.95 |
| SELL | 2024-03-22 09:15:00 | 1965.05 | 2024-04-03 09:15:00 | 2035.35 | EXIT_EMA400 | -70.30 |
| SELL | 2024-07-11 11:15:00 | 1774.65 | 2024-07-15 09:15:00 | 1889.85 | EXIT_EMA400 | -115.20 |
| BUY | 2024-09-25 10:15:00 | 2007.90 | 2024-09-27 09:15:00 | 1933.00 | EXIT_EMA400 | -74.90 |
| SELL | 2024-11-12 13:15:00 | 1829.70 | 2024-11-26 09:15:00 | 1894.25 | EXIT_EMA400 | -64.55 |
| SELL | 2025-03-21 13:15:00 | 1310.95 | 2025-05-14 11:15:00 | 1290.00 | EXIT_EMA400 | 20.95 |
