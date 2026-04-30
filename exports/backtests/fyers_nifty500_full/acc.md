# ACC Ltd. (ACC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1417.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 232.83
- **Avg P&L per closed trade:** 46.57

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 11:15:00 | 2363.10 | 2548.74 | 2549.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 13:15:00 | 2352.50 | 2544.88 | 2547.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 2408.25 | 2395.21 | 2448.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-07 11:15:00 | 2354.55 | 2452.10 | 2458.69 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 11:15:00 | 2303.95 | 2237.53 | 2296.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 14:15:00 | 2081.10 | 1972.43 | 1972.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 10:15:00 | 2091.70 | 1981.22 | 1976.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 1959.90 | 1985.95 | 1979.41 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 1877.20 | 1972.95 | 1973.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 1872.50 | 1966.14 | 1969.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 11:15:00 | 1927.40 | 1911.94 | 1936.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-30 09:15:00 | 1902.80 | 1924.30 | 1936.89 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-10 11:15:00 | 1926.50 | 1910.82 | 1926.30 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 1999.70 | 1917.35 | 1916.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 2002.70 | 1918.19 | 1917.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1922.00 | 1943.98 | 1932.51 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 1829.00 | 1923.25 | 1923.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 1814.20 | 1922.16 | 1922.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 1854.90 | 1851.47 | 1879.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 10:15:00 | 1833.90 | 1852.75 | 1877.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1851.00 | 1837.94 | 1862.65 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-04 12:15:00 | 1850.20 | 1838.37 | 1862.50 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1851.00 | 1839.93 | 1859.63 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-11 12:15:00 | 1840.00 | 1840.03 | 1859.48 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 1860.00 | 1841.02 | 1858.94 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-07 11:15:00 | 2354.55 | 2024-11-21 09:15:00 | 2042.12 | TARGET | 312.43 |
| SELL | 2025-05-30 09:15:00 | 1902.80 | 2025-06-10 11:15:00 | 1926.50 | EXIT_EMA400 | -23.70 |
| SELL | 2025-08-22 10:15:00 | 1833.90 | 2025-09-15 09:15:00 | 1860.00 | EXIT_EMA400 | -26.10 |
| SELL | 2025-09-04 12:15:00 | 1850.20 | 2025-09-15 09:15:00 | 1860.00 | EXIT_EMA400 | -9.80 |
| SELL | 2025-09-11 12:15:00 | 1840.00 | 2025-09-15 09:15:00 | 1860.00 | EXIT_EMA400 | -20.00 |
