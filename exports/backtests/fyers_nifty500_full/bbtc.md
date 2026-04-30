# Bombay Burmah Trading Corporation Ltd. (BBTC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1498.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** 38.37
- **Avg P&L per closed trade:** 4.80

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 14:15:00 | 2405.75 | 2570.01 | 2570.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 15:15:00 | 2397.00 | 2568.29 | 2569.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 2228.15 | 2139.11 | 2264.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-28 09:15:00 | 2026.35 | 2143.37 | 2253.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1882.80 | 1796.36 | 1904.49 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-03 09:15:00 | 1859.70 | 1798.79 | 1904.11 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-15 09:15:00 | 1894.90 | 1790.72 | 1879.66 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 2004.80 | 1894.81 | 1894.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 2019.00 | 1898.97 | 1896.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1927.90 | 1962.72 | 1937.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 12:15:00 | 1982.00 | 1939.71 | 1931.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1954.70 | 1958.75 | 1943.73 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-08 14:15:00 | 1959.80 | 1958.70 | 1943.85 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1954.00 | 1958.61 | 1943.96 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-09 15:15:00 | 1940.10 | 1958.50 | 1944.34 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 1862.60 | 1945.44 | 1945.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 1857.00 | 1942.99 | 1944.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 1888.10 | 1856.77 | 1888.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-05 10:15:00 | 1849.60 | 1858.06 | 1887.85 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 1893.00 | 1857.97 | 1884.95 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 2045.50 | 1886.96 | 1886.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 2047.90 | 1888.56 | 1887.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 15:15:00 | 1945.40 | 1947.11 | 1922.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-18 10:15:00 | 1982.80 | 1920.86 | 1914.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 1920.00 | 1928.40 | 1918.89 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-20 14:15:00 | 1917.50 | 1928.29 | 1918.89 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 14:15:00 | 1839.80 | 1910.54 | 1910.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 1836.70 | 1904.08 | 1907.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 1914.30 | 1890.55 | 1899.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-03 14:15:00 | 1871.30 | 1890.49 | 1899.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1871.30 | 1890.49 | 1899.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-04 09:15:00 | 1866.70 | 1890.13 | 1899.46 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-09 14:15:00 | 1897.00 | 1879.58 | 1892.71 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-03 09:15:00 | 1859.70 | 2025-04-07 09:15:00 | 1726.48 | TARGET | 133.22 |
| SELL | 2025-01-28 09:15:00 | 2026.35 | 2025-04-15 09:15:00 | 1894.90 | EXIT_EMA400 | 131.45 |
| BUY | 2025-06-27 12:15:00 | 1982.00 | 2025-07-09 15:15:00 | 1940.10 | EXIT_EMA400 | -41.90 |
| BUY | 2025-07-08 14:15:00 | 1959.80 | 2025-07-09 15:15:00 | 1940.10 | EXIT_EMA400 | -19.70 |
| SELL | 2025-09-05 10:15:00 | 1849.60 | 2025-09-10 09:15:00 | 1893.00 | EXIT_EMA400 | -43.40 |
| BUY | 2025-11-18 10:15:00 | 1982.80 | 2025-11-20 14:15:00 | 1917.50 | EXIT_EMA400 | -65.30 |
| SELL | 2025-12-03 14:15:00 | 1871.30 | 2025-12-09 14:15:00 | 1897.00 | EXIT_EMA400 | -25.70 |
| SELL | 2025-12-04 09:15:00 | 1866.70 | 2025-12-09 14:15:00 | 1897.00 | EXIT_EMA400 | -30.30 |
