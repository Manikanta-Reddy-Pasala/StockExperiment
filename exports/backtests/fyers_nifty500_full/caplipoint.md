# Caplin Point Laboratories Ltd. (CAPLIPOINT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1705.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 438.65
- **Avg P&L per closed trade:** 62.66

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 2024.05 | 2198.38 | 2198.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 1956.75 | 2153.13 | 2173.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 1956.55 | 1956.52 | 2030.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 1934.20 | 1979.74 | 2022.81 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1934.20 | 1979.74 | 2022.81 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-07 09:15:00 | 1811.85 | 1976.26 | 2019.56 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-05-12 13:15:00 | 1946.50 | 1902.50 | 1945.82 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 2271.90 | 1978.94 | 1977.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 2296.40 | 1993.77 | 1985.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 2055.70 | 2081.69 | 2041.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-11 15:15:00 | 2122.20 | 2078.21 | 2043.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2099.70 | 2085.91 | 2052.34 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 2051.60 | 2085.62 | 2053.51 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 1930.00 | 2053.01 | 2053.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1900.90 | 2047.97 | 2050.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 2069.00 | 2044.02 | 2048.58 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 2139.50 | 2053.21 | 2052.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 2143.10 | 2058.18 | 2055.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 2086.70 | 2098.61 | 2079.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-28 09:15:00 | 2119.80 | 2098.63 | 2079.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 2119.80 | 2098.63 | 2079.38 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-28 10:15:00 | 2143.40 | 2099.08 | 2079.70 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2115.50 | 2101.90 | 2082.40 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-02 15:15:00 | 2125.80 | 2103.65 | 2084.52 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-26 09:15:00 | 2080.90 | 2206.35 | 2156.98 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 2038.50 | 2120.31 | 2120.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 2009.50 | 2111.51 | 2116.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 1952.60 | 1949.76 | 1996.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-08 12:15:00 | 1919.80 | 1948.82 | 1992.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-23 09:15:00 | 1977.90 | 1941.96 | 1975.08 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-04 09:15:00 | 1934.20 | 2025-04-07 09:15:00 | 1668.38 | TARGET | 265.82 |
| SELL | 2025-04-07 09:15:00 | 1811.85 | 2025-05-12 13:15:00 | 1946.50 | EXIT_EMA400 | -134.65 |
| BUY | 2025-06-11 15:15:00 | 2122.20 | 2025-06-19 10:15:00 | 2051.60 | EXIT_EMA400 | -70.60 |
| BUY | 2025-08-28 09:15:00 | 2119.80 | 2025-09-11 09:15:00 | 2241.05 | TARGET | 121.25 |
| BUY | 2025-09-02 15:15:00 | 2125.80 | 2025-09-11 09:15:00 | 2249.63 | TARGET | 123.83 |
| BUY | 2025-08-28 10:15:00 | 2143.40 | 2025-09-17 10:15:00 | 2334.49 | TARGET | 191.09 |
| SELL | 2025-12-08 12:15:00 | 1919.80 | 2025-12-23 09:15:00 | 1977.90 | EXIT_EMA400 | -58.10 |
