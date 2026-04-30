# Lupin Ltd. (LUPIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2312.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** 240.47
- **Avg P&L per closed trade:** 26.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 2007.20 | 2110.77 | 2111.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 1999.00 | 2104.69 | 2108.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 13:15:00 | 2096.65 | 2092.92 | 2101.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-05 10:15:00 | 2072.90 | 2092.90 | 2100.85 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-05 12:15:00 | 2108.85 | 2093.06 | 2100.85 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 11:15:00 | 2177.70 | 2106.25 | 2106.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 10:15:00 | 2192.85 | 2114.17 | 2110.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 2207.95 | 2220.22 | 2174.04 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 15:15:00 | 2070.00 | 2149.77 | 2149.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 15:15:00 | 2053.20 | 2144.18 | 2147.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 2177.90 | 2125.41 | 2136.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 2097.35 | 2139.26 | 2142.57 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 2097.35 | 2139.26 | 2142.57 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-11 10:15:00 | 2078.95 | 2138.66 | 2142.25 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2113.00 | 2128.44 | 2136.75 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-14 09:15:00 | 1985.40 | 2123.65 | 2134.04 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 2038.85 | 2018.25 | 2062.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-10 10:15:00 | 2022.70 | 2018.29 | 2062.64 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-19 11:15:00 | 2050.55 | 2010.31 | 2049.47 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 2073.10 | 2043.73 | 2043.58 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1985.60 | 2043.56 | 2043.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1977.90 | 2042.91 | 2043.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 1998.10 | 1998.10 | 2016.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-06 10:15:00 | 1987.60 | 1997.99 | 2016.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 2004.80 | 1997.98 | 2014.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-11 09:15:00 | 2014.80 | 1998.38 | 2014.51 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 2040.50 | 1953.56 | 1953.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 14:15:00 | 2047.30 | 1954.50 | 1953.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 13:15:00 | 1969.70 | 1983.67 | 1970.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-03 09:15:00 | 1998.50 | 1973.86 | 1967.25 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-06 09:15:00 | 1933.80 | 1973.69 | 1967.40 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 11:15:00 | 1949.90 | 1962.12 | 1962.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 1941.10 | 1961.75 | 1961.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 12:15:00 | 1954.20 | 1950.56 | 1955.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-30 09:15:00 | 1934.70 | 1950.51 | 1955.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1934.70 | 1950.51 | 1955.56 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-31 10:15:00 | 1970.00 | 1950.08 | 1955.14 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 2020.00 | 1959.81 | 1959.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 2034.30 | 1964.82 | 1962.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 11:15:00 | 1991.60 | 1995.52 | 1980.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-25 09:15:00 | 2037.90 | 1995.79 | 1980.88 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-29 09:15:00 | 2098.90 | 2135.69 | 2099.14 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-05 10:15:00 | 2072.90 | 2024-12-05 12:15:00 | 2108.85 | EXIT_EMA400 | -35.95 |
| SELL | 2025-02-11 09:15:00 | 2097.35 | 2025-02-14 10:15:00 | 1961.69 | TARGET | 135.66 |
| SELL | 2025-02-11 10:15:00 | 2078.95 | 2025-02-24 09:15:00 | 1889.04 | TARGET | 189.91 |
| SELL | 2025-02-14 09:15:00 | 1985.40 | 2025-03-19 11:15:00 | 2050.55 | EXIT_EMA400 | -65.15 |
| SELL | 2025-03-10 10:15:00 | 2022.70 | 2025-03-19 11:15:00 | 2050.55 | EXIT_EMA400 | -27.85 |
| SELL | 2025-06-06 10:15:00 | 1987.60 | 2025-06-11 09:15:00 | 2014.80 | EXIT_EMA400 | -27.20 |
| BUY | 2025-10-03 09:15:00 | 1998.50 | 2025-10-06 09:15:00 | 1933.80 | EXIT_EMA400 | -64.70 |
| SELL | 2025-10-30 09:15:00 | 1934.70 | 2025-10-31 10:15:00 | 1970.00 | EXIT_EMA400 | -35.30 |
| BUY | 2025-11-25 09:15:00 | 2037.90 | 2026-01-07 09:15:00 | 2208.95 | TARGET | 171.05 |
