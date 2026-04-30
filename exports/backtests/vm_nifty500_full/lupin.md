# Lupin Ltd. (LUPIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 2305.20
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
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** 207.69
- **Avg P&L per closed trade:** 23.08

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 12:15:00 | 2004.15 | 2109.84 | 2110.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 1999.00 | 2104.84 | 2107.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 13:15:00 | 2096.35 | 2093.05 | 2100.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-05 10:15:00 | 2072.90 | 2092.95 | 2100.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-05 12:15:00 | 2109.00 | 2093.13 | 2100.42 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 2177.30 | 2105.62 | 2105.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 10:15:00 | 2192.85 | 2114.22 | 2109.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 2207.25 | 2220.12 | 2173.79 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 09:15:00 | 2077.00 | 2149.15 | 2149.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 15:15:00 | 2053.20 | 2144.27 | 2146.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 2177.90 | 2129.77 | 2138.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 2097.35 | 2142.63 | 2144.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 2097.35 | 2142.63 | 2144.72 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-11 10:15:00 | 2078.95 | 2142.00 | 2144.40 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2112.80 | 2131.37 | 2138.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-13 15:15:00 | 2052.00 | 2127.81 | 2136.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 2038.60 | 2019.08 | 2063.92 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-10 10:15:00 | 2022.70 | 2019.12 | 2063.71 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-19 11:15:00 | 2050.55 | 2010.96 | 2050.40 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 15:15:00 | 2072.90 | 2044.14 | 2044.04 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1985.60 | 2043.72 | 2043.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1977.80 | 2043.06 | 2043.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 13:15:00 | 2006.40 | 1998.10 | 2015.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-09 11:15:00 | 1990.20 | 1998.00 | 2015.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 2004.80 | 1998.05 | 2014.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-11 09:15:00 | 2014.80 | 1998.44 | 2014.69 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 2040.60 | 1953.44 | 1953.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 14:15:00 | 2047.30 | 1954.37 | 1953.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 13:15:00 | 1969.70 | 1983.52 | 1970.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-03 09:15:00 | 1998.50 | 1973.76 | 1967.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-06 09:15:00 | 1933.80 | 1973.60 | 1967.33 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 1952.30 | 1962.16 | 1962.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 1941.80 | 1961.90 | 1962.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 12:15:00 | 1954.20 | 1950.69 | 1955.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-30 09:15:00 | 1934.00 | 1950.62 | 1955.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1934.00 | 1950.62 | 1955.63 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-31 10:15:00 | 1970.00 | 1950.20 | 1955.21 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 2019.90 | 1959.62 | 1959.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 2034.30 | 1964.66 | 1962.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 11:15:00 | 1992.00 | 1995.39 | 1980.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-24 15:15:00 | 1999.60 | 1995.39 | 1980.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-29 09:15:00 | 2097.90 | 2135.57 | 2099.10 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-05 10:15:00 | 2072.90 | 2024-12-05 12:15:00 | 2109.00 | EXIT_EMA400 | -36.10 |
| SELL | 2025-02-11 09:15:00 | 2097.35 | 2025-02-14 10:15:00 | 1955.23 | TARGET | 142.12 |
| SELL | 2025-02-11 10:15:00 | 2078.95 | 2025-02-24 09:15:00 | 1882.61 | TARGET | 196.34 |
| SELL | 2025-02-13 15:15:00 | 2052.00 | 2025-03-19 11:15:00 | 2050.55 | EXIT_EMA400 | 1.45 |
| SELL | 2025-03-10 10:15:00 | 2022.70 | 2025-03-19 11:15:00 | 2050.55 | EXIT_EMA400 | -27.85 |
| SELL | 2025-06-09 11:15:00 | 1990.20 | 2025-06-11 09:15:00 | 2014.80 | EXIT_EMA400 | -24.60 |
| BUY | 2025-10-03 09:15:00 | 1998.50 | 2025-10-06 09:15:00 | 1933.80 | EXIT_EMA400 | -64.70 |
| SELL | 2025-10-30 09:15:00 | 1934.00 | 2025-10-31 10:15:00 | 1970.00 | EXIT_EMA400 | -36.00 |
| BUY | 2025-11-24 15:15:00 | 1999.60 | 2025-11-26 10:15:00 | 2056.64 | TARGET | 57.04 |
