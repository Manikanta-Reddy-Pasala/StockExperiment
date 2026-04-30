# United Breweries Ltd. (UBL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1455.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 94.30
- **Avg P&L per closed trade:** 13.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 1897.30 | 1995.39 | 1995.51 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 14:15:00 | 2032.80 | 1994.61 | 1994.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 09:15:00 | 2039.00 | 1997.78 | 1996.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 14:15:00 | 2008.70 | 2010.39 | 2003.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-09 09:15:00 | 2031.75 | 2010.65 | 2003.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 2031.75 | 2010.65 | 2003.75 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-09 10:15:00 | 2042.10 | 2010.96 | 2003.94 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 2051.90 | 2040.33 | 2022.07 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-19 09:15:00 | 2113.15 | 2041.22 | 2022.79 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 2067.10 | 2096.09 | 2061.19 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-07 15:15:00 | 2060.00 | 2094.73 | 2061.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 15:15:00 | 1983.95 | 2046.36 | 2046.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 1958.75 | 2044.76 | 2045.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 1942.90 | 1932.06 | 1970.96 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 2038.00 | 1982.20 | 1982.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 2045.30 | 1983.38 | 1982.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 13:15:00 | 1960.95 | 2024.23 | 2005.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-21 09:15:00 | 2091.10 | 2004.62 | 1999.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 2047.05 | 2008.89 | 2001.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-23 09:15:00 | 2078.80 | 2012.60 | 2003.82 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-01-28 09:15:00 | 2006.50 | 2020.31 | 2008.81 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 11:15:00 | 1915.85 | 2027.37 | 2027.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-05 09:15:00 | 1891.30 | 2021.57 | 2024.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 09:15:00 | 1992.75 | 1959.00 | 1986.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 13:15:00 | 1920.20 | 1957.35 | 1982.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1976.20 | 1955.07 | 1980.06 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-27 15:15:00 | 1984.10 | 1955.75 | 1980.03 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 2130.00 | 1991.94 | 1991.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 2145.90 | 1993.47 | 1992.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 2096.20 | 2103.80 | 2063.91 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 2000.00 | 2047.83 | 2047.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 1990.20 | 2047.26 | 2047.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1985.20 | 1984.67 | 2005.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-30 09:15:00 | 1951.20 | 1999.24 | 2007.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1825.00 | 1796.21 | 1836.55 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-28 09:15:00 | 1837.90 | 1801.61 | 1835.37 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 1694.20 | 1603.59 | 1603.29 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 1551.80 | 1604.61 | 1604.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1541.50 | 1601.63 | 1603.22 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-09 09:15:00 | 2031.75 | 2024-09-16 12:15:00 | 2115.76 | TARGET | 84.01 |
| BUY | 2024-09-09 10:15:00 | 2042.10 | 2024-09-23 13:15:00 | 2156.59 | TARGET | 114.49 |
| BUY | 2024-09-19 09:15:00 | 2113.15 | 2024-10-07 15:15:00 | 2060.00 | EXIT_EMA400 | -53.15 |
| BUY | 2025-01-21 09:15:00 | 2091.10 | 2025-01-28 09:15:00 | 2006.50 | EXIT_EMA400 | -84.60 |
| BUY | 2025-01-23 09:15:00 | 2078.80 | 2025-01-28 09:15:00 | 2006.50 | EXIT_EMA400 | -72.30 |
| SELL | 2025-03-25 13:15:00 | 1920.20 | 2025-03-27 15:15:00 | 1984.10 | EXIT_EMA400 | -63.90 |
| SELL | 2025-07-30 09:15:00 | 1951.20 | 2025-09-11 12:15:00 | 1781.45 | TARGET | 169.75 |
