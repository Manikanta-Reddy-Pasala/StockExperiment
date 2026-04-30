# United Breweries Ltd. (UBL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 1458.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -122.95
- **Avg P&L per closed trade:** -24.59

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 12:15:00 | 1690.00 | 1736.19 | 1736.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 10:15:00 | 1680.15 | 1732.59 | 1734.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 09:15:00 | 1732.90 | 1730.84 | 1733.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-13 14:15:00 | 1684.00 | 1729.24 | 1732.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 1720.75 | 1728.74 | 1732.34 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-15 14:15:00 | 1733.60 | 1727.02 | 1731.25 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 12:15:00 | 1819.80 | 1732.46 | 1732.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 11:15:00 | 1842.10 | 1744.98 | 1738.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 1903.75 | 1907.94 | 1847.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-13 12:15:00 | 1929.90 | 1908.32 | 1848.76 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-22 12:15:00 | 1854.10 | 1904.94 | 1858.35 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 1953.70 | 2043.68 | 2044.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 1940.60 | 2039.93 | 2042.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 1942.90 | 1932.55 | 1970.97 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 2039.70 | 1982.05 | 1981.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 2045.30 | 1983.78 | 1982.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 13:15:00 | 1960.95 | 2024.48 | 2006.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-21 09:15:00 | 2091.10 | 2004.72 | 1999.42 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-28 09:15:00 | 2006.50 | 2020.54 | 2008.98 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 12:15:00 | 1913.65 | 2024.37 | 2024.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-05 09:15:00 | 1889.95 | 2019.79 | 2022.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 09:15:00 | 1994.20 | 1958.25 | 1985.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 13:15:00 | 1920.20 | 1956.77 | 1981.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1976.15 | 1954.55 | 1978.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-28 09:15:00 | 2002.65 | 1955.64 | 1978.83 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 2130.00 | 1991.59 | 1990.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 2145.90 | 1993.12 | 1991.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 2096.20 | 2103.78 | 2063.41 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 1990.20 | 2047.39 | 2047.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 1982.10 | 2046.74 | 2047.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1985.20 | 1984.76 | 2005.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-30 09:15:00 | 1951.20 | 1999.16 | 2007.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1825.00 | 1796.17 | 1836.50 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-28 09:15:00 | 1837.90 | 1801.58 | 1835.33 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 1684.40 | 1606.80 | 1606.52 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 1551.30 | 1607.10 | 1607.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1541.50 | 1601.89 | 1604.50 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-13 14:15:00 | 1684.00 | 2024-03-15 14:15:00 | 1733.60 | EXIT_EMA400 | -49.60 |
| BUY | 2024-05-13 12:15:00 | 1929.90 | 2024-05-22 12:15:00 | 1854.10 | EXIT_EMA400 | -75.80 |
| BUY | 2025-01-21 09:15:00 | 2091.10 | 2025-01-28 09:15:00 | 2006.50 | EXIT_EMA400 | -84.60 |
| SELL | 2025-03-25 13:15:00 | 1920.20 | 2025-03-28 09:15:00 | 2002.65 | EXIT_EMA400 | -82.45 |
| SELL | 2025-07-30 09:15:00 | 1951.20 | 2025-09-11 12:15:00 | 1781.70 | TARGET | 169.50 |
