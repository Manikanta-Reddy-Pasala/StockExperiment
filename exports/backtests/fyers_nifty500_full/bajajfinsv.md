# Bajaj Finserv Ltd. (BAJAJFINSV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1740.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 9 |
| ENTRY1 | 7 |
| ENTRY2 | 4 |
| EXIT | 7 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / EMA400 exits:** 4 / 7
- **Total realized P&L (per unit):** 150.57
- **Avg P&L per closed trade:** 13.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 11:15:00 | 1538.95 | 1583.25 | 1583.44 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 1625.00 | 1583.58 | 1583.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 09:15:00 | 1631.00 | 1586.48 | 1584.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 12:15:00 | 1850.75 | 1860.69 | 1780.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 10:15:00 | 1877.30 | 1860.29 | 1782.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1811.00 | 1859.49 | 1799.01 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-18 11:15:00 | 1825.65 | 1858.78 | 1799.25 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-21 09:15:00 | 1797.80 | 1856.74 | 1799.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 1675.50 | 1769.49 | 1769.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 13:15:00 | 1667.45 | 1767.55 | 1768.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 1661.90 | 1661.20 | 1699.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-18 09:15:00 | 1626.05 | 1662.72 | 1693.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 1657.65 | 1620.86 | 1659.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-02 12:15:00 | 1711.90 | 1621.76 | 1659.75 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 1738.50 | 1679.28 | 1679.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 11:15:00 | 1749.05 | 1685.54 | 1682.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1693.40 | 1704.55 | 1693.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-01 13:15:00 | 1751.55 | 1705.01 | 1693.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1751.55 | 1705.01 | 1693.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-02-01 15:15:00 | 1769.00 | 1706.09 | 1694.07 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1777.60 | 1814.31 | 1771.21 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-03-05 10:15:00 | 1790.45 | 1814.07 | 1771.31 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1853.40 | 1880.96 | 1833.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-07 12:15:00 | 1832.05 | 1879.86 | 1834.00 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 1942.00 | 1992.10 | 1992.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1928.90 | 1991.47 | 1992.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 1939.90 | 1966.14 | 1975.31 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-03 15:15:00 | 1971.00 | 1957.43 | 1968.67 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 2036.30 | 1978.32 | 1978.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 2038.20 | 1978.92 | 1978.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 2015.00 | 2025.48 | 2006.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-07 09:15:00 | 2048.70 | 2019.64 | 2006.56 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 2016.50 | 2020.72 | 2007.76 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-09 11:15:00 | 2022.00 | 2020.35 | 2007.95 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 2010.00 | 2020.03 | 2008.27 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-10 13:15:00 | 2007.30 | 2019.90 | 2008.27 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 2009.50 | 2053.54 | 2053.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1988.60 | 2051.66 | 2052.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-07 09:15:00 | 2030.50 | 2046.22 | 2049.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2010.50 | 1986.60 | 2011.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 10:15:00 | 2015.00 | 1986.88 | 2011.47 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 2049.70 | 2022.50 | 2022.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 2052.80 | 2023.00 | 2022.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1966.20 | 2022.51 | 2022.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 1946.20 | 2021.16 | 2021.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.43 | 1865.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 12:15:00 | 1775.00 | 1783.29 | 1862.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1852.40 | 1797.68 | 1852.87 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-22 12:15:00 | 1857.20 | 1801.13 | 1852.73 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-09 10:15:00 | 1877.30 | 2024-10-21 09:15:00 | 1797.80 | EXIT_EMA400 | -79.50 |
| BUY | 2024-10-18 11:15:00 | 1825.65 | 2024-10-21 09:15:00 | 1797.80 | EXIT_EMA400 | -27.85 |
| SELL | 2024-12-18 09:15:00 | 1626.05 | 2025-01-02 12:15:00 | 1711.90 | EXIT_EMA400 | -85.85 |
| BUY | 2025-02-01 13:15:00 | 1751.55 | 2025-02-27 10:15:00 | 1925.96 | TARGET | 174.41 |
| BUY | 2025-03-05 10:15:00 | 1790.45 | 2025-03-06 13:15:00 | 1847.87 | TARGET | 57.42 |
| BUY | 2025-02-01 15:15:00 | 1769.00 | 2025-03-27 12:15:00 | 1993.79 | TARGET | 224.79 |
| SELL | 2025-08-26 09:15:00 | 1939.90 | 2025-09-03 15:15:00 | 1971.00 | EXIT_EMA400 | -31.10 |
| BUY | 2025-10-07 09:15:00 | 2048.70 | 2025-10-10 13:15:00 | 2007.30 | EXIT_EMA400 | -41.40 |
| BUY | 2025-10-09 11:15:00 | 2022.00 | 2025-10-10 13:15:00 | 2007.30 | EXIT_EMA400 | -14.70 |
| SELL | 2026-01-07 09:15:00 | 2030.50 | 2026-01-12 09:15:00 | 1973.95 | TARGET | 56.55 |
| SELL | 2026-04-09 12:15:00 | 1775.00 | 2026-04-22 12:15:00 | 1857.20 | EXIT_EMA400 | -82.20 |
