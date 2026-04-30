# BEML Ltd. (BEML.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1801.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 296.47
- **Avg P&L per closed trade:** 42.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 11:15:00 | 1940.43 | 2143.56 | 2144.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 13:15:00 | 1925.55 | 2139.41 | 2142.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 2032.58 | 2024.27 | 2072.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-18 13:15:00 | 1889.40 | 1992.95 | 2039.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1943.20 | 1889.28 | 1949.60 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-16 10:15:00 | 1949.83 | 1889.89 | 1949.60 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 2109.23 | 1971.19 | 1970.95 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 13:15:00 | 1885.03 | 1970.75 | 1970.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 09:15:00 | 1869.50 | 1968.22 | 1969.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1995.98 | 1947.18 | 1958.16 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 11:15:00 | 2145.00 | 1967.76 | 1967.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 2174.93 | 2022.14 | 1997.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 2086.35 | 2109.62 | 2057.19 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 1869.50 | 2031.14 | 2031.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 12:15:00 | 1855.00 | 2029.38 | 2030.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 1885.08 | 1874.10 | 1934.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 14:15:00 | 1836.38 | 1877.15 | 1932.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-27 14:15:00 | 1598.38 | 1389.91 | 1517.23 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 1732.50 | 1547.48 | 1547.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 1790.00 | 1566.41 | 1556.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 2206.85 | 2211.87 | 2077.21 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 1922.15 | 2040.72 | 2040.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 1913.30 | 2037.04 | 2038.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 2050.90 | 2023.70 | 2031.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 14:15:00 | 1976.00 | 2024.05 | 2031.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-05 09:15:00 | 2042.00 | 2023.70 | 2031.13 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 2187.50 | 2037.21 | 2036.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 2245.00 | 2084.24 | 2062.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 2096.35 | 2106.21 | 2077.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-01 09:15:00 | 2157.15 | 2101.30 | 2077.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 2157.15 | 2101.30 | 2077.37 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-03 09:15:00 | 2180.90 | 2105.29 | 2080.22 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 2138.90 | 2169.73 | 2136.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-06 09:15:00 | 2016.50 | 2167.74 | 2135.98 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 2031.10 | 2110.15 | 2110.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 2011.00 | 2109.17 | 2109.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 1857.30 | 1809.05 | 1905.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-12 09:15:00 | 1765.30 | 1835.25 | 1887.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1820.00 | 1779.51 | 1836.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-01 12:15:00 | 1713.40 | 1779.68 | 1833.71 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 1638.90 | 1558.61 | 1631.70 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 1809.50 | 1675.32 | 1675.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 1842.50 | 1676.98 | 1675.91 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-18 13:15:00 | 1889.40 | 2024-10-16 10:15:00 | 1949.83 | EXIT_EMA400 | -60.43 |
| SELL | 2025-02-01 14:15:00 | 1836.38 | 2025-02-10 10:15:00 | 1548.29 | TARGET | 288.09 |
| SELL | 2025-09-04 14:15:00 | 1976.00 | 2025-09-05 09:15:00 | 2042.00 | EXIT_EMA400 | -66.00 |
| BUY | 2025-10-01 09:15:00 | 2157.15 | 2025-11-06 09:15:00 | 2016.50 | EXIT_EMA400 | -140.65 |
| BUY | 2025-10-03 09:15:00 | 2180.90 | 2025-11-06 09:15:00 | 2016.50 | EXIT_EMA400 | -164.40 |
| SELL | 2026-01-12 09:15:00 | 1765.30 | 2026-03-23 15:15:00 | 1399.95 | TARGET | 365.35 |
| SELL | 2026-02-01 12:15:00 | 1713.40 | 2026-04-10 09:15:00 | 1638.90 | EXIT_EMA400 | 74.50 |
