# Glenmark Pharmaceuticals Ltd. (GLENMARK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2400.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -249.60
- **Avg P&L per closed trade:** -49.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 14:15:00 | 1476.55 | 1626.97 | 1627.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1465.95 | 1565.37 | 1588.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 1561.40 | 1561.25 | 1585.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 09:15:00 | 1541.50 | 1560.79 | 1584.52 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1574.40 | 1552.18 | 1574.46 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-27 10:15:00 | 1588.00 | 1552.53 | 1574.53 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 1518.20 | 1432.02 | 1431.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1558.10 | 1433.28 | 1432.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 09:15:00 | 2022.20 | 2024.63 | 1900.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-05 09:15:00 | 2046.90 | 1982.23 | 1919.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1987.00 | 2038.00 | 1981.73 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-26 10:15:00 | 1978.00 | 2037.40 | 1981.71 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 15:15:00 | 1855.00 | 1955.98 | 1956.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 1847.00 | 1948.52 | 1952.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 1909.20 | 1884.52 | 1911.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-17 13:15:00 | 1871.00 | 1886.59 | 1910.40 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1871.20 | 1879.07 | 1903.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-21 10:15:00 | 1868.00 | 1878.96 | 1903.52 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-11-26 10:15:00 | 1904.50 | 1876.63 | 1899.80 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 1949.40 | 1915.78 | 1915.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1961.00 | 1917.33 | 1916.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 14:15:00 | 2005.00 | 2006.95 | 1973.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-14 10:15:00 | 2031.60 | 2005.50 | 1975.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-20 09:15:00 | 1967.40 | 2004.93 | 1977.80 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-17 09:15:00 | 1541.50 | 2024-12-27 10:15:00 | 1588.00 | EXIT_EMA400 | -46.50 |
| BUY | 2025-09-05 09:15:00 | 2046.90 | 2025-09-26 10:15:00 | 1978.00 | EXIT_EMA400 | -68.90 |
| SELL | 2025-11-17 13:15:00 | 1871.00 | 2025-11-26 10:15:00 | 1904.50 | EXIT_EMA400 | -33.50 |
| SELL | 2025-11-21 10:15:00 | 1868.00 | 2025-11-26 10:15:00 | 1904.50 | EXIT_EMA400 | -36.50 |
| BUY | 2026-01-14 10:15:00 | 2031.60 | 2026-01-20 09:15:00 | 1967.40 | EXIT_EMA400 | -64.20 |
