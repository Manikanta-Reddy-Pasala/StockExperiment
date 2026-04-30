# Glenmark Pharmaceuticals Ltd. (GLENMARK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 2406.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -255.65
- **Avg P&L per closed trade:** -42.61

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 10:15:00 | 741.30 | 770.77 | 770.91 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 13:15:00 | 780.00 | 770.99 | 770.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 14:15:00 | 784.55 | 771.12 | 771.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 09:15:00 | 767.00 | 772.57 | 771.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-24 09:15:00 | 790.20 | 772.82 | 771.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 773.30 | 773.99 | 772.61 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-11-28 14:15:00 | 770.55 | 773.96 | 772.60 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 15:15:00 | 1478.20 | 1625.45 | 1625.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1464.95 | 1565.31 | 1588.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 1561.40 | 1561.18 | 1584.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 09:15:00 | 1541.50 | 1560.73 | 1584.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-27 09:15:00 | 1574.40 | 1552.09 | 1574.12 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 1512.80 | 1432.03 | 1431.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1558.70 | 1433.29 | 1432.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 09:15:00 | 2022.20 | 2024.77 | 1900.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-05 09:15:00 | 2046.90 | 1982.28 | 1919.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1986.50 | 2038.04 | 1981.81 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-26 10:15:00 | 1978.00 | 2037.45 | 1981.79 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 15:15:00 | 1853.60 | 1956.05 | 1956.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 1847.00 | 1948.52 | 1952.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 1909.00 | 1884.48 | 1911.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-17 13:15:00 | 1871.00 | 1886.50 | 1910.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1871.20 | 1879.00 | 1903.67 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-21 10:15:00 | 1868.00 | 1878.89 | 1903.49 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-11-26 10:15:00 | 1904.50 | 1876.68 | 1899.83 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 1949.40 | 1915.82 | 1915.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1961.00 | 1917.36 | 1916.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 14:15:00 | 2005.00 | 2007.00 | 1973.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-14 10:15:00 | 2031.60 | 2005.57 | 1975.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-20 09:15:00 | 1967.40 | 2004.96 | 1977.83 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-24 09:15:00 | 790.20 | 2023-11-28 14:15:00 | 770.55 | EXIT_EMA400 | -19.65 |
| SELL | 2024-12-17 09:15:00 | 1541.50 | 2024-12-27 09:15:00 | 1574.40 | EXIT_EMA400 | -32.90 |
| BUY | 2025-09-05 09:15:00 | 2046.90 | 2025-09-26 10:15:00 | 1978.00 | EXIT_EMA400 | -68.90 |
| SELL | 2025-11-17 13:15:00 | 1871.00 | 2025-11-26 10:15:00 | 1904.50 | EXIT_EMA400 | -33.50 |
| SELL | 2025-11-21 10:15:00 | 1868.00 | 2025-11-26 10:15:00 | 1904.50 | EXIT_EMA400 | -36.50 |
| BUY | 2026-01-14 10:15:00 | 2031.60 | 2026-01-20 09:15:00 | 1967.40 | EXIT_EMA400 | -64.20 |
