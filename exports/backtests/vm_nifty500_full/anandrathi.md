# Anand Rathi Wealth Ltd. (ANANDRATHI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 3601.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / EMA400 exits:** 4 / 4
- **Total realized P&L (per unit):** 592.02
- **Avg P&L per closed trade:** 74.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 14:15:00 | 1850.00 | 1943.35 | 1943.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1825.20 | 1923.94 | 1933.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 1891.47 | 1860.02 | 1892.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-23 14:15:00 | 1800.00 | 1860.11 | 1889.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1866.00 | 1856.22 | 1885.08 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-08-28 10:15:00 | 1862.50 | 1856.29 | 1884.97 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1873.78 | 1856.90 | 1884.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-30 11:15:00 | 1889.00 | 1858.05 | 1883.80 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 1959.62 | 1901.69 | 1901.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 1971.45 | 1903.53 | 1902.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 15:15:00 | 1910.00 | 1915.26 | 1908.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-25 11:15:00 | 1971.93 | 1919.77 | 1911.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1948.28 | 1929.43 | 1918.39 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-04 11:15:00 | 1978.00 | 1931.09 | 1919.71 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 1927.38 | 1932.33 | 1920.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-07 11:15:00 | 1951.55 | 1932.52 | 1920.84 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1969.38 | 2001.38 | 1968.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-28 11:15:00 | 1965.68 | 2000.72 | 1968.35 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 1908.30 | 2020.80 | 2021.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1883.00 | 1998.49 | 2008.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 1908.68 | 1906.71 | 1952.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-07 10:15:00 | 1864.40 | 1904.48 | 1947.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-20 09:15:00 | 1929.97 | 1871.10 | 1916.86 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 1892.90 | 1814.73 | 1814.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 1900.20 | 1816.23 | 1815.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 13:15:00 | 2876.80 | 2893.90 | 2710.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-06 11:15:00 | 2919.90 | 2877.41 | 2739.30 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-18 14:15:00 | 2945.00 | 3052.20 | 2955.38 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 11:15:00 | 2965.70 | 2987.42 | 2987.47 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 3016.10 | 2986.74 | 2986.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 3021.20 | 2987.74 | 2987.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 14:15:00 | 3064.00 | 3068.32 | 3035.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-01 10:15:00 | 3129.10 | 3038.04 | 3028.75 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-23 14:15:00 | 1800.00 | 2024-08-30 11:15:00 | 1889.00 | EXIT_EMA400 | -89.00 |
| SELL | 2024-08-28 10:15:00 | 1862.50 | 2024-08-30 11:15:00 | 1889.00 | EXIT_EMA400 | -26.50 |
| BUY | 2024-10-07 11:15:00 | 1951.55 | 2024-10-09 09:15:00 | 2043.69 | TARGET | 92.14 |
| BUY | 2024-09-25 11:15:00 | 1971.93 | 2024-10-18 12:15:00 | 2151.85 | TARGET | 179.93 |
| BUY | 2024-10-04 11:15:00 | 1978.00 | 2024-10-18 12:15:00 | 2152.86 | TARGET | 174.86 |
| SELL | 2025-02-07 10:15:00 | 1864.40 | 2025-02-20 09:15:00 | 1929.97 | EXIT_EMA400 | -65.57 |
| BUY | 2025-10-06 11:15:00 | 2919.90 | 2025-11-18 14:15:00 | 2945.00 | EXIT_EMA400 | 25.10 |
| BUY | 2026-04-01 10:15:00 | 3129.10 | 2026-04-09 09:15:00 | 3430.16 | TARGET | 301.06 |
