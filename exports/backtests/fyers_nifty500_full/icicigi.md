# ICICI Lombard General Insurance Company Ltd. (ICICIGI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1761.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -107.27
- **Avg P&L per closed trade:** -17.88

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 12:15:00 | 1935.50 | 2047.11 | 2047.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 1919.55 | 2045.85 | 2046.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 1919.90 | 1905.57 | 1952.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-18 13:15:00 | 1882.20 | 1925.96 | 1950.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 1897.00 | 1867.60 | 1899.96 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-16 09:15:00 | 1912.30 | 1868.26 | 1899.97 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 09:15:00 | 1862.00 | 1790.93 | 1790.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 11:15:00 | 1909.30 | 1829.47 | 1814.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 1839.40 | 1841.25 | 1823.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-28 12:15:00 | 1854.20 | 1841.34 | 1824.36 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-18 09:15:00 | 1938.90 | 1982.29 | 1939.29 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 1847.80 | 1923.02 | 1923.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 10:15:00 | 1832.50 | 1917.64 | 1920.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 1898.10 | 1883.97 | 1899.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-17 13:15:00 | 1868.00 | 1886.69 | 1899.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1891.70 | 1884.52 | 1896.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-24 10:15:00 | 1871.60 | 1884.80 | 1896.69 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-26 09:15:00 | 1900.10 | 1884.17 | 1895.62 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 2007.20 | 1900.19 | 1900.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 2027.80 | 1908.38 | 1904.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 1995.70 | 1996.81 | 1966.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-25 09:15:00 | 2008.00 | 1996.74 | 1966.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1970.70 | 1996.28 | 1969.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-28 10:15:00 | 1959.10 | 1995.91 | 1969.65 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 1950.00 | 1960.04 | 1960.07 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1979.90 | 1960.23 | 1960.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 1988.40 | 1960.51 | 1960.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1946.50 | 1965.51 | 1962.98 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1895.40 | 1960.57 | 1960.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 1874.00 | 1951.38 | 1955.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 13:15:00 | 1875.50 | 1873.77 | 1904.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-06 11:15:00 | 1848.70 | 1873.26 | 1903.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1899.90 | 1873.24 | 1901.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-10 12:15:00 | 1905.90 | 1873.82 | 1901.79 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-18 13:15:00 | 1882.20 | 2025-01-16 09:15:00 | 1912.30 | EXIT_EMA400 | -30.10 |
| BUY | 2025-05-28 12:15:00 | 1854.20 | 2025-06-06 13:15:00 | 1943.73 | TARGET | 89.53 |
| SELL | 2025-09-17 13:15:00 | 1868.00 | 2025-09-26 09:15:00 | 1900.10 | EXIT_EMA400 | -32.10 |
| SELL | 2025-09-24 10:15:00 | 1871.60 | 2025-09-26 09:15:00 | 1900.10 | EXIT_EMA400 | -28.50 |
| BUY | 2025-11-25 09:15:00 | 2008.00 | 2025-11-28 10:15:00 | 1959.10 | EXIT_EMA400 | -48.90 |
| SELL | 2026-02-06 11:15:00 | 1848.70 | 2026-02-10 12:15:00 | 1905.90 | EXIT_EMA400 | -57.20 |
