# BEML Ltd. (BEML.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1806.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 659.82
- **Avg P&L per closed trade:** 109.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 09:15:00 | 1911.00 | 2132.52 | 2132.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 10:15:00 | 1880.43 | 2130.01 | 2131.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 2033.70 | 2024.16 | 2068.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-18 13:15:00 | 1889.40 | 1992.80 | 2036.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1943.20 | 1889.06 | 1947.96 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-16 10:15:00 | 1950.80 | 1889.67 | 1947.97 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 2108.48 | 1969.60 | 1969.35 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 13:15:00 | 1885.03 | 1969.51 | 1969.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 09:15:00 | 1869.50 | 1967.05 | 1968.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1996.18 | 1946.27 | 1956.96 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 11:15:00 | 2145.00 | 1967.03 | 1966.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 2175.93 | 2021.69 | 1996.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 2086.35 | 2109.55 | 2056.67 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 1869.50 | 2031.15 | 2031.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 12:15:00 | 1854.05 | 2029.39 | 2030.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 1885.10 | 1874.15 | 1933.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 1762.50 | 1874.96 | 1932.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-27 14:15:00 | 1598.38 | 1389.92 | 1517.67 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 1732.50 | 1547.60 | 1547.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 1790.00 | 1566.54 | 1557.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 2206.05 | 2211.96 | 2077.32 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 1922.45 | 2040.60 | 2040.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 1913.30 | 2036.92 | 2038.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 2050.90 | 2023.60 | 2031.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 14:15:00 | 1975.50 | 2023.98 | 2031.34 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-05 09:15:00 | 2042.50 | 2023.69 | 2031.12 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 2188.30 | 2037.20 | 2036.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 2245.00 | 2084.26 | 2062.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 2096.35 | 2106.20 | 2077.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-01 09:15:00 | 2157.30 | 2101.17 | 2077.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 2138.70 | 2169.66 | 2136.42 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-06 09:15:00 | 2016.50 | 2167.67 | 2135.92 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 2033.10 | 2110.14 | 2110.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 2011.00 | 2109.15 | 2109.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 1857.30 | 1809.04 | 1905.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-12 09:15:00 | 1765.30 | 1835.09 | 1886.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1820.00 | 1779.47 | 1836.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-02 09:15:00 | 1692.80 | 1779.32 | 1834.31 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 1638.90 | 1558.80 | 1632.46 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 1833.30 | 1677.53 | 1676.77 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-18 13:15:00 | 1889.40 | 2024-10-16 10:15:00 | 1950.80 | EXIT_EMA400 | -61.40 |
| SELL | 2025-02-03 09:15:00 | 1762.50 | 2025-02-28 09:15:00 | 1252.40 | TARGET | 510.10 |
| SELL | 2025-09-04 14:15:00 | 1975.50 | 2025-09-05 09:15:00 | 2042.50 | EXIT_EMA400 | -67.00 |
| BUY | 2025-10-01 09:15:00 | 2157.30 | 2025-11-06 09:15:00 | 2016.50 | EXIT_EMA400 | -140.80 |
| SELL | 2026-01-12 09:15:00 | 1765.30 | 2026-03-23 15:15:00 | 1400.28 | TARGET | 365.02 |
| SELL | 2026-02-02 09:15:00 | 1692.80 | 2026-04-10 09:15:00 | 1638.90 | EXIT_EMA400 | 53.90 |
