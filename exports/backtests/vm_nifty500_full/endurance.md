# Endurance Technologies Ltd. (ENDURANCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 2330.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 226.01
- **Avg P&L per closed trade:** 28.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 13:15:00 | 1586.70 | 1616.74 | 1616.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 15:15:00 | 1576.35 | 1616.00 | 1616.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 1604.80 | 1604.71 | 1610.27 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 14:15:00 | 1720.10 | 1610.06 | 1609.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 15:15:00 | 1729.55 | 1633.94 | 1624.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 09:15:00 | 1648.95 | 1650.42 | 1635.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-12 09:15:00 | 1692.00 | 1651.10 | 1635.88 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-02-09 09:15:00 | 1896.45 | 1998.20 | 1903.03 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 09:15:00 | 1795.00 | 1859.29 | 1859.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 14:15:00 | 1719.85 | 1854.88 | 1857.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 13:15:00 | 1829.65 | 1821.34 | 1837.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-28 09:15:00 | 1791.85 | 1820.89 | 1836.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 1791.85 | 1820.89 | 1836.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-28 12:15:00 | 1837.75 | 1820.56 | 1836.37 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 14:15:00 | 1908.00 | 1847.75 | 1847.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 09:15:00 | 1914.15 | 1848.90 | 1848.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 12:15:00 | 1853.70 | 1855.48 | 1851.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-16 09:15:00 | 1882.95 | 1855.42 | 1851.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1882.95 | 1855.42 | 1851.88 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-16 10:15:00 | 1903.45 | 1855.90 | 1852.13 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 1855.85 | 1858.59 | 1853.77 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-19 10:15:00 | 1850.65 | 1858.51 | 1853.75 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 13:15:00 | 2381.00 | 2479.18 | 2479.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 09:15:00 | 2366.85 | 2476.18 | 2478.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 10:15:00 | 2391.10 | 2385.79 | 2423.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-29 09:15:00 | 2311.25 | 2389.65 | 2402.11 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-18 10:15:00 | 1980.55 | 1891.46 | 1978.40 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 2150.50 | 1961.12 | 1960.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 2187.80 | 1963.37 | 1962.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 2596.60 | 2617.20 | 2492.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-04 12:15:00 | 2645.00 | 2593.69 | 2505.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 2512.80 | 2589.76 | 2511.96 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-07 13:15:00 | 2507.00 | 2588.94 | 2511.93 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 2677.40 | 2797.10 | 2797.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 2648.00 | 2794.38 | 2796.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 2696.80 | 2670.43 | 2715.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-22 10:15:00 | 2656.10 | 2670.51 | 2714.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2492.80 | 2500.58 | 2577.08 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-03 10:15:00 | 2481.60 | 2500.39 | 2576.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2555.70 | 2492.65 | 2558.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-11 10:15:00 | 2603.80 | 2493.76 | 2558.24 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-12 09:15:00 | 1692.00 | 2023-12-22 10:15:00 | 1860.35 | TARGET | 168.35 |
| SELL | 2024-03-28 09:15:00 | 1791.85 | 2024-03-28 12:15:00 | 1837.75 | EXIT_EMA400 | -45.90 |
| BUY | 2024-04-16 09:15:00 | 1882.95 | 2024-04-19 10:15:00 | 1850.65 | EXIT_EMA400 | -32.30 |
| BUY | 2024-04-16 10:15:00 | 1903.45 | 2024-04-19 10:15:00 | 1850.65 | EXIT_EMA400 | -52.80 |
| SELL | 2024-11-29 09:15:00 | 2311.25 | 2025-01-13 12:15:00 | 2038.67 | TARGET | 272.58 |
| BUY | 2025-08-04 12:15:00 | 2645.00 | 2025-08-07 13:15:00 | 2507.00 | EXIT_EMA400 | -138.00 |
| SELL | 2025-12-22 10:15:00 | 2656.10 | 2026-01-16 09:15:00 | 2479.82 | TARGET | 176.28 |
| SELL | 2026-02-03 10:15:00 | 2481.60 | 2026-02-11 10:15:00 | 2603.80 | EXIT_EMA400 | -122.20 |
