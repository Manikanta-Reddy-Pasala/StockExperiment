# Poly Medicure Ltd. (POLYMED.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1512.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 454.66
- **Avg P&L per closed trade:** 90.93

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 14:15:00 | 1393.10 | 1467.43 | 1467.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 09:15:00 | 1386.80 | 1465.94 | 1467.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 09:15:00 | 1481.25 | 1459.44 | 1463.57 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 15:15:00 | 1521.90 | 1467.66 | 1467.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 09:15:00 | 1534.75 | 1468.33 | 1467.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 1558.00 | 1559.55 | 1529.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-27 12:15:00 | 1578.30 | 1525.48 | 1517.73 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 1532.90 | 1543.31 | 1529.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-09 09:15:00 | 1570.70 | 1543.52 | 1529.92 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-04-15 09:15:00 | 1520.50 | 1546.27 | 1532.75 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 15:15:00 | 2392.00 | 2620.45 | 2621.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 09:15:00 | 2348.30 | 2617.74 | 2620.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 11:15:00 | 2529.65 | 2492.94 | 2548.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 10:15:00 | 2407.95 | 2488.38 | 2542.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2318.00 | 2246.45 | 2325.86 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 2337.90 | 2247.36 | 2325.92 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 11:15:00 | 2585.80 | 2314.48 | 2313.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 2603.00 | 2366.74 | 2341.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 2450.30 | 2455.53 | 2395.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-20 10:15:00 | 2481.70 | 2437.29 | 2395.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 2408.90 | 2442.21 | 2402.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-26 09:15:00 | 2386.80 | 2440.97 | 2402.46 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 2243.60 | 2374.85 | 2375.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 2237.40 | 2372.22 | 2374.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 2233.10 | 2231.88 | 2285.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-15 12:15:00 | 2182.40 | 2228.34 | 2266.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 2051.90 | 2003.01 | 2088.94 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-25 12:15:00 | 2126.00 | 2008.27 | 2088.65 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-03-27 12:15:00 | 1578.30 | 2024-04-15 09:15:00 | 1520.50 | EXIT_EMA400 | -57.80 |
| BUY | 2024-04-09 09:15:00 | 1570.70 | 2024-04-15 09:15:00 | 1520.50 | EXIT_EMA400 | -50.20 |
| SELL | 2025-02-06 10:15:00 | 2407.95 | 2025-02-28 10:15:00 | 2003.71 | TARGET | 404.24 |
| BUY | 2025-05-20 10:15:00 | 2481.70 | 2025-05-26 09:15:00 | 2386.80 | EXIT_EMA400 | -94.90 |
| SELL | 2025-07-15 12:15:00 | 2182.40 | 2025-08-01 09:15:00 | 1929.08 | TARGET | 253.32 |
