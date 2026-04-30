# Indiamart Intermesh Ltd. (INDIAMART.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 2103.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 322.10
- **Avg P&L per closed trade:** 53.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 11:15:00 | 2873.45 | 3009.46 | 3010.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 15:15:00 | 2855.10 | 3003.74 | 3007.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 10:15:00 | 2650.05 | 2644.03 | 2729.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-09 13:15:00 | 2610.20 | 2718.50 | 2735.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 2643.25 | 2599.28 | 2649.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-02-07 12:15:00 | 2663.00 | 2599.92 | 2649.96 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 13:15:00 | 2800.05 | 2639.64 | 2639.32 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 14:15:00 | 2592.05 | 2644.69 | 2644.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 2577.60 | 2643.55 | 2644.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 11:15:00 | 2570.95 | 2561.21 | 2595.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-11 14:15:00 | 2550.90 | 2562.14 | 2593.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 2591.40 | 2562.31 | 2593.65 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-12 13:15:00 | 2606.50 | 2563.41 | 2593.58 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 15:15:00 | 2680.85 | 2610.36 | 2610.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 2688.00 | 2613.69 | 2611.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 2790.00 | 2832.99 | 2751.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-20 09:15:00 | 2878.00 | 2787.68 | 2750.06 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-25 12:15:00 | 2912.70 | 2992.74 | 2916.95 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 2490.75 | 2900.89 | 2901.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 2464.00 | 2880.18 | 2891.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 2358.95 | 2318.36 | 2417.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-10 09:15:00 | 2295.00 | 2319.06 | 2414.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-20 13:15:00 | 2129.50 | 2043.73 | 2126.87 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 2239.70 | 2135.04 | 2134.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 2254.20 | 2136.23 | 2135.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 2556.30 | 2561.83 | 2471.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-24 09:15:00 | 2626.90 | 2563.61 | 2478.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-05 09:15:00 | 2468.00 | 2567.07 | 2501.34 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 2375.00 | 2531.52 | 2531.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 15:15:00 | 2360.00 | 2529.81 | 2530.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 2439.00 | 2411.35 | 2453.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-20 13:15:00 | 2388.70 | 2440.43 | 2453.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-16 09:15:00 | 2282.80 | 2217.72 | 2281.55 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-09 13:15:00 | 2610.20 | 2024-02-07 12:15:00 | 2663.00 | EXIT_EMA400 | -52.80 |
| SELL | 2024-06-11 14:15:00 | 2550.90 | 2024-06-12 13:15:00 | 2606.50 | EXIT_EMA400 | -55.60 |
| BUY | 2024-08-20 09:15:00 | 2878.00 | 2024-09-25 12:15:00 | 2912.70 | EXIT_EMA400 | 34.70 |
| SELL | 2025-01-10 09:15:00 | 2295.00 | 2025-02-28 09:15:00 | 1935.21 | TARGET | 359.79 |
| BUY | 2025-07-24 09:15:00 | 2626.90 | 2025-08-05 09:15:00 | 2468.00 | EXIT_EMA400 | -158.90 |
| SELL | 2025-11-20 13:15:00 | 2388.70 | 2025-12-30 09:15:00 | 2193.79 | TARGET | 194.91 |
