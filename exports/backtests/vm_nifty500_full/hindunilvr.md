# Hindustan Unilever Ltd. (HINDUNILVR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2250.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| EXIT | 8 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 272.24
- **Avg P&L per closed trade:** 34.03

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 12:15:00 | 2572.50 | 2523.67 | 2523.45 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 15:15:00 | 2503.65 | 2523.24 | 2523.30 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 14:15:00 | 2560.00 | 2523.29 | 2523.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 12:15:00 | 2568.05 | 2525.19 | 2524.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 12:15:00 | 2566.55 | 2572.18 | 2552.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-08 14:15:00 | 2577.75 | 2572.23 | 2552.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-11 13:15:00 | 2544.75 | 2572.46 | 2554.60 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 11:15:00 | 2457.00 | 2543.39 | 2543.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 12:15:00 | 2440.00 | 2542.36 | 2543.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 2436.35 | 2430.06 | 2465.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-05 09:15:00 | 2399.85 | 2428.65 | 2462.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-07 09:15:00 | 2355.00 | 2260.49 | 2311.05 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 11:15:00 | 2438.75 | 2334.12 | 2333.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 13:15:00 | 2497.05 | 2337.09 | 2335.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 11:15:00 | 2435.30 | 2436.74 | 2397.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-21 13:15:00 | 2445.50 | 2436.82 | 2398.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 2815.05 | 2878.80 | 2804.14 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-09 09:15:00 | 2791.35 | 2877.34 | 2804.15 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 14:15:00 | 2527.95 | 2766.03 | 2766.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2519.15 | 2710.56 | 2736.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 13:15:00 | 2401.60 | 2398.37 | 2477.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 2383.75 | 2398.36 | 2476.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-09 10:15:00 | 2470.00 | 2397.59 | 2467.59 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 2347.20 | 2313.33 | 2313.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 2367.70 | 2314.53 | 2313.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2318.40 | 2324.04 | 2318.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 2385.50 | 2324.94 | 2319.54 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 2331.10 | 2342.15 | 2330.68 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-22 12:15:00 | 2325.20 | 2341.81 | 2330.68 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 2278.10 | 2336.79 | 2336.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 2269.80 | 2336.12 | 2336.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 2321.00 | 2320.27 | 2327.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-03 14:15:00 | 2314.20 | 2320.21 | 2327.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 2321.00 | 2320.17 | 2327.29 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-04 10:15:00 | 2328.10 | 2320.24 | 2327.29 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 2431.70 | 2334.16 | 2333.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 2516.00 | 2344.48 | 2339.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 2408.50 | 2415.21 | 2383.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-28 09:15:00 | 2440.60 | 2415.45 | 2384.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-23 10:15:00 | 2535.60 | 2588.37 | 2541.03 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 13:15:00 | 2456.10 | 2528.67 | 2528.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 2449.90 | 2526.54 | 2527.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 2459.10 | 2458.64 | 2483.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-27 12:15:00 | 2451.70 | 2458.57 | 2483.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2478.00 | 2458.82 | 2482.51 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-01 10:15:00 | 2486.00 | 2459.09 | 2482.52 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-08 14:15:00 | 2577.75 | 2024-01-11 13:15:00 | 2544.75 | EXIT_EMA400 | -33.00 |
| SELL | 2024-03-05 09:15:00 | 2399.85 | 2024-04-15 09:15:00 | 2211.15 | TARGET | 188.70 |
| BUY | 2024-06-21 13:15:00 | 2445.50 | 2024-07-08 09:15:00 | 2587.91 | TARGET | 142.41 |
| SELL | 2025-01-06 09:15:00 | 2383.75 | 2025-01-09 10:15:00 | 2470.00 | EXIT_EMA400 | -86.25 |
| BUY | 2025-05-12 09:15:00 | 2385.50 | 2025-05-22 12:15:00 | 2325.20 | EXIT_EMA400 | -60.30 |
| SELL | 2025-07-03 14:15:00 | 2314.20 | 2025-07-04 10:15:00 | 2328.10 | EXIT_EMA400 | -13.90 |
| BUY | 2025-07-28 09:15:00 | 2440.60 | 2025-08-01 09:15:00 | 2609.48 | TARGET | 168.87 |
| SELL | 2025-11-27 12:15:00 | 2451.70 | 2025-12-01 10:15:00 | 2486.00 | EXIT_EMA400 | -34.30 |
