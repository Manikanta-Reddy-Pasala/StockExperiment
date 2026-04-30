# Hindustan Unilever Ltd. (HINDUNILVR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2254.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 202.07
- **Avg P&L per closed trade:** 28.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 14:15:00 | 2485.93 | 2720.94 | 2722.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2476.39 | 2663.03 | 2690.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 13:15:00 | 2362.38 | 2359.08 | 2436.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 2344.83 | 2359.07 | 2435.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-09 10:15:00 | 2429.86 | 2358.38 | 2427.07 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 2306.71 | 2276.05 | 2275.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 2328.45 | 2276.89 | 2276.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2280.54 | 2286.19 | 2281.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 2346.55 | 2287.10 | 2282.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 2293.43 | 2304.01 | 2292.90 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-22 12:15:00 | 2287.23 | 2303.67 | 2292.90 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 2240.31 | 2298.59 | 2298.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 2232.74 | 2297.93 | 2298.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 2283.10 | 2282.41 | 2289.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-03 14:15:00 | 2276.31 | 2282.35 | 2289.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 2283.20 | 2282.30 | 2289.37 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-04 10:15:00 | 2290.08 | 2282.38 | 2289.37 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 2391.99 | 2296.04 | 2295.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 2475.01 | 2306.18 | 2301.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 2368.88 | 2375.75 | 2344.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-28 09:15:00 | 2400.75 | 2375.98 | 2345.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-23 10:15:00 | 2494.20 | 2546.03 | 2499.48 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 13:15:00 | 2416.78 | 2487.39 | 2487.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 2409.99 | 2485.29 | 2486.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 2418.95 | 2418.52 | 2443.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-27 12:15:00 | 2411.67 | 2418.46 | 2443.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2437.54 | 2418.77 | 2442.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-01 10:15:00 | 2445.41 | 2419.03 | 2442.04 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 2465.00 | 2379.87 | 2379.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 2466.90 | 2382.31 | 2380.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.93 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 2296.30 | 2379.30 | 2379.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 2283.40 | 2364.72 | 2371.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 11:15:00 | 2360.10 | 2358.38 | 2367.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-27 09:15:00 | 2343.30 | 2359.86 | 2367.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2343.30 | 2359.86 | 2367.68 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-02 09:15:00 | 2315.30 | 2358.81 | 2366.88 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-17 10:15:00 | 2226.10 | 2157.34 | 2216.96 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-06 09:15:00 | 2344.83 | 2025-01-09 10:15:00 | 2429.86 | EXIT_EMA400 | -85.03 |
| BUY | 2025-05-12 09:15:00 | 2346.55 | 2025-05-22 12:15:00 | 2287.23 | EXIT_EMA400 | -59.32 |
| SELL | 2025-07-03 14:15:00 | 2276.31 | 2025-07-04 10:15:00 | 2290.08 | EXIT_EMA400 | -13.77 |
| BUY | 2025-07-28 09:15:00 | 2400.75 | 2025-08-01 09:15:00 | 2566.79 | TARGET | 166.04 |
| SELL | 2025-11-27 12:15:00 | 2411.67 | 2025-12-01 10:15:00 | 2445.41 | EXIT_EMA400 | -33.74 |
| SELL | 2026-02-27 09:15:00 | 2343.30 | 2026-03-04 14:15:00 | 2270.16 | TARGET | 73.14 |
| SELL | 2026-03-02 09:15:00 | 2315.30 | 2026-03-11 12:15:00 | 2160.56 | TARGET | 154.74 |
