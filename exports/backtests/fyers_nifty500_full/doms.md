# DOMS Industries Ltd. (DOMS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2290.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -548.65
- **Avg P&L per closed trade:** -91.44

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 2586.00 | 2746.47 | 2746.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 12:15:00 | 2573.40 | 2744.75 | 2745.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 2550.70 | 2530.76 | 2614.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 13:15:00 | 2409.00 | 2593.69 | 2627.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 2599.85 | 2583.32 | 2619.87 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-21 10:15:00 | 2536.00 | 2582.13 | 2616.29 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 2565.90 | 2580.01 | 2614.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-24 14:15:00 | 2531.50 | 2579.11 | 2612.91 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 2558.75 | 2545.08 | 2589.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-05 09:15:00 | 2601.45 | 2545.64 | 2589.09 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 15:15:00 | 2796.00 | 2623.40 | 2622.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 2825.55 | 2625.41 | 2623.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 2739.90 | 2748.12 | 2696.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-03 11:15:00 | 2812.80 | 2752.31 | 2702.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-04 12:15:00 | 2667.00 | 2753.57 | 2705.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 2526.80 | 2727.23 | 2727.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 2490.00 | 2718.89 | 2723.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 2486.40 | 2483.48 | 2563.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-02 11:15:00 | 2445.70 | 2486.64 | 2549.38 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-11 09:15:00 | 2446.40 | 2381.72 | 2438.94 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 11:15:00 | 2661.50 | 2463.52 | 2462.78 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 2390.50 | 2544.20 | 2544.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 2351.70 | 2521.40 | 2532.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 13:15:00 | 2468.90 | 2462.08 | 2495.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 09:15:00 | 2418.40 | 2461.47 | 2495.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 2420.20 | 2459.73 | 2493.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-10 12:15:00 | 2492.70 | 2455.54 | 2488.10 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-14 13:15:00 | 2409.00 | 2025-03-05 09:15:00 | 2601.45 | EXIT_EMA400 | -192.45 |
| SELL | 2025-02-21 10:15:00 | 2536.00 | 2025-03-05 09:15:00 | 2601.45 | EXIT_EMA400 | -65.45 |
| SELL | 2025-02-24 14:15:00 | 2531.50 | 2025-03-05 09:15:00 | 2601.45 | EXIT_EMA400 | -69.95 |
| BUY | 2025-04-03 11:15:00 | 2812.80 | 2025-04-04 12:15:00 | 2667.00 | EXIT_EMA400 | -145.80 |
| SELL | 2025-07-02 11:15:00 | 2445.70 | 2025-08-11 09:15:00 | 2446.40 | EXIT_EMA400 | -0.70 |
| SELL | 2026-02-05 09:15:00 | 2418.40 | 2026-02-10 12:15:00 | 2492.70 | EXIT_EMA400 | -74.30 |
