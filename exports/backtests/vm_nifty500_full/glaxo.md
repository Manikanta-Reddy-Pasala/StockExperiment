# Glaxosmithkline Pharmaceuticals Ltd. (GLAXO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 2336.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 820.91
- **Avg P&L per closed trade:** 136.82

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 12:15:00 | 1943.00 | 2052.47 | 2052.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 13:15:00 | 1932.15 | 2051.27 | 2052.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 1992.70 | 1964.82 | 1998.27 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 14:15:00 | 2057.10 | 2023.25 | 2023.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 09:15:00 | 2059.45 | 2023.80 | 2023.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 12:15:00 | 2017.75 | 2023.90 | 2023.43 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 1998.30 | 2022.81 | 2022.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 13:15:00 | 1988.20 | 2022.47 | 2022.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 12:15:00 | 2037.45 | 2013.33 | 2017.69 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 15:15:00 | 2299.50 | 2023.76 | 2022.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 09:15:00 | 2350.75 | 2027.01 | 2024.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 13:15:00 | 2538.00 | 2539.42 | 2403.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-09 09:15:00 | 2594.60 | 2538.89 | 2409.68 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 2811.40 | 2825.22 | 2744.35 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-18 10:15:00 | 2825.70 | 2824.72 | 2745.30 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 2747.60 | 2822.93 | 2746.74 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-19 10:15:00 | 2740.00 | 2822.10 | 2746.71 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 11:15:00 | 2609.80 | 2723.13 | 2723.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 2574.25 | 2712.73 | 2717.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 2715.85 | 2695.32 | 2708.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-30 13:15:00 | 2667.05 | 2695.38 | 2707.97 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-31 09:15:00 | 2718.75 | 2695.22 | 2707.70 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 11:15:00 | 2592.80 | 2269.12 | 2268.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 13:15:00 | 2607.15 | 2275.65 | 2271.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2591.80 | 2684.96 | 2552.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 11:15:00 | 2675.60 | 2677.46 | 2554.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 2716.10 | 2806.74 | 2697.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-09 12:15:00 | 2738.10 | 2804.43 | 2698.03 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 3149.00 | 3237.98 | 3143.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-18 12:15:00 | 3132.60 | 3236.06 | 3142.98 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 2671.30 | 3097.99 | 3100.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 10:15:00 | 2644.70 | 3072.68 | 3087.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 2859.20 | 2845.59 | 2920.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 10:15:00 | 2808.10 | 2846.14 | 2917.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-11 14:15:00 | 2628.00 | 2557.53 | 2623.85 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 2623.80 | 2509.75 | 2509.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 2663.40 | 2511.28 | 2510.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 2532.50 | 2541.62 | 2526.90 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 2394.20 | 2516.15 | 2516.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 2375.80 | 2495.08 | 2505.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 2414.00 | 2400.16 | 2443.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-30 09:15:00 | 2348.00 | 2418.28 | 2439.47 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-07-09 09:15:00 | 2594.60 | 2024-09-19 10:15:00 | 2740.00 | EXIT_EMA400 | 145.40 |
| BUY | 2024-09-18 10:15:00 | 2825.70 | 2024-09-19 10:15:00 | 2740.00 | EXIT_EMA400 | -85.70 |
| SELL | 2024-10-30 13:15:00 | 2667.05 | 2024-10-31 09:15:00 | 2718.75 | EXIT_EMA400 | -51.70 |
| BUY | 2025-04-08 11:15:00 | 2675.60 | 2025-04-24 09:15:00 | 3038.64 | TARGET | 363.04 |
| BUY | 2025-05-09 12:15:00 | 2738.10 | 2025-05-14 09:15:00 | 2858.30 | TARGET | 120.20 |
| SELL | 2025-09-12 10:15:00 | 2808.10 | 2025-11-11 09:15:00 | 2478.42 | TARGET | 329.68 |
