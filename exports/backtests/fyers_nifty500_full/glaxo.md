# Glaxosmithkline Pharmaceuticals Ltd. (GLAXO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2348.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 3 / 1
- **Target hits / EMA400 exits:** 3 / 1
- **Total realized P&L (per unit):** 771.05
- **Avg P&L per closed trade:** 192.76

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 2638.45 | 2721.51 | 2721.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 2617.75 | 2720.47 | 2721.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 2712.70 | 2695.48 | 2707.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-30 13:15:00 | 2667.05 | 2695.53 | 2707.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-31 09:15:00 | 2718.75 | 2695.33 | 2707.25 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 10:15:00 | 2599.00 | 2261.76 | 2260.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 11:15:00 | 2651.20 | 2313.89 | 2288.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2591.80 | 2684.31 | 2550.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 11:15:00 | 2675.60 | 2677.10 | 2552.35 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 2716.10 | 2806.58 | 2696.35 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-09 12:15:00 | 2738.10 | 2804.28 | 2696.83 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 3149.00 | 3237.92 | 3142.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-18 12:15:00 | 3132.60 | 3236.00 | 3142.68 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 2671.30 | 3098.00 | 3099.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 10:15:00 | 2643.40 | 3072.66 | 3086.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 2859.20 | 2845.44 | 2920.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 10:15:00 | 2808.10 | 2845.97 | 2917.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-11 14:15:00 | 2628.00 | 2557.23 | 2623.62 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 2604.50 | 2506.42 | 2506.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 2623.80 | 2509.34 | 2507.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 2532.50 | 2541.51 | 2525.50 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 2399.80 | 2514.88 | 2515.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 2390.10 | 2496.18 | 2505.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 2414.00 | 2400.01 | 2443.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-30 09:15:00 | 2348.00 | 2418.81 | 2438.93 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-30 13:15:00 | 2667.05 | 2024-10-31 09:15:00 | 2718.75 | EXIT_EMA400 | -51.70 |
| BUY | 2025-04-08 11:15:00 | 2675.60 | 2025-04-24 09:15:00 | 3045.34 | TARGET | 369.74 |
| BUY | 2025-05-09 12:15:00 | 2738.10 | 2025-05-12 09:15:00 | 2861.91 | TARGET | 123.81 |
| SELL | 2025-09-12 10:15:00 | 2808.10 | 2025-11-11 09:15:00 | 2478.91 | TARGET | 329.19 |
