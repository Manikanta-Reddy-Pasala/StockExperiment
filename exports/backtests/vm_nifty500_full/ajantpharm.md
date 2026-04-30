# Ajanta Pharmaceuticals Ltd. (AJANTPHARM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 2822.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| EXIT | 4 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / EMA400 exits:** 6 / 2
- **Total realized P&L (per unit):** 1144.70
- **Avg P&L per closed trade:** 143.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 15:15:00 | 2842.00 | 3029.81 | 3030.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 14:15:00 | 2841.90 | 2981.09 | 3000.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 11:15:00 | 2924.85 | 2894.80 | 2941.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-10 09:15:00 | 2857.40 | 2922.60 | 2944.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2913.40 | 2883.25 | 2914.99 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-24 09:15:00 | 2845.05 | 2883.26 | 2914.06 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-03 14:15:00 | 3061.90 | 2812.13 | 2867.64 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 2757.20 | 2608.53 | 2607.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 12:15:00 | 2775.80 | 2610.20 | 2608.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 13:15:00 | 2695.50 | 2702.05 | 2665.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-19 15:15:00 | 2720.90 | 2674.13 | 2659.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 2691.50 | 2674.28 | 2659.64 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-21 13:15:00 | 2659.50 | 2674.42 | 2660.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 11:15:00 | 2486.50 | 2647.68 | 2648.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 2479.20 | 2646.01 | 2647.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 2615.90 | 2609.86 | 2627.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-09 12:15:00 | 2565.90 | 2607.57 | 2624.92 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-03 13:15:00 | 2530.20 | 2457.13 | 2501.70 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 2608.00 | 2520.71 | 2520.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 14:15:00 | 2624.00 | 2525.18 | 2522.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 2570.80 | 2575.50 | 2552.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-19 09:15:00 | 2665.50 | 2576.53 | 2554.26 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2682.40 | 2717.85 | 2658.32 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-19 10:15:00 | 2696.00 | 2717.63 | 2658.51 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2718.60 | 2713.14 | 2665.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-30 09:15:00 | 2766.40 | 2711.58 | 2669.20 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 2897.80 | 2953.40 | 2875.09 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-20 15:15:00 | 3019.00 | 2954.05 | 2875.80 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 2843.10 | 2952.95 | 2875.64 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 2782.10 | 2839.16 | 2839.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 14:15:00 | 2758.60 | 2838.36 | 2838.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 2833.30 | 2828.44 | 2833.41 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-10 09:15:00 | 2857.40 | 2025-01-28 09:15:00 | 2597.29 | TARGET | 260.11 |
| SELL | 2025-01-24 09:15:00 | 2845.05 | 2025-01-28 09:15:00 | 2638.02 | TARGET | 207.03 |
| BUY | 2025-08-19 15:15:00 | 2720.90 | 2025-08-21 13:15:00 | 2659.50 | EXIT_EMA400 | -61.40 |
| SELL | 2025-09-09 12:15:00 | 2565.90 | 2025-09-30 13:15:00 | 2388.83 | TARGET | 177.07 |
| BUY | 2025-12-19 09:15:00 | 2665.50 | 2026-01-02 12:15:00 | 2999.23 | TARGET | 333.73 |
| BUY | 2026-01-19 10:15:00 | 2696.00 | 2026-02-02 09:15:00 | 2808.47 | TARGET | 112.47 |
| BUY | 2026-01-30 09:15:00 | 2766.40 | 2026-03-11 09:15:00 | 3058.00 | TARGET | 291.60 |
| BUY | 2026-03-20 15:15:00 | 3019.00 | 2026-03-23 09:15:00 | 2843.10 | EXIT_EMA400 | -175.90 |
