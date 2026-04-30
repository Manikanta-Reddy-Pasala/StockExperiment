# Data Patterns (India) Ltd. (DATAPATTNS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4078.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / EMA400 exits:** 3 / 7
- **Total realized P&L (per unit):** -105.86
- **Avg P&L per closed trade:** -10.59

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 2882.30 | 3020.80 | 3021.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 15:15:00 | 2860.00 | 3003.24 | 3011.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 15:15:00 | 2533.00 | 2531.42 | 2666.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 09:15:00 | 2460.00 | 2530.71 | 2665.60 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 2531.65 | 2338.80 | 2444.29 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 2546.00 | 2504.32 | 2504.26 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 2481.30 | 2504.12 | 2504.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 2462.70 | 2502.85 | 2503.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 13:15:00 | 2500.45 | 2497.30 | 2500.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 2444.75 | 2496.99 | 2500.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2444.75 | 2496.99 | 2500.15 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-06 10:15:00 | 2388.85 | 2495.91 | 2499.60 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-15 10:15:00 | 1857.90 | 1685.08 | 1793.15 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 2345.00 | 1864.70 | 1864.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 11:15:00 | 2407.40 | 1870.10 | 1867.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 2807.20 | 2850.06 | 2607.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-02 11:15:00 | 2945.10 | 2859.18 | 2653.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 2755.00 | 2888.10 | 2746.33 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-21 09:15:00 | 2674.60 | 2885.98 | 2745.97 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 2512.00 | 2685.04 | 2685.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 2505.70 | 2683.25 | 2684.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2574.70 | 2568.39 | 2610.21 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 2836.00 | 2638.84 | 2638.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 13:15:00 | 2843.90 | 2640.88 | 2639.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 2661.00 | 2678.84 | 2660.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-03 09:15:00 | 2736.70 | 2660.46 | 2653.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 2736.70 | 2660.46 | 2653.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-03 13:15:00 | 2828.20 | 2665.05 | 2655.59 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2709.70 | 2713.73 | 2686.74 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-16 13:15:00 | 2741.50 | 2713.98 | 2687.40 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-30 12:15:00 | 2711.40 | 2745.00 | 2712.06 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 2570.30 | 2779.18 | 2779.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 2532.00 | 2770.81 | 2775.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 11:15:00 | 2720.60 | 2711.70 | 2741.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 13:15:00 | 2685.00 | 2711.23 | 2741.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2716.10 | 2710.70 | 2740.16 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-26 12:15:00 | 2710.80 | 2710.70 | 2740.01 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 2689.10 | 2683.01 | 2719.56 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-05 11:15:00 | 2722.00 | 2683.39 | 2719.57 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 2821.90 | 2655.62 | 2655.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 2863.10 | 2662.65 | 2658.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 13:15:00 | 3075.50 | 3076.66 | 2920.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-16 14:15:00 | 3098.90 | 3076.88 | 2921.37 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-04-02 09:15:00 | 2958.30 | 3127.96 | 3000.37 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-17 09:15:00 | 2460.00 | 2024-11-28 09:15:00 | 2531.65 | EXIT_EMA400 | -71.65 |
| SELL | 2025-01-06 09:15:00 | 2444.75 | 2025-01-10 09:15:00 | 2278.54 | TARGET | 166.21 |
| SELL | 2025-01-06 10:15:00 | 2388.85 | 2025-01-14 09:15:00 | 2056.61 | TARGET | 332.24 |
| BUY | 2025-07-02 11:15:00 | 2945.10 | 2025-07-21 09:15:00 | 2674.60 | EXIT_EMA400 | -270.50 |
| BUY | 2025-10-03 09:15:00 | 2736.70 | 2025-10-30 12:15:00 | 2711.40 | EXIT_EMA400 | -25.30 |
| BUY | 2025-10-03 13:15:00 | 2828.20 | 2025-10-30 12:15:00 | 2711.40 | EXIT_EMA400 | -116.80 |
| BUY | 2025-10-16 13:15:00 | 2741.50 | 2025-10-30 12:15:00 | 2711.40 | EXIT_EMA400 | -30.10 |
| SELL | 2025-12-26 12:15:00 | 2710.80 | 2025-12-30 09:15:00 | 2623.16 | TARGET | 87.64 |
| SELL | 2025-12-24 13:15:00 | 2685.00 | 2026-01-05 11:15:00 | 2722.00 | EXIT_EMA400 | -37.00 |
| BUY | 2026-03-16 14:15:00 | 3098.90 | 2026-04-02 09:15:00 | 2958.30 | EXIT_EMA400 | -140.60 |
