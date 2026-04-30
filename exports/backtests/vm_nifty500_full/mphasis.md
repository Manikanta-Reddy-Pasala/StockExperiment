# MphasiS Ltd. (MPHASIS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2276.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 8 |
| ENTRY1 | 8 |
| ENTRY2 | 3 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 5 / 6
- **Target hits / EMA400 exits:** 5 / 6
- **Total realized P&L (per unit):** 319.07
- **Avg P&L per closed trade:** 29.01

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 11:15:00 | 2130.80 | 2314.32 | 2314.82 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 15:15:00 | 2354.65 | 2291.95 | 2291.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 13:15:00 | 2356.45 | 2294.83 | 2293.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 14:15:00 | 2555.70 | 2564.78 | 2477.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-11 09:15:00 | 2578.60 | 2563.58 | 2480.67 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 2513.55 | 2581.60 | 2503.86 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-18 13:15:00 | 2503.00 | 2579.53 | 2504.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 2441.20 | 2555.54 | 2555.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 12:15:00 | 2417.85 | 2552.98 | 2554.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 10:15:00 | 2501.10 | 2492.34 | 2518.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-03 14:15:00 | 2480.00 | 2492.68 | 2518.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 2513.00 | 2492.76 | 2517.62 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-05 11:15:00 | 2499.80 | 2493.51 | 2517.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 14:15:00 | 2511.85 | 2493.84 | 2517.19 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-08 09:15:00 | 2472.20 | 2493.72 | 2516.90 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 2498.00 | 2491.56 | 2515.00 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-09 12:15:00 | 2458.25 | 2491.05 | 2514.40 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 2387.65 | 2335.49 | 2390.57 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-05-22 10:15:00 | 2401.75 | 2336.15 | 2390.63 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 2521.85 | 2400.38 | 2400.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 11:15:00 | 2529.60 | 2401.67 | 2400.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 2667.90 | 2732.53 | 2616.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-07 10:15:00 | 2714.60 | 2722.37 | 2619.91 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 2927.55 | 3019.83 | 2923.99 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-03 13:15:00 | 2915.00 | 3017.85 | 2923.95 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 14:15:00 | 2753.70 | 2923.64 | 2924.21 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 13:15:00 | 2950.55 | 2922.10 | 2922.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 2975.10 | 2923.35 | 2922.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 11:15:00 | 3016.80 | 3055.16 | 3003.15 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 11:15:00 | 2909.95 | 2969.46 | 2969.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 2890.85 | 2966.98 | 2968.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 2903.00 | 2889.66 | 2922.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-24 09:15:00 | 2790.90 | 2890.74 | 2922.36 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-24 11:15:00 | 3025.40 | 2891.68 | 2922.51 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 2565.30 | 2494.36 | 2494.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 2570.00 | 2495.12 | 2494.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 2465.10 | 2498.78 | 2496.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 10:15:00 | 2490.60 | 2498.70 | 2496.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 2490.60 | 2498.70 | 2496.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-02 11:15:00 | 2488.50 | 2498.60 | 2496.37 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 2740.70 | 2780.29 | 2780.40 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 2806.20 | 2780.55 | 2780.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 2826.80 | 2782.41 | 2781.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 2794.00 | 2804.25 | 2793.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-12 10:15:00 | 2843.00 | 2790.49 | 2787.97 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-14 09:15:00 | 2762.90 | 2794.66 | 2790.30 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 2662.50 | 2785.88 | 2786.11 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 2819.30 | 2783.48 | 2783.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 10:15:00 | 2868.10 | 2786.23 | 2784.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 2852.50 | 2856.77 | 2830.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-29 09:15:00 | 2863.80 | 2856.65 | 2831.07 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-29 11:15:00 | 2824.50 | 2856.20 | 2831.10 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 2752.90 | 2823.11 | 2823.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 2730.50 | 2819.61 | 2821.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2846.00 | 2816.17 | 2819.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 10:15:00 | 2825.50 | 2816.26 | 2819.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 2825.50 | 2816.26 | 2819.61 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 11:15:00 | 2829.20 | 2816.39 | 2819.66 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-11 09:15:00 | 2578.60 | 2024-01-18 13:15:00 | 2503.00 | EXIT_EMA400 | -75.60 |
| SELL | 2024-04-05 11:15:00 | 2499.80 | 2024-04-08 13:15:00 | 2447.05 | TARGET | 52.75 |
| SELL | 2024-04-03 14:15:00 | 2480.00 | 2024-04-16 09:15:00 | 2364.99 | TARGET | 115.01 |
| SELL | 2024-04-08 09:15:00 | 2472.20 | 2024-04-16 12:15:00 | 2338.09 | TARGET | 134.11 |
| SELL | 2024-04-09 12:15:00 | 2458.25 | 2024-04-19 09:15:00 | 2289.80 | TARGET | 168.45 |
| BUY | 2024-08-07 10:15:00 | 2714.60 | 2024-08-20 09:15:00 | 2998.66 | TARGET | 284.06 |
| SELL | 2025-01-24 09:15:00 | 2790.90 | 2025-01-24 11:15:00 | 3025.40 | EXIT_EMA400 | -234.50 |
| BUY | 2025-06-02 10:15:00 | 2490.60 | 2025-06-02 11:15:00 | 2488.50 | EXIT_EMA400 | -2.10 |
| BUY | 2025-11-12 10:15:00 | 2843.00 | 2025-11-14 09:15:00 | 2762.90 | EXIT_EMA400 | -80.10 |
| BUY | 2025-12-29 09:15:00 | 2863.80 | 2025-12-29 11:15:00 | 2824.50 | EXIT_EMA400 | -39.30 |
| SELL | 2026-02-03 10:15:00 | 2825.50 | 2026-02-03 11:15:00 | 2829.20 | EXIT_EMA400 | -3.70 |
