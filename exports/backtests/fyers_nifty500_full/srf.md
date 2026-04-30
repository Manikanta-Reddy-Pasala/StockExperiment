# SRF Ltd. (SRF.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2525.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 1 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 7 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -543.14
- **Avg P&L per closed trade:** -77.59

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 09:15:00 | 2332.80 | 2456.76 | 2457.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 2272.05 | 2409.69 | 2430.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 2324.80 | 2316.61 | 2365.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 15:15:00 | 2294.00 | 2322.29 | 2364.19 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 13:15:00 | 2318.00 | 2272.01 | 2314.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 11:15:00 | 2608.65 | 2314.15 | 2313.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 2617.10 | 2377.67 | 2347.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 2879.10 | 2889.80 | 2779.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-02 11:15:00 | 2938.10 | 2889.93 | 2784.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 2680.15 | 2887.00 | 2792.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 2843.00 | 3058.51 | 3058.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 2831.00 | 2980.92 | 3012.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2961.50 | 2938.82 | 2979.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-16 10:15:00 | 2931.20 | 2946.68 | 2978.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-17 09:15:00 | 2982.00 | 2946.94 | 2977.58 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 3200.50 | 2969.90 | 2969.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3204.10 | 2972.23 | 2970.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 3009.50 | 3013.44 | 2994.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-28 12:15:00 | 3027.50 | 3013.57 | 2994.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-30 09:15:00 | 2988.80 | 3014.95 | 2996.06 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 2902.50 | 2981.90 | 2981.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 2885.80 | 2979.54 | 2980.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 2919.80 | 2895.86 | 2930.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-03 13:15:00 | 2832.80 | 2894.07 | 2925.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-10 09:15:00 | 2936.70 | 2884.34 | 2915.57 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 3083.00 | 2938.21 | 2937.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 3126.90 | 2940.09 | 2938.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 2976.30 | 3009.61 | 2980.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-07 09:15:00 | 3084.80 | 3011.87 | 2982.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 2994.00 | 3023.52 | 2995.61 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2706.60 | 2971.95 | 2972.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 2693.80 | 2954.55 | 2963.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2929.60 | 2895.15 | 2928.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 13:15:00 | 2921.00 | 2896.62 | 2928.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2928.20 | 2897.26 | 2928.52 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 11:15:00 | 2939.50 | 2897.83 | 2928.49 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 15:15:00 | 2294.00 | 2024-12-03 13:15:00 | 2318.00 | EXIT_EMA400 | -24.00 |
| BUY | 2025-04-02 11:15:00 | 2938.10 | 2025-04-07 09:15:00 | 2680.15 | EXIT_EMA400 | -257.95 |
| SELL | 2025-09-16 10:15:00 | 2931.20 | 2025-09-17 09:15:00 | 2982.00 | EXIT_EMA400 | -50.80 |
| BUY | 2025-10-28 12:15:00 | 3027.50 | 2025-10-30 09:15:00 | 2988.80 | EXIT_EMA400 | -38.70 |
| SELL | 2025-12-03 13:15:00 | 2832.80 | 2025-12-10 09:15:00 | 2936.70 | EXIT_EMA400 | -103.90 |
| BUY | 2026-01-07 09:15:00 | 3084.80 | 2026-01-19 09:15:00 | 2994.00 | EXIT_EMA400 | -90.80 |
| SELL | 2026-02-03 13:15:00 | 2921.00 | 2026-02-04 09:15:00 | 2897.99 | TARGET | 23.01 |
