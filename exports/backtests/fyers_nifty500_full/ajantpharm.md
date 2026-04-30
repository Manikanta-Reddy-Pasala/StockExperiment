# Ajanta Pharmaceuticals Ltd. (AJANTPHARM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2830.00
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
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 615.83
- **Avg P&L per closed trade:** 87.98

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 12:15:00 | 2860.30 | 3034.45 | 3035.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 13:15:00 | 2837.15 | 3032.49 | 3034.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 3085.70 | 3006.35 | 3019.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 14:15:00 | 2968.65 | 3007.01 | 3019.52 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 2968.65 | 3007.01 | 3019.52 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-27 14:15:00 | 3034.05 | 3004.81 | 3017.53 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 2746.10 | 2607.04 | 2606.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 12:15:00 | 2775.80 | 2610.16 | 2608.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 13:15:00 | 2695.50 | 2701.95 | 2664.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-19 15:15:00 | 2737.00 | 2674.17 | 2659.35 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 2691.50 | 2674.31 | 2659.57 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-21 13:15:00 | 2659.50 | 2674.45 | 2660.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 11:15:00 | 2486.40 | 2647.65 | 2648.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 2478.90 | 2645.97 | 2647.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 2615.90 | 2609.98 | 2627.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-09 12:15:00 | 2564.10 | 2607.61 | 2624.90 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-03 13:15:00 | 2530.20 | 2457.08 | 2501.63 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 2608.00 | 2520.48 | 2520.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 2636.20 | 2526.96 | 2523.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 2570.80 | 2575.15 | 2552.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-19 09:15:00 | 2670.00 | 2576.24 | 2554.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2682.40 | 2717.60 | 2658.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-19 10:15:00 | 2696.00 | 2717.39 | 2658.25 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2718.60 | 2713.05 | 2665.09 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-30 09:15:00 | 2766.40 | 2711.52 | 2669.06 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 2898.80 | 2953.86 | 2876.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-20 15:15:00 | 3020.00 | 2954.52 | 2876.98 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 2843.10 | 2953.41 | 2876.81 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 2793.20 | 2840.03 | 2840.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 14:15:00 | 2758.60 | 2836.32 | 2838.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 2833.30 | 2826.83 | 2832.90 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-25 14:15:00 | 2968.65 | 2024-11-27 14:15:00 | 3034.05 | EXIT_EMA400 | -65.40 |
| BUY | 2025-08-19 15:15:00 | 2737.00 | 2025-08-21 13:15:00 | 2659.50 | EXIT_EMA400 | -77.50 |
| SELL | 2025-09-09 12:15:00 | 2564.10 | 2025-09-30 13:15:00 | 2381.70 | TARGET | 182.40 |
| BUY | 2025-12-19 09:15:00 | 2670.00 | 2026-01-02 12:15:00 | 3017.96 | TARGET | 347.96 |
| BUY | 2026-01-19 10:15:00 | 2696.00 | 2026-02-01 09:15:00 | 2809.24 | TARGET | 113.24 |
| BUY | 2026-01-30 09:15:00 | 2766.40 | 2026-03-11 09:15:00 | 3058.43 | TARGET | 292.03 |
| BUY | 2026-03-20 15:15:00 | 3020.00 | 2026-03-23 09:15:00 | 2843.10 | EXIT_EMA400 | -176.90 |
