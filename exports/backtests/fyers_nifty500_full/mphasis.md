# MphasiS Ltd. (MPHASIS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2275.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -360.00
- **Avg P&L per closed trade:** -72.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 09:15:00 | 2829.40 | 2920.33 | 2920.51 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 15:15:00 | 3025.00 | 2919.58 | 2919.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 13:15:00 | 3050.50 | 2930.39 | 2924.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 11:15:00 | 3016.80 | 3055.00 | 3002.03 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 14:15:00 | 2923.95 | 2968.14 | 2968.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 15:15:00 | 2920.00 | 2967.66 | 2967.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 2903.00 | 2889.58 | 2922.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-24 09:15:00 | 2790.90 | 2890.60 | 2921.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-24 11:15:00 | 3025.40 | 2891.54 | 2921.98 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 2557.50 | 2493.60 | 2493.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 2565.10 | 2494.31 | 2493.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 2465.10 | 2498.79 | 2496.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 10:15:00 | 2490.90 | 2498.71 | 2496.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 2490.90 | 2498.71 | 2496.15 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-02 11:15:00 | 2488.50 | 2498.61 | 2496.11 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 2740.00 | 2780.22 | 2780.30 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 2806.20 | 2780.48 | 2780.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 2826.80 | 2782.39 | 2781.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 2793.60 | 2804.16 | 2793.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-12 10:15:00 | 2843.00 | 2790.42 | 2787.88 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-14 09:15:00 | 2762.90 | 2794.69 | 2790.27 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 2662.50 | 2785.91 | 2786.07 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 2819.30 | 2783.43 | 2783.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 15:15:00 | 2846.00 | 2784.90 | 2784.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 2852.50 | 2856.65 | 2830.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-29 09:15:00 | 2863.80 | 2856.58 | 2830.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-29 11:15:00 | 2824.50 | 2856.13 | 2831.02 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 2753.40 | 2822.74 | 2822.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 2735.00 | 2819.92 | 2821.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2846.00 | 2812.04 | 2817.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 10:15:00 | 2825.50 | 2812.18 | 2817.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 2825.50 | 2812.18 | 2817.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 11:15:00 | 2829.20 | 2812.35 | 2817.37 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-24 09:15:00 | 2790.90 | 2025-01-24 11:15:00 | 3025.40 | EXIT_EMA400 | -234.50 |
| BUY | 2025-06-02 10:15:00 | 2490.90 | 2025-06-02 11:15:00 | 2488.50 | EXIT_EMA400 | -2.40 |
| BUY | 2025-11-12 10:15:00 | 2843.00 | 2025-11-14 09:15:00 | 2762.90 | EXIT_EMA400 | -80.10 |
| BUY | 2025-12-29 09:15:00 | 2863.80 | 2025-12-29 11:15:00 | 2824.50 | EXIT_EMA400 | -39.30 |
| SELL | 2026-02-03 10:15:00 | 2825.50 | 2026-02-03 11:15:00 | 2829.20 | EXIT_EMA400 | -3.70 |
