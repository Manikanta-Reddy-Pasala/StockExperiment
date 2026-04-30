# Timken India Ltd. (TIMKEN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3425.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / EMA400 exits:** 0 / 9
- **Total realized P&L (per unit):** -869.05
- **Avg P&L per closed trade:** -96.56

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 12:15:00 | 3679.95 | 4035.79 | 4035.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 13:15:00 | 3662.85 | 4032.08 | 4034.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 3920.20 | 3881.11 | 3940.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-05 14:15:00 | 3794.00 | 3877.39 | 3935.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 3876.00 | 3816.24 | 3880.53 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-23 11:15:00 | 3799.60 | 3818.23 | 3879.34 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 3863.00 | 3818.27 | 3876.95 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-24 13:15:00 | 3844.15 | 3818.52 | 3876.78 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-24 14:15:00 | 3932.00 | 3819.65 | 3877.06 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 3126.90 | 2692.85 | 2691.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 3142.70 | 2697.33 | 2693.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 15:15:00 | 3365.00 | 3366.19 | 3235.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-28 09:15:00 | 3442.10 | 3366.95 | 3236.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.83 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 2881.70 | 3173.78 | 3174.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 14:15:00 | 2852.10 | 3170.58 | 3172.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 3092.20 | 3089.02 | 3126.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 2955.70 | 3084.50 | 3122.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 3036.70 | 3002.07 | 3058.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-12 14:15:00 | 3027.20 | 3003.46 | 3057.92 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-16 09:15:00 | 3061.30 | 3005.94 | 3056.79 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 3100.00 | 3029.03 | 3028.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 3153.40 | 3041.41 | 3035.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 3069.50 | 3079.19 | 3059.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-15 11:15:00 | 3099.30 | 3066.86 | 3055.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3077.80 | 3068.92 | 3057.53 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-18 09:15:00 | 3025.60 | 3069.45 | 3058.20 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 2977.30 | 3050.30 | 3050.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 2948.00 | 3048.72 | 3049.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 3050.20 | 3042.74 | 3046.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 11:15:00 | 3006.80 | 3046.45 | 3047.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 3025.00 | 3030.04 | 3038.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-14 10:15:00 | 3049.30 | 3030.27 | 3038.98 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 3252.90 | 3037.02 | 3036.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 3293.50 | 3062.81 | 3050.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 3089.70 | 3098.62 | 3071.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-13 15:15:00 | 3120.00 | 3098.56 | 3072.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 3120.00 | 3098.56 | 3072.06 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-16 09:15:00 | 3057.10 | 3098.15 | 3071.99 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-05 14:15:00 | 3794.00 | 2024-09-24 14:15:00 | 3932.00 | EXIT_EMA400 | -138.00 |
| SELL | 2024-09-23 11:15:00 | 3799.60 | 2024-09-24 14:15:00 | 3932.00 | EXIT_EMA400 | -132.40 |
| SELL | 2024-09-24 13:15:00 | 3844.15 | 2024-09-24 14:15:00 | 3932.00 | EXIT_EMA400 | -87.85 |
| BUY | 2025-07-28 09:15:00 | 3442.10 | 2025-08-01 09:15:00 | 3250.10 | EXIT_EMA400 | -192.00 |
| SELL | 2025-08-26 09:15:00 | 2955.70 | 2025-09-16 09:15:00 | 3061.30 | EXIT_EMA400 | -105.60 |
| SELL | 2025-09-12 14:15:00 | 3027.20 | 2025-09-16 09:15:00 | 3061.30 | EXIT_EMA400 | -34.10 |
| BUY | 2025-12-15 11:15:00 | 3099.30 | 2025-12-18 09:15:00 | 3025.60 | EXIT_EMA400 | -73.70 |
| SELL | 2026-01-08 11:15:00 | 3006.80 | 2026-01-14 10:15:00 | 3049.30 | EXIT_EMA400 | -42.50 |
| BUY | 2026-02-13 15:15:00 | 3120.00 | 2026-02-16 09:15:00 | 3057.10 | EXIT_EMA400 | -62.90 |
