# Titan Company Ltd. (TITAN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 4385.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 3 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -13.86
- **Avg P&L per closed trade:** -1.54

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 11:15:00 | 3590.35 | 3647.13 | 3647.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 09:15:00 | 3579.10 | 3641.70 | 3644.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 13:15:00 | 3450.60 | 3445.87 | 3521.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-24 09:15:00 | 3418.70 | 3445.72 | 3519.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-13 12:15:00 | 3463.00 | 3390.82 | 3455.37 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 11:15:00 | 3612.25 | 3398.86 | 3398.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 3631.90 | 3427.19 | 3413.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 12:15:00 | 3682.10 | 3695.82 | 3604.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-04 10:15:00 | 3724.90 | 3695.75 | 3606.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-07 09:15:00 | 3582.00 | 3694.63 | 3609.06 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 13:15:00 | 3340.05 | 3557.76 | 3557.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 3309.80 | 3550.82 | 3554.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 14:15:00 | 3314.00 | 3298.71 | 3387.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-28 10:15:00 | 3250.80 | 3300.32 | 3378.16 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 10:15:00 | 3368.00 | 3295.84 | 3365.36 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 14:15:00 | 3437.55 | 3389.43 | 3389.37 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 3333.65 | 3389.25 | 3389.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 13:15:00 | 3321.50 | 3388.58 | 3388.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 3394.00 | 3374.52 | 3381.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 13:15:00 | 3344.10 | 3374.22 | 3380.69 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-23 10:15:00 | 3392.60 | 3373.98 | 3380.44 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 10:15:00 | 3590.00 | 3384.98 | 3384.65 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 11:15:00 | 3265.10 | 3389.25 | 3389.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 13:15:00 | 3242.85 | 3386.52 | 3387.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 3167.65 | 3153.07 | 3233.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-24 09:15:00 | 3090.75 | 3155.27 | 3229.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 3133.45 | 3109.84 | 3181.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-08 14:15:00 | 3120.60 | 3111.20 | 3180.00 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 12:15:00 | 3175.85 | 3112.89 | 3179.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-09 13:15:00 | 3183.20 | 3113.59 | 3179.17 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 3372.30 | 3222.12 | 3221.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 3379.70 | 3223.69 | 3222.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 3498.70 | 3499.55 | 3425.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-12 12:15:00 | 3512.00 | 3499.67 | 3426.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-13 09:15:00 | 3408.90 | 3497.42 | 3426.35 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 3355.10 | 3465.01 | 3465.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 3328.50 | 3460.33 | 3462.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 3449.00 | 3441.80 | 3452.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-07 12:15:00 | 3398.70 | 3440.07 | 3451.18 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-08 09:15:00 | 3469.40 | 3439.27 | 3450.55 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 3557.80 | 3459.01 | 3458.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 3562.70 | 3461.04 | 3459.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 3576.80 | 3579.79 | 3536.72 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 11:15:00 | 3380.00 | 3511.22 | 3511.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 14:15:00 | 3367.30 | 3499.48 | 3505.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 3571.00 | 3479.27 | 3493.91 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 3641.40 | 3506.56 | 3505.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3694.00 | 3511.06 | 3508.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 13:15:00 | 3810.00 | 3813.15 | 3724.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-09 11:15:00 | 3841.60 | 3811.65 | 3734.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 3999.30 | 4056.38 | 3958.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-29 09:15:00 | 3891.70 | 4048.83 | 3958.82 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-24 09:15:00 | 3418.70 | 2024-06-04 12:15:00 | 3115.38 | TARGET | 303.32 |
| BUY | 2024-10-04 10:15:00 | 3724.90 | 2024-10-07 09:15:00 | 3582.00 | EXIT_EMA400 | -142.90 |
| SELL | 2024-11-28 10:15:00 | 3250.80 | 2024-12-04 10:15:00 | 3368.00 | EXIT_EMA400 | -117.20 |
| SELL | 2025-01-22 13:15:00 | 3344.10 | 2025-01-23 10:15:00 | 3392.60 | EXIT_EMA400 | -48.50 |
| SELL | 2025-03-24 09:15:00 | 3090.75 | 2025-04-09 13:15:00 | 3183.20 | EXIT_EMA400 | -92.45 |
| SELL | 2025-04-08 14:15:00 | 3120.60 | 2025-04-09 13:15:00 | 3183.20 | EXIT_EMA400 | -62.60 |
| BUY | 2025-06-12 12:15:00 | 3512.00 | 2025-06-13 09:15:00 | 3408.90 | EXIT_EMA400 | -103.10 |
| SELL | 2025-08-07 12:15:00 | 3398.70 | 2025-08-08 09:15:00 | 3469.40 | EXIT_EMA400 | -70.70 |
| BUY | 2025-12-09 11:15:00 | 3841.60 | 2026-01-07 09:15:00 | 4161.87 | TARGET | 320.27 |
