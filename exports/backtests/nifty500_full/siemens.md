# Siemens Ltd. (SIEMENS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 3808.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 460.39
- **Avg P&L per closed trade:** 65.77

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 14:15:00 | 3503.95 | 3744.72 | 3745.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 13:15:00 | 3498.00 | 3657.95 | 3694.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 11:15:00 | 3514.35 | 3470.81 | 3556.36 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 13:15:00 | 3854.05 | 3594.17 | 3593.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 12:15:00 | 3873.10 | 3637.78 | 3616.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 6585.00 | 6618.88 | 6043.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-13 09:15:00 | 7211.10 | 6692.50 | 6204.55 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-19 09:15:00 | 6947.00 | 7462.78 | 7025.33 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 6614.35 | 6939.86 | 6940.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 6583.95 | 6936.32 | 6938.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 6825.10 | 6816.44 | 6867.22 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 10:15:00 | 7220.65 | 6906.46 | 6905.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 11:15:00 | 7274.50 | 6910.12 | 6907.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 14:15:00 | 6995.90 | 7015.64 | 6965.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 09:15:00 | 7104.95 | 7016.37 | 6966.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 7104.95 | 7016.37 | 6966.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-08 10:15:00 | 7194.80 | 7018.15 | 6967.55 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 7226.95 | 7356.09 | 7185.99 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-23 10:15:00 | 7162.55 | 7354.17 | 7185.87 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 15:15:00 | 6704.55 | 7080.72 | 7080.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 10:15:00 | 6674.60 | 7073.13 | 7076.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 7185.00 | 6978.80 | 7024.74 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 13:15:00 | 7440.65 | 7067.45 | 7066.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 7490.00 | 7078.47 | 7071.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 11:15:00 | 7461.35 | 7538.61 | 7364.24 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 13:15:00 | 6570.95 | 7221.31 | 7224.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 14:15:00 | 6538.90 | 7214.52 | 7220.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 12:15:00 | 5154.00 | 5143.20 | 5551.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-20 09:15:00 | 5117.00 | 5144.89 | 5544.52 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-26 12:15:00 | 5504.35 | 5170.19 | 5500.52 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 3295.30 | 3167.42 | 3167.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 14:15:00 | 3307.50 | 3180.65 | 3174.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 13:15:00 | 3196.30 | 3221.96 | 3197.79 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 09:15:00 | 3117.60 | 3181.54 | 3181.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 3095.20 | 3172.33 | 3176.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 3133.40 | 3131.05 | 3151.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 3097.80 | 3130.76 | 3151.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 3055.40 | 3014.90 | 3073.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-29 13:15:00 | 3075.40 | 3016.06 | 3073.92 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 3203.80 | 3109.71 | 3109.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 3226.20 | 3115.90 | 3112.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 12:15:00 | 3193.30 | 3196.27 | 3158.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 11:15:00 | 3221.50 | 3195.61 | 3159.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 3195.60 | 3202.16 | 3164.86 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-09 14:15:00 | 3230.90 | 3201.96 | 3165.67 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-16 09:15:00 | 3153.60 | 3221.07 | 3181.10 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 2948.30 | 3153.76 | 3154.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 2938.00 | 3149.74 | 3152.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3173.10 | 3117.31 | 3134.53 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 11:15:00 | 3360.20 | 3149.48 | 3149.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 3504.00 | 3161.00 | 3155.25 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-06-13 09:15:00 | 7211.10 | 2024-07-19 09:15:00 | 6947.00 | EXIT_EMA400 | -264.10 |
| BUY | 2024-10-08 09:15:00 | 7104.95 | 2024-10-09 09:15:00 | 7520.58 | TARGET | 415.63 |
| BUY | 2024-10-08 10:15:00 | 7194.80 | 2024-10-15 14:15:00 | 7876.56 | TARGET | 681.76 |
| SELL | 2025-03-20 09:15:00 | 5117.00 | 2025-03-26 12:15:00 | 5504.35 | EXIT_EMA400 | -387.35 |
| SELL | 2026-01-08 10:15:00 | 3097.80 | 2026-01-12 11:15:00 | 2938.15 | TARGET | 159.65 |
| BUY | 2026-03-05 11:15:00 | 3221.50 | 2026-03-16 09:15:00 | 3153.60 | EXIT_EMA400 | -67.90 |
| BUY | 2026-03-09 14:15:00 | 3230.90 | 2026-03-16 09:15:00 | 3153.60 | EXIT_EMA400 | -77.30 |
