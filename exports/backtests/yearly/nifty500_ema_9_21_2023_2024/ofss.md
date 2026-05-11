# Oracle Financial Services Software Ltd. (OFSS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 9321.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 213 |
| ALERT1 | 151 |
| ALERT2 | 148 |
| ALERT2_SKIP | 102 |
| ALERT3 | 302 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 129 |
| PARTIAL | 16 |
| TARGET_HIT | 3 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 147 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 97
- **Target hits / Stop hits / Partials:** 3 / 128 / 16
- **Avg / median % per leg:** 0.25% / -0.82%
- **Sum % (uncompounded):** 37.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 10 | 16.9% | 1 | 58 | 0 | -0.73% | -43.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.92% | -0.9% |
| BUY @ 3rd Alert (retest2) | 58 | 10 | 17.2% | 1 | 57 | 0 | -0.73% | -42.4% |
| SELL (all) | 88 | 40 | 45.5% | 2 | 70 | 16 | 0.91% | 80.4% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.98% | 2.0% |
| SELL @ 3rd Alert (retest2) | 87 | 39 | 44.8% | 2 | 69 | 16 | 0.90% | 78.4% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.53% | 1.1% |
| retest2 (combined) | 145 | 49 | 33.8% | 3 | 126 | 16 | 0.25% | 36.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 10:15:00 | 3515.85 | 3537.29 | 3537.39 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 11:15:00 | 3548.35 | 3537.22 | 3536.18 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 10:15:00 | 3504.20 | 3536.97 | 3537.73 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 12:15:00 | 3549.00 | 3538.27 | 3538.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 13:15:00 | 3551.00 | 3540.82 | 3539.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 10:15:00 | 3550.00 | 3551.09 | 3545.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 10:15:00 | 3550.00 | 3551.09 | 3545.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 3550.00 | 3551.09 | 3545.49 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 3502.50 | 3537.35 | 3540.93 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 13:15:00 | 3552.95 | 3542.46 | 3541.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 09:15:00 | 3557.55 | 3547.50 | 3544.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 11:15:00 | 3564.85 | 3571.93 | 3561.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 11:15:00 | 3564.85 | 3571.93 | 3561.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 3564.85 | 3571.93 | 3561.27 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-05-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 10:15:00 | 3546.25 | 3569.97 | 3570.17 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 12:15:00 | 3578.15 | 3570.17 | 3570.14 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-05-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 13:15:00 | 3565.60 | 3569.26 | 3569.73 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 3586.20 | 3572.39 | 3570.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 11:15:00 | 3617.75 | 3582.74 | 3576.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 09:15:00 | 3591.00 | 3592.58 | 3584.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 09:15:00 | 3591.00 | 3592.58 | 3584.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 3591.00 | 3592.58 | 3584.26 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 14:15:00 | 3575.00 | 3581.05 | 3581.05 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 09:15:00 | 3591.70 | 3581.57 | 3581.19 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 11:15:00 | 3577.50 | 3580.56 | 3580.78 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 14:15:00 | 3589.75 | 3582.00 | 3581.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 09:15:00 | 3619.95 | 3589.60 | 3584.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 14:15:00 | 3618.00 | 3629.02 | 3615.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 15:15:00 | 3621.00 | 3627.42 | 3615.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 3621.00 | 3627.42 | 3615.70 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 09:15:00 | 3583.45 | 3618.35 | 3620.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 3566.20 | 3607.92 | 3615.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 09:15:00 | 3607.00 | 3579.35 | 3594.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 09:15:00 | 3607.00 | 3579.35 | 3594.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 3607.00 | 3579.35 | 3594.06 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 11:15:00 | 3593.00 | 3564.07 | 3562.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 3636.25 | 3589.99 | 3576.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 14:15:00 | 3857.75 | 3866.80 | 3806.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 15:15:00 | 3870.05 | 3882.83 | 3849.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 3870.05 | 3882.83 | 3849.60 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 3849.00 | 3877.83 | 3881.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 3764.70 | 3847.31 | 3865.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 3824.95 | 3822.14 | 3841.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 3824.95 | 3822.14 | 3841.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 3824.95 | 3822.14 | 3841.11 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 15:15:00 | 3848.50 | 3824.44 | 3822.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 09:15:00 | 3883.75 | 3836.31 | 3827.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 09:15:00 | 3819.00 | 3848.13 | 3840.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 3819.00 | 3848.13 | 3840.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 3819.00 | 3848.13 | 3840.70 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 11:15:00 | 3808.70 | 3833.67 | 3835.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 09:15:00 | 3783.25 | 3816.21 | 3825.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 13:15:00 | 3830.35 | 3808.78 | 3817.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 13:15:00 | 3830.35 | 3808.78 | 3817.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 3830.35 | 3808.78 | 3817.98 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 09:15:00 | 3867.50 | 3826.26 | 3824.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 09:15:00 | 3891.35 | 3869.14 | 3856.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 3844.00 | 3864.11 | 3855.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 10:15:00 | 3844.00 | 3864.11 | 3855.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 3844.00 | 3864.11 | 3855.64 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 3824.75 | 3850.08 | 3850.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 14:15:00 | 3808.00 | 3838.08 | 3844.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 3828.95 | 3790.46 | 3808.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 3828.95 | 3790.46 | 3808.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 3828.95 | 3790.46 | 3808.29 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 14:15:00 | 3840.95 | 3818.82 | 3817.25 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 09:15:00 | 3778.95 | 3813.27 | 3815.15 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 13:15:00 | 3829.65 | 3815.06 | 3815.01 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 14:15:00 | 3806.95 | 3813.44 | 3814.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 15:15:00 | 3797.85 | 3810.32 | 3812.78 | Break + close below crossover candle low |

### Cycle 26 — BUY (started 2023-07-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 09:15:00 | 3867.00 | 3821.66 | 3817.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 10:15:00 | 3875.95 | 3832.51 | 3823.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 3839.70 | 3846.61 | 3833.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 14:15:00 | 3844.90 | 3846.27 | 3834.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 3844.90 | 3846.27 | 3834.08 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 14:15:00 | 3915.35 | 3930.03 | 3931.90 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 13:15:00 | 3963.00 | 3934.84 | 3932.80 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-07-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 12:15:00 | 3929.15 | 3934.21 | 3934.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 09:15:00 | 3900.00 | 3925.07 | 3929.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 12:15:00 | 3866.60 | 3856.41 | 3881.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 12:15:00 | 3866.60 | 3856.41 | 3881.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 3866.60 | 3856.41 | 3881.58 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 3859.75 | 3826.04 | 3824.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 10:15:00 | 3876.40 | 3836.11 | 3828.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 12:15:00 | 3874.50 | 3876.21 | 3859.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 15:15:00 | 3870.05 | 3873.02 | 3862.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 15:15:00 | 3870.05 | 3873.02 | 3862.14 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 15:15:00 | 3836.45 | 3864.89 | 3864.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 11:15:00 | 3821.95 | 3850.14 | 3857.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 3935.00 | 3849.08 | 3851.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 3935.00 | 3849.08 | 3851.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 3935.00 | 3849.08 | 3851.60 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 3924.95 | 3864.25 | 3858.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 14:15:00 | 3960.00 | 3922.61 | 3898.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 09:15:00 | 3965.80 | 3968.02 | 3951.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 09:15:00 | 3965.80 | 3968.02 | 3951.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 3965.80 | 3968.02 | 3951.34 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 11:15:00 | 3947.50 | 3958.97 | 3959.96 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 13:15:00 | 3977.00 | 3961.96 | 3961.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-14 14:15:00 | 3984.15 | 3966.40 | 3963.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 09:15:00 | 4054.00 | 4065.70 | 4028.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 10:15:00 | 4046.80 | 4061.92 | 4030.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 4046.80 | 4061.92 | 4030.22 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 12:15:00 | 3986.25 | 4029.80 | 4031.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-22 11:15:00 | 3980.00 | 3996.63 | 4008.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 09:15:00 | 4000.00 | 3969.84 | 3979.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 09:15:00 | 4000.00 | 3969.84 | 3979.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 4000.00 | 3969.84 | 3979.71 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-08-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 13:15:00 | 4010.10 | 3985.31 | 3984.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 15:15:00 | 4016.70 | 3995.55 | 3989.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 10:15:00 | 3977.00 | 3992.71 | 3989.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 10:15:00 | 3977.00 | 3992.71 | 3989.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 3977.00 | 3992.71 | 3989.49 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 12:15:00 | 3973.05 | 3984.69 | 3986.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 15:15:00 | 3959.25 | 3975.18 | 3981.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 10:15:00 | 3971.00 | 3968.30 | 3976.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 11:15:00 | 3997.10 | 3974.06 | 3978.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 3997.10 | 3974.06 | 3978.40 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 13:15:00 | 4000.50 | 3983.50 | 3982.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 10:15:00 | 4020.70 | 4005.67 | 3997.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 09:15:00 | 4122.55 | 4127.46 | 4101.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 10:15:00 | 4109.80 | 4123.93 | 4102.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 4109.80 | 4123.93 | 4102.64 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 4286.75 | 4307.32 | 4308.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 15:15:00 | 4266.40 | 4299.13 | 4305.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 4262.00 | 4231.90 | 4255.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 4262.00 | 4231.90 | 4255.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 4262.00 | 4231.90 | 4255.63 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 4346.00 | 4278.09 | 4270.00 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 14:15:00 | 4248.00 | 4277.76 | 4280.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 4195.00 | 4258.37 | 4270.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 10:15:00 | 4242.50 | 4225.33 | 4241.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 10:15:00 | 4242.50 | 4225.33 | 4241.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 4242.50 | 4225.33 | 4241.33 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 14:15:00 | 4097.55 | 4058.09 | 4057.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 15:15:00 | 4105.00 | 4067.47 | 4061.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 12:15:00 | 4091.50 | 4092.86 | 4077.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 14:15:00 | 4097.00 | 4092.47 | 4080.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 14:15:00 | 4097.00 | 4092.47 | 4080.05 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 4101.30 | 4142.91 | 4145.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 10:15:00 | 4085.35 | 4116.84 | 4131.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 4133.55 | 4107.17 | 4117.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 4133.55 | 4107.17 | 4117.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 4133.55 | 4107.17 | 4117.90 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 11:15:00 | 4161.00 | 4127.06 | 4123.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 12:15:00 | 4177.00 | 4137.05 | 4127.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 11:15:00 | 4132.05 | 4151.92 | 4141.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 11:15:00 | 4132.05 | 4151.92 | 4141.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 4132.05 | 4151.92 | 4141.68 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 4105.95 | 4137.27 | 4137.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 4040.00 | 4117.81 | 4128.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-23 10:15:00 | 3972.40 | 3957.10 | 4002.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 3963.70 | 3949.53 | 3978.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 3963.70 | 3949.53 | 3978.16 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 12:15:00 | 3906.70 | 3903.29 | 3903.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 3942.00 | 3918.77 | 3912.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 14:15:00 | 3932.50 | 3936.00 | 3925.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 10:15:00 | 3960.70 | 3971.74 | 3956.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 3960.70 | 3971.74 | 3956.29 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 10:15:00 | 3996.00 | 4028.97 | 4032.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 15:15:00 | 3991.00 | 4008.77 | 4020.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 4035.85 | 4014.19 | 4021.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 4035.85 | 4014.19 | 4021.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 4035.85 | 4014.19 | 4021.55 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-11-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 13:15:00 | 4032.05 | 4026.27 | 4025.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 14:15:00 | 4056.80 | 4032.38 | 4028.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 10:15:00 | 4040.15 | 4041.46 | 4034.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 10:15:00 | 4040.15 | 4041.46 | 4034.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 4040.15 | 4041.46 | 4034.31 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 10:15:00 | 4125.00 | 4162.96 | 4165.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 11:15:00 | 4079.70 | 4146.31 | 4157.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 15:15:00 | 4095.00 | 4094.83 | 4111.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 4063.55 | 4088.57 | 4107.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 4063.55 | 4088.57 | 4107.57 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 10:15:00 | 4087.10 | 4045.11 | 4041.10 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 13:15:00 | 4016.80 | 4039.34 | 4039.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-30 14:15:00 | 3993.25 | 4030.12 | 4035.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 15:15:00 | 4048.00 | 4033.70 | 4036.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 15:15:00 | 4048.00 | 4033.70 | 4036.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 15:15:00 | 4048.00 | 4033.70 | 4036.47 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 15:15:00 | 4039.00 | 4034.92 | 4034.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 4055.00 | 4038.94 | 4036.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 09:15:00 | 4012.90 | 4042.55 | 4041.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 4012.90 | 4042.55 | 4041.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 4012.90 | 4042.55 | 4041.55 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2023-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 10:15:00 | 4008.65 | 4035.77 | 4038.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 11:15:00 | 3982.30 | 4025.08 | 4033.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 14:15:00 | 4028.65 | 4017.06 | 4026.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 14:15:00 | 4028.65 | 4017.06 | 4026.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 14:15:00 | 4028.65 | 4017.06 | 4026.85 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 09:15:00 | 4082.00 | 4033.08 | 4032.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 11:15:00 | 4116.05 | 4057.52 | 4044.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 10:15:00 | 4133.50 | 4134.25 | 4108.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 12:15:00 | 4120.55 | 4131.62 | 4111.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 4120.55 | 4131.62 | 4111.48 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 4118.00 | 4154.81 | 4156.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 09:15:00 | 4071.10 | 4138.06 | 4148.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 15:15:00 | 4095.00 | 4091.81 | 4115.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 4160.50 | 4105.55 | 4119.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 4160.50 | 4105.55 | 4119.74 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 4204.50 | 4138.31 | 4132.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 4249.90 | 4181.49 | 4158.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 15:15:00 | 4363.65 | 4365.85 | 4335.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 11:15:00 | 4331.05 | 4360.12 | 4340.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 4331.05 | 4360.12 | 4340.31 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 4186.70 | 4308.62 | 4320.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 4179.90 | 4282.87 | 4307.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 14:15:00 | 4220.00 | 4200.23 | 4227.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 14:15:00 | 4220.00 | 4200.23 | 4227.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 4220.00 | 4200.23 | 4227.71 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2023-12-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 15:15:00 | 4209.00 | 4195.15 | 4194.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 09:15:00 | 4230.95 | 4202.31 | 4197.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 13:15:00 | 4200.00 | 4204.83 | 4200.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 13:15:00 | 4200.00 | 4204.83 | 4200.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 13:15:00 | 4200.00 | 4204.83 | 4200.59 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-01-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 14:15:00 | 4329.40 | 4350.12 | 4351.20 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 4519.90 | 4382.79 | 4365.79 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 4352.70 | 4400.65 | 4401.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 15:15:00 | 4334.00 | 4371.11 | 4385.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 14:15:00 | 4336.00 | 4334.43 | 4358.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 4371.10 | 4342.18 | 4357.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 4371.10 | 4342.18 | 4357.53 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 4540.00 | 4384.11 | 4370.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 13:15:00 | 4601.35 | 4492.44 | 4431.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 11:15:00 | 4912.95 | 4920.89 | 4775.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 5011.95 | 4934.48 | 4835.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 5011.95 | 4934.48 | 4835.38 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 10:15:00 | 6562.05 | 6615.52 | 6618.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 12:15:00 | 6558.50 | 6598.44 | 6609.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 12:15:00 | 6521.60 | 6503.28 | 6544.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 12:15:00 | 6521.60 | 6503.28 | 6544.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 6521.60 | 6503.28 | 6544.01 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 6587.00 | 6504.76 | 6504.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 09:15:00 | 6711.20 | 6574.88 | 6540.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 14:15:00 | 6624.50 | 6650.82 | 6597.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 11:15:00 | 7180.15 | 7188.05 | 7084.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 7180.15 | 7188.05 | 7084.90 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 13:15:00 | 7732.45 | 7794.45 | 7799.12 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 09:15:00 | 7832.05 | 7804.43 | 7802.64 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 7759.35 | 7803.56 | 7804.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 7687.45 | 7771.45 | 7789.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 10:15:00 | 7785.90 | 7774.34 | 7788.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 10:15:00 | 7785.90 | 7774.34 | 7788.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 7785.90 | 7774.34 | 7788.84 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-02-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 14:15:00 | 7910.65 | 7814.03 | 7803.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 09:15:00 | 8066.80 | 7878.46 | 7835.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 13:15:00 | 7923.00 | 7924.30 | 7875.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 15:15:00 | 7893.00 | 7911.18 | 7877.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 7893.00 | 7911.18 | 7877.32 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 11:15:00 | 7788.35 | 7858.84 | 7859.86 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 14:15:00 | 7890.50 | 7864.96 | 7862.12 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 15:15:00 | 7860.00 | 7867.65 | 7867.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 09:15:00 | 7823.00 | 7858.72 | 7863.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 7700.35 | 7669.08 | 7724.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 7713.40 | 7682.89 | 7721.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 7713.40 | 7682.89 | 7721.28 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 12:15:00 | 7730.10 | 7719.76 | 7718.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 11:15:00 | 7815.00 | 7744.90 | 7730.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 13:15:00 | 7738.75 | 7748.01 | 7734.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 13:15:00 | 7738.75 | 7748.01 | 7734.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 13:15:00 | 7738.75 | 7748.01 | 7734.88 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 7645.15 | 7720.99 | 7726.91 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 12:15:00 | 7803.30 | 7726.63 | 7722.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-06 13:15:00 | 7920.70 | 7765.44 | 7740.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 10:15:00 | 8114.00 | 8177.38 | 8048.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 11:15:00 | 8121.10 | 8227.41 | 8186.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 8121.10 | 8227.41 | 8186.92 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 14:15:00 | 8023.60 | 8137.76 | 8152.24 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 11:15:00 | 8296.00 | 8173.06 | 8162.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 12:15:00 | 8381.00 | 8214.65 | 8182.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 09:15:00 | 8273.45 | 8315.05 | 8248.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 09:15:00 | 8273.45 | 8315.05 | 8248.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 8273.45 | 8315.05 | 8248.28 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 15:15:00 | 8290.00 | 8386.13 | 8387.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 8204.45 | 8349.80 | 8370.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 8342.55 | 8281.47 | 8314.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 8342.55 | 8281.47 | 8314.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 8342.55 | 8281.47 | 8314.24 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 14:15:00 | 8422.00 | 8326.04 | 8324.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 09:15:00 | 8569.75 | 8408.81 | 8372.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 09:15:00 | 8758.00 | 8773.39 | 8670.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 09:15:00 | 8796.65 | 8900.37 | 8838.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 8796.65 | 8900.37 | 8838.08 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 11:15:00 | 8730.00 | 8820.53 | 8832.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 09:15:00 | 8664.35 | 8765.85 | 8791.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 7407.70 | 7405.55 | 7502.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 7251.65 | 7364.35 | 7435.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 7251.65 | 7364.35 | 7435.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:30:00 | 7487.25 | 7364.35 | 7435.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 7354.15 | 7256.73 | 7314.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 12:00:00 | 7354.15 | 7256.73 | 7314.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 12:15:00 | 7459.95 | 7297.37 | 7327.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 12:45:00 | 7465.90 | 7297.37 | 7327.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 14:15:00 | 7477.45 | 7365.81 | 7355.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 09:15:00 | 7566.95 | 7468.90 | 7425.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 09:15:00 | 7699.40 | 7811.45 | 7766.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 09:15:00 | 7699.40 | 7811.45 | 7766.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 7699.40 | 7811.45 | 7766.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:15:00 | 7665.00 | 7811.45 | 7766.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 7642.65 | 7777.69 | 7755.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:45:00 | 7697.50 | 7777.69 | 7755.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 7565.80 | 7711.49 | 7727.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 7560.70 | 7612.50 | 7649.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 14:15:00 | 7559.35 | 7558.73 | 7598.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 14:15:00 | 7559.35 | 7558.73 | 7598.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 7559.35 | 7558.73 | 7598.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 7559.35 | 7558.73 | 7598.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 7595.00 | 7565.98 | 7598.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 7600.00 | 7565.98 | 7598.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 7566.05 | 7566.00 | 7595.38 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 7695.00 | 7614.29 | 7611.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 7723.60 | 7636.15 | 7621.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 13:15:00 | 7814.55 | 7815.79 | 7762.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 14:00:00 | 7814.55 | 7815.79 | 7762.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 7769.95 | 7800.09 | 7764.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 7923.85 | 7800.09 | 7764.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 15:15:00 | 7777.00 | 7818.32 | 7819.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 15:15:00 | 7777.00 | 7818.32 | 7819.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 09:15:00 | 7695.90 | 7786.92 | 7804.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 7668.20 | 7667.51 | 7697.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 11:45:00 | 7676.00 | 7667.51 | 7697.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 7601.20 | 7642.10 | 7673.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:15:00 | 7572.05 | 7642.10 | 7673.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 11:30:00 | 7567.50 | 7617.67 | 7656.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 12:45:00 | 7558.35 | 7606.35 | 7647.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 15:15:00 | 7555.00 | 7584.71 | 7603.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 7555.00 | 7578.77 | 7598.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:15:00 | 7610.75 | 7578.77 | 7598.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 7563.20 | 7575.66 | 7595.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:30:00 | 7533.40 | 7552.25 | 7564.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 10:00:00 | 7530.05 | 7552.25 | 7564.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:00:00 | 7534.20 | 7548.64 | 7561.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:15:00 | 7535.00 | 7481.52 | 7489.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 12:15:00 | 7550.00 | 7495.22 | 7494.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 7550.00 | 7495.22 | 7494.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 7585.05 | 7513.19 | 7502.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 7479.95 | 7521.49 | 7510.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 7479.95 | 7521.49 | 7510.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 7479.95 | 7521.49 | 7510.56 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 7344.45 | 7486.08 | 7495.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 7200.00 | 7428.86 | 7468.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 7580.05 | 7399.73 | 7430.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 7580.05 | 7399.73 | 7430.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 7580.05 | 7399.73 | 7430.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 7580.05 | 7399.73 | 7430.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 7764.90 | 7472.76 | 7461.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 7872.60 | 7714.88 | 7612.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 8172.45 | 8299.37 | 8112.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 8172.45 | 8299.37 | 8112.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 9807.30 | 9771.91 | 9713.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 9790.00 | 9771.91 | 9713.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 9733.95 | 9774.46 | 9734.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:30:00 | 9763.75 | 9774.46 | 9734.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 9720.90 | 9763.74 | 9733.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 9667.95 | 9763.74 | 9733.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 9705.20 | 9752.04 | 9730.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 9635.00 | 9752.04 | 9730.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 9783.50 | 9758.33 | 9735.55 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 9722.00 | 9741.53 | 9743.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 09:15:00 | 9582.40 | 9709.71 | 9728.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 10:15:00 | 9568.00 | 9552.03 | 9616.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 10:30:00 | 9563.40 | 9552.03 | 9616.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 9600.00 | 9561.63 | 9614.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 9600.00 | 9561.63 | 9614.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 9594.90 | 9568.28 | 9613.07 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 9850.00 | 9674.20 | 9653.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 10209.25 | 9894.42 | 9787.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 10121.15 | 10176.61 | 10057.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 13:00:00 | 10121.15 | 10176.61 | 10057.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 10423.40 | 10424.21 | 10375.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:45:00 | 10384.95 | 10424.21 | 10375.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 10404.90 | 10421.92 | 10382.84 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 15:15:00 | 10336.10 | 10369.22 | 10369.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 09:15:00 | 10279.95 | 10351.36 | 10361.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 13:15:00 | 10396.05 | 10343.49 | 10352.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 10396.05 | 10343.49 | 10352.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 10396.05 | 10343.49 | 10352.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 10396.05 | 10343.49 | 10352.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 10374.20 | 10349.63 | 10354.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 10345.60 | 10351.70 | 10354.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 10590.20 | 10302.39 | 10272.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 10590.20 | 10302.39 | 10272.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 13:15:00 | 10611.00 | 10397.27 | 10323.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 10875.05 | 10991.33 | 10859.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 10875.05 | 10991.33 | 10859.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 10875.05 | 10991.33 | 10859.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:45:00 | 10917.50 | 10991.33 | 10859.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 10881.80 | 10950.81 | 10862.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 12:30:00 | 10956.05 | 10951.59 | 10871.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 10:15:00 | 10931.05 | 10935.35 | 10889.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 12:00:00 | 10987.95 | 10939.49 | 10898.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 10758.40 | 10897.68 | 10886.48 | SL hit (close<static) qty=1.00 sl=10827.35 alert=retest2 |

### Cycle 91 — SELL (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 13:15:00 | 10860.00 | 10881.35 | 10883.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 14:15:00 | 10791.35 | 10863.35 | 10875.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 11:15:00 | 10837.60 | 10832.16 | 10854.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 11:15:00 | 10837.60 | 10832.16 | 10854.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 10837.60 | 10832.16 | 10854.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:00:00 | 10837.60 | 10832.16 | 10854.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 10859.15 | 10837.56 | 10854.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:00:00 | 10859.15 | 10837.56 | 10854.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 10974.95 | 10865.04 | 10865.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 10974.95 | 10865.04 | 10865.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 10980.00 | 10888.03 | 10875.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 15:15:00 | 11045.00 | 10919.42 | 10891.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 14:15:00 | 11118.00 | 11121.25 | 11028.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 14:15:00 | 11118.00 | 11121.25 | 11028.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 11118.00 | 11121.25 | 11028.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 14:30:00 | 11060.00 | 11121.25 | 11028.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 11061.00 | 11113.00 | 11040.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 14:45:00 | 11379.95 | 11240.46 | 11178.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 14:15:00 | 11104.85 | 11166.28 | 11166.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 14:15:00 | 11104.85 | 11166.28 | 11166.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 15:15:00 | 11083.00 | 11149.63 | 11159.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 09:15:00 | 11185.00 | 11156.70 | 11161.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 11185.00 | 11156.70 | 11161.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 11185.00 | 11156.70 | 11161.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:00:00 | 11185.00 | 11156.70 | 11161.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 10993.20 | 11124.00 | 11146.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 12:45:00 | 10885.00 | 11055.06 | 11091.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 15:15:00 | 10340.75 | 10594.36 | 10789.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-05 10:15:00 | 9796.50 | 10313.61 | 10622.09 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 94 — BUY (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 12:15:00 | 10227.65 | 10139.13 | 10130.42 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 10058.35 | 10123.11 | 10124.66 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 10484.90 | 10188.57 | 10153.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 14:15:00 | 10565.65 | 10407.79 | 10288.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 13:15:00 | 10747.35 | 10760.87 | 10618.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:00:00 | 10747.35 | 10760.87 | 10618.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 10755.15 | 10750.63 | 10648.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 11:00:00 | 10834.90 | 10767.48 | 10665.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:00:00 | 10855.00 | 10784.99 | 10682.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 14:15:00 | 10615.40 | 10750.76 | 10692.66 | SL hit (close<static) qty=1.00 sl=10632.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 10975.05 | 11067.75 | 11070.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 11:15:00 | 10934.05 | 11041.01 | 11057.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 11148.45 | 11016.83 | 11033.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 11148.45 | 11016.83 | 11033.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 11148.45 | 11016.83 | 11033.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 11142.40 | 11016.83 | 11033.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 11074.00 | 11028.27 | 11037.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 12:45:00 | 11033.30 | 11032.48 | 11038.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 11061.75 | 10998.80 | 10998.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 11061.75 | 10998.80 | 10998.54 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 10948.55 | 11001.72 | 11003.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 10845.10 | 10970.40 | 10988.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 11000.00 | 10897.54 | 10936.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 15:15:00 | 11000.00 | 10897.54 | 10936.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 11000.00 | 10897.54 | 10936.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 10951.20 | 10897.54 | 10936.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 10972.00 | 10912.43 | 10939.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:15:00 | 10978.00 | 10912.43 | 10939.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 11076.00 | 10945.14 | 10952.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 11073.75 | 10945.14 | 10952.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 11104.55 | 10977.02 | 10966.08 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 10800.00 | 10968.34 | 10970.56 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 11105.00 | 10963.25 | 10954.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 11:15:00 | 11184.85 | 11029.45 | 10987.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 11:15:00 | 11267.50 | 11268.60 | 11159.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 12:00:00 | 11267.50 | 11268.60 | 11159.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 11178.65 | 11250.61 | 11161.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:30:00 | 11176.00 | 11250.61 | 11161.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 11156.65 | 11231.82 | 11161.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:00:00 | 11156.65 | 11231.82 | 11161.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 11243.00 | 11234.06 | 11168.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:30:00 | 11153.90 | 11234.06 | 11168.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 11240.00 | 11235.72 | 11180.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:45:00 | 11194.05 | 11235.72 | 11180.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 11077.35 | 11219.24 | 11202.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 11077.35 | 11219.24 | 11202.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 11069.95 | 11189.38 | 11190.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 11007.45 | 11144.61 | 11169.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 10950.25 | 10934.05 | 11005.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 11240.00 | 10934.05 | 11005.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 11045.55 | 10956.35 | 11009.37 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 11288.35 | 11074.86 | 11054.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 13:15:00 | 11361.60 | 11132.21 | 11082.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 14:15:00 | 12290.00 | 12327.70 | 12148.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 14:45:00 | 12266.35 | 12327.70 | 12148.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 11757.20 | 12203.57 | 12122.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 11757.20 | 12203.57 | 12122.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 11678.60 | 12098.57 | 12082.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 11678.60 | 12098.57 | 12082.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 11581.35 | 11995.13 | 12036.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 11363.15 | 11868.73 | 11975.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 11171.05 | 11088.57 | 11402.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 11171.05 | 11088.57 | 11402.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 11342.90 | 11109.27 | 11355.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 11368.00 | 11109.27 | 11355.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 11355.00 | 11158.41 | 11355.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 11227.25 | 11242.70 | 11350.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 11385.70 | 11271.30 | 11353.54 | SL hit (close>static) qty=1.00 sl=11373.65 alert=retest2 |

### Cycle 106 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 11521.05 | 11405.48 | 11391.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 10:15:00 | 11544.00 | 11450.96 | 11418.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 11347.40 | 11524.97 | 11483.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 11347.40 | 11524.97 | 11483.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 11347.40 | 11524.97 | 11483.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 11347.40 | 11524.97 | 11483.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 11309.80 | 11481.94 | 11467.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 11318.85 | 11481.94 | 11467.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 11160.00 | 11417.55 | 11439.50 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 11:15:00 | 11426.75 | 11346.57 | 11344.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 14:15:00 | 11479.80 | 11394.83 | 11377.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 10:15:00 | 11337.50 | 11392.46 | 11381.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 10:15:00 | 11337.50 | 11392.46 | 11381.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 11337.50 | 11392.46 | 11381.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 11368.00 | 11392.46 | 11381.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 11385.00 | 11390.97 | 11382.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 13:15:00 | 11438.95 | 11392.67 | 11383.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 09:30:00 | 11472.65 | 11408.93 | 11396.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 10:15:00 | 11251.00 | 11377.35 | 11383.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 11251.00 | 11377.35 | 11383.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 11192.40 | 11340.36 | 11365.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 11264.20 | 11235.80 | 11293.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 11264.20 | 11235.80 | 11293.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 11264.20 | 11235.80 | 11293.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 11264.20 | 11235.80 | 11293.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 11377.75 | 11264.19 | 11301.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 11377.75 | 11264.19 | 11301.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 11188.95 | 11249.14 | 11291.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:30:00 | 11118.25 | 11225.97 | 11276.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:00:00 | 11133.30 | 11225.97 | 11276.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 14:30:00 | 11121.35 | 11054.40 | 11072.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 15:15:00 | 11249.00 | 11093.32 | 11088.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 11249.00 | 11093.32 | 11088.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 11435.00 | 11161.66 | 11119.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 11612.85 | 11616.81 | 11443.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:45:00 | 11638.65 | 11616.81 | 11443.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 11439.40 | 11581.33 | 11442.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 11439.40 | 11581.33 | 11442.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 11435.00 | 11552.07 | 11442.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:30:00 | 11417.90 | 11552.07 | 11442.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 11448.05 | 11531.26 | 11442.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 11448.05 | 11531.26 | 11442.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 11513.40 | 11527.69 | 11449.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 11615.90 | 11527.69 | 11449.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 11490.55 | 11520.26 | 11452.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:45:00 | 11461.10 | 11520.26 | 11452.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 11585.90 | 11533.39 | 11464.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:30:00 | 11632.10 | 11544.64 | 11500.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 12:30:00 | 11658.15 | 11574.95 | 11526.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 13:30:00 | 11628.40 | 11604.96 | 11544.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 10:30:00 | 11627.60 | 11725.69 | 11681.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 11667.45 | 11714.04 | 11680.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 13:15:00 | 11708.80 | 11700.80 | 11677.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 14:15:00 | 11567.80 | 11671.14 | 11667.67 | SL hit (close<static) qty=1.00 sl=11606.25 alert=retest2 |

### Cycle 111 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 11589.80 | 11654.87 | 11660.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 11507.45 | 11589.53 | 11624.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 11577.55 | 11520.15 | 11565.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 11577.55 | 11520.15 | 11565.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 11577.55 | 11520.15 | 11565.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 11577.55 | 11520.15 | 11565.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 11529.60 | 11522.04 | 11562.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 11552.60 | 11522.04 | 11562.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 11555.40 | 11528.71 | 11561.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 11555.40 | 11528.71 | 11561.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 11586.80 | 11540.33 | 11563.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 11579.15 | 11540.33 | 11563.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 11584.95 | 11549.25 | 11565.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 11535.45 | 11549.25 | 11565.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 11502.45 | 11543.97 | 11560.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 11427.70 | 11520.72 | 11548.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 10856.32 | 11139.76 | 11299.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 11170.45 | 11106.05 | 11254.22 | SL hit (close>ema200) qty=0.50 sl=11106.05 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 11294.90 | 10993.73 | 10980.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 11373.60 | 11069.71 | 11016.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 10780.00 | 11132.48 | 11093.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 10780.00 | 11132.48 | 11093.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 10780.00 | 11132.48 | 11093.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:00:00 | 10780.00 | 11132.48 | 11093.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 10683.50 | 11042.69 | 11056.01 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 11312.40 | 10964.61 | 10929.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 11357.50 | 11043.19 | 10968.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 11360.25 | 11377.00 | 11215.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:45:00 | 11389.60 | 11377.00 | 11215.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 11804.20 | 11581.07 | 11460.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:15:00 | 11864.25 | 11581.07 | 11460.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:45:00 | 11873.25 | 11641.02 | 11498.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 12:45:00 | 11916.15 | 11729.30 | 11565.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 13:45:00 | 11900.00 | 11884.91 | 11753.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 11721.45 | 11847.83 | 11769.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:45:00 | 11831.40 | 11847.83 | 11769.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 11762.00 | 11830.66 | 11768.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 13:30:00 | 11885.10 | 11826.04 | 11780.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 09:15:00 | 11903.70 | 11797.71 | 11774.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 10:30:00 | 11927.95 | 11815.20 | 11787.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 11663.15 | 11778.47 | 11781.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 11663.15 | 11778.47 | 11781.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 11234.00 | 11593.89 | 11683.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 11473.00 | 11249.86 | 11341.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 11473.00 | 11249.86 | 11341.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 11473.00 | 11249.86 | 11341.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 11473.00 | 11249.86 | 11341.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 11466.05 | 11293.10 | 11352.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:30:00 | 11523.00 | 11293.10 | 11352.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 11534.90 | 11395.01 | 11388.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 11846.50 | 11520.55 | 11450.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 10:15:00 | 11743.50 | 11784.93 | 11659.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 11:00:00 | 11743.50 | 11784.93 | 11659.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 11739.95 | 11759.17 | 11692.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 11790.75 | 11759.17 | 11692.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:45:00 | 11770.60 | 11801.82 | 11759.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 10:15:00 | 11805.55 | 11801.82 | 11759.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 11600.00 | 11761.45 | 11745.32 | SL hit (close<static) qty=1.00 sl=11678.80 alert=retest2 |

### Cycle 117 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 11571.10 | 11723.38 | 11729.48 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 11899.35 | 11724.73 | 11709.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 11990.80 | 11810.39 | 11753.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 09:15:00 | 12395.25 | 12409.73 | 12212.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 10:00:00 | 12395.25 | 12409.73 | 12212.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 12347.20 | 12394.19 | 12311.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 12325.25 | 12394.19 | 12311.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 12586.60 | 12578.68 | 12512.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:30:00 | 12595.55 | 12578.68 | 12512.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 12166.80 | 12496.53 | 12500.30 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 12480.00 | 12273.99 | 12250.51 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 12168.20 | 12347.14 | 12365.18 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 12923.00 | 12477.49 | 12418.59 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 12215.00 | 12419.47 | 12435.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 12190.55 | 12352.92 | 12401.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 12345.35 | 12295.49 | 12339.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 12345.35 | 12295.49 | 12339.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 12345.35 | 12295.49 | 12339.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 12345.35 | 12295.49 | 12339.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 12379.75 | 12312.35 | 12343.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:00:00 | 12379.75 | 12312.35 | 12343.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 12253.45 | 12300.57 | 12335.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 12098.00 | 12270.74 | 12308.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 12467.00 | 12305.65 | 12293.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 12467.00 | 12305.65 | 12293.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 12620.65 | 12395.75 | 12338.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 12631.25 | 12633.44 | 12524.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 13:45:00 | 12633.85 | 12633.44 | 12524.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 12369.05 | 12658.14 | 12570.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 12369.05 | 12658.14 | 12570.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 12569.20 | 12640.35 | 12570.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 12:15:00 | 12629.65 | 12626.28 | 12570.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 15:00:00 | 12635.00 | 12628.32 | 12619.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 12656.05 | 12624.26 | 12618.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 10:15:00 | 12458.40 | 12607.98 | 12623.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 12458.40 | 12607.98 | 12623.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 12139.35 | 12446.75 | 12530.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 12201.20 | 12180.53 | 12335.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 12194.95 | 12180.53 | 12335.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 11985.65 | 11918.24 | 12059.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:45:00 | 12000.00 | 11918.24 | 12059.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 10160.35 | 10518.18 | 10749.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:45:00 | 9968.90 | 10415.30 | 10682.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:45:00 | 9980.75 | 10311.52 | 10610.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 11:15:00 | 9983.05 | 10195.43 | 10416.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 9864.00 | 10092.60 | 10276.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 10025.65 | 9993.22 | 10107.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 9959.25 | 9993.22 | 10107.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:00:00 | 9922.00 | 9963.34 | 10073.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 13:15:00 | 9481.71 | 9701.83 | 9855.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 13:15:00 | 9483.90 | 9701.83 | 9855.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 10207.40 | 9771.18 | 9845.69 | SL hit (close>ema200) qty=0.50 sl=9771.18 alert=retest2 |

### Cycle 126 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 10182.95 | 9894.39 | 9891.30 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 9796.15 | 9916.89 | 9928.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 9452.70 | 9806.95 | 9875.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 14:15:00 | 9254.40 | 9249.46 | 9437.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 14:45:00 | 9220.05 | 9249.46 | 9437.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 9550.05 | 9310.62 | 9432.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 9582.00 | 9310.62 | 9432.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 9513.60 | 9351.22 | 9440.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:30:00 | 9488.50 | 9426.09 | 9456.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 09:15:00 | 9449.00 | 9469.86 | 9472.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 9575.05 | 9490.90 | 9481.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 9575.05 | 9490.90 | 9481.95 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 9125.55 | 9417.00 | 9451.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 14:15:00 | 8987.50 | 9331.10 | 9409.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 11:15:00 | 9044.95 | 9004.98 | 9098.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 11:45:00 | 9053.45 | 9004.98 | 9098.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 9141.45 | 9032.27 | 9102.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:00:00 | 9141.45 | 9032.27 | 9102.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 9163.95 | 9058.61 | 9108.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:45:00 | 9128.70 | 9058.61 | 9108.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 9333.75 | 9156.04 | 9144.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 9403.85 | 9229.85 | 9186.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 9318.05 | 9332.96 | 9264.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 15:00:00 | 9318.05 | 9332.96 | 9264.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 9303.45 | 9340.31 | 9294.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 9303.45 | 9340.31 | 9294.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 9286.00 | 9329.44 | 9293.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:30:00 | 9295.55 | 9329.44 | 9293.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 9258.95 | 9315.35 | 9290.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:30:00 | 9270.00 | 9315.35 | 9290.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 9289.05 | 9310.09 | 9290.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 9286.75 | 9310.09 | 9290.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 9281.55 | 9304.38 | 9289.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:00:00 | 9413.05 | 9326.11 | 9301.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 9360.00 | 9329.32 | 9304.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 11:00:00 | 9376.80 | 9376.32 | 9344.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 15:15:00 | 9270.00 | 9319.77 | 9325.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 9270.00 | 9319.77 | 9325.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 9197.25 | 9295.27 | 9314.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 14:15:00 | 9056.15 | 9052.85 | 9135.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 15:00:00 | 9056.15 | 9052.85 | 9135.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 9126.90 | 9066.39 | 9127.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 8987.40 | 9052.88 | 9106.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 8998.25 | 9031.73 | 9086.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 14:45:00 | 8971.00 | 8944.19 | 9004.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 09:15:00 | 8538.03 | 8674.71 | 8739.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 09:15:00 | 8548.34 | 8674.71 | 8739.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 09:15:00 | 8522.45 | 8674.71 | 8739.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 8352.00 | 8342.98 | 8467.83 | SL hit (close>ema200) qty=0.50 sl=8342.98 alert=retest2 |

### Cycle 132 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 7953.95 | 7825.96 | 7823.30 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 15:15:00 | 7797.00 | 7846.84 | 7847.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 09:15:00 | 7618.25 | 7801.12 | 7827.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 7510.00 | 7481.73 | 7569.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 11:00:00 | 7510.00 | 7481.73 | 7569.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 7526.10 | 7479.15 | 7539.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:45:00 | 7541.70 | 7479.15 | 7539.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 7540.00 | 7491.32 | 7539.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 7420.30 | 7491.32 | 7539.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 7318.95 | 7456.85 | 7519.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 7292.00 | 7456.85 | 7519.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 7306.70 | 7343.56 | 7390.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 14:15:00 | 7488.45 | 7411.54 | 7402.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 7488.45 | 7411.54 | 7402.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 7585.05 | 7458.48 | 7426.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 7475.05 | 7529.70 | 7487.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 09:15:00 | 7475.05 | 7529.70 | 7487.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 7475.05 | 7529.70 | 7487.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 14:15:00 | 7628.00 | 7535.85 | 7503.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:15:00 | 7611.50 | 7587.76 | 7555.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:30:00 | 7676.00 | 7592.36 | 7565.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:00:00 | 7604.70 | 7615.30 | 7586.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 7765.65 | 7653.79 | 7612.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 7688.50 | 7653.79 | 7612.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 7898.35 | 7885.54 | 7795.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 7798.90 | 7885.54 | 7795.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 7947.80 | 7956.09 | 7897.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 7918.20 | 7956.09 | 7897.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 7912.00 | 7947.28 | 7898.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:45:00 | 7891.20 | 7947.28 | 7898.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 7868.00 | 7931.42 | 7895.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 7961.65 | 7931.42 | 7895.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 7940.00 | 7933.14 | 7899.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 7853.30 | 7894.91 | 7895.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 09:15:00 | 7853.30 | 7894.91 | 7895.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 7803.10 | 7870.81 | 7884.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 14:15:00 | 7860.10 | 7843.08 | 7867.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 14:15:00 | 7860.10 | 7843.08 | 7867.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 7860.10 | 7843.08 | 7867.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:45:00 | 7863.55 | 7843.08 | 7867.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 7820.00 | 7838.46 | 7863.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 7734.10 | 7838.46 | 7863.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 7347.40 | 7473.20 | 7553.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 15:15:00 | 7330.00 | 7304.42 | 7413.26 | SL hit (close>ema200) qty=0.50 sl=7304.42 alert=retest2 |

### Cycle 136 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 7560.00 | 7450.17 | 7439.89 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 7348.60 | 7429.86 | 7431.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 10:15:00 | 7322.55 | 7408.40 | 7421.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-09 12:15:00 | 7475.00 | 7415.98 | 7422.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 12:15:00 | 7475.00 | 7415.98 | 7422.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 12:15:00 | 7475.00 | 7415.98 | 7422.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 13:00:00 | 7475.00 | 7415.98 | 7422.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 7375.05 | 7407.80 | 7418.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:15:00 | 7360.50 | 7407.80 | 7418.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 7569.15 | 7438.01 | 7429.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 7569.15 | 7438.01 | 7429.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 7633.20 | 7496.57 | 7458.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 7515.15 | 7518.43 | 7476.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 14:00:00 | 7515.15 | 7518.43 | 7476.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 8203.50 | 8143.76 | 8030.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 8344.00 | 8189.01 | 8107.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 11:15:00 | 8675.00 | 8733.98 | 8734.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 8675.00 | 8733.98 | 8734.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 11:15:00 | 8614.00 | 8691.44 | 8709.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 15:15:00 | 8655.00 | 8648.12 | 8679.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-06 09:15:00 | 8574.00 | 8648.12 | 8679.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 8517.50 | 8621.99 | 8664.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 8498.50 | 8621.99 | 8664.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 8498.50 | 8597.30 | 8649.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 8303.00 | 8483.56 | 8515.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 8073.57 | 8220.60 | 8346.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 8073.57 | 8220.60 | 8346.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 8466.50 | 8138.57 | 8220.13 | SL hit (close>ema200) qty=0.50 sl=8138.57 alert=retest2 |

### Cycle 140 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 8545.00 | 8277.13 | 8272.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 8633.00 | 8348.30 | 8305.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 8532.50 | 8538.25 | 8450.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:45:00 | 8503.50 | 8538.25 | 8450.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 8512.00 | 8509.24 | 8461.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 8608.00 | 8520.82 | 8499.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:30:00 | 8594.00 | 8545.84 | 8517.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 8502.00 | 8600.34 | 8603.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 8502.00 | 8600.34 | 8603.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 8436.00 | 8567.47 | 8587.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 8513.50 | 8348.99 | 8393.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 8513.50 | 8348.99 | 8393.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 8513.50 | 8348.99 | 8393.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 8513.50 | 8348.99 | 8393.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 8462.00 | 8371.59 | 8400.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 8446.00 | 8371.59 | 8400.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 8389.00 | 8384.58 | 8399.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:15:00 | 8421.00 | 8384.58 | 8399.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 8369.50 | 8381.57 | 8397.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 15:15:00 | 8356.00 | 8381.57 | 8397.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 8455.00 | 8392.16 | 8399.01 | SL hit (close>static) qty=1.00 sl=8428.50 alert=retest2 |

### Cycle 142 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 8440.50 | 8406.84 | 8404.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 8462.50 | 8421.68 | 8412.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 8406.50 | 8430.90 | 8419.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 8406.50 | 8430.90 | 8419.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 8406.50 | 8430.90 | 8419.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 8441.50 | 8430.90 | 8419.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 8425.00 | 8429.72 | 8420.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 8469.00 | 8429.72 | 8420.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 15:15:00 | 8405.00 | 8416.82 | 8417.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 8405.00 | 8416.82 | 8417.27 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 8479.00 | 8429.25 | 8422.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 12:15:00 | 8570.50 | 8473.40 | 8446.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 14:15:00 | 8494.50 | 8505.90 | 8485.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 14:45:00 | 8499.00 | 8505.90 | 8485.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 8519.00 | 8508.52 | 8488.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 8513.50 | 8508.52 | 8488.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 8490.50 | 8504.92 | 8488.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:00:00 | 8559.50 | 8515.83 | 8495.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:15:00 | 8537.50 | 8515.97 | 8497.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 8430.50 | 8496.84 | 8493.05 | SL hit (close<static) qty=1.00 sl=8458.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 8442.00 | 8485.87 | 8488.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 8413.00 | 8471.29 | 8481.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 8371.50 | 8359.35 | 8395.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 15:00:00 | 8371.50 | 8359.35 | 8395.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 8547.00 | 8396.99 | 8406.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 8547.00 | 8396.99 | 8406.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 8714.50 | 8460.49 | 8434.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 8845.00 | 8661.43 | 8563.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 8728.00 | 8736.30 | 8661.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:45:00 | 8750.50 | 8736.30 | 8661.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 9588.00 | 9443.33 | 9368.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 9640.50 | 9443.33 | 9368.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 9625.00 | 9507.98 | 9445.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:00:00 | 9610.50 | 9528.49 | 9460.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:00:00 | 9627.00 | 9548.19 | 9475.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 9645.00 | 9689.18 | 9645.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 9618.00 | 9689.18 | 9645.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 9615.00 | 9674.34 | 9642.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 9615.00 | 9674.34 | 9642.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 9662.50 | 9671.97 | 9644.46 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 9459.00 | 9624.98 | 9627.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 9459.00 | 9624.98 | 9627.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 9364.00 | 9572.79 | 9603.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 9403.50 | 9377.41 | 9441.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 9403.50 | 9377.41 | 9441.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 9271.50 | 9214.03 | 9266.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:00:00 | 9190.00 | 9209.06 | 9255.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 14:15:00 | 9189.50 | 9209.96 | 9247.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 9082.50 | 9002.87 | 8993.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 9082.50 | 9002.87 | 8993.95 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 9011.00 | 9028.99 | 9030.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 8951.00 | 9013.40 | 9023.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 9053.00 | 9002.45 | 9014.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 9053.00 | 9002.45 | 9014.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 9053.00 | 9002.45 | 9014.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 9053.00 | 9002.45 | 9014.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 8981.50 | 8998.26 | 9011.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 8973.00 | 8998.26 | 9011.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 8975.00 | 9000.17 | 9010.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 10:00:00 | 8972.50 | 8994.63 | 9006.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 8974.50 | 8988.95 | 9001.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 8524.35 | 8666.06 | 8751.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 8526.25 | 8666.06 | 8751.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 8523.88 | 8666.06 | 8751.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 8525.77 | 8666.06 | 8751.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 8679.50 | 8668.75 | 8745.03 | SL hit (close>ema200) qty=0.50 sl=8668.75 alert=retest2 |

### Cycle 150 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 8815.00 | 8750.07 | 8750.04 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 8735.50 | 8776.17 | 8780.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 8717.00 | 8759.03 | 8771.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 11:15:00 | 8779.50 | 8763.12 | 8772.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 11:15:00 | 8779.50 | 8763.12 | 8772.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 8779.50 | 8763.12 | 8772.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 8775.50 | 8763.12 | 8772.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 8822.50 | 8775.00 | 8776.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:30:00 | 8823.00 | 8775.00 | 8776.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 13:15:00 | 8823.50 | 8784.70 | 8781.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 8845.50 | 8800.08 | 8789.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 8764.50 | 8807.29 | 8799.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 8764.50 | 8807.29 | 8799.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 8764.50 | 8807.29 | 8799.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 8764.50 | 8807.29 | 8799.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 8734.50 | 8792.73 | 8793.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 8700.00 | 8774.19 | 8784.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 8727.00 | 8724.25 | 8750.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 8727.00 | 8724.25 | 8750.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 8727.00 | 8724.25 | 8750.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 8745.00 | 8724.25 | 8750.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 8749.00 | 8729.20 | 8750.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 8749.00 | 8729.20 | 8750.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 8767.00 | 8736.76 | 8751.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:45:00 | 8800.00 | 8736.76 | 8751.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 8749.50 | 8739.31 | 8751.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 8768.50 | 8739.31 | 8751.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 8785.00 | 8748.45 | 8754.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 8785.00 | 8748.45 | 8754.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 8767.50 | 8752.26 | 8755.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 8817.00 | 8752.26 | 8755.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 8851.50 | 8772.11 | 8764.38 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 8670.00 | 8808.34 | 8813.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 8625.00 | 8771.68 | 8796.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 8696.00 | 8677.68 | 8726.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 8696.00 | 8677.68 | 8726.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 8602.00 | 8655.08 | 8707.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 8687.00 | 8655.08 | 8707.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 8662.50 | 8651.13 | 8684.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 8692.00 | 8651.13 | 8684.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 8607.00 | 8641.08 | 8673.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 8592.00 | 8641.08 | 8673.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:00:00 | 8605.00 | 8628.86 | 8657.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:30:00 | 8601.50 | 8619.88 | 8650.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 8570.00 | 8501.62 | 8497.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 8570.00 | 8501.62 | 8497.45 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 8456.00 | 8507.97 | 8511.78 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 8570.00 | 8513.26 | 8510.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 8590.00 | 8528.61 | 8517.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 8505.00 | 8523.89 | 8516.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 8505.00 | 8523.89 | 8516.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 8505.00 | 8523.89 | 8516.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 8505.00 | 8523.89 | 8516.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 8527.50 | 8524.61 | 8517.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 8509.00 | 8524.61 | 8517.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 8509.00 | 8521.49 | 8516.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 8509.00 | 8521.49 | 8516.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 8507.00 | 8518.59 | 8515.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:15:00 | 8497.00 | 8518.59 | 8515.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 8479.00 | 8510.67 | 8512.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 8415.00 | 8491.54 | 8503.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 8617.50 | 8462.29 | 8469.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 8617.50 | 8462.29 | 8469.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 8617.50 | 8462.29 | 8469.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 8617.50 | 8462.29 | 8469.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 8560.00 | 8481.83 | 8477.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 8624.50 | 8565.83 | 8536.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 8585.00 | 8595.10 | 8565.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 15:00:00 | 8585.00 | 8595.10 | 8565.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 8541.00 | 8588.79 | 8572.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 8541.00 | 8588.79 | 8572.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 8589.00 | 8588.83 | 8573.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 8535.50 | 8588.83 | 8573.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 8574.50 | 8585.96 | 8573.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:00:00 | 8574.50 | 8585.96 | 8573.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 8553.50 | 8579.47 | 8571.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 8553.50 | 8579.47 | 8571.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 8584.50 | 8580.48 | 8573.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 8510.00 | 8572.78 | 8570.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 8552.00 | 8568.63 | 8568.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:45:00 | 8576.00 | 8577.10 | 8572.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 13:15:00 | 8666.50 | 8690.34 | 8691.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 8666.50 | 8690.34 | 8691.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 8625.00 | 8677.27 | 8685.71 | Break + close below crossover candle low |

### Cycle 162 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 8845.00 | 8703.57 | 8695.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 8894.00 | 8741.66 | 8713.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 14:15:00 | 8753.50 | 8777.72 | 8743.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 14:15:00 | 8753.50 | 8777.72 | 8743.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 8753.50 | 8777.72 | 8743.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 8753.50 | 8777.72 | 8743.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 8751.00 | 8772.38 | 8744.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 8689.00 | 8772.38 | 8744.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 8723.00 | 8762.50 | 8742.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 8685.50 | 8762.50 | 8742.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 8696.00 | 8749.20 | 8738.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 8686.00 | 8749.20 | 8738.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 8685.00 | 8736.36 | 8733.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 8685.00 | 8736.36 | 8733.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 8655.00 | 8720.09 | 8726.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 8575.00 | 8691.07 | 8712.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 8445.50 | 8442.71 | 8530.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 8445.50 | 8442.71 | 8530.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 8417.00 | 8384.08 | 8457.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 8451.50 | 8384.08 | 8457.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 8471.00 | 8401.46 | 8459.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 8471.00 | 8401.46 | 8459.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 8425.00 | 8406.17 | 8455.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:30:00 | 8420.00 | 8415.54 | 8455.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 8500.00 | 8444.03 | 8459.51 | SL hit (close>static) qty=1.00 sl=8478.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 8540.00 | 8475.99 | 8471.39 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 8410.50 | 8462.84 | 8468.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 10:15:00 | 8359.50 | 8415.62 | 8438.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 8318.00 | 8286.25 | 8325.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 8318.00 | 8286.25 | 8325.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 8318.00 | 8286.25 | 8325.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 8336.50 | 8286.25 | 8325.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 8370.00 | 8303.00 | 8329.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 8370.00 | 8303.00 | 8329.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 8389.00 | 8320.20 | 8335.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 8389.00 | 8320.20 | 8335.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 8294.00 | 8322.14 | 8333.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:45:00 | 8315.50 | 8322.14 | 8333.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 8360.00 | 8326.17 | 8332.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 8396.00 | 8326.17 | 8332.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 8381.00 | 8337.13 | 8337.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 8388.50 | 8337.13 | 8337.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 8357.50 | 8341.21 | 8339.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 8394.50 | 8355.19 | 8346.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 9075.00 | 9103.23 | 8890.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 9075.00 | 9103.23 | 8890.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 8987.50 | 9092.56 | 8953.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 8940.50 | 9092.56 | 8953.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 8946.00 | 9043.40 | 8954.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:00:00 | 8946.00 | 9043.40 | 8954.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 8945.00 | 9023.72 | 8953.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 8973.00 | 9010.97 | 8954.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:15:00 | 8980.00 | 9010.97 | 8954.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 8922.50 | 8981.54 | 8953.90 | SL hit (close<static) qty=1.00 sl=8927.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 9055.00 | 9092.42 | 9093.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 9009.00 | 9075.74 | 9086.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 9089.00 | 9047.70 | 9066.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 11:15:00 | 9089.00 | 9047.70 | 9066.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 9089.00 | 9047.70 | 9066.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 9089.00 | 9047.70 | 9066.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 9063.50 | 9050.86 | 9066.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:30:00 | 9032.50 | 9054.83 | 9065.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 9102.00 | 9065.09 | 9068.46 | SL hit (close>static) qty=1.00 sl=9099.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 9119.00 | 9078.62 | 9073.84 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 8994.50 | 9059.97 | 9068.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 12:15:00 | 8967.00 | 9029.59 | 9047.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 8533.00 | 8437.21 | 8510.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 11:15:00 | 8533.00 | 8437.21 | 8510.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 8533.00 | 8437.21 | 8510.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 8533.00 | 8437.21 | 8510.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 8515.00 | 8452.77 | 8510.83 | EMA400 retest candle locked (from downside) |

### Cycle 170 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 8735.00 | 8581.47 | 8561.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 8872.50 | 8639.68 | 8589.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 9167.50 | 9196.20 | 9083.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 11:00:00 | 9167.50 | 9196.20 | 9083.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 9183.00 | 9186.11 | 9114.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 9121.50 | 9186.11 | 9114.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 9236.50 | 9269.86 | 9223.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:30:00 | 9243.50 | 9269.86 | 9223.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 9199.00 | 9255.69 | 9221.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 9199.00 | 9255.69 | 9221.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 9204.50 | 9245.45 | 9219.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 9255.50 | 9236.56 | 9218.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 9151.50 | 9219.38 | 9213.48 | SL hit (close<static) qty=1.00 sl=9190.50 alert=retest2 |

### Cycle 171 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 9118.00 | 9199.10 | 9204.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 9055.00 | 9148.32 | 9175.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 9034.50 | 9030.38 | 9094.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 11:45:00 | 8854.50 | 8979.72 | 9060.35 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 8644.50 | 8629.20 | 8693.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:30:00 | 8575.00 | 8603.52 | 8660.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:45:00 | 8583.00 | 8571.73 | 8615.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 8679.50 | 8592.32 | 8610.58 | SL hit (close>ema400) qty=1.00 sl=8610.58 alert=retest1 |

### Cycle 172 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 8671.00 | 8629.47 | 8624.73 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 8586.00 | 8624.39 | 8628.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 8552.00 | 8609.92 | 8621.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 15:15:00 | 8624.50 | 8611.09 | 8620.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 15:15:00 | 8624.50 | 8611.09 | 8620.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 8624.50 | 8611.09 | 8620.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 8664.00 | 8611.09 | 8620.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 8620.00 | 8612.87 | 8620.08 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 8680.50 | 8635.86 | 8629.86 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 8609.00 | 8629.53 | 8631.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 8589.00 | 8614.08 | 8622.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 8235.00 | 8140.67 | 8201.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 10:15:00 | 8235.00 | 8140.67 | 8201.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 8235.00 | 8140.67 | 8201.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 8235.00 | 8140.67 | 8201.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 8211.50 | 8154.83 | 8202.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:30:00 | 8205.50 | 8161.57 | 8201.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:30:00 | 8200.00 | 8170.05 | 8201.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:30:00 | 8205.00 | 8169.04 | 8198.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 8198.00 | 8185.59 | 8200.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 8191.00 | 8186.67 | 8199.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:15:00 | 8176.00 | 8186.67 | 8199.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:00:00 | 8184.00 | 8191.54 | 8199.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 8248.00 | 8210.11 | 8206.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 8248.00 | 8210.11 | 8206.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 8310.00 | 8230.09 | 8216.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 8357.00 | 8360.04 | 8316.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:45:00 | 8355.00 | 8360.04 | 8316.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 8298.50 | 8344.29 | 8317.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 8298.50 | 8344.29 | 8317.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 8300.50 | 8335.53 | 8315.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 8265.00 | 8335.53 | 8315.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 8264.00 | 8297.20 | 8300.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 8173.00 | 8272.36 | 8289.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 8297.50 | 8256.69 | 8273.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 8297.50 | 8256.69 | 8273.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 8297.50 | 8256.69 | 8273.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 8288.00 | 8256.69 | 8273.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 8306.50 | 8266.65 | 8276.93 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 8310.00 | 8288.21 | 8285.31 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 8210.50 | 8274.14 | 8279.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 13:15:00 | 8200.00 | 8238.96 | 8259.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 8296.00 | 8230.48 | 8248.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 8296.00 | 8230.48 | 8248.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 8296.00 | 8230.48 | 8248.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 8278.50 | 8230.48 | 8248.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 8282.50 | 8240.88 | 8251.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 8312.00 | 8240.88 | 8251.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 8336.00 | 8259.91 | 8259.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 13:15:00 | 8348.00 | 8290.98 | 8274.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 8370.50 | 8386.96 | 8344.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 15:00:00 | 8370.50 | 8386.96 | 8344.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 8272.50 | 8361.04 | 8340.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 8272.50 | 8361.04 | 8340.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 8259.00 | 8340.63 | 8332.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 8242.50 | 8340.63 | 8332.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 8242.00 | 8320.90 | 8324.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 12:15:00 | 8215.50 | 8299.82 | 8314.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 8172.00 | 8122.05 | 8164.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 8172.00 | 8122.05 | 8164.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 8172.00 | 8122.05 | 8164.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 8157.00 | 8122.05 | 8164.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 8133.50 | 8124.34 | 8161.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:15:00 | 8165.00 | 8124.34 | 8161.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 8185.00 | 8136.47 | 8164.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 8185.00 | 8136.47 | 8164.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 8175.50 | 8144.28 | 8165.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 8175.50 | 8144.28 | 8165.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 8181.00 | 8151.62 | 8166.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:00:00 | 8181.00 | 8151.62 | 8166.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 8200.00 | 8165.52 | 8170.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 8162.00 | 8165.52 | 8170.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 8225.00 | 8177.41 | 8175.45 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 8143.00 | 8168.31 | 8171.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 8105.00 | 8141.85 | 8155.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 8143.50 | 8124.92 | 8138.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 8143.50 | 8124.92 | 8138.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 8143.50 | 8124.92 | 8138.86 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 8174.50 | 8148.77 | 8146.87 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 8120.50 | 8142.67 | 8144.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 8105.00 | 8135.13 | 8141.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 15:15:00 | 8133.00 | 8117.16 | 8128.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 15:15:00 | 8133.00 | 8117.16 | 8128.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 8133.00 | 8117.16 | 8128.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 8127.50 | 8117.16 | 8128.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 8081.50 | 8110.03 | 8124.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:00:00 | 8078.00 | 8103.62 | 8120.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:45:00 | 8075.50 | 8094.28 | 8111.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 8075.00 | 8092.03 | 8109.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 8213.00 | 8113.50 | 8115.77 | SL hit (close>static) qty=1.00 sl=8190.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 8193.00 | 8129.40 | 8122.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 10:15:00 | 8221.00 | 8180.31 | 8156.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 8135.50 | 8189.18 | 8173.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 8135.50 | 8189.18 | 8173.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 8135.50 | 8189.18 | 8173.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 8135.50 | 8189.18 | 8173.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 8084.50 | 8168.24 | 8165.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 8084.50 | 8168.24 | 8165.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 8062.00 | 8146.99 | 8156.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 8035.00 | 8124.60 | 8145.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 8053.50 | 8044.45 | 8087.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 8053.50 | 8044.45 | 8087.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 8082.50 | 8053.99 | 8084.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 8082.50 | 8053.99 | 8084.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 8059.50 | 8055.09 | 8081.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:15:00 | 8062.00 | 8055.09 | 8081.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 8062.00 | 8056.47 | 8080.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 8068.00 | 8056.47 | 8080.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 8065.50 | 8058.28 | 8078.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 7988.00 | 8045.82 | 8071.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 09:15:00 | 7588.60 | 7941.80 | 8001.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 10:15:00 | 7961.50 | 7945.74 | 7997.73 | SL hit (close>ema200) qty=0.50 sl=7945.74 alert=retest2 |

### Cycle 188 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 7896.00 | 7769.99 | 7759.79 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 7779.50 | 7802.34 | 7802.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 7732.00 | 7783.10 | 7793.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 12:15:00 | 7782.00 | 7775.99 | 7786.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 12:45:00 | 7785.50 | 7775.99 | 7786.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 7760.00 | 7772.79 | 7784.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:30:00 | 7786.00 | 7772.79 | 7784.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 7778.50 | 7773.93 | 7783.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 7778.50 | 7773.93 | 7783.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 7760.00 | 7771.15 | 7781.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 7745.50 | 7771.15 | 7781.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 7718.00 | 7760.52 | 7775.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 7706.00 | 7760.52 | 7775.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 7725.00 | 7685.89 | 7684.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 7725.00 | 7685.89 | 7684.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 7740.00 | 7711.89 | 7698.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 7685.50 | 7706.61 | 7697.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 7685.50 | 7706.61 | 7697.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 7685.50 | 7706.61 | 7697.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 7638.00 | 7706.61 | 7697.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 7677.00 | 7700.69 | 7695.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 7673.00 | 7700.69 | 7695.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 7697.50 | 7700.05 | 7695.89 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 7670.00 | 7691.63 | 7692.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 7641.50 | 7668.02 | 7679.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 7655.00 | 7640.25 | 7660.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 7655.00 | 7640.25 | 7660.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 7655.00 | 7640.25 | 7660.06 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 7754.50 | 7676.50 | 7673.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 12:15:00 | 7776.50 | 7696.50 | 7683.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 7671.00 | 7746.88 | 7720.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 7671.00 | 7746.88 | 7720.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 7671.00 | 7746.88 | 7720.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 7671.00 | 7746.88 | 7720.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 7662.00 | 7729.90 | 7714.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:30:00 | 7675.00 | 7729.90 | 7714.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 7603.50 | 7690.24 | 7698.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 7575.00 | 7651.95 | 7678.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 12:15:00 | 7608.50 | 7607.47 | 7646.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 13:00:00 | 7608.50 | 7607.47 | 7646.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 7578.50 | 7565.73 | 7595.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 7600.00 | 7565.73 | 7595.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 7600.00 | 7572.58 | 7596.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 7633.50 | 7572.58 | 7596.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 7634.50 | 7584.97 | 7599.55 | EMA400 retest candle locked (from downside) |

### Cycle 194 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 7699.00 | 7623.87 | 7614.09 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 7584.00 | 7612.68 | 7613.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 7533.50 | 7596.84 | 7605.84 | Break + close below crossover candle low |

### Cycle 196 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 7759.00 | 7610.31 | 7607.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 7910.00 | 7670.25 | 7635.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 7937.50 | 7937.75 | 7858.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 09:45:00 | 7905.50 | 7937.75 | 7858.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 7811.50 | 7902.85 | 7866.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 7811.50 | 7902.85 | 7866.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 7818.00 | 7885.88 | 7861.83 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 7652.00 | 7824.88 | 7837.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 7600.00 | 7779.91 | 7815.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 7985.00 | 7761.90 | 7781.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 7985.00 | 7761.90 | 7781.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 7985.00 | 7761.90 | 7781.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 7996.00 | 7761.90 | 7781.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 7891.50 | 7810.72 | 7801.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 7997.00 | 7898.85 | 7855.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 7940.00 | 7947.68 | 7901.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:15:00 | 8029.50 | 7947.68 | 7901.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 8004.00 | 7979.18 | 7941.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 8054.50 | 7979.18 | 7941.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:45:00 | 8006.00 | 7986.29 | 7951.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 7955.50 | 7982.30 | 7961.34 | SL hit (close<ema400) qty=1.00 sl=7961.34 alert=retest1 |

### Cycle 199 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 7797.00 | 7925.68 | 7941.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 7772.50 | 7895.04 | 7926.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 12:15:00 | 7750.50 | 7745.95 | 7790.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 13:00:00 | 7750.50 | 7745.95 | 7790.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 7759.50 | 7748.66 | 7787.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 7686.50 | 7752.20 | 7782.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:00:00 | 7701.50 | 7742.06 | 7774.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 7702.00 | 7710.39 | 7742.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 7869.00 | 7746.13 | 7753.51 | SL hit (close>static) qty=1.00 sl=7792.00 alert=retest2 |

### Cycle 200 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 7919.50 | 7780.80 | 7768.60 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 7556.50 | 7772.70 | 7779.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 7463.00 | 7540.90 | 7629.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 7303.00 | 7283.51 | 7376.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 7303.00 | 7283.51 | 7376.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 7315.50 | 7284.90 | 7336.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 7363.50 | 7284.90 | 7336.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 7330.00 | 7293.92 | 7335.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:15:00 | 7348.00 | 7293.92 | 7335.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 7346.00 | 7304.34 | 7336.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:30:00 | 7354.50 | 7304.34 | 7336.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 7342.50 | 7311.97 | 7337.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:45:00 | 7354.50 | 7311.97 | 7337.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 7341.00 | 7324.02 | 7338.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 7341.00 | 7324.02 | 7338.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 7273.00 | 7314.77 | 7332.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:15:00 | 7257.50 | 7307.37 | 7325.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:15:00 | 6894.62 | 7138.25 | 7230.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-13 09:15:00 | 6531.75 | 6808.30 | 7007.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 202 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 6845.00 | 6808.29 | 6806.02 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 6739.50 | 6794.53 | 6799.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 14:15:00 | 6709.00 | 6744.04 | 6769.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 09:15:00 | 6772.00 | 6742.59 | 6764.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 6772.00 | 6742.59 | 6764.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 6772.00 | 6742.59 | 6764.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 6695.00 | 6731.06 | 6754.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:30:00 | 6695.00 | 6723.84 | 6749.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 6805.50 | 6617.17 | 6597.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 6805.50 | 6617.17 | 6597.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 12:15:00 | 6832.00 | 6719.06 | 6654.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 15:15:00 | 6897.00 | 6901.34 | 6814.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 09:15:00 | 6863.50 | 6901.34 | 6814.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 6949.00 | 6910.87 | 6827.15 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 10:15:00 | 6755.50 | 6820.49 | 6827.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 11:15:00 | 6734.00 | 6785.34 | 6804.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 6725.00 | 6692.96 | 6731.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 15:00:00 | 6725.00 | 6692.96 | 6731.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 6722.50 | 6698.87 | 6730.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 6738.00 | 6698.87 | 6730.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 6768.00 | 6712.70 | 6733.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 6768.00 | 6712.70 | 6733.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 6779.00 | 6725.96 | 6737.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 6781.00 | 6725.96 | 6737.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 6803.00 | 6756.15 | 6749.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 6814.00 | 6767.72 | 6755.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 6769.00 | 6781.43 | 6766.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 6769.00 | 6781.43 | 6766.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 6769.00 | 6781.43 | 6766.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:45:00 | 6787.50 | 6781.43 | 6766.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 6779.50 | 6781.04 | 6767.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 6761.50 | 6781.04 | 6767.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 6767.50 | 6778.34 | 6767.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 6767.50 | 6778.34 | 6767.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 6822.50 | 6787.17 | 6772.44 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 6675.50 | 6765.52 | 6766.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 6598.00 | 6700.60 | 6730.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 6611.00 | 6399.02 | 6444.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 6611.00 | 6399.02 | 6444.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 6611.00 | 6399.02 | 6444.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 6611.00 | 6399.02 | 6444.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 6637.50 | 6446.72 | 6462.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 6640.50 | 6446.72 | 6462.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 6610.00 | 6479.37 | 6475.77 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 6419.50 | 6485.04 | 6490.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 6395.50 | 6454.97 | 6475.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 6584.50 | 6458.32 | 6469.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 6584.50 | 6458.32 | 6469.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 6584.50 | 6458.32 | 6469.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 6584.50 | 6458.32 | 6469.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 6642.00 | 6495.06 | 6485.28 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 6414.50 | 6506.36 | 6508.55 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 6591.00 | 6507.22 | 6504.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 6626.00 | 6530.98 | 6515.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 6593.50 | 6622.16 | 6585.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 6593.50 | 6622.16 | 6585.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 6605.00 | 6618.73 | 6587.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 7014.50 | 6618.73 | 6587.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 7715.95 | 7487.61 | 7371.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 9494.50 | 9664.68 | 9681.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 10:15:00 | 9341.00 | 9491.44 | 9567.31 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 7923.85 | 2024-05-17 15:15:00 | 7777.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-05-24 10:15:00 | 7572.05 | 2024-06-03 12:15:00 | 7550.00 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2024-05-24 11:30:00 | 7567.50 | 2024-06-03 12:15:00 | 7550.00 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2024-05-24 12:45:00 | 7558.35 | 2024-06-03 12:15:00 | 7550.00 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2024-05-27 15:15:00 | 7555.00 | 2024-06-03 12:15:00 | 7550.00 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2024-05-30 09:30:00 | 7533.40 | 2024-06-03 12:15:00 | 7550.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-05-30 10:00:00 | 7530.05 | 2024-06-03 12:15:00 | 7550.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-05-30 11:00:00 | 7534.20 | 2024-06-03 12:15:00 | 7550.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-06-03 12:15:00 | 7535.00 | 2024-06-03 12:15:00 | 7550.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-07-10 09:15:00 | 10345.60 | 2024-07-12 11:15:00 | 10590.20 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-07-18 12:30:00 | 10956.05 | 2024-07-19 13:15:00 | 10758.40 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-07-19 10:15:00 | 10931.05 | 2024-07-19 13:15:00 | 10758.40 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-07-19 12:00:00 | 10987.95 | 2024-07-19 13:15:00 | 10758.40 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-07-22 09:15:00 | 10958.00 | 2024-07-22 13:15:00 | 10860.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-07-29 14:45:00 | 11379.95 | 2024-07-30 14:15:00 | 11104.85 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-08-01 12:45:00 | 10885.00 | 2024-08-02 15:15:00 | 10340.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 12:45:00 | 10885.00 | 2024-08-05 10:15:00 | 9796.50 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-14 11:00:00 | 10834.90 | 2024-08-14 14:15:00 | 10615.40 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-08-14 12:00:00 | 10855.00 | 2024-08-14 14:15:00 | 10615.40 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-08-16 09:15:00 | 10942.40 | 2024-08-23 10:15:00 | 10975.05 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2024-08-26 12:45:00 | 11033.30 | 2024-08-28 11:15:00 | 11061.75 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-09-20 13:30:00 | 11227.25 | 2024-09-20 14:15:00 | 11385.70 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-10-01 13:15:00 | 11438.95 | 2024-10-03 10:15:00 | 11251.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-10-03 09:30:00 | 11472.65 | 2024-10-03 10:15:00 | 11251.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-10-04 12:30:00 | 11118.25 | 2024-10-08 15:15:00 | 11249.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-10-04 13:00:00 | 11133.30 | 2024-10-08 15:15:00 | 11249.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-10-08 14:30:00 | 11121.35 | 2024-10-08 15:15:00 | 11249.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-10-14 09:30:00 | 11632.10 | 2024-10-16 14:15:00 | 11567.80 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-10-14 12:30:00 | 11658.15 | 2024-10-16 15:15:00 | 11589.80 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-10-14 13:30:00 | 11628.40 | 2024-10-16 15:15:00 | 11589.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-10-16 10:30:00 | 11627.60 | 2024-10-16 15:15:00 | 11589.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-10-16 13:15:00 | 11708.80 | 2024-10-16 15:15:00 | 11589.80 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-10-21 12:00:00 | 11427.70 | 2024-10-22 14:15:00 | 10856.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 11427.70 | 2024-10-23 09:15:00 | 11170.45 | STOP_HIT | 0.50 | 2.25% |
| BUY | retest2 | 2024-11-11 10:15:00 | 11864.25 | 2024-11-18 09:15:00 | 11663.15 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-11-11 10:45:00 | 11873.25 | 2024-11-18 09:15:00 | 11663.15 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-11-11 12:45:00 | 11916.15 | 2024-11-18 09:15:00 | 11663.15 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-11-12 13:45:00 | 11900.00 | 2024-11-18 09:15:00 | 11663.15 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-11-13 13:30:00 | 11885.10 | 2024-11-18 09:15:00 | 11663.15 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-11-14 09:15:00 | 11903.70 | 2024-11-18 09:15:00 | 11663.15 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-11-14 10:30:00 | 11927.95 | 2024-11-18 09:15:00 | 11663.15 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-11-27 09:15:00 | 11790.75 | 2024-11-28 10:15:00 | 11600.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-11-28 09:45:00 | 11770.60 | 2024-11-28 10:15:00 | 11600.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-11-28 10:15:00 | 11805.55 | 2024-11-28 10:15:00 | 11600.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-12-26 09:30:00 | 12098.00 | 2024-12-27 10:15:00 | 12467.00 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-12-31 12:15:00 | 12629.65 | 2025-01-03 10:15:00 | 12458.40 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-01-01 15:00:00 | 12635.00 | 2025-01-03 10:15:00 | 12458.40 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-01-02 09:15:00 | 12656.05 | 2025-01-03 10:15:00 | 12458.40 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-01-16 10:45:00 | 9968.90 | 2025-01-22 13:15:00 | 9481.71 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-01-16 11:45:00 | 9980.75 | 2025-01-22 13:15:00 | 9483.90 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2025-01-16 10:45:00 | 9968.90 | 2025-01-23 09:15:00 | 10207.40 | STOP_HIT | 0.50 | -2.39% |
| SELL | retest2 | 2025-01-16 11:45:00 | 9980.75 | 2025-01-23 09:15:00 | 10207.40 | STOP_HIT | 0.50 | -2.27% |
| SELL | retest2 | 2025-01-17 11:15:00 | 9983.05 | 2025-01-23 11:15:00 | 10182.95 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-01-20 09:15:00 | 9864.00 | 2025-01-23 11:15:00 | 10182.95 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-01-21 10:15:00 | 9959.25 | 2025-01-23 11:15:00 | 10182.95 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-01-21 12:00:00 | 9922.00 | 2025-01-23 11:15:00 | 10182.95 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-01-29 13:30:00 | 9488.50 | 2025-01-30 09:15:00 | 9575.05 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-01-30 09:15:00 | 9449.00 | 2025-01-30 09:15:00 | 9575.05 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-02-07 11:00:00 | 9413.05 | 2025-02-10 15:15:00 | 9270.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-02-07 12:15:00 | 9360.00 | 2025-02-10 15:15:00 | 9270.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-02-10 11:00:00 | 9376.80 | 2025-02-10 15:15:00 | 9270.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-02-13 13:00:00 | 8987.40 | 2025-02-21 09:15:00 | 8538.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 8998.25 | 2025-02-21 09:15:00 | 8548.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 14:45:00 | 8971.00 | 2025-02-21 09:15:00 | 8522.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:00:00 | 8987.40 | 2025-02-24 13:15:00 | 8352.00 | STOP_HIT | 0.50 | 7.07% |
| SELL | retest2 | 2025-02-13 15:00:00 | 8998.25 | 2025-02-24 13:15:00 | 8352.00 | STOP_HIT | 0.50 | 7.18% |
| SELL | retest2 | 2025-02-14 14:45:00 | 8971.00 | 2025-02-24 13:15:00 | 8352.00 | STOP_HIT | 0.50 | 6.90% |
| SELL | retest2 | 2025-03-12 10:15:00 | 7292.00 | 2025-03-17 14:15:00 | 7488.45 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-03-13 14:15:00 | 7306.70 | 2025-03-17 14:15:00 | 7488.45 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-03-19 14:15:00 | 7628.00 | 2025-03-28 09:15:00 | 7853.30 | STOP_HIT | 1.00 | 2.95% |
| BUY | retest2 | 2025-03-20 14:15:00 | 7611.50 | 2025-03-28 09:15:00 | 7853.30 | STOP_HIT | 1.00 | 3.18% |
| BUY | retest2 | 2025-03-21 09:30:00 | 7676.00 | 2025-03-28 09:15:00 | 7853.30 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2025-03-21 14:00:00 | 7604.70 | 2025-03-28 09:15:00 | 7853.30 | STOP_HIT | 1.00 | 3.27% |
| SELL | retest2 | 2025-04-01 09:15:00 | 7734.10 | 2025-04-07 09:15:00 | 7347.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 09:15:00 | 7734.10 | 2025-04-07 15:15:00 | 7330.00 | STOP_HIT | 0.50 | 5.22% |
| SELL | retest2 | 2025-04-09 14:15:00 | 7360.50 | 2025-04-11 09:15:00 | 7569.15 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-04-23 09:15:00 | 8344.00 | 2025-05-02 11:15:00 | 8675.00 | STOP_HIT | 1.00 | 3.97% |
| SELL | retest2 | 2025-05-06 10:15:00 | 8498.50 | 2025-05-09 09:15:00 | 8073.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:00:00 | 8498.50 | 2025-05-09 09:15:00 | 8073.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 10:15:00 | 8498.50 | 2025-05-12 09:15:00 | 8466.50 | STOP_HIT | 0.50 | 0.38% |
| SELL | retest2 | 2025-05-06 11:00:00 | 8498.50 | 2025-05-12 09:15:00 | 8466.50 | STOP_HIT | 0.50 | 0.38% |
| SELL | retest2 | 2025-05-08 09:15:00 | 8303.00 | 2025-05-12 11:15:00 | 8545.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-05-12 10:30:00 | 8488.50 | 2025-05-12 11:15:00 | 8545.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-05-16 09:15:00 | 8608.00 | 2025-05-20 13:15:00 | 8502.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-05-16 11:30:00 | 8594.00 | 2025-05-20 13:15:00 | 8502.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-05-23 15:15:00 | 8356.00 | 2025-05-26 09:15:00 | 8455.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-05-27 11:15:00 | 8469.00 | 2025-05-27 15:15:00 | 8405.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-05-30 11:00:00 | 8559.50 | 2025-05-30 14:15:00 | 8430.50 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-05-30 12:15:00 | 8537.50 | 2025-05-30 14:15:00 | 8430.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-13 10:15:00 | 9640.50 | 2025-06-19 09:15:00 | 9459.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-06-16 10:15:00 | 9625.00 | 2025-06-19 09:15:00 | 9459.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-06-16 11:00:00 | 9610.50 | 2025-06-19 09:15:00 | 9459.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-16 12:00:00 | 9627.00 | 2025-06-19 09:15:00 | 9459.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-06-25 12:00:00 | 9190.00 | 2025-07-03 12:15:00 | 9082.50 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2025-06-25 14:15:00 | 9189.50 | 2025-07-03 12:15:00 | 9082.50 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2025-07-08 14:15:00 | 8973.00 | 2025-07-14 09:15:00 | 8524.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 09:15:00 | 8975.00 | 2025-07-14 09:15:00 | 8526.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 10:00:00 | 8972.50 | 2025-07-14 09:15:00 | 8523.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 12:15:00 | 8974.50 | 2025-07-14 09:15:00 | 8525.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 14:15:00 | 8973.00 | 2025-07-14 10:15:00 | 8679.50 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-07-09 09:15:00 | 8975.00 | 2025-07-14 10:15:00 | 8679.50 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-07-09 10:00:00 | 8972.50 | 2025-07-14 10:15:00 | 8679.50 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-07-09 12:15:00 | 8974.50 | 2025-07-14 10:15:00 | 8679.50 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-07-15 09:45:00 | 8700.50 | 2025-07-15 10:15:00 | 8836.50 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-07-30 10:15:00 | 8592.00 | 2025-08-04 15:15:00 | 8570.00 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-07-30 14:00:00 | 8605.00 | 2025-08-04 15:15:00 | 8570.00 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-07-30 14:30:00 | 8601.50 | 2025-08-04 15:15:00 | 8570.00 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-08-19 11:45:00 | 8576.00 | 2025-08-22 13:15:00 | 8666.50 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2025-09-01 12:30:00 | 8420.00 | 2025-09-01 15:15:00 | 8500.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-09-02 09:15:00 | 8407.50 | 2025-09-02 09:15:00 | 8490.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-09-12 13:30:00 | 8973.00 | 2025-09-15 09:15:00 | 8922.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-12 14:15:00 | 8980.00 | 2025-09-15 09:15:00 | 8922.50 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-09-15 09:45:00 | 8970.00 | 2025-09-19 12:15:00 | 9056.00 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-09-15 14:45:00 | 8966.00 | 2025-09-19 12:15:00 | 9056.00 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-09-18 14:15:00 | 9090.00 | 2025-09-19 13:15:00 | 9055.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-09-19 11:15:00 | 9090.00 | 2025-09-19 13:15:00 | 9055.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-22 14:30:00 | 9032.50 | 2025-09-23 09:15:00 | 9102.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-23 11:00:00 | 9053.50 | 2025-09-23 12:15:00 | 9119.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-09-23 11:30:00 | 9045.00 | 2025-09-23 12:15:00 | 9119.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-13 09:15:00 | 9255.50 | 2025-10-13 10:15:00 | 9151.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest1 | 2025-10-15 11:45:00 | 8854.50 | 2025-10-27 09:15:00 | 8679.50 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2025-10-23 13:30:00 | 8575.00 | 2025-10-27 12:15:00 | 8671.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-24 12:45:00 | 8583.00 | 2025-10-27 12:15:00 | 8671.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-10 12:30:00 | 8205.50 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-10 13:30:00 | 8200.00 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-11-10 14:30:00 | 8205.00 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-11 09:30:00 | 8198.00 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-11-11 11:15:00 | 8176.00 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-11-11 14:00:00 | 8184.00 | 2025-11-11 15:15:00 | 8248.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-03 11:00:00 | 8078.00 | 2025-12-04 09:15:00 | 8213.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-12-03 13:45:00 | 8075.50 | 2025-12-04 09:15:00 | 8213.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-12-03 15:15:00 | 8075.00 | 2025-12-04 09:15:00 | 8213.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-10 10:45:00 | 7988.00 | 2025-12-11 09:15:00 | 7588.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 10:45:00 | 7988.00 | 2025-12-11 10:15:00 | 7961.50 | STOP_HIT | 0.50 | 0.33% |
| SELL | retest2 | 2025-12-29 10:15:00 | 7706.00 | 2026-01-02 10:15:00 | 7725.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-27 09:15:00 | 8029.50 | 2026-01-28 13:15:00 | 7955.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-01-27 15:15:00 | 8054.50 | 2026-01-29 09:15:00 | 7816.00 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-01-28 09:45:00 | 8006.00 | 2026-01-29 09:15:00 | 7816.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-01-28 15:00:00 | 8019.00 | 2026-01-29 09:15:00 | 7816.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-02-02 09:15:00 | 7686.50 | 2026-02-03 09:15:00 | 7869.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-02 10:00:00 | 7701.50 | 2026-02-03 09:15:00 | 7869.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-02-02 14:30:00 | 7702.00 | 2026-02-03 09:15:00 | 7869.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-02-11 12:15:00 | 7257.50 | 2026-02-12 10:15:00 | 6894.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 12:15:00 | 7257.50 | 2026-02-13 09:15:00 | 6531.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 6695.00 | 2026-02-26 09:15:00 | 6805.50 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-19 12:30:00 | 6695.00 | 2026-02-26 09:15:00 | 6805.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-03-27 09:15:00 | 7014.50 | 2026-04-16 09:15:00 | 7715.95 | TARGET_HIT | 1.00 | 10.00% |
