# PI Industries Ltd. (PIIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3103.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 247 |
| ALERT1 | 161 |
| ALERT2 | 159 |
| ALERT2_SKIP | 101 |
| ALERT3 | 364 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 132 |
| PARTIAL | 1 |
| TARGET_HIT | 4 |
| STOP_HIT | 131 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 92
- **Target hits / Stop hits / Partials:** 4 / 131 / 1
- **Avg / median % per leg:** -0.01% / -0.71%
- **Sum % (uncompounded):** -1.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 34 | 44.2% | 4 | 73 | 0 | 0.70% | 54.1% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.14% | -0.3% |
| BUY @ 3rd Alert (retest2) | 75 | 34 | 45.3% | 4 | 71 | 0 | 0.72% | 54.3% |
| SELL (all) | 59 | 10 | 16.9% | 0 | 58 | 1 | -0.95% | -55.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.63% | -2.6% |
| SELL @ 3rd Alert (retest2) | 58 | 10 | 17.2% | 0 | 57 | 1 | -0.92% | -53.2% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.97% | -2.9% |
| retest2 (combined) | 133 | 44 | 33.1% | 4 | 128 | 1 | 0.01% | 1.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 14:15:00 | 3318.65 | 3299.97 | 3298.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 10:15:00 | 3346.25 | 3314.03 | 3305.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 13:15:00 | 3423.70 | 3427.83 | 3398.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 3445.00 | 3425.99 | 3404.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 3445.00 | 3425.99 | 3404.22 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 10:15:00 | 3487.00 | 3522.52 | 3523.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 14:15:00 | 3480.90 | 3501.14 | 3511.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 09:15:00 | 3507.90 | 3499.43 | 3509.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 09:15:00 | 3507.90 | 3499.43 | 3509.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 3507.90 | 3499.43 | 3509.04 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 14:15:00 | 3536.45 | 3516.92 | 3514.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 09:15:00 | 3566.00 | 3528.51 | 3520.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 12:15:00 | 3656.00 | 3670.21 | 3631.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 14:15:00 | 3637.90 | 3658.12 | 3632.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 3637.90 | 3658.12 | 3632.38 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 15:15:00 | 3872.00 | 3882.89 | 3883.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 3817.05 | 3869.73 | 3877.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 3830.40 | 3825.98 | 3841.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 14:15:00 | 3850.00 | 3830.79 | 3842.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 3850.00 | 3830.79 | 3842.06 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 3857.75 | 3843.38 | 3842.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 12:15:00 | 3892.80 | 3856.26 | 3848.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 11:15:00 | 3871.25 | 3872.26 | 3861.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 13:15:00 | 3878.15 | 3896.76 | 3886.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 13:15:00 | 3878.15 | 3896.76 | 3886.06 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 11:15:00 | 3856.60 | 3878.95 | 3880.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 12:15:00 | 3838.50 | 3870.86 | 3876.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 3862.20 | 3861.69 | 3869.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 15:15:00 | 3860.00 | 3855.66 | 3862.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 3860.00 | 3855.66 | 3862.35 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 3686.10 | 3641.43 | 3640.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 11:15:00 | 3696.95 | 3652.53 | 3645.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 10:15:00 | 3665.05 | 3671.04 | 3659.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 11:15:00 | 3637.15 | 3664.26 | 3657.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 3637.15 | 3664.26 | 3657.69 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 3624.65 | 3649.51 | 3651.70 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 11:15:00 | 3683.00 | 3652.58 | 3649.59 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 12:15:00 | 3645.00 | 3652.71 | 3653.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 14:15:00 | 3633.95 | 3648.84 | 3651.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 11:15:00 | 3601.20 | 3585.59 | 3604.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 12:15:00 | 3581.65 | 3584.80 | 3602.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 12:15:00 | 3581.65 | 3584.80 | 3602.73 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 3608.00 | 3588.93 | 3586.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 3629.45 | 3597.91 | 3591.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 10:15:00 | 3582.95 | 3599.34 | 3594.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 10:15:00 | 3582.95 | 3599.34 | 3594.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 3582.95 | 3599.34 | 3594.32 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 3790.00 | 3847.68 | 3848.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 10:15:00 | 3758.10 | 3829.77 | 3840.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 3792.80 | 3781.07 | 3805.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 10:15:00 | 3801.65 | 3785.18 | 3805.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 3801.65 | 3785.18 | 3805.57 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 3856.25 | 3819.60 | 3816.21 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 3743.10 | 3813.45 | 3817.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 09:15:00 | 3732.00 | 3765.22 | 3786.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 09:15:00 | 3708.85 | 3703.33 | 3728.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 09:15:00 | 3714.00 | 3699.15 | 3713.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 3714.00 | 3699.15 | 3713.43 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 10:15:00 | 3739.95 | 3669.33 | 3668.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 11:15:00 | 3754.50 | 3686.37 | 3676.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 10:15:00 | 3695.55 | 3703.61 | 3691.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 10:15:00 | 3695.55 | 3703.61 | 3691.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 3695.55 | 3703.61 | 3691.18 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 11:15:00 | 3667.50 | 3688.72 | 3689.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 3639.00 | 3677.39 | 3684.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 15:15:00 | 3637.00 | 3635.05 | 3653.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 09:15:00 | 3614.30 | 3630.90 | 3649.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 3614.30 | 3630.90 | 3649.92 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 11:15:00 | 3648.35 | 3635.68 | 3634.53 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 14:15:00 | 3614.70 | 3632.30 | 3633.41 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 09:15:00 | 3648.80 | 3634.10 | 3633.94 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 13:15:00 | 3627.85 | 3633.63 | 3633.99 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-09-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 14:15:00 | 3652.30 | 3637.36 | 3635.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 15:15:00 | 3656.40 | 3641.17 | 3637.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 13:15:00 | 3647.00 | 3653.06 | 3648.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 13:15:00 | 3647.00 | 3653.06 | 3648.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 13:15:00 | 3647.00 | 3653.06 | 3648.55 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 3624.05 | 3648.35 | 3649.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 13:15:00 | 3609.80 | 3640.64 | 3646.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 3615.95 | 3612.15 | 3627.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 13:15:00 | 3629.95 | 3616.98 | 3627.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 3629.95 | 3616.98 | 3627.33 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 3683.50 | 3636.77 | 3634.34 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 14:15:00 | 3629.95 | 3641.80 | 3641.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 09:15:00 | 3584.45 | 3627.96 | 3635.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 12:15:00 | 3425.00 | 3417.78 | 3449.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 09:15:00 | 3388.90 | 3404.51 | 3420.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 3388.90 | 3404.51 | 3420.00 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 15:15:00 | 3462.00 | 3428.20 | 3424.99 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 3408.00 | 3427.79 | 3429.01 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 3447.95 | 3430.21 | 3429.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 11:15:00 | 3465.90 | 3437.35 | 3433.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 15:15:00 | 3445.00 | 3448.57 | 3440.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 3415.55 | 3441.97 | 3438.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 3415.55 | 3441.97 | 3438.51 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 11:15:00 | 3428.55 | 3434.76 | 3435.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 3400.00 | 3418.06 | 3426.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 3403.65 | 3401.07 | 3411.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 3403.65 | 3401.07 | 3411.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 3403.65 | 3401.07 | 3411.69 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 14:15:00 | 3425.00 | 3406.84 | 3404.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 11:15:00 | 3468.30 | 3426.43 | 3415.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 10:15:00 | 3485.00 | 3487.38 | 3468.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 3487.45 | 3489.09 | 3477.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 3487.45 | 3489.09 | 3477.62 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 3483.70 | 3502.49 | 3503.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 3459.50 | 3490.86 | 3496.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 3350.15 | 3314.62 | 3350.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 3350.15 | 3314.62 | 3350.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 3350.15 | 3314.62 | 3350.02 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 3372.50 | 3340.53 | 3338.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 12:15:00 | 3392.30 | 3363.61 | 3350.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 09:15:00 | 3362.15 | 3375.24 | 3361.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 10:15:00 | 3376.35 | 3375.46 | 3362.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 3376.35 | 3375.46 | 3362.91 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 14:15:00 | 3689.90 | 3704.13 | 3705.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 13:15:00 | 3673.00 | 3689.66 | 3696.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 3703.00 | 3688.01 | 3694.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 3703.00 | 3688.01 | 3694.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 3703.00 | 3688.01 | 3694.07 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 11:15:00 | 3760.95 | 3695.01 | 3689.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 12:15:00 | 3771.50 | 3710.31 | 3697.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 14:15:00 | 3762.50 | 3770.21 | 3744.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 3714.65 | 3755.86 | 3741.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 3714.65 | 3755.86 | 3741.98 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 3830.00 | 3841.87 | 3842.51 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 09:15:00 | 3880.00 | 3849.06 | 3845.64 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 12:15:00 | 3788.25 | 3841.78 | 3848.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 13:15:00 | 3685.15 | 3810.45 | 3834.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 09:15:00 | 3400.00 | 3394.49 | 3481.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 3376.45 | 3400.48 | 3422.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 3376.45 | 3400.48 | 3422.78 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 11:15:00 | 3440.85 | 3417.55 | 3417.52 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 3386.55 | 3414.98 | 3416.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 3369.35 | 3405.85 | 3412.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 09:15:00 | 3398.55 | 3398.06 | 3407.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 11:15:00 | 3417.00 | 3401.50 | 3407.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 11:15:00 | 3417.00 | 3401.50 | 3407.19 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-12-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 14:15:00 | 3425.80 | 3412.59 | 3411.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 10:15:00 | 3439.75 | 3420.62 | 3415.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-22 13:15:00 | 3424.05 | 3425.43 | 3419.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 13:15:00 | 3424.05 | 3425.43 | 3419.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 3424.05 | 3425.43 | 3419.45 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 3464.25 | 3480.03 | 3482.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-04 09:15:00 | 3449.75 | 3468.91 | 3474.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-05 09:15:00 | 3447.60 | 3444.40 | 3456.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 09:15:00 | 3447.60 | 3444.40 | 3456.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 3447.60 | 3444.40 | 3456.29 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 12:15:00 | 3450.75 | 3441.70 | 3440.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 14:15:00 | 3462.60 | 3447.93 | 3443.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 09:15:00 | 3477.00 | 3480.11 | 3467.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 3477.00 | 3480.11 | 3467.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 3477.00 | 3480.11 | 3467.65 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 14:15:00 | 3450.30 | 3460.72 | 3461.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 09:15:00 | 3435.60 | 3453.38 | 3458.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 11:15:00 | 3455.50 | 3453.74 | 3457.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 11:15:00 | 3455.50 | 3453.74 | 3457.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 11:15:00 | 3455.50 | 3453.74 | 3457.48 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 15:15:00 | 3466.90 | 3459.00 | 3458.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 11:15:00 | 3470.35 | 3461.74 | 3460.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 3450.00 | 3466.46 | 3463.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 3450.00 | 3466.46 | 3463.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 3450.00 | 3466.46 | 3463.75 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 10:15:00 | 3428.70 | 3458.91 | 3460.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 15:15:00 | 3413.00 | 3441.80 | 3451.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 3421.75 | 3392.21 | 3413.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 3421.75 | 3392.21 | 3413.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 3421.75 | 3392.21 | 3413.22 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 12:15:00 | 3297.15 | 3286.39 | 3285.20 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 14:15:00 | 3265.30 | 3280.99 | 3282.88 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 3304.50 | 3283.77 | 3283.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 11:15:00 | 3343.00 | 3298.82 | 3290.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 12:15:00 | 3392.00 | 3403.57 | 3374.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 14:15:00 | 3377.75 | 3396.06 | 3376.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 14:15:00 | 3377.75 | 3396.06 | 3376.28 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 11:15:00 | 3320.10 | 3365.42 | 3367.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 3283.20 | 3335.38 | 3351.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 11:15:00 | 3325.35 | 3317.49 | 3336.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 12:15:00 | 3306.45 | 3315.28 | 3333.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 12:15:00 | 3306.45 | 3315.28 | 3333.47 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 09:15:00 | 3415.40 | 3343.13 | 3336.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 10:15:00 | 3449.45 | 3364.39 | 3346.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 09:15:00 | 3416.25 | 3449.10 | 3422.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 3416.25 | 3449.10 | 3422.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 3416.25 | 3449.10 | 3422.76 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 14:15:00 | 3395.15 | 3408.73 | 3409.75 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 09:15:00 | 3443.60 | 3411.78 | 3410.71 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 12:15:00 | 3404.35 | 3409.83 | 3410.32 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 15:15:00 | 3415.00 | 3410.99 | 3410.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 10:15:00 | 3430.00 | 3415.43 | 3412.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 11:15:00 | 3402.00 | 3412.75 | 3411.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 11:15:00 | 3402.00 | 3412.75 | 3411.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 3402.00 | 3412.75 | 3411.73 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 3654.55 | 3679.82 | 3681.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 3644.60 | 3668.46 | 3674.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 3671.05 | 3642.81 | 3653.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 3671.05 | 3642.81 | 3653.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 3671.05 | 3642.81 | 3653.14 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 3666.00 | 3659.68 | 3658.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 3685.05 | 3664.75 | 3661.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 12:15:00 | 3676.25 | 3685.10 | 3676.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 12:15:00 | 3676.25 | 3685.10 | 3676.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 3676.25 | 3685.10 | 3676.40 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 14:15:00 | 3617.30 | 3666.50 | 3669.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 15:15:00 | 3615.00 | 3656.20 | 3664.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 12:15:00 | 3631.90 | 3628.82 | 3640.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 13:15:00 | 3655.25 | 3634.10 | 3641.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 3655.25 | 3634.10 | 3641.94 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 09:15:00 | 3669.95 | 3646.63 | 3646.19 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 15:15:00 | 3632.90 | 3647.23 | 3648.19 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 09:15:00 | 3658.25 | 3649.43 | 3649.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 10:15:00 | 3700.00 | 3659.55 | 3653.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 12:15:00 | 3658.00 | 3664.33 | 3657.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 13:15:00 | 3666.00 | 3664.67 | 3658.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 3666.00 | 3664.67 | 3658.03 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 3627.30 | 3651.93 | 3653.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 3608.00 | 3643.14 | 3649.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 14:15:00 | 3648.55 | 3631.38 | 3641.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 14:15:00 | 3648.55 | 3631.38 | 3641.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 3648.55 | 3631.38 | 3641.27 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 15:15:00 | 3650.00 | 3621.58 | 3619.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 11:15:00 | 3659.60 | 3643.15 | 3634.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 14:15:00 | 3698.15 | 3704.55 | 3682.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 14:15:00 | 3698.15 | 3704.55 | 3682.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 3698.15 | 3704.55 | 3682.97 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 11:15:00 | 3637.50 | 3670.72 | 3672.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 13:15:00 | 3625.50 | 3656.39 | 3665.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 3700.60 | 3658.94 | 3663.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 3700.60 | 3658.94 | 3663.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 3700.60 | 3658.94 | 3663.58 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 3728.95 | 3672.94 | 3669.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 3742.25 | 3693.57 | 3679.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 3762.00 | 3770.95 | 3742.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 14:15:00 | 3820.95 | 3818.38 | 3792.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 3820.95 | 3818.38 | 3792.27 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 13:15:00 | 3831.00 | 3863.38 | 3866.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 14:15:00 | 3811.65 | 3853.03 | 3861.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 10:15:00 | 3874.10 | 3851.60 | 3857.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 10:15:00 | 3874.10 | 3851.60 | 3857.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 3874.10 | 3851.60 | 3857.92 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 3872.05 | 3863.13 | 3862.20 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 10:15:00 | 3860.75 | 3862.01 | 3862.02 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 11:15:00 | 3895.00 | 3868.61 | 3865.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 3910.75 | 3877.04 | 3869.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 13:15:00 | 3906.00 | 3925.89 | 3906.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 13:15:00 | 3906.00 | 3925.89 | 3906.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 3906.00 | 3925.89 | 3906.24 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 10:15:00 | 3851.90 | 3894.30 | 3896.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 15:15:00 | 3830.00 | 3864.96 | 3876.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 3809.95 | 3806.62 | 3831.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 10:00:00 | 3809.95 | 3806.62 | 3831.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 3824.00 | 3810.10 | 3830.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 3824.00 | 3810.10 | 3830.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 3782.10 | 3772.58 | 3792.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 12:45:00 | 3786.45 | 3772.58 | 3792.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 3746.50 | 3726.05 | 3745.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 12:15:00 | 3676.85 | 3708.61 | 3725.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 14:15:00 | 3678.35 | 3700.56 | 3718.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 13:15:00 | 3736.30 | 3718.74 | 3718.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 13:15:00 | 3736.30 | 3718.74 | 3718.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 14:15:00 | 3754.95 | 3725.98 | 3721.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 10:15:00 | 3772.10 | 3774.34 | 3758.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 11:00:00 | 3772.10 | 3774.34 | 3758.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 3745.55 | 3768.58 | 3757.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 12:00:00 | 3745.55 | 3768.58 | 3757.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 12:15:00 | 3779.45 | 3770.75 | 3759.08 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 09:15:00 | 3678.65 | 3741.60 | 3748.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 10:15:00 | 3626.55 | 3718.59 | 3737.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 09:15:00 | 3621.40 | 3613.72 | 3638.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-06 09:30:00 | 3635.05 | 3613.72 | 3638.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 15:15:00 | 3536.95 | 3517.78 | 3541.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:15:00 | 3553.40 | 3517.78 | 3541.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 3564.00 | 3527.02 | 3543.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:45:00 | 3585.45 | 3527.02 | 3543.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 3535.35 | 3528.69 | 3542.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 11:15:00 | 3524.65 | 3528.69 | 3542.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 14:30:00 | 3519.00 | 3532.62 | 3540.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 15:00:00 | 3524.75 | 3532.62 | 3540.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 10:00:00 | 3523.85 | 3529.21 | 3537.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 3541.90 | 3531.75 | 3538.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:00:00 | 3541.90 | 3531.75 | 3538.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 3546.00 | 3534.60 | 3538.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-10 14:15:00 | 3560.00 | 3545.07 | 3543.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-05-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 14:15:00 | 3560.00 | 3545.07 | 3543.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 10:15:00 | 3588.85 | 3557.19 | 3549.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 09:15:00 | 3592.10 | 3603.86 | 3580.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 10:00:00 | 3592.10 | 3603.86 | 3580.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 3638.80 | 3633.75 | 3612.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 3618.00 | 3633.75 | 3612.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 3608.50 | 3640.85 | 3627.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:00:00 | 3608.50 | 3640.85 | 3627.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 3595.20 | 3631.72 | 3624.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:45:00 | 3590.05 | 3631.72 | 3624.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 3589.50 | 3618.31 | 3619.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 11:15:00 | 3539.85 | 3594.22 | 3606.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 14:15:00 | 3586.55 | 3582.00 | 3596.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 15:00:00 | 3586.55 | 3582.00 | 3596.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 3579.75 | 3582.83 | 3594.55 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 3651.10 | 3599.75 | 3595.28 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 3630.05 | 3641.83 | 3643.12 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 09:15:00 | 3694.00 | 3650.96 | 3646.96 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 14:15:00 | 3612.55 | 3646.09 | 3646.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 3605.00 | 3632.74 | 3640.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 10:15:00 | 3636.00 | 3633.39 | 3639.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 10:15:00 | 3636.00 | 3633.39 | 3639.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 3636.00 | 3633.39 | 3639.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:00:00 | 3636.00 | 3633.39 | 3639.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 3635.35 | 3631.00 | 3637.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:00:00 | 3635.35 | 3631.00 | 3637.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 3615.10 | 3627.82 | 3635.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 3585.25 | 3624.59 | 3632.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 10:15:00 | 3676.30 | 3584.22 | 3575.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 3676.30 | 3584.22 | 3575.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 3708.00 | 3631.82 | 3615.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 3650.00 | 3657.21 | 3638.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 10:00:00 | 3650.00 | 3657.21 | 3638.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 3650.00 | 3654.82 | 3640.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:45:00 | 3648.80 | 3654.82 | 3640.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 3650.20 | 3653.90 | 3641.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:00:00 | 3650.20 | 3653.90 | 3641.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 3655.70 | 3653.92 | 3643.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:30:00 | 3652.00 | 3653.92 | 3643.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 3645.00 | 3652.14 | 3643.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 3644.00 | 3652.14 | 3643.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 3628.90 | 3647.49 | 3642.54 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-07 12:15:00 | 3632.90 | 3639.32 | 3639.59 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-06-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 15:15:00 | 3646.00 | 3640.36 | 3639.94 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 09:15:00 | 3567.35 | 3625.76 | 3633.34 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 09:15:00 | 3655.55 | 3636.06 | 3633.80 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 11:15:00 | 3613.10 | 3631.33 | 3632.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 09:15:00 | 3603.30 | 3619.26 | 3625.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 11:15:00 | 3621.20 | 3617.46 | 3623.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 11:15:00 | 3621.20 | 3617.46 | 3623.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 3621.20 | 3617.46 | 3623.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:00:00 | 3621.20 | 3617.46 | 3623.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 3627.00 | 3619.36 | 3623.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:00:00 | 3627.00 | 3619.36 | 3623.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 3634.40 | 3622.37 | 3624.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:00:00 | 3634.40 | 3622.37 | 3624.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 3623.00 | 3625.40 | 3625.90 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 11:15:00 | 3640.40 | 3628.72 | 3627.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 13:15:00 | 3642.15 | 3632.90 | 3629.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 09:15:00 | 3630.55 | 3634.42 | 3631.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 09:15:00 | 3630.55 | 3634.42 | 3631.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 3630.55 | 3634.42 | 3631.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 3637.20 | 3634.42 | 3631.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 3640.05 | 3635.55 | 3632.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:15:00 | 3647.30 | 3635.55 | 3632.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 13:45:00 | 3642.00 | 3634.97 | 3632.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 15:15:00 | 3641.95 | 3634.80 | 3632.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 3803.60 | 3815.64 | 3816.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 10:15:00 | 3803.60 | 3815.64 | 3816.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 3771.00 | 3806.71 | 3811.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 3790.20 | 3778.36 | 3792.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 3790.20 | 3778.36 | 3792.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 3790.20 | 3778.36 | 3792.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:45:00 | 3792.10 | 3778.36 | 3792.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 3780.20 | 3778.73 | 3791.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 3773.00 | 3790.59 | 3791.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 10:45:00 | 3758.20 | 3779.73 | 3786.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 3878.60 | 3789.01 | 3784.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 3878.60 | 3789.01 | 3784.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 3915.00 | 3858.04 | 3839.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 3876.90 | 3882.51 | 3862.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 10:00:00 | 3876.90 | 3882.51 | 3862.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 3855.85 | 3877.18 | 3861.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:45:00 | 3860.25 | 3877.18 | 3861.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 3868.05 | 3875.35 | 3862.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 15:15:00 | 3880.00 | 3864.69 | 3859.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 12:00:00 | 3885.00 | 3879.13 | 3875.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 10:15:00 | 3846.00 | 3883.30 | 3881.39 | SL hit (close<static) qty=1.00 sl=3854.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 3821.15 | 3870.87 | 3875.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 3812.00 | 3846.90 | 3862.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 3862.30 | 3849.98 | 3862.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 3862.30 | 3849.98 | 3862.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 3862.30 | 3849.98 | 3862.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 3862.30 | 3849.98 | 3862.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 3930.85 | 3866.16 | 3868.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 3930.85 | 3866.16 | 3868.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 3959.10 | 3884.74 | 3876.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 12:15:00 | 3998.90 | 3907.58 | 3887.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 3925.00 | 3963.70 | 3934.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 3925.00 | 3963.70 | 3934.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 3925.00 | 3963.70 | 3934.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 3923.05 | 3963.70 | 3934.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 3993.00 | 3969.56 | 3940.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 3941.95 | 3969.56 | 3940.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 3992.45 | 3983.34 | 3959.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:30:00 | 3955.40 | 3983.34 | 3959.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 3946.00 | 3982.75 | 3969.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:15:00 | 4002.05 | 3983.10 | 3970.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 13:45:00 | 3989.95 | 3993.70 | 3979.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-29 10:15:00 | 4402.26 | 4233.30 | 4134.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 4318.50 | 4364.50 | 4370.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 4283.15 | 4340.79 | 4358.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 4288.00 | 4248.41 | 4286.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 4288.00 | 4248.41 | 4286.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 4288.00 | 4248.41 | 4286.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 4314.95 | 4248.41 | 4286.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 4275.00 | 4253.73 | 4285.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:15:00 | 4298.70 | 4253.73 | 4285.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 4287.50 | 4260.49 | 4285.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:30:00 | 4293.70 | 4260.49 | 4285.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 4294.50 | 4267.29 | 4286.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 4261.65 | 4266.16 | 4284.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 4270.00 | 4264.03 | 4281.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 4255.50 | 4264.03 | 4281.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 4357.00 | 4285.98 | 4288.59 | SL hit (close>static) qty=1.00 sl=4305.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 4396.20 | 4308.02 | 4298.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 11:15:00 | 4421.30 | 4330.68 | 4309.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 4448.25 | 4463.65 | 4409.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 4448.25 | 4463.65 | 4409.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 4479.70 | 4486.98 | 4457.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:15:00 | 4446.05 | 4486.98 | 4457.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 4391.70 | 4467.92 | 4451.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:00:00 | 4391.70 | 4467.92 | 4451.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 4389.70 | 4452.28 | 4445.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:45:00 | 4405.55 | 4452.28 | 4445.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 4428.95 | 4443.26 | 4442.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:45:00 | 4438.40 | 4443.26 | 4442.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 15:15:00 | 4431.00 | 4440.81 | 4441.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 4403.70 | 4433.39 | 4438.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 10:15:00 | 4348.35 | 4345.56 | 4369.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 10:45:00 | 4361.55 | 4345.56 | 4369.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 4381.80 | 4352.81 | 4370.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 4381.80 | 4352.81 | 4370.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 4394.15 | 4361.08 | 4373.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:00:00 | 4394.15 | 4361.08 | 4373.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 4417.80 | 4372.42 | 4377.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 4417.80 | 4372.42 | 4377.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 4437.85 | 4385.51 | 4382.66 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 12:15:00 | 4330.10 | 4378.71 | 4382.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 13:15:00 | 4301.35 | 4363.24 | 4374.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 14:15:00 | 4328.30 | 4310.15 | 4335.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 14:15:00 | 4328.30 | 4310.15 | 4335.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 4328.30 | 4310.15 | 4335.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 4328.30 | 4310.15 | 4335.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 4391.65 | 4328.66 | 4339.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:00:00 | 4391.65 | 4328.66 | 4339.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 4371.60 | 4337.25 | 4342.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 11:30:00 | 4351.05 | 4343.60 | 4344.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 13:00:00 | 4344.85 | 4343.85 | 4344.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 13:15:00 | 4366.95 | 4348.47 | 4346.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 4366.95 | 4348.47 | 4346.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 15:15:00 | 4369.25 | 4355.49 | 4350.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 4384.15 | 4408.39 | 4390.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 4384.15 | 4408.39 | 4390.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 4384.15 | 4408.39 | 4390.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 4384.95 | 4408.39 | 4390.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 4396.85 | 4406.08 | 4390.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 14:00:00 | 4416.00 | 4405.67 | 4394.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 13:15:00 | 4409.50 | 4424.02 | 4410.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 14:45:00 | 4414.00 | 4416.42 | 4409.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 4428.50 | 4412.94 | 4408.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 4458.40 | 4422.03 | 4412.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:45:00 | 4467.35 | 4429.78 | 4417.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 12:30:00 | 4464.70 | 4440.68 | 4424.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 13:30:00 | 4466.10 | 4447.54 | 4429.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 10:15:00 | 4476.80 | 4448.25 | 4434.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 4485.75 | 4455.75 | 4438.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 14:15:00 | 4512.00 | 4471.48 | 4450.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 09:30:00 | 4509.15 | 4474.26 | 4466.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 10:45:00 | 4499.00 | 4478.42 | 4469.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 11:15:00 | 4497.85 | 4478.42 | 4469.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 4496.20 | 4485.70 | 4475.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:30:00 | 4475.25 | 4485.70 | 4475.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 4543.25 | 4553.99 | 4532.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:30:00 | 4527.70 | 4553.99 | 4532.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 4540.50 | 4551.29 | 4532.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:00:00 | 4540.50 | 4551.29 | 4532.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 4536.85 | 4548.40 | 4533.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 4536.85 | 4548.40 | 4533.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 4544.00 | 4546.55 | 4535.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-05 09:15:00 | 4519.70 | 4531.35 | 4532.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 4519.70 | 4531.35 | 4532.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 13:15:00 | 4510.50 | 4520.19 | 4526.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 09:15:00 | 4521.00 | 4518.60 | 4523.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 4521.00 | 4518.60 | 4523.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 4521.00 | 4518.60 | 4523.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:15:00 | 4559.65 | 4518.60 | 4523.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 4505.80 | 4516.04 | 4522.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 4496.25 | 4516.04 | 4522.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 11:15:00 | 4572.20 | 4527.27 | 4526.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 12:15:00 | 4624.20 | 4546.66 | 4535.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 15:15:00 | 4598.10 | 4599.35 | 4579.08 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:15:00 | 4635.00 | 4599.35 | 4579.08 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:00:00 | 4629.70 | 4605.42 | 4583.68 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 4626.00 | 4666.85 | 4645.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-11 14:15:00 | 4626.00 | 4666.85 | 4645.20 | SL hit (close<ema400) qty=1.00 sl=4645.20 alert=retest1 |

### Cycle 96 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 4615.25 | 4647.46 | 4648.64 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 11:15:00 | 4657.00 | 4649.78 | 4649.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 12:15:00 | 4693.70 | 4658.56 | 4653.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 09:15:00 | 4665.00 | 4671.07 | 4662.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 4665.00 | 4671.07 | 4662.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 4665.00 | 4671.07 | 4662.08 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 4626.85 | 4651.21 | 4654.49 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 4680.00 | 4653.85 | 4653.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 4701.00 | 4665.07 | 4658.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 14:15:00 | 4710.50 | 4723.19 | 4694.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-20 15:00:00 | 4710.50 | 4723.19 | 4694.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 4733.25 | 4727.16 | 4701.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 10:30:00 | 4756.15 | 4718.15 | 4707.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 12:15:00 | 4662.35 | 4703.44 | 4702.66 | SL hit (close<static) qty=1.00 sl=4690.15 alert=retest2 |

### Cycle 100 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 4625.40 | 4687.83 | 4695.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 4622.20 | 4674.70 | 4688.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 4625.05 | 4620.38 | 4646.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 15:00:00 | 4625.05 | 4620.38 | 4646.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 4619.90 | 4610.54 | 4627.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 4619.90 | 4610.54 | 4627.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 4582.00 | 4604.84 | 4623.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 4636.10 | 4604.84 | 4623.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 4649.30 | 4613.73 | 4625.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 4646.95 | 4613.73 | 4625.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 4670.00 | 4624.98 | 4629.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 4670.00 | 4624.98 | 4629.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 4645.15 | 4631.18 | 4631.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:00:00 | 4645.15 | 4631.18 | 4631.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 4655.35 | 4636.01 | 4633.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 09:15:00 | 4697.70 | 4650.66 | 4640.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 12:15:00 | 4649.20 | 4656.84 | 4646.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 12:15:00 | 4649.20 | 4656.84 | 4646.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 4649.20 | 4656.84 | 4646.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:00:00 | 4649.20 | 4656.84 | 4646.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 4668.60 | 4659.19 | 4648.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 4690.00 | 4660.10 | 4650.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 12:30:00 | 4691.50 | 4696.61 | 4684.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 15:15:00 | 4677.90 | 4680.85 | 4678.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 09:45:00 | 4715.00 | 4682.49 | 4679.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 4676.60 | 4681.32 | 4679.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:15:00 | 4664.40 | 4681.32 | 4679.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 4667.65 | 4678.58 | 4678.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:30:00 | 4668.05 | 4678.58 | 4678.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-04 12:15:00 | 4616.85 | 4666.24 | 4672.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 4616.85 | 4666.24 | 4672.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 13:15:00 | 4597.40 | 4652.47 | 4666.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 4531.95 | 4531.12 | 4578.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 4522.20 | 4531.12 | 4578.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 4551.70 | 4535.24 | 4576.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 4574.95 | 4535.24 | 4576.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 4562.70 | 4543.35 | 4569.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 4566.80 | 4543.35 | 4569.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 4573.50 | 4549.38 | 4570.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 4573.50 | 4549.38 | 4570.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 4560.25 | 4551.55 | 4569.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 4556.00 | 4551.55 | 4569.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 4567.45 | 4554.73 | 4569.07 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 14:15:00 | 4602.90 | 4574.44 | 4574.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 4633.10 | 4589.36 | 4581.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 4588.10 | 4590.65 | 4583.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 11:15:00 | 4588.10 | 4590.65 | 4583.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 4588.10 | 4590.65 | 4583.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:30:00 | 4590.05 | 4590.65 | 4583.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 4570.75 | 4586.67 | 4582.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 4570.75 | 4586.67 | 4582.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 4551.00 | 4579.53 | 4579.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:45:00 | 4546.85 | 4579.53 | 4579.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 4562.05 | 4576.04 | 4577.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 11:15:00 | 4513.20 | 4557.23 | 4568.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 13:15:00 | 4558.00 | 4521.28 | 4536.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 13:15:00 | 4558.00 | 4521.28 | 4536.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 4558.00 | 4521.28 | 4536.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:00:00 | 4558.00 | 4521.28 | 4536.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 4564.75 | 4529.97 | 4539.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 15:15:00 | 4554.60 | 4529.97 | 4539.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 4529.35 | 4531.85 | 4538.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:45:00 | 4542.50 | 4531.85 | 4538.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 4531.45 | 4531.77 | 4537.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 4531.45 | 4531.77 | 4537.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 4537.75 | 4532.97 | 4537.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 4537.75 | 4532.97 | 4537.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 4553.90 | 4537.15 | 4539.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:00:00 | 4553.90 | 4537.15 | 4539.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 4557.35 | 4541.19 | 4540.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 4585.85 | 4550.50 | 4545.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 15:15:00 | 4562.10 | 4569.96 | 4559.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 09:15:00 | 4584.65 | 4569.96 | 4559.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 4553.70 | 4566.71 | 4558.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 4553.70 | 4566.71 | 4558.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 4520.00 | 4557.37 | 4555.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 4520.00 | 4557.37 | 4555.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 4513.35 | 4548.56 | 4551.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 4498.00 | 4538.45 | 4546.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 4551.00 | 4496.84 | 4516.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 4551.00 | 4496.84 | 4516.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 4551.00 | 4496.84 | 4516.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 4551.00 | 4496.84 | 4516.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 4544.90 | 4506.45 | 4518.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 4580.00 | 4506.45 | 4518.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 15:15:00 | 4560.00 | 4532.79 | 4529.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 09:15:00 | 4616.40 | 4549.51 | 4537.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 4531.45 | 4551.10 | 4540.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 11:15:00 | 4531.45 | 4551.10 | 4540.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 4531.45 | 4551.10 | 4540.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:00:00 | 4531.45 | 4551.10 | 4540.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 4489.95 | 4538.87 | 4535.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 4489.95 | 4538.87 | 4535.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 13:15:00 | 4407.55 | 4512.60 | 4524.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 4385.30 | 4443.01 | 4483.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 4373.80 | 4350.61 | 4397.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 12:30:00 | 4365.15 | 4350.61 | 4397.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 4395.35 | 4359.56 | 4397.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 4400.10 | 4359.56 | 4397.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 4351.45 | 4357.93 | 4393.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:30:00 | 4332.90 | 4356.18 | 4386.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 4329.80 | 4350.90 | 4381.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:30:00 | 4335.30 | 4323.65 | 4352.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:30:00 | 4320.80 | 4327.50 | 4339.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 4353.30 | 4332.66 | 4340.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:30:00 | 4312.35 | 4332.32 | 4338.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 15:15:00 | 4359.85 | 4341.57 | 4341.68 | SL hit (close>static) qty=1.00 sl=4358.60 alert=retest2 |

### Cycle 109 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 4355.40 | 4343.38 | 4341.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 4383.20 | 4351.34 | 4345.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 4468.35 | 4476.79 | 4443.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 12:15:00 | 4468.35 | 4476.79 | 4443.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 4468.35 | 4476.79 | 4443.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 4441.35 | 4476.79 | 4443.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 4466.40 | 4474.71 | 4445.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:45:00 | 4474.65 | 4478.72 | 4449.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 10:00:00 | 4471.40 | 4490.63 | 4465.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:30:00 | 4493.45 | 4482.48 | 4465.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:15:00 | 4490.00 | 4478.03 | 4465.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 4457.20 | 4473.87 | 4464.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:00:00 | 4457.20 | 4473.87 | 4464.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 4464.25 | 4471.94 | 4464.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 4481.15 | 4471.65 | 4465.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:30:00 | 4502.25 | 4478.37 | 4470.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 15:15:00 | 4561.00 | 4583.49 | 4585.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 4561.00 | 4583.49 | 4585.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 4503.90 | 4567.57 | 4577.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 11:15:00 | 4595.90 | 4570.63 | 4577.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 11:15:00 | 4595.90 | 4570.63 | 4577.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 4595.90 | 4570.63 | 4577.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:00:00 | 4595.90 | 4570.63 | 4577.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 4583.05 | 4573.11 | 4577.91 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 13:15:00 | 4614.60 | 4581.41 | 4581.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 09:15:00 | 4621.05 | 4593.79 | 4587.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 11:15:00 | 4575.00 | 4593.35 | 4588.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 11:15:00 | 4575.00 | 4593.35 | 4588.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 4575.00 | 4593.35 | 4588.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:45:00 | 4570.20 | 4593.35 | 4588.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 4562.30 | 4587.14 | 4586.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:00:00 | 4562.30 | 4587.14 | 4586.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 4559.70 | 4581.65 | 4583.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 4436.55 | 4538.66 | 4562.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 15:15:00 | 4499.15 | 4474.18 | 4511.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 09:15:00 | 4097.00 | 4474.18 | 4511.90 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 4140.85 | 4140.89 | 4174.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 10:15:00 | 4126.35 | 4140.89 | 4174.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 12:00:00 | 4120.70 | 4129.62 | 4163.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 4204.85 | 4143.12 | 4154.28 | SL hit (close>ema400) qty=1.00 sl=4154.28 alert=retest1 |

### Cycle 113 — BUY (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 15:15:00 | 4105.00 | 4080.78 | 4079.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 4132.70 | 4094.92 | 4086.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 4124.35 | 4127.59 | 4112.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 4124.35 | 4127.59 | 4112.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 4124.35 | 4127.59 | 4112.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 4124.35 | 4127.59 | 4112.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 4189.10 | 4139.89 | 4119.66 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 10:15:00 | 4110.75 | 4131.08 | 4131.51 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 15:15:00 | 4140.00 | 4131.54 | 4130.85 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 4110.25 | 4127.28 | 4128.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 11:15:00 | 4090.95 | 4116.92 | 4123.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 10:15:00 | 4086.65 | 4086.11 | 4101.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 10:15:00 | 4086.65 | 4086.11 | 4101.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 4086.65 | 4086.11 | 4101.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:30:00 | 4095.00 | 4086.11 | 4101.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 4099.55 | 4088.80 | 4101.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:45:00 | 4094.15 | 4088.80 | 4101.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 4106.50 | 4092.34 | 4102.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:45:00 | 4108.95 | 4092.34 | 4102.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 4098.30 | 4093.53 | 4101.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:15:00 | 4098.65 | 4093.53 | 4101.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 4124.70 | 4099.76 | 4103.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 4124.70 | 4099.76 | 4103.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 4137.00 | 4107.21 | 4106.91 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 12:15:00 | 4100.00 | 4106.03 | 4106.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 4017.40 | 4088.78 | 4098.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 4039.25 | 4038.00 | 4059.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 4039.25 | 4038.00 | 4059.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 3891.10 | 3872.73 | 3893.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 3863.30 | 3872.73 | 3893.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 3862.65 | 3870.71 | 3890.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:30:00 | 3855.00 | 3863.04 | 3882.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 09:15:00 | 3662.25 | 3696.79 | 3718.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-01 13:15:00 | 3691.65 | 3688.70 | 3706.55 | SL hit (close>ema200) qty=0.50 sl=3688.70 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 3730.00 | 3703.49 | 3703.25 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 09:15:00 | 3690.45 | 3700.88 | 3702.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 12:15:00 | 3660.20 | 3691.02 | 3697.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 09:15:00 | 3680.25 | 3677.33 | 3687.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-06 10:00:00 | 3680.25 | 3677.33 | 3687.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 3633.40 | 3668.55 | 3682.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:15:00 | 3629.85 | 3668.55 | 3682.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 3625.00 | 3616.95 | 3628.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:15:00 | 3625.25 | 3619.63 | 3628.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 14:30:00 | 3629.05 | 3624.17 | 3629.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 3635.00 | 3626.34 | 3629.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 3677.00 | 3626.34 | 3629.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 3718.00 | 3644.67 | 3637.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 3718.00 | 3644.67 | 3637.74 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 3615.00 | 3638.07 | 3640.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 3612.50 | 3632.96 | 3637.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 3494.15 | 3474.96 | 3529.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 3494.15 | 3474.96 | 3529.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 3452.40 | 3459.51 | 3481.46 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 09:15:00 | 3543.00 | 3484.09 | 3482.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 10:15:00 | 3561.15 | 3499.50 | 3489.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 3548.75 | 3571.62 | 3548.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 3548.75 | 3571.62 | 3548.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 3548.75 | 3571.62 | 3548.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 3548.75 | 3571.62 | 3548.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 3532.75 | 3563.85 | 3547.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 3532.75 | 3563.85 | 3547.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 3540.80 | 3559.24 | 3546.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:15:00 | 3539.20 | 3559.24 | 3546.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 3545.40 | 3556.47 | 3546.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:15:00 | 3537.95 | 3556.47 | 3546.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 3530.25 | 3551.23 | 3544.96 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 3521.00 | 3537.76 | 3539.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 3493.10 | 3523.92 | 3532.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 3541.35 | 3513.34 | 3521.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 3541.35 | 3513.34 | 3521.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 3541.35 | 3513.34 | 3521.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 3541.35 | 3513.34 | 3521.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 3559.25 | 3522.53 | 3524.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 3559.25 | 3522.53 | 3524.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 3528.35 | 3523.43 | 3524.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:45:00 | 3521.40 | 3523.43 | 3524.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 3527.75 | 3524.29 | 3524.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 3529.45 | 3524.29 | 3524.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 3526.35 | 3524.71 | 3525.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:30:00 | 3521.60 | 3524.71 | 3525.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 3527.10 | 3525.18 | 3525.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 3529.15 | 3525.18 | 3525.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 3499.30 | 3520.01 | 3522.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 3440.00 | 3510.75 | 3516.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 3463.20 | 3415.04 | 3414.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 3463.20 | 3415.04 | 3414.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 3484.10 | 3451.95 | 3438.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 3480.00 | 3487.47 | 3466.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 3480.00 | 3487.47 | 3466.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 3480.00 | 3487.47 | 3466.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 3472.10 | 3487.47 | 3466.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 3440.65 | 3478.10 | 3463.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 3459.90 | 3478.10 | 3463.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 3477.50 | 3477.98 | 3465.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:15:00 | 3499.10 | 3478.06 | 3466.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:30:00 | 3535.05 | 3615.42 | 3604.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 11:15:00 | 3510.00 | 3583.78 | 3591.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 3510.00 | 3583.78 | 3591.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 3496.40 | 3548.08 | 3567.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 11:15:00 | 3135.85 | 3128.32 | 3173.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 11:45:00 | 3130.40 | 3128.32 | 3173.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 3168.15 | 3142.85 | 3169.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 3168.15 | 3142.85 | 3169.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 3185.90 | 3153.31 | 3169.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 3185.90 | 3153.31 | 3169.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 3183.65 | 3159.38 | 3171.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 3189.20 | 3159.38 | 3171.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 3191.65 | 3165.83 | 3172.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:30:00 | 3194.30 | 3165.83 | 3172.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 13:15:00 | 3191.15 | 3172.46 | 3174.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 14:00:00 | 3191.15 | 3172.46 | 3174.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 3206.20 | 3179.21 | 3177.64 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 13:15:00 | 3174.05 | 3179.10 | 3179.12 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 3185.10 | 3180.30 | 3179.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 15:15:00 | 3203.40 | 3184.92 | 3181.82 | Break + close above crossover candle high |

### Cycle 130 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 3139.05 | 3175.75 | 3177.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 3098.45 | 3141.91 | 3157.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 3113.50 | 3108.42 | 3128.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 10:00:00 | 3113.50 | 3108.42 | 3128.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 2958.75 | 3000.86 | 3029.08 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 3036.10 | 3021.14 | 3019.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 3092.00 | 3038.97 | 3027.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 3170.90 | 3198.06 | 3148.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 3170.90 | 3198.06 | 3148.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 3170.90 | 3198.06 | 3148.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 3198.15 | 3198.06 | 3148.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:00:00 | 3220.25 | 3209.24 | 3170.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-24 10:15:00 | 3517.97 | 3465.08 | 3438.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 3446.15 | 3451.73 | 3452.06 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 12:15:00 | 3454.95 | 3452.72 | 3452.47 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 3432.85 | 3448.74 | 3450.69 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 3479.15 | 3454.98 | 3452.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 3500.00 | 3469.32 | 3460.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 09:15:00 | 3457.55 | 3466.97 | 3460.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 3457.55 | 3466.97 | 3460.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 3457.55 | 3466.97 | 3460.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 3457.55 | 3466.97 | 3460.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 3462.90 | 3466.15 | 3460.68 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 3422.65 | 3456.14 | 3457.00 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 11:15:00 | 3461.05 | 3454.75 | 3454.35 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 3437.00 | 3452.95 | 3454.18 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 3501.50 | 3462.19 | 3457.73 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 11:15:00 | 3442.60 | 3456.91 | 3456.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 3348.65 | 3429.30 | 3443.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 3288.85 | 3268.05 | 3308.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 3288.85 | 3268.05 | 3308.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 3298.30 | 3278.74 | 3306.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 3303.00 | 3278.74 | 3306.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 3298.00 | 3268.92 | 3282.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 15:00:00 | 3298.00 | 3268.92 | 3282.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 3282.35 | 3271.60 | 3282.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:15:00 | 3450.00 | 3271.60 | 3282.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 3478.00 | 3312.88 | 3300.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 3556.50 | 3395.06 | 3341.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 11:15:00 | 3633.00 | 3633.60 | 3590.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 11:45:00 | 3634.40 | 3633.60 | 3590.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 3670.00 | 3690.74 | 3670.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 3670.00 | 3690.74 | 3670.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 3655.60 | 3683.71 | 3669.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 3653.40 | 3683.71 | 3669.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 3644.60 | 3675.89 | 3667.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 3644.60 | 3675.89 | 3667.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 09:15:00 | 3626.60 | 3655.49 | 3659.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 3585.20 | 3637.87 | 3647.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 3635.10 | 3601.81 | 3617.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 3635.10 | 3601.81 | 3617.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 3635.10 | 3601.81 | 3617.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 3635.10 | 3601.81 | 3617.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 3621.10 | 3605.67 | 3618.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 3621.10 | 3605.67 | 3618.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 3648.80 | 3614.30 | 3620.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:00:00 | 3648.80 | 3614.30 | 3620.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 3647.40 | 3620.92 | 3623.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 3650.30 | 3620.92 | 3623.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 3653.70 | 3627.47 | 3626.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 3658.40 | 3640.31 | 3632.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 12:15:00 | 3632.00 | 3643.42 | 3636.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 12:15:00 | 3632.00 | 3643.42 | 3636.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 3632.00 | 3643.42 | 3636.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:00:00 | 3632.00 | 3643.42 | 3636.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 3630.00 | 3640.74 | 3636.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:15:00 | 3623.00 | 3640.74 | 3636.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 3645.00 | 3644.19 | 3638.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 12:00:00 | 3683.00 | 3649.60 | 3643.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 13:15:00 | 3637.30 | 3674.49 | 3676.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 3637.30 | 3674.49 | 3676.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 3636.40 | 3666.87 | 3673.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 12:15:00 | 3646.80 | 3639.82 | 3654.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 12:15:00 | 3646.80 | 3639.82 | 3654.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 3646.80 | 3639.82 | 3654.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 3658.60 | 3639.82 | 3654.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 3656.20 | 3643.10 | 3654.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 3608.60 | 3647.47 | 3654.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 10:30:00 | 3634.70 | 3646.61 | 3653.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 11:15:00 | 3662.90 | 3649.87 | 3653.94 | SL hit (close>static) qty=1.00 sl=3660.30 alert=retest2 |

### Cycle 145 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3712.10 | 3651.76 | 3647.00 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 14:15:00 | 3637.00 | 3656.31 | 3658.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 15:15:00 | 3630.10 | 3651.07 | 3656.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 14:15:00 | 3651.70 | 3643.35 | 3649.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 14:15:00 | 3651.70 | 3643.35 | 3649.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 3651.70 | 3643.35 | 3649.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 15:00:00 | 3651.70 | 3643.35 | 3649.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 3658.10 | 3646.30 | 3649.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:15:00 | 3650.00 | 3646.30 | 3649.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 3650.00 | 3647.04 | 3649.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 11:15:00 | 3640.40 | 3645.95 | 3649.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 3674.70 | 3652.38 | 3651.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 3674.70 | 3652.38 | 3651.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 15:15:00 | 3694.20 | 3664.83 | 3657.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 3746.50 | 3762.62 | 3743.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 14:00:00 | 3746.50 | 3762.62 | 3743.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 3728.80 | 3755.85 | 3741.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:15:00 | 3734.00 | 3755.85 | 3741.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 3734.00 | 3751.48 | 3741.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 3625.30 | 3751.48 | 3741.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 3701.30 | 3730.33 | 3732.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 3656.00 | 3715.46 | 3725.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 3705.30 | 3700.96 | 3715.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 3705.30 | 3700.96 | 3715.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 3705.30 | 3701.83 | 3714.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 3670.00 | 3701.83 | 3714.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 3692.00 | 3682.66 | 3681.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 3692.00 | 3682.66 | 3681.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 3715.40 | 3689.79 | 3685.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 3841.40 | 3854.90 | 3825.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:45:00 | 3843.40 | 3854.90 | 3825.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 3833.00 | 3850.52 | 3826.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 3833.00 | 3850.52 | 3826.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 3834.20 | 3845.99 | 3830.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 3834.20 | 3845.99 | 3830.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 3830.00 | 3842.79 | 3830.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 3817.50 | 3842.79 | 3830.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 3789.10 | 3832.05 | 3826.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:45:00 | 3797.50 | 3832.05 | 3826.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 10:15:00 | 3775.00 | 3820.64 | 3821.70 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 3874.30 | 3830.14 | 3824.81 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 3814.60 | 3824.02 | 3824.11 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 3866.00 | 3831.13 | 3827.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 3949.90 | 3854.88 | 3838.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 10:15:00 | 3924.80 | 3934.26 | 3898.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 10:45:00 | 3923.20 | 3934.26 | 3898.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 3903.50 | 3923.45 | 3901.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 3903.50 | 3923.45 | 3901.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 3904.90 | 3919.74 | 3902.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:30:00 | 3910.70 | 3919.74 | 3902.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 3907.40 | 3917.27 | 3902.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 3920.10 | 3917.27 | 3902.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:00:00 | 3917.80 | 3917.78 | 3905.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 3882.10 | 3910.65 | 3903.36 | SL hit (close<static) qty=1.00 sl=3895.10 alert=retest2 |

### Cycle 154 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 4079.20 | 4117.79 | 4121.02 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 4154.10 | 4120.06 | 4118.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 4167.70 | 4139.30 | 4128.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 4177.30 | 4188.77 | 4166.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 13:00:00 | 4177.30 | 4188.77 | 4166.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 4180.00 | 4187.02 | 4167.76 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 4113.00 | 4158.65 | 4160.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 10:15:00 | 4084.90 | 4128.33 | 4140.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 15:15:00 | 4112.00 | 4105.66 | 4122.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:15:00 | 4149.10 | 4105.66 | 4122.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 4148.90 | 4114.31 | 4124.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:15:00 | 4158.10 | 4114.31 | 4124.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 4159.10 | 4123.27 | 4128.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 4154.60 | 4123.27 | 4128.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 12:15:00 | 4152.40 | 4134.65 | 4132.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 4168.90 | 4144.25 | 4139.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 10:15:00 | 4209.00 | 4222.76 | 4195.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 4209.00 | 4222.76 | 4195.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 4207.70 | 4215.88 | 4198.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 4197.00 | 4215.88 | 4198.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 4185.00 | 4210.52 | 4200.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 4185.00 | 4210.52 | 4200.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 4168.60 | 4202.14 | 4197.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 4168.60 | 4202.14 | 4197.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 4145.00 | 4190.71 | 4192.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 12:15:00 | 4131.00 | 4156.60 | 4171.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 4169.60 | 4155.98 | 4165.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 4169.60 | 4155.98 | 4165.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 4169.60 | 4155.98 | 4165.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 4169.60 | 4155.98 | 4165.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 4187.30 | 4162.24 | 4167.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 4187.30 | 4162.24 | 4167.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 4174.40 | 4164.68 | 4168.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 4164.80 | 4166.95 | 4168.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 4101.70 | 4167.36 | 4168.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 4139.20 | 4077.84 | 4075.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 4139.20 | 4077.84 | 4075.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 4145.00 | 4101.26 | 4087.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 4131.00 | 4133.78 | 4109.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:00:00 | 4131.00 | 4133.78 | 4109.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 4146.20 | 4133.58 | 4113.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 4150.00 | 4133.58 | 4113.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 14:15:00 | 4107.90 | 4126.98 | 4115.57 | SL hit (close<static) qty=1.00 sl=4111.60 alert=retest2 |

### Cycle 160 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 4102.50 | 4127.92 | 4130.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 4044.70 | 4089.81 | 4108.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 4080.00 | 4065.14 | 4086.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 4080.00 | 4065.14 | 4086.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 4066.10 | 4065.33 | 4084.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 4081.50 | 4065.33 | 4084.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 4072.70 | 4066.81 | 4083.64 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 4118.00 | 4090.47 | 4089.77 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 4060.00 | 4085.90 | 4087.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 4037.80 | 4076.28 | 4083.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 13:15:00 | 4103.20 | 4080.38 | 4083.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 13:15:00 | 4103.20 | 4080.38 | 4083.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 4103.20 | 4080.38 | 4083.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 4103.20 | 4080.38 | 4083.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 4095.50 | 4083.41 | 4084.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 4099.90 | 4083.41 | 4084.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 15:15:00 | 4105.00 | 4087.73 | 4086.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 09:15:00 | 4105.80 | 4091.34 | 4088.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 14:15:00 | 4113.50 | 4114.53 | 4102.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 4113.50 | 4114.53 | 4102.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 4086.40 | 4108.90 | 4101.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 4149.30 | 4108.90 | 4101.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 4164.00 | 4119.92 | 4107.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 10:30:00 | 4213.00 | 4139.54 | 4117.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:00:00 | 4212.70 | 4248.01 | 4227.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:45:00 | 4212.00 | 4228.85 | 4227.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 4200.10 | 4223.10 | 4225.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 4200.10 | 4223.10 | 4225.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 4192.60 | 4217.00 | 4222.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 4142.20 | 4140.57 | 4172.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 15:00:00 | 4142.20 | 4140.57 | 4172.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 3754.90 | 3793.95 | 3835.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:15:00 | 3725.90 | 3771.19 | 3813.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:15:00 | 3738.30 | 3733.37 | 3774.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 11:15:00 | 3785.80 | 3770.08 | 3769.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 11:15:00 | 3785.80 | 3770.08 | 3769.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 3813.40 | 3778.74 | 3773.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 15:15:00 | 3889.50 | 3890.54 | 3867.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 09:15:00 | 3817.80 | 3890.54 | 3867.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3826.90 | 3877.81 | 3864.19 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 3797.00 | 3847.67 | 3853.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 3785.30 | 3835.20 | 3846.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 3725.10 | 3713.78 | 3744.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 3725.10 | 3713.78 | 3744.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 3747.00 | 3720.58 | 3740.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 3747.00 | 3720.58 | 3740.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 3762.40 | 3728.94 | 3742.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 3759.70 | 3728.94 | 3742.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 3769.30 | 3747.96 | 3748.66 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 3766.10 | 3751.59 | 3750.25 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 3734.60 | 3748.21 | 3749.88 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 3754.50 | 3749.24 | 3749.09 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 3729.60 | 3746.59 | 3748.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 3717.30 | 3736.79 | 3742.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 3706.10 | 3698.00 | 3717.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 10:00:00 | 3706.10 | 3698.00 | 3717.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 3715.60 | 3701.52 | 3717.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 3715.60 | 3701.52 | 3717.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 3716.80 | 3704.58 | 3717.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:45:00 | 3717.10 | 3704.58 | 3717.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 3704.20 | 3704.50 | 3715.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 3687.00 | 3703.30 | 3713.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:45:00 | 3690.60 | 3685.47 | 3696.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 3733.10 | 3698.60 | 3700.96 | SL hit (close>static) qty=1.00 sl=3717.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 3735.00 | 3705.88 | 3704.05 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 3697.70 | 3702.39 | 3702.78 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 3710.00 | 3704.36 | 3703.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 3712.20 | 3705.93 | 3704.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 3725.00 | 3737.53 | 3727.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 12:15:00 | 3725.00 | 3737.53 | 3727.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 3725.00 | 3737.53 | 3727.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 3731.40 | 3737.53 | 3727.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 3722.00 | 3734.42 | 3726.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:00:00 | 3722.00 | 3734.42 | 3726.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 3701.00 | 3727.74 | 3724.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 3701.00 | 3727.74 | 3724.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 3718.00 | 3722.22 | 3722.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 11:15:00 | 3706.30 | 3719.04 | 3720.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 3722.20 | 3719.67 | 3721.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 12:15:00 | 3722.20 | 3719.67 | 3721.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 3722.20 | 3719.67 | 3721.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 3722.20 | 3719.67 | 3721.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 3723.30 | 3720.40 | 3721.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 3720.00 | 3720.40 | 3721.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 3720.40 | 3720.40 | 3721.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 3721.40 | 3720.40 | 3721.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 3729.30 | 3722.18 | 3721.89 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 09:15:00 | 3710.00 | 3719.74 | 3720.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 11:15:00 | 3707.00 | 3717.38 | 3719.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 15:15:00 | 3715.00 | 3714.76 | 3717.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 15:15:00 | 3715.00 | 3714.76 | 3717.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 3715.00 | 3714.76 | 3717.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 3732.60 | 3714.76 | 3717.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 3774.40 | 3726.69 | 3722.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 3791.00 | 3739.55 | 3728.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 3750.60 | 3750.69 | 3737.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 3750.60 | 3750.69 | 3737.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 3723.80 | 3748.23 | 3739.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 3723.80 | 3748.23 | 3739.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 3709.00 | 3740.38 | 3736.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 3709.00 | 3740.38 | 3736.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 3687.00 | 3729.70 | 3732.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 3654.50 | 3714.66 | 3725.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 3722.90 | 3709.34 | 3718.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 3722.90 | 3709.34 | 3718.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 3722.90 | 3709.34 | 3718.56 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 3752.80 | 3729.12 | 3726.22 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 3696.10 | 3724.54 | 3726.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 11:15:00 | 3691.10 | 3717.85 | 3722.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 3625.90 | 3623.20 | 3647.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:15:00 | 3639.70 | 3623.20 | 3647.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 3619.30 | 3623.62 | 3641.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:30:00 | 3598.00 | 3615.42 | 3633.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 3568.70 | 3532.69 | 3528.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 3568.70 | 3532.69 | 3528.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 3574.10 | 3547.23 | 3536.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 3593.70 | 3616.42 | 3601.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 3593.70 | 3616.42 | 3601.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 3593.70 | 3616.42 | 3601.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 3593.70 | 3616.42 | 3601.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 3580.50 | 3609.23 | 3599.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 3590.40 | 3609.23 | 3599.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 3575.10 | 3602.41 | 3597.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:45:00 | 3575.00 | 3602.41 | 3597.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 3588.00 | 3594.19 | 3594.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 3555.00 | 3586.35 | 3590.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 3533.90 | 3531.94 | 3555.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:00:00 | 3533.90 | 3531.94 | 3555.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 3535.10 | 3530.34 | 3547.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:45:00 | 3531.40 | 3530.34 | 3547.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3563.00 | 3537.11 | 3546.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 3563.00 | 3537.11 | 3546.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 3568.80 | 3543.44 | 3548.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 3568.80 | 3543.44 | 3548.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 3568.60 | 3554.45 | 3552.82 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 3532.20 | 3550.14 | 3552.20 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 10:15:00 | 3562.90 | 3553.32 | 3552.90 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 3542.60 | 3552.37 | 3552.71 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 14:15:00 | 3555.30 | 3552.96 | 3552.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 3563.20 | 3555.43 | 3554.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 11:15:00 | 3555.00 | 3555.35 | 3554.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 11:15:00 | 3555.00 | 3555.35 | 3554.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 3555.00 | 3555.35 | 3554.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 3555.00 | 3555.35 | 3554.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 3556.90 | 3555.49 | 3554.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 3557.80 | 3555.49 | 3554.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 3556.00 | 3555.59 | 3554.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:30:00 | 3557.30 | 3555.59 | 3554.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 3555.00 | 3555.47 | 3554.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 3570.40 | 3555.47 | 3554.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 3543.70 | 3554.16 | 3554.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 10:15:00 | 3543.70 | 3554.16 | 3554.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 09:15:00 | 3525.40 | 3545.17 | 3549.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 11:15:00 | 3570.10 | 3549.21 | 3550.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 11:15:00 | 3570.10 | 3549.21 | 3550.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 3570.10 | 3549.21 | 3550.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:30:00 | 3565.60 | 3549.21 | 3550.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 12:15:00 | 3587.80 | 3556.93 | 3553.95 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 3535.30 | 3554.01 | 3556.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 15:15:00 | 3532.10 | 3549.63 | 3554.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 3586.70 | 3552.03 | 3553.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 3586.70 | 3552.03 | 3553.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 3586.70 | 3552.03 | 3553.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 3586.70 | 3552.03 | 3553.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 3566.70 | 3554.97 | 3554.95 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 3539.00 | 3553.21 | 3555.15 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 11:15:00 | 3575.10 | 3555.69 | 3555.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 14:15:00 | 3582.00 | 3563.94 | 3559.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 14:15:00 | 3606.90 | 3613.57 | 3591.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 3606.90 | 3613.57 | 3591.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 3595.10 | 3608.01 | 3593.00 | EMA400 retest candle locked (from upside) |

### Cycle 194 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 3584.60 | 3593.79 | 3593.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 3571.20 | 3588.39 | 3591.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 14:15:00 | 3579.10 | 3576.35 | 3583.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:30:00 | 3576.10 | 3576.35 | 3583.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 3578.50 | 3576.78 | 3583.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 3622.20 | 3576.78 | 3583.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 3680.40 | 3597.51 | 3592.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 3693.60 | 3630.98 | 3609.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 3654.00 | 3659.87 | 3634.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 11:15:00 | 3650.00 | 3659.87 | 3634.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 3630.10 | 3652.11 | 3635.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 3630.10 | 3652.11 | 3635.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 3648.00 | 3651.29 | 3636.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:30:00 | 3636.20 | 3651.29 | 3636.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 3764.70 | 3785.10 | 3765.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 3764.70 | 3785.10 | 3765.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 3764.70 | 3781.02 | 3765.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:15:00 | 3781.70 | 3781.02 | 3765.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 14:15:00 | 3794.20 | 3779.82 | 3766.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 3702.70 | 3764.92 | 3762.94 | SL hit (close<static) qty=1.00 sl=3750.00 alert=retest2 |

### Cycle 196 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 3712.00 | 3754.34 | 3758.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 11:15:00 | 3632.00 | 3729.87 | 3746.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 3563.90 | 3558.72 | 3594.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 14:30:00 | 3556.60 | 3558.72 | 3594.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 3454.20 | 3446.28 | 3476.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 3466.60 | 3446.28 | 3476.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 3430.30 | 3418.65 | 3430.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 3443.70 | 3423.66 | 3431.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 3433.60 | 3425.65 | 3431.91 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 3439.80 | 3432.30 | 3431.80 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 3422.00 | 3431.05 | 3431.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 3395.20 | 3418.18 | 3424.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 3391.40 | 3386.54 | 3401.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 3391.40 | 3386.54 | 3401.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 3370.00 | 3370.39 | 3382.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 3390.80 | 3370.39 | 3382.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 3376.20 | 3371.55 | 3382.20 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 3395.50 | 3387.21 | 3386.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 3399.00 | 3390.38 | 3388.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 14:15:00 | 3396.90 | 3407.04 | 3400.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 3396.90 | 3407.04 | 3400.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 3396.90 | 3407.04 | 3400.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 3384.70 | 3407.04 | 3400.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 3395.40 | 3404.71 | 3400.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 3376.80 | 3404.71 | 3400.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 3360.90 | 3395.95 | 3396.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 3349.80 | 3386.72 | 3392.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 3335.00 | 3331.82 | 3352.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:00:00 | 3335.00 | 3331.82 | 3352.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 3353.00 | 3339.08 | 3350.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 3380.60 | 3339.08 | 3350.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 3405.60 | 3352.39 | 3355.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 3405.60 | 3352.39 | 3355.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 3379.30 | 3357.77 | 3358.00 | EMA400 retest candle locked (from downside) |

### Cycle 201 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 3393.10 | 3364.84 | 3361.19 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 3338.80 | 3357.90 | 3359.40 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 3370.90 | 3360.87 | 3360.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 3375.00 | 3363.70 | 3361.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 11:15:00 | 3317.00 | 3368.46 | 3366.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 11:15:00 | 3317.00 | 3368.46 | 3366.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 3317.00 | 3368.46 | 3366.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:45:00 | 3321.90 | 3368.46 | 3366.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 3293.70 | 3353.51 | 3360.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 13:15:00 | 3273.30 | 3337.47 | 3352.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 3262.50 | 3252.46 | 3277.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 3262.50 | 3252.46 | 3277.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 3210.20 | 3211.09 | 3224.76 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 3258.00 | 3232.51 | 3230.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 3262.90 | 3243.03 | 3236.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 3254.10 | 3258.04 | 3248.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 3254.10 | 3258.04 | 3248.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 3254.10 | 3258.04 | 3248.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 3249.00 | 3258.04 | 3248.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 3246.30 | 3255.69 | 3248.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 3246.30 | 3255.69 | 3248.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 3250.00 | 3254.55 | 3248.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 3256.00 | 3246.78 | 3246.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 3223.40 | 3242.10 | 3244.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 3223.40 | 3242.10 | 3244.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 3215.00 | 3233.35 | 3239.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 11:15:00 | 3247.00 | 3231.47 | 3236.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 11:15:00 | 3247.00 | 3231.47 | 3236.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 3247.00 | 3231.47 | 3236.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 3247.00 | 3231.47 | 3236.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 3260.70 | 3237.31 | 3238.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:45:00 | 3261.40 | 3237.31 | 3238.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 3256.80 | 3241.21 | 3240.49 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 3232.00 | 3238.68 | 3239.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 3228.20 | 3236.58 | 3238.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 3234.00 | 3233.61 | 3236.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 3234.00 | 3233.61 | 3236.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 3234.00 | 3233.61 | 3236.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 3234.00 | 3233.61 | 3236.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 3242.50 | 3235.38 | 3236.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 3242.50 | 3235.38 | 3236.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 3231.70 | 3234.65 | 3236.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:45:00 | 3221.20 | 3232.38 | 3235.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 3215.80 | 3232.38 | 3235.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:00:00 | 3214.90 | 3207.29 | 3217.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 14:15:00 | 3237.60 | 3222.08 | 3221.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 3237.60 | 3222.08 | 3221.76 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 3208.60 | 3220.17 | 3221.27 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 15:15:00 | 3239.90 | 3222.65 | 3221.77 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 09:15:00 | 3193.10 | 3216.74 | 3219.16 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 3241.60 | 3223.55 | 3221.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 3248.70 | 3228.58 | 3224.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 3237.20 | 3237.29 | 3229.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 3237.20 | 3237.29 | 3229.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 3247.40 | 3256.11 | 3246.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 3247.40 | 3256.11 | 3246.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 3245.50 | 3253.99 | 3246.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:30:00 | 3260.90 | 3256.33 | 3248.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 3240.00 | 3264.24 | 3265.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 3240.00 | 3264.24 | 3265.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 3225.20 | 3251.87 | 3259.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 3214.20 | 3213.59 | 3232.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 09:45:00 | 3195.00 | 3213.59 | 3232.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 3201.70 | 3211.21 | 3229.48 | EMA400 retest candle locked (from downside) |

### Cycle 215 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 3240.40 | 3236.06 | 3235.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 11:15:00 | 3256.40 | 3240.13 | 3237.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 15:15:00 | 3250.00 | 3251.19 | 3244.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:15:00 | 3272.70 | 3251.19 | 3244.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 3258.00 | 3252.55 | 3245.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 13:00:00 | 3278.70 | 3262.20 | 3252.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 13:45:00 | 3282.50 | 3265.46 | 3254.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 14:30:00 | 3283.80 | 3270.93 | 3258.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:45:00 | 3279.30 | 3278.87 | 3263.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 3263.40 | 3277.46 | 3267.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 3263.40 | 3277.46 | 3267.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 3253.10 | 3272.59 | 3265.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 3253.10 | 3272.59 | 3265.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 3264.60 | 3270.99 | 3265.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 3258.80 | 3270.99 | 3265.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 3292.80 | 3275.35 | 3268.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 3245.90 | 3275.35 | 3268.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 3252.70 | 3270.82 | 3266.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 3237.00 | 3270.82 | 3266.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 3242.00 | 3265.06 | 3264.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3242.00 | 3265.06 | 3264.60 | SL hit (close<static) qty=1.00 sl=3242.60 alert=retest2 |

### Cycle 216 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 3215.00 | 3255.05 | 3260.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 3205.70 | 3245.18 | 3255.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 3162.20 | 3161.06 | 3189.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:45:00 | 3166.50 | 3161.06 | 3189.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 3194.80 | 3147.82 | 3172.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 3194.80 | 3147.82 | 3172.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 3166.30 | 3151.52 | 3171.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 3188.50 | 3151.52 | 3171.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 3149.00 | 3151.14 | 3168.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:30:00 | 3157.70 | 3151.14 | 3168.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 3176.20 | 3156.15 | 3168.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 3176.20 | 3156.15 | 3168.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 3172.70 | 3159.46 | 3169.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 3184.60 | 3159.46 | 3169.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 3155.20 | 3158.61 | 3167.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 3190.40 | 3158.61 | 3167.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 3170.40 | 3160.97 | 3168.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 3180.90 | 3160.97 | 3168.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 3159.10 | 3160.59 | 3167.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:15:00 | 3168.40 | 3160.59 | 3167.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 3170.70 | 3162.61 | 3167.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:45:00 | 3175.70 | 3162.61 | 3167.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 3176.30 | 3165.35 | 3168.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 15:00:00 | 3167.60 | 3167.59 | 3169.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 3179.90 | 3171.23 | 3170.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 3179.90 | 3171.23 | 3170.50 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 3158.70 | 3168.79 | 3169.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 3144.00 | 3163.83 | 3167.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 3175.00 | 3161.45 | 3164.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 3175.00 | 3161.45 | 3164.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 3175.00 | 3161.45 | 3164.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 3174.00 | 3161.45 | 3164.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3174.10 | 3163.98 | 3165.61 | EMA400 retest candle locked (from downside) |

### Cycle 219 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 3191.20 | 3169.42 | 3167.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 3202.80 | 3179.55 | 3173.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 3106.90 | 3180.07 | 3176.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 3106.90 | 3180.07 | 3176.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 3106.90 | 3180.07 | 3176.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 3106.90 | 3180.07 | 3176.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 3103.00 | 3164.66 | 3170.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 3097.00 | 3151.13 | 3163.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 14:15:00 | 3150.90 | 3145.01 | 3156.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 14:15:00 | 3150.90 | 3145.01 | 3156.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 3150.90 | 3145.01 | 3156.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 3150.90 | 3145.01 | 3156.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 3168.90 | 3149.79 | 3158.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 3150.00 | 3149.83 | 3157.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 3183.00 | 3156.47 | 3159.65 | SL hit (close>static) qty=1.00 sl=3170.00 alert=retest2 |

### Cycle 221 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 3194.30 | 3164.03 | 3162.80 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 3133.00 | 3165.20 | 3166.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 3126.60 | 3157.48 | 3162.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 3091.40 | 3079.64 | 3109.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 3091.40 | 3079.64 | 3109.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 3217.00 | 3110.37 | 3118.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 3191.00 | 3110.37 | 3118.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 3205.60 | 3129.41 | 3126.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 3205.60 | 3129.41 | 3126.14 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 3123.70 | 3147.73 | 3149.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 3093.30 | 3134.31 | 3142.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 3128.70 | 3123.38 | 3132.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 3128.70 | 3123.38 | 3132.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 3120.00 | 3122.71 | 3131.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 3138.30 | 3122.71 | 3131.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 3152.10 | 3128.59 | 3133.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 3152.10 | 3128.59 | 3133.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 3151.00 | 3133.07 | 3135.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 3152.00 | 3133.07 | 3135.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 3154.80 | 3139.37 | 3137.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 3181.90 | 3147.88 | 3141.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 12:15:00 | 3177.50 | 3177.64 | 3163.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 12:45:00 | 3182.30 | 3177.64 | 3163.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 3159.50 | 3174.01 | 3162.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 3159.50 | 3174.01 | 3162.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 3190.00 | 3177.21 | 3165.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 3207.90 | 3175.97 | 3165.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:15:00 | 3221.00 | 3180.04 | 3170.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:45:00 | 3201.00 | 3214.50 | 3210.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:15:00 | 3205.30 | 3214.50 | 3210.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 3203.80 | 3212.36 | 3209.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:15:00 | 3189.20 | 3212.36 | 3209.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 3205.40 | 3210.97 | 3209.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 3139.30 | 3196.63 | 3202.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 3139.30 | 3196.63 | 3202.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 3056.40 | 3152.35 | 3178.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 15:15:00 | 3084.00 | 3066.79 | 3092.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 09:15:00 | 3094.60 | 3066.79 | 3092.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 3085.00 | 3070.43 | 3092.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 3098.40 | 3070.43 | 3092.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 3029.80 | 3024.40 | 3042.30 | EMA400 retest candle locked (from downside) |

### Cycle 227 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 3068.50 | 3049.55 | 3047.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 11:15:00 | 3088.40 | 3057.32 | 3051.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 3098.40 | 3100.87 | 3087.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 3098.40 | 3100.87 | 3087.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 3098.40 | 3100.87 | 3087.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 3085.30 | 3100.87 | 3087.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 3087.70 | 3098.36 | 3090.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 3087.70 | 3098.36 | 3090.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 3081.70 | 3095.03 | 3089.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 3083.00 | 3095.03 | 3089.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 3109.80 | 3107.94 | 3100.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 3125.50 | 3111.46 | 3102.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:00:00 | 3136.30 | 3113.89 | 3107.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 3063.50 | 3102.08 | 3103.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 3063.50 | 3102.08 | 3103.67 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 3101.00 | 3080.50 | 3079.09 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 3020.40 | 3073.60 | 3077.78 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 3085.50 | 3066.27 | 3065.25 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 3042.00 | 3065.40 | 3065.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 3031.00 | 3054.31 | 3060.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 2895.30 | 2884.15 | 2916.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 2898.10 | 2884.15 | 2916.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2926.10 | 2892.18 | 2914.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 2926.10 | 2892.18 | 2914.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 2926.30 | 2899.00 | 2915.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:15:00 | 2907.20 | 2905.99 | 2915.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 2947.70 | 2919.04 | 2916.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 2947.70 | 2919.04 | 2916.32 | EMA200 above EMA400 |

### Cycle 234 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 2875.00 | 2913.46 | 2914.78 | EMA200 below EMA400 |

### Cycle 235 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 2927.00 | 2910.80 | 2909.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 2927.90 | 2914.22 | 2911.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 2899.60 | 2911.30 | 2910.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 2899.60 | 2911.30 | 2910.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 2899.60 | 2911.30 | 2910.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 2899.60 | 2911.30 | 2910.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 2921.00 | 2913.24 | 2911.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 2851.30 | 2913.24 | 2911.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 236 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 2843.80 | 2899.35 | 2905.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 2823.30 | 2884.14 | 2897.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 2800.40 | 2796.38 | 2831.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 2797.20 | 2796.38 | 2831.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 2879.30 | 2802.46 | 2822.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 2879.30 | 2802.46 | 2822.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 2901.10 | 2822.19 | 2829.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 2903.00 | 2822.19 | 2829.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 237 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 2931.50 | 2844.05 | 2838.59 | EMA200 above EMA400 |

### Cycle 238 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 2810.90 | 2848.49 | 2851.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2761.20 | 2825.19 | 2839.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2847.30 | 2772.61 | 2797.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2847.30 | 2772.61 | 2797.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2847.30 | 2772.61 | 2797.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 2847.30 | 2772.61 | 2797.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 2805.70 | 2779.23 | 2798.11 | EMA400 retest candle locked (from downside) |

### Cycle 239 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 2845.40 | 2814.68 | 2811.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 2848.90 | 2821.52 | 2814.45 | Break + close above crossover candle high |

### Cycle 240 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 2745.30 | 2806.28 | 2808.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 10:15:00 | 2731.60 | 2791.34 | 2801.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 2784.40 | 2781.01 | 2793.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 2784.40 | 2781.01 | 2793.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 2800.10 | 2784.83 | 2793.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 2800.10 | 2784.83 | 2793.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 2792.70 | 2786.40 | 2793.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 2770.60 | 2786.40 | 2793.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 2818.00 | 2798.50 | 2797.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 241 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 2818.00 | 2798.50 | 2797.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 2833.30 | 2805.46 | 2800.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 2812.40 | 2816.14 | 2807.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 2812.40 | 2816.14 | 2807.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 2812.40 | 2816.14 | 2807.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 2909.80 | 2822.36 | 2813.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:30:00 | 2860.00 | 2861.07 | 2850.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 3146.00 | 3090.82 | 3038.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 242 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 3026.10 | 3071.76 | 3072.82 | EMA200 below EMA400 |

### Cycle 243 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 3083.60 | 3074.13 | 3073.80 | EMA200 above EMA400 |

### Cycle 244 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 3050.00 | 3070.50 | 3072.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 12:15:00 | 3038.20 | 3064.04 | 3069.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 12:15:00 | 3062.10 | 3040.94 | 3051.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 12:15:00 | 3062.10 | 3040.94 | 3051.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 3062.10 | 3040.94 | 3051.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 3062.10 | 3040.94 | 3051.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 3058.00 | 3044.35 | 3052.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 3058.00 | 3044.35 | 3052.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 3057.90 | 3047.06 | 3052.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:45:00 | 3069.20 | 3047.06 | 3052.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 3080.00 | 3055.40 | 3055.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 3080.00 | 3055.40 | 3055.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 245 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 3079.00 | 3060.12 | 3057.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 3119.20 | 3079.34 | 3069.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 14:15:00 | 3079.30 | 3090.36 | 3079.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 3079.30 | 3090.36 | 3079.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 3079.30 | 3090.36 | 3079.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 3079.30 | 3090.36 | 3079.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 3075.00 | 3087.29 | 3079.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 3101.40 | 3087.29 | 3079.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 12:30:00 | 3094.90 | 3091.42 | 3084.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 14:45:00 | 3091.20 | 3089.57 | 3084.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 3092.00 | 3089.57 | 3084.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 3092.00 | 3090.06 | 3085.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 3075.90 | 3090.06 | 3085.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 3138.00 | 3099.65 | 3090.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:30:00 | 3080.70 | 3099.65 | 3090.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 3085.50 | 3102.86 | 3096.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 3085.50 | 3102.86 | 3096.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 3127.00 | 3107.69 | 3098.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 3159.00 | 3107.69 | 3098.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 3031.50 | 3090.06 | 3097.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 246 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 3031.50 | 3090.06 | 3097.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 3027.50 | 3077.55 | 3090.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 3080.00 | 3062.89 | 3076.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 15:15:00 | 3080.00 | 3062.89 | 3076.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 3080.00 | 3062.89 | 3076.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 3102.10 | 3062.89 | 3076.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 3100.00 | 3070.31 | 3078.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 3100.00 | 3070.31 | 3078.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 3088.80 | 3074.01 | 3079.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 3060.50 | 3071.31 | 3077.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 3073.00 | 3039.01 | 3045.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 3072.40 | 3050.62 | 3049.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 247 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 3072.40 | 3050.62 | 3049.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 3111.70 | 3067.21 | 3057.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 3080.20 | 3093.73 | 3082.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 11:15:00 | 3080.20 | 3093.73 | 3082.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 3080.20 | 3093.73 | 3082.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 3080.20 | 3093.73 | 3082.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 3075.00 | 3089.99 | 3081.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:30:00 | 3073.00 | 3089.99 | 3081.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-23 12:15:00 | 3676.85 | 2024-04-24 13:15:00 | 3736.30 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-04-23 14:15:00 | 3678.35 | 2024-04-24 13:15:00 | 3736.30 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-05-09 11:15:00 | 3524.65 | 2024-05-10 14:15:00 | 3560.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-05-09 14:30:00 | 3519.00 | 2024-05-10 14:15:00 | 3560.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-05-09 15:00:00 | 3524.75 | 2024-05-10 14:15:00 | 3560.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-05-10 10:00:00 | 3523.85 | 2024-05-10 14:15:00 | 3560.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-05-29 09:15:00 | 3585.25 | 2024-06-03 10:15:00 | 3676.30 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2024-06-14 11:15:00 | 3647.30 | 2024-07-02 10:15:00 | 3803.60 | STOP_HIT | 1.00 | 4.29% |
| BUY | retest2 | 2024-06-14 13:45:00 | 3642.00 | 2024-07-02 10:15:00 | 3803.60 | STOP_HIT | 1.00 | 4.44% |
| BUY | retest2 | 2024-06-14 15:15:00 | 3641.95 | 2024-07-02 10:15:00 | 3803.60 | STOP_HIT | 1.00 | 4.44% |
| SELL | retest2 | 2024-07-08 09:15:00 | 3773.00 | 2024-07-09 09:15:00 | 3878.60 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2024-07-08 10:45:00 | 3758.20 | 2024-07-09 09:15:00 | 3878.60 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2024-07-15 15:15:00 | 3880.00 | 2024-07-19 10:15:00 | 3846.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-07-18 12:00:00 | 3885.00 | 2024-07-19 10:15:00 | 3846.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-07-25 11:15:00 | 4002.05 | 2024-07-29 10:15:00 | 4402.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-25 13:45:00 | 3989.95 | 2024-07-29 10:15:00 | 4388.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-06 14:00:00 | 4261.65 | 2024-08-07 09:15:00 | 4357.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-08-06 14:30:00 | 4270.00 | 2024-08-07 09:15:00 | 4357.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-08-06 15:00:00 | 4255.50 | 2024-08-07 09:15:00 | 4357.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-08-21 11:30:00 | 4351.05 | 2024-08-21 13:15:00 | 4366.95 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-08-21 13:00:00 | 4344.85 | 2024-08-21 13:15:00 | 4366.95 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-08-23 14:00:00 | 4416.00 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2024-08-26 13:15:00 | 4409.50 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest2 | 2024-08-26 14:45:00 | 4414.00 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2024-08-27 09:15:00 | 4428.50 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2024-08-27 10:45:00 | 4467.35 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2024-08-27 12:30:00 | 4464.70 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2024-08-27 13:30:00 | 4466.10 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2024-08-28 10:15:00 | 4476.80 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2024-08-28 14:15:00 | 4512.00 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2024-08-30 09:30:00 | 4509.15 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-08-30 10:45:00 | 4499.00 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2024-08-30 11:15:00 | 4497.85 | 2024-09-05 09:15:00 | 4519.70 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest1 | 2024-09-10 09:15:00 | 4635.00 | 2024-09-11 14:15:00 | 4626.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-09-10 10:00:00 | 4629.70 | 2024-09-11 14:15:00 | 4626.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-09-12 09:15:00 | 4677.45 | 2024-09-17 09:15:00 | 4615.25 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-09-12 14:00:00 | 4629.70 | 2024-09-17 09:15:00 | 4615.25 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-09-12 14:30:00 | 4634.05 | 2024-09-17 09:15:00 | 4615.25 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-09-24 10:30:00 | 4756.15 | 2024-09-24 12:15:00 | 4662.35 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-10-01 09:15:00 | 4690.00 | 2024-10-04 12:15:00 | 4616.85 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-10-03 12:30:00 | 4691.50 | 2024-10-04 12:15:00 | 4616.85 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-10-03 15:15:00 | 4677.90 | 2024-10-04 12:15:00 | 4616.85 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-10-04 09:45:00 | 4715.00 | 2024-10-04 12:15:00 | 4616.85 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-10-24 09:30:00 | 4332.90 | 2024-10-28 15:15:00 | 4359.85 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-10-24 11:00:00 | 4329.80 | 2024-10-29 12:15:00 | 4355.40 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-10-25 09:30:00 | 4335.30 | 2024-10-29 12:15:00 | 4355.40 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-10-28 09:30:00 | 4320.80 | 2024-10-29 12:15:00 | 4355.40 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-10-28 12:30:00 | 4312.35 | 2024-10-29 12:15:00 | 4355.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-10-29 09:30:00 | 4308.20 | 2024-10-29 12:15:00 | 4355.40 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-10-29 10:30:00 | 4314.35 | 2024-10-29 12:15:00 | 4355.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-10-31 14:45:00 | 4474.65 | 2024-11-08 15:15:00 | 4561.00 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2024-11-04 10:00:00 | 4471.40 | 2024-11-08 15:15:00 | 4561.00 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2024-11-04 11:30:00 | 4493.45 | 2024-11-08 15:15:00 | 4561.00 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2024-11-04 13:15:00 | 4490.00 | 2024-11-08 15:15:00 | 4561.00 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2024-11-05 09:15:00 | 4481.15 | 2024-11-08 15:15:00 | 4561.00 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2024-11-05 13:30:00 | 4502.25 | 2024-11-08 15:15:00 | 4561.00 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest1 | 2024-11-14 09:15:00 | 4097.00 | 2024-11-25 10:15:00 | 4204.85 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-11-22 10:15:00 | 4126.35 | 2024-11-25 10:15:00 | 4204.85 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-11-22 12:00:00 | 4120.70 | 2024-11-25 10:15:00 | 4204.85 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-11-25 15:00:00 | 4124.60 | 2024-12-02 15:15:00 | 4105.00 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-11-26 09:45:00 | 4123.50 | 2024-12-02 15:15:00 | 4105.00 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2024-11-27 11:30:00 | 4113.70 | 2024-12-02 15:15:00 | 4105.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-11-28 09:30:00 | 4118.50 | 2024-12-02 15:15:00 | 4105.00 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2024-11-28 10:00:00 | 4113.50 | 2024-12-02 15:15:00 | 4105.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-12-20 12:30:00 | 3855.00 | 2025-01-01 09:15:00 | 3662.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:30:00 | 3855.00 | 2025-01-01 13:15:00 | 3691.65 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2025-01-06 11:15:00 | 3629.85 | 2025-01-09 09:15:00 | 3718.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-01-08 11:30:00 | 3625.00 | 2025-01-09 09:15:00 | 3718.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-01-08 13:15:00 | 3625.25 | 2025-01-09 09:15:00 | 3718.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-01-08 14:30:00 | 3629.05 | 2025-01-09 09:15:00 | 3718.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-01-27 09:15:00 | 3440.00 | 2025-01-30 09:15:00 | 3463.20 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-02-01 15:15:00 | 3499.10 | 2025-02-07 11:15:00 | 3510.00 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-02-07 09:30:00 | 3535.05 | 2025-02-07 11:15:00 | 3510.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-03-07 10:15:00 | 3198.15 | 2025-03-24 10:15:00 | 3517.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-07 14:00:00 | 3220.25 | 2025-03-26 10:15:00 | 3446.15 | STOP_HIT | 1.00 | 7.01% |
| BUY | retest2 | 2025-05-02 12:00:00 | 3683.00 | 2025-05-06 13:15:00 | 3637.30 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-05-08 09:15:00 | 3608.60 | 2025-05-08 11:15:00 | 3662.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-05-08 10:30:00 | 3634.70 | 2025-05-08 11:15:00 | 3662.90 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-05-08 14:00:00 | 3621.30 | 2025-05-12 09:15:00 | 3712.10 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-05-09 12:15:00 | 3635.40 | 2025-05-12 09:15:00 | 3712.10 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-05-15 11:15:00 | 3640.40 | 2025-05-15 13:15:00 | 3674.70 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-05-22 09:15:00 | 3670.00 | 2025-05-26 12:15:00 | 3692.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-06 09:15:00 | 3920.10 | 2025-06-06 11:15:00 | 3882.10 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-06 11:00:00 | 3917.80 | 2025-06-06 11:15:00 | 3882.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-06-06 12:30:00 | 3912.30 | 2025-06-06 14:15:00 | 3893.90 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-06-06 13:30:00 | 3910.00 | 2025-06-06 14:15:00 | 3893.90 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-06-09 09:15:00 | 3919.30 | 2025-06-20 09:15:00 | 4079.20 | STOP_HIT | 1.00 | 4.08% |
| BUY | retest2 | 2025-06-09 10:15:00 | 3902.50 | 2025-06-20 09:15:00 | 4079.20 | STOP_HIT | 1.00 | 4.53% |
| BUY | retest2 | 2025-06-09 11:15:00 | 3918.10 | 2025-06-20 09:15:00 | 4079.20 | STOP_HIT | 1.00 | 4.11% |
| BUY | retest2 | 2025-06-13 09:45:00 | 3936.00 | 2025-06-20 09:15:00 | 4079.20 | STOP_HIT | 1.00 | 3.64% |
| BUY | retest2 | 2025-06-13 14:15:00 | 3971.20 | 2025-06-20 09:15:00 | 4079.20 | STOP_HIT | 1.00 | 2.72% |
| BUY | retest2 | 2025-06-13 15:00:00 | 3976.00 | 2025-06-20 09:15:00 | 4079.20 | STOP_HIT | 1.00 | 2.60% |
| SELL | retest2 | 2025-07-09 14:30:00 | 4164.80 | 2025-07-15 10:15:00 | 4139.20 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-07-10 09:15:00 | 4101.70 | 2025-07-15 10:15:00 | 4139.20 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-16 12:15:00 | 4150.00 | 2025-07-16 14:15:00 | 4107.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-21 10:15:00 | 4152.00 | 2025-07-22 09:15:00 | 4104.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-29 10:30:00 | 4213.00 | 2025-08-01 13:15:00 | 4200.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-07-31 12:00:00 | 4212.70 | 2025-08-01 13:15:00 | 4200.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-08-01 12:45:00 | 4212.00 | 2025-08-01 13:15:00 | 4200.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-08-14 13:15:00 | 3725.90 | 2025-08-20 11:15:00 | 3785.80 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-08-18 11:15:00 | 3738.30 | 2025-08-20 11:15:00 | 3785.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-08 15:15:00 | 3687.00 | 2025-09-10 09:15:00 | 3733.10 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-09 14:45:00 | 3690.60 | 2025-09-10 09:15:00 | 3733.10 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-25 12:30:00 | 3598.00 | 2025-10-01 11:15:00 | 3568.70 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-10-16 09:15:00 | 3570.40 | 2025-10-16 10:15:00 | 3543.70 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-11-11 13:15:00 | 3781.70 | 2025-11-12 09:15:00 | 3702.70 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-11-11 14:15:00 | 3794.20 | 2025-11-12 09:15:00 | 3702.70 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-12-24 12:45:00 | 3256.00 | 2025-12-24 13:15:00 | 3223.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-12-30 09:45:00 | 3221.20 | 2025-12-31 14:15:00 | 3237.60 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-12-30 10:15:00 | 3215.80 | 2025-12-31 14:15:00 | 3237.60 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-12-31 11:00:00 | 3214.90 | 2025-12-31 14:15:00 | 3237.60 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-01-06 14:30:00 | 3260.90 | 2026-01-08 15:15:00 | 3240.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-01-14 13:00:00 | 3278.70 | 2026-01-19 10:15:00 | 3242.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-01-14 13:45:00 | 3282.50 | 2026-01-19 10:15:00 | 3242.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-01-14 14:30:00 | 3283.80 | 2026-01-19 10:15:00 | 3242.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-01-16 09:45:00 | 3279.30 | 2026-01-19 10:15:00 | 3242.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-01-23 15:00:00 | 3167.60 | 2026-01-27 09:15:00 | 3179.90 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-01-30 10:00:00 | 3150.00 | 2026-01-30 10:15:00 | 3183.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-02-03 10:15:00 | 3191.00 | 2026-02-03 10:15:00 | 3205.60 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-02-11 09:15:00 | 3207.90 | 2026-02-13 12:15:00 | 3139.30 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2026-02-11 12:15:00 | 3221.00 | 2026-02-13 12:15:00 | 3139.30 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2026-02-13 09:45:00 | 3201.00 | 2026-02-13 12:15:00 | 3139.30 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2026-02-13 10:15:00 | 3205.30 | 2026-02-13 12:15:00 | 3139.30 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-02-26 15:00:00 | 3125.50 | 2026-03-02 09:15:00 | 3063.50 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-27 14:00:00 | 3136.30 | 2026-03-02 09:15:00 | 3063.50 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2026-03-17 13:15:00 | 2907.20 | 2026-03-18 13:15:00 | 2947.70 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-04-06 09:15:00 | 2770.60 | 2026-04-06 12:15:00 | 2818.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-04-08 09:15:00 | 2909.80 | 2026-04-17 09:15:00 | 3146.00 | TARGET_HIT | 1.00 | 8.12% |
| BUY | retest2 | 2026-04-09 11:30:00 | 2860.00 | 2026-04-20 15:15:00 | 3026.10 | STOP_HIT | 1.00 | 5.81% |
| BUY | retest2 | 2026-04-27 09:15:00 | 3101.40 | 2026-04-30 09:15:00 | 3031.50 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-04-27 12:30:00 | 3094.90 | 2026-04-30 09:15:00 | 3031.50 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-04-27 14:45:00 | 3091.20 | 2026-04-30 09:15:00 | 3031.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2026-04-27 15:15:00 | 3092.00 | 2026-04-30 09:15:00 | 3031.50 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-04-29 09:15:00 | 3159.00 | 2026-04-30 09:15:00 | 3031.50 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2026-05-04 12:00:00 | 3060.50 | 2026-05-06 14:15:00 | 3072.40 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-05-06 11:00:00 | 3073.00 | 2026-05-06 14:15:00 | 3072.40 | STOP_HIT | 1.00 | 0.02% |
