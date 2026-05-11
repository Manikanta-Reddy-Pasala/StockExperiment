# Siemens Ltd. (SIEMENS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 3838.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 133 |
| ALERT1 | 91 |
| ALERT2 | 89 |
| ALERT2_SKIP | 41 |
| ALERT3 | 256 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 113 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 113 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 126 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 80
- **Target hits / Stop hits / Partials:** 2 / 113 / 11
- **Avg / median % per leg:** 0.19% / -0.77%
- **Sum % (uncompounded):** 24.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 19 | 34.5% | 0 | 54 | 1 | -0.10% | -5.3% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.58% | 7.8% |
| BUY @ 3rd Alert (retest2) | 52 | 17 | 32.7% | 0 | 52 | 0 | -0.25% | -13.1% |
| SELL (all) | 71 | 27 | 38.0% | 2 | 59 | 10 | 0.41% | 29.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 71 | 27 | 38.0% | 2 | 59 | 10 | 0.41% | 29.4% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.58% | 7.8% |
| retest2 (combined) | 123 | 44 | 35.8% | 2 | 111 | 10 | 0.13% | 16.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 11:15:00 | 3586.83 | 3606.51 | 3607.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 3547.11 | 3582.79 | 3593.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 11:15:00 | 3477.24 | 3476.36 | 3497.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 11:30:00 | 3482.98 | 3476.36 | 3497.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 3459.29 | 3473.14 | 3490.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 3459.29 | 3473.14 | 3490.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 3645.49 | 3506.64 | 3502.90 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 3273.66 | 3513.37 | 3525.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 3073.76 | 3425.45 | 3484.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 3236.42 | 3221.51 | 3313.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 3236.42 | 3221.51 | 3313.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 3375.08 | 3258.76 | 3307.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 3375.08 | 3258.76 | 3307.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 3374.63 | 3281.93 | 3313.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:15:00 | 3382.04 | 3281.93 | 3313.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 3378.11 | 3317.77 | 3323.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:00:00 | 3378.11 | 3317.77 | 3323.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 14:15:00 | 3377.64 | 3329.74 | 3328.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 3385.49 | 3340.89 | 3333.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 3408.09 | 3412.52 | 3388.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 3414.35 | 3412.52 | 3388.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 3776.32 | 3814.71 | 3781.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:45:00 | 3775.94 | 3814.71 | 3781.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 3745.97 | 3800.97 | 3778.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 14:00:00 | 3745.97 | 3800.97 | 3778.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 3743.03 | 3789.38 | 3775.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 14:45:00 | 3751.93 | 3789.38 | 3775.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 3705.62 | 3765.20 | 3766.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 15:15:00 | 3693.44 | 3719.63 | 3734.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 3744.03 | 3724.51 | 3735.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 3744.03 | 3724.51 | 3735.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 3744.03 | 3724.51 | 3735.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 3744.03 | 3724.51 | 3735.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 3744.92 | 3728.59 | 3735.96 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 3786.61 | 3744.50 | 3742.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 3802.34 | 3766.82 | 3754.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 12:15:00 | 3772.39 | 3776.04 | 3762.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 12:15:00 | 3772.39 | 3776.04 | 3762.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 3772.39 | 3776.04 | 3762.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:45:00 | 3757.67 | 3776.04 | 3762.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 3764.91 | 3773.82 | 3762.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:30:00 | 3759.12 | 3773.82 | 3762.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 3761.83 | 3771.42 | 3762.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:45:00 | 3749.15 | 3771.42 | 3762.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 3765.83 | 3770.30 | 3762.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 3769.51 | 3770.30 | 3762.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 3743.95 | 3765.03 | 3761.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:45:00 | 3738.93 | 3765.03 | 3761.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 3775.62 | 3767.15 | 3762.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 11:30:00 | 3796.38 | 3769.61 | 3764.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 12:15:00 | 3795.83 | 3769.61 | 3764.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 13:00:00 | 3781.24 | 3771.94 | 3765.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:30:00 | 3792.77 | 3780.67 | 3772.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 3841.07 | 3830.41 | 3807.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 3819.64 | 3830.41 | 3807.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 3824.02 | 3829.13 | 3808.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 3824.02 | 3829.13 | 3808.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 3817.33 | 3826.77 | 3809.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:00:00 | 3817.33 | 3826.77 | 3809.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 3827.50 | 3826.92 | 3811.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 3868.56 | 3828.42 | 3814.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 12:15:00 | 3836.07 | 3867.02 | 3852.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:30:00 | 3842.59 | 3856.83 | 3850.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 3821.38 | 3882.64 | 3887.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 3821.38 | 3882.64 | 3887.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 3819.24 | 3845.94 | 3859.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 11:15:00 | 3871.39 | 3851.03 | 3860.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 11:15:00 | 3871.39 | 3851.03 | 3860.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 3871.39 | 3851.03 | 3860.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:00:00 | 3871.39 | 3851.03 | 3860.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 3882.53 | 3857.33 | 3862.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:30:00 | 3902.66 | 3857.33 | 3862.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 3880.22 | 3861.91 | 3863.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:45:00 | 3891.11 | 3861.91 | 3863.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 3901.22 | 3869.77 | 3867.28 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 09:15:00 | 3844.23 | 3866.70 | 3868.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 10:15:00 | 3813.23 | 3856.00 | 3863.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 14:15:00 | 3795.48 | 3787.41 | 3810.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 14:45:00 | 3804.63 | 3787.41 | 3810.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 3792.08 | 3789.90 | 3807.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 10:15:00 | 3782.85 | 3789.90 | 3807.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:15:00 | 3593.71 | 3713.19 | 3758.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-07-19 15:15:00 | 3404.57 | 3454.67 | 3547.13 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 3440.87 | 3420.92 | 3418.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 3474.95 | 3439.59 | 3428.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 10:15:00 | 3487.48 | 3493.19 | 3469.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 14:30:00 | 3509.80 | 3492.62 | 3476.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 3471.07 | 3487.88 | 3477.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 3471.07 | 3487.88 | 3477.27 | SL hit (close<ema400) qty=1.00 sl=3477.27 alert=retest1 |

### Cycle 11 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 3482.48 | 3488.02 | 3488.43 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 14:15:00 | 3494.24 | 3489.27 | 3488.96 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 3412.86 | 3473.69 | 3481.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 3354.35 | 3425.29 | 3451.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 3352.13 | 3351.46 | 3392.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 3352.13 | 3351.46 | 3392.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 3352.13 | 3351.46 | 3392.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 3304.98 | 3336.00 | 3371.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 14:15:00 | 3400.38 | 3349.76 | 3354.62 | SL hit (close>static) qty=1.00 sl=3397.75 alert=retest2 |

### Cycle 14 — BUY (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 13:15:00 | 3384.07 | 3357.30 | 3356.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 3397.87 | 3366.11 | 3360.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 15:15:00 | 3430.26 | 3431.32 | 3410.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 09:15:00 | 3415.92 | 3431.32 | 3410.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 3423.30 | 3429.72 | 3411.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:45:00 | 3460.56 | 3442.08 | 3420.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:00:00 | 3467.67 | 3450.06 | 3437.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 13:15:00 | 3503.71 | 3518.26 | 3518.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 13:15:00 | 3503.71 | 3518.26 | 3518.88 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 15:15:00 | 3529.69 | 3520.94 | 3520.02 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 12:15:00 | 3513.98 | 3519.11 | 3519.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 13:15:00 | 3502.84 | 3515.86 | 3517.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 3528.19 | 3515.05 | 3516.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 3528.19 | 3515.05 | 3516.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 3528.19 | 3515.05 | 3516.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 3528.19 | 3515.05 | 3516.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 3520.86 | 3516.21 | 3517.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 12:15:00 | 3509.80 | 3516.51 | 3517.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 09:45:00 | 3509.80 | 3499.47 | 3503.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 10:15:00 | 3334.31 | 3355.52 | 3369.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 10:15:00 | 3334.31 | 3355.52 | 3369.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-05 11:15:00 | 3357.03 | 3355.82 | 3368.77 | SL hit (close>ema200) qty=0.50 sl=3355.82 alert=retest2 |

### Cycle 18 — BUY (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 12:15:00 | 3305.97 | 3291.20 | 3289.90 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 3277.34 | 3288.47 | 3288.89 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 3297.10 | 3289.03 | 3288.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 13:15:00 | 3304.73 | 3292.17 | 3290.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 3337.12 | 3341.78 | 3329.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 3337.12 | 3341.78 | 3329.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 3344.63 | 3341.92 | 3331.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 14:30:00 | 3355.69 | 3346.86 | 3337.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:30:00 | 3365.66 | 3355.56 | 3344.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 3392.45 | 3353.12 | 3347.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 3303.19 | 3339.84 | 3342.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 3303.19 | 3339.84 | 3342.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 3246.64 | 3321.20 | 3333.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 3353.00 | 3324.80 | 3331.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 3353.00 | 3324.80 | 3331.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 3353.00 | 3324.80 | 3331.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 3353.00 | 3324.80 | 3331.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 3346.02 | 3329.04 | 3333.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 3367.64 | 3329.04 | 3333.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 3378.56 | 3338.95 | 3337.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 3393.02 | 3349.76 | 3342.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 13:15:00 | 3410.99 | 3416.47 | 3390.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 14:00:00 | 3410.99 | 3416.47 | 3390.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 3499.91 | 3535.96 | 3514.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 3499.91 | 3535.96 | 3514.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 3485.29 | 3525.82 | 3511.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:45:00 | 3477.14 | 3525.82 | 3511.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 3511.32 | 3522.53 | 3512.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 3506.10 | 3522.53 | 3512.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 3554.54 | 3528.93 | 3516.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 10:45:00 | 3572.89 | 3535.12 | 3520.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 11:15:00 | 3566.97 | 3535.12 | 3520.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 11:45:00 | 3577.41 | 3544.45 | 3525.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 12:15:00 | 3613.23 | 3650.87 | 3650.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 3613.23 | 3650.87 | 3650.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 3606.62 | 3637.90 | 3644.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 3576.82 | 3542.37 | 3576.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 3576.82 | 3542.37 | 3576.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 3576.82 | 3542.37 | 3576.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:45:00 | 3568.36 | 3542.37 | 3576.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 3596.80 | 3553.26 | 3578.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:45:00 | 3599.93 | 3553.26 | 3578.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 3603.51 | 3563.31 | 3580.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 3603.51 | 3563.31 | 3580.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 3629.91 | 3576.63 | 3585.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 3629.91 | 3576.63 | 3585.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 3666.28 | 3594.56 | 3592.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 3809.10 | 3647.69 | 3617.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 3784.54 | 3805.49 | 3763.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:15:00 | 3767.32 | 3805.49 | 3763.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 3788.22 | 3802.04 | 3766.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 3805.60 | 3802.04 | 3766.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 3833.21 | 3920.95 | 3897.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 3833.21 | 3920.95 | 3897.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 3860.38 | 3908.83 | 3894.56 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 3839.06 | 3880.55 | 3883.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 3794.41 | 3851.87 | 3868.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 3886.04 | 3858.70 | 3870.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 10:15:00 | 3886.04 | 3858.70 | 3870.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 3886.04 | 3858.70 | 3870.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:00:00 | 3886.04 | 3858.70 | 3870.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 3882.65 | 3863.49 | 3871.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 3882.65 | 3863.49 | 3871.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 3866.72 | 3864.14 | 3871.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 3852.50 | 3872.68 | 3873.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 3659.88 | 3748.52 | 3800.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 14:15:00 | 3467.25 | 3540.26 | 3629.91 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 26 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 3421.81 | 3402.95 | 3402.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 3471.47 | 3421.69 | 3411.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 3440.20 | 3460.48 | 3447.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 18:15:00 | 3440.20 | 3460.48 | 3447.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 3440.20 | 3460.48 | 3447.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 3440.20 | 3460.48 | 3447.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 3415.54 | 3451.49 | 3444.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 3415.54 | 3451.49 | 3444.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 3384.32 | 3438.06 | 3439.19 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 13:15:00 | 3466.67 | 3442.09 | 3440.48 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 3405.82 | 3440.55 | 3440.68 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 3465.18 | 3443.54 | 3441.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 3517.66 | 3459.76 | 3448.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 3485.86 | 3494.86 | 3476.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 3485.86 | 3494.86 | 3476.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 3488.60 | 3493.60 | 3477.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 3488.35 | 3493.60 | 3477.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 3537.72 | 3510.30 | 3493.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 13:15:00 | 3579.45 | 3528.54 | 3506.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 3441.62 | 3510.55 | 3516.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 09:15:00 | 3441.62 | 3510.55 | 3516.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 3434.78 | 3477.33 | 3497.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 3348.23 | 3342.05 | 3379.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 12:45:00 | 3349.72 | 3342.05 | 3379.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 3338.44 | 3345.25 | 3369.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:30:00 | 3318.40 | 3340.96 | 3365.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:00:00 | 3323.80 | 3340.96 | 3365.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:00:00 | 3320.81 | 3336.35 | 3359.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 09:45:00 | 3315.57 | 3321.41 | 3343.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 3364.86 | 3330.10 | 3345.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 3364.86 | 3330.10 | 3345.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 3358.55 | 3335.79 | 3346.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:30:00 | 3350.49 | 3337.70 | 3345.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 3460.19 | 3351.37 | 3340.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 3460.19 | 3351.37 | 3340.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 3571.94 | 3427.14 | 3385.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 13:15:00 | 3598.34 | 3602.69 | 3541.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 13:30:00 | 3602.39 | 3602.69 | 3541.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 3723.57 | 3750.75 | 3725.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:00:00 | 3723.57 | 3750.75 | 3725.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 3758.62 | 3752.33 | 3728.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 15:15:00 | 3766.32 | 3752.33 | 3728.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 3880.12 | 3922.87 | 3927.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 3880.12 | 3922.87 | 3927.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 3797.82 | 3881.14 | 3904.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 3811.89 | 3799.62 | 3823.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 11:00:00 | 3811.89 | 3799.62 | 3823.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 3710.64 | 3781.83 | 3812.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 3710.64 | 3781.83 | 3812.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 3261.06 | 3286.74 | 3328.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:30:00 | 3247.44 | 3281.04 | 3322.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:30:00 | 3253.08 | 3275.01 | 3315.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:00:00 | 3250.87 | 3275.01 | 3315.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 3228.44 | 3249.51 | 3288.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 3266.67 | 3242.39 | 3268.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:15:00 | 3277.49 | 3242.39 | 3268.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 3250.92 | 3244.09 | 3267.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:30:00 | 3274.68 | 3244.09 | 3267.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 3262.57 | 3249.09 | 3265.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-01 13:15:00 | 3298.52 | 3276.95 | 3275.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 3298.52 | 3276.95 | 3275.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 3302.00 | 3281.96 | 3277.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 3269.68 | 3282.39 | 3278.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 3269.68 | 3282.39 | 3278.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 3269.68 | 3282.39 | 3278.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:45:00 | 3268.91 | 3282.39 | 3278.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 3286.26 | 3283.16 | 3279.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:15:00 | 3298.81 | 3286.89 | 3281.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 3258.57 | 3286.93 | 3290.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 3258.57 | 3286.93 | 3290.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 3219.50 | 3269.04 | 3280.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 3168.17 | 3156.10 | 3191.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:00:00 | 3168.17 | 3156.10 | 3191.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 3163.59 | 3157.60 | 3188.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:45:00 | 3156.83 | 3157.12 | 3185.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 2998.99 | 3043.16 | 3092.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 2924.05 | 2914.59 | 2961.05 | SL hit (close>ema200) qty=0.50 sl=2914.59 alert=retest2 |

### Cycle 36 — BUY (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 13:15:00 | 3003.07 | 2954.63 | 2949.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 3031.55 | 2980.49 | 2963.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 2979.83 | 2980.36 | 2964.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 2979.83 | 2980.36 | 2964.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 2998.77 | 3037.88 | 3023.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 2998.77 | 3037.88 | 3023.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 2980.84 | 3026.47 | 3019.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 2980.84 | 3026.47 | 3019.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 2978.63 | 3009.89 | 3012.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 2971.10 | 3002.13 | 3008.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 3032.62 | 2944.57 | 2961.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 3032.62 | 2944.57 | 2961.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 3032.62 | 2944.57 | 2961.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 3032.62 | 2944.57 | 2961.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 3012.66 | 2958.19 | 2965.69 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 3040.33 | 2984.10 | 2976.76 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 2950.94 | 2976.30 | 2978.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 2920.69 | 2965.18 | 2972.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 15:15:00 | 2880.87 | 2849.08 | 2877.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 15:15:00 | 2880.87 | 2849.08 | 2877.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 2880.87 | 2849.08 | 2877.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 09:15:00 | 2814.80 | 2849.08 | 2877.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 2933.59 | 2888.52 | 2884.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 2933.59 | 2888.52 | 2884.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 3000.48 | 2930.66 | 2910.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 2970.26 | 2998.37 | 2969.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 2970.26 | 2998.37 | 2969.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 2970.26 | 2998.37 | 2969.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 2959.54 | 2998.37 | 2969.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 2821.24 | 2962.94 | 2955.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 2821.24 | 2962.94 | 2955.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 2863.77 | 2943.11 | 2947.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 2685.94 | 2866.09 | 2909.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 12:15:00 | 2678.49 | 2672.07 | 2743.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 12:30:00 | 2681.57 | 2672.07 | 2743.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 2724.10 | 2688.54 | 2728.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 2727.80 | 2688.54 | 2728.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 2787.04 | 2708.24 | 2734.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 2787.04 | 2708.24 | 2734.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 2769.49 | 2720.49 | 2737.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:30:00 | 2792.06 | 2720.49 | 2737.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 14:15:00 | 2773.64 | 2750.78 | 2748.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 09:15:00 | 2808.59 | 2765.63 | 2756.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 2775.97 | 2783.35 | 2769.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 2775.97 | 2783.35 | 2769.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 2756.09 | 2783.71 | 2773.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 2756.09 | 2783.71 | 2773.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 2750.17 | 2777.01 | 2771.32 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 2714.18 | 2764.44 | 2766.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 2710.75 | 2739.54 | 2751.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 2596.81 | 2576.74 | 2620.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 2596.81 | 2576.74 | 2620.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 2557.58 | 2572.91 | 2615.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 14:00:00 | 2557.58 | 2572.91 | 2615.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 2617.26 | 2585.61 | 2607.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 2617.26 | 2585.61 | 2607.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 2612.07 | 2590.90 | 2608.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 2629.82 | 2590.90 | 2608.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 2567.97 | 2586.31 | 2604.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 2561.26 | 2586.31 | 2604.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 2433.20 | 2476.38 | 2523.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-18 09:15:00 | 2436.10 | 2428.66 | 2469.61 | SL hit (close>ema200) qty=0.50 sl=2428.66 alert=retest2 |

### Cycle 44 — BUY (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 15:15:00 | 2468.79 | 2443.23 | 2441.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 2491.31 | 2455.26 | 2448.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 2447.22 | 2466.31 | 2456.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 2447.22 | 2466.31 | 2456.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 2447.22 | 2466.31 | 2456.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 2447.22 | 2466.31 | 2456.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 2432.50 | 2459.55 | 2454.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 2432.50 | 2459.55 | 2454.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 2435.09 | 2454.66 | 2452.33 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 12:15:00 | 2434.89 | 2450.70 | 2450.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 2395.51 | 2437.47 | 2444.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 15:15:00 | 2424.82 | 2423.19 | 2432.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 09:15:00 | 2425.99 | 2423.19 | 2432.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 2430.81 | 2424.72 | 2432.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 2441.70 | 2424.72 | 2432.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 2418.58 | 2423.49 | 2430.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:15:00 | 2410.85 | 2423.49 | 2430.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 2290.31 | 2341.84 | 2371.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 11:15:00 | 2320.79 | 2304.69 | 2329.51 | SL hit (close>ema200) qty=0.50 sl=2304.69 alert=retest2 |

### Cycle 46 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 2405.93 | 2347.48 | 2344.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 2416.77 | 2369.27 | 2355.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 2547.84 | 2548.42 | 2509.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 2558.77 | 2548.42 | 2509.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 2514.75 | 2540.42 | 2518.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 2514.75 | 2540.42 | 2518.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 2541.52 | 2540.64 | 2520.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:15:00 | 2571.45 | 2540.64 | 2520.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 2494.92 | 2525.88 | 2519.79 | SL hit (close<static) qty=1.00 sl=2508.36 alert=retest2 |

### Cycle 47 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 2489.75 | 2512.42 | 2514.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 2473.94 | 2500.66 | 2508.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 2467.72 | 2461.07 | 2478.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 2449.20 | 2461.07 | 2478.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2428.23 | 2454.50 | 2473.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 2426.34 | 2454.50 | 2473.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 2412.34 | 2447.61 | 2468.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:30:00 | 2423.18 | 2432.09 | 2455.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 09:15:00 | 2493.15 | 2447.49 | 2456.92 | SL hit (close>static) qty=1.00 sl=2490.22 alert=retest2 |

### Cycle 48 — BUY (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 11:15:00 | 2493.47 | 2464.96 | 2463.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 12:15:00 | 2505.80 | 2473.13 | 2467.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 2445.40 | 2469.70 | 2467.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 14:15:00 | 2445.40 | 2469.70 | 2467.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 2445.40 | 2469.70 | 2467.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 2445.40 | 2469.70 | 2467.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 2450.77 | 2465.91 | 2465.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 2468.29 | 2465.91 | 2465.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 09:15:00 | 2454.23 | 2463.58 | 2464.58 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 2497.78 | 2462.81 | 2461.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 2521.12 | 2479.84 | 2470.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 2543.64 | 2558.40 | 2530.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 2543.64 | 2558.40 | 2530.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2585.89 | 2599.32 | 2576.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2585.89 | 2599.32 | 2576.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 2591.39 | 2604.60 | 2591.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 15:00:00 | 2591.39 | 2604.60 | 2591.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 2597.55 | 2603.19 | 2591.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 2590.94 | 2603.19 | 2591.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 2566.50 | 2595.85 | 2589.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 2566.50 | 2595.85 | 2589.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 2566.38 | 2589.96 | 2587.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 2565.41 | 2589.96 | 2587.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 2559.92 | 2583.95 | 2584.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 2539.76 | 2564.64 | 2574.66 | Break + close below crossover candle low |

### Cycle 52 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 2681.24 | 2587.96 | 2584.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 10:15:00 | 2694.14 | 2609.20 | 2594.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-27 09:15:00 | 2658.20 | 2671.37 | 2639.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-27 09:30:00 | 2658.65 | 2671.37 | 2639.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 2651.96 | 2661.83 | 2644.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:45:00 | 2654.72 | 2661.83 | 2644.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 2640.83 | 2657.63 | 2644.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 2637.84 | 2657.63 | 2644.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 2643.81 | 2654.87 | 2644.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 2625.89 | 2654.87 | 2644.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 2604.16 | 2644.73 | 2640.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 2604.16 | 2644.73 | 2640.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 2622.58 | 2640.30 | 2639.12 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 11:15:00 | 2616.72 | 2635.58 | 2637.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 2600.16 | 2628.50 | 2633.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 14:15:00 | 2621.94 | 2620.16 | 2628.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 14:45:00 | 2624.64 | 2620.16 | 2628.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 2617.44 | 2619.62 | 2627.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 2648.21 | 2619.62 | 2627.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 2619.47 | 2619.59 | 2626.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:45:00 | 2593.03 | 2614.87 | 2624.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 2592.55 | 2610.91 | 2621.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 2587.48 | 2604.79 | 2614.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 2622.61 | 2613.46 | 2613.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 2622.61 | 2613.46 | 2613.26 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 15:15:00 | 2609.98 | 2613.25 | 2613.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 2520.49 | 2594.70 | 2605.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 09:15:00 | 2695.00 | 2534.19 | 2556.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 2695.00 | 2534.19 | 2556.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 2695.00 | 2534.19 | 2556.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 10:00:00 | 2695.00 | 2534.19 | 2556.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 10:15:00 | 3013.50 | 2630.05 | 2598.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-07 11:15:00 | 3087.00 | 2721.44 | 2642.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-08 14:15:00 | 2762.65 | 2797.00 | 2745.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-08 15:00:00 | 2762.65 | 2797.00 | 2745.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 2678.50 | 2769.29 | 2741.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 10:00:00 | 2678.50 | 2769.29 | 2741.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 2684.75 | 2752.38 | 2736.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 10:30:00 | 2681.35 | 2752.38 | 2736.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 2734.80 | 2738.08 | 2732.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:45:00 | 2758.45 | 2738.83 | 2733.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 13:15:00 | 2717.60 | 2740.11 | 2736.12 | SL hit (close<static) qty=1.00 sl=2723.05 alert=retest2 |

### Cycle 57 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 2814.50 | 2854.48 | 2859.75 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 2885.50 | 2857.36 | 2857.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 2917.00 | 2877.13 | 2866.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 2922.50 | 2925.26 | 2900.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 13:45:00 | 2926.50 | 2925.26 | 2900.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 2880.50 | 2920.06 | 2904.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:00:00 | 2880.50 | 2920.06 | 2904.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 2879.00 | 2911.85 | 2902.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:00:00 | 2879.00 | 2911.85 | 2902.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 2862.00 | 2890.55 | 2893.82 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 2948.80 | 2902.10 | 2898.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 12:15:00 | 2950.90 | 2930.39 | 2918.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 2929.20 | 2939.58 | 2927.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 2929.20 | 2939.58 | 2927.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 2929.20 | 2939.58 | 2927.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 2925.30 | 2939.58 | 2927.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 2938.20 | 2939.30 | 2928.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 15:00:00 | 2946.50 | 2940.02 | 2932.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 2968.80 | 2939.02 | 2932.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 09:45:00 | 2946.20 | 2940.40 | 2933.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 2909.50 | 2952.85 | 2947.49 | SL hit (close<static) qty=1.00 sl=2919.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 10:15:00 | 2907.10 | 2943.70 | 2943.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 12:15:00 | 2868.60 | 2920.73 | 2932.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 2859.50 | 2859.43 | 2891.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 12:00:00 | 2859.50 | 2859.43 | 2891.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2932.30 | 2872.62 | 2884.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 2932.30 | 2872.62 | 2884.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 12:15:00 | 2907.40 | 2888.40 | 2889.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 13:00:00 | 2907.40 | 2888.40 | 2889.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 2904.40 | 2891.60 | 2891.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 2931.40 | 2901.63 | 2895.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 2912.00 | 2915.38 | 2906.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 15:15:00 | 2936.60 | 2915.38 | 2906.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 09:15:00 | 3083.43 | 3014.87 | 2986.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 3049.90 | 3055.82 | 3026.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 3049.90 | 3055.82 | 3026.36 | SL hit (close<ema200) qty=0.50 sl=3055.82 alert=retest1 |

### Cycle 63 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 3267.00 | 3312.57 | 3318.44 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 12:15:00 | 3318.80 | 3309.65 | 3308.70 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 3286.50 | 3305.40 | 3307.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 10:15:00 | 3277.00 | 3299.72 | 3304.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 3300.80 | 3299.60 | 3303.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 14:00:00 | 3300.80 | 3299.60 | 3303.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 3309.90 | 3301.66 | 3303.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 3309.90 | 3301.66 | 3303.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 3309.00 | 3303.13 | 3304.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 3320.40 | 3303.13 | 3304.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 3310.40 | 3304.58 | 3304.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:15:00 | 3304.80 | 3304.58 | 3304.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 3306.60 | 3302.11 | 3302.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 3309.70 | 3303.62 | 3302.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 3309.70 | 3303.62 | 3302.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 3344.00 | 3312.08 | 3307.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 3346.30 | 3366.77 | 3356.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 10:15:00 | 3346.30 | 3366.77 | 3356.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 3346.30 | 3366.77 | 3356.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 3346.30 | 3366.77 | 3356.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3350.90 | 3363.60 | 3356.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 12:30:00 | 3366.00 | 3365.26 | 3357.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 3307.90 | 3348.60 | 3351.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 3307.90 | 3348.60 | 3351.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 3291.40 | 3326.75 | 3340.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 3292.50 | 3269.65 | 3287.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 3292.50 | 3269.65 | 3287.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 3292.50 | 3269.65 | 3287.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 3292.50 | 3269.65 | 3287.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 3277.60 | 3271.24 | 3286.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 3281.00 | 3271.24 | 3286.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 3278.60 | 3273.15 | 3284.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 3281.80 | 3273.15 | 3284.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 3330.90 | 3284.28 | 3286.75 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 3318.10 | 3291.04 | 3289.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 11:15:00 | 3358.20 | 3304.48 | 3295.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 3313.90 | 3317.47 | 3307.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 3313.90 | 3317.47 | 3307.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 3313.90 | 3317.47 | 3307.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 3313.40 | 3317.47 | 3307.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 3307.40 | 3315.46 | 3307.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 3315.00 | 3315.46 | 3307.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 3316.60 | 3315.68 | 3308.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 13:15:00 | 3328.70 | 3315.68 | 3308.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 3284.60 | 3324.10 | 3317.47 | SL hit (close<static) qty=1.00 sl=3303.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 3272.20 | 3310.14 | 3312.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 3208.60 | 3278.82 | 3295.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 3166.50 | 3136.33 | 3182.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-24 10:00:00 | 3166.50 | 3136.33 | 3182.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 3200.20 | 3149.11 | 3184.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 3196.90 | 3149.11 | 3184.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 3226.00 | 3164.49 | 3188.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:45:00 | 3226.00 | 3164.49 | 3188.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 3115.90 | 3156.16 | 3176.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:15:00 | 3113.60 | 3156.16 | 3176.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:00:00 | 3105.60 | 3140.40 | 3165.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 09:15:00 | 3254.00 | 3173.43 | 3164.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 3254.00 | 3173.43 | 3164.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 3276.70 | 3232.37 | 3205.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 11:15:00 | 3308.90 | 3315.90 | 3286.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 11:30:00 | 3312.80 | 3315.90 | 3286.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 3286.30 | 3309.98 | 3286.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 3280.50 | 3309.98 | 3286.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 3294.00 | 3306.78 | 3287.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 3270.30 | 3306.78 | 3287.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 3289.20 | 3303.26 | 3287.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:45:00 | 3283.10 | 3303.26 | 3287.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 3286.20 | 3299.85 | 3287.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 3304.70 | 3299.85 | 3287.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 3263.20 | 3295.84 | 3294.35 | SL hit (close<static) qty=1.00 sl=3284.10 alert=retest2 |

### Cycle 71 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 3255.30 | 3287.73 | 3290.80 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 3318.80 | 3286.22 | 3285.60 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 3264.00 | 3285.91 | 3287.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 11:15:00 | 3258.00 | 3280.33 | 3285.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 3144.80 | 3138.92 | 3173.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:30:00 | 3139.00 | 3138.92 | 3173.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 3153.80 | 3125.97 | 3152.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 3153.80 | 3125.97 | 3152.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 3150.00 | 3130.78 | 3151.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 3139.90 | 3130.78 | 3151.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3127.10 | 3130.04 | 3149.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:30:00 | 3092.20 | 3131.53 | 3142.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:00:00 | 3107.40 | 3126.70 | 3139.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 13:45:00 | 3104.20 | 3094.42 | 3108.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 15:15:00 | 3105.70 | 3098.32 | 3109.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 3105.70 | 3099.79 | 3108.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 3126.00 | 3099.79 | 3108.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 3119.90 | 3103.82 | 3109.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:30:00 | 3137.30 | 3103.82 | 3109.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 3134.90 | 3110.03 | 3112.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:45:00 | 3127.60 | 3110.03 | 3112.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 3136.30 | 3115.29 | 3114.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 11:15:00 | 3136.30 | 3115.29 | 3114.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 3152.00 | 3130.58 | 3123.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 3127.80 | 3140.22 | 3132.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 3127.80 | 3140.22 | 3132.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3127.80 | 3140.22 | 3132.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 3127.80 | 3140.22 | 3132.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 3144.00 | 3140.98 | 3133.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 3146.60 | 3141.98 | 3134.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:30:00 | 3147.60 | 3142.61 | 3136.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 14:00:00 | 3146.00 | 3142.61 | 3136.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 14:30:00 | 3149.30 | 3142.57 | 3136.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 3134.40 | 3140.94 | 3136.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 3133.60 | 3140.94 | 3136.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 3123.60 | 3137.47 | 3135.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 3123.60 | 3137.47 | 3135.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 3119.20 | 3133.82 | 3133.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 3119.20 | 3133.82 | 3133.91 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 3141.50 | 3134.70 | 3133.81 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 3107.40 | 3129.24 | 3131.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 3086.30 | 3115.71 | 3122.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 3063.00 | 3061.14 | 3078.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 3063.00 | 3061.14 | 3078.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 3058.10 | 3060.35 | 3075.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 3067.30 | 3060.35 | 3075.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 3066.50 | 3058.04 | 3069.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 3066.80 | 3058.04 | 3069.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 3071.70 | 3060.78 | 3069.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 3069.80 | 3060.78 | 3069.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 3061.30 | 3060.88 | 3068.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 3059.30 | 3060.88 | 3068.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 3044.30 | 3057.56 | 3066.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 3039.90 | 3057.56 | 3066.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 3073.60 | 3062.74 | 3065.64 | SL hit (close>static) qty=1.00 sl=3073.20 alert=retest2 |

### Cycle 78 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 3040.40 | 3018.39 | 3016.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 3088.00 | 3035.91 | 3025.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 3071.90 | 3077.28 | 3059.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 3071.90 | 3077.28 | 3059.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 3071.90 | 3077.28 | 3059.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:15:00 | 3055.80 | 3077.28 | 3059.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 3073.30 | 3076.48 | 3060.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:45:00 | 3070.00 | 3076.48 | 3060.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 3079.30 | 3075.51 | 3064.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:15:00 | 3107.20 | 3075.51 | 3064.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 3032.40 | 3080.94 | 3077.89 | SL hit (close<static) qty=1.00 sl=3064.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 3030.00 | 3070.75 | 3073.53 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 3124.00 | 3077.92 | 3076.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 3132.00 | 3107.36 | 3093.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 3120.00 | 3129.87 | 3116.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 3120.00 | 3129.87 | 3116.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 3120.00 | 3129.87 | 3116.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 3120.00 | 3129.87 | 3116.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 3136.20 | 3131.14 | 3118.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:45:00 | 3113.50 | 3131.14 | 3118.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 3135.70 | 3139.93 | 3127.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 3129.90 | 3139.93 | 3127.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 3158.70 | 3159.17 | 3145.18 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 14:15:00 | 3113.50 | 3139.93 | 3140.28 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 12:15:00 | 3152.70 | 3138.93 | 3137.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 3220.60 | 3161.34 | 3149.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 12:15:00 | 3200.30 | 3205.86 | 3186.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 13:00:00 | 3200.30 | 3205.86 | 3186.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 3173.30 | 3198.41 | 3186.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 3173.30 | 3198.41 | 3186.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 3162.60 | 3191.25 | 3184.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 3160.60 | 3191.25 | 3184.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 3197.00 | 3190.57 | 3185.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 3199.80 | 3190.57 | 3185.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 3166.00 | 3185.09 | 3184.51 | SL hit (close<static) qty=1.00 sl=3176.10 alert=retest2 |

### Cycle 83 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 3162.00 | 3180.47 | 3182.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 3091.70 | 3162.72 | 3174.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 3082.00 | 3060.08 | 3085.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 3082.00 | 3060.08 | 3085.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 3081.00 | 3064.26 | 3085.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 3069.90 | 3064.26 | 3085.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:30:00 | 3074.40 | 3068.05 | 3082.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 3099.90 | 3076.68 | 3082.65 | SL hit (close>static) qty=1.00 sl=3090.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 3132.00 | 3094.05 | 3089.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 3139.80 | 3108.86 | 3097.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 3200.00 | 3202.57 | 3179.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:45:00 | 3203.10 | 3202.57 | 3179.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 3165.50 | 3191.72 | 3180.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 3165.50 | 3191.72 | 3180.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 3152.00 | 3183.78 | 3177.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 3152.00 | 3183.78 | 3177.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 3144.30 | 3168.49 | 3171.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 3106.20 | 3156.03 | 3165.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 3123.30 | 3121.75 | 3142.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 10:00:00 | 3123.30 | 3121.75 | 3142.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 3132.30 | 3123.23 | 3137.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:45:00 | 3138.30 | 3123.23 | 3137.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 3122.00 | 3121.65 | 3134.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 3132.00 | 3121.65 | 3134.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 3108.10 | 3117.97 | 3130.38 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 3175.80 | 3138.00 | 3136.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 3192.40 | 3148.88 | 3141.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 3148.70 | 3170.17 | 3157.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 12:15:00 | 3148.70 | 3170.17 | 3157.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 3148.70 | 3170.17 | 3157.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 3148.00 | 3170.17 | 3157.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 3144.30 | 3165.00 | 3156.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 3144.30 | 3165.00 | 3156.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 3249.40 | 3207.74 | 3191.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 3255.70 | 3207.74 | 3191.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 3282.40 | 3312.46 | 3314.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 3282.40 | 3312.46 | 3314.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 3267.00 | 3295.92 | 3305.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 3131.30 | 3122.47 | 3156.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 3131.30 | 3122.47 | 3156.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 3129.60 | 3117.98 | 3128.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 3129.60 | 3117.98 | 3128.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 3121.00 | 3118.58 | 3127.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 3142.90 | 3118.58 | 3127.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 3137.60 | 3122.39 | 3128.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 3154.40 | 3122.39 | 3128.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 3109.40 | 3119.79 | 3126.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:30:00 | 3105.80 | 3116.59 | 3124.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 3138.00 | 3127.10 | 3125.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 3138.00 | 3127.10 | 3125.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 3162.10 | 3134.55 | 3129.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 3228.00 | 3236.47 | 3212.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 3228.00 | 3236.47 | 3212.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 3228.00 | 3236.47 | 3212.46 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 3182.70 | 3211.18 | 3211.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 11:15:00 | 3177.00 | 3204.34 | 3208.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 3097.90 | 3090.74 | 3122.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 09:30:00 | 3106.00 | 3090.74 | 3122.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 3107.30 | 3091.62 | 3110.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 3105.20 | 3091.62 | 3110.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 3100.00 | 3093.29 | 3109.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 3104.00 | 3093.29 | 3109.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 3112.00 | 3097.04 | 3109.48 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 3133.40 | 3116.73 | 3115.75 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 3109.80 | 3117.32 | 3117.34 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 3142.20 | 3122.30 | 3119.60 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 3109.50 | 3118.77 | 3119.02 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3125.70 | 3120.16 | 3119.63 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 3111.40 | 3119.28 | 3119.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 14:15:00 | 3098.50 | 3113.60 | 3116.65 | Break + close below crossover candle low |

### Cycle 96 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 3151.50 | 3116.69 | 3116.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 10:15:00 | 3164.50 | 3149.44 | 3137.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 12:15:00 | 3145.20 | 3151.00 | 3140.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 3145.20 | 3151.00 | 3140.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 3145.20 | 3151.00 | 3140.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 3145.20 | 3151.00 | 3140.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 3140.40 | 3148.88 | 3140.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 3137.70 | 3148.88 | 3140.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 3154.90 | 3150.08 | 3141.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 3137.30 | 3150.08 | 3141.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3149.10 | 3151.47 | 3143.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 3149.10 | 3151.47 | 3143.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 3146.40 | 3154.99 | 3150.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 3146.40 | 3154.99 | 3150.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 3148.60 | 3153.71 | 3150.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 3149.30 | 3153.71 | 3150.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 3117.10 | 3146.39 | 3147.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 3104.80 | 3138.07 | 3143.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 3145.80 | 3126.83 | 3134.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 3145.80 | 3126.83 | 3134.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 3145.80 | 3126.83 | 3134.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 3157.40 | 3126.83 | 3134.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 3147.50 | 3130.97 | 3135.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:30:00 | 3142.00 | 3138.02 | 3138.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 15:15:00 | 3165.00 | 3143.74 | 3140.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 3165.00 | 3143.74 | 3140.86 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 3121.80 | 3139.20 | 3140.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 3112.00 | 3128.44 | 3134.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 3116.80 | 3114.20 | 3123.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 3116.80 | 3114.20 | 3123.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3116.80 | 3114.20 | 3123.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 3122.80 | 3114.20 | 3123.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 3135.00 | 3118.36 | 3124.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 3135.00 | 3118.36 | 3124.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 3118.50 | 3118.39 | 3124.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 3108.00 | 3120.85 | 3123.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:00:00 | 3106.70 | 3117.44 | 3120.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 3107.90 | 3053.36 | 3046.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 3107.90 | 3053.36 | 3046.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 3179.60 | 3100.63 | 3081.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 3214.80 | 3226.03 | 3188.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 10:00:00 | 3214.80 | 3226.03 | 3188.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 3203.00 | 3221.43 | 3189.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 3203.00 | 3221.43 | 3189.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 3192.70 | 3214.13 | 3208.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 3192.70 | 3214.13 | 3208.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 3161.90 | 3203.68 | 3203.99 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 3276.00 | 3194.57 | 3185.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 3305.00 | 3216.66 | 3196.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 3283.20 | 3283.51 | 3248.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:30:00 | 3282.00 | 3283.51 | 3248.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 3288.00 | 3292.93 | 3282.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 3287.00 | 3292.93 | 3282.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 3287.90 | 3291.93 | 3282.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 3276.00 | 3291.93 | 3282.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 3324.30 | 3341.09 | 3323.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 3324.30 | 3341.09 | 3323.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 3341.30 | 3341.13 | 3324.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:45:00 | 3352.10 | 3339.76 | 3330.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 13:45:00 | 3349.00 | 3339.57 | 3332.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:15:00 | 3358.00 | 3339.57 | 3332.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:30:00 | 3350.00 | 3348.57 | 3339.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 3301.40 | 3339.13 | 3336.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 3301.40 | 3339.13 | 3336.41 | SL hit (close<static) qty=1.00 sl=3308.80 alert=retest2 |

### Cycle 103 — SELL (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 12:15:00 | 3303.90 | 3332.09 | 3333.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 3267.30 | 3303.87 | 3317.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 11:15:00 | 3163.00 | 3156.94 | 3198.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 11:45:00 | 3162.90 | 3156.94 | 3198.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 3192.90 | 3176.14 | 3185.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 3192.90 | 3176.14 | 3185.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 3197.00 | 3180.31 | 3186.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 3193.30 | 3180.31 | 3186.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 3158.90 | 3177.31 | 3184.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 3170.60 | 3177.31 | 3184.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 3172.80 | 3152.83 | 3165.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 3172.80 | 3152.83 | 3165.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 3169.10 | 3156.08 | 3165.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 12:30:00 | 3153.00 | 3153.87 | 3163.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 3149.50 | 3154.00 | 3161.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 3151.80 | 3154.56 | 3160.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:15:00 | 3144.90 | 3151.55 | 3157.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 3161.70 | 3153.62 | 3157.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 3161.70 | 3153.62 | 3157.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 3165.70 | 3156.04 | 3158.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 3157.90 | 3156.04 | 3158.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3155.70 | 3155.97 | 3158.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 3145.00 | 3155.97 | 3158.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 3147.60 | 3149.83 | 3154.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 3131.30 | 3114.70 | 3113.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 3131.30 | 3114.70 | 3113.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 3144.30 | 3124.11 | 3118.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 3125.00 | 3126.17 | 3120.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 3125.00 | 3126.17 | 3120.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 3125.00 | 3126.17 | 3120.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 3142.60 | 3130.86 | 3126.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 14:15:00 | 3095.00 | 3123.51 | 3124.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 3095.00 | 3123.51 | 3124.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 3090.00 | 3116.81 | 3121.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 3049.00 | 3034.75 | 3055.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:30:00 | 3046.40 | 3034.75 | 3055.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 3060.50 | 3039.90 | 3056.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 3059.00 | 3039.90 | 3056.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 3076.50 | 3047.22 | 3057.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 3076.60 | 3047.22 | 3057.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 3070.40 | 3059.88 | 3061.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 3077.40 | 3059.88 | 3061.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 3060.00 | 3059.91 | 3061.27 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 3073.20 | 3064.18 | 3063.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 3094.00 | 3072.40 | 3067.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 3071.50 | 3078.96 | 3072.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 3071.50 | 3078.96 | 3072.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 3071.50 | 3078.96 | 3072.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:45:00 | 3068.50 | 3078.96 | 3072.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 3092.00 | 3081.57 | 3074.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 15:00:00 | 3096.90 | 3085.61 | 3077.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:00:00 | 3096.10 | 3092.73 | 3088.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:00:00 | 3096.90 | 3088.98 | 3087.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 11:30:00 | 3095.00 | 3109.82 | 3105.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 3093.40 | 3102.85 | 3103.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 3093.40 | 3102.85 | 3103.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 3010.00 | 3084.28 | 3094.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 3069.00 | 3067.74 | 3084.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 3069.00 | 3067.74 | 3084.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3069.00 | 3067.74 | 3084.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 3069.00 | 3067.74 | 3084.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 3065.20 | 3067.23 | 3082.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:30:00 | 3053.60 | 3071.53 | 3080.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 3059.90 | 3071.53 | 3080.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 2906.90 | 2967.17 | 2979.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2940.10 | 2936.88 | 2955.28 | SL hit (close>ema200) qty=0.50 sl=2936.88 alert=retest2 |

### Cycle 108 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 2929.00 | 2899.45 | 2899.42 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 2889.50 | 2902.60 | 2903.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 2858.40 | 2893.76 | 2899.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 2895.00 | 2888.46 | 2895.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 2895.00 | 2888.46 | 2895.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 2895.00 | 2888.46 | 2895.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 2895.00 | 2888.46 | 2895.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 2893.10 | 2889.39 | 2895.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 2921.00 | 2889.39 | 2895.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 2891.70 | 2889.85 | 2894.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 2912.60 | 2889.85 | 2894.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 2938.80 | 2899.64 | 2898.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 2980.00 | 2915.71 | 2906.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 3046.90 | 3069.17 | 3036.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 3046.90 | 3069.17 | 3036.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 3046.90 | 3069.17 | 3036.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 3040.90 | 3069.17 | 3036.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 3028.30 | 3061.13 | 3038.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 3028.30 | 3061.13 | 3038.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 3037.50 | 3056.40 | 3038.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 13:30:00 | 3064.90 | 3051.30 | 3037.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 2985.00 | 3026.71 | 3028.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 2985.00 | 3026.71 | 3028.02 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 3068.10 | 3034.41 | 3030.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 3092.50 | 3046.03 | 3036.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 3257.30 | 3266.63 | 3212.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 3256.50 | 3266.63 | 3212.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 3197.40 | 3273.43 | 3258.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:45:00 | 3197.40 | 3273.43 | 3258.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 3183.80 | 3255.51 | 3251.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:15:00 | 3139.00 | 3255.51 | 3251.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 3139.00 | 3232.21 | 3241.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 09:15:00 | 3111.80 | 3208.12 | 3229.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 3134.60 | 3119.97 | 3141.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 3134.60 | 3119.97 | 3141.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 3134.60 | 3119.97 | 3141.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 3137.20 | 3119.97 | 3141.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 3133.20 | 3122.61 | 3141.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:30:00 | 3138.80 | 3122.61 | 3141.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 3146.10 | 3129.45 | 3141.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 3120.80 | 3137.27 | 3142.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 3133.40 | 3139.44 | 3142.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 3157.60 | 3142.86 | 3143.66 | SL hit (close>static) qty=1.00 sl=3154.00 alert=retest2 |

### Cycle 114 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 3172.80 | 3140.11 | 3138.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 3200.00 | 3152.09 | 3144.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 11:15:00 | 3166.10 | 3178.22 | 3163.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 11:15:00 | 3166.10 | 3178.22 | 3163.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 3166.10 | 3178.22 | 3163.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 3166.10 | 3178.22 | 3163.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 3178.00 | 3181.91 | 3171.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 3172.70 | 3181.91 | 3171.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 3186.00 | 3189.50 | 3180.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 3194.10 | 3189.50 | 3180.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 3190.20 | 3189.64 | 3181.36 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 3147.20 | 3173.05 | 3175.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 3127.20 | 3163.88 | 3170.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 3174.90 | 3149.88 | 3161.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 3174.90 | 3149.88 | 3161.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 3174.90 | 3149.88 | 3161.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:15:00 | 3191.70 | 3149.88 | 3161.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 3226.20 | 3165.14 | 3167.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 3218.90 | 3165.14 | 3167.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 3269.10 | 3185.93 | 3176.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 13:15:00 | 3304.80 | 3221.73 | 3195.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 14:15:00 | 3206.70 | 3218.73 | 3196.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 15:00:00 | 3206.70 | 3218.73 | 3196.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 3190.20 | 3213.02 | 3195.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 3247.30 | 3213.02 | 3195.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 3321.90 | 3344.52 | 3344.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2026-03-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 15:15:00 | 3321.90 | 3344.52 | 3344.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 3251.00 | 3325.81 | 3336.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 11:15:00 | 3221.50 | 3210.53 | 3252.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:00:00 | 3221.50 | 3210.53 | 3252.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 3285.10 | 3225.30 | 3242.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 3284.70 | 3225.30 | 3242.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 3269.20 | 3234.08 | 3245.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:15:00 | 3253.10 | 3234.08 | 3245.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 3296.20 | 3246.50 | 3249.99 | SL hit (close>static) qty=1.00 sl=3292.50 alert=retest2 |

### Cycle 118 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 3310.10 | 3259.22 | 3255.46 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 3195.60 | 3254.48 | 3255.80 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 3304.80 | 3248.90 | 3244.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 3335.70 | 3293.22 | 3280.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 3288.00 | 3307.13 | 3293.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 3288.00 | 3307.13 | 3293.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 3288.00 | 3307.13 | 3293.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:45:00 | 3292.80 | 3307.13 | 3293.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 3257.50 | 3297.20 | 3290.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 3257.50 | 3297.20 | 3290.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 3278.10 | 3293.38 | 3289.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:15:00 | 3254.60 | 3293.38 | 3289.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3231.50 | 3281.01 | 3284.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 3210.00 | 3260.55 | 3273.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 11:15:00 | 3160.30 | 3158.31 | 3191.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 11:45:00 | 3173.00 | 3158.31 | 3191.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 3193.00 | 3169.79 | 3184.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 3188.80 | 3169.79 | 3184.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 3206.10 | 3177.06 | 3186.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:30:00 | 3215.00 | 3177.06 | 3186.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 3222.60 | 3194.80 | 3192.91 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 3149.80 | 3191.67 | 3192.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 3117.90 | 3170.14 | 3182.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 3172.90 | 3130.12 | 3153.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 3172.90 | 3130.12 | 3153.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 3172.90 | 3130.12 | 3153.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 3172.90 | 3130.12 | 3153.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 3173.00 | 3138.70 | 3155.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 3171.00 | 3138.70 | 3155.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 3155.00 | 3143.00 | 3154.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 3155.00 | 3143.00 | 3154.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 3141.50 | 3142.70 | 3153.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 3137.10 | 3142.70 | 3153.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 2980.24 | 3079.32 | 3119.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 3015.40 | 3009.63 | 3051.31 | SL hit (close>ema200) qty=0.50 sl=3009.63 alert=retest2 |

### Cycle 124 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 3112.50 | 3064.56 | 3058.62 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 3045.00 | 3059.98 | 3061.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2980.50 | 3044.09 | 3054.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3051.10 | 2988.82 | 3013.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3051.10 | 2988.82 | 3013.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3051.10 | 2988.82 | 3013.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 3051.10 | 2988.82 | 3013.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 2999.20 | 2990.89 | 3011.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 2920.70 | 3013.49 | 3017.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 2986.00 | 2991.64 | 2999.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 3011.70 | 3004.02 | 3003.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 3011.70 | 3004.02 | 3003.86 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 2969.20 | 2997.05 | 3000.71 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 3035.10 | 3007.13 | 3004.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 3046.70 | 3015.04 | 3008.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 14:15:00 | 3340.00 | 3347.42 | 3299.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 3340.00 | 3347.42 | 3299.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 3846.50 | 3852.66 | 3817.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 3825.00 | 3852.66 | 3817.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 3774.50 | 3837.03 | 3813.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 3774.50 | 3837.03 | 3813.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 3800.20 | 3829.67 | 3812.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 3776.30 | 3829.67 | 3812.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 3789.50 | 3814.65 | 3807.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 14:30:00 | 3810.00 | 3813.24 | 3807.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 3808.30 | 3824.38 | 3825.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 11:15:00 | 3808.30 | 3824.38 | 3825.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 3792.40 | 3817.04 | 3821.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 3819.00 | 3801.47 | 3810.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 11:15:00 | 3819.00 | 3801.47 | 3810.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 3819.00 | 3801.47 | 3810.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 3819.00 | 3801.47 | 3810.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 3805.00 | 3802.18 | 3809.96 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 3905.80 | 3827.28 | 3819.29 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 3807.50 | 3830.61 | 3832.91 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 3848.90 | 3833.42 | 3832.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 3864.00 | 3839.53 | 3835.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 3821.60 | 3847.06 | 3842.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 3821.60 | 3847.06 | 3842.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 3821.60 | 3847.06 | 3842.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 3818.00 | 3847.06 | 3842.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 3813.70 | 3840.39 | 3839.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 3813.70 | 3840.39 | 3839.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 3804.10 | 3833.13 | 3836.50 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-26 11:30:00 | 3796.38 | 2024-07-08 11:15:00 | 3821.38 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2024-06-26 12:15:00 | 3795.83 | 2024-07-08 11:15:00 | 3821.38 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2024-06-26 13:00:00 | 3781.24 | 2024-07-08 11:15:00 | 3821.38 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2024-06-27 09:30:00 | 3792.77 | 2024-07-08 11:15:00 | 3821.38 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2024-07-01 09:15:00 | 3868.56 | 2024-07-08 11:15:00 | 3821.38 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-07-02 12:15:00 | 3836.07 | 2024-07-08 11:15:00 | 3821.38 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-07-02 13:30:00 | 3842.59 | 2024-07-08 11:15:00 | 3821.38 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-07-16 10:15:00 | 3782.85 | 2024-07-18 09:15:00 | 3593.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 10:15:00 | 3782.85 | 2024-07-19 15:15:00 | 3404.57 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-07-30 14:30:00 | 3509.80 | 2024-07-31 09:15:00 | 3471.07 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-07-31 11:30:00 | 3494.07 | 2024-08-01 13:15:00 | 3482.48 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-08-06 13:30:00 | 3304.98 | 2024-08-07 14:15:00 | 3400.38 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2024-08-08 09:15:00 | 3323.37 | 2024-08-08 13:15:00 | 3384.07 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-08-08 10:00:00 | 3322.63 | 2024-08-08 13:15:00 | 3384.07 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-08-08 10:45:00 | 3322.11 | 2024-08-08 13:15:00 | 3384.07 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-08-13 11:45:00 | 3460.56 | 2024-08-21 13:15:00 | 3503.71 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2024-08-14 12:00:00 | 3467.67 | 2024-08-21 13:15:00 | 3503.71 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2024-08-23 12:15:00 | 3509.80 | 2024-09-05 10:15:00 | 3334.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-27 09:45:00 | 3509.80 | 2024-09-05 10:15:00 | 3334.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-23 12:15:00 | 3509.80 | 2024-09-05 11:15:00 | 3357.03 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2024-08-27 09:45:00 | 3509.80 | 2024-09-05 11:15:00 | 3357.03 | STOP_HIT | 0.50 | 4.35% |
| BUY | retest2 | 2024-09-17 14:30:00 | 3355.69 | 2024-09-19 10:15:00 | 3303.19 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-09-18 10:30:00 | 3365.66 | 2024-09-19 10:15:00 | 3303.19 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-09-19 09:15:00 | 3392.45 | 2024-09-19 10:15:00 | 3303.19 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-09-27 10:45:00 | 3572.89 | 2024-10-04 12:15:00 | 3613.23 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2024-09-27 11:15:00 | 3566.97 | 2024-10-04 12:15:00 | 3613.23 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-09-27 11:45:00 | 3577.41 | 2024-10-04 12:15:00 | 3613.23 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 3852.50 | 2024-10-22 09:15:00 | 3659.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 3852.50 | 2024-10-23 14:15:00 | 3467.25 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-08 13:15:00 | 3579.45 | 2024-11-12 09:15:00 | 3441.62 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2024-11-18 10:30:00 | 3318.40 | 2024-11-22 10:15:00 | 3460.19 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2024-11-18 11:00:00 | 3323.80 | 2024-11-22 10:15:00 | 3460.19 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2024-11-18 13:00:00 | 3320.81 | 2024-11-22 10:15:00 | 3460.19 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2024-11-19 09:45:00 | 3315.57 | 2024-11-22 10:15:00 | 3460.19 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2024-11-19 13:30:00 | 3350.49 | 2024-11-22 10:15:00 | 3460.19 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2024-12-02 15:15:00 | 3766.32 | 2024-12-17 12:15:00 | 3880.12 | STOP_HIT | 1.00 | 3.02% |
| SELL | retest2 | 2024-12-30 10:30:00 | 3247.44 | 2025-01-01 13:15:00 | 3298.52 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-12-30 11:30:00 | 3253.08 | 2025-01-01 13:15:00 | 3298.52 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-12-30 12:00:00 | 3250.87 | 2025-01-01 13:15:00 | 3298.52 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-12-31 09:15:00 | 3228.44 | 2025-01-01 13:15:00 | 3298.52 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-01-02 13:15:00 | 3298.81 | 2025-01-06 10:15:00 | 3258.57 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-01-09 11:45:00 | 3156.83 | 2025-01-13 09:15:00 | 2998.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:45:00 | 3156.83 | 2025-01-14 15:15:00 | 2924.05 | STOP_HIT | 0.50 | 7.37% |
| SELL | retest2 | 2025-01-29 09:15:00 | 2814.80 | 2025-01-30 09:15:00 | 2933.59 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2025-02-13 13:15:00 | 2561.26 | 2025-02-17 09:15:00 | 2433.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:15:00 | 2561.26 | 2025-02-18 09:15:00 | 2436.10 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest2 | 2025-02-25 11:15:00 | 2410.85 | 2025-02-28 09:15:00 | 2290.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:15:00 | 2410.85 | 2025-03-03 11:15:00 | 2320.79 | STOP_HIT | 0.50 | 3.74% |
| BUY | retest2 | 2025-03-07 14:15:00 | 2571.45 | 2025-03-10 10:15:00 | 2494.92 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-03-12 10:15:00 | 2426.34 | 2025-03-13 09:15:00 | 2493.15 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-03-12 10:45:00 | 2412.34 | 2025-03-13 09:15:00 | 2493.15 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-03-12 13:30:00 | 2423.18 | 2025-03-13 09:15:00 | 2493.15 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-04-01 10:45:00 | 2593.03 | 2025-04-03 09:15:00 | 2622.61 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-04-01 11:30:00 | 2592.55 | 2025-04-03 09:15:00 | 2622.61 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-04-02 09:15:00 | 2587.48 | 2025-04-03 09:15:00 | 2622.61 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-04-11 10:45:00 | 2758.45 | 2025-04-11 13:15:00 | 2717.60 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-04-15 09:15:00 | 2754.50 | 2025-04-25 11:15:00 | 2814.50 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2025-05-06 15:00:00 | 2946.50 | 2025-05-08 09:15:00 | 2909.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-05-07 09:15:00 | 2968.80 | 2025-05-08 09:15:00 | 2909.50 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-05-07 09:45:00 | 2946.20 | 2025-05-08 09:15:00 | 2909.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest1 | 2025-05-13 15:15:00 | 2936.60 | 2025-05-16 09:15:00 | 3083.43 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-13 15:15:00 | 2936.60 | 2025-05-19 09:15:00 | 3049.90 | STOP_HIT | 0.50 | 3.86% |
| BUY | retest2 | 2025-05-20 10:45:00 | 3148.00 | 2025-05-30 15:15:00 | 3267.00 | STOP_HIT | 1.00 | 3.78% |
| BUY | retest2 | 2025-05-20 13:00:00 | 3147.00 | 2025-05-30 15:15:00 | 3267.00 | STOP_HIT | 1.00 | 3.81% |
| BUY | retest2 | 2025-05-21 09:30:00 | 3145.60 | 2025-05-30 15:15:00 | 3267.00 | STOP_HIT | 1.00 | 3.86% |
| BUY | retest2 | 2025-05-21 10:00:00 | 3163.50 | 2025-05-30 15:15:00 | 3267.00 | STOP_HIT | 1.00 | 3.27% |
| BUY | retest2 | 2025-05-27 09:15:00 | 3281.70 | 2025-05-30 15:15:00 | 3267.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-06-05 10:15:00 | 3304.80 | 2025-06-06 11:15:00 | 3309.70 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-06-06 10:30:00 | 3306.60 | 2025-06-06 11:15:00 | 3309.70 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-06-11 12:30:00 | 3366.00 | 2025-06-12 09:15:00 | 3307.90 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-06-18 13:15:00 | 3328.70 | 2025-06-19 10:15:00 | 3284.60 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-06-25 10:15:00 | 3113.60 | 2025-06-27 09:15:00 | 3254.00 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2025-06-25 12:00:00 | 3105.60 | 2025-06-27 09:15:00 | 3254.00 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest2 | 2025-07-03 09:15:00 | 3304.70 | 2025-07-04 10:15:00 | 3263.20 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-16 09:30:00 | 3092.20 | 2025-07-18 11:15:00 | 3136.30 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-07-16 11:00:00 | 3107.40 | 2025-07-18 11:15:00 | 3136.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-17 13:45:00 | 3104.20 | 2025-07-18 11:15:00 | 3136.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-17 15:15:00 | 3105.70 | 2025-07-18 11:15:00 | 3136.30 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-22 11:30:00 | 3146.60 | 2025-07-23 10:15:00 | 3119.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-22 13:30:00 | 3147.60 | 2025-07-23 10:15:00 | 3119.20 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-22 14:00:00 | 3146.00 | 2025-07-23 10:15:00 | 3119.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-22 14:30:00 | 3149.30 | 2025-07-23 10:15:00 | 3119.20 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-30 10:15:00 | 3039.90 | 2025-07-30 14:15:00 | 3073.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-31 09:15:00 | 3038.50 | 2025-08-05 12:15:00 | 3040.40 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-07-31 13:15:00 | 3036.40 | 2025-08-05 12:15:00 | 3040.40 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-08-05 09:30:00 | 3028.20 | 2025-08-05 12:15:00 | 3040.40 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-08-07 14:15:00 | 3107.20 | 2025-08-08 14:15:00 | 3032.40 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-08-25 11:15:00 | 3199.80 | 2025-08-25 14:15:00 | 3166.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-08-29 12:15:00 | 3069.90 | 2025-09-01 10:15:00 | 3099.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-29 14:30:00 | 3074.40 | 2025-09-01 10:15:00 | 3099.90 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-15 10:15:00 | 3255.70 | 2025-09-22 10:15:00 | 3282.40 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2025-10-01 11:30:00 | 3105.80 | 2025-10-03 12:15:00 | 3138.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-29 13:30:00 | 3142.00 | 2025-10-29 15:15:00 | 3165.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-04 11:15:00 | 3108.00 | 2025-11-13 09:15:00 | 3107.90 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-11-06 10:00:00 | 3106.70 | 2025-11-13 09:15:00 | 3107.90 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-12-04 09:45:00 | 3352.10 | 2025-12-05 11:15:00 | 3301.40 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-12-04 13:45:00 | 3349.00 | 2025-12-05 11:15:00 | 3301.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-04 14:15:00 | 3358.00 | 2025-12-05 11:15:00 | 3301.40 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-05 10:30:00 | 3350.00 | 2025-12-05 11:15:00 | 3301.40 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-12-15 12:30:00 | 3153.00 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-12-16 09:15:00 | 3149.50 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-12-16 10:15:00 | 3151.80 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2025-12-16 13:15:00 | 3144.90 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-12-17 10:15:00 | 3145.00 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-12-17 14:00:00 | 3147.60 | 2025-12-22 12:15:00 | 3131.30 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-12-24 12:15:00 | 3142.60 | 2025-12-24 14:15:00 | 3095.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-01-02 15:00:00 | 3096.90 | 2026-01-08 13:15:00 | 3093.40 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-01-06 10:00:00 | 3096.10 | 2026-01-08 13:15:00 | 3093.40 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2026-01-06 14:00:00 | 3096.90 | 2026-01-08 13:15:00 | 3093.40 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-01-08 11:30:00 | 3095.00 | 2026-01-08 13:15:00 | 3093.40 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-01-09 14:30:00 | 3053.60 | 2026-01-16 09:15:00 | 2906.90 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2026-01-09 14:30:00 | 3053.60 | 2026-01-19 09:15:00 | 2940.10 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2026-01-09 15:00:00 | 3059.90 | 2026-01-20 11:15:00 | 2900.92 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2026-01-09 15:00:00 | 3059.90 | 2026-01-21 11:15:00 | 2886.10 | STOP_HIT | 0.50 | 5.68% |
| BUY | retest2 | 2026-02-01 13:30:00 | 3064.90 | 2026-02-01 15:15:00 | 2985.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-02-12 09:15:00 | 3120.80 | 2026-02-12 13:15:00 | 3157.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-02-12 12:15:00 | 3133.40 | 2026-02-12 13:15:00 | 3157.60 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3102.80 | 2026-02-13 11:15:00 | 3175.80 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-02-13 13:45:00 | 3129.30 | 2026-02-16 11:15:00 | 3159.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-02-23 09:15:00 | 3247.30 | 2026-03-02 15:15:00 | 3321.90 | STOP_HIT | 1.00 | 2.30% |
| SELL | retest2 | 2026-03-06 11:15:00 | 3253.10 | 2026-03-06 11:15:00 | 3296.20 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-03-20 14:15:00 | 3137.10 | 2026-03-23 10:15:00 | 2980.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:15:00 | 3137.10 | 2026-03-24 11:15:00 | 3015.40 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2026-04-02 09:15:00 | 2920.70 | 2026-04-06 09:15:00 | 3011.70 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2026-04-02 14:30:00 | 2986.00 | 2026-04-06 09:15:00 | 3011.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-04-24 14:30:00 | 3810.00 | 2026-04-29 11:15:00 | 3808.30 | STOP_HIT | 1.00 | -0.04% |
