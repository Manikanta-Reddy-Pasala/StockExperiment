# TITAN (TITAN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 4359.60
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 9 |
| PENDING | 35 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 6 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 21
- **Target hits / Stop hits / Partials:** 0 / 24 / 0
- **Avg / median % per leg:** -1.42% / -1.46%
- **Sum % (uncompounded):** -33.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 3 | 17.6% | 0 | 17 | 0 | -0.94% | -16.0% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | 0.58% | 1.7% |
| BUY @ 3rd Alert (retest2) | 14 | 1 | 7.1% | 0 | 14 | 0 | -1.27% | -17.8% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.56% | -18.0% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.25% | -9.8% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.05% | -8.2% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 6 | 0 | -1.34% | -8.0% |
| retest2 (combined) | 18 | 1 | 5.6% | 0 | 18 | 0 | -1.44% | -26.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 11:15:00 | 3612.25 | 3398.86 | 3398.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 3631.90 | 3427.19 | 3413.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 12:15:00 | 3682.10 | 3695.82 | 3604.78 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-10-04 10:15:00 | 3724.90 | 3695.75 | 3606.99 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 11:15:00 | 3742.35 | 3696.21 | 3607.66 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 3582.00 | 3694.63 | 3609.06 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 3609.06 | 3694.63 | 3609.06 | SL hit qty=1.00 sl=3609.06 alert=retest1 |

### Cycle 2 — SELL (started 2024-10-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 13:15:00 | 3340.05 | 3557.76 | 3557.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 3309.80 | 3550.82 | 3554.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 14:15:00 | 3314.00 | 3298.71 | 3387.28 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-11-28 10:15:00 | 3250.80 | 3300.32 | 3378.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:15:00 | 3228.10 | 3299.60 | 3377.41 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 3368.00 | 3295.84 | 3365.36 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-12-04 10:15:00 | 3365.36 | 3295.84 | 3365.36 | SL hit qty=1.00 sl=3365.36 alert=retest1 |
| Cross detected — sustain check pending | 2024-12-26 11:15:00 | 3345.30 | 3372.04 | 3386.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 12:15:00 | 3342.85 | 3371.75 | 3386.40 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-02 14:15:00 | 3375.00 | 3345.68 | 3369.76 | SL hit qty=1.00 sl=3375.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-01-10 14:15:00 | 3437.55 | 3389.43 | 3389.37 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-01-14 11:15:00 | 3335.05 | 3389.81 | 3389.60 | ENTRY2 cross detected — sustain check pending (15m) |

### Cycle 3 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 3333.65 | 3389.25 | 3389.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 13:15:00 | 3321.50 | 3388.58 | 3388.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 3394.00 | 3374.52 | 3381.41 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 3394.00 | 3374.52 | 3381.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 3394.00 | 3374.52 | 3381.41 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-22 13:15:00 | 3344.10 | 3374.22 | 3380.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-22 14:15:00 | 3358.30 | 3374.06 | 3380.58 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-27 13:15:00 | 3316.85 | 3376.58 | 3381.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 14:15:00 | 3318.50 | 3376.00 | 3380.99 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-28 14:15:00 | 3329.45 | 3373.30 | 3379.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:15:00 | 3327.10 | 3372.84 | 3379.18 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 3395.00 | 3373.08 | 3378.82 | SL hit qty=1.00 sl=3395.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 3395.00 | 3373.08 | 3378.82 | SL hit qty=1.00 sl=3395.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 10:15:00 | 3590.00 | 3384.98 | 3384.65 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 11:15:00 | 3265.10 | 3389.25 | 3389.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 13:15:00 | 3242.85 | 3386.52 | 3387.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 3167.65 | 3153.07 | 3233.43 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-03-24 09:15:00 | 3090.75 | 3155.27 | 3229.53 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 10:15:00 | 3093.55 | 3154.65 | 3228.85 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-04 09:15:00 | 3088.00 | 3119.51 | 3190.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 10:15:00 | 3098.30 | 3119.30 | 3190.42 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 3133.45 | 3109.84 | 3181.07 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 3181.07 | 3109.84 | 3181.07 | SL hit qty=1.00 sl=3181.07 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 3181.07 | 3109.84 | 3181.07 | SL hit qty=1.00 sl=3181.07 alert=retest1 |
| Cross detected — sustain check pending | 2025-04-08 14:15:00 | 3120.60 | 3111.20 | 3180.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:15:00 | 3122.30 | 3111.31 | 3179.71 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 3212.55 | 3115.98 | 3179.40 | SL hit qty=1.00 sl=3212.55 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 3372.30 | 3222.12 | 3221.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 3379.70 | 3223.69 | 3222.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 3498.70 | 3499.55 | 3425.60 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-12 12:15:00 | 3512.00 | 3499.67 | 3426.03 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-12 13:15:00 | 3460.80 | 3499.28 | 3426.20 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 3408.90 | 3497.42 | 3426.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 3408.90 | 3497.42 | 3426.35 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-13 11:15:00 | 3427.00 | 3496.00 | 3426.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 3432.00 | 3495.36 | 3426.37 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-16 10:15:00 | 3427.40 | 3491.81 | 3426.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 3429.60 | 3491.19 | 3426.30 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-18 09:15:00 | 3427.40 | 3483.12 | 3425.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-18 10:15:00 | 3418.40 | 3482.48 | 3425.89 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-18 11:15:00 | 3425.10 | 3481.91 | 3425.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 3427.50 | 3481.37 | 3425.90 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-10 12:15:00 | 3427.00 | 3556.20 | 3498.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-10 13:15:00 | 3421.90 | 3554.87 | 3497.80 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-11 13:15:00 | 3383.00 | 3544.36 | 3494.42 | SL hit qty=1.00 sl=3383.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 13:15:00 | 3383.00 | 3544.36 | 3494.42 | SL hit qty=1.00 sl=3383.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 13:15:00 | 3383.00 | 3544.36 | 3494.42 | SL hit qty=1.00 sl=3383.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-17 10:15:00 | 3436.00 | 3512.84 | 3483.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:15:00 | 3439.20 | 3512.10 | 3483.26 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 3442.60 | 3511.41 | 3483.05 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 3383.00 | 3501.15 | 3479.31 | SL hit qty=1.00 sl=3383.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-22 09:15:00 | 3471.00 | 3496.18 | 3477.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 3483.70 | 3496.06 | 3477.56 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 3438.00 | 3490.14 | 3476.76 | SL hit qty=1.00 sl=3438.00 alert=retest2 |
| CROSSOVER_SKIP | 2025-07-31 12:15:00 | 3355.10 | 3465.01 | 3465.02 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-08-06 11:15:00 | 3449.00 | 3441.80 | 3452.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 3452.70 | 3441.91 | 3452.47 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-06 13:15:00 | 3438.00 | 3441.78 | 3452.35 | SL hit qty=1.00 sl=3438.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-08 09:15:00 | 3469.40 | 3439.27 | 3450.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 3466.70 | 3439.55 | 3450.63 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 3438.00 | 3440.66 | 3450.87 | SL hit qty=1.00 sl=3438.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-11 10:15:00 | 3451.00 | 3440.76 | 3450.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 11:15:00 | 3465.00 | 3441.00 | 3450.94 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 3557.80 | 3459.01 | 3458.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 3557.80 | 3459.01 | 3458.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 3562.70 | 3461.04 | 3459.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 3576.80 | 3579.79 | 3536.72 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 10:15:00 | 3526.10 | 3578.82 | 3537.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 3526.10 | 3578.82 | 3537.30 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-09-29 11:15:00 | 3380.00 | 3511.22 | 3511.85 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-10-08 09:15:00 | 3571.00 | 3479.27 | 3493.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-08 10:15:00 | 3560.00 | 3480.08 | 3494.24 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-08 12:15:00 | 3573.90 | 3481.80 | 3494.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-08 13:15:00 | 3562.30 | 3482.60 | 3495.30 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-08 14:15:00 | 3566.10 | 3483.43 | 3495.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 3567.00 | 3484.26 | 3496.01 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-10 11:15:00 | 3569.20 | 3490.53 | 3498.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-10 12:15:00 | 3556.50 | 3491.18 | 3498.93 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 3515.00 | 3492.66 | 3499.52 | SL hit qty=1.00 sl=3515.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-16 09:15:00 | 3636.30 | 3501.02 | 3503.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 3643.80 | 3502.44 | 3503.90 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 3641.40 | 3506.56 | 3505.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 3641.40 | 3506.56 | 3505.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3694.00 | 3511.06 | 3508.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 13:15:00 | 3810.00 | 3813.15 | 3724.85 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-09 11:15:00 | 3841.60 | 3811.65 | 3734.84 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 12:15:00 | 3867.60 | 3812.20 | 3735.51 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-11 13:15:00 | 3839.40 | 3815.97 | 3743.01 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 14:15:00 | 3845.10 | 3816.26 | 3743.52 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 3999.30 | 4056.38 | 3958.36 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 3958.36 | 4056.38 | 3958.36 | SL hit qty=1.00 sl=3958.36 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 3958.36 | 4056.38 | 3958.36 | SL hit qty=1.00 sl=3958.36 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 4068.60 | 4028.80 | 3957.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 4085.10 | 4029.36 | 3957.82 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 3955.00 | 4154.68 | 4111.81 | SL hit qty=1.00 sl=3955.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 4013.60 | 4121.43 | 4097.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 4059.90 | 4120.81 | 4097.29 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 4024.50 | 4115.86 | 4095.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 4024.80 | 4114.95 | 4095.23 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 3955.00 | 4108.51 | 4092.46 | SL hit qty=1.00 sl=3955.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 3955.00 | 4108.51 | 4092.46 | SL hit qty=1.00 sl=3955.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 4073.80 | 4098.84 | 4088.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 4083.20 | 4098.68 | 4088.06 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 4085.00 | 4098.54 | 4088.04 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-01 12:15:00 | 4089.70 | 4098.46 | 4088.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-01 13:15:00 | 4072.00 | 4098.19 | 4087.97 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 3955.00 | 4096.42 | 4087.23 | SL hit qty=1.00 sl=3955.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-02 14:15:00 | 4100.40 | 4092.55 | 4085.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-02 15:15:00 | 4080.00 | 4092.43 | 4085.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-06 09:15:00 | 4174.90 | 4093.25 | 4085.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 4175.00 | 4094.06 | 4086.35 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-04 11:15:00 | 3742.35 | 2024-10-07 09:15:00 | 3609.06 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest1 | 2024-11-28 11:15:00 | 3228.10 | 2024-12-04 10:15:00 | 3365.36 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2024-12-26 12:15:00 | 3342.85 | 2025-01-02 14:15:00 | 3375.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-01-27 14:15:00 | 3318.50 | 2025-01-31 09:15:00 | 3395.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-01-28 15:15:00 | 3327.10 | 2025-01-31 09:15:00 | 3395.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest1 | 2025-03-24 10:15:00 | 3093.55 | 2025-04-08 09:15:00 | 3181.07 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest1 | 2025-04-04 10:15:00 | 3098.30 | 2025-04-08 09:15:00 | 3181.07 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-04-08 15:15:00 | 3122.30 | 2025-04-11 09:15:00 | 3212.55 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-06-13 12:15:00 | 3432.00 | 2025-07-11 13:15:00 | 3383.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-16 11:15:00 | 3429.60 | 2025-07-11 13:15:00 | 3383.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-06-18 12:15:00 | 3427.50 | 2025-07-11 13:15:00 | 3383.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-17 11:15:00 | 3439.20 | 2025-07-21 09:15:00 | 3383.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-07-22 10:15:00 | 3483.70 | 2025-07-28 09:15:00 | 3438.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-08-06 12:15:00 | 3452.70 | 2025-08-06 13:15:00 | 3438.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-08-08 10:15:00 | 3466.70 | 2025-08-11 09:15:00 | 3438.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-08-11 11:15:00 | 3465.00 | 2025-08-19 10:15:00 | 3557.80 | STOP_HIT | 1.00 | 2.68% |
| BUY | retest2 | 2025-10-08 15:15:00 | 3567.00 | 2025-10-13 09:15:00 | 3515.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-16 10:15:00 | 3643.80 | 2025-10-16 13:15:00 | 3641.40 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest1 | 2025-12-09 12:15:00 | 3867.60 | 2026-01-27 14:15:00 | 3958.36 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest1 | 2025-12-11 14:15:00 | 3845.10 | 2026-01-27 14:15:00 | 3958.36 | STOP_HIT | 1.00 | 2.95% |
| BUY | retest2 | 2026-02-03 10:15:00 | 4085.10 | 2026-03-23 09:15:00 | 3955.00 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-03-25 10:15:00 | 4059.90 | 2026-03-30 09:15:00 | 3955.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-03-27 11:15:00 | 4024.80 | 2026-03-30 09:15:00 | 3955.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-04-01 10:15:00 | 4083.20 | 2026-04-02 09:15:00 | 3955.00 | STOP_HIT | 1.00 | -3.14% |
