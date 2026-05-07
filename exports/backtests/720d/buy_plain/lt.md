# LT (LT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 4023.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 6 |
| PENDING | 13 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 1 |
| ENTRY2 | 9 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 6
- **Target hits / Stop hits / Partials:** 0 / 9 / 2
- **Avg / median % per leg:** 3.96% / -1.02%
- **Sum % (uncompounded):** 43.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 0 | 9 | 2 | 3.96% | 43.6% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.69% | -1.7% |
| BUY @ 3rd Alert (retest2) | 10 | 5 | 50.0% | 0 | 8 | 2 | 4.53% | 45.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.69% | -1.7% |
| retest2 (combined) | 10 | 5 | 50.0% | 0 | 8 | 2 | 4.53% | 45.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 14:15:00 | 3698.85 | 3584.53 | 3584.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 3706.60 | 3586.85 | 3585.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 3722.05 | 3745.36 | 3683.98 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 3677.95 | 3743.00 | 3684.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 3677.95 | 3743.00 | 3684.89 | EMA400 retest candle locked |

### Cycle 2 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 3593.10 | 3351.74 | 3350.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 3639.00 | 3373.98 | 3362.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 3553.50 | 3578.39 | 3503.39 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 3553.50 | 3578.39 | 3503.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 3553.50 | 3578.39 | 3503.39 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-13 12:15:00 | 3568.50 | 3577.96 | 3504.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:15:00 | 3585.20 | 3578.03 | 3504.69 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 3484.80 | 3581.03 | 3553.68 | SL hit (close<static) qty=1.00 sl=3485.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 3637.80 | 3528.21 | 3530.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 3642.50 | 3529.34 | 3531.18 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 3678.50 | 3533.26 | 3533.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 3678.50 | 3533.26 | 3533.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 3715.00 | 3578.00 | 3558.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 3589.90 | 3601.96 | 3575.08 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-08-21 09:15:00 | 3633.50 | 3602.12 | 3576.08 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:15:00 | 3634.60 | 3602.45 | 3576.38 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-21 13:15:00 | 3623.60 | 3603.05 | 3577.07 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-21 14:15:00 | 3611.00 | 3603.12 | 3577.24 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-22 09:15:00 | 3628.80 | 3603.54 | 3577.70 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-22 10:15:00 | 3609.00 | 3603.59 | 3577.86 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3573.20 | 3602.53 | 3578.93 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 3573.20 | 3602.53 | 3578.93 | SL hit (close<ema400) qty=1.00 sl=3578.93 alert=retest1 |
| Cross detected — sustain check pending | 2025-08-29 10:15:00 | 3607.70 | 3597.59 | 3578.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 11:15:00 | 3613.00 | 3597.74 | 3578.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-29 15:15:00 | 3605.00 | 3598.00 | 3578.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-01 09:15:00 | 3589.00 | 3597.91 | 3578.81 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2025-09-01 12:15:00 | 3603.70 | 3597.90 | 3579.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 13:15:00 | 3607.10 | 3597.99 | 3579.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-02 09:15:00 | 3608.90 | 3598.09 | 3579.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 10:15:00 | 3606.90 | 3598.18 | 3579.69 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 3566.00 | 3598.01 | 3579.88 | SL hit (close<static) qty=1.00 sl=3567.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 3566.00 | 3598.01 | 3579.88 | SL hit (close<static) qty=1.00 sl=3567.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 3566.00 | 3598.01 | 3579.88 | SL hit (close<static) qty=1.00 sl=3567.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-03 14:15:00 | 3600.80 | 3596.75 | 3579.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 15:15:00 | 3601.70 | 3596.80 | 3580.06 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 3589.30 | 3596.72 | 3580.35 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 3565.00 | 3595.77 | 3580.35 | SL hit (close<static) qty=1.00 sl=3567.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-15 09:15:00 | 3602.50 | 3582.04 | 3575.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 3600.00 | 3582.22 | 3575.60 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-16 09:15:00 | 3618.30 | 3583.05 | 3576.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 3624.80 | 3583.47 | 3576.47 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-11-27 10:15:00 | 4140.00 | 3945.03 | 3851.95 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 3979.30 | 3980.27 | 3894.92 | SL hit (close<ema200) qty=0.50 sl=3980.27 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-01-02 12:15:00 | 4168.52 | 4042.08 | 3972.17 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 14:15:00 | 4029.60 | 4068.58 | 3996.27 | SL hit (close<ema200) qty=0.50 sl=4068.58 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 09:15:00 | 4044.00 | 3948.93 | 3948.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 4107.80 | 3964.27 | 3956.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4057.70 | 4162.32 | 4081.08 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 4057.70 | 4162.32 | 4081.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 4057.70 | 4162.32 | 4081.08 | EMA400 retest candle locked |

### Cycle 5 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 4071.00 | 3903.71 | 3903.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 4082.30 | 3922.45 | 3912.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 3919.50 | 3956.54 | 3932.45 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 3919.50 | 3956.54 | 3932.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3919.50 | 3956.54 | 3932.45 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-05-06 14:15:00 | 4011.00 | 3956.04 | 3932.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 15:15:00 | 4014.90 | 3956.62 | 3933.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 13:15:00 | 3585.20 | 2025-07-17 10:15:00 | 3484.80 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-07-30 10:15:00 | 3642.50 | 2025-07-30 13:15:00 | 3678.50 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest1 | 2025-08-21 10:15:00 | 3634.60 | 2025-08-26 09:15:00 | 3573.20 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-08-29 11:15:00 | 3613.00 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-01 13:15:00 | 3607.10 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-09-02 10:15:00 | 3606.90 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-03 15:15:00 | 3601.70 | 2025-09-05 11:15:00 | 3565.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-09-15 10:15:00 | 3600.00 | 2025-11-27 10:15:00 | 4140.00 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-09-15 10:15:00 | 3600.00 | 2025-12-09 09:15:00 | 3979.30 | STOP_HIT | 0.50 | 10.54% |
| BUY | retest2 | 2025-09-16 10:15:00 | 3624.80 | 2026-01-02 12:15:00 | 4168.52 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-09-16 10:15:00 | 3624.80 | 2026-01-08 14:15:00 | 4029.60 | STOP_HIT | 0.50 | 11.17% |
