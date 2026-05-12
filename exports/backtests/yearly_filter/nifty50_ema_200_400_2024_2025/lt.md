# LT (LT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3978.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 7 |
| ALERT3 | 65 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 55 |
| PARTIAL | 1 |
| TARGET_HIT | 5 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 10 / 43
- **Target hits / Stop hits / Partials:** 5 / 47 / 1
- **Avg / median % per leg:** -0.64% / -1.30%
- **Sum % (uncompounded):** -33.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 8 | 20.0% | 4 | 36 | 0 | -0.58% | -23.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 40 | 8 | 20.0% | 4 | 36 | 0 | -0.58% | -23.1% |
| SELL (all) | 13 | 2 | 15.4% | 1 | 11 | 1 | -0.84% | -10.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 1 | 11 | 1 | -0.84% | -10.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 10 | 18.9% | 5 | 47 | 1 | -0.64% | -34.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 3259.85 | 3545.42 | 3546.12 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 09:15:00 | 3616.25 | 3535.39 | 3535.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 12:15:00 | 3639.45 | 3538.25 | 3536.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3526.00 | 3576.06 | 3556.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 3526.00 | 3576.06 | 3556.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3526.00 | 3576.06 | 3556.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 3526.00 | 3576.06 | 3556.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 3419.75 | 3574.51 | 3555.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 3419.75 | 3574.51 | 3555.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 3490.95 | 3573.67 | 3555.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 11:00:00 | 3528.90 | 3553.29 | 3545.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 11:30:00 | 3537.15 | 3548.67 | 3543.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 12:15:00 | 3524.70 | 3548.67 | 3543.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 13:30:00 | 3523.85 | 3548.22 | 3543.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 3577.95 | 3548.13 | 3543.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 3585.05 | 3548.23 | 3543.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 15:15:00 | 3606.95 | 3585.16 | 3564.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:30:00 | 3587.10 | 3585.27 | 3565.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 11:00:00 | 3592.00 | 3585.33 | 3565.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 3558.35 | 3585.16 | 3565.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 3555.10 | 3585.16 | 3565.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 3554.80 | 3584.85 | 3565.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:15:00 | 3553.60 | 3584.85 | 3565.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-21 12:15:00 | 3530.00 | 3584.31 | 3565.64 | SL hit (close<static) qty=1.00 sl=3539.75 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 3483.10 | 3629.86 | 3630.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 3478.70 | 3622.38 | 3626.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 3601.00 | 3593.49 | 3609.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 3601.00 | 3593.49 | 3609.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 3601.00 | 3593.49 | 3609.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:00:00 | 3601.00 | 3593.49 | 3609.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 3603.00 | 3593.58 | 3609.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:45:00 | 3601.45 | 3593.58 | 3609.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 3603.60 | 3593.63 | 3609.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:00:00 | 3603.60 | 3593.63 | 3609.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 3586.95 | 3593.57 | 3609.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:30:00 | 3608.00 | 3593.57 | 3609.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 3627.15 | 3522.48 | 3566.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:00:00 | 3627.15 | 3522.48 | 3566.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 3647.95 | 3523.73 | 3566.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 3647.95 | 3523.73 | 3566.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 3555.65 | 3531.63 | 3568.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 3576.20 | 3531.63 | 3568.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 3576.55 | 3532.77 | 3568.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 3576.55 | 3532.77 | 3568.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 3575.90 | 3533.20 | 3568.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 3562.55 | 3533.20 | 3568.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 3557.50 | 3533.44 | 3568.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 15:00:00 | 3541.25 | 3564.15 | 3578.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 09:15:00 | 3530.05 | 3564.00 | 3578.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 10:00:00 | 3541.60 | 3563.77 | 3578.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 3518.15 | 3561.21 | 3576.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 3575.75 | 3560.18 | 3575.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 3570.00 | 3560.18 | 3575.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 3583.95 | 3560.42 | 3575.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 3581.30 | 3560.42 | 3575.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 3581.90 | 3560.63 | 3575.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:30:00 | 3590.10 | 3560.63 | 3575.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 3580.05 | 3553.97 | 3570.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 3580.05 | 3553.97 | 3570.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 3602.40 | 3554.45 | 3570.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-22 14:15:00 | 3602.40 | 3554.45 | 3570.87 | SL hit (close>static) qty=1.00 sl=3585.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 15:15:00 | 3697.00 | 3585.65 | 3585.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 3706.60 | 3586.85 | 3586.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 3722.05 | 3745.36 | 3684.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 3677.95 | 3743.00 | 3685.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 3677.95 | 3743.00 | 3685.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 3677.95 | 3743.00 | 3685.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 3691.50 | 3742.49 | 3685.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:15:00 | 3675.40 | 3742.49 | 3685.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 3670.05 | 3741.77 | 3685.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 3670.05 | 3741.77 | 3685.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 3646.35 | 3740.82 | 3685.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 3646.35 | 3740.82 | 3685.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 3653.95 | 3736.00 | 3684.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:45:00 | 3653.95 | 3736.00 | 3684.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 3664.10 | 3695.22 | 3672.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 15:00:00 | 3664.10 | 3695.22 | 3672.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 3670.10 | 3694.97 | 3672.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 3657.10 | 3694.97 | 3672.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 3678.40 | 3694.80 | 3672.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:30:00 | 3661.00 | 3694.80 | 3672.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 3679.10 | 3695.01 | 3673.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:30:00 | 3676.60 | 3695.01 | 3673.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 3664.60 | 3694.70 | 3673.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 3664.60 | 3694.70 | 3673.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 3663.25 | 3694.39 | 3673.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 3663.25 | 3694.39 | 3673.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 3641.75 | 3693.43 | 3673.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 3641.75 | 3693.43 | 3673.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 3620.00 | 3692.70 | 3672.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 3600.00 | 3692.70 | 3672.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 3519.15 | 3655.77 | 3656.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 3475.45 | 3653.98 | 3655.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 3561.95 | 3553.70 | 3594.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 3619.00 | 3555.01 | 3593.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 3619.00 | 3555.01 | 3593.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:00:00 | 3619.00 | 3555.01 | 3593.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 3611.00 | 3555.57 | 3593.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 3519.35 | 3555.13 | 3593.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 3343.38 | 3548.00 | 3588.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-28 09:15:00 | 3167.41 | 3351.27 | 3444.55 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 3593.10 | 3351.74 | 3350.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 3639.00 | 3373.98 | 3362.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 3553.50 | 3578.39 | 3503.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 3553.50 | 3578.39 | 3503.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 3553.50 | 3578.39 | 3503.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 3576.10 | 3578.39 | 3503.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 3570.80 | 3578.05 | 3503.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:00:00 | 3568.50 | 3577.96 | 3504.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 3484.80 | 3581.03 | 3553.68 | SL hit (close<static) qty=1.00 sl=3485.90 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 3422.00 | 3532.73 | 3532.98 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 3678.50 | 3533.26 | 3533.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 3715.00 | 3578.00 | 3558.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 3589.90 | 3601.96 | 3575.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:30:00 | 3583.90 | 3601.96 | 3575.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3573.20 | 3602.53 | 3578.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 3572.50 | 3602.53 | 3578.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 3575.40 | 3602.26 | 3578.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 3576.90 | 3602.26 | 3578.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 3575.00 | 3601.99 | 3578.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 3575.00 | 3601.99 | 3578.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 3575.60 | 3601.73 | 3578.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:30:00 | 3573.60 | 3601.73 | 3578.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 3607.70 | 3597.59 | 3578.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 12:00:00 | 3613.00 | 3597.74 | 3578.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 12:30:00 | 3611.80 | 3597.77 | 3578.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 13:30:00 | 3611.90 | 3597.91 | 3578.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 14:00:00 | 3611.60 | 3597.91 | 3578.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3566.00 | 3598.01 | 3579.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 3566.00 | 3598.01 | 3579.88 | SL hit (close<static) qty=1.00 sl=3570.20 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 3770.80 | 3950.99 | 3951.81 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 09:15:00 | 4044.00 | 3948.93 | 3948.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 4107.80 | 3964.27 | 3956.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4057.70 | 4162.32 | 4081.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 4057.70 | 4162.32 | 4081.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 4057.70 | 4162.32 | 4081.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:00:00 | 4057.70 | 4162.32 | 4081.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 4044.80 | 4161.15 | 4080.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:45:00 | 4052.90 | 4161.15 | 4080.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 4027.00 | 4124.27 | 4068.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 3960.60 | 4124.27 | 4068.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3485.60 | 4023.27 | 4024.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 3474.90 | 4017.82 | 4021.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 3732.90 | 3727.11 | 3839.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 3732.90 | 3727.11 | 3839.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3971.00 | 3728.58 | 3835.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 3930.30 | 3746.39 | 3840.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 3930.60 | 3746.39 | 3840.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:00:00 | 3930.00 | 3749.92 | 3841.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 3902.20 | 3769.06 | 3846.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 3928.50 | 3772.16 | 3847.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:45:00 | 3920.00 | 3772.16 | 3847.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4081.60 | 3783.87 | 3850.80 | SL hit (close>static) qty=1.00 sl=4023.40 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 4071.00 | 3903.71 | 3903.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 4082.30 | 3922.45 | 3912.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 3919.50 | 3956.54 | 3932.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 3919.50 | 3956.54 | 3932.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3919.50 | 3956.54 | 3932.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 3992.20 | 3955.49 | 3932.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:15:00 | 3977.80 | 3961.37 | 3936.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 3980.70 | 3961.62 | 3937.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-06 11:00:00 | 3528.90 | 2024-06-21 12:15:00 | 3530.00 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-06-07 11:30:00 | 3537.15 | 2024-06-21 12:15:00 | 3530.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-06-07 12:15:00 | 3524.70 | 2024-06-21 12:15:00 | 3530.00 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-06-07 13:30:00 | 3523.85 | 2024-06-21 12:15:00 | 3530.00 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2024-06-11 09:15:00 | 3585.05 | 2024-06-28 14:15:00 | 3545.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-06-19 15:15:00 | 3606.95 | 2024-06-28 14:15:00 | 3545.10 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-06-20 09:30:00 | 3587.10 | 2024-06-28 14:15:00 | 3545.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-06-20 11:00:00 | 3592.00 | 2024-07-23 11:15:00 | 3533.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-06-25 13:15:00 | 3582.75 | 2024-07-23 11:15:00 | 3533.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-06-27 09:30:00 | 3581.15 | 2024-07-23 11:15:00 | 3533.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-06-27 11:30:00 | 3581.05 | 2024-07-23 11:15:00 | 3533.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-07-02 11:15:00 | 3583.55 | 2024-07-23 11:15:00 | 3533.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-07-05 10:15:00 | 3630.05 | 2024-08-23 12:15:00 | 3595.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-05 11:15:00 | 3628.25 | 2024-09-06 09:15:00 | 3584.60 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-07-05 13:15:00 | 3630.60 | 2024-09-10 14:15:00 | 3593.65 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-07-05 15:00:00 | 3629.90 | 2024-10-03 09:15:00 | 3577.35 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-07-25 09:15:00 | 3563.25 | 2024-10-03 09:15:00 | 3577.35 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2024-08-06 09:15:00 | 3615.00 | 2024-10-07 10:15:00 | 3445.70 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2024-08-13 14:15:00 | 3562.30 | 2024-10-07 10:15:00 | 3445.70 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2024-08-14 09:15:00 | 3570.75 | 2024-10-07 10:15:00 | 3445.70 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2024-08-23 10:00:00 | 3617.50 | 2024-10-07 10:15:00 | 3445.70 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2024-08-26 09:45:00 | 3610.80 | 2024-10-09 14:15:00 | 3483.10 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2024-09-10 13:00:00 | 3615.25 | 2024-10-09 14:15:00 | 3483.10 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2024-09-12 15:00:00 | 3631.70 | 2024-10-09 14:15:00 | 3483.10 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2024-09-16 09:15:00 | 3634.70 | 2024-10-09 14:15:00 | 3483.10 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2024-11-13 15:00:00 | 3541.25 | 2024-11-22 14:15:00 | 3602.40 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-11-14 09:15:00 | 3530.05 | 2024-11-22 14:15:00 | 3602.40 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-11-14 10:00:00 | 3541.60 | 2024-11-22 14:15:00 | 3602.40 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-11-18 09:30:00 | 3518.15 | 2024-11-22 14:15:00 | 3602.40 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-02-01 11:45:00 | 3519.35 | 2025-02-03 09:15:00 | 3343.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 11:45:00 | 3519.35 | 2025-02-28 09:15:00 | 3167.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-13 09:15:00 | 3561.80 | 2025-05-14 10:15:00 | 3593.10 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-05-13 11:00:00 | 3575.90 | 2025-05-14 10:15:00 | 3593.10 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-05-13 12:15:00 | 3575.40 | 2025-05-14 10:15:00 | 3593.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-06-13 10:15:00 | 3576.10 | 2025-07-17 10:15:00 | 3484.80 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-06-13 12:15:00 | 3570.80 | 2025-07-17 10:15:00 | 3484.80 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-06-13 13:00:00 | 3568.50 | 2025-07-17 10:15:00 | 3484.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-08-29 12:00:00 | 3613.00 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-08-29 12:30:00 | 3611.80 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-29 13:30:00 | 3611.90 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-29 14:00:00 | 3611.60 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-09-03 09:45:00 | 3584.10 | 2025-09-05 14:15:00 | 3552.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-03 13:45:00 | 3582.40 | 2025-09-05 14:15:00 | 3552.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-04 12:30:00 | 3588.10 | 2025-09-05 14:15:00 | 3552.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-09-04 14:30:00 | 3589.20 | 2025-09-05 14:15:00 | 3552.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-09-12 11:00:00 | 3588.50 | 2025-10-23 11:15:00 | 3947.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-12 11:45:00 | 3593.00 | 2025-10-23 11:15:00 | 3952.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-15 09:15:00 | 3590.00 | 2025-10-23 11:15:00 | 3949.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-15 09:45:00 | 3601.40 | 2025-10-23 12:15:00 | 3961.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-09 09:45:00 | 3930.30 | 2026-04-15 09:15:00 | 4081.60 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2026-04-09 10:15:00 | 3930.60 | 2026-04-15 09:15:00 | 4081.60 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2026-04-09 12:00:00 | 3930.00 | 2026-04-15 09:15:00 | 4081.60 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2026-04-13 09:15:00 | 3902.20 | 2026-04-15 09:15:00 | 4081.60 | STOP_HIT | 1.00 | -4.60% |
