# Larsen & Toubro Ltd. (LT)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 3978.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 91 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 14 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 77
- **Target hits / Stop hits / Partials:** 14 / 77 / 36
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 9.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 29 | 43.9% | 9 | 37 | 20 | 0.08% | 5.3% |
| BUY @ 2nd Alert (retest1) | 66 | 29 | 43.9% | 9 | 37 | 20 | 0.08% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 61 | 21 | 34.4% | 5 | 40 | 16 | 0.07% | 4.5% |
| SELL @ 2nd Alert (retest1) | 61 | 21 | 34.4% | 5 | 40 | 16 | 0.07% | 4.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 127 | 50 | 39.4% | 14 | 77 | 36 | 0.08% | 9.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:15:00 | 3605.10 | 3602.29 | 0.00 | ORB-long ORB[3568.00,3595.00] vol=3.3x ATR=9.89 |
| Stop hit — per-position SL triggered | 2025-05-14 10:55:00 | 3595.21 | 3601.94 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:40:00 | 3556.60 | 3564.35 | 0.00 | ORB-short ORB[3564.00,3597.00] vol=2.4x ATR=7.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 09:45:00 | 3544.76 | 3561.92 | 0.00 | T1 1.5R @ 3544.76 |
| Stop hit — per-position SL triggered | 2025-05-15 09:50:00 | 3556.60 | 3561.54 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 10:50:00 | 3560.00 | 3565.12 | 0.00 | ORB-short ORB[3564.00,3592.00] vol=3.6x ATR=6.92 |
| Stop hit — per-position SL triggered | 2025-05-22 11:50:00 | 3566.92 | 3563.34 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:45:00 | 3615.00 | 3583.22 | 0.00 | ORB-long ORB[3551.20,3577.20] vol=3.4x ATR=9.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 10:55:00 | 3628.74 | 3596.17 | 0.00 | T1 1.5R @ 3628.74 |
| Stop hit — per-position SL triggered | 2025-05-23 11:00:00 | 3615.00 | 3597.24 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:35:00 | 3647.80 | 3638.45 | 0.00 | ORB-long ORB[3608.40,3645.60] vol=2.1x ATR=7.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:40:00 | 3658.67 | 3642.72 | 0.00 | T1 1.5R @ 3658.67 |
| Stop hit — per-position SL triggered | 2025-05-26 10:10:00 | 3647.80 | 3651.30 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:50:00 | 3645.60 | 3648.39 | 0.00 | ORB-short ORB[3647.00,3668.90] vol=1.8x ATR=6.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 12:05:00 | 3635.38 | 3645.62 | 0.00 | T1 1.5R @ 3635.38 |
| Target hit | 2025-05-29 14:20:00 | 3639.40 | 3639.37 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:15:00 | 3656.70 | 3643.68 | 0.00 | ORB-long ORB[3636.10,3650.00] vol=1.5x ATR=8.20 |
| Stop hit — per-position SL triggered | 2025-06-06 11:30:00 | 3648.50 | 3644.83 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:40:00 | 3706.10 | 3691.27 | 0.00 | ORB-long ORB[3662.20,3701.70] vol=2.9x ATR=8.65 |
| Stop hit — per-position SL triggered | 2025-06-09 09:45:00 | 3697.45 | 3692.28 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 10:55:00 | 3610.00 | 3622.65 | 0.00 | ORB-short ORB[3612.00,3629.90] vol=2.0x ATR=4.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 11:30:00 | 3602.73 | 3619.42 | 0.00 | T1 1.5R @ 3602.73 |
| Target hit | 2025-06-18 15:20:00 | 3600.00 | 3605.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:15:00 | 3654.20 | 3634.51 | 0.00 | ORB-long ORB[3614.00,3646.00] vol=1.7x ATR=9.08 |
| Stop hit — per-position SL triggered | 2025-06-20 10:55:00 | 3645.12 | 3642.85 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-23 11:05:00 | 3625.30 | 3634.27 | 0.00 | ORB-short ORB[3634.20,3660.00] vol=2.8x ATR=7.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 11:45:00 | 3614.80 | 3631.11 | 0.00 | T1 1.5R @ 3614.80 |
| Target hit | 2025-06-23 15:20:00 | 3578.00 | 3599.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 11:15:00 | 3665.00 | 3658.92 | 0.00 | ORB-long ORB[3615.70,3658.80] vol=1.8x ATR=9.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 12:05:00 | 3679.21 | 3660.82 | 0.00 | T1 1.5R @ 3679.21 |
| Stop hit — per-position SL triggered | 2025-06-24 12:35:00 | 3665.00 | 3661.84 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 10:20:00 | 3494.10 | 3495.14 | 0.00 | ORB-short ORB[3500.80,3519.00] vol=6.4x ATR=5.96 |
| Stop hit — per-position SL triggered | 2025-07-15 10:30:00 | 3500.06 | 3495.30 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:45:00 | 3460.00 | 3476.61 | 0.00 | ORB-short ORB[3478.70,3499.00] vol=3.1x ATR=5.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 11:40:00 | 3451.38 | 3464.66 | 0.00 | T1 1.5R @ 3451.38 |
| Stop hit — per-position SL triggered | 2025-07-18 12:40:00 | 3460.00 | 3462.88 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 11:15:00 | 3484.00 | 3473.85 | 0.00 | ORB-long ORB[3444.90,3477.70] vol=3.0x ATR=5.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 12:00:00 | 3492.84 | 3478.73 | 0.00 | T1 1.5R @ 3492.84 |
| Target hit | 2025-07-21 15:20:00 | 3502.00 | 3492.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-07-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:55:00 | 3479.00 | 3483.81 | 0.00 | ORB-short ORB[3489.10,3507.60] vol=7.1x ATR=5.69 |
| Stop hit — per-position SL triggered | 2025-07-22 10:00:00 | 3484.69 | 3483.78 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:50:00 | 3464.30 | 3474.05 | 0.00 | ORB-short ORB[3475.00,3490.00] vol=3.0x ATR=5.11 |
| Stop hit — per-position SL triggered | 2025-07-23 12:40:00 | 3469.41 | 3467.63 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 11:10:00 | 3460.10 | 3461.67 | 0.00 | ORB-short ORB[3462.20,3474.30] vol=5.8x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-07-25 11:30:00 | 3464.29 | 3459.54 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 09:40:00 | 3422.60 | 3428.12 | 0.00 | ORB-short ORB[3429.20,3446.10] vol=6.5x ATR=6.10 |
| Stop hit — per-position SL triggered | 2025-07-28 09:50:00 | 3428.70 | 3427.78 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 11:10:00 | 3442.50 | 3424.49 | 0.00 | ORB-long ORB[3407.00,3437.00] vol=1.9x ATR=8.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 11:35:00 | 3455.24 | 3428.72 | 0.00 | T1 1.5R @ 3455.24 |
| Stop hit — per-position SL triggered | 2025-07-29 12:05:00 | 3442.50 | 3431.06 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:30:00 | 3645.60 | 3626.56 | 0.00 | ORB-long ORB[3590.00,3639.90] vol=3.5x ATR=16.42 |
| Stop hit — per-position SL triggered | 2025-07-30 09:55:00 | 3629.18 | 3631.78 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:00:00 | 3650.20 | 3643.00 | 0.00 | ORB-long ORB[3621.00,3650.00] vol=2.3x ATR=6.93 |
| Stop hit — per-position SL triggered | 2025-07-31 11:10:00 | 3643.27 | 3643.12 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:10:00 | 3642.20 | 3651.97 | 0.00 | ORB-short ORB[3650.00,3671.00] vol=1.8x ATR=5.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:25:00 | 3634.17 | 3650.56 | 0.00 | T1 1.5R @ 3634.17 |
| Stop hit — per-position SL triggered | 2025-08-06 11:50:00 | 3642.20 | 3647.22 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:40:00 | 3602.10 | 3613.27 | 0.00 | ORB-short ORB[3610.30,3627.90] vol=2.9x ATR=5.62 |
| Stop hit — per-position SL triggered | 2025-08-07 10:45:00 | 3607.72 | 3611.52 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 10:45:00 | 3638.20 | 3627.78 | 0.00 | ORB-long ORB[3600.00,3628.00] vol=2.0x ATR=6.85 |
| Stop hit — per-position SL triggered | 2025-08-11 11:10:00 | 3631.35 | 3630.36 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 09:55:00 | 3718.10 | 3701.06 | 0.00 | ORB-long ORB[3673.10,3712.30] vol=2.3x ATR=7.64 |
| Stop hit — per-position SL triggered | 2025-08-12 10:25:00 | 3710.46 | 3705.88 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:45:00 | 3701.00 | 3697.25 | 0.00 | ORB-long ORB[3681.30,3698.80] vol=1.6x ATR=5.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 10:50:00 | 3709.26 | 3698.61 | 0.00 | T1 1.5R @ 3709.26 |
| Stop hit — per-position SL triggered | 2025-08-13 14:15:00 | 3701.00 | 3705.42 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:50:00 | 3662.30 | 3676.89 | 0.00 | ORB-short ORB[3678.00,3708.50] vol=1.9x ATR=5.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 11:10:00 | 3653.43 | 3673.07 | 0.00 | T1 1.5R @ 3653.43 |
| Stop hit — per-position SL triggered | 2025-08-14 12:30:00 | 3662.30 | 3665.17 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 09:35:00 | 3644.40 | 3661.04 | 0.00 | ORB-short ORB[3652.00,3701.50] vol=2.7x ATR=9.02 |
| Stop hit — per-position SL triggered | 2025-08-18 09:45:00 | 3653.42 | 3658.56 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:40:00 | 3611.40 | 3601.94 | 0.00 | ORB-long ORB[3588.00,3608.60] vol=1.8x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 09:45:00 | 3619.57 | 3604.36 | 0.00 | T1 1.5R @ 3619.57 |
| Target hit | 2025-08-21 12:15:00 | 3622.90 | 3624.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2025-09-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 11:10:00 | 3591.00 | 3600.33 | 0.00 | ORB-short ORB[3600.00,3628.60] vol=2.0x ATR=5.53 |
| Stop hit — per-position SL triggered | 2025-09-01 11:35:00 | 3596.53 | 3598.82 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 11:10:00 | 3530.00 | 3538.49 | 0.00 | ORB-short ORB[3538.00,3550.00] vol=5.5x ATR=5.22 |
| Stop hit — per-position SL triggered | 2025-09-09 11:50:00 | 3535.22 | 3536.26 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:45:00 | 3545.40 | 3552.52 | 0.00 | ORB-short ORB[3547.20,3564.70] vol=1.5x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:20:00 | 3539.17 | 3550.41 | 0.00 | T1 1.5R @ 3539.17 |
| Stop hit — per-position SL triggered | 2025-09-11 14:45:00 | 3545.40 | 3543.09 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 11:05:00 | 3589.90 | 3574.81 | 0.00 | ORB-long ORB[3548.10,3567.70] vol=2.6x ATR=5.38 |
| Stop hit — per-position SL triggered | 2025-09-12 11:45:00 | 3584.52 | 3577.44 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:45:00 | 3598.10 | 3587.28 | 0.00 | ORB-long ORB[3575.50,3592.00] vol=2.3x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 10:00:00 | 3605.96 | 3592.86 | 0.00 | T1 1.5R @ 3605.96 |
| Stop hit — per-position SL triggered | 2025-09-15 10:05:00 | 3598.10 | 3593.35 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:00:00 | 3628.80 | 3631.60 | 0.00 | ORB-short ORB[3632.00,3655.00] vol=1.6x ATR=5.06 |
| Stop hit — per-position SL triggered | 2025-09-23 12:00:00 | 3633.86 | 3630.67 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 10:30:00 | 3731.00 | 3703.31 | 0.00 | ORB-long ORB[3661.00,3704.00] vol=1.7x ATR=10.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:05:00 | 3747.03 | 3712.25 | 0.00 | T1 1.5R @ 3747.03 |
| Target hit | 2025-09-26 14:00:00 | 3737.30 | 3744.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — SELL (started 2025-10-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:55:00 | 3618.30 | 3645.59 | 0.00 | ORB-short ORB[3639.90,3659.00] vol=1.5x ATR=6.95 |
| Stop hit — per-position SL triggered | 2025-10-01 11:10:00 | 3625.25 | 3643.45 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:50:00 | 3764.50 | 3738.15 | 0.00 | ORB-long ORB[3703.00,3738.80] vol=1.8x ATR=8.04 |
| Stop hit — per-position SL triggered | 2025-10-06 10:55:00 | 3756.46 | 3739.46 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:55:00 | 3734.80 | 3749.15 | 0.00 | ORB-short ORB[3742.70,3766.00] vol=1.6x ATR=6.33 |
| Stop hit — per-position SL triggered | 2025-10-07 11:40:00 | 3741.13 | 3745.86 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:25:00 | 3763.10 | 3750.93 | 0.00 | ORB-long ORB[3730.10,3748.40] vol=1.9x ATR=6.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:35:00 | 3773.48 | 3756.77 | 0.00 | T1 1.5R @ 3773.48 |
| Target hit | 2025-10-09 12:30:00 | 3765.70 | 3772.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2025-10-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:40:00 | 3784.00 | 3780.63 | 0.00 | ORB-long ORB[3760.00,3783.90] vol=2.5x ATR=5.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 10:45:00 | 3792.99 | 3782.74 | 0.00 | T1 1.5R @ 3792.99 |
| Stop hit — per-position SL triggered | 2025-10-10 11:30:00 | 3784.00 | 3786.63 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:55:00 | 3768.00 | 3771.60 | 0.00 | ORB-short ORB[3771.30,3786.80] vol=1.8x ATR=7.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:15:00 | 3757.45 | 3769.94 | 0.00 | T1 1.5R @ 3757.45 |
| Stop hit — per-position SL triggered | 2025-10-14 11:35:00 | 3768.00 | 3767.74 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:35:00 | 3804.00 | 3785.26 | 0.00 | ORB-long ORB[3744.30,3781.90] vol=3.7x ATR=8.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:05:00 | 3816.21 | 3796.01 | 0.00 | T1 1.5R @ 3816.21 |
| Target hit | 2025-10-15 15:20:00 | 3827.40 | 3822.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:15:00 | 3865.10 | 3846.34 | 0.00 | ORB-long ORB[3830.00,3849.00] vol=1.9x ATR=7.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 10:30:00 | 3877.08 | 3851.66 | 0.00 | T1 1.5R @ 3877.08 |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 3865.10 | 3863.51 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 11:00:00 | 3884.00 | 3868.90 | 0.00 | ORB-long ORB[3845.40,3878.00] vol=1.7x ATR=7.25 |
| Stop hit — per-position SL triggered | 2025-10-20 11:20:00 | 3876.75 | 3870.90 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:55:00 | 3917.00 | 3905.85 | 0.00 | ORB-long ORB[3890.00,3914.80] vol=1.7x ATR=7.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:00:00 | 3928.37 | 3908.57 | 0.00 | T1 1.5R @ 3928.37 |
| Target hit | 2025-10-23 14:55:00 | 3935.20 | 3939.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2025-10-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:45:00 | 3935.40 | 3927.91 | 0.00 | ORB-long ORB[3907.00,3932.90] vol=1.9x ATR=9.45 |
| Stop hit — per-position SL triggered | 2025-10-27 10:00:00 | 3925.95 | 3928.43 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 3962.00 | 3946.78 | 0.00 | ORB-long ORB[3917.30,3954.30] vol=2.3x ATR=8.51 |
| Stop hit — per-position SL triggered | 2025-10-28 09:50:00 | 3953.49 | 3950.93 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-10-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:50:00 | 3967.70 | 3992.38 | 0.00 | ORB-short ORB[3982.00,4015.00] vol=1.8x ATR=11.84 |
| Stop hit — per-position SL triggered | 2025-10-29 11:00:00 | 3979.54 | 3991.71 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:15:00 | 3949.00 | 3961.97 | 0.00 | ORB-short ORB[3959.30,3990.00] vol=1.9x ATR=7.14 |
| Stop hit — per-position SL triggered | 2025-11-04 10:20:00 | 3956.14 | 3961.42 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 10:30:00 | 3850.40 | 3861.90 | 0.00 | ORB-short ORB[3860.00,3879.00] vol=1.8x ATR=9.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 11:05:00 | 3836.36 | 3854.42 | 0.00 | T1 1.5R @ 3836.36 |
| Stop hit — per-position SL triggered | 2025-11-07 11:10:00 | 3850.40 | 3854.01 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:40:00 | 3901.70 | 3919.15 | 0.00 | ORB-short ORB[3913.00,3935.00] vol=1.6x ATR=6.56 |
| Stop hit — per-position SL triggered | 2025-11-11 11:50:00 | 3908.26 | 3912.61 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:45:00 | 4070.60 | 4037.93 | 0.00 | ORB-long ORB[3998.00,4034.90] vol=4.2x ATR=9.32 |
| Stop hit — per-position SL triggered | 2025-11-26 10:50:00 | 4061.28 | 4040.32 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:25:00 | 4127.00 | 4101.34 | 0.00 | ORB-long ORB[4059.60,4108.80] vol=1.7x ATR=8.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:50:00 | 4139.97 | 4110.10 | 0.00 | T1 1.5R @ 4139.97 |
| Stop hit — per-position SL triggered | 2025-11-27 11:10:00 | 4127.00 | 4111.63 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-12-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:05:00 | 4060.10 | 4085.13 | 0.00 | ORB-short ORB[4076.50,4097.00] vol=2.8x ATR=7.11 |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 4067.21 | 4083.83 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:45:00 | 4050.10 | 4065.03 | 0.00 | ORB-short ORB[4066.50,4088.60] vol=1.8x ATR=5.52 |
| Stop hit — per-position SL triggered | 2025-12-02 11:00:00 | 4055.62 | 4062.96 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:05:00 | 3992.80 | 4005.06 | 0.00 | ORB-short ORB[4011.20,4046.90] vol=1.6x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:20:00 | 3983.38 | 4001.88 | 0.00 | T1 1.5R @ 3983.38 |
| Stop hit — per-position SL triggered | 2025-12-03 12:20:00 | 3992.80 | 3996.18 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:00:00 | 4002.00 | 3995.28 | 0.00 | ORB-long ORB[3961.70,3994.90] vol=3.5x ATR=6.22 |
| Stop hit — per-position SL triggered | 2025-12-04 13:35:00 | 3995.78 | 3997.93 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:10:00 | 4008.00 | 4015.60 | 0.00 | ORB-short ORB[4020.10,4047.90] vol=2.8x ATR=6.02 |
| Stop hit — per-position SL triggered | 2025-12-08 11:30:00 | 4014.02 | 4015.26 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:40:00 | 4023.30 | 4003.03 | 0.00 | ORB-long ORB[3970.00,4005.30] vol=1.5x ATR=7.74 |
| Stop hit — per-position SL triggered | 2025-12-10 10:50:00 | 4015.56 | 4005.16 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 09:30:00 | 4054.80 | 4068.58 | 0.00 | ORB-short ORB[4061.20,4103.80] vol=2.9x ATR=7.27 |
| Stop hit — per-position SL triggered | 2025-12-16 09:40:00 | 4062.07 | 4067.90 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:30:00 | 4079.00 | 4066.66 | 0.00 | ORB-long ORB[4040.10,4064.40] vol=1.5x ATR=6.25 |
| Stop hit — per-position SL triggered | 2025-12-19 11:45:00 | 4072.75 | 4071.18 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 11:00:00 | 4073.10 | 4081.69 | 0.00 | ORB-short ORB[4075.30,4098.50] vol=1.9x ATR=4.62 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 4077.72 | 4081.07 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:05:00 | 4041.50 | 4044.84 | 0.00 | ORB-short ORB[4045.50,4061.50] vol=2.7x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:35:00 | 4035.05 | 4043.65 | 0.00 | T1 1.5R @ 4035.05 |
| Stop hit — per-position SL triggered | 2025-12-26 13:30:00 | 4041.50 | 4038.48 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-12-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 10:35:00 | 4053.00 | 4054.39 | 0.00 | ORB-short ORB[4053.60,4072.60] vol=2.0x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-12-31 10:45:00 | 4057.79 | 4054.55 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:05:00 | 4126.00 | 4113.71 | 0.00 | ORB-long ORB[4088.30,4114.80] vol=1.5x ATR=7.99 |
| Stop hit — per-position SL triggered | 2026-01-01 10:15:00 | 4118.01 | 4114.43 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:15:00 | 4168.60 | 4158.58 | 0.00 | ORB-long ORB[4136.50,4160.60] vol=1.6x ATR=8.20 |
| Stop hit — per-position SL triggered | 2026-01-06 10:20:00 | 4160.40 | 4159.10 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 4149.00 | 4157.86 | 0.00 | ORB-short ORB[4151.00,4169.90] vol=1.8x ATR=6.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:30:00 | 4139.31 | 4153.59 | 0.00 | T1 1.5R @ 4139.31 |
| Target hit | 2026-01-08 15:20:00 | 4044.00 | 4076.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 11:15:00 | 4025.20 | 4051.21 | 0.00 | ORB-short ORB[4026.80,4068.60] vol=1.8x ATR=13.55 |
| Stop hit — per-position SL triggered | 2026-01-09 15:20:00 | 4029.80 | 4038.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2026-01-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:20:00 | 3905.70 | 3890.18 | 0.00 | ORB-long ORB[3860.40,3895.80] vol=1.7x ATR=8.71 |
| Stop hit — per-position SL triggered | 2026-01-16 10:50:00 | 3896.99 | 3893.75 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:50:00 | 3842.30 | 3854.42 | 0.00 | ORB-short ORB[3847.10,3874.30] vol=1.5x ATR=7.93 |
| Stop hit — per-position SL triggered | 2026-01-20 11:50:00 | 3850.23 | 3849.94 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 11:15:00 | 3767.60 | 3764.22 | 0.00 | ORB-long ORB[3739.00,3767.00] vol=1.6x ATR=8.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 11:45:00 | 3780.55 | 3764.91 | 0.00 | T1 1.5R @ 3780.55 |
| Stop hit — per-position SL triggered | 2026-01-27 12:15:00 | 3767.60 | 3766.99 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:05:00 | 3999.40 | 3967.36 | 0.00 | ORB-long ORB[3921.10,3957.40] vol=3.4x ATR=10.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:10:00 | 4014.81 | 3989.77 | 0.00 | T1 1.5R @ 4014.81 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 3999.40 | 3991.66 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:40:00 | 4039.00 | 4052.41 | 0.00 | ORB-short ORB[4040.70,4088.50] vol=1.9x ATR=7.81 |
| Stop hit — per-position SL triggered | 2026-02-05 10:00:00 | 4046.81 | 4050.12 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:45:00 | 4117.20 | 4102.96 | 0.00 | ORB-long ORB[4062.40,4099.80] vol=5.1x ATR=8.92 |
| Stop hit — per-position SL triggered | 2026-02-09 09:55:00 | 4108.28 | 4104.85 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 4227.60 | 4210.69 | 0.00 | ORB-long ORB[4190.10,4220.00] vol=1.6x ATR=8.38 |
| Stop hit — per-position SL triggered | 2026-02-17 09:55:00 | 4219.22 | 4213.35 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 4303.10 | 4283.18 | 0.00 | ORB-long ORB[4261.80,4299.90] vol=2.9x ATR=8.32 |
| Stop hit — per-position SL triggered | 2026-02-18 10:55:00 | 4294.78 | 4283.84 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-02-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:10:00 | 4384.30 | 4405.54 | 0.00 | ORB-short ORB[4401.00,4440.00] vol=1.7x ATR=9.63 |
| Stop hit — per-position SL triggered | 2026-02-24 10:20:00 | 4393.93 | 4404.26 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-02-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:55:00 | 4337.10 | 4310.15 | 0.00 | ORB-long ORB[4254.20,4306.00] vol=2.8x ATR=11.45 |
| Stop hit — per-position SL triggered | 2026-02-25 11:30:00 | 4325.65 | 4312.25 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 11:05:00 | 4298.80 | 4282.35 | 0.00 | ORB-long ORB[4263.60,4298.00] vol=3.2x ATR=7.44 |
| Stop hit — per-position SL triggered | 2026-02-27 11:25:00 | 4291.36 | 4284.21 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 11:00:00 | 3465.00 | 3491.48 | 0.00 | ORB-short ORB[3471.20,3516.90] vol=1.9x ATR=10.27 |
| Stop hit — per-position SL triggered | 2026-03-20 11:20:00 | 3475.27 | 3485.92 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-03-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:45:00 | 3602.00 | 3575.90 | 0.00 | ORB-long ORB[3546.00,3575.00] vol=1.6x ATR=14.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 09:50:00 | 3623.73 | 3580.90 | 0.00 | T1 1.5R @ 3623.73 |
| Target hit | 2026-03-25 15:20:00 | 3650.00 | 3620.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — SELL (started 2026-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:55:00 | 3615.70 | 3625.30 | 0.00 | ORB-short ORB[3621.00,3655.00] vol=1.8x ATR=11.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:45:00 | 3597.99 | 3623.71 | 0.00 | T1 1.5R @ 3597.99 |
| Stop hit — per-position SL triggered | 2026-04-01 12:00:00 | 3615.70 | 3623.28 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-04-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:50:00 | 3705.00 | 3696.67 | 0.00 | ORB-long ORB[3658.00,3699.50] vol=2.0x ATR=10.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 11:55:00 | 3721.48 | 3700.57 | 0.00 | T1 1.5R @ 3721.48 |
| Target hit | 2026-04-07 15:20:00 | 3722.00 | 3710.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — SELL (started 2026-04-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:50:00 | 4104.00 | 4119.56 | 0.00 | ORB-short ORB[4105.00,4142.90] vol=2.7x ATR=8.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:15:00 | 4091.65 | 4112.06 | 0.00 | T1 1.5R @ 4091.65 |
| Stop hit — per-position SL triggered | 2026-04-17 14:45:00 | 4104.00 | 4104.87 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-04-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 10:55:00 | 4077.20 | 4099.07 | 0.00 | ORB-short ORB[4078.00,4130.00] vol=2.1x ATR=9.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 12:15:00 | 4062.83 | 4088.59 | 0.00 | T1 1.5R @ 4062.83 |
| Target hit | 2026-04-20 15:20:00 | 4041.20 | 4066.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — BUY (started 2026-04-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 11:05:00 | 4034.80 | 4024.88 | 0.00 | ORB-long ORB[3978.50,4019.40] vol=1.9x ATR=7.88 |
| Stop hit — per-position SL triggered | 2026-04-23 11:20:00 | 4026.92 | 4025.20 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 4003.90 | 4026.23 | 0.00 | ORB-short ORB[4037.20,4088.10] vol=2.6x ATR=8.69 |
| Stop hit — per-position SL triggered | 2026-04-24 12:15:00 | 4012.59 | 4016.59 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 4088.00 | 4076.24 | 0.00 | ORB-long ORB[4051.30,4078.30] vol=2.3x ATR=8.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:15:00 | 4100.72 | 4082.30 | 0.00 | T1 1.5R @ 4100.72 |
| Target hit | 2026-04-29 15:05:00 | 4100.20 | 4100.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 91 — SELL (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 3975.70 | 3989.55 | 0.00 | ORB-short ORB[3987.90,4014.00] vol=1.8x ATR=6.52 |
| Stop hit — per-position SL triggered | 2026-05-08 11:05:00 | 3982.22 | 3988.76 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 10:15:00 | 3605.10 | 2025-05-14 10:55:00 | 3595.21 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-15 09:40:00 | 3556.60 | 2025-05-15 09:45:00 | 3544.76 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-05-15 09:40:00 | 3556.60 | 2025-05-15 09:50:00 | 3556.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-22 10:50:00 | 3560.00 | 2025-05-22 11:50:00 | 3566.92 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-05-23 10:45:00 | 3615.00 | 2025-05-23 10:55:00 | 3628.74 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-05-23 10:45:00 | 3615.00 | 2025-05-23 11:00:00 | 3615.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-26 09:35:00 | 3647.80 | 2025-05-26 09:40:00 | 3658.67 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-05-26 09:35:00 | 3647.80 | 2025-05-26 10:10:00 | 3647.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-29 10:50:00 | 3645.60 | 2025-05-29 12:05:00 | 3635.38 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-05-29 10:50:00 | 3645.60 | 2025-05-29 14:20:00 | 3639.40 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-06-06 11:15:00 | 3656.70 | 2025-06-06 11:30:00 | 3648.50 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-09 09:40:00 | 3706.10 | 2025-06-09 09:45:00 | 3697.45 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-18 10:55:00 | 3610.00 | 2025-06-18 11:30:00 | 3602.73 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-06-18 10:55:00 | 3610.00 | 2025-06-18 15:20:00 | 3600.00 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2025-06-20 10:15:00 | 3654.20 | 2025-06-20 10:55:00 | 3645.12 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-06-23 11:05:00 | 3625.30 | 2025-06-23 11:45:00 | 3614.80 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-06-23 11:05:00 | 3625.30 | 2025-06-23 15:20:00 | 3578.00 | TARGET_HIT | 0.50 | 1.30% |
| BUY | retest1 | 2025-06-24 11:15:00 | 3665.00 | 2025-06-24 12:05:00 | 3679.21 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-24 11:15:00 | 3665.00 | 2025-06-24 12:35:00 | 3665.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-15 10:20:00 | 3494.10 | 2025-07-15 10:30:00 | 3500.06 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-07-18 10:45:00 | 3460.00 | 2025-07-18 11:40:00 | 3451.38 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-18 10:45:00 | 3460.00 | 2025-07-18 12:40:00 | 3460.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 11:15:00 | 3484.00 | 2025-07-21 12:00:00 | 3492.84 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-07-21 11:15:00 | 3484.00 | 2025-07-21 15:20:00 | 3502.00 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2025-07-22 09:55:00 | 3479.00 | 2025-07-22 10:00:00 | 3484.69 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-23 10:50:00 | 3464.30 | 2025-07-23 12:40:00 | 3469.41 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-07-25 11:10:00 | 3460.10 | 2025-07-25 11:30:00 | 3464.29 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-07-28 09:40:00 | 3422.60 | 2025-07-28 09:50:00 | 3428.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-29 11:10:00 | 3442.50 | 2025-07-29 11:35:00 | 3455.24 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-29 11:10:00 | 3442.50 | 2025-07-29 12:05:00 | 3442.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-30 09:30:00 | 3645.60 | 2025-07-30 09:55:00 | 3629.18 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-07-31 11:00:00 | 3650.20 | 2025-07-31 11:10:00 | 3643.27 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-06 11:10:00 | 3642.20 | 2025-08-06 11:25:00 | 3634.17 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-08-06 11:10:00 | 3642.20 | 2025-08-06 11:50:00 | 3642.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 10:40:00 | 3602.10 | 2025-08-07 10:45:00 | 3607.72 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-08-11 10:45:00 | 3638.20 | 2025-08-11 11:10:00 | 3631.35 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-08-12 09:55:00 | 3718.10 | 2025-08-12 10:25:00 | 3710.46 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-08-13 10:45:00 | 3701.00 | 2025-08-13 10:50:00 | 3709.26 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2025-08-13 10:45:00 | 3701.00 | 2025-08-13 14:15:00 | 3701.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-14 10:50:00 | 3662.30 | 2025-08-14 11:10:00 | 3653.43 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-08-14 10:50:00 | 3662.30 | 2025-08-14 12:30:00 | 3662.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-18 09:35:00 | 3644.40 | 2025-08-18 09:45:00 | 3653.42 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-21 09:40:00 | 3611.40 | 2025-08-21 09:45:00 | 3619.57 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-08-21 09:40:00 | 3611.40 | 2025-08-21 12:15:00 | 3622.90 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2025-09-01 11:10:00 | 3591.00 | 2025-09-01 11:35:00 | 3596.53 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-09-09 11:10:00 | 3530.00 | 2025-09-09 11:50:00 | 3535.22 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-09-11 10:45:00 | 3545.40 | 2025-09-11 11:20:00 | 3539.17 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2025-09-11 10:45:00 | 3545.40 | 2025-09-11 14:45:00 | 3545.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-12 11:05:00 | 3589.90 | 2025-09-12 11:45:00 | 3584.52 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-09-15 09:45:00 | 3598.10 | 2025-09-15 10:00:00 | 3605.96 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2025-09-15 09:45:00 | 3598.10 | 2025-09-15 10:05:00 | 3598.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 11:00:00 | 3628.80 | 2025-09-23 12:00:00 | 3633.86 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-09-26 10:30:00 | 3731.00 | 2025-09-26 11:05:00 | 3747.03 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-26 10:30:00 | 3731.00 | 2025-09-26 14:00:00 | 3737.30 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2025-10-01 10:55:00 | 3618.30 | 2025-10-01 11:10:00 | 3625.25 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-06 10:50:00 | 3764.50 | 2025-10-06 10:55:00 | 3756.46 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-07 10:55:00 | 3734.80 | 2025-10-07 11:40:00 | 3741.13 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-10-09 10:25:00 | 3763.10 | 2025-10-09 10:35:00 | 3773.48 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-10-09 10:25:00 | 3763.10 | 2025-10-09 12:30:00 | 3765.70 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2025-10-10 10:40:00 | 3784.00 | 2025-10-10 10:45:00 | 3792.99 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-10-10 10:40:00 | 3784.00 | 2025-10-10 11:30:00 | 3784.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 10:55:00 | 3768.00 | 2025-10-14 11:15:00 | 3757.45 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-10-14 10:55:00 | 3768.00 | 2025-10-14 11:35:00 | 3768.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 10:35:00 | 3804.00 | 2025-10-15 11:05:00 | 3816.21 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-10-15 10:35:00 | 3804.00 | 2025-10-15 15:20:00 | 3827.40 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2025-10-16 10:15:00 | 3865.10 | 2025-10-16 10:30:00 | 3877.08 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-10-16 10:15:00 | 3865.10 | 2025-10-16 11:15:00 | 3865.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 11:00:00 | 3884.00 | 2025-10-20 11:20:00 | 3876.75 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-23 09:55:00 | 3917.00 | 2025-10-23 10:00:00 | 3928.37 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-10-23 09:55:00 | 3917.00 | 2025-10-23 14:55:00 | 3935.20 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2025-10-27 09:45:00 | 3935.40 | 2025-10-27 10:00:00 | 3925.95 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-28 09:30:00 | 3962.00 | 2025-10-28 09:50:00 | 3953.49 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-29 10:50:00 | 3967.70 | 2025-10-29 11:00:00 | 3979.54 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-04 10:15:00 | 3949.00 | 2025-11-04 10:20:00 | 3956.14 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-11-07 10:30:00 | 3850.40 | 2025-11-07 11:05:00 | 3836.36 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-11-07 10:30:00 | 3850.40 | 2025-11-07 11:10:00 | 3850.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 10:40:00 | 3901.70 | 2025-11-11 11:50:00 | 3908.26 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-26 10:45:00 | 4070.60 | 2025-11-26 10:50:00 | 4061.28 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-27 10:25:00 | 4127.00 | 2025-11-27 10:50:00 | 4139.97 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-11-27 10:25:00 | 4127.00 | 2025-11-27 11:10:00 | 4127.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-01 11:05:00 | 4060.10 | 2025-12-01 11:15:00 | 4067.21 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-02 10:45:00 | 4050.10 | 2025-12-02 11:00:00 | 4055.62 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-12-03 11:05:00 | 3992.80 | 2025-12-03 11:20:00 | 3983.38 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-03 11:05:00 | 3992.80 | 2025-12-03 12:20:00 | 3992.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-04 11:00:00 | 4002.00 | 2025-12-04 13:35:00 | 3995.78 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-12-08 11:10:00 | 4008.00 | 2025-12-08 11:30:00 | 4014.02 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-12-10 10:40:00 | 4023.30 | 2025-12-10 10:50:00 | 4015.56 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-16 09:30:00 | 4054.80 | 2025-12-16 09:40:00 | 4062.07 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-12-19 10:30:00 | 4079.00 | 2025-12-19 11:45:00 | 4072.75 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-12-22 11:00:00 | 4073.10 | 2025-12-22 11:15:00 | 4077.72 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2025-12-26 11:05:00 | 4041.50 | 2025-12-26 11:35:00 | 4035.05 | PARTIAL | 0.50 | 0.16% |
| SELL | retest1 | 2025-12-26 11:05:00 | 4041.50 | 2025-12-26 13:30:00 | 4041.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-31 10:35:00 | 4053.00 | 2025-12-31 10:45:00 | 4057.79 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2026-01-01 10:05:00 | 4126.00 | 2026-01-01 10:15:00 | 4118.01 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-01-06 10:15:00 | 4168.60 | 2026-01-06 10:20:00 | 4160.40 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-08 11:15:00 | 4149.00 | 2026-01-08 11:30:00 | 4139.31 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2026-01-08 11:15:00 | 4149.00 | 2026-01-08 15:20:00 | 4044.00 | TARGET_HIT | 0.50 | 2.53% |
| SELL | retest1 | 2026-01-09 11:15:00 | 4025.20 | 2026-01-09 15:20:00 | 4029.80 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2026-01-16 10:20:00 | 3905.70 | 2026-01-16 10:50:00 | 3896.99 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-20 10:50:00 | 3842.30 | 2026-01-20 11:50:00 | 3850.23 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-27 11:15:00 | 3767.60 | 2026-01-27 11:45:00 | 3780.55 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-01-27 11:15:00 | 3767.60 | 2026-01-27 12:15:00 | 3767.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:05:00 | 3999.40 | 2026-02-01 11:10:00 | 4014.81 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-01 11:05:00 | 3999.40 | 2026-02-01 11:15:00 | 3999.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-05 09:40:00 | 4039.00 | 2026-02-05 10:00:00 | 4046.81 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-09 09:45:00 | 4117.20 | 2026-02-09 09:55:00 | 4108.28 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-17 09:40:00 | 4227.60 | 2026-02-17 09:55:00 | 4219.22 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-18 10:50:00 | 4303.10 | 2026-02-18 10:55:00 | 4294.78 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-24 10:10:00 | 4384.30 | 2026-02-24 10:20:00 | 4393.93 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-25 10:55:00 | 4337.10 | 2026-02-25 11:30:00 | 4325.65 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-27 11:05:00 | 4298.80 | 2026-02-27 11:25:00 | 4291.36 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-03-20 11:00:00 | 3465.00 | 2026-03-20 11:20:00 | 3475.27 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-25 09:45:00 | 3602.00 | 2026-03-25 09:50:00 | 3623.73 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-25 09:45:00 | 3602.00 | 2026-03-25 15:20:00 | 3650.00 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2026-04-01 10:55:00 | 3615.70 | 2026-04-01 11:45:00 | 3597.99 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-01 10:55:00 | 3615.70 | 2026-04-01 12:00:00 | 3615.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 10:50:00 | 3705.00 | 2026-04-07 11:55:00 | 3721.48 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-07 10:50:00 | 3705.00 | 2026-04-07 15:20:00 | 3722.00 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-17 10:50:00 | 4104.00 | 2026-04-17 12:15:00 | 4091.65 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-04-17 10:50:00 | 4104.00 | 2026-04-17 14:45:00 | 4104.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-20 10:55:00 | 4077.20 | 2026-04-20 12:15:00 | 4062.83 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-20 10:55:00 | 4077.20 | 2026-04-20 15:20:00 | 4041.20 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2026-04-23 11:05:00 | 4034.80 | 2026-04-23 11:20:00 | 4026.92 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-04-24 11:10:00 | 4003.90 | 2026-04-24 12:15:00 | 4012.59 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-29 10:20:00 | 4088.00 | 2026-04-29 11:15:00 | 4100.72 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-04-29 10:20:00 | 4088.00 | 2026-04-29 15:05:00 | 4100.20 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-05-08 10:50:00 | 3975.70 | 2026-05-08 11:05:00 | 3982.22 | STOP_HIT | 1.00 | -0.16% |
