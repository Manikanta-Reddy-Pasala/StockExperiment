# TVS Motor Company Ltd. (TVSMOTOR)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3701.00
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 7
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 3.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.20% | 2.2% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.20% | 2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.08% | 0.9% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.08% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 10 | 45.5% | 3 | 12 | 7 | 0.14% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:05:00 | 3750.20 | 3755.89 | 0.00 | ORB-short ORB[3752.00,3772.40] vol=4.4x ATR=6.85 |
| Stop hit — per-position SL triggered | 2026-02-10 11:15:00 | 3757.05 | 3755.82 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:40:00 | 3885.50 | 3877.26 | 0.00 | ORB-long ORB[3856.60,3881.10] vol=1.6x ATR=9.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:00:00 | 3899.82 | 3882.69 | 0.00 | T1 1.5R @ 3899.82 |
| Stop hit — per-position SL triggered | 2026-02-12 10:25:00 | 3885.50 | 3885.94 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:35:00 | 3874.70 | 3866.98 | 0.00 | ORB-long ORB[3841.50,3867.90] vol=2.1x ATR=9.10 |
| Stop hit — per-position SL triggered | 2026-02-16 09:40:00 | 3865.60 | 3866.77 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 3880.60 | 3866.29 | 0.00 | ORB-long ORB[3832.70,3876.00] vol=2.4x ATR=11.82 |
| Stop hit — per-position SL triggered | 2026-02-23 09:50:00 | 3868.78 | 3866.85 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 3901.50 | 3864.74 | 0.00 | ORB-long ORB[3800.20,3831.00] vol=2.1x ATR=8.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:55:00 | 3914.67 | 3874.39 | 0.00 | T1 1.5R @ 3914.67 |
| Stop hit — per-position SL triggered | 2026-02-25 13:20:00 | 3901.50 | 3902.61 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:55:00 | 3305.60 | 3338.58 | 0.00 | ORB-short ORB[3315.00,3357.70] vol=2.5x ATR=14.38 |
| Stop hit — per-position SL triggered | 2026-03-16 11:00:00 | 3319.98 | 3337.26 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:35:00 | 3405.70 | 3386.16 | 0.00 | ORB-long ORB[3352.20,3390.00] vol=1.9x ATR=15.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:00:00 | 3429.46 | 3402.97 | 0.00 | T1 1.5R @ 3429.46 |
| Target hit | 2026-03-17 11:20:00 | 3417.00 | 3420.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:05:00 | 3563.10 | 3541.98 | 0.00 | ORB-long ORB[3500.00,3551.40] vol=2.4x ATR=9.63 |
| Stop hit — per-position SL triggered | 2026-03-18 11:45:00 | 3553.47 | 3549.25 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:45:00 | 3422.00 | 3438.97 | 0.00 | ORB-short ORB[3438.30,3474.00] vol=2.3x ATR=11.92 |
| Stop hit — per-position SL triggered | 2026-03-24 11:40:00 | 3433.92 | 3435.80 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 3443.90 | 3469.60 | 0.00 | ORB-short ORB[3476.10,3521.00] vol=2.4x ATR=11.89 |
| Stop hit — per-position SL triggered | 2026-03-27 11:20:00 | 3455.79 | 3467.22 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:50:00 | 3420.50 | 3457.63 | 0.00 | ORB-short ORB[3425.00,3474.40] vol=2.5x ATR=14.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:10:00 | 3398.93 | 3446.39 | 0.00 | T1 1.5R @ 3398.93 |
| Stop hit — per-position SL triggered | 2026-04-01 11:30:00 | 3420.50 | 3438.80 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 3777.20 | 3795.19 | 0.00 | ORB-short ORB[3794.00,3828.00] vol=1.7x ATR=8.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 12:00:00 | 3765.17 | 3791.89 | 0.00 | T1 1.5R @ 3765.17 |
| Stop hit — per-position SL triggered | 2026-04-16 14:45:00 | 3777.20 | 3773.97 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 3574.10 | 3598.58 | 0.00 | ORB-short ORB[3589.00,3636.00] vol=1.7x ATR=14.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:55:00 | 3552.54 | 3587.35 | 0.00 | T1 1.5R @ 3552.54 |
| Target hit | 2026-04-23 10:45:00 | 3535.40 | 3531.47 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — BUY (started 2026-05-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:10:00 | 3505.00 | 3495.52 | 0.00 | ORB-long ORB[3462.80,3503.30] vol=1.9x ATR=8.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 12:00:00 | 3517.20 | 3497.78 | 0.00 | T1 1.5R @ 3517.20 |
| Target hit | 2026-05-05 15:20:00 | 3536.10 | 3509.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 3521.90 | 3566.38 | 0.00 | ORB-short ORB[3552.20,3604.40] vol=3.3x ATR=14.20 |
| Stop hit — per-position SL triggered | 2026-05-06 11:05:00 | 3536.10 | 3562.87 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 11:05:00 | 3750.20 | 2026-02-10 11:15:00 | 3757.05 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-12 09:40:00 | 3885.50 | 2026-02-12 10:00:00 | 3899.82 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-12 09:40:00 | 3885.50 | 2026-02-12 10:25:00 | 3885.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 09:35:00 | 3874.70 | 2026-02-16 09:40:00 | 3865.60 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-23 09:45:00 | 3880.60 | 2026-02-23 09:50:00 | 3868.78 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-25 10:50:00 | 3901.50 | 2026-02-25 10:55:00 | 3914.67 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-25 10:50:00 | 3901.50 | 2026-02-25 13:20:00 | 3901.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:55:00 | 3305.60 | 2026-03-16 11:00:00 | 3319.98 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-17 09:35:00 | 3405.70 | 2026-03-17 10:00:00 | 3429.46 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-17 09:35:00 | 3405.70 | 2026-03-17 11:20:00 | 3417.00 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2026-03-18 11:05:00 | 3563.10 | 2026-03-18 11:45:00 | 3553.47 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-24 10:45:00 | 3422.00 | 2026-03-24 11:40:00 | 3433.92 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-27 11:00:00 | 3443.90 | 2026-03-27 11:20:00 | 3455.79 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-01 10:50:00 | 3420.50 | 2026-04-01 11:10:00 | 3398.93 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-04-01 10:50:00 | 3420.50 | 2026-04-01 11:30:00 | 3420.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 11:15:00 | 3777.20 | 2026-04-16 12:00:00 | 3765.17 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-04-16 11:15:00 | 3777.20 | 2026-04-16 14:45:00 | 3777.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 09:30:00 | 3574.10 | 2026-04-23 09:55:00 | 3552.54 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-04-23 09:30:00 | 3574.10 | 2026-04-23 10:45:00 | 3535.40 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2026-05-05 11:10:00 | 3505.00 | 2026-05-05 12:00:00 | 3517.20 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-05-05 11:10:00 | 3505.00 | 2026-05-05 15:20:00 | 3536.10 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2026-05-06 11:00:00 | 3521.90 | 2026-05-06 11:05:00 | 3536.10 | STOP_HIT | 1.00 | -0.40% |
