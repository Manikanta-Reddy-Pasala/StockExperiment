# AIA Engineering Ltd. (AIAENG)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3955.00
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
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 5
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 0.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 6 | 46.2% | 3 | 7 | 3 | 0.04% | 0.6% |
| BUY @ 2nd Alert (retest1) | 13 | 6 | 46.2% | 3 | 7 | 3 | 0.04% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.02% | 0.1% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | 0.02% | 0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 8 | 40.0% | 3 | 12 | 5 | 0.04% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 4037.10 | 4050.74 | 0.00 | ORB-short ORB[4044.70,4087.90] vol=1.5x ATR=13.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 12:40:00 | 4017.60 | 4042.15 | 0.00 | T1 1.5R @ 4017.60 |
| Stop hit — per-position SL triggered | 2026-02-11 15:00:00 | 4037.10 | 4030.27 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 3951.40 | 3930.64 | 0.00 | ORB-long ORB[3884.40,3939.10] vol=3.7x ATR=13.43 |
| Stop hit — per-position SL triggered | 2026-02-13 11:20:00 | 3937.97 | 3931.48 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 3915.00 | 3885.62 | 0.00 | ORB-long ORB[3855.60,3878.30] vol=3.0x ATR=13.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:50:00 | 3934.56 | 3894.49 | 0.00 | T1 1.5R @ 3934.56 |
| Target hit | 2026-02-17 12:45:00 | 3918.90 | 3920.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 3978.90 | 3974.88 | 0.00 | ORB-long ORB[3951.30,3969.00] vol=2.5x ATR=11.21 |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 3967.69 | 3978.75 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 3822.00 | 3841.06 | 0.00 | ORB-short ORB[3846.80,3869.90] vol=2.5x ATR=8.47 |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 3830.47 | 3839.89 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 3924.70 | 3919.29 | 0.00 | ORB-long ORB[3859.80,3912.10] vol=1.6x ATR=14.85 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 3909.85 | 3919.59 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 11:05:00 | 3654.40 | 3660.10 | 0.00 | ORB-short ORB[3666.50,3707.80] vol=3.3x ATR=16.79 |
| Stop hit — per-position SL triggered | 2026-03-04 14:40:00 | 3671.19 | 3658.92 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 3649.40 | 3620.16 | 0.00 | ORB-long ORB[3577.10,3610.60] vol=2.0x ATR=12.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:35:00 | 3668.01 | 3628.90 | 0.00 | T1 1.5R @ 3668.01 |
| Target hit | 2026-03-06 15:20:00 | 3674.80 | 3659.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:45:00 | 3704.00 | 3654.69 | 0.00 | ORB-long ORB[3610.00,3660.00] vol=1.6x ATR=16.47 |
| Stop hit — per-position SL triggered | 2026-03-16 09:55:00 | 3687.53 | 3663.47 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:45:00 | 3708.00 | 3683.70 | 0.00 | ORB-long ORB[3644.00,3696.00] vol=5.2x ATR=12.91 |
| Stop hit — per-position SL triggered | 2026-04-07 10:50:00 | 3695.09 | 3684.67 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 4094.50 | 4068.99 | 0.00 | ORB-long ORB[4017.00,4072.80] vol=4.6x ATR=14.47 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 4080.03 | 4074.92 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 4013.60 | 4023.61 | 0.00 | ORB-short ORB[4014.10,4054.10] vol=2.0x ATR=18.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 3985.26 | 4014.13 | 0.00 | T1 1.5R @ 3985.26 |
| Stop hit — per-position SL triggered | 2026-04-28 12:00:00 | 4013.60 | 4011.97 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 3923.10 | 3941.14 | 0.00 | ORB-short ORB[3938.10,3967.70] vol=4.2x ATR=14.13 |
| Stop hit — per-position SL triggered | 2026-05-05 10:25:00 | 3937.23 | 3939.60 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 3987.30 | 3959.25 | 0.00 | ORB-long ORB[3920.10,3979.00] vol=3.0x ATR=15.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 12:15:00 | 4010.07 | 3969.93 | 0.00 | T1 1.5R @ 4010.07 |
| Target hit | 2026-05-07 15:20:00 | 4013.00 | 3992.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 4037.50 | 4024.37 | 0.00 | ORB-long ORB[4005.50,4031.90] vol=3.3x ATR=12.51 |
| Stop hit — per-position SL triggered | 2026-05-08 09:35:00 | 4024.99 | 4022.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 10:55:00 | 4037.10 | 2026-02-11 12:40:00 | 4017.60 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-11 10:55:00 | 4037.10 | 2026-02-11 15:00:00 | 4037.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-13 11:15:00 | 3951.40 | 2026-02-13 11:20:00 | 3937.97 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-17 10:35:00 | 3915.00 | 2026-02-17 10:50:00 | 3934.56 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-17 10:35:00 | 3915.00 | 2026-02-17 12:45:00 | 3918.90 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-02-20 10:35:00 | 3978.90 | 2026-02-20 11:15:00 | 3967.69 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-25 11:00:00 | 3822.00 | 2026-02-25 11:15:00 | 3830.47 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-26 10:55:00 | 3924.70 | 2026-02-26 11:30:00 | 3909.85 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-03-04 11:05:00 | 3654.40 | 2026-03-04 14:40:00 | 3671.19 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-03-06 10:45:00 | 3649.40 | 2026-03-06 11:35:00 | 3668.01 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-06 10:45:00 | 3649.40 | 2026-03-06 15:20:00 | 3674.80 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-16 09:45:00 | 3704.00 | 2026-03-16 09:55:00 | 3687.53 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-07 10:45:00 | 3708.00 | 2026-04-07 10:50:00 | 3695.09 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-27 09:50:00 | 4094.50 | 2026-04-27 10:05:00 | 4080.03 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-28 09:45:00 | 4013.60 | 2026-04-28 11:25:00 | 3985.26 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-04-28 09:45:00 | 4013.60 | 2026-04-28 12:00:00 | 4013.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:10:00 | 3923.10 | 2026-05-05 10:25:00 | 3937.23 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-07 11:00:00 | 3987.30 | 2026-05-07 12:15:00 | 4010.07 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-05-07 11:00:00 | 3987.30 | 2026-05-07 15:20:00 | 4013.00 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2026-05-08 09:30:00 | 4037.50 | 2026-05-08 09:35:00 | 4024.99 | STOP_HIT | 1.00 | -0.31% |
