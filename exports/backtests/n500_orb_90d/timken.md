# Timken India Ltd. (TIMKEN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3600.00
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
| TARGET_HIT | 4 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 4 / 11 / 7
- **Avg / median % per leg:** 0.35% / 0.35%
- **Sum % (uncompounded):** 7.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 3 | 5 | 4 | 0.57% | 6.9% |
| BUY @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 3 | 5 | 4 | 0.57% | 6.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.09% | 0.9% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.09% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 11 | 50.0% | 4 | 11 | 7 | 0.35% | 7.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:10:00 | 3219.80 | 3242.00 | 0.00 | ORB-short ORB[3224.30,3257.10] vol=2.1x ATR=9.94 |
| Stop hit — per-position SL triggered | 2026-02-11 11:30:00 | 3229.74 | 3240.96 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 3087.20 | 3067.75 | 0.00 | ORB-long ORB[3028.10,3069.60] vol=3.2x ATR=11.02 |
| Stop hit — per-position SL triggered | 2026-02-17 10:50:00 | 3076.18 | 3080.51 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 3160.20 | 3148.43 | 0.00 | ORB-long ORB[3130.00,3150.00] vol=1.8x ATR=9.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:50:00 | 3174.36 | 3153.95 | 0.00 | T1 1.5R @ 3174.36 |
| Stop hit — per-position SL triggered | 2026-02-20 12:55:00 | 3160.20 | 3166.81 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:00:00 | 3311.40 | 3282.49 | 0.00 | ORB-long ORB[3245.30,3282.50] vol=3.3x ATR=12.87 |
| Stop hit — per-position SL triggered | 2026-02-25 10:30:00 | 3298.53 | 3289.38 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:55:00 | 3373.20 | 3344.80 | 0.00 | ORB-long ORB[3320.00,3350.00] vol=3.0x ATR=16.24 |
| Target hit | 2026-03-06 15:20:00 | 3385.00 | 3380.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:40:00 | 3388.00 | 3365.58 | 0.00 | ORB-long ORB[3338.50,3384.00] vol=1.7x ATR=14.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:45:00 | 3409.40 | 3373.30 | 0.00 | T1 1.5R @ 3409.40 |
| Stop hit — per-position SL triggered | 2026-03-18 10:50:00 | 3388.00 | 3374.77 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:15:00 | 3287.80 | 3297.03 | 0.00 | ORB-short ORB[3304.00,3339.90] vol=2.1x ATR=8.81 |
| Stop hit — per-position SL triggered | 2026-03-19 12:05:00 | 3296.61 | 3296.55 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 3218.30 | 3193.00 | 0.00 | ORB-long ORB[3154.30,3200.00] vol=1.9x ATR=21.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:10:00 | 3249.89 | 3207.92 | 0.00 | T1 1.5R @ 3249.89 |
| Target hit | 2026-03-25 15:10:00 | 3296.70 | 3305.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2026-04-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:50:00 | 3510.20 | 3554.38 | 0.00 | ORB-short ORB[3570.00,3618.00] vol=1.8x ATR=13.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 12:50:00 | 3489.49 | 3530.12 | 0.00 | T1 1.5R @ 3489.49 |
| Target hit | 2026-04-16 15:20:00 | 3492.50 | 3517.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-04-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:25:00 | 3472.50 | 3501.71 | 0.00 | ORB-short ORB[3489.70,3522.60] vol=3.6x ATR=10.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:30:00 | 3456.24 | 3499.96 | 0.00 | T1 1.5R @ 3456.24 |
| Stop hit — per-position SL triggered | 2026-04-17 11:20:00 | 3472.50 | 3490.11 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:50:00 | 3543.20 | 3522.99 | 0.00 | ORB-long ORB[3483.00,3530.00] vol=4.7x ATR=10.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:05:00 | 3559.68 | 3526.54 | 0.00 | T1 1.5R @ 3559.68 |
| Target hit | 2026-04-22 15:20:00 | 3639.70 | 3615.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-04-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 11:05:00 | 3530.70 | 3553.04 | 0.00 | ORB-short ORB[3540.50,3577.00] vol=4.6x ATR=11.67 |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 3542.37 | 3551.39 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:35:00 | 3549.10 | 3572.34 | 0.00 | ORB-short ORB[3557.50,3600.00] vol=2.0x ATR=11.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:40:00 | 3531.41 | 3564.42 | 0.00 | T1 1.5R @ 3531.41 |
| Stop hit — per-position SL triggered | 2026-04-28 11:45:00 | 3549.10 | 3562.55 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:10:00 | 3516.90 | 3542.31 | 0.00 | ORB-short ORB[3532.90,3571.20] vol=1.7x ATR=8.42 |
| Stop hit — per-position SL triggered | 2026-04-29 11:20:00 | 3525.32 | 3541.47 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 3454.50 | 3426.26 | 0.00 | ORB-long ORB[3384.90,3429.80] vol=1.8x ATR=13.70 |
| Stop hit — per-position SL triggered | 2026-05-04 11:30:00 | 3440.80 | 3430.77 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 11:10:00 | 3219.80 | 2026-02-11 11:30:00 | 3229.74 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-17 09:50:00 | 3087.20 | 2026-02-17 10:50:00 | 3076.18 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-20 10:35:00 | 3160.20 | 2026-02-20 10:50:00 | 3174.36 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-20 10:35:00 | 3160.20 | 2026-02-20 12:55:00 | 3160.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:00:00 | 3311.40 | 2026-02-25 10:30:00 | 3298.53 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-06 09:55:00 | 3373.20 | 2026-03-06 15:20:00 | 3385.00 | TARGET_HIT | 1.00 | 0.35% |
| BUY | retest1 | 2026-03-18 10:40:00 | 3388.00 | 2026-03-18 10:45:00 | 3409.40 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-03-18 10:40:00 | 3388.00 | 2026-03-18 10:50:00 | 3388.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 11:15:00 | 3287.80 | 2026-03-19 12:05:00 | 3296.61 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-25 09:30:00 | 3218.30 | 2026-03-25 10:10:00 | 3249.89 | PARTIAL | 0.50 | 0.98% |
| BUY | retest1 | 2026-03-25 09:30:00 | 3218.30 | 2026-03-25 15:10:00 | 3296.70 | TARGET_HIT | 0.50 | 2.44% |
| SELL | retest1 | 2026-04-16 10:50:00 | 3510.20 | 2026-04-16 12:50:00 | 3489.49 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-04-16 10:50:00 | 3510.20 | 2026-04-16 15:20:00 | 3492.50 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-17 10:25:00 | 3472.50 | 2026-04-17 10:30:00 | 3456.24 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-17 10:25:00 | 3472.50 | 2026-04-17 11:20:00 | 3472.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:50:00 | 3543.20 | 2026-04-22 11:05:00 | 3559.68 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-22 10:50:00 | 3543.20 | 2026-04-22 15:20:00 | 3639.70 | TARGET_HIT | 0.50 | 2.72% |
| SELL | retest1 | 2026-04-27 11:05:00 | 3530.70 | 2026-04-27 11:15:00 | 3542.37 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-28 10:35:00 | 3549.10 | 2026-04-28 11:40:00 | 3531.41 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-28 10:35:00 | 3549.10 | 2026-04-28 11:45:00 | 3549.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 11:10:00 | 3516.90 | 2026-04-29 11:20:00 | 3525.32 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-05-04 11:00:00 | 3454.50 | 2026-05-04 11:30:00 | 3440.80 | STOP_HIT | 1.00 | -0.40% |
