# Escorts Kubota Ltd. (ESCORTS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3148.00
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 12
- **Target hits / Stop hits / Partials:** 4 / 12 / 8
- **Avg / median % per leg:** 0.28% / 0.15%
- **Sum % (uncompounded):** 6.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.19% | -1.1% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.19% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 11 | 61.1% | 4 | 7 | 7 | 0.44% | 8.0% |
| SELL @ 2nd Alert (retest1) | 18 | 11 | 61.1% | 4 | 7 | 7 | 0.44% | 8.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 12 | 50.0% | 4 | 12 | 8 | 0.28% | 6.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:05:00 | 3631.00 | 3670.89 | 0.00 | ORB-short ORB[3644.10,3697.00] vol=2.4x ATR=12.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 3612.58 | 3660.81 | 0.00 | T1 1.5R @ 3612.58 |
| Stop hit — per-position SL triggered | 2026-02-12 11:30:00 | 3631.00 | 3643.79 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 3551.50 | 3538.39 | 0.00 | ORB-long ORB[3514.90,3545.90] vol=1.9x ATR=13.13 |
| Stop hit — per-position SL triggered | 2026-02-18 09:40:00 | 3538.37 | 3539.47 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:25:00 | 3482.10 | 3522.08 | 0.00 | ORB-short ORB[3525.30,3568.00] vol=2.6x ATR=11.60 |
| Stop hit — per-position SL triggered | 2026-02-19 10:30:00 | 3493.70 | 3518.19 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:05:00 | 3598.30 | 3615.35 | 0.00 | ORB-short ORB[3599.80,3626.30] vol=1.7x ATR=10.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:30:00 | 3581.82 | 3610.39 | 0.00 | T1 1.5R @ 3581.82 |
| Stop hit — per-position SL triggered | 2026-02-26 14:35:00 | 3598.30 | 3597.56 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:50:00 | 3256.60 | 3267.52 | 0.00 | ORB-short ORB[3258.90,3288.30] vol=1.8x ATR=9.36 |
| Stop hit — per-position SL triggered | 2026-03-05 11:20:00 | 3265.96 | 3264.04 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:35:00 | 3370.80 | 3369.04 | 0.00 | ORB-long ORB[3335.10,3370.00] vol=2.1x ATR=15.45 |
| Stop hit — per-position SL triggered | 2026-03-11 10:20:00 | 3355.35 | 3369.52 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:10:00 | 3171.00 | 3199.35 | 0.00 | ORB-short ORB[3205.00,3229.80] vol=2.3x ATR=9.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:20:00 | 3156.58 | 3189.84 | 0.00 | T1 1.5R @ 3156.58 |
| Stop hit — per-position SL triggered | 2026-03-13 11:40:00 | 3171.00 | 3186.34 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:50:00 | 3006.30 | 3030.55 | 0.00 | ORB-short ORB[3016.00,3060.00] vol=1.9x ATR=11.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:55:00 | 2989.61 | 3021.96 | 0.00 | T1 1.5R @ 2989.61 |
| Target hit | 2026-03-19 15:20:00 | 2912.00 | 2943.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 3293.00 | 3269.60 | 0.00 | ORB-long ORB[3242.10,3287.00] vol=1.6x ATR=7.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:35:00 | 3303.72 | 3276.07 | 0.00 | T1 1.5R @ 3303.72 |
| Stop hit — per-position SL triggered | 2026-04-16 11:40:00 | 3293.00 | 3276.36 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 3302.90 | 3281.65 | 0.00 | ORB-long ORB[3273.00,3296.10] vol=1.7x ATR=10.13 |
| Stop hit — per-position SL triggered | 2026-04-17 10:30:00 | 3292.77 | 3287.24 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:05:00 | 3325.60 | 3354.96 | 0.00 | ORB-short ORB[3333.10,3377.90] vol=1.8x ATR=7.43 |
| Stop hit — per-position SL triggered | 2026-04-22 11:10:00 | 3333.03 | 3353.59 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:30:00 | 3318.00 | 3351.13 | 0.00 | ORB-short ORB[3333.00,3382.00] vol=1.7x ATR=10.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:45:00 | 3302.40 | 3344.41 | 0.00 | T1 1.5R @ 3302.40 |
| Target hit | 2026-04-29 15:20:00 | 3301.70 | 3306.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-04-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:00:00 | 3267.10 | 3291.21 | 0.00 | ORB-short ORB[3277.00,3314.40] vol=1.5x ATR=12.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:00:00 | 3248.80 | 3273.54 | 0.00 | T1 1.5R @ 3248.80 |
| Target hit | 2026-04-30 13:55:00 | 3262.20 | 3242.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 3261.00 | 3292.19 | 0.00 | ORB-short ORB[3280.00,3325.60] vol=3.1x ATR=11.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:50:00 | 3243.14 | 3282.37 | 0.00 | T1 1.5R @ 3243.14 |
| Target hit | 2026-05-04 15:20:00 | 3203.90 | 3234.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 3200.00 | 3227.78 | 0.00 | ORB-short ORB[3202.90,3242.10] vol=1.9x ATR=9.05 |
| Stop hit — per-position SL triggered | 2026-05-05 11:35:00 | 3209.05 | 3226.11 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 3279.90 | 3262.05 | 0.00 | ORB-long ORB[3220.30,3263.90] vol=6.5x ATR=10.72 |
| Stop hit — per-position SL triggered | 2026-05-06 09:35:00 | 3269.18 | 3263.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:05:00 | 3631.00 | 2026-02-12 11:15:00 | 3612.58 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-12 11:05:00 | 3631.00 | 2026-02-12 11:30:00 | 3631.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:35:00 | 3551.50 | 2026-02-18 09:40:00 | 3538.37 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-19 10:25:00 | 3482.10 | 2026-02-19 10:30:00 | 3493.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-26 11:05:00 | 3598.30 | 2026-02-26 11:30:00 | 3581.82 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-26 11:05:00 | 3598.30 | 2026-02-26 14:35:00 | 3598.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 09:50:00 | 3256.60 | 2026-03-05 11:20:00 | 3265.96 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-11 09:35:00 | 3370.80 | 2026-03-11 10:20:00 | 3355.35 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-03-13 11:10:00 | 3171.00 | 2026-03-13 11:20:00 | 3156.58 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-13 11:10:00 | 3171.00 | 2026-03-13 11:40:00 | 3171.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 09:50:00 | 3006.30 | 2026-03-19 09:55:00 | 2989.61 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-19 09:50:00 | 3006.30 | 2026-03-19 15:20:00 | 2912.00 | TARGET_HIT | 0.50 | 3.14% |
| BUY | retest1 | 2026-04-16 11:15:00 | 3293.00 | 2026-04-16 11:35:00 | 3303.72 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-04-16 11:15:00 | 3293.00 | 2026-04-16 11:40:00 | 3293.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:05:00 | 3302.90 | 2026-04-17 10:30:00 | 3292.77 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-22 11:05:00 | 3325.60 | 2026-04-22 11:10:00 | 3333.03 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-29 10:30:00 | 3318.00 | 2026-04-29 10:45:00 | 3302.40 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-29 10:30:00 | 3318.00 | 2026-04-29 15:20:00 | 3301.70 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-30 10:00:00 | 3267.10 | 2026-04-30 11:00:00 | 3248.80 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-30 10:00:00 | 3267.10 | 2026-04-30 13:55:00 | 3262.20 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-05-04 11:10:00 | 3261.00 | 2026-05-04 11:50:00 | 3243.14 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-05-04 11:10:00 | 3261.00 | 2026-05-04 15:20:00 | 3203.90 | TARGET_HIT | 0.50 | 1.75% |
| SELL | retest1 | 2026-05-05 11:00:00 | 3200.00 | 2026-05-05 11:35:00 | 3209.05 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-05-06 09:30:00 | 3279.90 | 2026-05-06 09:35:00 | 3269.18 | STOP_HIT | 1.00 | -0.33% |
