# Supreme Industries Ltd. (SUPREMEIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3654.00
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 7
- **Target hits / Stop hits / Partials:** 3 / 7 / 7
- **Avg / median % per leg:** 0.39% / 0.32%
- **Sum % (uncompounded):** 6.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 2 | 3 | 4 | 0.54% | 4.9% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 2 | 3 | 4 | 0.54% | 4.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.23% | 1.8% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.23% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 10 | 58.8% | 3 | 7 | 7 | 0.39% | 6.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 11:10:00 | 3728.40 | 3693.21 | 0.00 | ORB-long ORB[3680.00,3726.90] vol=3.2x ATR=9.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:30:00 | 3742.45 | 3700.52 | 0.00 | T1 1.5R @ 3742.45 |
| Target hit | 2026-02-11 15:20:00 | 3849.80 | 3796.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:00:00 | 3757.00 | 3784.91 | 0.00 | ORB-short ORB[3784.50,3831.70] vol=1.8x ATR=10.00 |
| Stop hit — per-position SL triggered | 2026-02-13 11:40:00 | 3767.00 | 3780.21 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 3822.00 | 3800.84 | 0.00 | ORB-long ORB[3749.50,3791.00] vol=3.7x ATR=10.30 |
| Stop hit — per-position SL triggered | 2026-02-16 10:50:00 | 3811.70 | 3807.18 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:30:00 | 3917.90 | 3909.14 | 0.00 | ORB-long ORB[3860.90,3916.70] vol=15.4x ATR=12.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:10:00 | 3936.41 | 3911.81 | 0.00 | T1 1.5R @ 3936.41 |
| Stop hit — per-position SL triggered | 2026-02-20 15:10:00 | 3917.90 | 3919.66 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 3863.90 | 3894.91 | 0.00 | ORB-short ORB[3892.30,3942.50] vol=3.9x ATR=8.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:20:00 | 3850.66 | 3868.66 | 0.00 | T1 1.5R @ 3850.66 |
| Stop hit — per-position SL triggered | 2026-03-05 12:00:00 | 3863.90 | 3864.88 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:00:00 | 3991.00 | 3979.21 | 0.00 | ORB-long ORB[3900.00,3956.00] vol=1.8x ATR=14.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:15:00 | 4013.18 | 3988.75 | 0.00 | T1 1.5R @ 4013.18 |
| Target hit | 2026-03-18 10:40:00 | 3991.20 | 3994.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 11:15:00 | 3735.50 | 3687.73 | 0.00 | ORB-long ORB[3646.20,3691.70] vol=2.2x ATR=12.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 11:55:00 | 3754.06 | 3698.25 | 0.00 | T1 1.5R @ 3754.06 |
| Stop hit — per-position SL triggered | 2026-04-07 12:05:00 | 3735.50 | 3701.84 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 3638.30 | 3650.17 | 0.00 | ORB-short ORB[3640.10,3662.00] vol=3.4x ATR=7.89 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 3646.19 | 3651.66 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:40:00 | 3715.00 | 3725.34 | 0.00 | ORB-short ORB[3720.10,3746.30] vol=2.0x ATR=9.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:00:00 | 3700.12 | 3721.43 | 0.00 | T1 1.5R @ 3700.12 |
| Target hit | 2026-05-07 15:20:00 | 3669.00 | 3682.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:05:00 | 3663.00 | 3675.78 | 0.00 | ORB-short ORB[3670.10,3693.80] vol=1.7x ATR=7.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:55:00 | 3651.22 | 3670.64 | 0.00 | T1 1.5R @ 3651.22 |
| Stop hit — per-position SL triggered | 2026-05-08 12:00:00 | 3663.00 | 3670.45 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 11:10:00 | 3728.40 | 2026-02-11 11:30:00 | 3742.45 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-11 11:10:00 | 3728.40 | 2026-02-11 15:20:00 | 3849.80 | TARGET_HIT | 0.50 | 3.26% |
| SELL | retest1 | 2026-02-13 11:00:00 | 3757.00 | 2026-02-13 11:40:00 | 3767.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-16 10:45:00 | 3822.00 | 2026-02-16 10:50:00 | 3811.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-20 10:30:00 | 3917.90 | 2026-02-20 11:10:00 | 3936.41 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-20 10:30:00 | 3917.90 | 2026-02-20 15:10:00 | 3917.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 11:05:00 | 3863.90 | 2026-03-05 11:20:00 | 3850.66 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-03-05 11:05:00 | 3863.90 | 2026-03-05 12:00:00 | 3863.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 10:00:00 | 3991.00 | 2026-03-18 10:15:00 | 4013.18 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-03-18 10:00:00 | 3991.00 | 2026-03-18 10:40:00 | 3991.20 | TARGET_HIT | 0.50 | 0.01% |
| BUY | retest1 | 2026-04-07 11:15:00 | 3735.50 | 2026-04-07 11:55:00 | 3754.06 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-07 11:15:00 | 3735.50 | 2026-04-07 12:05:00 | 3735.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 11:00:00 | 3638.30 | 2026-05-06 11:10:00 | 3646.19 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-07 10:40:00 | 3715.00 | 2026-05-07 11:00:00 | 3700.12 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-05-07 10:40:00 | 3715.00 | 2026-05-07 15:20:00 | 3669.00 | TARGET_HIT | 0.50 | 1.24% |
| SELL | retest1 | 2026-05-08 11:05:00 | 3663.00 | 2026-05-08 11:55:00 | 3651.22 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-05-08 11:05:00 | 3663.00 | 2026-05-08 12:00:00 | 3663.00 | STOP_HIT | 0.50 | 0.00% |
