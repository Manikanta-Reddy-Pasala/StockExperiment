# Avenue Supermarts Ltd. (DMART)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4396.10
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 16
- **Target hits / Stop hits / Partials:** 1 / 16 / 4
- **Avg / median % per leg:** 0.03% / -0.24%
- **Sum % (uncompounded):** 0.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 3 | 27.3% | 1 | 8 | 2 | 0.13% | 1.4% |
| BUY @ 2nd Alert (retest1) | 11 | 3 | 27.3% | 1 | 8 | 2 | 0.13% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.08% | -0.8% |
| SELL @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.08% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 5 | 23.8% | 1 | 16 | 4 | 0.03% | 0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 3964.50 | 3940.39 | 0.00 | ORB-long ORB[3920.00,3956.00] vol=1.6x ATR=9.56 |
| Stop hit — per-position SL triggered | 2026-02-10 09:55:00 | 3954.94 | 3944.13 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 3998.20 | 3999.79 | 0.00 | ORB-short ORB[4005.70,4032.20] vol=2.8x ATR=8.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:10:00 | 3985.66 | 3999.42 | 0.00 | T1 1.5R @ 3985.66 |
| Stop hit — per-position SL triggered | 2026-02-11 11:35:00 | 3998.20 | 3999.02 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 3871.00 | 3915.53 | 0.00 | ORB-short ORB[3908.00,3944.00] vol=2.0x ATR=8.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:50:00 | 3857.70 | 3909.17 | 0.00 | T1 1.5R @ 3857.70 |
| Stop hit — per-position SL triggered | 2026-02-19 13:20:00 | 3871.00 | 3900.49 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 3851.10 | 3860.86 | 0.00 | ORB-short ORB[3871.20,3900.00] vol=2.6x ATR=10.08 |
| Stop hit — per-position SL triggered | 2026-02-23 11:30:00 | 3861.18 | 3858.15 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 3808.30 | 3823.69 | 0.00 | ORB-short ORB[3816.00,3847.10] vol=2.4x ATR=8.57 |
| Stop hit — per-position SL triggered | 2026-02-24 11:55:00 | 3816.87 | 3820.73 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:10:00 | 3940.90 | 3904.91 | 0.00 | ORB-long ORB[3869.50,3906.40] vol=3.7x ATR=10.08 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 3930.82 | 3906.92 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:05:00 | 3814.90 | 3824.12 | 0.00 | ORB-short ORB[3840.80,3880.20] vol=3.4x ATR=9.37 |
| Stop hit — per-position SL triggered | 2026-02-27 10:30:00 | 3824.27 | 3822.86 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 3749.00 | 3750.50 | 0.00 | ORB-short ORB[3754.60,3790.50] vol=2.8x ATR=8.36 |
| Stop hit — per-position SL triggered | 2026-03-05 11:05:00 | 3757.36 | 3751.43 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 09:30:00 | 3897.00 | 3861.30 | 0.00 | ORB-long ORB[3814.50,3868.00] vol=3.2x ATR=16.52 |
| Stop hit — per-position SL triggered | 2026-03-09 09:45:00 | 3880.48 | 3869.80 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 3846.90 | 3826.43 | 0.00 | ORB-long ORB[3810.00,3842.00] vol=1.8x ATR=11.40 |
| Stop hit — per-position SL triggered | 2026-03-17 11:05:00 | 3835.50 | 3830.38 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:45:00 | 3839.10 | 3816.96 | 0.00 | ORB-long ORB[3802.20,3829.00] vol=1.6x ATR=11.82 |
| Stop hit — per-position SL triggered | 2026-03-20 11:50:00 | 3827.28 | 3821.93 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 10:00:00 | 3925.80 | 3918.28 | 0.00 | ORB-long ORB[3865.50,3915.90] vol=4.5x ATR=13.51 |
| Stop hit — per-position SL triggered | 2026-03-27 10:15:00 | 3912.29 | 3918.01 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 09:35:00 | 3922.80 | 3912.71 | 0.00 | ORB-long ORB[3877.00,3922.50] vol=2.6x ATR=15.85 |
| Stop hit — per-position SL triggered | 2026-03-30 09:40:00 | 3906.95 | 3912.89 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:50:00 | 4442.90 | 4466.43 | 0.00 | ORB-short ORB[4466.50,4506.90] vol=1.6x ATR=10.90 |
| Stop hit — per-position SL triggered | 2026-04-16 11:05:00 | 4453.80 | 4464.47 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:45:00 | 4505.00 | 4473.35 | 0.00 | ORB-long ORB[4435.10,4475.50] vol=2.4x ATR=13.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 09:50:00 | 4525.57 | 4483.25 | 0.00 | T1 1.5R @ 4525.57 |
| Target hit | 2026-04-17 15:20:00 | 4638.90 | 4585.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-04-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:35:00 | 4459.00 | 4491.43 | 0.00 | ORB-short ORB[4507.40,4559.50] vol=1.9x ATR=11.93 |
| Stop hit — per-position SL triggered | 2026-04-24 10:45:00 | 4470.93 | 4488.19 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 4544.00 | 4511.48 | 0.00 | ORB-long ORB[4495.00,4533.10] vol=1.8x ATR=8.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:45:00 | 4556.55 | 4524.55 | 0.00 | T1 1.5R @ 4556.55 |
| Stop hit — per-position SL triggered | 2026-04-28 15:00:00 | 4544.00 | 4560.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 3964.50 | 2026-02-10 09:55:00 | 3954.94 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-11 11:00:00 | 3998.20 | 2026-02-11 11:10:00 | 3985.66 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-11 11:00:00 | 3998.20 | 2026-02-11 11:35:00 | 3998.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:15:00 | 3871.00 | 2026-02-19 11:50:00 | 3857.70 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-19 11:15:00 | 3871.00 | 2026-02-19 13:20:00 | 3871.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 10:40:00 | 3851.10 | 2026-02-23 11:30:00 | 3861.18 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-24 11:10:00 | 3808.30 | 2026-02-24 11:55:00 | 3816.87 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-25 11:10:00 | 3940.90 | 2026-02-25 11:20:00 | 3930.82 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-27 10:05:00 | 3814.90 | 2026-02-27 10:30:00 | 3824.27 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-05 10:55:00 | 3749.00 | 2026-03-05 11:05:00 | 3757.36 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-03-09 09:30:00 | 3897.00 | 2026-03-09 09:45:00 | 3880.48 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-17 10:35:00 | 3846.90 | 2026-03-17 11:05:00 | 3835.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-20 10:45:00 | 3839.10 | 2026-03-20 11:50:00 | 3827.28 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-27 10:00:00 | 3925.80 | 2026-03-27 10:15:00 | 3912.29 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-30 09:35:00 | 3922.80 | 2026-03-30 09:40:00 | 3906.95 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-16 10:50:00 | 4442.90 | 2026-04-16 11:05:00 | 4453.80 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-17 09:45:00 | 4505.00 | 2026-04-17 09:50:00 | 4525.57 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-17 09:45:00 | 4505.00 | 2026-04-17 15:20:00 | 4638.90 | TARGET_HIT | 0.50 | 2.97% |
| SELL | retest1 | 2026-04-24 10:35:00 | 4459.00 | 2026-04-24 10:45:00 | 4470.93 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-28 11:10:00 | 4544.00 | 2026-04-28 11:45:00 | 4556.55 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-04-28 11:10:00 | 4544.00 | 2026-04-28 15:00:00 | 4544.00 | STOP_HIT | 0.50 | 0.00% |
