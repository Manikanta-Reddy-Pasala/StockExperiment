# TRENT (TRENT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4249.10
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
| ENTRY1 | 21 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 6 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 15
- **Target hits / Stop hits / Partials:** 6 / 15 / 9
- **Avg / median % per leg:** 0.17% / 0.27%
- **Sum % (uncompounded):** 5.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 8 | 50.0% | 3 | 8 | 5 | 0.08% | 1.3% |
| BUY @ 2nd Alert (retest1) | 16 | 8 | 50.0% | 3 | 8 | 5 | 0.08% | 1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.27% | 3.8% |
| SELL @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.27% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 30 | 15 | 50.0% | 6 | 15 | 9 | 0.17% | 5.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 4163.50 | 4127.90 | 0.00 | ORB-long ORB[4101.10,4130.70] vol=1.6x ATR=18.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:50:00 | 4191.55 | 4150.55 | 0.00 | T1 1.5R @ 4191.55 |
| Target hit | 2026-02-09 15:15:00 | 4174.90 | 4176.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:25:00 | 4254.40 | 4226.57 | 0.00 | ORB-long ORB[4182.80,4228.10] vol=2.7x ATR=13.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:50:00 | 4273.98 | 4237.74 | 0.00 | T1 1.5R @ 4273.98 |
| Stop hit — per-position SL triggered | 2026-02-12 11:20:00 | 4254.40 | 4243.62 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:55:00 | 4289.50 | 4260.79 | 0.00 | ORB-long ORB[4222.80,4252.00] vol=1.9x ATR=13.86 |
| Stop hit — per-position SL triggered | 2026-02-16 10:05:00 | 4275.64 | 4268.07 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 4156.20 | 4166.51 | 0.00 | ORB-short ORB[4168.50,4201.00] vol=2.9x ATR=9.19 |
| Stop hit — per-position SL triggered | 2026-02-18 12:05:00 | 4165.39 | 4163.59 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 4123.40 | 4153.10 | 0.00 | ORB-short ORB[4160.00,4181.10] vol=2.0x ATR=8.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:50:00 | 4110.32 | 4147.38 | 0.00 | T1 1.5R @ 4110.32 |
| Target hit | 2026-02-19 15:20:00 | 4070.90 | 4083.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 4057.70 | 4083.15 | 0.00 | ORB-short ORB[4079.50,4119.90] vol=3.4x ATR=9.94 |
| Stop hit — per-position SL triggered | 2026-02-23 10:45:00 | 4067.64 | 4079.29 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 3916.30 | 3922.65 | 0.00 | ORB-short ORB[3921.20,3946.70] vol=1.8x ATR=8.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:30:00 | 3903.38 | 3918.70 | 0.00 | T1 1.5R @ 3903.38 |
| Target hit | 2026-02-26 15:20:00 | 3860.30 | 3875.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 11:15:00 | 3754.40 | 3771.97 | 0.00 | ORB-short ORB[3755.80,3798.00] vol=4.7x ATR=12.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:45:00 | 3736.06 | 3768.16 | 0.00 | T1 1.5R @ 3736.06 |
| Stop hit — per-position SL triggered | 2026-03-04 14:15:00 | 3754.40 | 3756.55 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 3730.60 | 3744.41 | 0.00 | ORB-short ORB[3738.10,3785.10] vol=1.9x ATR=8.78 |
| Stop hit — per-position SL triggered | 2026-03-05 11:30:00 | 3739.38 | 3743.47 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 3676.00 | 3701.18 | 0.00 | ORB-short ORB[3701.00,3727.10] vol=1.7x ATR=9.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:30:00 | 3662.12 | 3695.24 | 0.00 | T1 1.5R @ 3662.12 |
| Target hit | 2026-03-11 15:20:00 | 3629.60 | 3659.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 3540.00 | 3524.58 | 0.00 | ORB-long ORB[3500.00,3533.00] vol=1.8x ATR=13.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:00:00 | 3560.58 | 3535.27 | 0.00 | T1 1.5R @ 3560.58 |
| Target hit | 2026-03-13 12:25:00 | 3550.60 | 3551.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2026-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:10:00 | 3559.50 | 3532.75 | 0.00 | ORB-long ORB[3506.40,3533.00] vol=2.6x ATR=13.33 |
| Stop hit — per-position SL triggered | 2026-03-20 10:20:00 | 3546.17 | 3534.79 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 09:35:00 | 3854.50 | 3812.76 | 0.00 | ORB-long ORB[3776.30,3828.60] vol=1.6x ATR=23.13 |
| Stop hit — per-position SL triggered | 2026-04-07 09:40:00 | 3831.37 | 3820.78 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:40:00 | 3926.30 | 3903.66 | 0.00 | ORB-long ORB[3871.30,3912.90] vol=3.4x ATR=14.56 |
| Stop hit — per-position SL triggered | 2026-04-10 10:45:00 | 3911.74 | 3903.94 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:40:00 | 4063.00 | 4002.02 | 0.00 | ORB-long ORB[4000.20,4034.80] vol=2.9x ATR=16.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:55:00 | 4088.41 | 4025.69 | 0.00 | T1 1.5R @ 4088.41 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 4063.00 | 4034.53 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:15:00 | 4162.90 | 4125.98 | 0.00 | ORB-long ORB[4090.00,4129.20] vol=1.6x ATR=16.33 |
| Stop hit — per-position SL triggered | 2026-04-17 10:50:00 | 4146.57 | 4131.88 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 4318.00 | 4278.06 | 0.00 | ORB-long ORB[4242.80,4278.70] vol=1.8x ATR=12.66 |
| Stop hit — per-position SL triggered | 2026-04-21 10:25:00 | 4305.34 | 4286.27 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:05:00 | 4294.60 | 4315.75 | 0.00 | ORB-short ORB[4297.30,4348.50] vol=1.5x ATR=17.30 |
| Stop hit — per-position SL triggered | 2026-04-27 10:30:00 | 4311.90 | 4313.83 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 4207.50 | 4234.70 | 0.00 | ORB-short ORB[4223.10,4269.90] vol=1.6x ATR=13.25 |
| Stop hit — per-position SL triggered | 2026-04-28 09:45:00 | 4220.75 | 4232.94 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:10:00 | 4260.00 | 4267.82 | 0.00 | ORB-short ORB[4262.40,4304.70] vol=2.3x ATR=9.40 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 4269.40 | 4268.59 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 4223.00 | 4202.30 | 0.00 | ORB-long ORB[4156.50,4216.50] vol=1.8x ATR=11.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:45:00 | 4239.93 | 4218.05 | 0.00 | T1 1.5R @ 4239.93 |
| Target hit | 2026-05-06 11:55:00 | 4239.30 | 4239.82 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 4163.50 | 2026-02-09 12:50:00 | 4191.55 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-02-09 10:30:00 | 4163.50 | 2026-02-09 15:15:00 | 4174.90 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-12 10:25:00 | 4254.40 | 2026-02-12 10:50:00 | 4273.98 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-12 10:25:00 | 4254.40 | 2026-02-12 11:20:00 | 4254.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 09:55:00 | 4289.50 | 2026-02-16 10:05:00 | 4275.64 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-18 11:15:00 | 4156.20 | 2026-02-18 12:05:00 | 4165.39 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-19 10:40:00 | 4123.40 | 2026-02-19 10:50:00 | 4110.32 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-19 10:40:00 | 4123.40 | 2026-02-19 15:20:00 | 4070.90 | TARGET_HIT | 0.50 | 1.27% |
| SELL | retest1 | 2026-02-23 10:40:00 | 4057.70 | 2026-02-23 10:45:00 | 4067.64 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-26 10:50:00 | 3916.30 | 2026-02-26 11:30:00 | 3903.38 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-26 10:50:00 | 3916.30 | 2026-02-26 15:20:00 | 3860.30 | TARGET_HIT | 0.50 | 1.43% |
| SELL | retest1 | 2026-03-04 11:15:00 | 3754.40 | 2026-03-04 11:45:00 | 3736.06 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-04 11:15:00 | 3754.40 | 2026-03-04 14:15:00 | 3754.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 11:00:00 | 3730.60 | 2026-03-05 11:30:00 | 3739.38 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-11 11:10:00 | 3676.00 | 2026-03-11 11:30:00 | 3662.12 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-03-11 11:10:00 | 3676.00 | 2026-03-11 15:20:00 | 3629.60 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2026-03-13 09:45:00 | 3540.00 | 2026-03-13 10:00:00 | 3560.58 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-13 09:45:00 | 3540.00 | 2026-03-13 12:25:00 | 3550.60 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2026-03-20 10:10:00 | 3559.50 | 2026-03-20 10:20:00 | 3546.17 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-07 09:35:00 | 3854.50 | 2026-04-07 09:40:00 | 3831.37 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2026-04-10 10:40:00 | 3926.30 | 2026-04-10 10:45:00 | 3911.74 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-16 10:40:00 | 4063.00 | 2026-04-16 10:55:00 | 4088.41 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-16 10:40:00 | 4063.00 | 2026-04-16 11:25:00 | 4063.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:15:00 | 4162.90 | 2026-04-17 10:50:00 | 4146.57 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-21 10:10:00 | 4318.00 | 2026-04-21 10:25:00 | 4305.34 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-27 10:05:00 | 4294.60 | 2026-04-27 10:30:00 | 4311.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-28 09:40:00 | 4207.50 | 2026-04-28 09:45:00 | 4220.75 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-29 11:10:00 | 4260.00 | 2026-04-29 11:15:00 | 4269.40 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-05-06 09:40:00 | 4223.00 | 2026-05-06 09:45:00 | 4239.93 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-05-06 09:40:00 | 4223.00 | 2026-05-06 11:55:00 | 4239.30 | TARGET_HIT | 0.50 | 0.39% |
