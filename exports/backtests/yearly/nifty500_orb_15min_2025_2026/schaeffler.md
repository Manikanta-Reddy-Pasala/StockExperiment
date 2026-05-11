# Schaeffler India Ltd. (SCHAEFFLER)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 4226.20
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
| ENTRY1 | 86 |
| ENTRY2 | 0 |
| PARTIAL | 41 |
| TARGET_HIT | 17 |
| STOP_HIT | 69 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 58 / 69
- **Target hits / Stop hits / Partials:** 17 / 69 / 41
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 26.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 26 | 40.6% | 6 | 38 | 20 | 0.13% | 8.5% |
| BUY @ 2nd Alert (retest1) | 64 | 26 | 40.6% | 6 | 38 | 20 | 0.13% | 8.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 63 | 32 | 50.8% | 11 | 31 | 21 | 0.29% | 18.5% |
| SELL @ 2nd Alert (retest1) | 63 | 32 | 50.8% | 11 | 31 | 21 | 0.29% | 18.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 127 | 58 | 45.7% | 17 | 69 | 41 | 0.21% | 27.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:35:00 | 3808.70 | 3764.67 | 0.00 | ORB-long ORB[3710.20,3766.40] vol=3.6x ATR=16.60 |
| Stop hit — per-position SL triggered | 2025-05-13 09:45:00 | 3792.10 | 3786.94 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 09:35:00 | 4043.10 | 4017.61 | 0.00 | ORB-long ORB[3991.50,4032.00] vol=1.9x ATR=17.04 |
| Stop hit — per-position SL triggered | 2025-05-27 09:45:00 | 4026.06 | 4022.15 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:35:00 | 3979.20 | 3992.74 | 0.00 | ORB-short ORB[3985.20,4006.10] vol=3.4x ATR=12.03 |
| Stop hit — per-position SL triggered | 2025-05-28 10:40:00 | 3991.23 | 3995.40 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 10:35:00 | 4160.90 | 4180.67 | 0.00 | ORB-short ORB[4170.10,4209.90] vol=2.8x ATR=18.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 11:30:00 | 4133.10 | 4168.03 | 0.00 | T1 1.5R @ 4133.10 |
| Target hit | 2025-06-02 15:20:00 | 4128.90 | 4150.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 11:15:00 | 3982.60 | 3998.60 | 0.00 | ORB-short ORB[3992.80,4027.60] vol=4.4x ATR=9.45 |
| Stop hit — per-position SL triggered | 2025-06-18 11:50:00 | 3992.05 | 3985.61 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 09:50:00 | 3947.60 | 3964.33 | 0.00 | ORB-short ORB[3957.50,3997.50] vol=2.2x ATR=7.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:05:00 | 3935.92 | 3959.70 | 0.00 | T1 1.5R @ 3935.92 |
| Target hit | 2025-06-19 15:20:00 | 3860.00 | 3884.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2025-06-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:40:00 | 3903.80 | 3872.08 | 0.00 | ORB-long ORB[3857.30,3890.00] vol=6.9x ATR=13.51 |
| Stop hit — per-position SL triggered | 2025-06-20 10:45:00 | 3890.29 | 3872.52 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 10:50:00 | 3912.80 | 3899.87 | 0.00 | ORB-long ORB[3877.00,3908.60] vol=1.6x ATR=11.77 |
| Stop hit — per-position SL triggered | 2025-06-23 12:35:00 | 3901.03 | 3902.82 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:15:00 | 3878.50 | 3902.28 | 0.00 | ORB-short ORB[3915.00,3964.50] vol=2.2x ATR=8.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 11:35:00 | 3865.06 | 3898.86 | 0.00 | T1 1.5R @ 3865.06 |
| Stop hit — per-position SL triggered | 2025-06-26 12:40:00 | 3878.50 | 3889.20 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 09:45:00 | 3984.20 | 3953.85 | 0.00 | ORB-long ORB[3920.00,3960.10] vol=6.0x ATR=20.17 |
| Stop hit — per-position SL triggered | 2025-06-30 10:05:00 | 3964.03 | 3961.58 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 10:10:00 | 3956.90 | 3993.94 | 0.00 | ORB-short ORB[3998.90,4050.40] vol=2.5x ATR=12.75 |
| Stop hit — per-position SL triggered | 2025-07-04 10:20:00 | 3969.65 | 3988.56 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:45:00 | 4294.80 | 4270.28 | 0.00 | ORB-long ORB[4235.00,4284.90] vol=2.4x ATR=17.19 |
| Stop hit — per-position SL triggered | 2025-07-10 09:50:00 | 4277.61 | 4272.83 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:50:00 | 4281.80 | 4259.05 | 0.00 | ORB-long ORB[4224.10,4273.20] vol=3.2x ATR=15.13 |
| Stop hit — per-position SL triggered | 2025-07-15 11:00:00 | 4266.67 | 4260.43 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 10:05:00 | 4164.10 | 4179.80 | 0.00 | ORB-short ORB[4183.60,4223.00] vol=2.0x ATR=10.05 |
| Stop hit — per-position SL triggered | 2025-07-16 10:20:00 | 4174.15 | 4178.81 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 10:30:00 | 4179.10 | 4187.61 | 0.00 | ORB-short ORB[4182.00,4220.00] vol=2.2x ATR=9.77 |
| Stop hit — per-position SL triggered | 2025-07-17 10:35:00 | 4188.87 | 4188.04 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:50:00 | 4160.00 | 4161.82 | 0.00 | ORB-short ORB[4165.00,4199.90] vol=2.2x ATR=6.98 |
| Stop hit — per-position SL triggered | 2025-07-18 11:00:00 | 4166.98 | 4162.33 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:20:00 | 4241.20 | 4212.69 | 0.00 | ORB-long ORB[4170.90,4224.40] vol=3.9x ATR=11.88 |
| Stop hit — per-position SL triggered | 2025-07-21 10:25:00 | 4229.32 | 4215.32 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 10:55:00 | 4079.30 | 4046.52 | 0.00 | ORB-long ORB[4005.00,4055.70] vol=1.6x ATR=13.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:05:00 | 4099.46 | 4054.49 | 0.00 | T1 1.5R @ 4099.46 |
| Target hit | 2025-07-31 15:20:00 | 4126.70 | 4105.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-08-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 10:50:00 | 4075.10 | 4092.15 | 0.00 | ORB-short ORB[4081.00,4138.00] vol=2.3x ATR=16.07 |
| Stop hit — per-position SL triggered | 2025-08-01 11:00:00 | 4091.17 | 4092.01 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 11:15:00 | 4045.50 | 4091.00 | 0.00 | ORB-short ORB[4081.60,4138.10] vol=4.4x ATR=14.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 12:10:00 | 4023.47 | 4073.05 | 0.00 | T1 1.5R @ 4023.47 |
| Stop hit — per-position SL triggered | 2025-08-04 15:05:00 | 4045.50 | 4055.01 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 11:15:00 | 4120.80 | 4086.65 | 0.00 | ORB-long ORB[4047.60,4106.20] vol=2.3x ATR=16.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 11:20:00 | 4145.12 | 4106.44 | 0.00 | T1 1.5R @ 4145.12 |
| Stop hit — per-position SL triggered | 2025-08-05 14:00:00 | 4120.80 | 4142.88 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:10:00 | 4004.20 | 4031.33 | 0.00 | ORB-short ORB[4005.10,4060.60] vol=2.4x ATR=10.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:15:00 | 3988.70 | 4028.20 | 0.00 | T1 1.5R @ 3988.70 |
| Stop hit — per-position SL triggered | 2025-08-07 12:40:00 | 4004.20 | 4004.01 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:50:00 | 3900.80 | 3893.90 | 0.00 | ORB-long ORB[3865.00,3891.90] vol=2.1x ATR=12.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:55:00 | 3918.80 | 3895.32 | 0.00 | T1 1.5R @ 3918.80 |
| Stop hit — per-position SL triggered | 2025-08-18 11:30:00 | 3900.80 | 3899.72 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:20:00 | 3963.80 | 3971.42 | 0.00 | ORB-short ORB[3976.10,4000.00] vol=1.5x ATR=11.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 11:45:00 | 3946.29 | 3965.42 | 0.00 | T1 1.5R @ 3946.29 |
| Target hit | 2025-08-22 15:20:00 | 3941.70 | 3954.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-09-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:05:00 | 3924.40 | 3901.50 | 0.00 | ORB-long ORB[3861.00,3897.90] vol=2.7x ATR=13.11 |
| Stop hit — per-position SL triggered | 2025-09-02 10:35:00 | 3911.29 | 3904.63 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:25:00 | 3852.00 | 3831.37 | 0.00 | ORB-long ORB[3750.50,3806.00] vol=13.5x ATR=11.49 |
| Stop hit — per-position SL triggered | 2025-09-08 10:30:00 | 3840.51 | 3831.73 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:35:00 | 3955.00 | 3934.90 | 0.00 | ORB-long ORB[3897.30,3950.00] vol=3.0x ATR=12.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 09:50:00 | 3974.41 | 3945.16 | 0.00 | T1 1.5R @ 3974.41 |
| Stop hit — per-position SL triggered | 2025-09-10 10:00:00 | 3955.00 | 3947.68 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:40:00 | 4015.60 | 3993.73 | 0.00 | ORB-long ORB[3953.30,4002.80] vol=1.6x ATR=12.49 |
| Stop hit — per-position SL triggered | 2025-09-15 09:50:00 | 4003.11 | 3995.65 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:50:00 | 4070.00 | 4043.27 | 0.00 | ORB-long ORB[3992.80,4045.00] vol=3.5x ATR=15.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 10:45:00 | 4093.93 | 4063.65 | 0.00 | T1 1.5R @ 4093.93 |
| Stop hit — per-position SL triggered | 2025-09-16 11:00:00 | 4070.00 | 4064.70 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:20:00 | 4135.20 | 4101.93 | 0.00 | ORB-long ORB[4057.20,4087.90] vol=4.5x ATR=13.13 |
| Stop hit — per-position SL triggered | 2025-09-17 10:30:00 | 4122.07 | 4107.29 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:30:00 | 4162.30 | 4142.21 | 0.00 | ORB-long ORB[4116.20,4149.70] vol=5.6x ATR=10.76 |
| Stop hit — per-position SL triggered | 2025-09-19 10:55:00 | 4151.54 | 4145.63 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 10:45:00 | 3973.70 | 3981.21 | 0.00 | ORB-short ORB[3980.60,4017.30] vol=1.9x ATR=13.54 |
| Stop hit — per-position SL triggered | 2025-09-29 12:30:00 | 3987.24 | 3975.37 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:20:00 | 4173.90 | 4164.84 | 0.00 | ORB-long ORB[4127.10,4166.80] vol=4.4x ATR=20.55 |
| Stop hit — per-position SL triggered | 2025-10-01 15:20:00 | 4173.90 | 4171.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2025-10-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 11:00:00 | 4218.20 | 4185.98 | 0.00 | ORB-long ORB[4142.20,4190.00] vol=5.1x ATR=14.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 11:10:00 | 4239.93 | 4202.29 | 0.00 | T1 1.5R @ 4239.93 |
| Stop hit — per-position SL triggered | 2025-10-03 12:05:00 | 4218.20 | 4225.77 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 09:40:00 | 4208.40 | 4213.49 | 0.00 | ORB-short ORB[4213.70,4275.00] vol=2.3x ATR=21.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 12:00:00 | 4175.59 | 4204.75 | 0.00 | T1 1.5R @ 4175.59 |
| Stop hit — per-position SL triggered | 2025-10-06 13:10:00 | 4208.40 | 4203.30 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:40:00 | 4264.70 | 4271.06 | 0.00 | ORB-short ORB[4266.40,4299.90] vol=1.9x ATR=17.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:40:00 | 4238.76 | 4262.71 | 0.00 | T1 1.5R @ 4238.76 |
| Target hit | 2025-10-08 15:20:00 | 4222.70 | 4237.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-10-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:50:00 | 4202.10 | 4222.20 | 0.00 | ORB-short ORB[4219.50,4246.30] vol=4.1x ATR=17.95 |
| Stop hit — per-position SL triggered | 2025-10-09 10:25:00 | 4220.05 | 4214.50 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 11:15:00 | 4150.30 | 4185.51 | 0.00 | ORB-short ORB[4177.00,4210.00] vol=1.8x ATR=12.41 |
| Stop hit — per-position SL triggered | 2025-10-10 11:40:00 | 4162.71 | 4183.13 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 11:00:00 | 4070.50 | 4087.72 | 0.00 | ORB-short ORB[4078.70,4127.20] vol=2.0x ATR=12.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 11:20:00 | 4052.11 | 4083.77 | 0.00 | T1 1.5R @ 4052.11 |
| Target hit | 2025-10-13 15:20:00 | 4020.50 | 4045.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2025-10-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:55:00 | 3975.00 | 3999.28 | 0.00 | ORB-short ORB[3991.50,4039.60] vol=1.5x ATR=11.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:30:00 | 3957.80 | 3992.11 | 0.00 | T1 1.5R @ 3957.80 |
| Target hit | 2025-10-14 12:10:00 | 3954.70 | 3952.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — SELL (started 2025-10-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:40:00 | 3864.70 | 3888.63 | 0.00 | ORB-short ORB[3880.00,3924.00] vol=1.6x ATR=11.86 |
| Stop hit — per-position SL triggered | 2025-10-20 11:35:00 | 3876.56 | 3877.72 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:50:00 | 3896.50 | 3905.89 | 0.00 | ORB-short ORB[3899.00,3932.50] vol=2.9x ATR=8.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:00:00 | 3884.16 | 3903.67 | 0.00 | T1 1.5R @ 3884.16 |
| Stop hit — per-position SL triggered | 2025-10-24 13:45:00 | 3896.50 | 3898.58 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 10:40:00 | 3902.00 | 3923.18 | 0.00 | ORB-short ORB[3912.70,3957.80] vol=1.8x ATR=8.46 |
| Stop hit — per-position SL triggered | 2025-10-27 12:00:00 | 3910.46 | 3917.92 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 09:35:00 | 3918.20 | 3929.84 | 0.00 | ORB-short ORB[3919.40,3948.20] vol=2.4x ATR=10.09 |
| Stop hit — per-position SL triggered | 2025-10-28 10:30:00 | 3928.29 | 3925.73 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:40:00 | 3901.00 | 3920.66 | 0.00 | ORB-short ORB[3923.00,3950.00] vol=2.8x ATR=10.05 |
| Stop hit — per-position SL triggered | 2025-10-30 11:00:00 | 3911.05 | 3911.53 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:40:00 | 3972.70 | 3953.78 | 0.00 | ORB-long ORB[3924.40,3955.00] vol=1.8x ATR=10.51 |
| Stop hit — per-position SL triggered | 2025-10-31 09:45:00 | 3962.19 | 3954.24 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 10:50:00 | 4168.80 | 4158.60 | 0.00 | ORB-long ORB[4120.10,4168.70] vol=5.2x ATR=16.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:40:00 | 4194.20 | 4160.61 | 0.00 | T1 1.5R @ 4194.20 |
| Stop hit — per-position SL triggered | 2025-11-06 13:00:00 | 4168.80 | 4162.13 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 11:00:00 | 4138.50 | 4112.73 | 0.00 | ORB-long ORB[4086.10,4111.90] vol=5.4x ATR=12.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:05:00 | 4157.23 | 4120.78 | 0.00 | T1 1.5R @ 4157.23 |
| Stop hit — per-position SL triggered | 2025-11-10 11:20:00 | 4138.50 | 4123.03 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:55:00 | 4125.40 | 4107.54 | 0.00 | ORB-long ORB[4076.00,4121.50] vol=1.9x ATR=12.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:00:00 | 4143.46 | 4115.86 | 0.00 | T1 1.5R @ 4143.46 |
| Target hit | 2025-11-12 12:30:00 | 4171.70 | 4174.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2025-11-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:25:00 | 4064.90 | 4077.93 | 0.00 | ORB-short ORB[4081.10,4131.70] vol=2.3x ATR=12.60 |
| Stop hit — per-position SL triggered | 2025-11-14 11:10:00 | 4077.50 | 4076.04 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 09:35:00 | 4174.60 | 4157.07 | 0.00 | ORB-long ORB[4100.20,4142.30] vol=2.6x ATR=16.12 |
| Stop hit — per-position SL triggered | 2025-11-19 10:05:00 | 4158.48 | 4169.86 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 11:15:00 | 4061.20 | 4071.55 | 0.00 | ORB-short ORB[4069.00,4109.00] vol=2.3x ATR=7.47 |
| Stop hit — per-position SL triggered | 2025-11-20 11:25:00 | 4068.67 | 4071.31 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:45:00 | 3936.50 | 3943.78 | 0.00 | ORB-short ORB[3945.00,3991.90] vol=2.3x ATR=11.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 15:05:00 | 3918.65 | 3936.11 | 0.00 | T1 1.5R @ 3918.65 |
| Target hit | 2025-11-25 15:20:00 | 3902.80 | 3929.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2025-11-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:10:00 | 3866.00 | 3877.67 | 0.00 | ORB-short ORB[3878.30,3919.00] vol=6.7x ATR=7.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 12:15:00 | 3855.35 | 3868.14 | 0.00 | T1 1.5R @ 3855.35 |
| Stop hit — per-position SL triggered | 2025-11-27 14:55:00 | 3866.00 | 3856.75 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:50:00 | 3894.60 | 3903.56 | 0.00 | ORB-short ORB[3895.00,3936.00] vol=2.4x ATR=12.04 |
| Stop hit — per-position SL triggered | 2025-12-01 10:55:00 | 3906.64 | 3904.60 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 11:05:00 | 3944.20 | 3916.60 | 0.00 | ORB-long ORB[3861.50,3916.50] vol=2.6x ATR=9.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 11:15:00 | 3958.75 | 3923.50 | 0.00 | T1 1.5R @ 3958.75 |
| Stop hit — per-position SL triggered | 2025-12-02 12:00:00 | 3944.20 | 3934.70 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 11:05:00 | 3910.00 | 3893.35 | 0.00 | ORB-long ORB[3879.90,3900.00] vol=1.5x ATR=10.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 11:15:00 | 3925.18 | 3897.28 | 0.00 | T1 1.5R @ 3925.18 |
| Stop hit — per-position SL triggered | 2025-12-05 11:55:00 | 3910.00 | 3901.96 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:30:00 | 3850.00 | 3880.68 | 0.00 | ORB-short ORB[3890.00,3930.00] vol=2.4x ATR=14.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:40:00 | 3828.76 | 3874.43 | 0.00 | T1 1.5R @ 3828.76 |
| Target hit | 2025-12-08 14:50:00 | 3810.70 | 3806.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 59 — SELL (started 2025-12-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:35:00 | 3720.20 | 3740.05 | 0.00 | ORB-short ORB[3764.10,3787.80] vol=3.2x ATR=12.40 |
| Stop hit — per-position SL triggered | 2025-12-18 10:20:00 | 3732.60 | 3731.14 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:45:00 | 3813.00 | 3807.85 | 0.00 | ORB-long ORB[3778.00,3800.50] vol=3.4x ATR=12.08 |
| Stop hit — per-position SL triggered | 2025-12-19 11:05:00 | 3800.92 | 3807.57 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:35:00 | 3796.10 | 3782.70 | 0.00 | ORB-long ORB[3757.50,3790.80] vol=1.6x ATR=9.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:25:00 | 3811.02 | 3792.26 | 0.00 | T1 1.5R @ 3811.02 |
| Stop hit — per-position SL triggered | 2025-12-26 11:50:00 | 3796.10 | 3793.55 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:20:00 | 3834.00 | 3820.74 | 0.00 | ORB-long ORB[3796.40,3823.60] vol=2.0x ATR=10.70 |
| Stop hit — per-position SL triggered | 2025-12-30 10:55:00 | 3823.30 | 3824.43 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:05:00 | 3775.30 | 3787.70 | 0.00 | ORB-short ORB[3782.10,3821.60] vol=4.9x ATR=11.33 |
| Stop hit — per-position SL triggered | 2026-01-06 11:40:00 | 3786.63 | 3787.02 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 11:00:00 | 3800.00 | 3785.56 | 0.00 | ORB-long ORB[3762.30,3795.00] vol=3.1x ATR=9.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 13:05:00 | 3814.69 | 3803.48 | 0.00 | T1 1.5R @ 3814.69 |
| Target hit | 2026-01-14 13:30:00 | 3805.00 | 3805.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 65 — SELL (started 2026-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:50:00 | 3683.00 | 3708.39 | 0.00 | ORB-short ORB[3711.70,3754.30] vol=1.8x ATR=10.35 |
| Stop hit — per-position SL triggered | 2026-01-20 10:55:00 | 3693.35 | 3708.02 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-01-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 10:55:00 | 3544.50 | 3576.66 | 0.00 | ORB-short ORB[3551.00,3593.50] vol=2.6x ATR=11.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:05:00 | 3527.57 | 3569.14 | 0.00 | T1 1.5R @ 3527.57 |
| Stop hit — per-position SL triggered | 2026-01-22 11:25:00 | 3544.50 | 3563.88 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 10:40:00 | 3592.70 | 3568.40 | 0.00 | ORB-long ORB[3524.90,3556.50] vol=1.5x ATR=10.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:10:00 | 3609.09 | 3580.11 | 0.00 | T1 1.5R @ 3609.09 |
| Stop hit — per-position SL triggered | 2026-01-23 11:20:00 | 3592.70 | 3580.79 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 11:10:00 | 3573.10 | 3598.03 | 0.00 | ORB-short ORB[3590.50,3635.00] vol=4.3x ATR=8.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 11:15:00 | 3560.29 | 3594.80 | 0.00 | T1 1.5R @ 3560.29 |
| Stop hit — per-position SL triggered | 2026-01-29 12:10:00 | 3573.10 | 3584.48 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-02-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 10:05:00 | 3599.70 | 3571.13 | 0.00 | ORB-long ORB[3550.00,3583.00] vol=1.6x ATR=13.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:55:00 | 3620.39 | 3588.88 | 0.00 | T1 1.5R @ 3620.39 |
| Target hit | 2026-02-02 15:20:00 | 3690.00 | 3644.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 3944.00 | 3896.66 | 0.00 | ORB-long ORB[3833.50,3888.60] vol=2.1x ATR=14.42 |
| Stop hit — per-position SL triggered | 2026-02-09 12:00:00 | 3929.58 | 3917.56 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 3911.80 | 3921.60 | 0.00 | ORB-short ORB[3912.70,3944.20] vol=2.8x ATR=10.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:20:00 | 3896.07 | 3914.36 | 0.00 | T1 1.5R @ 3896.07 |
| Target hit | 2026-02-11 15:20:00 | 3816.20 | 3841.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 3812.30 | 3795.64 | 0.00 | ORB-long ORB[3752.00,3785.00] vol=2.6x ATR=10.65 |
| Stop hit — per-position SL triggered | 2026-02-17 11:10:00 | 3801.65 | 3796.14 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:10:00 | 3752.60 | 3770.08 | 0.00 | ORB-short ORB[3754.40,3799.00] vol=2.8x ATR=7.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 12:05:00 | 3740.97 | 3764.56 | 0.00 | T1 1.5R @ 3740.97 |
| Stop hit — per-position SL triggered | 2026-02-18 13:05:00 | 3752.60 | 3759.50 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:00:00 | 3877.20 | 3849.66 | 0.00 | ORB-long ORB[3803.90,3845.70] vol=4.5x ATR=11.76 |
| Stop hit — per-position SL triggered | 2026-02-20 11:10:00 | 3865.44 | 3850.30 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-02-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:55:00 | 4004.00 | 3978.27 | 0.00 | ORB-long ORB[3947.40,3999.90] vol=2.3x ATR=15.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:10:00 | 4027.71 | 3988.68 | 0.00 | T1 1.5R @ 4027.71 |
| Target hit | 2026-02-24 15:20:00 | 4062.20 | 4054.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — BUY (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 4220.10 | 4186.64 | 0.00 | ORB-long ORB[4158.00,4200.80] vol=3.1x ATR=13.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:10:00 | 4239.81 | 4199.19 | 0.00 | T1 1.5R @ 4239.81 |
| Stop hit — per-position SL triggered | 2026-03-05 11:45:00 | 4220.10 | 4206.48 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-03-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:35:00 | 4363.70 | 4324.23 | 0.00 | ORB-long ORB[4267.70,4317.00] vol=2.3x ATR=14.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:45:00 | 4385.16 | 4338.20 | 0.00 | T1 1.5R @ 4385.16 |
| Target hit | 2026-03-06 14:15:00 | 4375.00 | 4395.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 78 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:15:00 | 4156.10 | 4166.25 | 0.00 | ORB-short ORB[4163.10,4202.80] vol=4.0x ATR=11.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:35:00 | 4138.42 | 4164.56 | 0.00 | T1 1.5R @ 4138.42 |
| Target hit | 2026-03-11 15:20:00 | 4043.20 | 4102.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2026-03-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:45:00 | 3846.90 | 3820.84 | 0.00 | ORB-long ORB[3777.20,3834.60] vol=2.2x ATR=21.75 |
| Stop hit — per-position SL triggered | 2026-03-16 10:10:00 | 3825.15 | 3821.82 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-03-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:30:00 | 3820.00 | 3806.10 | 0.00 | ORB-long ORB[3764.70,3802.00] vol=1.9x ATR=10.78 |
| Stop hit — per-position SL triggered | 2026-03-17 10:40:00 | 3809.22 | 3806.79 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 3956.30 | 3937.87 | 0.00 | ORB-long ORB[3901.00,3944.40] vol=5.0x ATR=15.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:30:00 | 3979.29 | 3954.49 | 0.00 | T1 1.5R @ 3979.29 |
| Stop hit — per-position SL triggered | 2026-04-15 11:00:00 | 3956.30 | 3973.44 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:50:00 | 4327.00 | 4305.89 | 0.00 | ORB-long ORB[4252.00,4295.00] vol=2.0x ATR=17.88 |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 4309.12 | 4308.76 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 4270.00 | 4242.54 | 0.00 | ORB-long ORB[4225.00,4256.10] vol=1.8x ATR=15.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:55:00 | 4292.69 | 4257.71 | 0.00 | T1 1.5R @ 4292.69 |
| Stop hit — per-position SL triggered | 2026-04-24 10:20:00 | 4270.00 | 4289.36 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-04-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:30:00 | 4300.00 | 4310.33 | 0.00 | ORB-short ORB[4305.00,4350.00] vol=1.8x ATR=14.18 |
| Stop hit — per-position SL triggered | 2026-04-27 10:35:00 | 4314.18 | 4310.34 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-04-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:30:00 | 4339.00 | 4355.18 | 0.00 | ORB-short ORB[4350.60,4388.50] vol=2.1x ATR=19.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 4309.71 | 4347.60 | 0.00 | T1 1.5R @ 4309.71 |
| Target hit | 2026-04-28 15:20:00 | 4268.00 | 4311.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 4151.70 | 4158.50 | 0.00 | ORB-short ORB[4155.00,4182.60] vol=3.5x ATR=11.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 12:20:00 | 4135.20 | 4154.43 | 0.00 | T1 1.5R @ 4135.20 |
| Stop hit — per-position SL triggered | 2026-05-06 13:20:00 | 4151.70 | 4148.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 09:35:00 | 3808.70 | 2025-05-13 09:45:00 | 3792.10 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-05-27 09:35:00 | 4043.10 | 2025-05-27 09:45:00 | 4026.06 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-05-28 10:35:00 | 3979.20 | 2025-05-28 10:40:00 | 3991.23 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-02 10:35:00 | 4160.90 | 2025-06-02 11:30:00 | 4133.10 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-06-02 10:35:00 | 4160.90 | 2025-06-02 15:20:00 | 4128.90 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2025-06-18 11:15:00 | 3982.60 | 2025-06-18 11:50:00 | 3992.05 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-19 09:50:00 | 3947.60 | 2025-06-19 10:05:00 | 3935.92 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-06-19 09:50:00 | 3947.60 | 2025-06-19 15:20:00 | 3860.00 | TARGET_HIT | 0.50 | 2.22% |
| BUY | retest1 | 2025-06-20 10:40:00 | 3903.80 | 2025-06-20 10:45:00 | 3890.29 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-23 10:50:00 | 3912.80 | 2025-06-23 12:35:00 | 3901.03 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-26 11:15:00 | 3878.50 | 2025-06-26 11:35:00 | 3865.06 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-06-26 11:15:00 | 3878.50 | 2025-06-26 12:40:00 | 3878.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-30 09:45:00 | 3984.20 | 2025-06-30 10:05:00 | 3964.03 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-07-04 10:10:00 | 3956.90 | 2025-07-04 10:20:00 | 3969.65 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-10 09:45:00 | 4294.80 | 2025-07-10 09:50:00 | 4277.61 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-07-15 10:50:00 | 4281.80 | 2025-07-15 11:00:00 | 4266.67 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-16 10:05:00 | 4164.10 | 2025-07-16 10:20:00 | 4174.15 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-17 10:30:00 | 4179.10 | 2025-07-17 10:35:00 | 4188.87 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-18 10:50:00 | 4160.00 | 2025-07-18 11:00:00 | 4166.98 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-07-21 10:20:00 | 4241.20 | 2025-07-21 10:25:00 | 4229.32 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-31 10:55:00 | 4079.30 | 2025-07-31 11:05:00 | 4099.46 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-07-31 10:55:00 | 4079.30 | 2025-07-31 15:20:00 | 4126.70 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2025-08-01 10:50:00 | 4075.10 | 2025-08-01 11:00:00 | 4091.17 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-08-04 11:15:00 | 4045.50 | 2025-08-04 12:10:00 | 4023.47 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-08-04 11:15:00 | 4045.50 | 2025-08-04 15:05:00 | 4045.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-05 11:15:00 | 4120.80 | 2025-08-05 11:20:00 | 4145.12 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-08-05 11:15:00 | 4120.80 | 2025-08-05 14:00:00 | 4120.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 11:10:00 | 4004.20 | 2025-08-07 11:15:00 | 3988.70 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-08-07 11:10:00 | 4004.20 | 2025-08-07 12:40:00 | 4004.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-18 10:50:00 | 3900.80 | 2025-08-18 10:55:00 | 3918.80 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-08-18 10:50:00 | 3900.80 | 2025-08-18 11:30:00 | 3900.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 10:20:00 | 3963.80 | 2025-08-22 11:45:00 | 3946.29 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-08-22 10:20:00 | 3963.80 | 2025-08-22 15:20:00 | 3941.70 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2025-09-02 10:05:00 | 3924.40 | 2025-09-02 10:35:00 | 3911.29 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-08 10:25:00 | 3852.00 | 2025-09-08 10:30:00 | 3840.51 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-10 09:35:00 | 3955.00 | 2025-09-10 09:50:00 | 3974.41 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-09-10 09:35:00 | 3955.00 | 2025-09-10 10:00:00 | 3955.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-15 09:40:00 | 4015.60 | 2025-09-15 09:50:00 | 4003.11 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-16 09:50:00 | 4070.00 | 2025-09-16 10:45:00 | 4093.93 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-09-16 09:50:00 | 4070.00 | 2025-09-16 11:00:00 | 4070.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 10:20:00 | 4135.20 | 2025-09-17 10:30:00 | 4122.07 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-19 10:30:00 | 4162.30 | 2025-09-19 10:55:00 | 4151.54 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-29 10:45:00 | 3973.70 | 2025-09-29 12:30:00 | 3987.24 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-01 10:20:00 | 4173.90 | 2025-10-01 15:20:00 | 4173.90 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest1 | 2025-10-03 11:00:00 | 4218.20 | 2025-10-03 11:10:00 | 4239.93 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-10-03 11:00:00 | 4218.20 | 2025-10-03 12:05:00 | 4218.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-06 09:40:00 | 4208.40 | 2025-10-06 12:00:00 | 4175.59 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2025-10-06 09:40:00 | 4208.40 | 2025-10-06 13:10:00 | 4208.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 09:40:00 | 4264.70 | 2025-10-08 10:40:00 | 4238.76 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-10-08 09:40:00 | 4264.70 | 2025-10-08 15:20:00 | 4222.70 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2025-10-09 09:50:00 | 4202.10 | 2025-10-09 10:25:00 | 4220.05 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-10-10 11:15:00 | 4150.30 | 2025-10-10 11:40:00 | 4162.71 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-13 11:00:00 | 4070.50 | 2025-10-13 11:20:00 | 4052.11 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-10-13 11:00:00 | 4070.50 | 2025-10-13 15:20:00 | 4020.50 | TARGET_HIT | 0.50 | 1.23% |
| SELL | retest1 | 2025-10-14 09:55:00 | 3975.00 | 2025-10-14 10:30:00 | 3957.80 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-14 09:55:00 | 3975.00 | 2025-10-14 12:10:00 | 3954.70 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2025-10-20 10:40:00 | 3864.70 | 2025-10-20 11:35:00 | 3876.56 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-10-24 10:50:00 | 3896.50 | 2025-10-24 11:00:00 | 3884.16 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-10-24 10:50:00 | 3896.50 | 2025-10-24 13:45:00 | 3896.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-27 10:40:00 | 3902.00 | 2025-10-27 12:00:00 | 3910.46 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-28 09:35:00 | 3918.20 | 2025-10-28 10:30:00 | 3928.29 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-30 10:40:00 | 3901.00 | 2025-10-30 11:00:00 | 3911.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-31 09:40:00 | 3972.70 | 2025-10-31 09:45:00 | 3962.19 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-06 10:50:00 | 4168.80 | 2025-11-06 11:40:00 | 4194.20 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-11-06 10:50:00 | 4168.80 | 2025-11-06 13:00:00 | 4168.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-10 11:00:00 | 4138.50 | 2025-11-10 11:05:00 | 4157.23 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-11-10 11:00:00 | 4138.50 | 2025-11-10 11:20:00 | 4138.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 09:55:00 | 4125.40 | 2025-11-12 10:00:00 | 4143.46 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-11-12 09:55:00 | 4125.40 | 2025-11-12 12:30:00 | 4171.70 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2025-11-14 10:25:00 | 4064.90 | 2025-11-14 11:10:00 | 4077.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-19 09:35:00 | 4174.60 | 2025-11-19 10:05:00 | 4158.48 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-11-20 11:15:00 | 4061.20 | 2025-11-20 11:25:00 | 4068.67 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-11-25 10:45:00 | 3936.50 | 2025-11-25 15:05:00 | 3918.65 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-11-25 10:45:00 | 3936.50 | 2025-11-25 15:20:00 | 3902.80 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2025-11-27 11:10:00 | 3866.00 | 2025-11-27 12:15:00 | 3855.35 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-11-27 11:10:00 | 3866.00 | 2025-11-27 14:55:00 | 3866.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-01 10:50:00 | 3894.60 | 2025-12-01 10:55:00 | 3906.64 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-02 11:05:00 | 3944.20 | 2025-12-02 11:15:00 | 3958.75 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-02 11:05:00 | 3944.20 | 2025-12-02 12:00:00 | 3944.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 11:05:00 | 3910.00 | 2025-12-05 11:15:00 | 3925.18 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-12-05 11:05:00 | 3910.00 | 2025-12-05 11:55:00 | 3910.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 10:30:00 | 3850.00 | 2025-12-08 10:40:00 | 3828.76 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-12-08 10:30:00 | 3850.00 | 2025-12-08 14:50:00 | 3810.70 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2025-12-18 09:35:00 | 3720.20 | 2025-12-18 10:20:00 | 3732.60 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-19 10:45:00 | 3813.00 | 2025-12-19 11:05:00 | 3800.92 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-12-26 09:35:00 | 3796.10 | 2025-12-26 11:25:00 | 3811.02 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-12-26 09:35:00 | 3796.10 | 2025-12-26 11:50:00 | 3796.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-30 10:20:00 | 3834.00 | 2025-12-30 10:55:00 | 3823.30 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-06 11:05:00 | 3775.30 | 2026-01-06 11:40:00 | 3786.63 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-14 11:00:00 | 3800.00 | 2026-01-14 13:05:00 | 3814.69 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-01-14 11:00:00 | 3800.00 | 2026-01-14 13:30:00 | 3805.00 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2026-01-20 10:50:00 | 3683.00 | 2026-01-20 10:55:00 | 3693.35 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-22 10:55:00 | 3544.50 | 2026-01-22 11:05:00 | 3527.57 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-01-22 10:55:00 | 3544.50 | 2026-01-22 11:25:00 | 3544.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-23 10:40:00 | 3592.70 | 2026-01-23 11:10:00 | 3609.09 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-01-23 10:40:00 | 3592.70 | 2026-01-23 11:20:00 | 3592.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 11:10:00 | 3573.10 | 2026-01-29 11:15:00 | 3560.29 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-29 11:10:00 | 3573.10 | 2026-01-29 12:10:00 | 3573.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-02 10:05:00 | 3599.70 | 2026-02-02 10:55:00 | 3620.39 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-02-02 10:05:00 | 3599.70 | 2026-02-02 15:20:00 | 3690.00 | TARGET_HIT | 0.50 | 2.51% |
| BUY | retest1 | 2026-02-09 10:35:00 | 3944.00 | 2026-02-09 12:00:00 | 3929.58 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-11 10:55:00 | 3911.80 | 2026-02-11 11:20:00 | 3896.07 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-11 10:55:00 | 3911.80 | 2026-02-11 15:20:00 | 3816.20 | TARGET_HIT | 0.50 | 2.44% |
| BUY | retest1 | 2026-02-17 11:00:00 | 3812.30 | 2026-02-17 11:10:00 | 3801.65 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-18 11:10:00 | 3752.60 | 2026-02-18 12:05:00 | 3740.97 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-18 11:10:00 | 3752.60 | 2026-02-18 13:05:00 | 3752.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 11:00:00 | 3877.20 | 2026-02-20 11:10:00 | 3865.44 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-24 10:55:00 | 4004.00 | 2026-02-24 11:10:00 | 4027.71 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-02-24 10:55:00 | 4004.00 | 2026-02-24 15:20:00 | 4062.20 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2026-03-05 10:55:00 | 4220.10 | 2026-03-05 11:10:00 | 4239.81 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-05 10:55:00 | 4220.10 | 2026-03-05 11:45:00 | 4220.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 10:35:00 | 4363.70 | 2026-03-06 10:45:00 | 4385.16 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-03-06 10:35:00 | 4363.70 | 2026-03-06 14:15:00 | 4375.00 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2026-03-11 10:15:00 | 4156.10 | 2026-03-11 10:35:00 | 4138.42 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-03-11 10:15:00 | 4156.10 | 2026-03-11 15:20:00 | 4043.20 | TARGET_HIT | 0.50 | 2.72% |
| BUY | retest1 | 2026-03-16 09:45:00 | 3846.90 | 2026-03-16 10:10:00 | 3825.15 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-03-17 10:30:00 | 3820.00 | 2026-03-17 10:40:00 | 3809.22 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-15 09:35:00 | 3956.30 | 2026-04-15 10:30:00 | 3979.29 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-15 09:35:00 | 3956.30 | 2026-04-15 11:00:00 | 3956.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 09:50:00 | 4327.00 | 2026-04-23 10:15:00 | 4309.12 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-24 09:45:00 | 4270.00 | 2026-04-24 09:55:00 | 4292.69 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-24 09:45:00 | 4270.00 | 2026-04-24 10:20:00 | 4270.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-27 10:30:00 | 4300.00 | 2026-04-27 10:35:00 | 4314.18 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-28 10:30:00 | 4339.00 | 2026-04-28 11:25:00 | 4309.71 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-04-28 10:30:00 | 4339.00 | 2026-04-28 15:20:00 | 4268.00 | TARGET_HIT | 0.50 | 1.64% |
| SELL | retest1 | 2026-05-06 11:10:00 | 4151.70 | 2026-05-06 12:20:00 | 4135.20 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-05-06 11:10:00 | 4151.70 | 2026-05-06 13:20:00 | 4151.70 | STOP_HIT | 0.50 | 0.00% |
