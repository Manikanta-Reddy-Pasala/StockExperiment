# Schaeffler India Ltd. (SCHAEFFLER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 12
- **Target hits / Stop hits / Partials:** 5 / 12 / 10
- **Avg / median % per leg:** 0.39% / 0.31%
- **Sum % (uncompounded):** 10.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.12% | 1.9% |
| BUY @ 2nd Alert (retest1) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.12% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 8 | 72.7% | 3 | 3 | 5 | 0.79% | 8.7% |
| SELL @ 2nd Alert (retest1) | 11 | 8 | 72.7% | 3 | 3 | 5 | 0.79% | 8.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 15 | 55.6% | 5 | 12 | 10 | 0.39% | 10.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 3944.00 | 3896.66 | 0.00 | ORB-long ORB[3833.50,3888.60] vol=3.4x ATR=26.10 |
| Stop hit — per-position SL triggered | 2026-02-09 12:00:00 | 3917.90 | 3917.56 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 3911.80 | 3921.60 | 0.00 | ORB-short ORB[3912.70,3944.20] vol=2.8x ATR=10.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:20:00 | 3896.07 | 3914.36 | 0.00 | T1 1.5R @ 3896.07 |
| Target hit | 2026-02-11 15:20:00 | 3816.20 | 3841.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 3812.30 | 3795.64 | 0.00 | ORB-long ORB[3752.00,3785.00] vol=2.6x ATR=10.65 |
| Stop hit — per-position SL triggered | 2026-02-17 11:10:00 | 3801.65 | 3796.14 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:10:00 | 3752.60 | 3770.08 | 0.00 | ORB-short ORB[3754.40,3799.00] vol=2.8x ATR=7.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 12:05:00 | 3740.97 | 3764.56 | 0.00 | T1 1.5R @ 3740.97 |
| Stop hit — per-position SL triggered | 2026-02-18 13:05:00 | 3752.60 | 3759.50 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:00:00 | 3877.20 | 3849.66 | 0.00 | ORB-long ORB[3803.90,3845.70] vol=4.5x ATR=11.76 |
| Stop hit — per-position SL triggered | 2026-02-20 11:10:00 | 3865.44 | 3850.30 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:55:00 | 4004.00 | 3978.27 | 0.00 | ORB-long ORB[3947.40,3999.90] vol=2.3x ATR=15.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:10:00 | 4027.71 | 3988.68 | 0.00 | T1 1.5R @ 4027.71 |
| Target hit | 2026-02-24 15:20:00 | 4062.20 | 4054.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 4220.10 | 4186.64 | 0.00 | ORB-long ORB[4158.00,4200.80] vol=3.1x ATR=13.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:10:00 | 4239.81 | 4199.19 | 0.00 | T1 1.5R @ 4239.81 |
| Stop hit — per-position SL triggered | 2026-03-05 11:45:00 | 4220.10 | 4206.48 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:35:00 | 4363.70 | 4324.23 | 0.00 | ORB-long ORB[4267.70,4317.00] vol=2.3x ATR=14.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:45:00 | 4385.16 | 4338.20 | 0.00 | T1 1.5R @ 4385.16 |
| Target hit | 2026-03-06 14:15:00 | 4375.00 | 4395.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:15:00 | 4156.10 | 4166.25 | 0.00 | ORB-short ORB[4163.10,4202.80] vol=4.0x ATR=11.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:35:00 | 4138.42 | 4164.56 | 0.00 | T1 1.5R @ 4138.42 |
| Target hit | 2026-03-11 15:20:00 | 4043.20 | 4102.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-03-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:45:00 | 3846.90 | 3820.84 | 0.00 | ORB-long ORB[3777.20,3834.60] vol=2.2x ATR=21.75 |
| Stop hit — per-position SL triggered | 2026-03-16 10:10:00 | 3825.15 | 3821.82 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:30:00 | 3820.00 | 3806.10 | 0.00 | ORB-long ORB[3764.70,3802.00] vol=1.9x ATR=10.78 |
| Stop hit — per-position SL triggered | 2026-03-17 10:40:00 | 3809.22 | 3806.79 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 3956.30 | 3937.87 | 0.00 | ORB-long ORB[3901.00,3944.40] vol=5.0x ATR=15.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:30:00 | 3979.29 | 3954.49 | 0.00 | T1 1.5R @ 3979.29 |
| Stop hit — per-position SL triggered | 2026-04-15 11:00:00 | 3956.30 | 3973.44 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:50:00 | 4327.00 | 4305.89 | 0.00 | ORB-long ORB[4252.00,4295.00] vol=2.0x ATR=17.88 |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 4309.12 | 4308.76 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 4270.00 | 4242.54 | 0.00 | ORB-long ORB[4225.00,4256.10] vol=1.8x ATR=15.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:55:00 | 4292.69 | 4257.71 | 0.00 | T1 1.5R @ 4292.69 |
| Stop hit — per-position SL triggered | 2026-04-24 10:20:00 | 4270.00 | 4289.36 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:30:00 | 4300.00 | 4310.33 | 0.00 | ORB-short ORB[4305.00,4350.00] vol=1.8x ATR=14.18 |
| Stop hit — per-position SL triggered | 2026-04-27 10:35:00 | 4314.18 | 4310.34 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:30:00 | 4339.00 | 4355.18 | 0.00 | ORB-short ORB[4350.60,4388.50] vol=2.1x ATR=19.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 4309.71 | 4347.60 | 0.00 | T1 1.5R @ 4309.71 |
| Target hit | 2026-04-28 15:20:00 | 4268.00 | 4311.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2026-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:10:00 | 4151.70 | 4158.50 | 0.00 | ORB-short ORB[4155.00,4182.60] vol=3.5x ATR=11.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 12:20:00 | 4135.20 | 4154.43 | 0.00 | T1 1.5R @ 4135.20 |
| Stop hit — per-position SL triggered | 2026-05-06 13:20:00 | 4151.70 | 4148.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 3944.00 | 2026-02-09 12:00:00 | 3917.90 | STOP_HIT | 1.00 | -0.66% |
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
