# Torrent Pharmaceuticals Ltd. (TORNTPHARM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4385.00
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
| ENTRY1 | 20 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 16
- **Target hits / Stop hits / Partials:** 4 / 16 / 7
- **Avg / median % per leg:** 0.03% / 0.00%
- **Sum % (uncompounded):** 0.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 7 | 41.2% | 3 | 10 | 4 | 0.04% | 0.6% |
| BUY @ 2nd Alert (retest1) | 17 | 7 | 41.2% | 3 | 10 | 4 | 0.04% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.03% | 0.3% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.03% | 0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 11 | 40.7% | 4 | 16 | 7 | 0.03% | 0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 3993.60 | 3969.84 | 0.00 | ORB-long ORB[3936.20,3971.50] vol=1.7x ATR=16.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:00:00 | 4018.14 | 3998.92 | 0.00 | T1 1.5R @ 4018.14 |
| Target hit | 2026-02-09 11:45:00 | 3999.20 | 4002.12 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 4067.40 | 4040.25 | 0.00 | ORB-long ORB[4008.20,4052.70] vol=2.2x ATR=7.58 |
| Stop hit — per-position SL triggered | 2026-02-10 11:10:00 | 4059.82 | 4041.90 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:45:00 | 4111.10 | 4092.59 | 0.00 | ORB-long ORB[4067.20,4091.50] vol=2.6x ATR=12.45 |
| Stop hit — per-position SL triggered | 2026-02-13 09:55:00 | 4098.65 | 4094.53 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 4462.30 | 4424.15 | 0.00 | ORB-long ORB[4392.50,4420.00] vol=2.1x ATR=13.62 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 4448.68 | 4436.84 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 4365.60 | 4383.98 | 0.00 | ORB-short ORB[4383.30,4425.00] vol=9.9x ATR=8.69 |
| Stop hit — per-position SL triggered | 2026-02-27 11:10:00 | 4374.29 | 4382.89 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 4386.90 | 4357.28 | 0.00 | ORB-long ORB[4328.90,4362.50] vol=2.7x ATR=13.39 |
| Stop hit — per-position SL triggered | 2026-03-05 09:35:00 | 4373.51 | 4360.79 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:35:00 | 4342.00 | 4357.89 | 0.00 | ORB-short ORB[4342.10,4380.30] vol=1.6x ATR=11.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:25:00 | 4325.35 | 4344.41 | 0.00 | T1 1.5R @ 4325.35 |
| Target hit | 2026-03-06 12:05:00 | 4339.50 | 4325.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 4407.00 | 4439.68 | 0.00 | ORB-short ORB[4422.10,4468.60] vol=2.4x ATR=13.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:10:00 | 4386.91 | 4431.74 | 0.00 | T1 1.5R @ 4386.91 |
| Stop hit — per-position SL triggered | 2026-03-13 11:45:00 | 4407.00 | 4425.10 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:15:00 | 4290.90 | 4271.28 | 0.00 | ORB-long ORB[4242.70,4280.00] vol=2.4x ATR=13.47 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 4277.43 | 4278.55 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:15:00 | 4254.70 | 4285.12 | 0.00 | ORB-short ORB[4268.20,4315.10] vol=2.2x ATR=8.96 |
| Stop hit — per-position SL triggered | 2026-03-18 11:25:00 | 4263.66 | 4280.60 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:40:00 | 4187.50 | 4214.88 | 0.00 | ORB-short ORB[4216.70,4270.30] vol=3.2x ATR=12.04 |
| Stop hit — per-position SL triggered | 2026-03-23 10:45:00 | 4199.54 | 4214.46 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:40:00 | 4295.10 | 4289.80 | 0.00 | ORB-long ORB[4231.00,4287.00] vol=3.4x ATR=12.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:40:00 | 4313.43 | 4294.18 | 0.00 | T1 1.5R @ 4313.43 |
| Target hit | 2026-03-25 12:50:00 | 4298.10 | 4298.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2026-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:00:00 | 4089.70 | 4057.20 | 0.00 | ORB-long ORB[4029.30,4075.20] vol=2.7x ATR=11.44 |
| Stop hit — per-position SL triggered | 2026-04-09 13:20:00 | 4078.26 | 4069.06 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:05:00 | 4123.10 | 4106.52 | 0.00 | ORB-long ORB[4074.60,4116.90] vol=2.0x ATR=10.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 12:15:00 | 4138.56 | 4113.32 | 0.00 | T1 1.5R @ 4138.56 |
| Target hit | 2026-04-10 15:20:00 | 4159.90 | 4130.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 4112.70 | 4137.87 | 0.00 | ORB-short ORB[4125.20,4158.90] vol=1.9x ATR=9.65 |
| Stop hit — per-position SL triggered | 2026-04-16 11:20:00 | 4122.35 | 4137.27 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:15:00 | 4088.80 | 4108.21 | 0.00 | ORB-short ORB[4093.50,4140.60] vol=1.8x ATR=8.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:25:00 | 4076.42 | 4106.35 | 0.00 | T1 1.5R @ 4076.42 |
| Stop hit — per-position SL triggered | 2026-04-22 11:40:00 | 4088.80 | 4097.93 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:45:00 | 4136.10 | 4117.12 | 0.00 | ORB-long ORB[4065.10,4126.00] vol=1.5x ATR=17.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:10:00 | 4161.88 | 4129.22 | 0.00 | T1 1.5R @ 4161.88 |
| Stop hit — per-position SL triggered | 2026-04-23 11:40:00 | 4136.10 | 4142.20 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 4195.70 | 4167.99 | 0.00 | ORB-long ORB[4118.60,4175.00] vol=2.6x ATR=14.09 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 4181.61 | 4176.66 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:40:00 | 4240.00 | 4231.06 | 0.00 | ORB-long ORB[4198.40,4237.50] vol=3.0x ATR=9.78 |
| Stop hit — per-position SL triggered | 2026-04-28 11:20:00 | 4230.22 | 4233.41 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:45:00 | 4323.20 | 4299.84 | 0.00 | ORB-long ORB[4276.80,4307.20] vol=2.5x ATR=11.85 |
| Stop hit — per-position SL triggered | 2026-05-06 09:55:00 | 4311.35 | 4302.41 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 3993.60 | 2026-02-09 11:00:00 | 4018.14 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-09 10:30:00 | 3993.60 | 2026-02-09 11:45:00 | 3999.20 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2026-02-10 11:00:00 | 4067.40 | 2026-02-10 11:10:00 | 4059.82 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-13 09:45:00 | 4111.10 | 2026-02-13 09:55:00 | 4098.65 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-26 09:45:00 | 4462.30 | 2026-02-26 09:55:00 | 4448.68 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-27 10:55:00 | 4365.60 | 2026-02-27 11:10:00 | 4374.29 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-03-05 09:30:00 | 4386.90 | 2026-03-05 09:35:00 | 4373.51 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-06 09:35:00 | 4342.00 | 2026-03-06 10:25:00 | 4325.35 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-03-06 09:35:00 | 4342.00 | 2026-03-06 12:05:00 | 4339.50 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2026-03-13 10:40:00 | 4407.00 | 2026-03-13 11:10:00 | 4386.91 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-13 10:40:00 | 4407.00 | 2026-03-13 11:45:00 | 4407.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:15:00 | 4290.90 | 2026-03-17 10:25:00 | 4277.43 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-18 11:15:00 | 4254.70 | 2026-03-18 11:25:00 | 4263.66 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-23 10:40:00 | 4187.50 | 2026-03-23 10:45:00 | 4199.54 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-25 10:40:00 | 4295.10 | 2026-03-25 11:40:00 | 4313.43 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-03-25 10:40:00 | 4295.10 | 2026-03-25 12:50:00 | 4298.10 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2026-04-09 11:00:00 | 4089.70 | 2026-04-09 13:20:00 | 4078.26 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-10 11:05:00 | 4123.10 | 2026-04-10 12:15:00 | 4138.56 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-04-10 11:05:00 | 4123.10 | 2026-04-10 15:20:00 | 4159.90 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2026-04-16 11:15:00 | 4112.70 | 2026-04-16 11:20:00 | 4122.35 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-04-22 11:15:00 | 4088.80 | 2026-04-22 11:25:00 | 4076.42 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-04-22 11:15:00 | 4088.80 | 2026-04-22 11:40:00 | 4088.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 09:45:00 | 4136.10 | 2026-04-23 10:10:00 | 4161.88 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-23 09:45:00 | 4136.10 | 2026-04-23 11:40:00 | 4136.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:40:00 | 4195.70 | 2026-04-27 10:05:00 | 4181.61 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-28 10:40:00 | 4240.00 | 2026-04-28 11:20:00 | 4230.22 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-05-06 09:45:00 | 4323.20 | 2026-05-06 09:55:00 | 4311.35 | STOP_HIT | 1.00 | -0.27% |
