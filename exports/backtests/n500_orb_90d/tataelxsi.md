# Tata Elxsi Ltd. (TATAELXSI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4313.00
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
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 13
- **Target hits / Stop hits / Partials:** 2 / 13 / 5
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 1.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.08% | 0.8% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.08% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.11% | 1.1% |
| SELL @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.11% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 7 | 35.0% | 2 | 13 | 5 | 0.09% | 1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 5289.00 | 5262.26 | 0.00 | ORB-long ORB[5228.00,5266.00] vol=2.0x ATR=12.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 5308.27 | 5283.63 | 0.00 | T1 1.5R @ 5308.27 |
| Target hit | 2026-02-10 11:45:00 | 5325.00 | 5328.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:55:00 | 4773.00 | 4802.11 | 0.00 | ORB-short ORB[4778.50,4836.50] vol=2.5x ATR=15.93 |
| Stop hit — per-position SL triggered | 2026-02-16 12:10:00 | 4788.93 | 4792.51 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:55:00 | 4800.00 | 4832.06 | 0.00 | ORB-short ORB[4802.00,4873.00] vol=1.7x ATR=19.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:40:00 | 4771.21 | 4813.57 | 0.00 | T1 1.5R @ 4771.21 |
| Target hit | 2026-02-23 15:20:00 | 4703.50 | 4744.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-03-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:35:00 | 4271.00 | 4275.67 | 0.00 | ORB-short ORB[4272.00,4327.00] vol=1.6x ATR=12.96 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 4283.96 | 4275.62 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:20:00 | 4198.70 | 4223.20 | 0.00 | ORB-short ORB[4215.00,4255.00] vol=3.5x ATR=15.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:30:00 | 4175.79 | 4214.12 | 0.00 | T1 1.5R @ 4175.79 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 4198.70 | 4193.47 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:00:00 | 4119.20 | 4156.44 | 0.00 | ORB-short ORB[4135.80,4190.00] vol=1.6x ATR=17.20 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 4136.40 | 4155.96 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:45:00 | 4246.00 | 4216.61 | 0.00 | ORB-long ORB[4187.50,4228.60] vol=1.7x ATR=18.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:20:00 | 4274.22 | 4243.64 | 0.00 | T1 1.5R @ 4274.22 |
| Stop hit — per-position SL triggered | 2026-03-25 11:35:00 | 4246.00 | 4243.97 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:55:00 | 4055.60 | 4080.11 | 0.00 | ORB-short ORB[4073.60,4130.00] vol=1.8x ATR=13.48 |
| Stop hit — per-position SL triggered | 2026-03-30 11:05:00 | 4069.08 | 4078.93 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 4399.40 | 4437.45 | 0.00 | ORB-short ORB[4425.20,4490.00] vol=1.7x ATR=16.53 |
| Stop hit — per-position SL triggered | 2026-04-10 09:55:00 | 4415.93 | 4424.94 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 4598.00 | 4554.98 | 0.00 | ORB-long ORB[4516.60,4561.70] vol=2.8x ATR=14.95 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 4583.05 | 4557.18 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 4258.90 | 4225.20 | 0.00 | ORB-long ORB[4198.60,4244.70] vol=1.5x ATR=19.26 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 4239.64 | 4233.51 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 4181.50 | 4158.48 | 0.00 | ORB-long ORB[4138.50,4180.00] vol=2.0x ATR=12.61 |
| Stop hit — per-position SL triggered | 2026-04-29 09:55:00 | 4168.89 | 4166.08 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 4208.00 | 4187.90 | 0.00 | ORB-long ORB[4151.00,4205.80] vol=1.8x ATR=12.38 |
| Stop hit — per-position SL triggered | 2026-05-05 09:55:00 | 4195.62 | 4189.81 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 4295.20 | 4278.23 | 0.00 | ORB-long ORB[4245.00,4290.00] vol=3.7x ATR=12.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:55:00 | 4314.56 | 4290.09 | 0.00 | T1 1.5R @ 4314.56 |
| Stop hit — per-position SL triggered | 2026-05-06 10:35:00 | 4295.20 | 4293.88 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:50:00 | 4276.60 | 4296.62 | 0.00 | ORB-short ORB[4284.00,4328.70] vol=1.6x ATR=13.30 |
| Stop hit — per-position SL triggered | 2026-05-07 12:20:00 | 4289.90 | 4288.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:35:00 | 5289.00 | 2026-02-10 09:40:00 | 5308.27 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-10 09:35:00 | 5289.00 | 2026-02-10 11:45:00 | 5325.00 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2026-02-16 10:55:00 | 4773.00 | 2026-02-16 12:10:00 | 4788.93 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-23 09:55:00 | 4800.00 | 2026-02-23 10:40:00 | 4771.21 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-23 09:55:00 | 4800.00 | 2026-02-23 15:20:00 | 4703.50 | TARGET_HIT | 0.50 | 2.01% |
| SELL | retest1 | 2026-03-13 10:35:00 | 4271.00 | 2026-03-13 10:50:00 | 4283.96 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-16 10:20:00 | 4198.70 | 2026-03-16 10:30:00 | 4175.79 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-16 10:20:00 | 4198.70 | 2026-03-16 11:15:00 | 4198.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-23 11:00:00 | 4119.20 | 2026-03-23 11:05:00 | 4136.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-25 09:45:00 | 4246.00 | 2026-03-25 11:20:00 | 4274.22 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-03-25 09:45:00 | 4246.00 | 2026-03-25 11:35:00 | 4246.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-30 10:55:00 | 4055.60 | 2026-03-30 11:05:00 | 4069.08 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-10 09:40:00 | 4399.40 | 2026-04-10 09:55:00 | 4415.93 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-21 10:10:00 | 4598.00 | 2026-04-21 10:15:00 | 4583.05 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-27 09:45:00 | 4258.90 | 2026-04-27 10:05:00 | 4239.64 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-29 09:40:00 | 4181.50 | 2026-04-29 09:55:00 | 4168.89 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 09:50:00 | 4208.00 | 2026-05-05 09:55:00 | 4195.62 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-06 09:30:00 | 4295.20 | 2026-05-06 09:55:00 | 4314.56 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-05-06 09:30:00 | 4295.20 | 2026-05-06 10:35:00 | 4295.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 09:50:00 | 4276.60 | 2026-05-07 12:20:00 | 4289.90 | STOP_HIT | 1.00 | -0.31% |
