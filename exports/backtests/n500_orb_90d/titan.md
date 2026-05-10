# Titan Company Ltd. (TITAN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4517.00
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 14
- **Target hits / Stop hits / Partials:** 4 / 14 / 7
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.15% | -1.2% |
| BUY @ 2nd Alert (retest1) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.15% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 10 | 58.8% | 4 | 7 | 6 | 0.24% | 4.0% |
| SELL @ 2nd Alert (retest1) | 17 | 10 | 58.8% | 4 | 7 | 6 | 0.24% | 4.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 11 | 44.0% | 4 | 14 | 7 | 0.11% | 2.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 4315.40 | 4296.58 | 0.00 | ORB-long ORB[4267.60,4312.10] vol=1.9x ATR=14.00 |
| Stop hit — per-position SL triggered | 2026-02-10 09:55:00 | 4301.40 | 4301.58 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:00:00 | 4232.80 | 4248.85 | 0.00 | ORB-short ORB[4238.70,4271.80] vol=1.6x ATR=8.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:55:00 | 4220.13 | 4244.84 | 0.00 | T1 1.5R @ 4220.13 |
| Stop hit — per-position SL triggered | 2026-02-13 12:25:00 | 4232.80 | 4239.25 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:35:00 | 4180.30 | 4164.91 | 0.00 | ORB-long ORB[4128.80,4163.80] vol=2.1x ATR=9.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:05:00 | 4194.76 | 4170.07 | 0.00 | T1 1.5R @ 4194.76 |
| Stop hit — per-position SL triggered | 2026-02-16 11:25:00 | 4180.30 | 4172.40 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 11:00:00 | 4287.60 | 4263.51 | 0.00 | ORB-long ORB[4232.40,4270.30] vol=2.3x ATR=10.80 |
| Stop hit — per-position SL triggered | 2026-03-06 12:35:00 | 4276.80 | 4272.95 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:00:00 | 4027.00 | 4060.24 | 0.00 | ORB-short ORB[4051.20,4097.70] vol=2.2x ATR=14.27 |
| Stop hit — per-position SL triggered | 2026-03-16 11:05:00 | 4041.27 | 4058.20 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:00:00 | 3930.40 | 3961.99 | 0.00 | ORB-short ORB[4001.10,4059.00] vol=1.6x ATR=14.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:05:00 | 3908.75 | 3945.10 | 0.00 | T1 1.5R @ 3908.75 |
| Target hit | 2026-03-23 15:20:00 | 3849.50 | 3897.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-04-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 10:30:00 | 4176.80 | 4195.00 | 0.00 | ORB-short ORB[4185.00,4239.00] vol=2.0x ATR=13.66 |
| Stop hit — per-position SL triggered | 2026-04-07 10:35:00 | 4190.46 | 4194.70 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:55:00 | 4451.10 | 4462.46 | 0.00 | ORB-short ORB[4461.00,4497.40] vol=1.9x ATR=9.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 13:05:00 | 4436.51 | 4456.50 | 0.00 | T1 1.5R @ 4436.51 |
| Target hit | 2026-04-09 15:05:00 | 4450.00 | 4449.85 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — SELL (started 2026-04-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:55:00 | 4498.00 | 4515.74 | 0.00 | ORB-short ORB[4502.00,4554.00] vol=3.1x ATR=11.09 |
| Stop hit — per-position SL triggered | 2026-04-16 11:00:00 | 4509.09 | 4514.95 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:35:00 | 4365.60 | 4456.65 | 0.00 | ORB-short ORB[4427.10,4484.40] vol=2.5x ATR=20.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:40:00 | 4334.81 | 4432.06 | 0.00 | T1 1.5R @ 4334.81 |
| Stop hit — per-position SL triggered | 2026-04-17 10:45:00 | 4365.60 | 4426.70 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 4457.00 | 4467.53 | 0.00 | ORB-short ORB[4460.00,4487.40] vol=1.5x ATR=9.97 |
| Stop hit — per-position SL triggered | 2026-04-22 10:30:00 | 4466.97 | 4463.83 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 11:05:00 | 4460.30 | 4428.50 | 0.00 | ORB-long ORB[4393.50,4430.00] vol=1.5x ATR=10.74 |
| Stop hit — per-position SL triggered | 2026-04-23 11:10:00 | 4449.56 | 4429.53 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:10:00 | 4399.00 | 4434.41 | 0.00 | ORB-short ORB[4431.00,4464.00] vol=1.5x ATR=11.65 |
| Stop hit — per-position SL triggered | 2026-04-27 10:15:00 | 4410.65 | 4433.31 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:55:00 | 4474.90 | 4452.31 | 0.00 | ORB-long ORB[4395.40,4460.30] vol=1.7x ATR=8.78 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 4466.12 | 4453.23 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:50:00 | 4459.30 | 4440.89 | 0.00 | ORB-long ORB[4410.20,4453.50] vol=2.0x ATR=10.29 |
| Stop hit — per-position SL triggered | 2026-04-29 11:20:00 | 4449.01 | 4449.03 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 4378.90 | 4371.43 | 0.00 | ORB-long ORB[4330.30,4376.90] vol=1.6x ATR=12.29 |
| Stop hit — per-position SL triggered | 2026-05-05 10:45:00 | 4366.61 | 4377.60 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 4360.80 | 4396.14 | 0.00 | ORB-short ORB[4390.00,4441.10] vol=1.9x ATR=15.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:45:00 | 4337.99 | 4389.72 | 0.00 | T1 1.5R @ 4337.99 |
| Target hit | 2026-05-06 12:15:00 | 4338.90 | 4337.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2026-05-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:25:00 | 4299.70 | 4324.16 | 0.00 | ORB-short ORB[4333.50,4359.00] vol=1.8x ATR=12.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:30:00 | 4280.69 | 4318.94 | 0.00 | T1 1.5R @ 4280.69 |
| Target hit | 2026-05-07 12:35:00 | 4298.90 | 4293.78 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:35:00 | 4315.40 | 2026-02-10 09:55:00 | 4301.40 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-13 11:00:00 | 4232.80 | 2026-02-13 11:55:00 | 4220.13 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-13 11:00:00 | 4232.80 | 2026-02-13 12:25:00 | 4232.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:35:00 | 4180.30 | 2026-02-16 11:05:00 | 4194.76 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-16 10:35:00 | 4180.30 | 2026-02-16 11:25:00 | 4180.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 11:00:00 | 4287.60 | 2026-03-06 12:35:00 | 4276.80 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-16 11:00:00 | 4027.00 | 2026-03-16 11:05:00 | 4041.27 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-23 11:00:00 | 3930.40 | 2026-03-23 12:05:00 | 3908.75 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-23 11:00:00 | 3930.40 | 2026-03-23 15:20:00 | 3849.50 | TARGET_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2026-04-07 10:30:00 | 4176.80 | 2026-04-07 10:35:00 | 4190.46 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-09 10:55:00 | 4451.10 | 2026-04-09 13:05:00 | 4436.51 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-04-09 10:55:00 | 4451.10 | 2026-04-09 15:05:00 | 4450.00 | TARGET_HIT | 0.50 | 0.02% |
| SELL | retest1 | 2026-04-16 10:55:00 | 4498.00 | 2026-04-16 11:00:00 | 4509.09 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-17 10:35:00 | 4365.60 | 2026-04-17 10:40:00 | 4334.81 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-04-17 10:35:00 | 4365.60 | 2026-04-17 10:45:00 | 4365.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 09:45:00 | 4457.00 | 2026-04-22 10:30:00 | 4466.97 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-23 11:05:00 | 4460.30 | 2026-04-23 11:10:00 | 4449.56 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-27 10:10:00 | 4399.00 | 2026-04-27 10:15:00 | 4410.65 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-28 10:55:00 | 4474.90 | 2026-04-28 11:05:00 | 4466.12 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-04-29 10:50:00 | 4459.30 | 2026-04-29 11:20:00 | 4449.01 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-05-05 09:35:00 | 4378.90 | 2026-05-05 10:45:00 | 4366.61 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-06 09:40:00 | 4360.80 | 2026-05-06 09:45:00 | 4337.99 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-05-06 09:40:00 | 4360.80 | 2026-05-06 12:15:00 | 4338.90 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-05-07 10:25:00 | 4299.70 | 2026-05-07 10:30:00 | 4280.69 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-05-07 10:25:00 | 4299.70 | 2026-05-07 12:35:00 | 4298.90 | TARGET_HIT | 0.50 | 0.02% |
