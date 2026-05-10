# Hindustan Aeronautics Ltd. (HAL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4790.00
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 5
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 3.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.33% | 4.3% |
| BUY @ 2nd Alert (retest1) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.33% | 4.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.11% | -0.6% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.11% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 7 | 36.8% | 2 | 12 | 5 | 0.19% | 3.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 4176.20 | 4193.65 | 0.00 | ORB-short ORB[4177.10,4228.70] vol=5.8x ATR=11.21 |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 4187.41 | 4192.56 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:10:00 | 4200.00 | 4182.62 | 0.00 | ORB-long ORB[4121.20,4178.70] vol=1.8x ATR=10.38 |
| Stop hit — per-position SL triggered | 2026-02-20 11:50:00 | 4189.62 | 4184.54 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:00:00 | 3963.90 | 3974.45 | 0.00 | ORB-short ORB[3965.50,4001.00] vol=3.1x ATR=8.52 |
| Stop hit — per-position SL triggered | 2026-02-27 10:25:00 | 3972.42 | 3973.63 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 3887.60 | 3873.15 | 0.00 | ORB-long ORB[3840.90,3881.60] vol=1.5x ATR=11.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:05:00 | 3905.05 | 3877.29 | 0.00 | T1 1.5R @ 3905.05 |
| Stop hit — per-position SL triggered | 2026-03-05 11:45:00 | 3887.60 | 3888.76 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:05:00 | 3928.50 | 3947.78 | 0.00 | ORB-short ORB[3968.60,4004.10] vol=5.6x ATR=11.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 3911.26 | 3931.74 | 0.00 | T1 1.5R @ 3911.26 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 3928.50 | 3924.49 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:20:00 | 3941.40 | 3917.85 | 0.00 | ORB-long ORB[3892.50,3930.30] vol=1.8x ATR=11.64 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 3929.76 | 3918.66 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:15:00 | 3582.00 | 3624.92 | 0.00 | ORB-short ORB[3619.00,3665.00] vol=1.8x ATR=11.34 |
| Stop hit — per-position SL triggered | 2026-03-27 10:20:00 | 3593.34 | 3622.32 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:45:00 | 4212.00 | 4179.12 | 0.00 | ORB-long ORB[4147.00,4196.90] vol=5.2x ATR=16.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:50:00 | 4236.95 | 4185.66 | 0.00 | T1 1.5R @ 4236.95 |
| Stop hit — per-position SL triggered | 2026-04-15 10:05:00 | 4212.00 | 4189.23 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 4396.00 | 4366.78 | 0.00 | ORB-long ORB[4335.50,4380.00] vol=2.7x ATR=12.57 |
| Stop hit — per-position SL triggered | 2026-04-21 10:00:00 | 4383.43 | 4375.42 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 4312.70 | 4337.12 | 0.00 | ORB-short ORB[4324.00,4387.30] vol=2.0x ATR=11.96 |
| Stop hit — per-position SL triggered | 2026-04-24 10:05:00 | 4324.66 | 4325.08 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 4328.00 | 4306.28 | 0.00 | ORB-long ORB[4280.00,4325.00] vol=1.8x ATR=12.95 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 4315.05 | 4313.92 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 4399.50 | 4358.30 | 0.00 | ORB-long ORB[4318.00,4374.60] vol=2.5x ATR=14.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:50:00 | 4421.93 | 4386.11 | 0.00 | T1 1.5R @ 4421.93 |
| Target hit | 2026-05-04 15:20:00 | 4560.00 | 4532.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 4650.50 | 4634.94 | 0.00 | ORB-long ORB[4614.50,4644.00] vol=2.3x ATR=10.61 |
| Stop hit — per-position SL triggered | 2026-05-06 09:35:00 | 4639.89 | 4635.64 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 4659.00 | 4646.90 | 0.00 | ORB-long ORB[4628.10,4655.90] vol=2.1x ATR=10.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:40:00 | 4675.42 | 4658.30 | 0.00 | T1 1.5R @ 4675.42 |
| Target hit | 2026-05-07 09:55:00 | 4662.30 | 4666.76 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-16 11:05:00 | 4176.20 | 2026-02-16 11:15:00 | 4187.41 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-20 11:10:00 | 4200.00 | 2026-02-20 11:50:00 | 4189.62 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-27 10:00:00 | 3963.90 | 2026-02-27 10:25:00 | 3972.42 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-03-05 10:45:00 | 3887.60 | 2026-03-05 11:05:00 | 3905.05 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-03-05 10:45:00 | 3887.60 | 2026-03-05 11:45:00 | 3887.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:05:00 | 3928.50 | 2026-03-13 10:10:00 | 3911.26 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-03-13 10:05:00 | 3928.50 | 2026-03-13 10:50:00 | 3928.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:20:00 | 3941.40 | 2026-03-17 10:25:00 | 3929.76 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-27 10:15:00 | 3582.00 | 2026-03-27 10:20:00 | 3593.34 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-15 09:45:00 | 4212.00 | 2026-04-15 09:50:00 | 4236.95 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-15 09:45:00 | 4212.00 | 2026-04-15 10:05:00 | 4212.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:35:00 | 4396.00 | 2026-04-21 10:00:00 | 4383.43 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-24 09:30:00 | 4312.70 | 2026-04-24 10:05:00 | 4324.66 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-27 09:45:00 | 4328.00 | 2026-04-27 10:05:00 | 4315.05 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-04 09:45:00 | 4399.50 | 2026-05-04 09:50:00 | 4421.93 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-05-04 09:45:00 | 4399.50 | 2026-05-04 15:20:00 | 4560.00 | TARGET_HIT | 0.50 | 3.65% |
| BUY | retest1 | 2026-05-06 09:30:00 | 4650.50 | 2026-05-06 09:35:00 | 4639.89 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-05-07 09:35:00 | 4659.00 | 2026-05-07 09:40:00 | 4675.42 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-05-07 09:35:00 | 4659.00 | 2026-05-07 09:55:00 | 4662.30 | TARGET_HIT | 0.50 | 0.07% |
