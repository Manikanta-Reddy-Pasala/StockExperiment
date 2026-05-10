# Kaynes Technology India Ltd. (KAYNES)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4497.00
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
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 1 / 13 / 7
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 3.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 3 | 27.3% | 0 | 8 | 3 | 0.03% | 0.3% |
| BUY @ 2nd Alert (retest1) | 11 | 3 | 27.3% | 0 | 8 | 3 | 0.03% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.29% | 2.9% |
| SELL @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.29% | 2.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 8 | 38.1% | 1 | 13 | 7 | 0.15% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:10:00 | 4019.70 | 3981.24 | 0.00 | ORB-long ORB[3950.10,3980.00] vol=2.1x ATR=16.81 |
| Stop hit — per-position SL triggered | 2026-02-11 10:30:00 | 4002.89 | 3992.03 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 3958.90 | 3990.78 | 0.00 | ORB-short ORB[3971.00,4019.80] vol=1.6x ATR=15.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:50:00 | 3935.19 | 3982.09 | 0.00 | T1 1.5R @ 3935.19 |
| Stop hit — per-position SL triggered | 2026-02-18 11:40:00 | 3958.90 | 3954.97 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:35:00 | 3923.10 | 3891.72 | 0.00 | ORB-long ORB[3865.10,3911.00] vol=3.0x ATR=17.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:00:00 | 3948.65 | 3901.51 | 0.00 | T1 1.5R @ 3948.65 |
| Stop hit — per-position SL triggered | 2026-02-25 12:40:00 | 3923.10 | 3928.66 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:45:00 | 3875.20 | 3921.36 | 0.00 | ORB-short ORB[3914.80,3958.10] vol=2.0x ATR=16.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:00:00 | 3850.00 | 3907.41 | 0.00 | T1 1.5R @ 3850.00 |
| Stop hit — per-position SL triggered | 2026-02-27 11:45:00 | 3875.20 | 3878.53 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 3675.50 | 3719.99 | 0.00 | ORB-short ORB[3705.20,3754.90] vol=1.8x ATR=19.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:30:00 | 3646.62 | 3689.53 | 0.00 | T1 1.5R @ 3646.62 |
| Stop hit — per-position SL triggered | 2026-03-05 11:25:00 | 3675.50 | 3677.36 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 3861.90 | 3825.29 | 0.00 | ORB-long ORB[3805.00,3852.30] vol=2.1x ATR=20.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 09:45:00 | 3893.07 | 3838.74 | 0.00 | T1 1.5R @ 3893.07 |
| Stop hit — per-position SL triggered | 2026-03-11 09:55:00 | 3861.90 | 3845.65 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:35:00 | 3600.00 | 3565.45 | 0.00 | ORB-long ORB[3538.00,3587.30] vol=2.5x ATR=20.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:50:00 | 3630.73 | 3588.17 | 0.00 | T1 1.5R @ 3630.73 |
| Stop hit — per-position SL triggered | 2026-03-17 10:50:00 | 3600.00 | 3602.63 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 3914.30 | 3895.49 | 0.00 | ORB-long ORB[3869.90,3908.80] vol=1.9x ATR=16.69 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 3897.61 | 3895.97 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 4109.00 | 4088.97 | 0.00 | ORB-long ORB[4060.00,4100.00] vol=2.4x ATR=15.92 |
| Stop hit — per-position SL triggered | 2026-04-16 09:35:00 | 4093.08 | 4092.15 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 4204.80 | 4230.49 | 0.00 | ORB-short ORB[4206.10,4263.90] vol=1.6x ATR=19.68 |
| Stop hit — per-position SL triggered | 2026-04-17 10:05:00 | 4224.48 | 4217.31 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 4280.00 | 4243.20 | 0.00 | ORB-long ORB[4213.60,4269.00] vol=2.4x ATR=16.76 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 4263.24 | 4244.36 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 4232.90 | 4272.51 | 0.00 | ORB-short ORB[4254.20,4315.00] vol=2.0x ATR=22.27 |
| Stop hit — per-position SL triggered | 2026-04-27 09:40:00 | 4255.17 | 4268.03 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 4165.60 | 4190.56 | 0.00 | ORB-short ORB[4185.40,4237.70] vol=1.7x ATR=12.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 12:05:00 | 4147.04 | 4184.53 | 0.00 | T1 1.5R @ 4147.04 |
| Target hit | 2026-04-28 15:20:00 | 4108.60 | 4142.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 4364.60 | 4334.00 | 0.00 | ORB-long ORB[4306.30,4352.00] vol=1.9x ATR=17.44 |
| Stop hit — per-position SL triggered | 2026-05-07 09:40:00 | 4347.16 | 4342.40 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:10:00 | 4019.70 | 2026-02-11 10:30:00 | 4002.89 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-18 09:45:00 | 3958.90 | 2026-02-18 09:50:00 | 3935.19 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-18 09:45:00 | 3958.90 | 2026-02-18 11:40:00 | 3958.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:35:00 | 3923.10 | 2026-02-25 11:00:00 | 3948.65 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-02-25 10:35:00 | 3923.10 | 2026-02-25 12:40:00 | 3923.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:45:00 | 3875.20 | 2026-02-27 10:00:00 | 3850.00 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-02-27 09:45:00 | 3875.20 | 2026-02-27 11:45:00 | 3875.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 09:30:00 | 3675.50 | 2026-03-05 10:30:00 | 3646.62 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2026-03-05 09:30:00 | 3675.50 | 2026-03-05 11:25:00 | 3675.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 09:40:00 | 3861.90 | 2026-03-11 09:45:00 | 3893.07 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2026-03-11 09:40:00 | 3861.90 | 2026-03-11 09:55:00 | 3861.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 09:35:00 | 3600.00 | 2026-03-17 09:50:00 | 3630.73 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2026-03-17 09:35:00 | 3600.00 | 2026-03-17 10:50:00 | 3600.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:30:00 | 3914.30 | 2026-04-10 09:35:00 | 3897.61 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-16 09:30:00 | 4109.00 | 2026-04-16 09:35:00 | 4093.08 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-17 09:30:00 | 4204.80 | 2026-04-17 10:05:00 | 4224.48 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-21 10:10:00 | 4280.00 | 2026-04-21 10:15:00 | 4263.24 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-27 09:30:00 | 4232.90 | 2026-04-27 09:40:00 | 4255.17 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2026-04-28 11:10:00 | 4165.60 | 2026-04-28 12:05:00 | 4147.04 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-28 11:10:00 | 4165.60 | 2026-04-28 15:20:00 | 4108.60 | TARGET_HIT | 0.50 | 1.37% |
| BUY | retest1 | 2026-05-07 09:35:00 | 4364.60 | 2026-05-07 09:40:00 | 4347.16 | STOP_HIT | 1.00 | -0.40% |
