# InterGlobe Aviation Ltd. (INDIGO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4522.00
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 13
- **Target hits / Stop hits / Partials:** 1 / 13 / 3
- **Avg / median % per leg:** -0.13% / -0.21%
- **Sum % (uncompounded):** -2.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 3 | 25.0% | 1 | 9 | 2 | -0.15% | -1.8% |
| BUY @ 2nd Alert (retest1) | 12 | 3 | 25.0% | 1 | 9 | 2 | -0.15% | -1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.09% | -0.5% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.09% | -0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 4 | 23.5% | 1 | 13 | 3 | -0.13% | -2.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 4963.60 | 4945.05 | 0.00 | ORB-long ORB[4888.70,4948.00] vol=2.0x ATR=27.49 |
| Stop hit — per-position SL triggered | 2026-02-09 11:50:00 | 4936.11 | 4954.58 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 4931.30 | 4940.24 | 0.00 | ORB-short ORB[4938.40,4960.10] vol=2.8x ATR=7.95 |
| Stop hit — per-position SL triggered | 2026-02-10 09:35:00 | 4939.25 | 4939.84 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:30:00 | 5005.00 | 4975.30 | 0.00 | ORB-long ORB[4960.10,4990.00] vol=2.1x ATR=9.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:35:00 | 5019.75 | 4984.10 | 0.00 | T1 1.5R @ 5019.75 |
| Target hit | 2026-02-11 13:25:00 | 5011.60 | 5011.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2026-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:55:00 | 4925.00 | 4945.04 | 0.00 | ORB-short ORB[4949.00,4988.00] vol=1.6x ATR=8.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:15:00 | 4911.76 | 4942.95 | 0.00 | T1 1.5R @ 4911.76 |
| Stop hit — per-position SL triggered | 2026-02-13 11:45:00 | 4925.00 | 4937.33 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:25:00 | 4951.50 | 4932.62 | 0.00 | ORB-long ORB[4903.00,4939.00] vol=1.6x ATR=10.43 |
| Stop hit — per-position SL triggered | 2026-02-16 10:30:00 | 4941.07 | 4933.12 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 5000.40 | 4980.09 | 0.00 | ORB-long ORB[4937.50,5000.00] vol=1.6x ATR=12.10 |
| Stop hit — per-position SL triggered | 2026-02-17 09:40:00 | 4988.30 | 4982.98 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:05:00 | 4996.30 | 4981.42 | 0.00 | ORB-long ORB[4967.20,4994.90] vol=2.4x ATR=6.88 |
| Stop hit — per-position SL triggered | 2026-02-18 12:25:00 | 4989.42 | 4985.87 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 4870.00 | 4888.62 | 0.00 | ORB-short ORB[4875.00,4939.00] vol=1.8x ATR=14.84 |
| Stop hit — per-position SL triggered | 2026-02-19 10:05:00 | 4884.84 | 4881.28 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 4830.00 | 4825.48 | 0.00 | ORB-long ORB[4783.90,4827.20] vol=10.2x ATR=12.10 |
| Stop hit — per-position SL triggered | 2026-02-20 11:20:00 | 4817.90 | 4825.38 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:50:00 | 4261.20 | 4231.46 | 0.00 | ORB-long ORB[4181.00,4236.00] vol=1.5x ATR=12.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:00:00 | 4279.92 | 4237.77 | 0.00 | T1 1.5R @ 4279.92 |
| Stop hit — per-position SL triggered | 2026-03-25 12:50:00 | 4261.20 | 4259.23 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 4646.70 | 4616.02 | 0.00 | ORB-long ORB[4575.70,4640.20] vol=2.6x ATR=26.55 |
| Stop hit — per-position SL triggered | 2026-04-15 09:45:00 | 4620.15 | 4620.01 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 4742.80 | 4712.53 | 0.00 | ORB-long ORB[4677.70,4717.80] vol=1.8x ATR=13.61 |
| Stop hit — per-position SL triggered | 2026-04-21 09:35:00 | 4729.19 | 4716.96 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:55:00 | 4636.60 | 4663.43 | 0.00 | ORB-short ORB[4655.30,4694.80] vol=1.8x ATR=11.83 |
| Stop hit — per-position SL triggered | 2026-04-22 10:05:00 | 4648.43 | 4659.12 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 4372.00 | 4347.03 | 0.00 | ORB-long ORB[4315.60,4362.00] vol=1.7x ATR=17.52 |
| Stop hit — per-position SL triggered | 2026-05-06 12:15:00 | 4354.48 | 4365.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 4963.60 | 2026-02-09 11:50:00 | 4936.11 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2026-02-10 09:30:00 | 4931.30 | 2026-02-10 09:35:00 | 4939.25 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-02-11 10:30:00 | 5005.00 | 2026-02-11 10:35:00 | 5019.75 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-11 10:30:00 | 5005.00 | 2026-02-11 13:25:00 | 5011.60 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2026-02-13 10:55:00 | 4925.00 | 2026-02-13 11:15:00 | 4911.76 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-13 10:55:00 | 4925.00 | 2026-02-13 11:45:00 | 4925.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:25:00 | 4951.50 | 2026-02-16 10:30:00 | 4941.07 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-17 09:30:00 | 5000.40 | 2026-02-17 09:40:00 | 4988.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-18 11:05:00 | 4996.30 | 2026-02-18 12:25:00 | 4989.42 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2026-02-19 09:35:00 | 4870.00 | 2026-02-19 10:05:00 | 4884.84 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-20 10:50:00 | 4830.00 | 2026-02-20 11:20:00 | 4817.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-25 10:50:00 | 4261.20 | 2026-03-25 11:00:00 | 4279.92 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-03-25 10:50:00 | 4261.20 | 2026-03-25 12:50:00 | 4261.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 09:30:00 | 4646.70 | 2026-04-15 09:45:00 | 4620.15 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-04-21 09:30:00 | 4742.80 | 2026-04-21 09:35:00 | 4729.19 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-22 09:55:00 | 4636.60 | 2026-04-22 10:05:00 | 4648.43 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-05-06 09:35:00 | 4372.00 | 2026-05-06 12:15:00 | 4354.48 | STOP_HIT | 1.00 | -0.40% |
