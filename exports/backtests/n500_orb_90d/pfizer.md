# Pfizer Ltd. (PFIZER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4793.00
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 18
- **Target hits / Stop hits / Partials:** 3 / 18 / 4
- **Avg / median % per leg:** -0.02% / -0.25%
- **Sum % (uncompounded):** -0.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 2 | 16.7% | 1 | 10 | 1 | -0.13% | -1.6% |
| BUY @ 2nd Alert (retest1) | 12 | 2 | 16.7% | 1 | 10 | 1 | -0.13% | -1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.08% | 1.0% |
| SELL @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.08% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 7 | 28.0% | 3 | 18 | 4 | -0.02% | -0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 5029.60 | 5042.05 | 0.00 | ORB-short ORB[5031.20,5072.30] vol=1.7x ATR=12.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:05:00 | 5010.29 | 5039.03 | 0.00 | T1 1.5R @ 5010.29 |
| Stop hit — per-position SL triggered | 2026-02-16 12:40:00 | 5029.60 | 5038.25 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:05:00 | 5138.80 | 5129.37 | 0.00 | ORB-long ORB[5070.60,5110.00] vol=2.9x ATR=16.06 |
| Stop hit — per-position SL triggered | 2026-02-25 11:25:00 | 5122.74 | 5132.43 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:25:00 | 5127.70 | 5108.72 | 0.00 | ORB-long ORB[5060.00,5111.00] vol=2.2x ATR=16.47 |
| Stop hit — per-position SL triggered | 2026-02-27 10:45:00 | 5111.23 | 5109.63 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:55:00 | 4811.00 | 4831.92 | 0.00 | ORB-short ORB[4817.00,4864.00] vol=2.9x ATR=20.25 |
| Stop hit — per-position SL triggered | 2026-03-04 10:00:00 | 4831.25 | 4831.70 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:50:00 | 4898.00 | 4877.58 | 0.00 | ORB-long ORB[4854.00,4894.50] vol=2.6x ATR=18.06 |
| Stop hit — per-position SL triggered | 2026-03-05 10:10:00 | 4879.94 | 4878.29 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 4763.00 | 4774.78 | 0.00 | ORB-short ORB[4766.50,4825.00] vol=2.2x ATR=17.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 13:10:00 | 4736.02 | 4766.39 | 0.00 | T1 1.5R @ 4736.02 |
| Target hit | 2026-03-11 15:20:00 | 4703.00 | 4740.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:35:00 | 4617.00 | 4639.29 | 0.00 | ORB-short ORB[4633.00,4690.50] vol=1.8x ATR=14.82 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 4631.82 | 4638.54 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 09:55:00 | 4682.50 | 4625.45 | 0.00 | ORB-long ORB[4587.00,4653.00] vol=2.2x ATR=28.72 |
| Stop hit — per-position SL triggered | 2026-03-24 10:10:00 | 4653.78 | 4636.12 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 4716.00 | 4691.08 | 0.00 | ORB-long ORB[4651.00,4707.00] vol=5.0x ATR=12.61 |
| Stop hit — per-position SL triggered | 2026-03-27 11:20:00 | 4703.39 | 4692.41 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 11:15:00 | 4722.00 | 4661.67 | 0.00 | ORB-long ORB[4611.00,4671.00] vol=5.9x ATR=15.80 |
| Stop hit — per-position SL triggered | 2026-03-30 11:25:00 | 4706.20 | 4669.75 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:30:00 | 4774.80 | 4792.85 | 0.00 | ORB-short ORB[4781.00,4839.90] vol=1.7x ATR=15.26 |
| Stop hit — per-position SL triggered | 2026-04-07 09:40:00 | 4790.06 | 4789.63 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 11:00:00 | 4802.10 | 4808.06 | 0.00 | ORB-short ORB[4812.50,4860.00] vol=1.7x ATR=7.36 |
| Stop hit — per-position SL triggered | 2026-04-09 11:05:00 | 4809.46 | 4808.04 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 4802.80 | 4794.17 | 0.00 | ORB-long ORB[4750.00,4783.30] vol=3.8x ATR=14.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:50:00 | 4824.97 | 4803.34 | 0.00 | T1 1.5R @ 4824.97 |
| Target hit | 2026-04-10 15:20:00 | 4868.00 | 4828.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-04-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:35:00 | 4923.20 | 4898.04 | 0.00 | ORB-long ORB[4865.40,4920.30] vol=1.8x ATR=14.38 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 4908.82 | 4899.62 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:35:00 | 4979.40 | 4933.03 | 0.00 | ORB-long ORB[4888.00,4950.00] vol=2.6x ATR=16.06 |
| Stop hit — per-position SL triggered | 2026-04-16 10:40:00 | 4963.34 | 4935.20 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:10:00 | 4934.10 | 4903.76 | 0.00 | ORB-long ORB[4864.80,4887.90] vol=2.0x ATR=12.16 |
| Stop hit — per-position SL triggered | 2026-04-17 10:20:00 | 4921.94 | 4905.98 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:15:00 | 4875.20 | 4885.76 | 0.00 | ORB-short ORB[4880.40,4917.20] vol=1.5x ATR=10.19 |
| Stop hit — per-position SL triggered | 2026-04-21 10:30:00 | 4885.39 | 4885.37 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 4783.30 | 4796.46 | 0.00 | ORB-short ORB[4790.00,4825.00] vol=1.8x ATR=8.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 13:10:00 | 4771.03 | 4789.10 | 0.00 | T1 1.5R @ 4771.03 |
| Target hit | 2026-04-29 15:20:00 | 4760.10 | 4780.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2026-04-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:50:00 | 4712.30 | 4731.75 | 0.00 | ORB-short ORB[4721.00,4789.00] vol=2.5x ATR=11.73 |
| Stop hit — per-position SL triggered | 2026-04-30 11:10:00 | 4724.03 | 4726.96 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 4767.60 | 4748.50 | 0.00 | ORB-long ORB[4701.10,4756.90] vol=2.0x ATR=14.55 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 4753.05 | 4755.89 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-05-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:50:00 | 4752.10 | 4777.97 | 0.00 | ORB-short ORB[4783.00,4802.30] vol=3.0x ATR=12.00 |
| Stop hit — per-position SL triggered | 2026-05-07 10:05:00 | 4764.10 | 4773.91 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-16 11:15:00 | 5029.60 | 2026-02-16 12:05:00 | 5010.29 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-16 11:15:00 | 5029.60 | 2026-02-16 12:40:00 | 5029.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:05:00 | 5138.80 | 2026-02-25 11:25:00 | 5122.74 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-27 10:25:00 | 5127.70 | 2026-02-27 10:45:00 | 5111.23 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-04 09:55:00 | 4811.00 | 2026-03-04 10:00:00 | 4831.25 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-05 09:50:00 | 4898.00 | 2026-03-05 10:10:00 | 4879.94 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-11 09:40:00 | 4763.00 | 2026-03-11 13:10:00 | 4736.02 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-11 09:40:00 | 4763.00 | 2026-03-11 15:20:00 | 4703.00 | TARGET_HIT | 0.50 | 1.26% |
| SELL | retest1 | 2026-03-13 10:35:00 | 4617.00 | 2026-03-13 10:50:00 | 4631.82 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-24 09:55:00 | 4682.50 | 2026-03-24 10:10:00 | 4653.78 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2026-03-27 11:05:00 | 4716.00 | 2026-03-27 11:20:00 | 4703.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-30 11:15:00 | 4722.00 | 2026-03-30 11:25:00 | 4706.20 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-07 09:30:00 | 4774.80 | 2026-04-07 09:40:00 | 4790.06 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-09 11:00:00 | 4802.10 | 2026-04-09 11:05:00 | 4809.46 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2026-04-10 09:30:00 | 4802.80 | 2026-04-10 09:50:00 | 4824.97 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-10 09:30:00 | 4802.80 | 2026-04-10 15:20:00 | 4868.00 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2026-04-15 10:35:00 | 4923.20 | 2026-04-15 10:50:00 | 4908.82 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-16 10:35:00 | 4979.40 | 2026-04-16 10:40:00 | 4963.34 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-17 10:10:00 | 4934.10 | 2026-04-17 10:20:00 | 4921.94 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-21 10:15:00 | 4875.20 | 2026-04-21 10:30:00 | 4885.39 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-04-29 10:45:00 | 4783.30 | 2026-04-29 13:10:00 | 4771.03 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-04-29 10:45:00 | 4783.30 | 2026-04-29 15:20:00 | 4760.10 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-30 09:50:00 | 4712.30 | 2026-04-30 11:10:00 | 4724.03 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-05-04 09:40:00 | 4767.60 | 2026-05-04 10:15:00 | 4753.05 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-07 09:50:00 | 4752.10 | 2026-05-07 10:05:00 | 4764.10 | STOP_HIT | 1.00 | -0.25% |
