# CRISIL Ltd. (CRISIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4160.70
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 4
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 0.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 2 | 3 | 0.33% | 2.0% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 2 | 3 | 0.33% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.16% | -1.1% |
| SELL @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.16% | -1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.07% | 0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:00:00 | 4615.30 | 4530.73 | 0.00 | ORB-long ORB[4398.70,4467.10] vol=1.6x ATR=26.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:15:00 | 4655.54 | 4574.22 | 0.00 | T1 1.5R @ 4655.54 |
| Stop hit — per-position SL triggered | 2026-02-16 11:25:00 | 4615.30 | 4594.30 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 4434.50 | 4452.31 | 0.00 | ORB-short ORB[4440.00,4482.60] vol=2.1x ATR=12.80 |
| Stop hit — per-position SL triggered | 2026-02-27 09:40:00 | 4447.30 | 4450.68 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 4282.00 | 4301.70 | 0.00 | ORB-short ORB[4299.50,4319.90] vol=3.6x ATR=10.07 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 4292.07 | 4300.61 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:00:00 | 4306.50 | 4280.31 | 0.00 | ORB-long ORB[4238.60,4298.90] vol=2.1x ATR=11.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 12:25:00 | 4323.42 | 4295.68 | 0.00 | T1 1.5R @ 4323.42 |
| Target hit | 2026-03-11 15:20:00 | 4313.90 | 4310.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-04-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-02 10:20:00 | 3691.90 | 3728.64 | 0.00 | ORB-short ORB[3721.20,3770.00] vol=2.2x ATR=14.53 |
| Stop hit — per-position SL triggered | 2026-04-02 11:55:00 | 3706.43 | 3713.16 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 4138.00 | 4160.47 | 0.00 | ORB-short ORB[4145.00,4191.00] vol=2.5x ATR=13.14 |
| Stop hit — per-position SL triggered | 2026-04-16 11:30:00 | 4151.14 | 4158.50 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:40:00 | 4204.80 | 4164.35 | 0.00 | ORB-long ORB[4103.50,4151.90] vol=6.2x ATR=15.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:45:00 | 4227.61 | 4182.34 | 0.00 | T1 1.5R @ 4227.61 |
| Stop hit — per-position SL triggered | 2026-04-17 12:00:00 | 4204.80 | 4187.92 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 4263.50 | 4284.99 | 0.00 | ORB-short ORB[4280.10,4317.50] vol=2.6x ATR=11.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 4245.87 | 4279.66 | 0.00 | T1 1.5R @ 4245.87 |
| Stop hit — per-position SL triggered | 2026-04-28 11:40:00 | 4263.50 | 4277.68 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 4154.50 | 4175.30 | 0.00 | ORB-short ORB[4200.00,4252.70] vol=3.0x ATR=11.86 |
| Stop hit — per-position SL triggered | 2026-05-05 11:35:00 | 4166.36 | 4173.84 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 10:00:00 | 4615.30 | 2026-02-16 10:15:00 | 4655.54 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2026-02-16 10:00:00 | 4615.30 | 2026-02-16 11:25:00 | 4615.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:30:00 | 4434.50 | 2026-02-27 09:40:00 | 4447.30 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-06 10:45:00 | 4282.00 | 2026-03-06 11:00:00 | 4292.07 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-11 11:00:00 | 4306.50 | 2026-03-11 12:25:00 | 4323.42 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-03-11 11:00:00 | 4306.50 | 2026-03-11 15:20:00 | 4313.90 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2026-04-02 10:20:00 | 3691.90 | 2026-04-02 11:55:00 | 3706.43 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-16 11:00:00 | 4138.00 | 2026-04-16 11:30:00 | 4151.14 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-17 10:40:00 | 4204.80 | 2026-04-17 11:45:00 | 4227.61 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-17 10:40:00 | 4204.80 | 2026-04-17 12:00:00 | 4204.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 11:00:00 | 4263.50 | 2026-04-28 11:25:00 | 4245.87 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-04-28 11:00:00 | 4263.50 | 2026-04-28 11:40:00 | 4263.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 11:00:00 | 4154.50 | 2026-05-05 11:35:00 | 4166.36 | STOP_HIT | 1.00 | -0.29% |
