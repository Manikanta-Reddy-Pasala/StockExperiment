# Navin Fluorine International Ltd. (NAVINFLUOR)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 7039.50
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
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 5
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 2.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.09% | 1.1% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.09% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.17% | 1.2% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.17% | 1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 8 | 40.0% | 3 | 12 | 5 | 0.12% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 6262.00 | 6308.62 | 0.00 | ORB-short ORB[6270.00,6348.00] vol=1.5x ATR=19.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:35:00 | 6233.22 | 6303.26 | 0.00 | T1 1.5R @ 6233.22 |
| Stop hit — per-position SL triggered | 2026-02-16 13:25:00 | 6262.00 | 6286.01 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 6302.00 | 6270.84 | 0.00 | ORB-long ORB[6232.50,6287.50] vol=2.1x ATR=19.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:50:00 | 6330.99 | 6284.56 | 0.00 | T1 1.5R @ 6330.99 |
| Target hit | 2026-02-18 15:00:00 | 6382.50 | 6388.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 6459.00 | 6397.61 | 0.00 | ORB-long ORB[6350.00,6433.00] vol=2.6x ATR=19.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:15:00 | 6487.75 | 6430.80 | 0.00 | T1 1.5R @ 6487.75 |
| Target hit | 2026-02-20 15:20:00 | 6523.00 | 6489.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:05:00 | 6497.00 | 6561.77 | 0.00 | ORB-short ORB[6545.00,6632.00] vol=1.6x ATR=22.63 |
| Stop hit — per-position SL triggered | 2026-02-23 10:10:00 | 6519.63 | 6556.50 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 6496.50 | 6460.20 | 0.00 | ORB-long ORB[6409.00,6468.00] vol=3.1x ATR=20.94 |
| Stop hit — per-position SL triggered | 2026-02-26 09:35:00 | 6475.56 | 6461.93 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:35:00 | 6585.00 | 6538.14 | 0.00 | ORB-long ORB[6480.00,6571.00] vol=2.0x ATR=24.61 |
| Stop hit — per-position SL triggered | 2026-03-10 11:15:00 | 6560.39 | 6559.23 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:30:00 | 6062.50 | 6177.76 | 0.00 | ORB-short ORB[6169.00,6260.00] vol=2.4x ATR=27.16 |
| Stop hit — per-position SL triggered | 2026-03-13 10:35:00 | 6089.66 | 6173.15 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 11:15:00 | 6217.00 | 6184.43 | 0.00 | ORB-long ORB[6100.00,6189.50] vol=1.6x ATR=21.21 |
| Stop hit — per-position SL triggered | 2026-03-16 11:30:00 | 6195.79 | 6185.67 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:20:00 | 6277.50 | 6252.60 | 0.00 | ORB-long ORB[6180.00,6233.00] vol=3.3x ATR=21.05 |
| Stop hit — per-position SL triggered | 2026-03-17 12:10:00 | 6256.45 | 6259.63 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:35:00 | 6321.50 | 6296.94 | 0.00 | ORB-long ORB[6258.50,6318.00] vol=1.5x ATR=16.74 |
| Stop hit — per-position SL triggered | 2026-04-17 11:00:00 | 6304.76 | 6302.43 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:05:00 | 6432.50 | 6528.10 | 0.00 | ORB-short ORB[6505.50,6600.00] vol=1.7x ATR=29.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:25:00 | 6388.68 | 6488.93 | 0.00 | T1 1.5R @ 6388.68 |
| Target hit | 2026-04-21 15:20:00 | 6362.50 | 6440.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-04-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:20:00 | 6405.50 | 6376.45 | 0.00 | ORB-long ORB[6335.00,6400.00] vol=2.1x ATR=20.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:30:00 | 6436.65 | 6383.44 | 0.00 | T1 1.5R @ 6436.65 |
| Stop hit — per-position SL triggered | 2026-04-22 10:45:00 | 6405.50 | 6386.88 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:40:00 | 6409.00 | 6346.41 | 0.00 | ORB-long ORB[6262.00,6350.00] vol=5.0x ATR=28.54 |
| Stop hit — per-position SL triggered | 2026-04-24 09:50:00 | 6380.46 | 6367.69 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 6910.50 | 6849.73 | 0.00 | ORB-long ORB[6765.00,6848.00] vol=2.4x ATR=30.01 |
| Stop hit — per-position SL triggered | 2026-05-05 09:40:00 | 6880.49 | 6863.87 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 7009.50 | 7021.32 | 0.00 | ORB-short ORB[7010.00,7071.00] vol=1.7x ATR=16.03 |
| Stop hit — per-position SL triggered | 2026-05-08 09:40:00 | 7025.53 | 7024.20 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-16 11:15:00 | 6262.00 | 2026-02-16 11:35:00 | 6233.22 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-16 11:15:00 | 6262.00 | 2026-02-16 13:25:00 | 6262.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:40:00 | 6302.00 | 2026-02-18 09:50:00 | 6330.99 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-18 09:40:00 | 6302.00 | 2026-02-18 15:00:00 | 6382.50 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2026-02-20 10:45:00 | 6459.00 | 2026-02-20 11:15:00 | 6487.75 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-20 10:45:00 | 6459.00 | 2026-02-20 15:20:00 | 6523.00 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2026-02-23 10:05:00 | 6497.00 | 2026-02-23 10:10:00 | 6519.63 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-26 09:30:00 | 6496.50 | 2026-02-26 09:35:00 | 6475.56 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-10 10:35:00 | 6585.00 | 2026-03-10 11:15:00 | 6560.39 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-13 10:30:00 | 6062.50 | 2026-03-13 10:35:00 | 6089.66 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-16 11:15:00 | 6217.00 | 2026-03-16 11:30:00 | 6195.79 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-17 10:20:00 | 6277.50 | 2026-03-17 12:10:00 | 6256.45 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-17 10:35:00 | 6321.50 | 2026-04-17 11:00:00 | 6304.76 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-21 10:05:00 | 6432.50 | 2026-04-21 11:25:00 | 6388.68 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-04-21 10:05:00 | 6432.50 | 2026-04-21 15:20:00 | 6362.50 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2026-04-22 10:20:00 | 6405.50 | 2026-04-22 10:30:00 | 6436.65 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-22 10:20:00 | 6405.50 | 2026-04-22 10:45:00 | 6405.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 09:40:00 | 6409.00 | 2026-04-24 09:50:00 | 6380.46 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-05 09:30:00 | 6910.50 | 2026-05-05 09:40:00 | 6880.49 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-05-08 09:30:00 | 7009.50 | 2026-05-08 09:40:00 | 7025.53 | STOP_HIT | 1.00 | -0.23% |
