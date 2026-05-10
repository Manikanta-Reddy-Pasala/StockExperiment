# UltraTech Cement Ltd. (ULTRACEMCO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 11930.00
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 4
- **Avg / median % per leg:** 0.10% / -0.17%
- **Sum % (uncompounded):** 2.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.06% | 0.6% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.06% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 2 | 7 | 2 | 0.14% | 1.5% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 2 | 7 | 2 | 0.14% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 7 | 33.3% | 3 | 14 | 4 | 0.10% | 2.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 12876.00 | 12827.84 | 0.00 | ORB-long ORB[12722.00,12823.00] vol=7.7x ATR=46.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:55:00 | 12945.38 | 12867.96 | 0.00 | T1 1.5R @ 12945.38 |
| Target hit | 2026-02-09 15:20:00 | 13041.00 | 12940.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:15:00 | 12995.00 | 13037.98 | 0.00 | ORB-short ORB[13010.00,13110.00] vol=4.4x ATR=25.48 |
| Stop hit — per-position SL triggered | 2026-02-10 12:05:00 | 13020.48 | 13026.09 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 12943.00 | 12973.25 | 0.00 | ORB-short ORB[12955.00,13087.00] vol=2.6x ATR=20.51 |
| Stop hit — per-position SL triggered | 2026-02-11 11:20:00 | 12963.51 | 12971.16 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:00:00 | 12986.00 | 12956.34 | 0.00 | ORB-long ORB[12912.00,12969.00] vol=1.5x ATR=21.70 |
| Stop hit — per-position SL triggered | 2026-02-12 10:10:00 | 12964.30 | 12957.80 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:35:00 | 12893.00 | 12924.18 | 0.00 | ORB-short ORB[12918.00,13025.00] vol=2.2x ATR=28.65 |
| Stop hit — per-position SL triggered | 2026-02-13 11:00:00 | 12921.65 | 12918.09 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:25:00 | 12989.00 | 13010.51 | 0.00 | ORB-short ORB[13008.00,13059.00] vol=1.5x ATR=17.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:00:00 | 12962.57 | 13000.31 | 0.00 | T1 1.5R @ 12962.57 |
| Target hit | 2026-02-19 15:20:00 | 12669.00 | 12774.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 12758.00 | 12690.40 | 0.00 | ORB-long ORB[12578.00,12727.00] vol=2.0x ATR=29.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:45:00 | 12801.76 | 12707.55 | 0.00 | T1 1.5R @ 12801.76 |
| Stop hit — per-position SL triggered | 2026-02-20 12:05:00 | 12758.00 | 12714.40 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:00:00 | 12980.00 | 12943.47 | 0.00 | ORB-long ORB[12910.00,12972.00] vol=2.2x ATR=20.37 |
| Stop hit — per-position SL triggered | 2026-02-24 11:45:00 | 12959.63 | 12956.48 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 12043.00 | 12113.17 | 0.00 | ORB-short ORB[12073.00,12208.00] vol=2.4x ATR=33.27 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 12076.27 | 12107.23 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:20:00 | 11577.00 | 11596.18 | 0.00 | ORB-short ORB[11581.00,11663.00] vol=3.9x ATR=27.72 |
| Stop hit — per-position SL triggered | 2026-03-11 11:00:00 | 11604.72 | 11594.54 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:20:00 | 11238.00 | 11293.51 | 0.00 | ORB-short ORB[11251.00,11405.00] vol=2.2x ATR=33.51 |
| Stop hit — per-position SL triggered | 2026-03-12 10:35:00 | 11271.51 | 11282.44 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:30:00 | 10840.00 | 10871.83 | 0.00 | ORB-short ORB[10843.00,11000.00] vol=2.1x ATR=43.37 |
| Stop hit — per-position SL triggered | 2026-03-13 09:40:00 | 10883.37 | 10867.68 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:30:00 | 10719.00 | 10687.28 | 0.00 | ORB-long ORB[10540.00,10693.00] vol=2.9x ATR=36.89 |
| Stop hit — per-position SL triggered | 2026-04-06 11:00:00 | 10682.11 | 10695.03 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:15:00 | 12030.00 | 11950.43 | 0.00 | ORB-long ORB[11833.00,11940.00] vol=5.1x ATR=21.54 |
| Stop hit — per-position SL triggered | 2026-04-29 11:25:00 | 12008.46 | 11955.78 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 11749.00 | 11685.89 | 0.00 | ORB-long ORB[11612.00,11746.00] vol=3.8x ATR=33.56 |
| Stop hit — per-position SL triggered | 2026-05-04 12:10:00 | 11715.44 | 11700.50 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 11795.00 | 11742.47 | 0.00 | ORB-long ORB[11624.00,11768.00] vol=1.6x ATR=51.28 |
| Stop hit — per-position SL triggered | 2026-05-05 10:05:00 | 11743.72 | 11752.60 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 11991.00 | 12047.26 | 0.00 | ORB-short ORB[12030.00,12110.00] vol=2.0x ATR=30.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:00:00 | 11945.38 | 12014.00 | 0.00 | T1 1.5R @ 11945.38 |
| Target hit | 2026-05-08 14:05:00 | 11957.00 | 11949.61 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 12876.00 | 2026-02-09 12:55:00 | 12945.38 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-09 10:40:00 | 12876.00 | 2026-02-09 15:20:00 | 13041.00 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2026-02-10 11:15:00 | 12995.00 | 2026-02-10 12:05:00 | 13020.48 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-11 11:00:00 | 12943.00 | 2026-02-11 11:20:00 | 12963.51 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-02-12 10:00:00 | 12986.00 | 2026-02-12 10:10:00 | 12964.30 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-13 10:35:00 | 12893.00 | 2026-02-13 11:00:00 | 12921.65 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-19 10:25:00 | 12989.00 | 2026-02-19 11:00:00 | 12962.57 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2026-02-19 10:25:00 | 12989.00 | 2026-02-19 15:20:00 | 12669.00 | TARGET_HIT | 0.50 | 2.46% |
| BUY | retest1 | 2026-02-20 10:55:00 | 12758.00 | 2026-02-20 11:45:00 | 12801.76 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-20 10:55:00 | 12758.00 | 2026-02-20 12:05:00 | 12758.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 11:00:00 | 12980.00 | 2026-02-24 11:45:00 | 12959.63 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2026-03-06 10:55:00 | 12043.00 | 2026-03-06 11:00:00 | 12076.27 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-11 10:20:00 | 11577.00 | 2026-03-11 11:00:00 | 11604.72 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-12 10:20:00 | 11238.00 | 2026-03-12 10:35:00 | 11271.51 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-13 09:30:00 | 10840.00 | 2026-03-13 09:40:00 | 10883.37 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-06 10:30:00 | 10719.00 | 2026-04-06 11:00:00 | 10682.11 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-29 11:15:00 | 12030.00 | 2026-04-29 11:25:00 | 12008.46 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-05-04 11:10:00 | 11749.00 | 2026-05-04 12:10:00 | 11715.44 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-05 09:35:00 | 11795.00 | 2026-05-05 10:05:00 | 11743.72 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-05-08 09:35:00 | 11991.00 | 2026-05-08 10:00:00 | 11945.38 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-05-08 09:35:00 | 11991.00 | 2026-05-08 14:05:00 | 11957.00 | TARGET_HIT | 0.50 | 0.28% |
