# PTC Industries Ltd. (PTCIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 16790.00
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 13
- **Target hits / Stop hits / Partials:** 3 / 13 / 6
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 2.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | -0.00% | -0.0% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | -0.00% | -0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.25% | 2.7% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.25% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 9 | 40.9% | 3 | 13 | 6 | 0.12% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 18425.00 | 18336.02 | 0.00 | ORB-long ORB[18234.00,18342.00] vol=2.6x ATR=51.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:05:00 | 18501.80 | 18387.17 | 0.00 | T1 1.5R @ 18501.80 |
| Stop hit — per-position SL triggered | 2026-02-12 10:25:00 | 18425.00 | 18391.08 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 18326.00 | 18191.16 | 0.00 | ORB-long ORB[17881.00,18142.00] vol=1.7x ATR=56.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:20:00 | 18410.47 | 18211.93 | 0.00 | T1 1.5R @ 18410.47 |
| Target hit | 2026-02-17 12:20:00 | 18355.00 | 18385.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 18230.00 | 18331.00 | 0.00 | ORB-short ORB[18287.00,18510.00] vol=1.6x ATR=38.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:30:00 | 18172.82 | 18292.47 | 0.00 | T1 1.5R @ 18172.82 |
| Stop hit — per-position SL triggered | 2026-02-18 12:25:00 | 18230.00 | 18234.67 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 17995.00 | 17953.84 | 0.00 | ORB-long ORB[17838.00,17940.00] vol=2.0x ATR=46.78 |
| Stop hit — per-position SL triggered | 2026-02-25 10:20:00 | 17948.22 | 17961.42 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:05:00 | 18232.00 | 18105.18 | 0.00 | ORB-long ORB[17950.00,18047.00] vol=1.6x ATR=46.32 |
| Stop hit — per-position SL triggered | 2026-02-26 10:10:00 | 18185.68 | 18114.65 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 17797.00 | 18065.17 | 0.00 | ORB-short ORB[17990.00,18250.00] vol=1.7x ATR=63.41 |
| Stop hit — per-position SL triggered | 2026-03-06 14:40:00 | 17860.41 | 17964.68 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:05:00 | 17529.00 | 17721.95 | 0.00 | ORB-short ORB[17755.00,17912.00] vol=1.9x ATR=49.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:50:00 | 17454.60 | 17671.33 | 0.00 | T1 1.5R @ 17454.60 |
| Target hit | 2026-03-13 15:20:00 | 17006.00 | 17195.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-03-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:30:00 | 17270.00 | 17097.19 | 0.00 | ORB-long ORB[16869.00,17098.00] vol=1.7x ATR=85.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 12:00:00 | 17397.85 | 17178.21 | 0.00 | T1 1.5R @ 17397.85 |
| Stop hit — per-position SL triggered | 2026-03-17 14:00:00 | 17270.00 | 17315.82 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 11:05:00 | 17240.00 | 17354.69 | 0.00 | ORB-short ORB[17360.00,17561.00] vol=2.0x ATR=44.33 |
| Stop hit — per-position SL triggered | 2026-03-20 11:45:00 | 17284.33 | 17328.80 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:40:00 | 15691.00 | 15877.06 | 0.00 | ORB-short ORB[15863.00,15963.00] vol=4.3x ATR=52.14 |
| Stop hit — per-position SL triggered | 2026-04-17 11:25:00 | 15743.14 | 15814.39 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 16335.00 | 16252.66 | 0.00 | ORB-long ORB[16073.00,16226.00] vol=1.8x ATR=58.61 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 16276.39 | 16254.65 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:30:00 | 15886.00 | 16000.90 | 0.00 | ORB-short ORB[15964.00,16126.00] vol=1.8x ATR=47.07 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 15933.07 | 15976.96 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:00:00 | 15947.00 | 15996.89 | 0.00 | ORB-short ORB[15999.00,16195.00] vol=3.7x ATR=58.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:05:00 | 15859.96 | 15982.94 | 0.00 | T1 1.5R @ 15859.96 |
| Target hit | 2026-04-24 14:40:00 | 15940.00 | 15812.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — BUY (started 2026-05-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:50:00 | 16416.00 | 16263.86 | 0.00 | ORB-long ORB[16068.00,16307.00] vol=2.1x ATR=85.40 |
| Stop hit — per-position SL triggered | 2026-05-04 10:10:00 | 16330.60 | 16289.96 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:10:00 | 17021.00 | 17088.79 | 0.00 | ORB-short ORB[17068.00,17241.00] vol=1.8x ATR=61.15 |
| Stop hit — per-position SL triggered | 2026-05-07 10:50:00 | 17082.15 | 17076.28 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 17028.00 | 16924.35 | 0.00 | ORB-long ORB[16715.00,16950.00] vol=2.5x ATR=68.55 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 16959.45 | 16965.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 09:35:00 | 18425.00 | 2026-02-12 10:05:00 | 18501.80 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-12 09:35:00 | 18425.00 | 2026-02-12 10:25:00 | 18425.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 11:05:00 | 18326.00 | 2026-02-17 11:20:00 | 18410.47 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-17 11:05:00 | 18326.00 | 2026-02-17 12:20:00 | 18355.00 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2026-02-18 11:15:00 | 18230.00 | 2026-02-18 11:30:00 | 18172.82 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-18 11:15:00 | 18230.00 | 2026-02-18 12:25:00 | 18230.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:55:00 | 17995.00 | 2026-02-25 10:20:00 | 17948.22 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-26 10:05:00 | 18232.00 | 2026-02-26 10:10:00 | 18185.68 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-06 10:55:00 | 17797.00 | 2026-03-06 14:40:00 | 17860.41 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-03-13 11:05:00 | 17529.00 | 2026-03-13 11:50:00 | 17454.60 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-13 11:05:00 | 17529.00 | 2026-03-13 15:20:00 | 17006.00 | TARGET_HIT | 0.50 | 2.98% |
| BUY | retest1 | 2026-03-17 10:30:00 | 17270.00 | 2026-03-17 12:00:00 | 17397.85 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-03-17 10:30:00 | 17270.00 | 2026-03-17 14:00:00 | 17270.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 11:05:00 | 17240.00 | 2026-03-20 11:45:00 | 17284.33 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-17 10:40:00 | 15691.00 | 2026-04-17 11:25:00 | 15743.14 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-21 09:45:00 | 16335.00 | 2026-04-21 09:50:00 | 16276.39 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-22 10:30:00 | 15886.00 | 2026-04-22 11:05:00 | 15933.07 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-24 10:00:00 | 15947.00 | 2026-04-24 10:05:00 | 15859.96 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-24 10:00:00 | 15947.00 | 2026-04-24 14:40:00 | 15940.00 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2026-05-04 09:50:00 | 16416.00 | 2026-05-04 10:10:00 | 16330.60 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-05-07 10:10:00 | 17021.00 | 2026-05-07 10:50:00 | 17082.15 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-08 09:40:00 | 17028.00 | 2026-05-08 10:10:00 | 16959.45 | STOP_HIT | 1.00 | -0.40% |
