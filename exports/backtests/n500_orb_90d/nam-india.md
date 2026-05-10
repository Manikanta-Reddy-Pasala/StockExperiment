# Nippon Life India Asset Management Ltd. (NAM-INDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1100.20
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 7
- **Target hits / Stop hits / Partials:** 5 / 7 / 5
- **Avg / median % per leg:** 0.57% / 0.48%
- **Sum % (uncompounded):** 9.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 8 | 66.7% | 4 | 4 | 4 | 0.70% | 8.4% |
| BUY @ 2nd Alert (retest1) | 12 | 8 | 66.7% | 4 | 4 | 4 | 0.70% | 8.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.25% | 1.2% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.25% | 1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 10 | 58.8% | 5 | 7 | 5 | 0.57% | 9.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 921.00 | 917.31 | 0.00 | ORB-long ORB[909.00,919.70] vol=1.6x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:20:00 | 925.76 | 919.87 | 0.00 | T1 1.5R @ 925.76 |
| Target hit | 2026-02-11 15:20:00 | 950.00 | 940.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:40:00 | 957.00 | 952.87 | 0.00 | ORB-long ORB[942.00,955.95] vol=1.8x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:25:00 | 963.04 | 954.93 | 0.00 | T1 1.5R @ 963.04 |
| Target hit | 2026-02-12 13:55:00 | 958.55 | 962.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:15:00 | 954.50 | 948.64 | 0.00 | ORB-long ORB[940.00,947.95] vol=1.8x ATR=3.49 |
| Stop hit — per-position SL triggered | 2026-02-20 11:25:00 | 951.01 | 948.98 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 959.15 | 965.51 | 0.00 | ORB-short ORB[964.00,975.00] vol=1.6x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:15:00 | 954.56 | 963.76 | 0.00 | T1 1.5R @ 954.56 |
| Target hit | 2026-02-26 15:20:00 | 942.80 | 946.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:15:00 | 929.30 | 932.34 | 0.00 | ORB-short ORB[931.00,942.60] vol=1.6x ATR=2.59 |
| Stop hit — per-position SL triggered | 2026-02-27 11:45:00 | 931.89 | 932.20 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:10:00 | 852.15 | 859.07 | 0.00 | ORB-short ORB[857.00,865.45] vol=2.1x ATR=2.66 |
| Stop hit — per-position SL triggered | 2026-03-06 11:55:00 | 854.81 | 858.10 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 11:10:00 | 829.65 | 827.44 | 0.00 | ORB-long ORB[815.10,827.55] vol=1.6x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-03-16 11:50:00 | 826.08 | 827.84 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:00:00 | 840.35 | 835.85 | 0.00 | ORB-long ORB[829.85,836.90] vol=3.0x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:15:00 | 845.64 | 837.59 | 0.00 | T1 1.5R @ 845.64 |
| Target hit | 2026-03-17 15:20:00 | 846.85 | 845.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:00:00 | 822.75 | 826.90 | 0.00 | ORB-short ORB[827.25,835.90] vol=1.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2026-03-24 11:05:00 | 825.62 | 826.77 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 970.00 | 965.26 | 0.00 | ORB-long ORB[958.65,966.70] vol=3.5x ATR=2.92 |
| Stop hit — per-position SL triggered | 2026-04-16 11:20:00 | 967.08 | 965.33 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:45:00 | 981.55 | 973.76 | 0.00 | ORB-long ORB[962.00,976.60] vol=1.8x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 09:55:00 | 987.55 | 976.41 | 0.00 | T1 1.5R @ 987.55 |
| Target hit | 2026-04-17 15:20:00 | 1015.65 | 998.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1073.80 | 1065.93 | 0.00 | ORB-long ORB[1058.40,1071.80] vol=1.9x ATR=4.31 |
| Stop hit — per-position SL triggered | 2026-05-06 10:45:00 | 1069.49 | 1067.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 09:35:00 | 921.00 | 2026-02-11 10:20:00 | 925.76 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-11 09:35:00 | 921.00 | 2026-02-11 15:20:00 | 950.00 | TARGET_HIT | 0.50 | 3.15% |
| BUY | retest1 | 2026-02-12 09:40:00 | 957.00 | 2026-02-12 10:25:00 | 963.04 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-02-12 09:40:00 | 957.00 | 2026-02-12 13:55:00 | 958.55 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-02-20 11:15:00 | 954.50 | 2026-02-20 11:25:00 | 951.01 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-26 10:50:00 | 959.15 | 2026-02-26 11:15:00 | 954.56 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-26 10:50:00 | 959.15 | 2026-02-26 15:20:00 | 942.80 | TARGET_HIT | 0.50 | 1.70% |
| SELL | retest1 | 2026-02-27 11:15:00 | 929.30 | 2026-02-27 11:45:00 | 931.89 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-06 11:10:00 | 852.15 | 2026-03-06 11:55:00 | 854.81 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-16 11:10:00 | 829.65 | 2026-03-16 11:50:00 | 826.08 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-17 10:00:00 | 840.35 | 2026-03-17 10:15:00 | 845.64 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-03-17 10:00:00 | 840.35 | 2026-03-17 15:20:00 | 846.85 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2026-03-24 11:00:00 | 822.75 | 2026-03-24 11:05:00 | 825.62 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-16 11:15:00 | 970.00 | 2026-04-16 11:20:00 | 967.08 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-17 09:45:00 | 981.55 | 2026-04-17 09:55:00 | 987.55 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-17 09:45:00 | 981.55 | 2026-04-17 15:20:00 | 1015.65 | TARGET_HIT | 0.50 | 3.47% |
| BUY | retest1 | 2026-05-06 10:15:00 | 1073.80 | 2026-05-06 10:45:00 | 1069.49 | STOP_HIT | 1.00 | -0.40% |
