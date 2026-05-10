# Indian Bank (INDIANB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 862.50
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
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 13
- **Target hits / Stop hits / Partials:** 4 / 13 / 5
- **Avg / median % per leg:** 0.21% / -0.25%
- **Sum % (uncompounded):** 4.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 8 | 50.0% | 4 | 8 | 4 | 0.36% | 5.7% |
| BUY @ 2nd Alert (retest1) | 16 | 8 | 50.0% | 4 | 8 | 4 | 0.36% | 5.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.20% | -1.2% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.20% | -1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 9 | 40.9% | 4 | 13 | 5 | 0.21% | 4.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 911.50 | 906.09 | 0.00 | ORB-long ORB[897.80,907.25] vol=3.1x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-02-10 11:20:00 | 909.22 | 906.78 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:45:00 | 880.85 | 874.40 | 0.00 | ORB-long ORB[864.15,876.30] vol=1.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2026-02-13 10:50:00 | 877.97 | 874.56 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 873.00 | 868.54 | 0.00 | ORB-long ORB[863.10,870.45] vol=2.2x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:00:00 | 877.47 | 872.15 | 0.00 | T1 1.5R @ 877.47 |
| Target hit | 2026-02-16 15:20:00 | 889.90 | 878.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:55:00 | 903.15 | 893.76 | 0.00 | ORB-long ORB[881.00,890.10] vol=2.2x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:50:00 | 907.72 | 897.35 | 0.00 | T1 1.5R @ 907.72 |
| Target hit | 2026-02-17 15:20:00 | 920.35 | 909.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:10:00 | 933.50 | 927.20 | 0.00 | ORB-long ORB[920.45,928.00] vol=2.1x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:25:00 | 937.43 | 928.02 | 0.00 | T1 1.5R @ 937.43 |
| Target hit | 2026-02-18 15:20:00 | 936.95 | 933.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:35:00 | 943.10 | 935.54 | 0.00 | ORB-long ORB[924.00,937.50] vol=1.8x ATR=3.75 |
| Stop hit — per-position SL triggered | 2026-02-20 11:20:00 | 939.35 | 940.89 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:30:00 | 988.05 | 981.87 | 0.00 | ORB-long ORB[976.15,986.40] vol=2.0x ATR=2.97 |
| Stop hit — per-position SL triggered | 2026-02-24 10:45:00 | 985.08 | 982.36 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 975.65 | 983.24 | 0.00 | ORB-short ORB[982.05,990.40] vol=3.1x ATR=2.97 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 978.62 | 981.06 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 926.60 | 931.20 | 0.00 | ORB-short ORB[928.20,935.00] vol=2.1x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:20:00 | 922.78 | 930.46 | 0.00 | T1 1.5R @ 922.78 |
| Stop hit — per-position SL triggered | 2026-03-11 11:45:00 | 926.60 | 929.80 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:55:00 | 856.40 | 863.03 | 0.00 | ORB-short ORB[858.15,870.15] vol=1.8x ATR=4.23 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 860.63 | 862.35 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:25:00 | 864.80 | 871.91 | 0.00 | ORB-short ORB[872.05,881.60] vol=2.0x ATR=4.67 |
| Stop hit — per-position SL triggered | 2026-04-01 11:25:00 | 869.47 | 870.86 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:30:00 | 968.30 | 959.76 | 0.00 | ORB-long ORB[948.55,962.60] vol=1.8x ATR=4.16 |
| Stop hit — per-position SL triggered | 2026-04-09 10:35:00 | 964.14 | 960.44 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 940.60 | 935.46 | 0.00 | ORB-long ORB[931.05,937.30] vol=1.6x ATR=2.83 |
| Stop hit — per-position SL triggered | 2026-04-21 09:35:00 | 937.77 | 935.84 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 930.60 | 926.97 | 0.00 | ORB-long ORB[921.05,930.15] vol=5.3x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-04-22 11:10:00 | 928.16 | 927.13 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:05:00 | 907.15 | 913.01 | 0.00 | ORB-short ORB[911.20,923.50] vol=2.1x ATR=2.35 |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 909.50 | 912.27 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:25:00 | 915.00 | 904.59 | 0.00 | ORB-long ORB[893.60,907.25] vol=2.0x ATR=3.13 |
| Stop hit — per-position SL triggered | 2026-04-29 10:30:00 | 911.87 | 904.97 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 828.80 | 823.74 | 0.00 | ORB-long ORB[818.35,827.85] vol=2.8x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 12:10:00 | 832.57 | 826.00 | 0.00 | T1 1.5R @ 832.57 |
| Target hit | 2026-05-05 15:20:00 | 847.05 | 836.25 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 11:00:00 | 911.50 | 2026-02-10 11:20:00 | 909.22 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-13 10:45:00 | 880.85 | 2026-02-13 10:50:00 | 877.97 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-16 09:30:00 | 873.00 | 2026-02-16 11:00:00 | 877.47 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-16 09:30:00 | 873.00 | 2026-02-16 15:20:00 | 889.90 | TARGET_HIT | 0.50 | 1.94% |
| BUY | retest1 | 2026-02-17 10:55:00 | 903.15 | 2026-02-17 11:50:00 | 907.72 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-17 10:55:00 | 903.15 | 2026-02-17 15:20:00 | 920.35 | TARGET_HIT | 0.50 | 1.90% |
| BUY | retest1 | 2026-02-18 11:10:00 | 933.50 | 2026-02-18 11:25:00 | 937.43 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-18 11:10:00 | 933.50 | 2026-02-18 15:20:00 | 936.95 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-20 09:35:00 | 943.10 | 2026-02-20 11:20:00 | 939.35 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-24 10:30:00 | 988.05 | 2026-02-24 10:45:00 | 985.08 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-25 10:45:00 | 975.65 | 2026-02-25 11:20:00 | 978.62 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-11 11:10:00 | 926.60 | 2026-03-11 11:20:00 | 922.78 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-11 11:10:00 | 926.60 | 2026-03-11 11:45:00 | 926.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:55:00 | 856.40 | 2026-03-16 11:15:00 | 860.63 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-01 10:25:00 | 864.80 | 2026-04-01 11:25:00 | 869.47 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-04-09 10:30:00 | 968.30 | 2026-04-09 10:35:00 | 964.14 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-21 09:30:00 | 940.60 | 2026-04-21 09:35:00 | 937.77 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-22 10:55:00 | 930.60 | 2026-04-22 11:10:00 | 928.16 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-24 11:05:00 | 907.15 | 2026-04-24 11:15:00 | 909.50 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-29 10:25:00 | 915.00 | 2026-04-29 10:30:00 | 911.87 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-05 11:00:00 | 828.80 | 2026-05-05 12:10:00 | 832.57 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-05-05 11:00:00 | 828.80 | 2026-05-05 15:20:00 | 847.05 | TARGET_HIT | 0.50 | 2.20% |
