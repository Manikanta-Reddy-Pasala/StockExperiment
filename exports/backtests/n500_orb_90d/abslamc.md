# Aditya Birla Sun Life AMC Ltd. (ABSLAMC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1075.00
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 14 |
| TARGET_HIT | 9 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 10
- **Target hits / Stop hits / Partials:** 9 / 10 / 14
- **Avg / median % per leg:** 0.69% / 0.52%
- **Sum % (uncompounded):** 22.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 19 | 82.6% | 9 | 4 | 10 | 0.92% | 21.1% |
| BUY @ 2nd Alert (retest1) | 23 | 19 | 82.6% | 9 | 4 | 10 | 0.92% | 21.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 0 | 6 | 4 | 0.17% | 1.7% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 0 | 6 | 4 | 0.17% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 23 | 69.7% | 9 | 10 | 14 | 0.69% | 22.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:10:00 | 819.45 | 814.74 | 0.00 | ORB-long ORB[810.20,817.80] vol=1.7x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:25:00 | 822.65 | 817.05 | 0.00 | T1 1.5R @ 822.65 |
| Stop hit — per-position SL triggered | 2026-02-09 11:30:00 | 819.45 | 817.07 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 821.25 | 817.83 | 0.00 | ORB-long ORB[811.40,820.55] vol=3.5x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:05:00 | 825.50 | 820.75 | 0.00 | T1 1.5R @ 825.50 |
| Target hit | 2026-02-10 15:20:00 | 838.00 | 834.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:50:00 | 867.60 | 862.37 | 0.00 | ORB-long ORB[856.00,864.55] vol=2.6x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:05:00 | 872.61 | 864.60 | 0.00 | T1 1.5R @ 872.61 |
| Target hit | 2026-02-12 12:45:00 | 870.05 | 870.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:30:00 | 889.45 | 885.67 | 0.00 | ORB-long ORB[875.00,888.00] vol=3.9x ATR=2.32 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 887.13 | 885.88 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 897.10 | 890.58 | 0.00 | ORB-long ORB[882.40,889.35] vol=3.1x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:45:00 | 901.97 | 896.27 | 0.00 | T1 1.5R @ 901.97 |
| Target hit | 2026-02-18 10:25:00 | 898.65 | 898.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 905.35 | 907.80 | 0.00 | ORB-short ORB[906.00,913.75] vol=2.0x ATR=3.92 |
| Stop hit — per-position SL triggered | 2026-02-19 11:35:00 | 909.27 | 907.78 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:20:00 | 903.00 | 895.70 | 0.00 | ORB-long ORB[887.55,895.60] vol=1.5x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:55:00 | 909.24 | 898.37 | 0.00 | T1 1.5R @ 909.24 |
| Target hit | 2026-02-20 12:45:00 | 912.30 | 912.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 914.50 | 924.07 | 0.00 | ORB-short ORB[921.00,933.90] vol=1.6x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 12:05:00 | 909.28 | 922.52 | 0.00 | T1 1.5R @ 909.28 |
| Stop hit — per-position SL triggered | 2026-02-23 12:55:00 | 914.50 | 921.44 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 931.50 | 925.80 | 0.00 | ORB-long ORB[917.35,928.00] vol=1.7x ATR=3.23 |
| Stop hit — per-position SL triggered | 2026-02-25 09:55:00 | 928.27 | 927.46 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 923.10 | 920.86 | 0.00 | ORB-long ORB[915.00,922.80] vol=2.2x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:55:00 | 926.33 | 922.81 | 0.00 | T1 1.5R @ 926.33 |
| Target hit | 2026-02-26 11:55:00 | 925.30 | 925.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:50:00 | 898.40 | 905.34 | 0.00 | ORB-short ORB[904.25,917.75] vol=1.6x ATR=3.22 |
| Stop hit — per-position SL triggered | 2026-02-27 10:05:00 | 901.62 | 904.03 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 889.70 | 880.75 | 0.00 | ORB-long ORB[867.95,879.60] vol=1.5x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:30:00 | 895.51 | 885.78 | 0.00 | T1 1.5R @ 895.51 |
| Target hit | 2026-03-05 11:20:00 | 891.85 | 892.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2026-03-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:55:00 | 893.05 | 887.61 | 0.00 | ORB-long ORB[883.05,892.00] vol=1.7x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:15:00 | 900.08 | 892.08 | 0.00 | T1 1.5R @ 900.08 |
| Target hit | 2026-03-10 13:35:00 | 987.00 | 1000.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2026-03-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:55:00 | 919.70 | 929.04 | 0.00 | ORB-short ORB[926.15,939.90] vol=2.5x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:45:00 | 913.12 | 925.77 | 0.00 | T1 1.5R @ 913.12 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 919.70 | 924.90 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-03-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:55:00 | 904.20 | 915.91 | 0.00 | ORB-short ORB[916.00,928.00] vol=1.7x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:00:00 | 899.34 | 913.25 | 0.00 | T1 1.5R @ 899.34 |
| Stop hit — per-position SL triggered | 2026-03-27 11:35:00 | 904.20 | 907.71 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:40:00 | 959.90 | 953.85 | 0.00 | ORB-long ORB[948.15,956.90] vol=2.9x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:00:00 | 967.19 | 957.26 | 0.00 | T1 1.5R @ 967.19 |
| Target hit | 2026-04-08 15:05:00 | 971.45 | 972.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2026-04-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:40:00 | 1033.90 | 1038.55 | 0.00 | ORB-short ORB[1036.10,1044.40] vol=7.1x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:10:00 | 1026.68 | 1037.33 | 0.00 | T1 1.5R @ 1026.68 |
| Stop hit — per-position SL triggered | 2026-04-22 13:10:00 | 1033.90 | 1035.21 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 10:45:00 | 1082.80 | 1066.36 | 0.00 | ORB-long ORB[1032.50,1048.85] vol=3.9x ATR=8.50 |
| Stop hit — per-position SL triggered | 2026-04-24 10:50:00 | 1074.30 | 1067.24 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 1061.10 | 1046.99 | 0.00 | ORB-long ORB[1030.00,1044.70] vol=2.0x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:55:00 | 1069.40 | 1059.76 | 0.00 | T1 1.5R @ 1069.40 |
| Target hit | 2026-05-05 10:30:00 | 1068.60 | 1069.31 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:10:00 | 819.45 | 2026-02-09 11:25:00 | 822.65 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-09 11:10:00 | 819.45 | 2026-02-09 11:30:00 | 819.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 09:45:00 | 821.25 | 2026-02-10 10:05:00 | 825.50 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-10 09:45:00 | 821.25 | 2026-02-10 15:20:00 | 838.00 | TARGET_HIT | 0.50 | 2.04% |
| BUY | retest1 | 2026-02-12 09:50:00 | 867.60 | 2026-02-12 10:05:00 | 872.61 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-12 09:50:00 | 867.60 | 2026-02-12 12:45:00 | 870.05 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2026-02-17 10:30:00 | 889.45 | 2026-02-17 10:40:00 | 887.13 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-18 09:40:00 | 897.10 | 2026-02-18 09:45:00 | 901.97 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-18 09:40:00 | 897.10 | 2026-02-18 10:25:00 | 898.65 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2026-02-19 11:15:00 | 905.35 | 2026-02-19 11:35:00 | 909.27 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-02-20 10:20:00 | 903.00 | 2026-02-20 10:55:00 | 909.24 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-02-20 10:20:00 | 903.00 | 2026-02-20 12:45:00 | 912.30 | TARGET_HIT | 0.50 | 1.03% |
| SELL | retest1 | 2026-02-23 10:55:00 | 914.50 | 2026-02-23 12:05:00 | 909.28 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-23 10:55:00 | 914.50 | 2026-02-23 12:55:00 | 914.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:35:00 | 931.50 | 2026-02-25 09:55:00 | 928.27 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-26 10:50:00 | 923.10 | 2026-02-26 10:55:00 | 926.33 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-26 10:50:00 | 923.10 | 2026-02-26 11:55:00 | 925.30 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-27 09:50:00 | 898.40 | 2026-02-27 10:05:00 | 901.62 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-05 10:15:00 | 889.70 | 2026-03-05 10:30:00 | 895.51 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-03-05 10:15:00 | 889.70 | 2026-03-05 11:20:00 | 891.85 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2026-03-10 09:55:00 | 893.05 | 2026-03-10 11:15:00 | 900.08 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-03-10 09:55:00 | 893.05 | 2026-03-10 13:35:00 | 987.00 | TARGET_HIT | 0.50 | 10.52% |
| SELL | retest1 | 2026-03-23 09:55:00 | 919.70 | 2026-03-23 10:45:00 | 913.12 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-03-23 09:55:00 | 919.70 | 2026-03-23 11:05:00 | 919.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 10:55:00 | 904.20 | 2026-03-27 11:00:00 | 899.34 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-27 10:55:00 | 904.20 | 2026-03-27 11:35:00 | 904.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 09:40:00 | 959.90 | 2026-04-08 10:00:00 | 967.19 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2026-04-08 09:40:00 | 959.90 | 2026-04-08 15:05:00 | 971.45 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2026-04-22 10:40:00 | 1033.90 | 2026-04-22 11:10:00 | 1026.68 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-22 10:40:00 | 1033.90 | 2026-04-22 13:10:00 | 1033.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 10:45:00 | 1082.80 | 2026-04-24 10:50:00 | 1074.30 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2026-05-05 09:50:00 | 1061.10 | 2026-05-05 09:55:00 | 1069.40 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-05-05 09:50:00 | 1061.10 | 2026-05-05 10:30:00 | 1068.60 | TARGET_HIT | 0.50 | 0.71% |
