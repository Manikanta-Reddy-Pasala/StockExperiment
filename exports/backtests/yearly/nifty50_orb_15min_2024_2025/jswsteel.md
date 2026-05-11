# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1272.00
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
| ENTRY1 | 89 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 16 |
| STOP_HIT | 73 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 73
- **Target hits / Stop hits / Partials:** 16 / 73 / 39
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 18.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 35 | 49.3% | 10 | 36 | 25 | 0.21% | 15.0% |
| BUY @ 2nd Alert (retest1) | 71 | 35 | 49.3% | 10 | 36 | 25 | 0.21% | 15.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 57 | 20 | 35.1% | 6 | 37 | 14 | 0.06% | 3.5% |
| SELL @ 2nd Alert (retest1) | 57 | 20 | 35.1% | 6 | 37 | 14 | 0.06% | 3.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 128 | 55 | 43.0% | 16 | 73 | 39 | 0.14% | 18.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 09:55:00 | 871.15 | 877.37 | 0.00 | ORB-short ORB[874.45,883.50] vol=2.1x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:15:00 | 866.75 | 874.62 | 0.00 | T1 1.5R @ 866.75 |
| Stop hit — per-position SL triggered | 2024-05-15 11:00:00 | 871.15 | 872.33 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 865.90 | 870.37 | 0.00 | ORB-short ORB[870.95,877.40] vol=1.6x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-05-16 11:25:00 | 868.37 | 870.16 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:25:00 | 890.75 | 886.57 | 0.00 | ORB-long ORB[884.00,888.70] vol=1.9x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:35:00 | 895.55 | 888.42 | 0.00 | T1 1.5R @ 895.55 |
| Target hit | 2024-05-17 15:20:00 | 908.00 | 902.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:35:00 | 911.35 | 916.43 | 0.00 | ORB-short ORB[914.00,927.30] vol=2.2x ATR=3.99 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 915.34 | 914.32 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 11:05:00 | 906.30 | 912.54 | 0.00 | ORB-short ORB[910.50,919.15] vol=1.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-05-27 11:50:00 | 909.17 | 911.53 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:55:00 | 918.25 | 915.02 | 0.00 | ORB-long ORB[908.05,916.65] vol=3.1x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 12:35:00 | 921.40 | 917.27 | 0.00 | T1 1.5R @ 921.40 |
| Stop hit — per-position SL triggered | 2024-06-12 13:50:00 | 918.25 | 918.60 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 09:30:00 | 936.45 | 932.75 | 0.00 | ORB-long ORB[925.55,935.55] vol=1.9x ATR=2.57 |
| Stop hit — per-position SL triggered | 2024-06-19 09:35:00 | 933.88 | 933.03 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:30:00 | 923.70 | 919.66 | 0.00 | ORB-long ORB[911.30,920.45] vol=3.3x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 11:55:00 | 927.61 | 922.10 | 0.00 | T1 1.5R @ 927.61 |
| Target hit | 2024-06-20 15:20:00 | 930.15 | 928.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 935.85 | 931.65 | 0.00 | ORB-long ORB[926.35,933.35] vol=1.7x ATR=2.80 |
| Stop hit — per-position SL triggered | 2024-06-21 09:40:00 | 933.05 | 932.64 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:50:00 | 950.25 | 943.30 | 0.00 | ORB-long ORB[938.20,946.55] vol=2.9x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 11:00:00 | 953.10 | 945.19 | 0.00 | T1 1.5R @ 953.10 |
| Stop hit — per-position SL triggered | 2024-07-02 11:10:00 | 950.25 | 945.69 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 952.70 | 949.90 | 0.00 | ORB-long ORB[945.20,952.00] vol=1.5x ATR=2.30 |
| Stop hit — per-position SL triggered | 2024-07-05 09:50:00 | 950.40 | 950.37 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 943.30 | 948.12 | 0.00 | ORB-short ORB[948.15,955.90] vol=2.3x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:50:00 | 940.43 | 946.48 | 0.00 | T1 1.5R @ 940.43 |
| Target hit | 2024-07-08 15:20:00 | 942.95 | 941.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 926.30 | 928.90 | 0.00 | ORB-short ORB[926.45,935.30] vol=2.6x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 928.76 | 928.80 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:10:00 | 923.50 | 926.47 | 0.00 | ORB-short ORB[925.40,932.00] vol=1.5x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 12:00:00 | 919.64 | 923.88 | 0.00 | T1 1.5R @ 919.64 |
| Target hit | 2024-07-11 14:00:00 | 922.40 | 921.96 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2024-07-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:50:00 | 931.75 | 927.97 | 0.00 | ORB-long ORB[925.00,931.20] vol=3.7x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:00:00 | 935.44 | 930.41 | 0.00 | T1 1.5R @ 935.44 |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 931.75 | 930.77 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:40:00 | 938.80 | 935.83 | 0.00 | ORB-long ORB[928.15,937.00] vol=5.4x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-07-16 10:45:00 | 937.16 | 936.03 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:35:00 | 923.60 | 927.56 | 0.00 | ORB-short ORB[925.55,934.50] vol=1.5x ATR=2.30 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 925.90 | 927.48 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:30:00 | 913.50 | 918.15 | 0.00 | ORB-short ORB[915.90,927.50] vol=1.8x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-07-19 09:35:00 | 916.46 | 917.93 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:00:00 | 895.45 | 888.93 | 0.00 | ORB-long ORB[875.30,887.00] vol=1.8x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 11:50:00 | 899.31 | 893.19 | 0.00 | T1 1.5R @ 899.31 |
| Target hit | 2024-07-26 15:05:00 | 897.80 | 898.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2024-07-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:30:00 | 899.80 | 895.67 | 0.00 | ORB-long ORB[890.15,896.70] vol=2.2x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 10:25:00 | 903.42 | 898.49 | 0.00 | T1 1.5R @ 903.42 |
| Stop hit — per-position SL triggered | 2024-07-30 13:15:00 | 899.80 | 900.62 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 909.65 | 906.84 | 0.00 | ORB-long ORB[902.70,908.95] vol=3.2x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:35:00 | 913.34 | 908.66 | 0.00 | T1 1.5R @ 913.34 |
| Stop hit — per-position SL triggered | 2024-07-31 09:45:00 | 909.65 | 909.05 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 11:05:00 | 904.10 | 911.41 | 0.00 | ORB-short ORB[912.70,923.85] vol=2.2x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-08-02 11:45:00 | 907.55 | 910.65 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:55:00 | 889.90 | 893.62 | 0.00 | ORB-short ORB[891.30,900.00] vol=1.6x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 892.65 | 893.39 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:40:00 | 905.00 | 906.26 | 0.00 | ORB-short ORB[905.10,911.95] vol=1.6x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 10:45:00 | 901.10 | 905.54 | 0.00 | T1 1.5R @ 901.10 |
| Target hit | 2024-08-14 12:55:00 | 899.25 | 897.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — SELL (started 2024-08-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 09:50:00 | 895.30 | 900.91 | 0.00 | ORB-short ORB[898.35,907.30] vol=1.8x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 10:55:00 | 890.14 | 897.03 | 0.00 | T1 1.5R @ 890.14 |
| Stop hit — per-position SL triggered | 2024-08-16 12:05:00 | 895.30 | 894.76 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:55:00 | 912.10 | 915.02 | 0.00 | ORB-short ORB[912.20,923.90] vol=1.7x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-08-20 11:10:00 | 914.01 | 914.71 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:35:00 | 921.35 | 916.28 | 0.00 | ORB-long ORB[908.10,918.90] vol=2.4x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:10:00 | 924.53 | 919.60 | 0.00 | T1 1.5R @ 924.53 |
| Stop hit — per-position SL triggered | 2024-08-21 10:50:00 | 921.35 | 920.21 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:55:00 | 936.10 | 938.98 | 0.00 | ORB-short ORB[938.80,947.50] vol=1.5x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-08-28 11:20:00 | 938.10 | 938.49 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:45:00 | 941.70 | 946.38 | 0.00 | ORB-short ORB[944.25,949.00] vol=2.6x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-08-29 10:50:00 | 943.79 | 945.79 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:35:00 | 942.35 | 938.79 | 0.00 | ORB-long ORB[933.10,938.50] vol=2.5x ATR=2.77 |
| Stop hit — per-position SL triggered | 2024-09-05 10:30:00 | 939.58 | 940.03 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:30:00 | 967.55 | 964.10 | 0.00 | ORB-long ORB[958.10,965.60] vol=3.0x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 09:45:00 | 970.80 | 965.64 | 0.00 | T1 1.5R @ 970.80 |
| Stop hit — per-position SL triggered | 2024-09-18 09:50:00 | 967.55 | 965.72 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:50:00 | 991.00 | 983.57 | 0.00 | ORB-long ORB[977.45,984.60] vol=1.6x ATR=2.30 |
| Stop hit — per-position SL triggered | 2024-09-26 11:10:00 | 988.70 | 984.39 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:45:00 | 1014.90 | 1011.29 | 0.00 | ORB-long ORB[1005.40,1013.95] vol=1.7x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-09-27 10:25:00 | 1011.52 | 1013.33 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:20:00 | 1019.95 | 1028.77 | 0.00 | ORB-short ORB[1031.00,1044.10] vol=1.9x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 11:05:00 | 1013.76 | 1024.86 | 0.00 | T1 1.5R @ 1013.76 |
| Stop hit — per-position SL triggered | 2024-10-07 11:10:00 | 1019.95 | 1024.67 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:45:00 | 1003.10 | 994.28 | 0.00 | ORB-long ORB[985.00,999.90] vol=1.9x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:15:00 | 1008.66 | 997.07 | 0.00 | T1 1.5R @ 1008.66 |
| Stop hit — per-position SL triggered | 2024-10-09 11:50:00 | 1003.10 | 998.11 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 1019.35 | 1015.25 | 0.00 | ORB-long ORB[1006.00,1016.45] vol=2.0x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:00:00 | 1024.42 | 1018.42 | 0.00 | T1 1.5R @ 1024.42 |
| Stop hit — per-position SL triggered | 2024-10-11 10:40:00 | 1019.35 | 1019.53 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:35:00 | 985.25 | 989.48 | 0.00 | ORB-short ORB[988.05,995.20] vol=1.6x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:00:00 | 981.07 | 987.39 | 0.00 | T1 1.5R @ 981.07 |
| Target hit | 2024-10-17 15:05:00 | 980.10 | 979.55 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — SELL (started 2024-10-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 09:30:00 | 976.15 | 980.05 | 0.00 | ORB-short ORB[977.35,983.05] vol=1.9x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-10-22 09:35:00 | 979.03 | 979.84 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:45:00 | 938.50 | 945.89 | 0.00 | ORB-short ORB[947.00,959.95] vol=1.7x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:55:00 | 934.03 | 945.16 | 0.00 | T1 1.5R @ 934.03 |
| Stop hit — per-position SL triggered | 2024-10-25 11:00:00 | 938.50 | 945.05 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-11-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:25:00 | 951.95 | 959.55 | 0.00 | ORB-short ORB[960.70,968.85] vol=2.1x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:50:00 | 947.51 | 957.33 | 0.00 | T1 1.5R @ 947.51 |
| Stop hit — per-position SL triggered | 2024-11-04 14:10:00 | 951.95 | 951.11 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-11-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 10:45:00 | 944.05 | 946.75 | 0.00 | ORB-short ORB[948.20,958.25] vol=1.5x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 11:10:00 | 938.57 | 946.43 | 0.00 | T1 1.5R @ 938.57 |
| Target hit | 2024-11-13 15:20:00 | 937.65 | 938.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-11-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:35:00 | 957.90 | 953.67 | 0.00 | ORB-long ORB[945.60,954.30] vol=1.7x ATR=2.60 |
| Stop hit — per-position SL triggered | 2024-11-19 10:45:00 | 955.30 | 953.86 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:15:00 | 963.95 | 967.28 | 0.00 | ORB-short ORB[964.30,974.95] vol=1.5x ATR=2.42 |
| Stop hit — per-position SL triggered | 2024-11-28 12:00:00 | 966.37 | 966.55 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:50:00 | 966.15 | 960.57 | 0.00 | ORB-long ORB[953.40,962.65] vol=1.7x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:25:00 | 970.76 | 962.50 | 0.00 | T1 1.5R @ 970.76 |
| Stop hit — per-position SL triggered | 2024-11-29 11:50:00 | 966.15 | 963.13 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 979.15 | 983.85 | 0.00 | ORB-short ORB[986.05,995.20] vol=2.1x ATR=2.69 |
| Stop hit — per-position SL triggered | 2024-12-05 12:00:00 | 981.84 | 982.97 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:40:00 | 1000.70 | 1002.61 | 0.00 | ORB-short ORB[1000.95,1011.90] vol=1.7x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 11:15:00 | 997.48 | 1001.62 | 0.00 | T1 1.5R @ 997.48 |
| Stop hit — per-position SL triggered | 2024-12-09 13:00:00 | 1000.70 | 1000.48 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:40:00 | 1018.25 | 1013.20 | 0.00 | ORB-long ORB[1008.35,1015.90] vol=1.8x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-12-11 10:40:00 | 1015.47 | 1015.40 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:10:00 | 980.90 | 983.98 | 0.00 | ORB-short ORB[983.00,990.80] vol=1.9x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-12-17 10:15:00 | 982.86 | 983.85 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:35:00 | 932.40 | 920.71 | 0.00 | ORB-long ORB[915.00,927.90] vol=1.6x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:50:00 | 938.05 | 924.40 | 0.00 | T1 1.5R @ 938.05 |
| Stop hit — per-position SL triggered | 2024-12-20 11:05:00 | 932.40 | 925.98 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 11:10:00 | 926.75 | 930.76 | 0.00 | ORB-short ORB[928.50,937.95] vol=1.5x ATR=2.49 |
| Stop hit — per-position SL triggered | 2024-12-24 11:20:00 | 929.24 | 930.67 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 904.30 | 897.26 | 0.00 | ORB-long ORB[893.45,903.75] vol=1.6x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:05:00 | 908.06 | 898.70 | 0.00 | T1 1.5R @ 908.06 |
| Stop hit — per-position SL triggered | 2025-01-01 11:35:00 | 904.30 | 899.49 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:15:00 | 899.35 | 901.56 | 0.00 | ORB-short ORB[900.60,908.95] vol=4.0x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:00:00 | 895.49 | 900.38 | 0.00 | T1 1.5R @ 895.49 |
| Stop hit — per-position SL triggered | 2025-01-09 11:55:00 | 899.35 | 899.78 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:50:00 | 882.10 | 886.87 | 0.00 | ORB-short ORB[887.35,896.05] vol=1.5x ATR=2.75 |
| Stop hit — per-position SL triggered | 2025-01-10 10:05:00 | 884.85 | 885.73 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-01-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:35:00 | 910.45 | 908.76 | 0.00 | ORB-long ORB[903.05,910.00] vol=4.6x ATR=2.26 |
| Stop hit — per-position SL triggered | 2025-01-15 11:30:00 | 908.19 | 909.28 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-01-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:25:00 | 905.50 | 909.14 | 0.00 | ORB-short ORB[907.15,913.80] vol=3.1x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 10:50:00 | 902.49 | 907.77 | 0.00 | T1 1.5R @ 902.49 |
| Stop hit — per-position SL triggered | 2025-01-16 11:25:00 | 905.50 | 905.80 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:30:00 | 916.05 | 911.85 | 0.00 | ORB-long ORB[905.25,912.45] vol=1.7x ATR=2.47 |
| Stop hit — per-position SL triggered | 2025-01-17 09:35:00 | 913.58 | 912.30 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-01-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:35:00 | 914.50 | 907.71 | 0.00 | ORB-long ORB[901.30,909.05] vol=2.0x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 11:00:00 | 917.56 | 909.84 | 0.00 | T1 1.5R @ 917.56 |
| Target hit | 2025-01-20 15:20:00 | 919.05 | 916.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2025-01-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:40:00 | 931.85 | 926.81 | 0.00 | ORB-long ORB[921.50,928.00] vol=2.8x ATR=2.48 |
| Stop hit — per-position SL triggered | 2025-01-21 09:55:00 | 929.37 | 927.81 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:35:00 | 921.70 | 917.85 | 0.00 | ORB-long ORB[912.80,919.95] vol=2.0x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:15:00 | 925.72 | 921.57 | 0.00 | T1 1.5R @ 925.72 |
| Target hit | 2025-01-23 15:20:00 | 929.85 | 928.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2025-01-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 11:00:00 | 946.80 | 938.05 | 0.00 | ORB-long ORB[930.00,943.65] vol=1.7x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:45:00 | 951.85 | 941.13 | 0.00 | T1 1.5R @ 951.85 |
| Stop hit — per-position SL triggered | 2025-01-24 12:00:00 | 946.80 | 942.17 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:55:00 | 911.00 | 913.83 | 0.00 | ORB-short ORB[915.05,925.60] vol=1.6x ATR=2.63 |
| Stop hit — per-position SL triggered | 2025-01-28 11:25:00 | 913.63 | 913.55 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:35:00 | 916.95 | 913.10 | 0.00 | ORB-long ORB[908.00,915.50] vol=1.5x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:00:00 | 921.93 | 915.49 | 0.00 | T1 1.5R @ 921.93 |
| Target hit | 2025-01-29 15:20:00 | 938.35 | 929.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2025-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:35:00 | 948.50 | 944.54 | 0.00 | ORB-long ORB[936.20,947.40] vol=1.7x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:55:00 | 953.06 | 947.31 | 0.00 | T1 1.5R @ 953.06 |
| Target hit | 2025-01-30 10:50:00 | 949.20 | 950.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — BUY (started 2025-02-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:35:00 | 952.70 | 946.25 | 0.00 | ORB-long ORB[941.10,947.95] vol=2.2x ATR=2.70 |
| Stop hit — per-position SL triggered | 2025-02-05 09:45:00 | 950.00 | 947.65 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-02-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 10:45:00 | 955.20 | 951.97 | 0.00 | ORB-long ORB[946.55,954.90] vol=2.3x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-02-06 11:05:00 | 953.52 | 952.49 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-02-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 09:35:00 | 964.55 | 959.69 | 0.00 | ORB-long ORB[947.80,961.00] vol=1.9x ATR=3.02 |
| Stop hit — per-position SL triggered | 2025-02-07 09:50:00 | 961.53 | 960.88 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 10:15:00 | 940.90 | 949.36 | 0.00 | ORB-short ORB[950.45,962.95] vol=3.6x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-02-12 10:35:00 | 945.09 | 948.61 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:35:00 | 970.85 | 963.42 | 0.00 | ORB-long ORB[953.25,964.75] vol=2.0x ATR=3.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 10:00:00 | 976.58 | 969.35 | 0.00 | T1 1.5R @ 976.58 |
| Target hit | 2025-02-13 13:20:00 | 974.85 | 975.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 69 — SELL (started 2025-02-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:55:00 | 969.85 | 973.76 | 0.00 | ORB-short ORB[970.55,983.70] vol=1.9x ATR=3.22 |
| Stop hit — per-position SL triggered | 2025-02-14 11:50:00 | 973.07 | 972.24 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 10:45:00 | 969.20 | 974.47 | 0.00 | ORB-short ORB[970.05,981.70] vol=3.1x ATR=2.60 |
| Stop hit — per-position SL triggered | 2025-02-24 11:20:00 | 971.80 | 972.84 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-03-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 11:00:00 | 953.75 | 960.65 | 0.00 | ORB-short ORB[955.05,968.60] vol=1.8x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-03-03 11:10:00 | 957.10 | 960.46 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:45:00 | 986.30 | 979.13 | 0.00 | ORB-long ORB[970.90,980.90] vol=1.9x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 10:25:00 | 990.62 | 985.36 | 0.00 | T1 1.5R @ 990.62 |
| Target hit | 2025-03-05 15:20:00 | 1006.40 | 995.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:40:00 | 1023.00 | 1017.01 | 0.00 | ORB-long ORB[1010.25,1022.15] vol=1.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 10:00:00 | 1027.73 | 1019.70 | 0.00 | T1 1.5R @ 1027.73 |
| Target hit | 2025-03-10 12:15:00 | 1024.45 | 1025.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — BUY (started 2025-03-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:50:00 | 1015.10 | 1010.26 | 0.00 | ORB-long ORB[1001.45,1012.70] vol=1.5x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-03-11 11:40:00 | 1012.17 | 1011.29 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:10:00 | 999.85 | 1009.38 | 0.00 | ORB-short ORB[1013.10,1024.95] vol=1.9x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-03-12 11:40:00 | 1002.03 | 1006.52 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:40:00 | 1015.25 | 1009.29 | 0.00 | ORB-long ORB[1003.45,1014.90] vol=1.5x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:45:00 | 1018.45 | 1010.33 | 0.00 | T1 1.5R @ 1018.45 |
| Stop hit — per-position SL triggered | 2025-03-18 12:30:00 | 1015.25 | 1012.75 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 11:10:00 | 1038.85 | 1032.80 | 0.00 | ORB-long ORB[1027.30,1038.00] vol=1.7x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-03-20 11:15:00 | 1036.53 | 1033.09 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-03-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:40:00 | 1064.50 | 1059.81 | 0.00 | ORB-long ORB[1052.00,1063.55] vol=1.9x ATR=3.09 |
| Stop hit — per-position SL triggered | 2025-03-25 10:40:00 | 1061.41 | 1062.25 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2025-04-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:55:00 | 1043.20 | 1048.26 | 0.00 | ORB-short ORB[1045.00,1054.45] vol=1.6x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 1046.09 | 1047.46 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-04-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:35:00 | 986.15 | 978.46 | 0.00 | ORB-long ORB[968.35,981.75] vol=2.7x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-11 09:50:00 | 993.19 | 982.90 | 0.00 | T1 1.5R @ 993.19 |
| Stop hit — per-position SL triggered | 2025-04-11 09:55:00 | 986.15 | 983.32 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 10:15:00 | 995.10 | 1000.26 | 0.00 | ORB-short ORB[1000.80,1007.30] vol=1.8x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-04-16 10:45:00 | 997.35 | 998.50 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 09:30:00 | 988.00 | 993.57 | 0.00 | ORB-short ORB[991.10,1004.10] vol=2.9x ATR=3.31 |
| Stop hit — per-position SL triggered | 2025-04-17 09:40:00 | 991.31 | 993.31 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:40:00 | 1031.70 | 1040.24 | 0.00 | ORB-short ORB[1044.10,1053.90] vol=1.5x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-04-23 11:05:00 | 1034.51 | 1038.89 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-04-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:55:00 | 1022.30 | 1037.36 | 0.00 | ORB-short ORB[1044.00,1056.90] vol=2.7x ATR=3.53 |
| Stop hit — per-position SL triggered | 2025-04-25 12:10:00 | 1025.83 | 1029.36 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 11:05:00 | 1050.70 | 1039.36 | 0.00 | ORB-long ORB[1019.20,1033.20] vol=1.7x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-04-28 11:10:00 | 1048.05 | 1039.67 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 11:00:00 | 1060.50 | 1056.65 | 0.00 | ORB-long ORB[1053.40,1059.60] vol=2.6x ATR=3.01 |
| Stop hit — per-position SL triggered | 2025-04-29 11:10:00 | 1057.49 | 1056.99 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2025-05-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 10:50:00 | 1008.80 | 1024.39 | 0.00 | ORB-short ORB[1013.50,1028.40] vol=1.7x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 11:00:00 | 1002.90 | 1020.32 | 0.00 | T1 1.5R @ 1002.90 |
| Target hit | 2025-05-02 15:20:00 | 973.70 | 976.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — BUY (started 2025-05-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 10:40:00 | 966.30 | 961.66 | 0.00 | ORB-long ORB[955.00,962.90] vol=1.8x ATR=2.61 |
| Stop hit — per-position SL triggered | 2025-05-06 11:15:00 | 963.69 | 963.09 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2025-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:30:00 | 959.40 | 963.72 | 0.00 | ORB-short ORB[961.50,969.50] vol=1.7x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-05-08 09:35:00 | 961.79 | 963.46 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 09:55:00 | 871.15 | 2024-05-15 10:15:00 | 866.75 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-15 09:55:00 | 871.15 | 2024-05-15 11:00:00 | 871.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 11:15:00 | 865.90 | 2024-05-16 11:25:00 | 868.37 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-17 10:25:00 | 890.75 | 2024-05-17 10:35:00 | 895.55 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-05-17 10:25:00 | 890.75 | 2024-05-17 15:20:00 | 908.00 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2024-05-22 09:35:00 | 911.35 | 2024-05-22 09:55:00 | 915.34 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-05-27 11:05:00 | 906.30 | 2024-05-27 11:50:00 | 909.17 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-12 10:55:00 | 918.25 | 2024-06-12 12:35:00 | 921.40 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-06-12 10:55:00 | 918.25 | 2024-06-12 13:50:00 | 918.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-19 09:30:00 | 936.45 | 2024-06-19 09:35:00 | 933.88 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-20 10:30:00 | 923.70 | 2024-06-20 11:55:00 | 927.61 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-06-20 10:30:00 | 923.70 | 2024-06-20 15:20:00 | 930.15 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2024-06-21 09:30:00 | 935.85 | 2024-06-21 09:40:00 | 933.05 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-02 10:50:00 | 950.25 | 2024-07-02 11:00:00 | 953.10 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-07-02 10:50:00 | 950.25 | 2024-07-02 11:10:00 | 950.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 09:30:00 | 952.70 | 2024-07-05 09:50:00 | 950.40 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-07-08 11:10:00 | 943.30 | 2024-07-08 11:50:00 | 940.43 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-07-08 11:10:00 | 943.30 | 2024-07-08 15:20:00 | 942.95 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2024-07-10 10:10:00 | 926.30 | 2024-07-10 10:15:00 | 928.76 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-11 10:10:00 | 923.50 | 2024-07-11 12:00:00 | 919.64 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-07-11 10:10:00 | 923.50 | 2024-07-11 14:00:00 | 922.40 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2024-07-12 10:50:00 | 931.75 | 2024-07-12 11:00:00 | 935.44 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-12 10:50:00 | 931.75 | 2024-07-12 11:15:00 | 931.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 10:40:00 | 938.80 | 2024-07-16 10:45:00 | 937.16 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-07-18 09:35:00 | 923.60 | 2024-07-18 09:40:00 | 925.90 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-19 09:30:00 | 913.50 | 2024-07-19 09:35:00 | 916.46 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-26 10:00:00 | 895.45 | 2024-07-26 11:50:00 | 899.31 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-26 10:00:00 | 895.45 | 2024-07-26 15:05:00 | 897.80 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-07-30 09:30:00 | 899.80 | 2024-07-30 10:25:00 | 903.42 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-30 09:30:00 | 899.80 | 2024-07-30 13:15:00 | 899.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 09:30:00 | 909.65 | 2024-07-31 09:35:00 | 913.34 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-31 09:30:00 | 909.65 | 2024-07-31 09:45:00 | 909.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-02 11:05:00 | 904.10 | 2024-08-02 11:45:00 | 907.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-08 10:55:00 | 889.90 | 2024-08-08 11:15:00 | 892.65 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-14 10:40:00 | 905.00 | 2024-08-14 10:45:00 | 901.10 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-14 10:40:00 | 905.00 | 2024-08-14 12:55:00 | 899.25 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2024-08-16 09:50:00 | 895.30 | 2024-08-16 10:55:00 | 890.14 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-08-16 09:50:00 | 895.30 | 2024-08-16 12:05:00 | 895.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-20 10:55:00 | 912.10 | 2024-08-20 11:10:00 | 914.01 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-21 09:35:00 | 921.35 | 2024-08-21 10:10:00 | 924.53 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-21 09:35:00 | 921.35 | 2024-08-21 10:50:00 | 921.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 10:55:00 | 936.10 | 2024-08-28 11:20:00 | 938.10 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-29 10:45:00 | 941.70 | 2024-08-29 10:50:00 | 943.79 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-05 09:35:00 | 942.35 | 2024-09-05 10:30:00 | 939.58 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-18 09:30:00 | 967.55 | 2024-09-18 09:45:00 | 970.80 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-18 09:30:00 | 967.55 | 2024-09-18 09:50:00 | 967.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 10:50:00 | 991.00 | 2024-09-26 11:10:00 | 988.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-27 09:45:00 | 1014.90 | 2024-09-27 10:25:00 | 1011.52 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-07 10:20:00 | 1019.95 | 2024-10-07 11:05:00 | 1013.76 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-10-07 10:20:00 | 1019.95 | 2024-10-07 11:10:00 | 1019.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 10:45:00 | 1003.10 | 2024-10-09 11:15:00 | 1008.66 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-10-09 10:45:00 | 1003.10 | 2024-10-09 11:50:00 | 1003.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 09:35:00 | 1019.35 | 2024-10-11 10:00:00 | 1024.42 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-10-11 09:35:00 | 1019.35 | 2024-10-11 10:40:00 | 1019.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 09:35:00 | 985.25 | 2024-10-17 10:00:00 | 981.07 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-10-17 09:35:00 | 985.25 | 2024-10-17 15:05:00 | 980.10 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-22 09:30:00 | 976.15 | 2024-10-22 09:35:00 | 979.03 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-25 10:45:00 | 938.50 | 2024-10-25 10:55:00 | 934.03 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-10-25 10:45:00 | 938.50 | 2024-10-25 11:00:00 | 938.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-04 10:25:00 | 951.95 | 2024-11-04 10:50:00 | 947.51 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-11-04 10:25:00 | 951.95 | 2024-11-04 14:10:00 | 951.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 10:45:00 | 944.05 | 2024-11-13 11:10:00 | 938.57 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-11-13 10:45:00 | 944.05 | 2024-11-13 15:20:00 | 937.65 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2024-11-19 10:35:00 | 957.90 | 2024-11-19 10:45:00 | 955.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-11-28 11:15:00 | 963.95 | 2024-11-28 12:00:00 | 966.37 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-11-29 10:50:00 | 966.15 | 2024-11-29 11:25:00 | 970.76 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-11-29 10:50:00 | 966.15 | 2024-11-29 11:50:00 | 966.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 10:55:00 | 979.15 | 2024-12-05 12:00:00 | 981.84 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-09 10:40:00 | 1000.70 | 2024-12-09 11:15:00 | 997.48 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-12-09 10:40:00 | 1000.70 | 2024-12-09 13:00:00 | 1000.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 09:40:00 | 1018.25 | 2024-12-11 10:40:00 | 1015.47 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-17 10:10:00 | 980.90 | 2024-12-17 10:15:00 | 982.86 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-12-20 10:35:00 | 932.40 | 2024-12-20 10:50:00 | 938.05 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-12-20 10:35:00 | 932.40 | 2024-12-20 11:05:00 | 932.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-24 11:10:00 | 926.75 | 2024-12-24 11:20:00 | 929.24 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-01 10:50:00 | 904.30 | 2025-01-01 11:05:00 | 908.06 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-01-01 10:50:00 | 904.30 | 2025-01-01 11:35:00 | 904.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-09 10:15:00 | 899.35 | 2025-01-09 11:00:00 | 895.49 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-09 10:15:00 | 899.35 | 2025-01-09 11:55:00 | 899.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-10 09:50:00 | 882.10 | 2025-01-10 10:05:00 | 884.85 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-15 10:35:00 | 910.45 | 2025-01-15 11:30:00 | 908.19 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-16 10:25:00 | 905.50 | 2025-01-16 10:50:00 | 902.49 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-01-16 10:25:00 | 905.50 | 2025-01-16 11:25:00 | 905.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-17 09:30:00 | 916.05 | 2025-01-17 09:35:00 | 913.58 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-20 10:35:00 | 914.50 | 2025-01-20 11:00:00 | 917.56 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-01-20 10:35:00 | 914.50 | 2025-01-20 15:20:00 | 919.05 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-21 09:40:00 | 931.85 | 2025-01-21 09:55:00 | 929.37 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-23 09:35:00 | 921.70 | 2025-01-23 10:15:00 | 925.72 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-23 09:35:00 | 921.70 | 2025-01-23 15:20:00 | 929.85 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2025-01-24 11:00:00 | 946.80 | 2025-01-24 11:45:00 | 951.85 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-01-24 11:00:00 | 946.80 | 2025-01-24 12:00:00 | 946.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-28 10:55:00 | 911.00 | 2025-01-28 11:25:00 | 913.63 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-29 09:35:00 | 916.95 | 2025-01-29 10:00:00 | 921.93 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-01-29 09:35:00 | 916.95 | 2025-01-29 15:20:00 | 938.35 | TARGET_HIT | 0.50 | 2.33% |
| BUY | retest1 | 2025-01-30 09:35:00 | 948.50 | 2025-01-30 09:55:00 | 953.06 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-01-30 09:35:00 | 948.50 | 2025-01-30 10:50:00 | 949.20 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2025-02-05 09:35:00 | 952.70 | 2025-02-05 09:45:00 | 950.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-02-06 10:45:00 | 955.20 | 2025-02-06 11:05:00 | 953.52 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-02-07 09:35:00 | 964.55 | 2025-02-07 09:50:00 | 961.53 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-02-12 10:15:00 | 940.90 | 2025-02-12 10:35:00 | 945.09 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-02-13 09:35:00 | 970.85 | 2025-02-13 10:00:00 | 976.58 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-02-13 09:35:00 | 970.85 | 2025-02-13 13:20:00 | 974.85 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2025-02-14 10:55:00 | 969.85 | 2025-02-14 11:50:00 | 973.07 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-24 10:45:00 | 969.20 | 2025-02-24 11:20:00 | 971.80 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-03-03 11:00:00 | 953.75 | 2025-03-03 11:10:00 | 957.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-05 09:45:00 | 986.30 | 2025-03-05 10:25:00 | 990.62 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-03-05 09:45:00 | 986.30 | 2025-03-05 15:20:00 | 1006.40 | TARGET_HIT | 0.50 | 2.04% |
| BUY | retest1 | 2025-03-10 09:40:00 | 1023.00 | 2025-03-10 10:00:00 | 1027.73 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-03-10 09:40:00 | 1023.00 | 2025-03-10 12:15:00 | 1024.45 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2025-03-11 10:50:00 | 1015.10 | 2025-03-11 11:40:00 | 1012.17 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-03-12 11:10:00 | 999.85 | 2025-03-12 11:40:00 | 1002.03 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-03-18 10:40:00 | 1015.25 | 2025-03-18 10:45:00 | 1018.45 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-03-18 10:40:00 | 1015.25 | 2025-03-18 12:30:00 | 1015.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-20 11:10:00 | 1038.85 | 2025-03-20 11:15:00 | 1036.53 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-03-25 09:40:00 | 1064.50 | 2025-03-25 10:40:00 | 1061.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-03 09:55:00 | 1043.20 | 2025-04-03 10:15:00 | 1046.09 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-11 09:35:00 | 986.15 | 2025-04-11 09:50:00 | 993.19 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-04-11 09:35:00 | 986.15 | 2025-04-11 09:55:00 | 986.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-16 10:15:00 | 995.10 | 2025-04-16 10:45:00 | 997.35 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-04-17 09:30:00 | 988.00 | 2025-04-17 09:40:00 | 991.31 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-04-23 10:40:00 | 1031.70 | 2025-04-23 11:05:00 | 1034.51 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-04-25 10:55:00 | 1022.30 | 2025-04-25 12:10:00 | 1025.83 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-28 11:05:00 | 1050.70 | 2025-04-28 11:10:00 | 1048.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-04-29 11:00:00 | 1060.50 | 2025-04-29 11:10:00 | 1057.49 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-02 10:50:00 | 1008.80 | 2025-05-02 11:00:00 | 1002.90 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-05-02 10:50:00 | 1008.80 | 2025-05-02 15:20:00 | 973.70 | TARGET_HIT | 0.50 | 3.48% |
| BUY | retest1 | 2025-05-06 10:40:00 | 966.30 | 2025-05-06 11:15:00 | 963.69 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-08 09:30:00 | 959.40 | 2025-05-08 09:35:00 | 961.79 | STOP_HIT | 1.00 | -0.25% |
