# Chalet Hotels Ltd. (CHALET)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-08-08 15:25:00 (4875 bars)
- **Last close:** 868.05
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
| ENTRY1 | 23 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 1 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 22
- **Target hits / Stop hits / Partials:** 1 / 22 / 9
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 1.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.10% | -1.0% |
| BUY @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.10% | -1.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 8 | 36.4% | 1 | 14 | 7 | 0.10% | 2.2% |
| SELL @ 2nd Alert (retest1) | 22 | 8 | 36.4% | 1 | 14 | 7 | 0.10% | 2.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 32 | 10 | 31.2% | 1 | 22 | 9 | 0.04% | 1.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:40:00 | 893.95 | 889.97 | 0.00 | ORB-long ORB[880.00,888.55] vol=1.7x ATR=4.12 |
| Stop hit — per-position SL triggered | 2025-05-23 10:05:00 | 889.83 | 890.09 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:20:00 | 918.35 | 921.40 | 0.00 | ORB-short ORB[920.00,930.70] vol=1.9x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:50:00 | 914.24 | 920.57 | 0.00 | T1 1.5R @ 914.24 |
| Stop hit — per-position SL triggered | 2025-05-28 12:05:00 | 918.35 | 918.38 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:55:00 | 926.25 | 934.07 | 0.00 | ORB-short ORB[930.20,943.70] vol=3.5x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 11:20:00 | 922.09 | 933.23 | 0.00 | T1 1.5R @ 922.09 |
| Stop hit — per-position SL triggered | 2025-05-29 13:40:00 | 926.25 | 930.69 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 11:15:00 | 914.20 | 920.49 | 0.00 | ORB-short ORB[926.35,935.35] vol=6.5x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-05-30 11:40:00 | 916.99 | 919.15 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 11:10:00 | 914.25 | 918.78 | 0.00 | ORB-short ORB[918.05,925.40] vol=1.8x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 12:15:00 | 910.66 | 917.43 | 0.00 | T1 1.5R @ 910.66 |
| Stop hit — per-position SL triggered | 2025-06-05 13:00:00 | 914.25 | 916.36 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 926.35 | 921.54 | 0.00 | ORB-long ORB[915.65,922.40] vol=1.8x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-06-06 10:20:00 | 923.42 | 922.01 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 09:35:00 | 919.00 | 925.36 | 0.00 | ORB-short ORB[923.30,932.85] vol=1.7x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-06-10 09:45:00 | 922.00 | 924.40 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:35:00 | 894.95 | 889.96 | 0.00 | ORB-long ORB[882.00,889.95] vol=2.4x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 12:20:00 | 900.96 | 894.23 | 0.00 | T1 1.5R @ 900.96 |
| Stop hit — per-position SL triggered | 2025-06-17 12:50:00 | 894.95 | 894.40 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:55:00 | 893.00 | 901.51 | 0.00 | ORB-short ORB[897.85,909.00] vol=2.1x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:35:00 | 888.43 | 900.62 | 0.00 | T1 1.5R @ 888.43 |
| Stop hit — per-position SL triggered | 2025-06-19 14:00:00 | 893.00 | 896.81 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:00:00 | 909.50 | 904.64 | 0.00 | ORB-long ORB[897.60,904.85] vol=1.9x ATR=3.13 |
| Stop hit — per-position SL triggered | 2025-06-25 11:30:00 | 906.37 | 907.46 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:05:00 | 913.15 | 916.74 | 0.00 | ORB-short ORB[919.00,927.50] vol=4.5x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 11:10:00 | 909.18 | 915.82 | 0.00 | T1 1.5R @ 909.18 |
| Stop hit — per-position SL triggered | 2025-06-26 11:45:00 | 913.15 | 915.54 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:35:00 | 926.75 | 924.58 | 0.00 | ORB-long ORB[917.55,923.55] vol=4.5x ATR=3.26 |
| Stop hit — per-position SL triggered | 2025-07-01 09:55:00 | 923.49 | 924.61 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:05:00 | 892.75 | 904.29 | 0.00 | ORB-short ORB[902.50,914.85] vol=3.2x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-07-02 11:20:00 | 894.92 | 903.76 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 11:05:00 | 890.00 | 894.32 | 0.00 | ORB-short ORB[890.40,901.65] vol=1.5x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-07-03 12:10:00 | 892.36 | 893.63 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 09:40:00 | 887.80 | 890.16 | 0.00 | ORB-short ORB[888.25,894.00] vol=1.6x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-07-04 10:30:00 | 890.59 | 892.08 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:35:00 | 870.35 | 878.96 | 0.00 | ORB-short ORB[875.00,887.60] vol=2.3x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:15:00 | 864.78 | 873.98 | 0.00 | T1 1.5R @ 864.78 |
| Target hit | 2025-07-08 14:25:00 | 861.90 | 856.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2025-07-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:55:00 | 874.55 | 867.02 | 0.00 | ORB-long ORB[857.40,869.85] vol=3.3x ATR=3.04 |
| Stop hit — per-position SL triggered | 2025-07-09 11:05:00 | 871.51 | 867.78 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:25:00 | 861.00 | 864.67 | 0.00 | ORB-short ORB[870.00,880.00] vol=7.0x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-07-11 10:40:00 | 864.34 | 863.81 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 10:55:00 | 933.45 | 926.74 | 0.00 | ORB-long ORB[919.25,926.60] vol=7.2x ATR=3.48 |
| Stop hit — per-position SL triggered | 2025-07-18 11:20:00 | 929.97 | 927.80 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:50:00 | 946.65 | 941.75 | 0.00 | ORB-long ORB[935.35,944.90] vol=3.3x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 10:00:00 | 951.82 | 944.72 | 0.00 | T1 1.5R @ 951.82 |
| Stop hit — per-position SL triggered | 2025-07-24 10:05:00 | 946.65 | 945.20 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 11:00:00 | 893.25 | 900.90 | 0.00 | ORB-short ORB[898.45,909.30] vol=1.7x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 895.61 | 900.23 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:45:00 | 893.70 | 898.91 | 0.00 | ORB-short ORB[897.15,908.15] vol=2.1x ATR=2.57 |
| Stop hit — per-position SL triggered | 2025-07-30 09:50:00 | 896.27 | 898.75 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:50:00 | 883.85 | 888.19 | 0.00 | ORB-short ORB[888.15,900.05] vol=3.2x ATR=4.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 10:45:00 | 877.00 | 884.77 | 0.00 | T1 1.5R @ 877.00 |
| Stop hit — per-position SL triggered | 2025-08-05 10:55:00 | 883.85 | 884.65 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-23 09:40:00 | 893.95 | 2025-05-23 10:05:00 | 889.83 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-05-28 10:20:00 | 918.35 | 2025-05-28 10:50:00 | 914.24 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-05-28 10:20:00 | 918.35 | 2025-05-28 12:05:00 | 918.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-29 10:55:00 | 926.25 | 2025-05-29 11:20:00 | 922.09 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-05-29 10:55:00 | 926.25 | 2025-05-29 13:40:00 | 926.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-30 11:15:00 | 914.20 | 2025-05-30 11:40:00 | 916.99 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-06-05 11:10:00 | 914.25 | 2025-06-05 12:15:00 | 910.66 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-05 11:10:00 | 914.25 | 2025-06-05 13:00:00 | 914.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-06 10:15:00 | 926.35 | 2025-06-06 10:20:00 | 923.42 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-06-10 09:35:00 | 919.00 | 2025-06-10 09:45:00 | 922.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-17 09:35:00 | 894.95 | 2025-06-17 12:20:00 | 900.96 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-06-17 09:35:00 | 894.95 | 2025-06-17 12:50:00 | 894.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-19 10:55:00 | 893.00 | 2025-06-19 11:35:00 | 888.43 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-06-19 10:55:00 | 893.00 | 2025-06-19 14:00:00 | 893.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-25 10:00:00 | 909.50 | 2025-06-25 11:30:00 | 906.37 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-06-26 11:05:00 | 913.15 | 2025-06-26 11:10:00 | 909.18 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-06-26 11:05:00 | 913.15 | 2025-06-26 11:45:00 | 913.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-01 09:35:00 | 926.75 | 2025-07-01 09:55:00 | 923.49 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-02 11:05:00 | 892.75 | 2025-07-02 11:20:00 | 894.92 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-03 11:05:00 | 890.00 | 2025-07-03 12:10:00 | 892.36 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-04 09:40:00 | 887.80 | 2025-07-04 10:30:00 | 890.59 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-08 09:35:00 | 870.35 | 2025-07-08 10:15:00 | 864.78 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-07-08 09:35:00 | 870.35 | 2025-07-08 14:25:00 | 861.90 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2025-07-09 10:55:00 | 874.55 | 2025-07-09 11:05:00 | 871.51 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-11 10:25:00 | 861.00 | 2025-07-11 10:40:00 | 864.34 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-18 10:55:00 | 933.45 | 2025-07-18 11:20:00 | 929.97 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-07-24 09:50:00 | 946.65 | 2025-07-24 10:00:00 | 951.82 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-07-24 09:50:00 | 946.65 | 2025-07-24 10:05:00 | 946.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-29 11:00:00 | 893.25 | 2025-07-29 11:15:00 | 895.61 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-30 09:45:00 | 893.70 | 2025-07-30 09:50:00 | 896.27 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-05 09:50:00 | 883.85 | 2025-08-05 10:45:00 | 877.00 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2025-08-05 09:50:00 | 883.85 | 2025-08-05 10:55:00 | 883.85 | STOP_HIT | 0.50 | 0.00% |
