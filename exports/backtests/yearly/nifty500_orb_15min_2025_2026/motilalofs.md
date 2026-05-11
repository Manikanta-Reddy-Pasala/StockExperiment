# Motilal Oswal Financial Services Ltd. (MOTILALOFS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 882.20
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 15 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 48
- **Target hits / Stop hits / Partials:** 15 / 48 / 27
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 16.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 20 | 41.7% | 7 | 28 | 13 | 0.11% | 5.3% |
| BUY @ 2nd Alert (retest1) | 48 | 20 | 41.7% | 7 | 28 | 13 | 0.11% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 22 | 52.4% | 8 | 20 | 14 | 0.26% | 10.8% |
| SELL @ 2nd Alert (retest1) | 42 | 22 | 52.4% | 8 | 20 | 14 | 0.26% | 10.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 90 | 42 | 46.7% | 15 | 48 | 27 | 0.18% | 16.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 10:35:00 | 777.05 | 780.47 | 0.00 | ORB-short ORB[777.60,787.05] vol=1.9x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-05-27 11:10:00 | 779.85 | 780.03 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:45:00 | 819.30 | 815.30 | 0.00 | ORB-long ORB[807.15,818.45] vol=2.3x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-05-30 09:55:00 | 816.25 | 815.58 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:40:00 | 827.00 | 821.70 | 0.00 | ORB-long ORB[816.75,825.15] vol=1.6x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 09:45:00 | 832.12 | 826.60 | 0.00 | T1 1.5R @ 832.12 |
| Target hit | 2025-06-05 10:25:00 | 828.55 | 831.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2025-06-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:45:00 | 805.25 | 814.69 | 0.00 | ORB-short ORB[813.80,824.65] vol=1.5x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-06-16 09:50:00 | 809.09 | 812.88 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 826.55 | 822.95 | 0.00 | ORB-long ORB[815.25,825.80] vol=2.2x ATR=3.51 |
| Stop hit — per-position SL triggered | 2025-06-17 09:40:00 | 823.04 | 823.57 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:50:00 | 829.00 | 821.77 | 0.00 | ORB-long ORB[815.00,824.90] vol=2.4x ATR=3.79 |
| Stop hit — per-position SL triggered | 2025-06-20 09:55:00 | 825.21 | 822.43 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:00:00 | 851.10 | 844.56 | 0.00 | ORB-long ORB[839.95,849.20] vol=2.0x ATR=4.15 |
| Stop hit — per-position SL triggered | 2025-06-24 10:05:00 | 846.95 | 844.81 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:00:00 | 871.50 | 864.95 | 0.00 | ORB-long ORB[859.55,867.45] vol=1.6x ATR=3.22 |
| Stop hit — per-position SL triggered | 2025-06-25 10:05:00 | 868.28 | 865.10 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:45:00 | 880.40 | 873.79 | 0.00 | ORB-long ORB[861.10,872.75] vol=2.3x ATR=3.07 |
| Stop hit — per-position SL triggered | 2025-06-27 10:50:00 | 877.33 | 873.99 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:50:00 | 860.60 | 866.50 | 0.00 | ORB-short ORB[866.40,874.90] vol=1.5x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:00:00 | 856.80 | 865.17 | 0.00 | T1 1.5R @ 856.80 |
| Target hit | 2025-07-02 15:05:00 | 858.00 | 856.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2025-07-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:35:00 | 945.15 | 938.59 | 0.00 | ORB-long ORB[933.15,944.90] vol=2.8x ATR=3.97 |
| Stop hit — per-position SL triggered | 2025-07-15 15:20:00 | 944.70 | 941.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:35:00 | 923.25 | 926.93 | 0.00 | ORB-short ORB[924.00,935.40] vol=1.8x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-07-18 10:50:00 | 926.10 | 926.30 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:45:00 | 925.80 | 920.64 | 0.00 | ORB-long ORB[909.15,918.00] vol=1.6x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 11:05:00 | 930.89 | 921.86 | 0.00 | T1 1.5R @ 930.89 |
| Target hit | 2025-07-21 14:15:00 | 928.60 | 929.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2025-07-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:00:00 | 922.50 | 932.71 | 0.00 | ORB-short ORB[934.50,940.95] vol=2.1x ATR=2.67 |
| Stop hit — per-position SL triggered | 2025-07-22 11:45:00 | 925.17 | 931.59 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:45:00 | 896.50 | 890.95 | 0.00 | ORB-long ORB[881.20,892.30] vol=1.5x ATR=4.87 |
| Stop hit — per-position SL triggered | 2025-07-29 10:25:00 | 891.63 | 892.41 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 10:00:00 | 908.70 | 917.10 | 0.00 | ORB-short ORB[914.00,927.45] vol=1.6x ATR=4.45 |
| Stop hit — per-position SL triggered | 2025-08-01 10:10:00 | 913.15 | 916.37 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:15:00 | 920.10 | 926.10 | 0.00 | ORB-short ORB[922.05,933.90] vol=1.5x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 13:00:00 | 914.86 | 922.25 | 0.00 | T1 1.5R @ 914.86 |
| Target hit | 2025-08-05 15:20:00 | 911.30 | 917.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-08-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:00:00 | 905.10 | 915.82 | 0.00 | ORB-short ORB[913.00,920.00] vol=2.5x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:35:00 | 900.34 | 912.90 | 0.00 | T1 1.5R @ 900.34 |
| Stop hit — per-position SL triggered | 2025-08-06 11:50:00 | 905.10 | 912.64 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:10:00 | 908.00 | 914.27 | 0.00 | ORB-short ORB[911.90,922.00] vol=5.6x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:25:00 | 903.67 | 913.44 | 0.00 | T1 1.5R @ 903.67 |
| Target hit | 2025-08-07 14:30:00 | 905.05 | 903.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — SELL (started 2025-08-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 11:00:00 | 927.85 | 929.08 | 0.00 | ORB-short ORB[928.00,935.05] vol=2.3x ATR=2.57 |
| Stop hit — per-position SL triggered | 2025-08-14 11:50:00 | 930.42 | 929.27 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 11:00:00 | 951.95 | 943.33 | 0.00 | ORB-long ORB[936.60,945.00] vol=7.3x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-08-19 11:20:00 | 949.07 | 946.62 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 09:30:00 | 950.60 | 954.35 | 0.00 | ORB-short ORB[952.75,963.00] vol=3.2x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 09:40:00 | 946.61 | 953.07 | 0.00 | T1 1.5R @ 946.61 |
| Stop hit — per-position SL triggered | 2025-08-21 10:00:00 | 950.60 | 951.85 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:35:00 | 935.05 | 939.13 | 0.00 | ORB-short ORB[938.00,949.00] vol=3.3x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 10:45:00 | 931.14 | 938.44 | 0.00 | T1 1.5R @ 931.14 |
| Stop hit — per-position SL triggered | 2025-08-22 11:00:00 | 935.05 | 937.91 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:45:00 | 880.55 | 870.06 | 0.00 | ORB-long ORB[859.60,869.40] vol=2.8x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:55:00 | 887.83 | 873.49 | 0.00 | T1 1.5R @ 887.83 |
| Stop hit — per-position SL triggered | 2025-09-01 10:05:00 | 880.55 | 874.29 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:30:00 | 876.20 | 871.07 | 0.00 | ORB-long ORB[865.80,874.35] vol=2.6x ATR=3.38 |
| Stop hit — per-position SL triggered | 2025-09-05 09:35:00 | 872.82 | 871.34 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:40:00 | 881.20 | 876.49 | 0.00 | ORB-long ORB[870.05,877.70] vol=1.9x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 10:45:00 | 884.80 | 877.05 | 0.00 | T1 1.5R @ 884.80 |
| Target hit | 2025-09-08 15:20:00 | 894.20 | 892.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:35:00 | 908.65 | 897.79 | 0.00 | ORB-long ORB[889.00,900.00] vol=3.5x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 09:55:00 | 914.19 | 905.36 | 0.00 | T1 1.5R @ 914.19 |
| Stop hit — per-position SL triggered | 2025-09-10 11:50:00 | 908.65 | 911.28 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 09:30:00 | 934.85 | 938.91 | 0.00 | ORB-short ORB[936.05,945.00] vol=1.7x ATR=2.83 |
| Stop hit — per-position SL triggered | 2025-09-16 09:40:00 | 937.68 | 938.54 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:30:00 | 909.25 | 913.39 | 0.00 | ORB-short ORB[911.05,923.75] vol=2.0x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:00:00 | 905.23 | 911.43 | 0.00 | T1 1.5R @ 905.23 |
| Target hit | 2025-09-30 15:20:00 | 894.90 | 901.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2025-10-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:25:00 | 930.00 | 919.66 | 0.00 | ORB-long ORB[905.50,917.55] vol=3.6x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:30:00 | 935.81 | 921.22 | 0.00 | T1 1.5R @ 935.81 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 930.00 | 926.80 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-10-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 10:00:00 | 940.55 | 935.06 | 0.00 | ORB-long ORB[928.20,936.40] vol=2.3x ATR=3.47 |
| Stop hit — per-position SL triggered | 2025-10-07 10:05:00 | 937.08 | 935.78 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:35:00 | 936.20 | 928.71 | 0.00 | ORB-long ORB[920.80,932.85] vol=2.2x ATR=3.09 |
| Stop hit — per-position SL triggered | 2025-10-09 09:40:00 | 933.11 | 929.00 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:45:00 | 979.85 | 971.58 | 0.00 | ORB-long ORB[962.10,973.85] vol=4.6x ATR=4.75 |
| Stop hit — per-position SL triggered | 2025-10-13 09:55:00 | 975.10 | 972.36 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:10:00 | 1013.00 | 1004.64 | 0.00 | ORB-long ORB[995.60,1005.35] vol=1.9x ATR=3.81 |
| Stop hit — per-position SL triggered | 2025-10-15 11:55:00 | 1009.19 | 1006.75 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 09:35:00 | 983.75 | 988.57 | 0.00 | ORB-short ORB[985.50,995.00] vol=2.3x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-11-12 09:40:00 | 987.10 | 988.19 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:35:00 | 1003.55 | 1000.56 | 0.00 | ORB-long ORB[992.15,1003.00] vol=2.3x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:50:00 | 1008.04 | 1002.17 | 0.00 | T1 1.5R @ 1008.04 |
| Target hit | 2025-11-13 12:30:00 | 1009.55 | 1012.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 998.45 | 994.55 | 0.00 | ORB-long ORB[988.00,996.70] vol=2.3x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-11-14 09:40:00 | 995.10 | 994.95 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:30:00 | 1000.75 | 993.86 | 0.00 | ORB-long ORB[986.10,997.50] vol=1.7x ATR=4.06 |
| Stop hit — per-position SL triggered | 2025-11-17 09:40:00 | 996.69 | 994.67 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:45:00 | 974.80 | 976.01 | 0.00 | ORB-short ORB[977.30,988.35] vol=4.8x ATR=2.69 |
| Stop hit — per-position SL triggered | 2025-11-18 11:00:00 | 977.49 | 975.99 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 10:45:00 | 965.65 | 969.56 | 0.00 | ORB-short ORB[967.55,973.50] vol=7.8x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 11:30:00 | 961.20 | 969.06 | 0.00 | T1 1.5R @ 961.20 |
| Stop hit — per-position SL triggered | 2025-11-20 13:00:00 | 965.65 | 967.81 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:30:00 | 959.95 | 954.36 | 0.00 | ORB-long ORB[948.75,954.70] vol=2.1x ATR=3.36 |
| Stop hit — per-position SL triggered | 2025-11-21 09:55:00 | 956.59 | 956.20 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 11:15:00 | 918.85 | 924.69 | 0.00 | ORB-short ORB[920.00,930.50] vol=3.3x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 11:55:00 | 915.07 | 923.03 | 0.00 | T1 1.5R @ 915.07 |
| Stop hit — per-position SL triggered | 2025-12-04 12:25:00 | 918.85 | 922.23 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:05:00 | 907.05 | 910.99 | 0.00 | ORB-short ORB[915.00,919.80] vol=1.7x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:30:00 | 902.64 | 909.40 | 0.00 | T1 1.5R @ 902.64 |
| Stop hit — per-position SL triggered | 2025-12-05 10:50:00 | 907.05 | 908.72 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:15:00 | 833.85 | 838.19 | 0.00 | ORB-short ORB[837.05,847.00] vol=1.7x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-12-16 11:20:00 | 835.51 | 837.77 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:10:00 | 872.50 | 865.23 | 0.00 | ORB-long ORB[858.45,869.90] vol=1.6x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 10:25:00 | 877.69 | 868.21 | 0.00 | T1 1.5R @ 877.69 |
| Stop hit — per-position SL triggered | 2025-12-22 10:45:00 | 872.50 | 870.56 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:55:00 | 888.45 | 882.85 | 0.00 | ORB-long ORB[877.10,885.00] vol=4.5x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 10:00:00 | 892.58 | 883.80 | 0.00 | T1 1.5R @ 892.58 |
| Stop hit — per-position SL triggered | 2025-12-24 10:05:00 | 888.45 | 884.30 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 856.85 | 850.31 | 0.00 | ORB-long ORB[840.75,853.35] vol=1.6x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 853.98 | 850.57 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:45:00 | 856.85 | 853.01 | 0.00 | ORB-long ORB[845.90,854.00] vol=2.3x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:20:00 | 861.12 | 855.62 | 0.00 | T1 1.5R @ 861.12 |
| Target hit | 2026-01-02 15:20:00 | 866.50 | 862.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2026-01-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 11:10:00 | 840.00 | 848.56 | 0.00 | ORB-short ORB[851.70,856.95] vol=3.5x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-01-07 11:40:00 | 842.26 | 846.16 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-01-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:50:00 | 842.15 | 846.56 | 0.00 | ORB-short ORB[844.05,853.65] vol=3.0x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:00:00 | 838.77 | 845.31 | 0.00 | T1 1.5R @ 838.77 |
| Target hit | 2026-01-08 15:20:00 | 823.95 | 833.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2026-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 09:30:00 | 875.45 | 867.78 | 0.00 | ORB-long ORB[856.00,867.75] vol=2.7x ATR=4.00 |
| Stop hit — per-position SL triggered | 2026-01-16 09:35:00 | 871.45 | 868.44 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-02-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:55:00 | 801.60 | 794.06 | 0.00 | ORB-long ORB[786.15,797.20] vol=3.6x ATR=3.74 |
| Stop hit — per-position SL triggered | 2026-02-09 11:30:00 | 797.86 | 797.59 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 774.10 | 767.02 | 0.00 | ORB-long ORB[759.95,767.90] vol=2.0x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:45:00 | 778.07 | 768.19 | 0.00 | T1 1.5R @ 778.07 |
| Stop hit — per-position SL triggered | 2026-02-20 11:00:00 | 774.10 | 769.32 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 751.85 | 754.00 | 0.00 | ORB-short ORB[752.50,763.00] vol=3.1x ATR=4.00 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 755.85 | 754.06 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-02-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:35:00 | 724.85 | 733.23 | 0.00 | ORB-short ORB[733.10,740.60] vol=2.0x ATR=2.51 |
| Stop hit — per-position SL triggered | 2026-02-26 10:55:00 | 727.36 | 731.47 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 687.00 | 690.68 | 0.00 | ORB-short ORB[691.00,697.25] vol=2.2x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:10:00 | 683.08 | 688.18 | 0.00 | T1 1.5R @ 683.08 |
| Target hit | 2026-03-13 15:20:00 | 678.70 | 683.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2026-03-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:05:00 | 642.35 | 647.58 | 0.00 | ORB-short ORB[646.60,654.80] vol=1.5x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 637.91 | 645.48 | 0.00 | T1 1.5R @ 637.91 |
| Target hit | 2026-03-23 13:50:00 | 632.35 | 632.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 58 — BUY (started 2026-04-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 11:00:00 | 667.40 | 659.69 | 0.00 | ORB-long ORB[656.05,665.15] vol=2.0x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 11:35:00 | 671.53 | 662.13 | 0.00 | T1 1.5R @ 671.53 |
| Target hit | 2026-04-02 15:20:00 | 684.65 | 671.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2026-04-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:55:00 | 779.30 | 773.00 | 0.00 | ORB-long ORB[769.00,778.15] vol=1.6x ATR=2.41 |
| Stop hit — per-position SL triggered | 2026-04-10 11:15:00 | 776.89 | 773.69 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-04-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:00:00 | 818.60 | 815.50 | 0.00 | ORB-long ORB[809.35,815.60] vol=1.9x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:45:00 | 821.43 | 816.25 | 0.00 | T1 1.5R @ 821.43 |
| Target hit | 2026-04-21 15:00:00 | 820.75 | 823.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — SELL (started 2026-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:10:00 | 815.95 | 820.30 | 0.00 | ORB-short ORB[816.35,827.35] vol=1.7x ATR=2.13 |
| Stop hit — per-position SL triggered | 2026-04-22 11:50:00 | 818.08 | 819.82 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-04-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:05:00 | 799.15 | 803.89 | 0.00 | ORB-short ORB[802.20,809.40] vol=2.1x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:20:00 | 796.24 | 802.42 | 0.00 | T1 1.5R @ 796.24 |
| Target hit | 2026-04-23 15:20:00 | 792.80 | 794.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2026-05-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:10:00 | 901.90 | 893.74 | 0.00 | ORB-long ORB[884.50,894.45] vol=2.1x ATR=3.30 |
| Stop hit — per-position SL triggered | 2026-05-08 11:25:00 | 898.60 | 894.50 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-27 10:35:00 | 777.05 | 2025-05-27 11:10:00 | 779.85 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-05-30 09:45:00 | 819.30 | 2025-05-30 09:55:00 | 816.25 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-05 09:40:00 | 827.00 | 2025-06-05 09:45:00 | 832.12 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-06-05 09:40:00 | 827.00 | 2025-06-05 10:25:00 | 828.55 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2025-06-16 09:45:00 | 805.25 | 2025-06-16 09:50:00 | 809.09 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-06-17 09:30:00 | 826.55 | 2025-06-17 09:40:00 | 823.04 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-06-20 09:50:00 | 829.00 | 2025-06-20 09:55:00 | 825.21 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-06-24 10:00:00 | 851.10 | 2025-06-24 10:05:00 | 846.95 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-06-25 10:00:00 | 871.50 | 2025-06-25 10:05:00 | 868.28 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-27 10:45:00 | 880.40 | 2025-06-27 10:50:00 | 877.33 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-02 09:50:00 | 860.60 | 2025-07-02 10:00:00 | 856.80 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-07-02 09:50:00 | 860.60 | 2025-07-02 15:05:00 | 858.00 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-07-15 10:35:00 | 945.15 | 2025-07-15 15:20:00 | 944.70 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest1 | 2025-07-18 10:35:00 | 923.25 | 2025-07-18 10:50:00 | 926.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-21 10:45:00 | 925.80 | 2025-07-21 11:05:00 | 930.89 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-07-21 10:45:00 | 925.80 | 2025-07-21 14:15:00 | 928.60 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-22 11:00:00 | 922.50 | 2025-07-22 11:45:00 | 925.17 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-29 09:45:00 | 896.50 | 2025-07-29 10:25:00 | 891.63 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-08-01 10:00:00 | 908.70 | 2025-08-01 10:10:00 | 913.15 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-08-05 10:15:00 | 920.10 | 2025-08-05 13:00:00 | 914.86 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-08-05 10:15:00 | 920.10 | 2025-08-05 15:20:00 | 911.30 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2025-08-06 11:00:00 | 905.10 | 2025-08-06 11:35:00 | 900.34 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-08-06 11:00:00 | 905.10 | 2025-08-06 11:50:00 | 905.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 11:10:00 | 908.00 | 2025-08-07 11:25:00 | 903.67 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-08-07 11:10:00 | 908.00 | 2025-08-07 14:30:00 | 905.05 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2025-08-14 11:00:00 | 927.85 | 2025-08-14 11:50:00 | 930.42 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-19 11:00:00 | 951.95 | 2025-08-19 11:20:00 | 949.07 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-21 09:30:00 | 950.60 | 2025-08-21 09:40:00 | 946.61 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-08-21 09:30:00 | 950.60 | 2025-08-21 10:00:00 | 950.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 10:35:00 | 935.05 | 2025-08-22 10:45:00 | 931.14 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-08-22 10:35:00 | 935.05 | 2025-08-22 11:00:00 | 935.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 09:45:00 | 880.55 | 2025-09-01 09:55:00 | 887.83 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2025-09-01 09:45:00 | 880.55 | 2025-09-01 10:05:00 | 880.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-05 09:30:00 | 876.20 | 2025-09-05 09:35:00 | 872.82 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-08 10:40:00 | 881.20 | 2025-09-08 10:45:00 | 884.80 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-09-08 10:40:00 | 881.20 | 2025-09-08 15:20:00 | 894.20 | TARGET_HIT | 0.50 | 1.48% |
| BUY | retest1 | 2025-09-10 09:35:00 | 908.65 | 2025-09-10 09:55:00 | 914.19 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-09-10 09:35:00 | 908.65 | 2025-09-10 11:50:00 | 908.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-16 09:30:00 | 934.85 | 2025-09-16 09:40:00 | 937.68 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-30 10:30:00 | 909.25 | 2025-09-30 11:00:00 | 905.23 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-09-30 10:30:00 | 909.25 | 2025-09-30 15:20:00 | 894.90 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2025-10-06 10:25:00 | 930.00 | 2025-10-06 10:30:00 | 935.81 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-10-06 10:25:00 | 930.00 | 2025-10-06 11:15:00 | 930.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-07 10:00:00 | 940.55 | 2025-10-07 10:05:00 | 937.08 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-09 09:35:00 | 936.20 | 2025-10-09 09:40:00 | 933.11 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-13 09:45:00 | 979.85 | 2025-10-13 09:55:00 | 975.10 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-10-15 11:10:00 | 1013.00 | 2025-10-15 11:55:00 | 1009.19 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-11-12 09:35:00 | 983.75 | 2025-11-12 09:40:00 | 987.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-11-13 09:35:00 | 1003.55 | 2025-11-13 09:50:00 | 1008.04 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-11-13 09:35:00 | 1003.55 | 2025-11-13 12:30:00 | 1009.55 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2025-11-14 09:30:00 | 998.45 | 2025-11-14 09:40:00 | 995.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-11-17 09:30:00 | 1000.75 | 2025-11-17 09:40:00 | 996.69 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-11-18 10:45:00 | 974.80 | 2025-11-18 11:00:00 | 977.49 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-20 10:45:00 | 965.65 | 2025-11-20 11:30:00 | 961.20 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-11-20 10:45:00 | 965.65 | 2025-11-20 13:00:00 | 965.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-21 09:30:00 | 959.95 | 2025-11-21 09:55:00 | 956.59 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-04 11:15:00 | 918.85 | 2025-12-04 11:55:00 | 915.07 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-12-04 11:15:00 | 918.85 | 2025-12-04 12:25:00 | 918.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-05 10:05:00 | 907.05 | 2025-12-05 10:30:00 | 902.64 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-12-05 10:05:00 | 907.05 | 2025-12-05 10:50:00 | 907.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-16 11:15:00 | 833.85 | 2025-12-16 11:20:00 | 835.51 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-22 10:10:00 | 872.50 | 2025-12-22 10:25:00 | 877.69 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-12-22 10:10:00 | 872.50 | 2025-12-22 10:45:00 | 872.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-24 09:55:00 | 888.45 | 2025-12-24 10:00:00 | 892.58 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-12-24 09:55:00 | 888.45 | 2025-12-24 10:05:00 | 888.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 11:00:00 | 856.85 | 2025-12-31 11:15:00 | 853.98 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-01-02 09:45:00 | 856.85 | 2026-01-02 10:20:00 | 861.12 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-01-02 09:45:00 | 856.85 | 2026-01-02 15:20:00 | 866.50 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2026-01-07 11:10:00 | 840.00 | 2026-01-07 11:40:00 | 842.26 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-08 10:50:00 | 842.15 | 2026-01-08 11:00:00 | 838.77 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-08 10:50:00 | 842.15 | 2026-01-08 15:20:00 | 823.95 | TARGET_HIT | 0.50 | 2.16% |
| BUY | retest1 | 2026-01-16 09:30:00 | 875.45 | 2026-01-16 09:35:00 | 871.45 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-02-09 09:55:00 | 801.60 | 2026-02-09 11:30:00 | 797.86 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-02-20 10:35:00 | 774.10 | 2026-02-20 10:45:00 | 778.07 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-20 10:35:00 | 774.10 | 2026-02-20 11:00:00 | 774.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 751.85 | 2026-02-24 09:45:00 | 755.85 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2026-02-26 10:35:00 | 724.85 | 2026-02-26 10:55:00 | 727.36 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-13 09:50:00 | 687.00 | 2026-03-13 12:10:00 | 683.08 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-13 09:50:00 | 687.00 | 2026-03-13 15:20:00 | 678.70 | TARGET_HIT | 0.50 | 1.21% |
| SELL | retest1 | 2026-03-23 10:05:00 | 642.35 | 2026-03-23 10:15:00 | 637.91 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-23 10:05:00 | 642.35 | 2026-03-23 13:50:00 | 632.35 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2026-04-02 11:00:00 | 667.40 | 2026-04-02 11:35:00 | 671.53 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-02 11:00:00 | 667.40 | 2026-04-02 15:20:00 | 684.65 | TARGET_HIT | 0.50 | 2.58% |
| BUY | retest1 | 2026-04-10 10:55:00 | 779.30 | 2026-04-10 11:15:00 | 776.89 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-21 11:00:00 | 818.60 | 2026-04-21 11:45:00 | 821.43 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-21 11:00:00 | 818.60 | 2026-04-21 15:00:00 | 820.75 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2026-04-22 11:10:00 | 815.95 | 2026-04-22 11:50:00 | 818.08 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-23 11:05:00 | 799.15 | 2026-04-23 11:20:00 | 796.24 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-04-23 11:05:00 | 799.15 | 2026-04-23 15:20:00 | 792.80 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2026-05-08 11:10:00 | 901.90 | 2026-05-08 11:25:00 | 898.60 | STOP_HIT | 1.00 | -0.37% |
