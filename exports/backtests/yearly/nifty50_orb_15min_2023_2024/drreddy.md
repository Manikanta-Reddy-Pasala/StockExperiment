# DRREDDY (DRREDDY)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55354 bars)
- **Last close:** 1294.50
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
| ENTRY1 | 96 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 20 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 76
- **Target hits / Stop hits / Partials:** 20 / 76 / 32
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 9.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 84 | 35 | 41.7% | 14 | 49 | 21 | 0.06% | 5.3% |
| BUY @ 2nd Alert (retest1) | 84 | 35 | 41.7% | 14 | 49 | 21 | 0.06% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 44 | 17 | 38.6% | 6 | 27 | 11 | 0.09% | 3.8% |
| SELL @ 2nd Alert (retest1) | 44 | 17 | 38.6% | 6 | 27 | 11 | 0.09% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 128 | 52 | 40.6% | 20 | 76 | 32 | 0.07% | 9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-16 11:15:00 | 903.20 | 906.04 | 0.00 | ORB-short ORB[904.00,909.58] vol=1.6x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-16 11:35:00 | 901.31 | 905.58 | 0.00 | T1 1.5R @ 901.31 |
| Stop hit — per-position SL triggered | 2023-05-16 11:55:00 | 903.20 | 905.41 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 10:50:00 | 899.85 | 905.15 | 0.00 | ORB-short ORB[903.22,909.01] vol=1.6x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 11:15:00 | 897.37 | 904.12 | 0.00 | T1 1.5R @ 897.37 |
| Target hit | 2023-05-17 15:15:00 | 897.20 | 896.97 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2023-05-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 09:30:00 | 893.49 | 896.86 | 0.00 | ORB-short ORB[896.04,900.44] vol=2.1x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-18 09:55:00 | 891.02 | 895.24 | 0.00 | T1 1.5R @ 891.02 |
| Stop hit — per-position SL triggered | 2023-05-18 10:40:00 | 893.49 | 893.85 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 09:30:00 | 888.51 | 884.47 | 0.00 | ORB-long ORB[876.85,887.51] vol=1.5x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-22 09:45:00 | 891.14 | 886.35 | 0.00 | T1 1.5R @ 891.14 |
| Stop hit — per-position SL triggered | 2023-05-22 10:00:00 | 888.51 | 887.39 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:55:00 | 904.60 | 899.89 | 0.00 | ORB-long ORB[891.98,903.00] vol=1.9x ATR=1.78 |
| Stop hit — per-position SL triggered | 2023-05-24 10:05:00 | 902.82 | 900.27 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-05-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 11:00:00 | 897.16 | 901.71 | 0.00 | ORB-short ORB[900.61,906.78] vol=2.1x ATR=1.44 |
| Stop hit — per-position SL triggered | 2023-05-25 12:00:00 | 898.60 | 900.39 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 09:40:00 | 911.60 | 907.62 | 0.00 | ORB-long ORB[904.00,909.58] vol=3.1x ATR=1.57 |
| Stop hit — per-position SL triggered | 2023-05-29 09:45:00 | 910.03 | 908.28 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-05-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 10:45:00 | 900.30 | 905.94 | 0.00 | ORB-short ORB[903.08,911.25] vol=2.0x ATR=1.59 |
| Stop hit — per-position SL triggered | 2023-05-30 11:00:00 | 901.89 | 905.32 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 10:55:00 | 919.60 | 916.45 | 0.00 | ORB-long ORB[913.01,919.20] vol=1.6x ATR=1.47 |
| Stop hit — per-position SL triggered | 2023-06-02 11:00:00 | 918.13 | 916.53 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 10:30:00 | 926.40 | 924.42 | 0.00 | ORB-long ORB[921.00,925.15] vol=1.7x ATR=1.56 |
| Stop hit — per-position SL triggered | 2023-06-06 10:45:00 | 924.84 | 924.52 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:05:00 | 940.51 | 936.44 | 0.00 | ORB-long ORB[932.48,940.06] vol=1.7x ATR=1.76 |
| Stop hit — per-position SL triggered | 2023-06-13 10:15:00 | 938.75 | 936.88 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 11:00:00 | 957.61 | 949.82 | 0.00 | ORB-long ORB[939.20,948.60] vol=1.7x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 11:05:00 | 959.96 | 950.79 | 0.00 | T1 1.5R @ 959.96 |
| Target hit | 2023-06-15 15:20:00 | 960.70 | 959.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2023-06-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 10:55:00 | 986.58 | 991.56 | 0.00 | ORB-short ORB[990.20,1004.80] vol=3.0x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 13:20:00 | 982.87 | 989.06 | 0.00 | T1 1.5R @ 982.87 |
| Target hit | 2023-06-19 15:20:00 | 975.00 | 985.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2023-06-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-23 10:25:00 | 994.59 | 986.51 | 0.00 | ORB-long ORB[980.20,989.45] vol=1.8x ATR=2.21 |
| Stop hit — per-position SL triggered | 2023-06-23 10:30:00 | 992.38 | 986.93 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-06-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:45:00 | 1001.99 | 995.76 | 0.00 | ORB-long ORB[990.40,997.98] vol=1.6x ATR=2.65 |
| Stop hit — per-position SL triggered | 2023-06-26 09:50:00 | 999.34 | 996.61 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-06-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 10:25:00 | 1014.99 | 1008.09 | 0.00 | ORB-long ORB[1003.20,1009.57] vol=2.0x ATR=2.69 |
| Stop hit — per-position SL triggered | 2023-06-27 10:30:00 | 1012.30 | 1008.52 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-06-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 09:30:00 | 1011.66 | 1007.45 | 0.00 | ORB-long ORB[1000.87,1009.12] vol=1.7x ATR=2.16 |
| Stop hit — per-position SL triggered | 2023-06-28 09:35:00 | 1009.50 | 1007.84 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-06-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 10:40:00 | 1032.00 | 1029.23 | 0.00 | ORB-long ORB[1022.02,1028.20] vol=1.5x ATR=1.94 |
| Stop hit — per-position SL triggered | 2023-06-30 11:00:00 | 1030.06 | 1029.37 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 09:40:00 | 1030.48 | 1026.46 | 0.00 | ORB-long ORB[1017.01,1027.99] vol=1.7x ATR=2.10 |
| Stop hit — per-position SL triggered | 2023-07-04 10:00:00 | 1028.38 | 1027.37 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:45:00 | 1043.65 | 1038.60 | 0.00 | ORB-long ORB[1034.00,1039.80] vol=2.4x ATR=2.36 |
| Stop hit — per-position SL triggered | 2023-07-06 10:00:00 | 1041.29 | 1039.90 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 10:00:00 | 1037.75 | 1032.63 | 0.00 | ORB-long ORB[1026.00,1031.71] vol=1.8x ATR=1.87 |
| Stop hit — per-position SL triggered | 2023-07-12 10:15:00 | 1035.88 | 1033.16 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-07-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-14 09:30:00 | 1015.80 | 1023.00 | 0.00 | ORB-short ORB[1020.65,1032.98] vol=2.1x ATR=2.72 |
| Stop hit — per-position SL triggered | 2023-07-14 09:35:00 | 1018.52 | 1021.94 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 11:15:00 | 1034.92 | 1032.58 | 0.00 | ORB-long ORB[1018.60,1033.19] vol=1.8x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 11:20:00 | 1038.17 | 1032.96 | 0.00 | T1 1.5R @ 1038.17 |
| Target hit | 2023-07-17 15:20:00 | 1045.55 | 1039.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2023-07-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:30:00 | 1045.28 | 1042.35 | 0.00 | ORB-long ORB[1032.32,1044.99] vol=1.9x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 09:35:00 | 1048.83 | 1043.92 | 0.00 | T1 1.5R @ 1048.83 |
| Target hit | 2023-07-19 09:50:00 | 1047.33 | 1047.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — BUY (started 2023-07-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 09:55:00 | 1061.89 | 1055.89 | 0.00 | ORB-long ORB[1048.00,1058.70] vol=3.2x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 10:00:00 | 1066.36 | 1057.28 | 0.00 | T1 1.5R @ 1066.36 |
| Target hit | 2023-07-20 15:20:00 | 1066.51 | 1066.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2023-07-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:40:00 | 1088.96 | 1086.56 | 0.00 | ORB-long ORB[1082.30,1088.76] vol=2.1x ATR=2.66 |
| Stop hit — per-position SL triggered | 2023-07-26 09:55:00 | 1086.30 | 1087.36 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-07-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:35:00 | 1120.80 | 1114.27 | 0.00 | ORB-long ORB[1102.40,1118.55] vol=1.6x ATR=4.32 |
| Stop hit — per-position SL triggered | 2023-07-28 10:05:00 | 1116.48 | 1116.50 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 09:30:00 | 1137.19 | 1130.43 | 0.00 | ORB-long ORB[1121.42,1133.79] vol=2.5x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 09:50:00 | 1141.81 | 1133.62 | 0.00 | T1 1.5R @ 1141.81 |
| Stop hit — per-position SL triggered | 2023-08-01 10:30:00 | 1137.19 | 1134.83 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-08-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 09:50:00 | 1144.49 | 1140.70 | 0.00 | ORB-long ORB[1130.40,1143.84] vol=2.4x ATR=2.70 |
| Stop hit — per-position SL triggered | 2023-08-08 10:20:00 | 1141.79 | 1141.88 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-08-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 09:35:00 | 1170.71 | 1167.05 | 0.00 | ORB-long ORB[1160.00,1169.03] vol=1.7x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-14 09:45:00 | 1174.49 | 1168.46 | 0.00 | T1 1.5R @ 1174.49 |
| Stop hit — per-position SL triggered | 2023-08-14 11:30:00 | 1170.71 | 1170.32 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-16 09:30:00 | 1157.20 | 1163.60 | 0.00 | ORB-short ORB[1159.10,1169.60] vol=2.4x ATR=2.55 |
| Stop hit — per-position SL triggered | 2023-08-16 09:40:00 | 1159.75 | 1162.60 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 11:10:00 | 1189.53 | 1192.53 | 0.00 | ORB-short ORB[1192.09,1197.94] vol=1.6x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 11:15:00 | 1186.64 | 1192.00 | 0.00 | T1 1.5R @ 1186.64 |
| Stop hit — per-position SL triggered | 2023-08-24 11:50:00 | 1189.53 | 1191.20 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 09:30:00 | 1161.61 | 1167.53 | 0.00 | ORB-short ORB[1163.61,1178.00] vol=2.1x ATR=4.28 |
| Stop hit — per-position SL triggered | 2023-08-25 09:40:00 | 1165.89 | 1166.72 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 11:00:00 | 1164.02 | 1156.43 | 0.00 | ORB-long ORB[1146.00,1159.98] vol=2.9x ATR=2.51 |
| Stop hit — per-position SL triggered | 2023-08-28 11:20:00 | 1161.51 | 1157.09 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 09:30:00 | 1148.50 | 1151.75 | 0.00 | ORB-short ORB[1149.03,1157.30] vol=2.0x ATR=2.85 |
| Stop hit — per-position SL triggered | 2023-08-29 10:30:00 | 1151.35 | 1149.89 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 09:50:00 | 1124.06 | 1120.54 | 0.00 | ORB-long ORB[1115.71,1122.68] vol=1.7x ATR=1.91 |
| Stop hit — per-position SL triggered | 2023-09-04 09:55:00 | 1122.15 | 1120.79 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 10:20:00 | 1124.92 | 1117.71 | 0.00 | ORB-long ORB[1111.85,1117.95] vol=2.1x ATR=2.17 |
| Stop hit — per-position SL triggered | 2023-09-06 10:40:00 | 1122.75 | 1119.09 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 10:50:00 | 1118.02 | 1120.57 | 0.00 | ORB-short ORB[1120.83,1126.73] vol=1.7x ATR=2.12 |
| Target hit | 2023-09-08 15:20:00 | 1115.97 | 1118.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2023-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 09:35:00 | 1130.99 | 1127.30 | 0.00 | ORB-long ORB[1117.01,1129.20] vol=1.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-09-11 09:45:00 | 1129.04 | 1127.99 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 1142.87 | 1137.66 | 0.00 | ORB-long ORB[1125.20,1141.69] vol=3.0x ATR=2.51 |
| Stop hit — per-position SL triggered | 2023-09-12 09:35:00 | 1140.36 | 1138.23 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 09:45:00 | 1130.61 | 1133.32 | 0.00 | ORB-short ORB[1131.91,1144.37] vol=2.2x ATR=2.49 |
| Stop hit — per-position SL triggered | 2023-09-20 09:55:00 | 1133.10 | 1133.17 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-09-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:30:00 | 1126.46 | 1131.43 | 0.00 | ORB-short ORB[1129.21,1138.78] vol=1.6x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 09:40:00 | 1121.98 | 1128.42 | 0.00 | T1 1.5R @ 1121.98 |
| Target hit | 2023-09-22 15:20:00 | 1104.05 | 1110.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2023-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 09:30:00 | 1103.32 | 1099.28 | 0.00 | ORB-long ORB[1089.80,1101.10] vol=1.6x ATR=2.75 |
| Stop hit — per-position SL triggered | 2023-09-26 09:50:00 | 1100.57 | 1100.16 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-09-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:20:00 | 1109.99 | 1102.09 | 0.00 | ORB-long ORB[1090.65,1099.72] vol=1.5x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 10:35:00 | 1114.71 | 1103.97 | 0.00 | T1 1.5R @ 1114.71 |
| Target hit | 2023-09-29 15:20:00 | 1116.00 | 1116.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 11:15:00 | 1084.69 | 1092.30 | 0.00 | ORB-short ORB[1085.85,1093.73] vol=1.7x ATR=2.21 |
| Stop hit — per-position SL triggered | 2023-10-06 11:20:00 | 1086.90 | 1091.87 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-10-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 09:40:00 | 1092.77 | 1089.04 | 0.00 | ORB-long ORB[1080.03,1089.58] vol=1.6x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:50:00 | 1096.95 | 1090.31 | 0.00 | T1 1.5R @ 1096.95 |
| Target hit | 2023-10-09 15:10:00 | 1099.02 | 1099.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2023-10-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 10:05:00 | 1116.19 | 1111.78 | 0.00 | ORB-long ORB[1102.00,1114.81] vol=2.2x ATR=2.41 |
| Stop hit — per-position SL triggered | 2023-10-11 10:15:00 | 1113.78 | 1112.49 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-10-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 10:55:00 | 1102.10 | 1097.44 | 0.00 | ORB-long ORB[1084.32,1098.29] vol=1.8x ATR=2.01 |
| Stop hit — per-position SL triggered | 2023-10-13 11:25:00 | 1100.09 | 1098.36 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:50:00 | 1109.61 | 1103.84 | 0.00 | ORB-long ORB[1097.01,1106.00] vol=1.5x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 11:25:00 | 1112.49 | 1105.93 | 0.00 | T1 1.5R @ 1112.49 |
| Stop hit — per-position SL triggered | 2023-10-17 13:15:00 | 1109.61 | 1108.04 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-10-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:40:00 | 1123.80 | 1116.67 | 0.00 | ORB-long ORB[1108.36,1118.59] vol=1.8x ATR=2.03 |
| Stop hit — per-position SL triggered | 2023-10-18 09:50:00 | 1121.77 | 1118.33 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 10:45:00 | 1071.20 | 1071.91 | 0.00 | ORB-short ORB[1072.78,1084.08] vol=2.1x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 13:00:00 | 1067.04 | 1071.40 | 0.00 | T1 1.5R @ 1067.04 |
| Stop hit — per-position SL triggered | 2023-11-01 13:35:00 | 1071.20 | 1070.67 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 10:00:00 | 1082.00 | 1077.77 | 0.00 | ORB-long ORB[1063.96,1074.80] vol=1.6x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 10:15:00 | 1086.23 | 1079.56 | 0.00 | T1 1.5R @ 1086.23 |
| Target hit | 2023-11-07 13:10:00 | 1085.24 | 1086.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — BUY (started 2023-11-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 09:45:00 | 1091.79 | 1088.80 | 0.00 | ORB-long ORB[1084.17,1091.16] vol=1.6x ATR=2.38 |
| Stop hit — per-position SL triggered | 2023-11-16 10:00:00 | 1089.41 | 1089.09 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 10:25:00 | 1119.49 | 1114.91 | 0.00 | ORB-long ORB[1109.00,1115.00] vol=1.6x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 11:05:00 | 1123.43 | 1116.81 | 0.00 | T1 1.5R @ 1123.43 |
| Target hit | 2023-11-17 15:20:00 | 1122.45 | 1120.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2023-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:55:00 | 1129.49 | 1124.97 | 0.00 | ORB-long ORB[1121.10,1126.64] vol=1.6x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 10:10:00 | 1132.51 | 1127.10 | 0.00 | T1 1.5R @ 1132.51 |
| Target hit | 2023-11-21 13:15:00 | 1130.65 | 1130.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — BUY (started 2023-11-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:35:00 | 1142.66 | 1138.41 | 0.00 | ORB-long ORB[1131.02,1137.60] vol=2.4x ATR=2.42 |
| Stop hit — per-position SL triggered | 2023-11-22 09:40:00 | 1140.24 | 1138.71 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-29 10:55:00 | 1133.21 | 1136.54 | 0.00 | ORB-short ORB[1133.95,1140.55] vol=2.0x ATR=2.13 |
| Stop hit — per-position SL triggered | 2023-11-29 11:15:00 | 1135.34 | 1135.90 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 10:45:00 | 1156.50 | 1152.92 | 0.00 | ORB-long ORB[1143.20,1156.00] vol=2.8x ATR=3.08 |
| Stop hit — per-position SL triggered | 2023-11-30 10:55:00 | 1153.42 | 1153.02 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-12-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 10:55:00 | 1149.44 | 1156.70 | 0.00 | ORB-short ORB[1157.00,1164.00] vol=2.9x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 11:50:00 | 1146.27 | 1152.85 | 0.00 | T1 1.5R @ 1146.27 |
| Target hit | 2023-12-06 15:20:00 | 1147.02 | 1148.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2023-12-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 09:50:00 | 1119.00 | 1106.95 | 0.00 | ORB-long ORB[1095.48,1110.00] vol=1.5x ATR=3.67 |
| Stop hit — per-position SL triggered | 2023-12-12 09:55:00 | 1115.33 | 1108.44 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:55:00 | 1126.03 | 1124.27 | 0.00 | ORB-long ORB[1112.00,1124.94] vol=3.4x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 11:50:00 | 1130.74 | 1125.95 | 0.00 | T1 1.5R @ 1130.74 |
| Stop hit — per-position SL triggered | 2023-12-22 12:20:00 | 1126.03 | 1126.22 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:10:00 | 1133.11 | 1128.83 | 0.00 | ORB-long ORB[1120.60,1129.50] vol=1.7x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 11:15:00 | 1136.70 | 1130.97 | 0.00 | T1 1.5R @ 1136.70 |
| Stop hit — per-position SL triggered | 2023-12-26 12:35:00 | 1133.11 | 1134.31 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-01-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 09:50:00 | 1193.34 | 1185.41 | 0.00 | ORB-long ORB[1177.64,1185.36] vol=2.1x ATR=3.27 |
| Stop hit — per-position SL triggered | 2024-01-03 09:55:00 | 1190.07 | 1186.18 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-01-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 10:00:00 | 1138.55 | 1147.35 | 0.00 | ORB-short ORB[1144.22,1159.60] vol=1.5x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-01-10 10:05:00 | 1141.77 | 1146.65 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-01-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 10:50:00 | 1150.66 | 1145.19 | 0.00 | ORB-long ORB[1141.22,1147.98] vol=2.9x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 11:05:00 | 1153.98 | 1146.62 | 0.00 | T1 1.5R @ 1153.98 |
| Target hit | 2024-01-15 15:20:00 | 1156.99 | 1154.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2024-01-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 11:00:00 | 1129.89 | 1132.23 | 0.00 | ORB-short ORB[1133.65,1140.00] vol=2.5x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-01-20 11:10:00 | 1131.76 | 1132.33 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 09:35:00 | 1144.56 | 1136.17 | 0.00 | ORB-long ORB[1128.19,1140.60] vol=3.0x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-01-23 09:50:00 | 1141.92 | 1138.16 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-01-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 09:40:00 | 1154.60 | 1147.51 | 0.00 | ORB-long ORB[1138.60,1151.59] vol=1.8x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-24 09:55:00 | 1160.92 | 1151.43 | 0.00 | T1 1.5R @ 1160.92 |
| Target hit | 2024-01-24 15:20:00 | 1181.40 | 1170.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 1227.36 | 1233.58 | 0.00 | ORB-short ORB[1234.05,1243.99] vol=2.1x ATR=2.44 |
| Stop hit — per-position SL triggered | 2024-02-08 11:30:00 | 1229.80 | 1232.15 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 09:45:00 | 1278.99 | 1274.07 | 0.00 | ORB-long ORB[1262.21,1275.99] vol=2.6x ATR=4.95 |
| Stop hit — per-position SL triggered | 2024-02-13 10:20:00 | 1274.04 | 1275.01 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-02-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 09:50:00 | 1260.40 | 1255.63 | 0.00 | ORB-long ORB[1249.47,1255.59] vol=1.8x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 10:20:00 | 1265.65 | 1258.00 | 0.00 | T1 1.5R @ 1265.65 |
| Target hit | 2024-02-16 14:30:00 | 1261.76 | 1262.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — SELL (started 2024-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 09:30:00 | 1283.04 | 1288.63 | 0.00 | ORB-short ORB[1284.40,1293.00] vol=2.2x ATR=2.60 |
| Stop hit — per-position SL triggered | 2024-02-20 09:35:00 | 1285.64 | 1288.32 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-02-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 09:55:00 | 1290.24 | 1282.97 | 0.00 | ORB-long ORB[1272.60,1281.71] vol=1.5x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-23 10:00:00 | 1295.60 | 1286.48 | 0.00 | T1 1.5R @ 1295.60 |
| Target hit | 2024-02-23 11:15:00 | 1291.06 | 1291.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — BUY (started 2024-02-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-28 09:40:00 | 1299.69 | 1294.69 | 0.00 | ORB-long ORB[1288.00,1294.16] vol=2.6x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-02-28 09:45:00 | 1296.86 | 1295.18 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-03-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 11:10:00 | 1279.84 | 1283.04 | 0.00 | ORB-short ORB[1281.69,1292.09] vol=1.8x ATR=2.94 |
| Stop hit — per-position SL triggered | 2024-03-01 11:40:00 | 1282.78 | 1282.67 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 11:00:00 | 1251.42 | 1261.39 | 0.00 | ORB-short ORB[1258.29,1268.43] vol=1.6x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-03-05 11:30:00 | 1253.81 | 1258.89 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 09:30:00 | 1276.46 | 1279.31 | 0.00 | ORB-short ORB[1277.00,1285.60] vol=1.7x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:40:00 | 1271.63 | 1278.02 | 0.00 | T1 1.5R @ 1271.63 |
| Target hit | 2024-03-12 15:20:00 | 1261.74 | 1265.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — SELL (started 2024-03-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 11:00:00 | 1250.00 | 1252.78 | 0.00 | ORB-short ORB[1252.36,1261.71] vol=2.2x ATR=2.93 |
| Stop hit — per-position SL triggered | 2024-03-15 11:15:00 | 1252.93 | 1252.53 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-18 11:05:00 | 1268.01 | 1264.32 | 0.00 | ORB-long ORB[1253.61,1261.73] vol=1.7x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-03-18 14:10:00 | 1265.60 | 1266.52 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-03-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 11:00:00 | 1228.44 | 1222.07 | 0.00 | ORB-long ORB[1212.00,1225.36] vol=2.3x ATR=2.76 |
| Stop hit — per-position SL triggered | 2024-03-21 11:20:00 | 1225.68 | 1222.50 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-03-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 11:05:00 | 1232.96 | 1227.46 | 0.00 | ORB-long ORB[1221.32,1231.24] vol=1.6x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 11:20:00 | 1236.91 | 1228.49 | 0.00 | T1 1.5R @ 1236.91 |
| Target hit | 2024-03-22 15:20:00 | 1240.57 | 1239.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — SELL (started 2024-03-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 10:00:00 | 1226.71 | 1230.98 | 0.00 | ORB-short ORB[1229.62,1238.58] vol=2.0x ATR=4.08 |
| Stop hit — per-position SL triggered | 2024-03-26 10:15:00 | 1230.79 | 1230.79 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-03-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:45:00 | 1233.27 | 1225.12 | 0.00 | ORB-long ORB[1215.60,1229.00] vol=1.8x ATR=4.01 |
| Stop hit — per-position SL triggered | 2024-03-28 10:10:00 | 1229.26 | 1227.24 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 10:55:00 | 1243.72 | 1239.52 | 0.00 | ORB-long ORB[1231.58,1243.28] vol=2.7x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 11:15:00 | 1247.43 | 1241.17 | 0.00 | T1 1.5R @ 1247.43 |
| Stop hit — per-position SL triggered | 2024-04-01 11:40:00 | 1243.72 | 1241.67 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-04-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-02 09:35:00 | 1242.99 | 1246.17 | 0.00 | ORB-short ORB[1243.61,1253.79] vol=3.3x ATR=3.30 |
| Stop hit — per-position SL triggered | 2024-04-02 10:00:00 | 1246.29 | 1245.68 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-04-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-03 10:45:00 | 1233.93 | 1240.39 | 0.00 | ORB-short ORB[1238.42,1253.60] vol=1.8x ATR=2.90 |
| Stop hit — per-position SL triggered | 2024-04-03 11:30:00 | 1236.83 | 1239.24 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 1214.64 | 1221.30 | 0.00 | ORB-short ORB[1221.53,1231.40] vol=3.3x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 11:10:00 | 1209.93 | 1220.26 | 0.00 | T1 1.5R @ 1209.93 |
| Stop hit — per-position SL triggered | 2024-04-04 11:15:00 | 1214.64 | 1220.19 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 11:05:00 | 1229.80 | 1239.40 | 0.00 | ORB-short ORB[1231.03,1245.35] vol=2.0x ATR=2.79 |
| Stop hit — per-position SL triggered | 2024-04-08 11:25:00 | 1232.59 | 1238.18 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-04-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-09 10:45:00 | 1234.99 | 1238.57 | 0.00 | ORB-short ORB[1235.05,1246.04] vol=2.4x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-04-09 10:50:00 | 1237.02 | 1238.44 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 11:00:00 | 1212.31 | 1204.39 | 0.00 | ORB-long ORB[1198.00,1204.79] vol=1.6x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-04-16 11:15:00 | 1209.44 | 1205.14 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 10:55:00 | 1211.00 | 1203.34 | 0.00 | ORB-long ORB[1190.70,1207.80] vol=2.2x ATR=2.92 |
| Stop hit — per-position SL triggered | 2024-04-22 11:10:00 | 1208.08 | 1204.21 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:55:00 | 1196.40 | 1191.67 | 0.00 | ORB-long ORB[1186.40,1195.58] vol=1.5x ATR=2.58 |
| Stop hit — per-position SL triggered | 2024-04-24 13:05:00 | 1193.82 | 1193.53 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2024-05-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 10:55:00 | 1250.99 | 1248.23 | 0.00 | ORB-long ORB[1239.05,1249.76] vol=2.1x ATR=3.13 |
| Stop hit — per-position SL triggered | 2024-05-02 11:15:00 | 1247.86 | 1248.69 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2024-05-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 10:45:00 | 1269.53 | 1263.34 | 0.00 | ORB-long ORB[1251.60,1259.98] vol=1.6x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1266.79 | 1264.99 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2024-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 11:10:00 | 1251.79 | 1262.00 | 0.00 | ORB-short ORB[1259.08,1270.00] vol=1.5x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:35:00 | 1246.69 | 1260.85 | 0.00 | T1 1.5R @ 1246.69 |
| Stop hit — per-position SL triggered | 2024-05-07 11:40:00 | 1251.79 | 1259.99 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-05-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 11:05:00 | 1182.66 | 1174.89 | 0.00 | ORB-long ORB[1163.62,1179.05] vol=1.8x ATR=2.59 |
| Stop hit — per-position SL triggered | 2024-05-10 11:10:00 | 1180.07 | 1175.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-16 11:15:00 | 903.20 | 2023-05-16 11:35:00 | 901.31 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2023-05-16 11:15:00 | 903.20 | 2023-05-16 11:55:00 | 903.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-17 10:50:00 | 899.85 | 2023-05-17 11:15:00 | 897.37 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-05-17 10:50:00 | 899.85 | 2023-05-17 15:15:00 | 897.20 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2023-05-18 09:30:00 | 893.49 | 2023-05-18 09:55:00 | 891.02 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-05-18 09:30:00 | 893.49 | 2023-05-18 10:40:00 | 893.49 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-22 09:30:00 | 888.51 | 2023-05-22 09:45:00 | 891.14 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-05-22 09:30:00 | 888.51 | 2023-05-22 10:00:00 | 888.51 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-24 09:55:00 | 904.60 | 2023-05-24 10:05:00 | 902.82 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-05-25 11:00:00 | 897.16 | 2023-05-25 12:00:00 | 898.60 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-05-29 09:40:00 | 911.60 | 2023-05-29 09:45:00 | 910.03 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-05-30 10:45:00 | 900.30 | 2023-05-30 11:00:00 | 901.89 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-06-02 10:55:00 | 919.60 | 2023-06-02 11:00:00 | 918.13 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-06-06 10:30:00 | 926.40 | 2023-06-06 10:45:00 | 924.84 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-06-13 10:05:00 | 940.51 | 2023-06-13 10:15:00 | 938.75 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-06-15 11:00:00 | 957.61 | 2023-06-15 11:05:00 | 959.96 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-06-15 11:00:00 | 957.61 | 2023-06-15 15:20:00 | 960.70 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2023-06-19 10:55:00 | 986.58 | 2023-06-19 13:20:00 | 982.87 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-06-19 10:55:00 | 986.58 | 2023-06-19 15:20:00 | 975.00 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2023-06-23 10:25:00 | 994.59 | 2023-06-23 10:30:00 | 992.38 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-26 09:45:00 | 1001.99 | 2023-06-26 09:50:00 | 999.34 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-27 10:25:00 | 1014.99 | 2023-06-27 10:30:00 | 1012.30 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-28 09:30:00 | 1011.66 | 2023-06-28 09:35:00 | 1009.50 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-30 10:40:00 | 1032.00 | 2023-06-30 11:00:00 | 1030.06 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-07-04 09:40:00 | 1030.48 | 2023-07-04 10:00:00 | 1028.38 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-06 09:45:00 | 1043.65 | 2023-07-06 10:00:00 | 1041.29 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-12 10:00:00 | 1037.75 | 2023-07-12 10:15:00 | 1035.88 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-07-14 09:30:00 | 1015.80 | 2023-07-14 09:35:00 | 1018.52 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-17 11:15:00 | 1034.92 | 2023-07-17 11:20:00 | 1038.17 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-07-17 11:15:00 | 1034.92 | 2023-07-17 15:20:00 | 1045.55 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2023-07-19 09:30:00 | 1045.28 | 2023-07-19 09:35:00 | 1048.83 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-07-19 09:30:00 | 1045.28 | 2023-07-19 09:50:00 | 1047.33 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2023-07-20 09:55:00 | 1061.89 | 2023-07-20 10:00:00 | 1066.36 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-07-20 09:55:00 | 1061.89 | 2023-07-20 15:20:00 | 1066.51 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2023-07-26 09:40:00 | 1088.96 | 2023-07-26 09:55:00 | 1086.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-28 09:35:00 | 1120.80 | 2023-07-28 10:05:00 | 1116.48 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-08-01 09:30:00 | 1137.19 | 2023-08-01 09:50:00 | 1141.81 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-08-01 09:30:00 | 1137.19 | 2023-08-01 10:30:00 | 1137.19 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-08 09:50:00 | 1144.49 | 2023-08-08 10:20:00 | 1141.79 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-08-14 09:35:00 | 1170.71 | 2023-08-14 09:45:00 | 1174.49 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-08-14 09:35:00 | 1170.71 | 2023-08-14 11:30:00 | 1170.71 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-16 09:30:00 | 1157.20 | 2023-08-16 09:40:00 | 1159.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-24 11:10:00 | 1189.53 | 2023-08-24 11:15:00 | 1186.64 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-08-24 11:10:00 | 1189.53 | 2023-08-24 11:50:00 | 1189.53 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-25 09:30:00 | 1161.61 | 2023-08-25 09:40:00 | 1165.89 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-08-28 11:00:00 | 1164.02 | 2023-08-28 11:20:00 | 1161.51 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-29 09:30:00 | 1148.50 | 2023-08-29 10:30:00 | 1151.35 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-04 09:50:00 | 1124.06 | 2023-09-04 09:55:00 | 1122.15 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-09-06 10:20:00 | 1124.92 | 2023-09-06 10:40:00 | 1122.75 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-09-08 10:50:00 | 1118.02 | 2023-09-08 15:20:00 | 1115.97 | TARGET_HIT | 1.00 | 0.18% |
| BUY | retest1 | 2023-09-11 09:35:00 | 1130.99 | 2023-09-11 09:45:00 | 1129.04 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-09-12 09:30:00 | 1142.87 | 2023-09-12 09:35:00 | 1140.36 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-09-20 09:45:00 | 1130.61 | 2023-09-20 09:55:00 | 1133.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-09-22 09:30:00 | 1126.46 | 2023-09-22 09:40:00 | 1121.98 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-09-22 09:30:00 | 1126.46 | 2023-09-22 15:20:00 | 1104.05 | TARGET_HIT | 0.50 | 1.99% |
| BUY | retest1 | 2023-09-26 09:30:00 | 1103.32 | 2023-09-26 09:50:00 | 1100.57 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-29 10:20:00 | 1109.99 | 2023-09-29 10:35:00 | 1114.71 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-09-29 10:20:00 | 1109.99 | 2023-09-29 15:20:00 | 1116.00 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2023-10-06 11:15:00 | 1084.69 | 2023-10-06 11:20:00 | 1086.90 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-10-09 09:40:00 | 1092.77 | 2023-10-09 09:50:00 | 1096.95 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-10-09 09:40:00 | 1092.77 | 2023-10-09 15:10:00 | 1099.02 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2023-10-11 10:05:00 | 1116.19 | 2023-10-11 10:15:00 | 1113.78 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-10-13 10:55:00 | 1102.10 | 2023-10-13 11:25:00 | 1100.09 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-10-17 09:50:00 | 1109.61 | 2023-10-17 11:25:00 | 1112.49 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-10-17 09:50:00 | 1109.61 | 2023-10-17 13:15:00 | 1109.61 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-18 09:40:00 | 1123.80 | 2023-10-18 09:50:00 | 1121.77 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-11-01 10:45:00 | 1071.20 | 2023-11-01 13:00:00 | 1067.04 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-11-01 10:45:00 | 1071.20 | 2023-11-01 13:35:00 | 1071.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-07 10:00:00 | 1082.00 | 2023-11-07 10:15:00 | 1086.23 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-11-07 10:00:00 | 1082.00 | 2023-11-07 13:10:00 | 1085.24 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2023-11-16 09:45:00 | 1091.79 | 2023-11-16 10:00:00 | 1089.41 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-11-17 10:25:00 | 1119.49 | 2023-11-17 11:05:00 | 1123.43 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-11-17 10:25:00 | 1119.49 | 2023-11-17 15:20:00 | 1122.45 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2023-11-21 09:55:00 | 1129.49 | 2023-11-21 10:10:00 | 1132.51 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-11-21 09:55:00 | 1129.49 | 2023-11-21 13:15:00 | 1130.65 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2023-11-22 09:35:00 | 1142.66 | 2023-11-22 09:40:00 | 1140.24 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-11-29 10:55:00 | 1133.21 | 2023-11-29 11:15:00 | 1135.34 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-11-30 10:45:00 | 1156.50 | 2023-11-30 10:55:00 | 1153.42 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-12-06 10:55:00 | 1149.44 | 2023-12-06 11:50:00 | 1146.27 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-12-06 10:55:00 | 1149.44 | 2023-12-06 15:20:00 | 1147.02 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2023-12-12 09:50:00 | 1119.00 | 2023-12-12 09:55:00 | 1115.33 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-12-22 09:55:00 | 1126.03 | 2023-12-22 11:50:00 | 1130.74 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-12-22 09:55:00 | 1126.03 | 2023-12-22 12:20:00 | 1126.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 10:10:00 | 1133.11 | 2023-12-26 11:15:00 | 1136.70 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-12-26 10:10:00 | 1133.11 | 2023-12-26 12:35:00 | 1133.11 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-03 09:50:00 | 1193.34 | 2024-01-03 09:55:00 | 1190.07 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-10 10:00:00 | 1138.55 | 2024-01-10 10:05:00 | 1141.77 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-01-15 10:50:00 | 1150.66 | 2024-01-15 11:05:00 | 1153.98 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-01-15 10:50:00 | 1150.66 | 2024-01-15 15:20:00 | 1156.99 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2024-01-20 11:00:00 | 1129.89 | 2024-01-20 11:10:00 | 1131.76 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-01-23 09:35:00 | 1144.56 | 2024-01-23 09:50:00 | 1141.92 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-01-24 09:40:00 | 1154.60 | 2024-01-24 09:55:00 | 1160.92 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-01-24 09:40:00 | 1154.60 | 2024-01-24 15:20:00 | 1181.40 | TARGET_HIT | 0.50 | 2.32% |
| SELL | retest1 | 2024-02-08 11:00:00 | 1227.36 | 2024-02-08 11:30:00 | 1229.80 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-02-13 09:45:00 | 1278.99 | 2024-02-13 10:20:00 | 1274.04 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-02-16 09:50:00 | 1260.40 | 2024-02-16 10:20:00 | 1265.65 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-02-16 09:50:00 | 1260.40 | 2024-02-16 14:30:00 | 1261.76 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2024-02-20 09:30:00 | 1283.04 | 2024-02-20 09:35:00 | 1285.64 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-02-23 09:55:00 | 1290.24 | 2024-02-23 10:00:00 | 1295.60 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-02-23 09:55:00 | 1290.24 | 2024-02-23 11:15:00 | 1291.06 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2024-02-28 09:40:00 | 1299.69 | 2024-02-28 09:45:00 | 1296.86 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-03-01 11:10:00 | 1279.84 | 2024-03-01 11:40:00 | 1282.78 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-03-05 11:00:00 | 1251.42 | 2024-03-05 11:30:00 | 1253.81 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-03-12 09:30:00 | 1276.46 | 2024-03-12 09:40:00 | 1271.63 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-03-12 09:30:00 | 1276.46 | 2024-03-12 15:20:00 | 1261.74 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2024-03-15 11:00:00 | 1250.00 | 2024-03-15 11:15:00 | 1252.93 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-03-18 11:05:00 | 1268.01 | 2024-03-18 14:10:00 | 1265.60 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-03-21 11:00:00 | 1228.44 | 2024-03-21 11:20:00 | 1225.68 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-03-22 11:05:00 | 1232.96 | 2024-03-22 11:20:00 | 1236.91 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-03-22 11:05:00 | 1232.96 | 2024-03-22 15:20:00 | 1240.57 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2024-03-26 10:00:00 | 1226.71 | 2024-03-26 10:15:00 | 1230.79 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-28 09:45:00 | 1233.27 | 2024-03-28 10:10:00 | 1229.26 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-04-01 10:55:00 | 1243.72 | 2024-04-01 11:15:00 | 1247.43 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-04-01 10:55:00 | 1243.72 | 2024-04-01 11:40:00 | 1243.72 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-02 09:35:00 | 1242.99 | 2024-04-02 10:00:00 | 1246.29 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-04-03 10:45:00 | 1233.93 | 2024-04-03 11:30:00 | 1236.83 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-04-04 10:50:00 | 1214.64 | 2024-04-04 11:10:00 | 1209.93 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-04-04 10:50:00 | 1214.64 | 2024-04-04 11:15:00 | 1214.64 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-08 11:05:00 | 1229.80 | 2024-04-08 11:25:00 | 1232.59 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-04-09 10:45:00 | 1234.99 | 2024-04-09 10:50:00 | 1237.02 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-04-16 11:00:00 | 1212.31 | 2024-04-16 11:15:00 | 1209.44 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-22 10:55:00 | 1211.00 | 2024-04-22 11:10:00 | 1208.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-24 10:55:00 | 1196.40 | 2024-04-24 13:05:00 | 1193.82 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-05-02 10:55:00 | 1250.99 | 2024-05-02 11:15:00 | 1247.86 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-03 10:45:00 | 1269.53 | 2024-05-03 11:15:00 | 1266.79 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-07 11:10:00 | 1251.79 | 2024-05-07 11:35:00 | 1246.69 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-07 11:10:00 | 1251.79 | 2024-05-07 11:40:00 | 1251.79 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-10 11:05:00 | 1182.66 | 2024-05-10 11:10:00 | 1180.07 | STOP_HIT | 1.00 | -0.22% |
