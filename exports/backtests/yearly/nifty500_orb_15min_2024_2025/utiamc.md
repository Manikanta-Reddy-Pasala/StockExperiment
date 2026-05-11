# UTI Asset Management Company Ltd. (UTIAMC)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35433 bars)
- **Last close:** 973.15
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
| ENTRY1 | 50 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 13 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 37
- **Target hits / Stop hits / Partials:** 13 / 37 / 22
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 15.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 16 | 51.6% | 6 | 15 | 10 | 0.30% | 9.3% |
| BUY @ 2nd Alert (retest1) | 31 | 16 | 51.6% | 6 | 15 | 10 | 0.30% | 9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 41 | 19 | 46.3% | 7 | 22 | 12 | 0.16% | 6.5% |
| SELL @ 2nd Alert (retest1) | 41 | 19 | 46.3% | 7 | 22 | 12 | 0.16% | 6.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 72 | 35 | 48.6% | 13 | 37 | 22 | 0.22% | 15.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 11:00:00 | 896.95 | 899.52 | 0.00 | ORB-short ORB[902.05,913.50] vol=4.3x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-05-13 11:05:00 | 900.65 | 899.54 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 11:15:00 | 915.30 | 907.62 | 0.00 | ORB-long ORB[897.85,909.50] vol=3.9x ATR=2.49 |
| Stop hit — per-position SL triggered | 2024-05-14 11:20:00 | 912.81 | 907.76 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 11:00:00 | 914.50 | 918.89 | 0.00 | ORB-short ORB[921.05,926.00] vol=1.6x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 11:25:00 | 911.54 | 918.11 | 0.00 | T1 1.5R @ 911.54 |
| Target hit | 2024-05-15 15:20:00 | 905.00 | 908.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-05-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:00:00 | 916.40 | 912.33 | 0.00 | ORB-long ORB[906.35,912.90] vol=1.8x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-05-16 10:05:00 | 914.03 | 912.41 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 10:35:00 | 929.50 | 924.64 | 0.00 | ORB-long ORB[920.75,927.95] vol=3.7x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 11:25:00 | 934.12 | 926.96 | 0.00 | T1 1.5R @ 934.12 |
| Stop hit — per-position SL triggered | 2024-05-22 11:35:00 | 929.50 | 927.25 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:35:00 | 927.35 | 930.75 | 0.00 | ORB-short ORB[928.35,937.80] vol=2.2x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 09:55:00 | 922.92 | 929.25 | 0.00 | T1 1.5R @ 922.92 |
| Target hit | 2024-05-27 10:45:00 | 924.95 | 924.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2024-05-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:50:00 | 914.65 | 916.61 | 0.00 | ORB-short ORB[914.80,921.95] vol=3.8x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-05-28 10:55:00 | 916.66 | 917.25 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-05-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:00:00 | 916.90 | 919.41 | 0.00 | ORB-short ORB[919.00,924.80] vol=1.8x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:15:00 | 912.96 | 918.48 | 0.00 | T1 1.5R @ 912.96 |
| Stop hit — per-position SL triggered | 2024-05-30 10:20:00 | 916.90 | 918.44 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:15:00 | 902.70 | 903.02 | 0.00 | ORB-short ORB[906.40,911.45] vol=13.4x ATR=2.44 |
| Stop hit — per-position SL triggered | 2024-05-31 10:50:00 | 905.14 | 903.03 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 10:00:00 | 955.00 | 965.66 | 0.00 | ORB-short ORB[965.00,973.50] vol=1.5x ATR=4.42 |
| Stop hit — per-position SL triggered | 2024-06-10 10:25:00 | 959.42 | 962.32 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 09:40:00 | 1026.50 | 1032.06 | 0.00 | ORB-short ORB[1029.30,1036.90] vol=1.7x ATR=4.32 |
| Stop hit — per-position SL triggered | 2024-06-21 10:10:00 | 1030.82 | 1029.68 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:40:00 | 1015.90 | 1021.60 | 0.00 | ORB-short ORB[1020.20,1027.95] vol=1.7x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:35:00 | 1009.72 | 1016.24 | 0.00 | T1 1.5R @ 1009.72 |
| Stop hit — per-position SL triggered | 2024-06-26 14:55:00 | 1015.90 | 1012.58 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 11:05:00 | 1033.55 | 1035.53 | 0.00 | ORB-short ORB[1036.85,1045.60] vol=1.9x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:15:00 | 1029.23 | 1034.52 | 0.00 | T1 1.5R @ 1029.23 |
| Target hit | 2024-07-02 15:20:00 | 1024.00 | 1030.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:15:00 | 1042.00 | 1048.55 | 0.00 | ORB-short ORB[1044.35,1053.50] vol=1.5x ATR=2.70 |
| Stop hit — per-position SL triggered | 2024-07-09 10:20:00 | 1044.70 | 1048.42 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 1038.20 | 1047.95 | 0.00 | ORB-short ORB[1045.00,1050.00] vol=1.8x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 1034.11 | 1044.89 | 0.00 | T1 1.5R @ 1034.11 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 1038.20 | 1041.08 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:40:00 | 1055.00 | 1048.05 | 0.00 | ORB-long ORB[1040.00,1052.00] vol=2.9x ATR=5.68 |
| Stop hit — per-position SL triggered | 2024-07-11 09:45:00 | 1049.32 | 1048.71 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:35:00 | 1089.10 | 1081.41 | 0.00 | ORB-long ORB[1073.05,1086.80] vol=2.2x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 11:40:00 | 1095.47 | 1086.59 | 0.00 | T1 1.5R @ 1095.47 |
| Target hit | 2024-07-16 15:20:00 | 1108.00 | 1097.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:30:00 | 982.85 | 974.28 | 0.00 | ORB-long ORB[964.80,978.90] vol=1.6x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 09:35:00 | 989.13 | 982.59 | 0.00 | T1 1.5R @ 989.13 |
| Stop hit — per-position SL triggered | 2024-07-24 09:40:00 | 982.85 | 983.05 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:45:00 | 1061.70 | 1042.63 | 0.00 | ORB-long ORB[1025.00,1040.80] vol=1.6x ATR=5.70 |
| Stop hit — per-position SL triggered | 2024-07-26 11:10:00 | 1056.00 | 1046.99 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:40:00 | 1043.30 | 1037.78 | 0.00 | ORB-long ORB[1030.00,1040.00] vol=1.7x ATR=3.03 |
| Stop hit — per-position SL triggered | 2024-07-31 09:45:00 | 1040.27 | 1038.21 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:40:00 | 1046.00 | 1041.03 | 0.00 | ORB-long ORB[1034.70,1043.25] vol=1.7x ATR=3.75 |
| Stop hit — per-position SL triggered | 2024-08-01 09:55:00 | 1042.25 | 1044.33 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 09:45:00 | 985.85 | 992.72 | 0.00 | ORB-short ORB[988.60,997.00] vol=1.9x ATR=5.26 |
| Stop hit — per-position SL triggered | 2024-08-07 10:25:00 | 991.11 | 989.55 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:10:00 | 1007.00 | 1003.27 | 0.00 | ORB-long ORB[991.95,1004.95] vol=2.0x ATR=4.98 |
| Stop hit — per-position SL triggered | 2024-08-08 14:55:00 | 1002.02 | 1005.88 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:35:00 | 1011.60 | 1005.36 | 0.00 | ORB-long ORB[998.55,1007.95] vol=1.5x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:40:00 | 1017.89 | 1010.65 | 0.00 | T1 1.5R @ 1017.89 |
| Target hit | 2024-08-09 10:15:00 | 1012.75 | 1013.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — BUY (started 2024-08-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:45:00 | 1020.60 | 1004.80 | 0.00 | ORB-long ORB[983.90,999.05] vol=2.3x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 11:45:00 | 1027.29 | 1009.93 | 0.00 | T1 1.5R @ 1027.29 |
| Stop hit — per-position SL triggered | 2024-08-12 14:50:00 | 1020.60 | 1020.65 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:35:00 | 1039.95 | 1034.54 | 0.00 | ORB-long ORB[1024.00,1038.80] vol=2.8x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:50:00 | 1047.17 | 1039.18 | 0.00 | T1 1.5R @ 1047.17 |
| Target hit | 2024-08-13 10:15:00 | 1042.40 | 1042.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2024-08-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 11:10:00 | 1047.50 | 1052.46 | 0.00 | ORB-short ORB[1050.05,1064.00] vol=2.3x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-08-19 11:15:00 | 1050.70 | 1052.31 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:45:00 | 1090.55 | 1082.74 | 0.00 | ORB-long ORB[1066.00,1080.00] vol=11.7x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:50:00 | 1096.63 | 1086.42 | 0.00 | T1 1.5R @ 1096.63 |
| Target hit | 2024-08-20 15:20:00 | 1125.30 | 1107.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2024-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:45:00 | 1117.20 | 1127.86 | 0.00 | ORB-short ORB[1129.05,1139.95] vol=1.5x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:20:00 | 1111.73 | 1123.46 | 0.00 | T1 1.5R @ 1111.73 |
| Stop hit — per-position SL triggered | 2024-08-29 12:15:00 | 1117.20 | 1121.04 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:45:00 | 1114.10 | 1119.96 | 0.00 | ORB-short ORB[1116.90,1129.90] vol=1.6x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 10:15:00 | 1108.08 | 1117.15 | 0.00 | T1 1.5R @ 1108.08 |
| Stop hit — per-position SL triggered | 2024-08-30 11:05:00 | 1114.10 | 1114.53 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 11:10:00 | 1287.55 | 1294.54 | 0.00 | ORB-short ORB[1292.20,1302.45] vol=10.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-09-11 11:25:00 | 1290.43 | 1294.35 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:30:00 | 1294.00 | 1289.58 | 0.00 | ORB-long ORB[1284.30,1292.70] vol=2.3x ATR=4.30 |
| Stop hit — per-position SL triggered | 2024-09-12 09:50:00 | 1289.70 | 1289.89 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 1282.90 | 1272.32 | 0.00 | ORB-long ORB[1262.05,1275.00] vol=1.6x ATR=5.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 09:40:00 | 1291.53 | 1281.50 | 0.00 | T1 1.5R @ 1291.53 |
| Stop hit — per-position SL triggered | 2024-09-25 09:50:00 | 1282.90 | 1282.41 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:35:00 | 1246.30 | 1257.07 | 0.00 | ORB-short ORB[1258.00,1270.00] vol=2.2x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 10:05:00 | 1237.21 | 1247.49 | 0.00 | T1 1.5R @ 1237.21 |
| Target hit | 2024-09-26 14:10:00 | 1237.75 | 1237.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2024-10-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:55:00 | 1217.70 | 1225.15 | 0.00 | ORB-short ORB[1218.55,1234.00] vol=2.6x ATR=3.94 |
| Target hit | 2024-10-11 15:20:00 | 1214.05 | 1220.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-10-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:00:00 | 1192.95 | 1203.48 | 0.00 | ORB-short ORB[1197.75,1213.50] vol=2.2x ATR=3.37 |
| Stop hit — per-position SL triggered | 2024-10-14 13:30:00 | 1196.32 | 1198.24 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 1225.80 | 1216.01 | 0.00 | ORB-long ORB[1197.00,1215.00] vol=2.7x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:35:00 | 1232.75 | 1218.45 | 0.00 | T1 1.5R @ 1232.75 |
| Target hit | 2024-10-15 10:30:00 | 1236.10 | 1236.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — SELL (started 2024-10-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:45:00 | 1251.75 | 1263.30 | 0.00 | ORB-short ORB[1262.65,1280.20] vol=4.7x ATR=6.87 |
| Stop hit — per-position SL triggered | 2024-10-22 10:55:00 | 1258.62 | 1262.91 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 10:25:00 | 1276.35 | 1289.54 | 0.00 | ORB-short ORB[1280.60,1294.50] vol=1.9x ATR=5.13 |
| Stop hit — per-position SL triggered | 2024-11-26 15:05:00 | 1281.48 | 1281.47 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 11:10:00 | 1291.60 | 1297.44 | 0.00 | ORB-short ORB[1295.00,1302.00] vol=1.6x ATR=3.44 |
| Stop hit — per-position SL triggered | 2024-12-02 12:20:00 | 1295.04 | 1296.36 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 1319.70 | 1314.38 | 0.00 | ORB-long ORB[1303.90,1318.00] vol=2.2x ATR=3.59 |
| Stop hit — per-position SL triggered | 2024-12-05 09:35:00 | 1316.11 | 1314.64 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-12-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:55:00 | 1394.70 | 1380.94 | 0.00 | ORB-long ORB[1362.50,1380.00] vol=7.7x ATR=6.99 |
| Stop hit — per-position SL triggered | 2024-12-10 10:00:00 | 1387.71 | 1382.49 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:30:00 | 1353.95 | 1361.56 | 0.00 | ORB-short ORB[1360.00,1368.55] vol=1.9x ATR=5.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:15:00 | 1345.68 | 1355.43 | 0.00 | T1 1.5R @ 1345.68 |
| Target hit | 2024-12-13 11:25:00 | 1350.45 | 1350.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — SELL (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 1355.75 | 1364.02 | 0.00 | ORB-short ORB[1358.70,1374.00] vol=2.0x ATR=4.79 |
| Stop hit — per-position SL triggered | 2024-12-16 09:35:00 | 1360.54 | 1363.29 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:50:00 | 1271.95 | 1282.22 | 0.00 | ORB-short ORB[1280.10,1297.50] vol=2.1x ATR=4.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:00:00 | 1265.25 | 1278.41 | 0.00 | T1 1.5R @ 1265.25 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 1271.95 | 1277.47 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-01-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:45:00 | 1228.80 | 1223.14 | 0.00 | ORB-long ORB[1215.00,1224.95] vol=1.6x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 12:25:00 | 1236.01 | 1228.10 | 0.00 | T1 1.5R @ 1236.01 |
| Target hit | 2025-01-20 15:20:00 | 1247.85 | 1239.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2025-01-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:35:00 | 1204.10 | 1212.19 | 0.00 | ORB-short ORB[1216.35,1233.95] vol=2.8x ATR=5.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:05:00 | 1195.45 | 1209.82 | 0.00 | T1 1.5R @ 1195.45 |
| Target hit | 2025-01-24 15:20:00 | 1174.25 | 1183.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 09:30:00 | 953.35 | 944.31 | 0.00 | ORB-long ORB[935.00,947.30] vol=4.0x ATR=5.50 |
| Stop hit — per-position SL triggered | 2025-02-24 09:40:00 | 947.85 | 944.74 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 09:50:00 | 954.10 | 957.58 | 0.00 | ORB-short ORB[955.00,966.80] vol=1.6x ATR=3.51 |
| Stop hit — per-position SL triggered | 2025-03-19 10:40:00 | 957.61 | 955.78 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 1107.30 | 1119.01 | 0.00 | ORB-short ORB[1116.50,1129.40] vol=2.1x ATR=3.98 |
| Stop hit — per-position SL triggered | 2025-04-25 09:45:00 | 1111.28 | 1115.95 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 11:00:00 | 896.95 | 2024-05-13 11:05:00 | 900.65 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-05-14 11:15:00 | 915.30 | 2024-05-14 11:20:00 | 912.81 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-05-15 11:00:00 | 914.50 | 2024-05-15 11:25:00 | 911.54 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-15 11:00:00 | 914.50 | 2024-05-15 15:20:00 | 905.00 | TARGET_HIT | 0.50 | 1.04% |
| BUY | retest1 | 2024-05-16 10:00:00 | 916.40 | 2024-05-16 10:05:00 | 914.03 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-05-22 10:35:00 | 929.50 | 2024-05-22 11:25:00 | 934.12 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-05-22 10:35:00 | 929.50 | 2024-05-22 11:35:00 | 929.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-27 09:35:00 | 927.35 | 2024-05-27 09:55:00 | 922.92 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-05-27 09:35:00 | 927.35 | 2024-05-27 10:45:00 | 924.95 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2024-05-28 10:50:00 | 914.65 | 2024-05-28 10:55:00 | 916.66 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-30 10:00:00 | 916.90 | 2024-05-30 10:15:00 | 912.96 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-30 10:00:00 | 916.90 | 2024-05-30 10:20:00 | 916.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-31 10:15:00 | 902.70 | 2024-05-31 10:50:00 | 905.14 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-10 10:00:00 | 955.00 | 2024-06-10 10:25:00 | 959.42 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-06-21 09:40:00 | 1026.50 | 2024-06-21 10:10:00 | 1030.82 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-06-26 09:40:00 | 1015.90 | 2024-06-26 11:35:00 | 1009.72 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-06-26 09:40:00 | 1015.90 | 2024-06-26 14:55:00 | 1015.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 11:05:00 | 1033.55 | 2024-07-02 12:15:00 | 1029.23 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-07-02 11:05:00 | 1033.55 | 2024-07-02 15:20:00 | 1024.00 | TARGET_HIT | 0.50 | 0.92% |
| SELL | retest1 | 2024-07-09 10:15:00 | 1042.00 | 2024-07-09 10:20:00 | 1044.70 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-10 10:05:00 | 1038.20 | 2024-07-10 10:20:00 | 1034.11 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-10 10:05:00 | 1038.20 | 2024-07-10 10:55:00 | 1038.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 09:40:00 | 1055.00 | 2024-07-11 09:45:00 | 1049.32 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-07-16 09:35:00 | 1089.10 | 2024-07-16 11:40:00 | 1095.47 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-07-16 09:35:00 | 1089.10 | 2024-07-16 15:20:00 | 1108.00 | TARGET_HIT | 0.50 | 1.74% |
| BUY | retest1 | 2024-07-24 09:30:00 | 982.85 | 2024-07-24 09:35:00 | 989.13 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-07-24 09:30:00 | 982.85 | 2024-07-24 09:40:00 | 982.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:45:00 | 1061.70 | 2024-07-26 11:10:00 | 1056.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-07-31 09:40:00 | 1043.30 | 2024-07-31 09:45:00 | 1040.27 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-01 09:40:00 | 1046.00 | 2024-08-01 09:55:00 | 1042.25 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-07 09:45:00 | 985.85 | 2024-08-07 10:25:00 | 991.11 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-08-08 10:10:00 | 1007.00 | 2024-08-08 14:55:00 | 1002.02 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-08-09 09:35:00 | 1011.60 | 2024-08-09 09:40:00 | 1017.89 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-08-09 09:35:00 | 1011.60 | 2024-08-09 10:15:00 | 1012.75 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-08-12 10:45:00 | 1020.60 | 2024-08-12 11:45:00 | 1027.29 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-08-12 10:45:00 | 1020.60 | 2024-08-12 14:50:00 | 1020.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-13 09:35:00 | 1039.95 | 2024-08-13 09:50:00 | 1047.17 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-08-13 09:35:00 | 1039.95 | 2024-08-13 10:15:00 | 1042.40 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2024-08-19 11:10:00 | 1047.50 | 2024-08-19 11:15:00 | 1050.70 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-20 09:45:00 | 1090.55 | 2024-08-20 09:50:00 | 1096.63 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-20 09:45:00 | 1090.55 | 2024-08-20 15:20:00 | 1125.30 | TARGET_HIT | 0.50 | 3.19% |
| SELL | retest1 | 2024-08-29 10:45:00 | 1117.20 | 2024-08-29 11:20:00 | 1111.73 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-08-29 10:45:00 | 1117.20 | 2024-08-29 12:15:00 | 1117.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-30 09:45:00 | 1114.10 | 2024-08-30 10:15:00 | 1108.08 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-08-30 09:45:00 | 1114.10 | 2024-08-30 11:05:00 | 1114.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-11 11:10:00 | 1287.55 | 2024-09-11 11:25:00 | 1290.43 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-12 09:30:00 | 1294.00 | 2024-09-12 09:50:00 | 1289.70 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-25 09:30:00 | 1282.90 | 2024-09-25 09:40:00 | 1291.53 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-09-25 09:30:00 | 1282.90 | 2024-09-25 09:50:00 | 1282.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-26 09:35:00 | 1246.30 | 2024-09-26 10:05:00 | 1237.21 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-09-26 09:35:00 | 1246.30 | 2024-09-26 14:10:00 | 1237.75 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2024-10-11 10:55:00 | 1217.70 | 2024-10-11 15:20:00 | 1214.05 | TARGET_HIT | 1.00 | 0.30% |
| SELL | retest1 | 2024-10-14 11:00:00 | 1192.95 | 2024-10-14 13:30:00 | 1196.32 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-15 09:30:00 | 1225.80 | 2024-10-15 09:35:00 | 1232.75 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-10-15 09:30:00 | 1225.80 | 2024-10-15 10:30:00 | 1236.10 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2024-10-22 10:45:00 | 1251.75 | 2024-10-22 10:55:00 | 1258.62 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-11-26 10:25:00 | 1276.35 | 2024-11-26 15:05:00 | 1281.48 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-12-02 11:10:00 | 1291.60 | 2024-12-02 12:20:00 | 1295.04 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-05 09:30:00 | 1319.70 | 2024-12-05 09:35:00 | 1316.11 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-10 09:55:00 | 1394.70 | 2024-12-10 10:00:00 | 1387.71 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-12-13 09:30:00 | 1353.95 | 2024-12-13 10:15:00 | 1345.68 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-12-13 09:30:00 | 1353.95 | 2024-12-13 11:25:00 | 1350.45 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2024-12-16 09:30:00 | 1355.75 | 2024-12-16 09:35:00 | 1360.54 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-20 09:50:00 | 1271.95 | 2024-12-20 10:00:00 | 1265.25 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-12-20 09:50:00 | 1271.95 | 2024-12-20 10:15:00 | 1271.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-20 09:45:00 | 1228.80 | 2025-01-20 12:25:00 | 1236.01 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-01-20 09:45:00 | 1228.80 | 2025-01-20 15:20:00 | 1247.85 | TARGET_HIT | 0.50 | 1.55% |
| SELL | retest1 | 2025-01-24 10:35:00 | 1204.10 | 2025-01-24 11:05:00 | 1195.45 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2025-01-24 10:35:00 | 1204.10 | 2025-01-24 15:20:00 | 1174.25 | TARGET_HIT | 0.50 | 2.48% |
| BUY | retest1 | 2025-02-24 09:30:00 | 953.35 | 2025-02-24 09:40:00 | 947.85 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-03-19 09:50:00 | 954.10 | 2025-03-19 10:40:00 | 957.61 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-25 09:35:00 | 1107.30 | 2025-04-25 09:45:00 | 1111.28 | STOP_HIT | 1.00 | -0.36% |
