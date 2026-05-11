# Max Healthcare Institute Ltd. (MAXHEALTH)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:25:00 (36496 bars)
- **Last close:** 992.00
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
| ENTRY1 | 31 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 7 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 24
- **Target hits / Stop hits / Partials:** 7 / 24 / 12
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 6.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 7 | 35.0% | 2 | 13 | 5 | 0.09% | 1.9% |
| BUY @ 2nd Alert (retest1) | 20 | 7 | 35.0% | 2 | 13 | 5 | 0.09% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 12 | 52.2% | 5 | 11 | 7 | 0.22% | 5.0% |
| SELL @ 2nd Alert (retest1) | 23 | 12 | 52.2% | 5 | 11 | 7 | 0.22% | 5.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 43 | 19 | 44.2% | 7 | 24 | 12 | 0.16% | 6.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:25:00 | 803.25 | 808.53 | 0.00 | ORB-short ORB[807.95,818.95] vol=7.4x ATR=4.17 |
| Stop hit — per-position SL triggered | 2024-05-14 11:05:00 | 807.42 | 807.91 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 11:15:00 | 834.30 | 837.57 | 0.00 | ORB-short ORB[837.10,846.45] vol=1.8x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-05-17 11:30:00 | 836.40 | 837.31 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:55:00 | 861.35 | 848.89 | 0.00 | ORB-long ORB[836.25,846.40] vol=1.8x ATR=5.24 |
| Stop hit — per-position SL triggered | 2024-05-22 10:00:00 | 856.11 | 849.80 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 763.25 | 769.21 | 0.00 | ORB-short ORB[772.45,783.40] vol=2.0x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:10:00 | 758.09 | 765.33 | 0.00 | T1 1.5R @ 758.09 |
| Target hit | 2024-05-31 15:20:00 | 755.90 | 753.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-06-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 10:25:00 | 799.30 | 806.13 | 0.00 | ORB-short ORB[804.20,815.35] vol=1.7x ATR=3.04 |
| Stop hit — per-position SL triggered | 2024-06-11 10:50:00 | 802.34 | 804.92 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 10:50:00 | 910.80 | 898.95 | 0.00 | ORB-long ORB[890.05,903.05] vol=1.9x ATR=4.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 10:55:00 | 917.01 | 900.48 | 0.00 | T1 1.5R @ 917.01 |
| Stop hit — per-position SL triggered | 2024-06-19 11:10:00 | 910.80 | 904.42 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:00:00 | 892.00 | 896.77 | 0.00 | ORB-short ORB[892.10,904.85] vol=1.9x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 12:15:00 | 888.57 | 894.80 | 0.00 | T1 1.5R @ 888.57 |
| Target hit | 2024-06-25 15:00:00 | 891.40 | 891.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2024-07-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:20:00 | 931.10 | 924.42 | 0.00 | ORB-long ORB[914.90,928.00] vol=2.1x ATR=3.53 |
| Stop hit — per-position SL triggered | 2024-07-08 12:05:00 | 927.57 | 928.71 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:30:00 | 913.40 | 919.00 | 0.00 | ORB-short ORB[915.00,927.50] vol=4.0x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:50:00 | 906.25 | 917.46 | 0.00 | T1 1.5R @ 906.25 |
| Target hit | 2024-07-10 15:20:00 | 907.75 | 912.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 925.15 | 921.18 | 0.00 | ORB-long ORB[915.25,925.00] vol=1.6x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-07-31 09:35:00 | 922.76 | 921.79 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-08-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:55:00 | 857.00 | 857.94 | 0.00 | ORB-short ORB[860.75,871.45] vol=9.3x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-08-14 10:15:00 | 859.62 | 858.03 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-08-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:20:00 | 876.95 | 877.41 | 0.00 | ORB-short ORB[877.10,884.35] vol=9.7x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-08-20 15:05:00 | 879.38 | 877.27 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 11:15:00 | 862.15 | 866.78 | 0.00 | ORB-short ORB[863.00,873.05] vol=1.8x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:20:00 | 859.31 | 866.11 | 0.00 | T1 1.5R @ 859.31 |
| Stop hit — per-position SL triggered | 2024-08-23 11:25:00 | 862.15 | 865.63 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:55:00 | 891.95 | 888.71 | 0.00 | ORB-long ORB[879.25,889.00] vol=2.6x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:00:00 | 895.49 | 889.24 | 0.00 | T1 1.5R @ 895.49 |
| Target hit | 2024-09-11 13:35:00 | 893.00 | 893.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2024-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:10:00 | 972.35 | 977.34 | 0.00 | ORB-short ORB[975.00,988.70] vol=2.4x ATR=3.14 |
| Stop hit — per-position SL triggered | 2024-10-01 11:45:00 | 975.49 | 976.35 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:30:00 | 925.35 | 922.49 | 0.00 | ORB-long ORB[913.00,920.95] vol=10.5x ATR=5.53 |
| Stop hit — per-position SL triggered | 2024-10-08 09:35:00 | 919.82 | 922.61 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-12-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:50:00 | 1041.80 | 1037.29 | 0.00 | ORB-long ORB[1024.30,1034.65] vol=1.6x ATR=3.59 |
| Stop hit — per-position SL triggered | 2024-12-03 11:25:00 | 1038.21 | 1038.13 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-12-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:50:00 | 1125.20 | 1119.63 | 0.00 | ORB-long ORB[1110.80,1124.80] vol=1.7x ATR=3.86 |
| Stop hit — per-position SL triggered | 2024-12-10 10:00:00 | 1121.34 | 1120.01 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-13 09:30:00 | 1178.10 | 1172.68 | 0.00 | ORB-long ORB[1161.80,1178.00] vol=4.0x ATR=4.48 |
| Stop hit — per-position SL triggered | 2024-12-13 10:25:00 | 1173.62 | 1175.55 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:15:00 | 1184.15 | 1180.89 | 0.00 | ORB-long ORB[1170.05,1183.35] vol=6.2x ATR=3.21 |
| Stop hit — per-position SL triggered | 2024-12-16 10:20:00 | 1180.94 | 1180.96 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-01-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:50:00 | 1147.80 | 1142.49 | 0.00 | ORB-long ORB[1136.10,1147.30] vol=2.5x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-01-02 11:00:00 | 1145.52 | 1142.78 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 11:15:00 | 1178.75 | 1169.52 | 0.00 | ORB-long ORB[1154.90,1164.45] vol=3.1x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 11:45:00 | 1184.17 | 1171.15 | 0.00 | T1 1.5R @ 1184.17 |
| Stop hit — per-position SL triggered | 2025-01-03 12:05:00 | 1178.75 | 1172.47 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-02-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:00:00 | 1094.65 | 1083.81 | 0.00 | ORB-long ORB[1067.25,1076.45] vol=2.1x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 10:15:00 | 1101.44 | 1090.44 | 0.00 | T1 1.5R @ 1101.44 |
| Stop hit — per-position SL triggered | 2025-02-01 10:20:00 | 1094.65 | 1091.30 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:30:00 | 1027.40 | 1021.07 | 0.00 | ORB-long ORB[1011.80,1023.70] vol=1.7x ATR=3.66 |
| Stop hit — per-position SL triggered | 2025-02-20 09:55:00 | 1023.74 | 1023.02 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 11:15:00 | 989.35 | 994.45 | 0.00 | ORB-short ORB[994.00,1004.95] vol=1.6x ATR=3.30 |
| Stop hit — per-position SL triggered | 2025-02-24 11:50:00 | 992.65 | 993.66 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:55:00 | 1023.45 | 1029.62 | 0.00 | ORB-short ORB[1023.50,1037.85] vol=4.5x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 10:05:00 | 1017.49 | 1029.11 | 0.00 | T1 1.5R @ 1017.49 |
| Target hit | 2025-02-27 15:15:00 | 1022.65 | 1018.93 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — SELL (started 2025-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:40:00 | 1009.15 | 1011.51 | 0.00 | ORB-short ORB[1010.30,1023.45] vol=3.5x ATR=4.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 09:50:00 | 1001.92 | 1009.70 | 0.00 | T1 1.5R @ 1001.92 |
| Stop hit — per-position SL triggered | 2025-03-06 09:55:00 | 1009.15 | 1009.58 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-04-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-02 10:40:00 | 1082.15 | 1091.03 | 0.00 | ORB-short ORB[1085.65,1099.15] vol=1.9x ATR=4.27 |
| Stop hit — per-position SL triggered | 2025-04-02 10:50:00 | 1086.42 | 1090.40 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-04-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:35:00 | 1068.60 | 1073.75 | 0.00 | ORB-short ORB[1071.00,1083.70] vol=1.9x ATR=3.33 |
| Stop hit — per-position SL triggered | 2025-04-17 11:30:00 | 1071.93 | 1072.17 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:35:00 | 1098.90 | 1082.47 | 0.00 | ORB-long ORB[1068.80,1081.90] vol=1.7x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 09:40:00 | 1104.90 | 1089.64 | 0.00 | T1 1.5R @ 1104.90 |
| Target hit | 2025-04-22 15:20:00 | 1129.30 | 1115.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2025-04-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:10:00 | 1089.80 | 1092.54 | 0.00 | ORB-short ORB[1113.40,1125.00] vol=12.3x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:25:00 | 1081.76 | 1091.88 | 0.00 | T1 1.5R @ 1081.76 |
| Target hit | 2025-04-25 15:20:00 | 1065.80 | 1082.66 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:25:00 | 803.25 | 2024-05-14 11:05:00 | 807.42 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-05-17 11:15:00 | 834.30 | 2024-05-17 11:30:00 | 836.40 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-22 09:55:00 | 861.35 | 2024-05-22 10:00:00 | 856.11 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2024-05-31 09:45:00 | 763.25 | 2024-05-31 10:10:00 | 758.09 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-05-31 09:45:00 | 763.25 | 2024-05-31 15:20:00 | 755.90 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2024-06-11 10:25:00 | 799.30 | 2024-06-11 10:50:00 | 802.34 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-19 10:50:00 | 910.80 | 2024-06-19 10:55:00 | 917.01 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-06-19 10:50:00 | 910.80 | 2024-06-19 11:10:00 | 910.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 11:00:00 | 892.00 | 2024-06-25 12:15:00 | 888.57 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-06-25 11:00:00 | 892.00 | 2024-06-25 15:00:00 | 891.40 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2024-07-08 10:20:00 | 931.10 | 2024-07-08 12:05:00 | 927.57 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-07-10 10:30:00 | 913.40 | 2024-07-10 10:50:00 | 906.25 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2024-07-10 10:30:00 | 913.40 | 2024-07-10 15:20:00 | 907.75 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2024-07-31 09:30:00 | 925.15 | 2024-07-31 09:35:00 | 922.76 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-14 09:55:00 | 857.00 | 2024-08-14 10:15:00 | 859.62 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-20 10:20:00 | 876.95 | 2024-08-20 15:05:00 | 879.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-23 11:15:00 | 862.15 | 2024-08-23 11:20:00 | 859.31 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-08-23 11:15:00 | 862.15 | 2024-08-23 11:25:00 | 862.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 10:55:00 | 891.95 | 2024-09-11 11:00:00 | 895.49 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-09-11 10:55:00 | 891.95 | 2024-09-11 13:35:00 | 893.00 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2024-10-01 11:10:00 | 972.35 | 2024-10-01 11:45:00 | 975.49 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-08 09:30:00 | 925.35 | 2024-10-08 09:35:00 | 919.82 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2024-12-03 10:50:00 | 1041.80 | 2024-12-03 11:25:00 | 1038.21 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-10 09:50:00 | 1125.20 | 2024-12-10 10:00:00 | 1121.34 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-13 09:30:00 | 1178.10 | 2024-12-13 10:25:00 | 1173.62 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-16 10:15:00 | 1184.15 | 2024-12-16 10:20:00 | 1180.94 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-02 10:50:00 | 1147.80 | 2025-01-02 11:00:00 | 1145.52 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-01-03 11:15:00 | 1178.75 | 2025-01-03 11:45:00 | 1184.17 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-03 11:15:00 | 1178.75 | 2025-01-03 12:05:00 | 1178.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-01 10:00:00 | 1094.65 | 2025-02-01 10:15:00 | 1101.44 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-02-01 10:00:00 | 1094.65 | 2025-02-01 10:20:00 | 1094.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 09:30:00 | 1027.40 | 2025-02-20 09:55:00 | 1023.74 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-02-24 11:15:00 | 989.35 | 2025-02-24 11:50:00 | 992.65 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-27 09:55:00 | 1023.45 | 2025-02-27 10:05:00 | 1017.49 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-02-27 09:55:00 | 1023.45 | 2025-02-27 15:15:00 | 1022.65 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2025-03-06 09:40:00 | 1009.15 | 2025-03-06 09:50:00 | 1001.92 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2025-03-06 09:40:00 | 1009.15 | 2025-03-06 09:55:00 | 1009.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-02 10:40:00 | 1082.15 | 2025-04-02 10:50:00 | 1086.42 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-04-17 10:35:00 | 1068.60 | 2025-04-17 11:30:00 | 1071.93 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-22 09:35:00 | 1098.90 | 2025-04-22 09:40:00 | 1104.90 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-04-22 09:35:00 | 1098.90 | 2025-04-22 15:20:00 | 1129.30 | TARGET_HIT | 0.50 | 2.77% |
| SELL | retest1 | 2025-04-25 10:10:00 | 1089.80 | 2025-04-25 10:25:00 | 1081.76 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2025-04-25 10:10:00 | 1089.80 | 2025-04-25 15:20:00 | 1065.80 | TARGET_HIT | 0.50 | 2.20% |
