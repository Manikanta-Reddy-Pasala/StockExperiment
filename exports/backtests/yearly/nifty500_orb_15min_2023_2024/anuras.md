# Anupam Rasayan India Ltd. (ANURAS)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (53841 bars)
- **Last close:** 1369.00
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
| ENTRY1 | 82 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 17 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 115 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 65
- **Target hits / Stop hits / Partials:** 17 / 65 / 33
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 27.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 9 | 30.0% | 3 | 21 | 6 | 0.08% | 2.4% |
| BUY @ 2nd Alert (retest1) | 30 | 9 | 30.0% | 3 | 21 | 6 | 0.08% | 2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 85 | 41 | 48.2% | 14 | 44 | 27 | 0.29% | 24.9% |
| SELL @ 2nd Alert (retest1) | 85 | 41 | 48.2% | 14 | 44 | 27 | 0.29% | 24.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 115 | 50 | 43.5% | 17 | 65 | 33 | 0.24% | 27.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 11:00:00 | 1183.00 | 1168.79 | 0.00 | ORB-long ORB[1156.00,1172.50] vol=1.9x ATR=4.17 |
| Stop hit — per-position SL triggered | 2023-05-15 11:15:00 | 1178.83 | 1173.63 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 10:45:00 | 1177.75 | 1170.78 | 0.00 | ORB-long ORB[1162.85,1174.30] vol=3.0x ATR=2.98 |
| Stop hit — per-position SL triggered | 2023-05-16 11:00:00 | 1174.77 | 1171.22 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:50:00 | 1165.15 | 1169.73 | 0.00 | ORB-short ORB[1168.65,1174.95] vol=2.1x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 10:05:00 | 1161.51 | 1167.89 | 0.00 | T1 1.5R @ 1161.51 |
| Stop hit — per-position SL triggered | 2023-05-19 10:10:00 | 1165.15 | 1167.64 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 09:55:00 | 1214.20 | 1212.44 | 0.00 | ORB-long ORB[1198.00,1210.55] vol=12.9x ATR=2.99 |
| Stop hit — per-position SL triggered | 2023-05-26 10:10:00 | 1211.21 | 1212.38 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-05-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 10:00:00 | 1197.00 | 1204.17 | 0.00 | ORB-short ORB[1208.25,1223.00] vol=2.2x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 10:10:00 | 1189.93 | 1203.11 | 0.00 | T1 1.5R @ 1189.93 |
| Target hit | 2023-05-29 15:20:00 | 1140.05 | 1175.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2023-06-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 11:00:00 | 1118.20 | 1121.58 | 0.00 | ORB-short ORB[1120.80,1132.00] vol=6.7x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 11:30:00 | 1114.73 | 1118.60 | 0.00 | T1 1.5R @ 1114.73 |
| Target hit | 2023-06-06 15:20:00 | 1091.85 | 1110.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2023-06-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-07 09:30:00 | 1091.65 | 1097.02 | 0.00 | ORB-short ORB[1095.35,1109.75] vol=1.9x ATR=5.41 |
| Stop hit — per-position SL triggered | 2023-06-07 09:35:00 | 1097.06 | 1097.35 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 10:45:00 | 1077.00 | 1080.41 | 0.00 | ORB-short ORB[1077.35,1088.00] vol=1.8x ATR=2.33 |
| Stop hit — per-position SL triggered | 2023-06-09 11:00:00 | 1079.33 | 1080.35 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 10:35:00 | 1075.00 | 1078.27 | 0.00 | ORB-short ORB[1077.45,1083.45] vol=1.7x ATR=1.69 |
| Stop hit — per-position SL triggered | 2023-06-13 10:40:00 | 1076.69 | 1078.22 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 10:25:00 | 1080.00 | 1081.33 | 0.00 | ORB-short ORB[1080.25,1086.65] vol=2.1x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 10:30:00 | 1077.33 | 1080.81 | 0.00 | T1 1.5R @ 1077.33 |
| Target hit | 2023-06-20 10:55:00 | 1078.65 | 1078.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — SELL (started 2023-07-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-11 10:35:00 | 984.35 | 987.66 | 0.00 | ORB-short ORB[987.00,993.80] vol=2.0x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 11:25:00 | 981.30 | 986.53 | 0.00 | T1 1.5R @ 981.30 |
| Stop hit — per-position SL triggered | 2023-07-11 15:15:00 | 984.35 | 984.62 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 10:15:00 | 975.65 | 978.08 | 0.00 | ORB-short ORB[977.60,985.40] vol=1.6x ATR=1.62 |
| Stop hit — per-position SL triggered | 2023-07-25 11:15:00 | 977.27 | 977.08 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-26 11:15:00 | 980.00 | 981.71 | 0.00 | ORB-short ORB[980.05,985.95] vol=3.6x ATR=1.47 |
| Stop hit — per-position SL triggered | 2023-07-26 12:40:00 | 981.47 | 981.29 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 09:50:00 | 972.50 | 976.38 | 0.00 | ORB-short ORB[975.90,981.00] vol=1.8x ATR=2.43 |
| Stop hit — per-position SL triggered | 2023-07-27 09:55:00 | 974.93 | 976.22 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 09:35:00 | 971.85 | 975.14 | 0.00 | ORB-short ORB[973.20,980.00] vol=2.8x ATR=3.86 |
| Stop hit — per-position SL triggered | 2023-08-01 09:40:00 | 975.71 | 974.97 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-08-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:10:00 | 944.90 | 946.87 | 0.00 | ORB-short ORB[945.15,955.00] vol=2.3x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 11:05:00 | 940.92 | 946.33 | 0.00 | T1 1.5R @ 940.92 |
| Target hit | 2023-08-02 15:20:00 | 926.35 | 935.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2023-08-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 10:40:00 | 926.75 | 927.56 | 0.00 | ORB-short ORB[927.65,933.40] vol=2.1x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-03 12:55:00 | 923.09 | 926.63 | 0.00 | T1 1.5R @ 923.09 |
| Target hit | 2023-08-03 15:00:00 | 926.25 | 925.01 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2023-08-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 10:45:00 | 927.15 | 928.57 | 0.00 | ORB-short ORB[927.25,934.90] vol=1.8x ATR=1.65 |
| Stop hit — per-position SL triggered | 2023-08-04 10:50:00 | 928.80 | 928.57 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:55:00 | 938.60 | 939.10 | 0.00 | ORB-short ORB[938.65,942.80] vol=3.3x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-08-10 11:05:00 | 939.72 | 939.11 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 11:00:00 | 996.00 | 987.77 | 0.00 | ORB-long ORB[978.00,990.00] vol=2.0x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 11:25:00 | 1000.74 | 991.88 | 0.00 | T1 1.5R @ 1000.74 |
| Target hit | 2023-08-22 15:20:00 | 1008.00 | 997.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2023-08-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:55:00 | 994.75 | 1003.86 | 0.00 | ORB-short ORB[1000.00,1009.65] vol=2.5x ATR=3.70 |
| Stop hit — per-position SL triggered | 2023-08-25 14:15:00 | 998.45 | 997.96 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 10:05:00 | 1003.20 | 1004.77 | 0.00 | ORB-short ORB[1003.40,1013.00] vol=4.3x ATR=1.67 |
| Stop hit — per-position SL triggered | 2023-08-30 10:10:00 | 1004.87 | 1005.01 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 09:50:00 | 1012.50 | 1009.89 | 0.00 | ORB-long ORB[1003.20,1010.80] vol=3.0x ATR=2.71 |
| Stop hit — per-position SL triggered | 2023-08-31 10:00:00 | 1009.79 | 1010.13 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-09-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 10:55:00 | 997.00 | 1004.25 | 0.00 | ORB-short ORB[1005.40,1017.90] vol=3.7x ATR=2.73 |
| Stop hit — per-position SL triggered | 2023-09-04 11:00:00 | 999.73 | 1004.08 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-09-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:40:00 | 1005.40 | 1001.76 | 0.00 | ORB-long ORB[997.05,1003.40] vol=4.3x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 10:55:00 | 1008.46 | 1002.41 | 0.00 | T1 1.5R @ 1008.46 |
| Stop hit — per-position SL triggered | 2023-09-05 11:10:00 | 1005.40 | 1005.34 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-09-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 09:30:00 | 992.75 | 999.77 | 0.00 | ORB-short ORB[999.80,1009.80] vol=5.9x ATR=3.28 |
| Stop hit — per-position SL triggered | 2023-09-07 09:35:00 | 996.03 | 999.66 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-09-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 11:00:00 | 992.45 | 998.52 | 0.00 | ORB-short ORB[995.00,1008.20] vol=2.7x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 11:30:00 | 989.87 | 997.26 | 0.00 | T1 1.5R @ 989.87 |
| Target hit | 2023-09-08 15:20:00 | 983.05 | 991.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2023-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:35:00 | 970.50 | 976.30 | 0.00 | ORB-short ORB[978.60,987.45] vol=2.4x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:50:00 | 966.20 | 971.54 | 0.00 | T1 1.5R @ 966.20 |
| Stop hit — per-position SL triggered | 2023-09-12 10:00:00 | 970.50 | 970.67 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 09:30:00 | 956.95 | 960.31 | 0.00 | ORB-short ORB[957.55,965.25] vol=2.8x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 11:30:00 | 953.57 | 957.61 | 0.00 | T1 1.5R @ 953.57 |
| Target hit | 2023-09-15 15:20:00 | 943.05 | 946.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2023-09-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 09:55:00 | 933.60 | 937.85 | 0.00 | ORB-short ORB[937.25,944.00] vol=2.0x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 13:00:00 | 929.94 | 933.95 | 0.00 | T1 1.5R @ 929.94 |
| Target hit | 2023-09-20 15:20:00 | 924.60 | 931.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2023-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 09:40:00 | 888.40 | 885.60 | 0.00 | ORB-long ORB[880.15,886.45] vol=2.4x ATR=3.02 |
| Stop hit — per-position SL triggered | 2023-09-26 10:40:00 | 885.38 | 886.24 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 10:20:00 | 886.35 | 883.28 | 0.00 | ORB-long ORB[880.75,885.95] vol=2.0x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 10:30:00 | 889.68 | 885.40 | 0.00 | T1 1.5R @ 889.68 |
| Target hit | 2023-09-27 15:05:00 | 894.05 | 894.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2023-09-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 09:55:00 | 889.75 | 893.52 | 0.00 | ORB-short ORB[893.55,900.80] vol=2.3x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 10:15:00 | 886.07 | 891.27 | 0.00 | T1 1.5R @ 886.07 |
| Stop hit — per-position SL triggered | 2023-09-28 10:20:00 | 889.75 | 891.02 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-10-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 10:25:00 | 895.15 | 889.04 | 0.00 | ORB-long ORB[881.50,893.00] vol=2.3x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 10:40:00 | 899.42 | 890.93 | 0.00 | T1 1.5R @ 899.42 |
| Stop hit — per-position SL triggered | 2023-10-05 10:45:00 | 895.15 | 892.63 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-10-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 10:45:00 | 892.80 | 896.84 | 0.00 | ORB-short ORB[895.20,904.70] vol=2.4x ATR=2.09 |
| Stop hit — per-position SL triggered | 2023-10-06 10:50:00 | 894.89 | 896.73 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-10-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-10 10:25:00 | 880.00 | 883.12 | 0.00 | ORB-short ORB[882.00,889.75] vol=5.5x ATR=1.99 |
| Stop hit — per-position SL triggered | 2023-10-10 10:30:00 | 881.99 | 882.05 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 09:50:00 | 882.55 | 885.54 | 0.00 | ORB-short ORB[883.25,893.45] vol=1.8x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-10-12 09:55:00 | 884.50 | 885.63 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-10-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 10:30:00 | 879.95 | 883.10 | 0.00 | ORB-short ORB[882.10,889.05] vol=2.7x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 11:10:00 | 876.93 | 882.41 | 0.00 | T1 1.5R @ 876.93 |
| Stop hit — per-position SL triggered | 2023-10-16 14:45:00 | 879.95 | 881.36 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 09:55:00 | 852.90 | 853.99 | 0.00 | ORB-short ORB[854.15,864.00] vol=6.5x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:20:00 | 848.84 | 853.54 | 0.00 | T1 1.5R @ 848.84 |
| Stop hit — per-position SL triggered | 2023-10-18 12:35:00 | 852.90 | 853.19 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-10-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 10:40:00 | 855.50 | 855.50 | 0.00 | ORB-long ORB[846.55,854.95] vol=3.8x ATR=2.47 |
| Stop hit — per-position SL triggered | 2023-10-19 11:00:00 | 853.03 | 855.35 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-10-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-25 09:35:00 | 852.35 | 848.77 | 0.00 | ORB-long ORB[845.55,852.20] vol=1.5x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:40:00 | 856.16 | 849.53 | 0.00 | T1 1.5R @ 856.16 |
| Stop hit — per-position SL triggered | 2023-10-25 09:45:00 | 852.35 | 849.62 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-10-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 10:25:00 | 842.05 | 844.37 | 0.00 | ORB-short ORB[842.55,854.45] vol=1.9x ATR=1.65 |
| Stop hit — per-position SL triggered | 2023-10-26 10:30:00 | 843.70 | 844.28 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-10-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 10:50:00 | 851.70 | 848.19 | 0.00 | ORB-long ORB[842.10,847.50] vol=1.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2023-10-27 11:25:00 | 850.28 | 848.75 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 09:45:00 | 855.00 | 854.34 | 0.00 | ORB-long ORB[849.55,854.95] vol=4.0x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 12:00:00 | 858.96 | 855.71 | 0.00 | T1 1.5R @ 858.96 |
| Target hit | 2023-10-30 15:20:00 | 881.65 | 865.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2023-11-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 10:20:00 | 882.80 | 880.61 | 0.00 | ORB-long ORB[875.65,881.05] vol=4.4x ATR=1.68 |
| Stop hit — per-position SL triggered | 2023-11-01 11:05:00 | 881.12 | 880.74 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-12-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 09:55:00 | 1037.80 | 1040.27 | 0.00 | ORB-short ORB[1039.30,1045.20] vol=2.3x ATR=2.00 |
| Stop hit — per-position SL triggered | 2023-12-11 10:00:00 | 1039.80 | 1040.49 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-12-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 10:50:00 | 1039.00 | 1046.63 | 0.00 | ORB-short ORB[1050.15,1059.40] vol=1.7x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 10:55:00 | 1035.89 | 1044.31 | 0.00 | T1 1.5R @ 1035.89 |
| Stop hit — per-position SL triggered | 2023-12-13 11:35:00 | 1039.00 | 1041.34 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-12-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 11:05:00 | 1039.90 | 1041.51 | 0.00 | ORB-short ORB[1040.00,1050.95] vol=1.6x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 13:15:00 | 1036.88 | 1039.77 | 0.00 | T1 1.5R @ 1036.88 |
| Target hit | 2023-12-14 15:20:00 | 1032.55 | 1038.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2023-12-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:10:00 | 1027.25 | 1028.53 | 0.00 | ORB-short ORB[1028.15,1040.00] vol=1.6x ATR=3.18 |
| Stop hit — per-position SL triggered | 2023-12-19 10:20:00 | 1030.43 | 1028.60 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-12-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-22 10:10:00 | 1017.55 | 1022.96 | 0.00 | ORB-short ORB[1021.10,1035.80] vol=5.9x ATR=2.58 |
| Stop hit — per-position SL triggered | 2023-12-22 10:15:00 | 1020.13 | 1021.95 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 11:15:00 | 1047.50 | 1050.39 | 0.00 | ORB-short ORB[1047.55,1053.45] vol=1.6x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-01-01 12:15:00 | 1049.46 | 1049.11 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:45:00 | 1082.30 | 1079.61 | 0.00 | ORB-long ORB[1066.45,1081.00] vol=1.7x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-01-02 09:55:00 | 1077.57 | 1079.21 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 11:15:00 | 1056.50 | 1065.54 | 0.00 | ORB-short ORB[1064.75,1076.90] vol=1.9x ATR=2.66 |
| Stop hit — per-position SL triggered | 2024-01-08 14:35:00 | 1059.16 | 1064.41 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-01-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 09:30:00 | 1004.25 | 1010.83 | 0.00 | ORB-short ORB[1006.05,1019.30] vol=1.5x ATR=5.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 11:40:00 | 995.68 | 1005.00 | 0.00 | T1 1.5R @ 995.68 |
| Target hit | 2024-01-10 15:20:00 | 956.15 | 969.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2024-01-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 10:10:00 | 967.10 | 955.97 | 0.00 | ORB-long ORB[948.20,959.90] vol=4.0x ATR=3.44 |
| Stop hit — per-position SL triggered | 2024-01-12 10:15:00 | 963.66 | 957.28 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-01-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:45:00 | 920.45 | 930.01 | 0.00 | ORB-short ORB[934.90,944.65] vol=1.6x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:55:00 | 914.12 | 925.96 | 0.00 | T1 1.5R @ 914.12 |
| Stop hit — per-position SL triggered | 2024-01-18 10:00:00 | 920.45 | 925.19 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 09:45:00 | 939.00 | 941.02 | 0.00 | ORB-short ORB[941.00,947.90] vol=2.6x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 10:45:00 | 935.01 | 939.16 | 0.00 | T1 1.5R @ 935.01 |
| Target hit | 2024-01-20 15:05:00 | 936.50 | 935.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 58 — SELL (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 11:15:00 | 943.35 | 950.56 | 0.00 | ORB-short ORB[956.75,967.50] vol=6.3x ATR=2.70 |
| Stop hit — per-position SL triggered | 2024-01-29 11:45:00 | 946.05 | 950.43 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-02-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 10:30:00 | 939.55 | 944.30 | 0.00 | ORB-short ORB[940.00,952.90] vol=2.8x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 10:55:00 | 934.67 | 942.66 | 0.00 | T1 1.5R @ 934.67 |
| Target hit | 2024-02-05 15:20:00 | 917.50 | 925.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — SELL (started 2024-02-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 11:05:00 | 918.00 | 922.35 | 0.00 | ORB-short ORB[919.95,930.00] vol=5.2x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 11:40:00 | 914.80 | 919.25 | 0.00 | T1 1.5R @ 914.80 |
| Stop hit — per-position SL triggered | 2024-02-07 12:50:00 | 918.00 | 918.33 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-02-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:10:00 | 908.00 | 913.33 | 0.00 | ORB-short ORB[910.00,917.00] vol=5.3x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-02-08 11:15:00 | 909.76 | 913.33 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-02-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 10:00:00 | 905.70 | 907.70 | 0.00 | ORB-short ORB[906.55,913.70] vol=1.8x ATR=3.82 |
| Stop hit — per-position SL triggered | 2024-02-09 10:10:00 | 909.52 | 907.79 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-02-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 10:25:00 | 886.00 | 893.56 | 0.00 | ORB-short ORB[913.85,923.00] vol=10.4x ATR=9.30 |
| Stop hit — per-position SL triggered | 2024-02-12 10:30:00 | 895.30 | 889.40 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-02-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 10:45:00 | 907.95 | 903.77 | 0.00 | ORB-long ORB[895.25,906.15] vol=1.7x ATR=2.36 |
| Stop hit — per-position SL triggered | 2024-02-19 11:00:00 | 905.59 | 904.06 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 950.95 | 952.67 | 0.00 | ORB-short ORB[952.00,957.40] vol=3.2x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 11:20:00 | 948.62 | 952.01 | 0.00 | T1 1.5R @ 948.62 |
| Stop hit — per-position SL triggered | 2024-02-28 11:30:00 | 950.95 | 951.57 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-03-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 11:10:00 | 955.05 | 957.13 | 0.00 | ORB-short ORB[956.00,963.00] vol=4.2x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-03-01 11:15:00 | 956.61 | 957.13 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-03-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 10:00:00 | 949.55 | 952.42 | 0.00 | ORB-short ORB[952.35,959.50] vol=3.3x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-03-06 10:50:00 | 951.41 | 952.00 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-03-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 11:05:00 | 949.25 | 952.24 | 0.00 | ORB-short ORB[950.00,956.60] vol=3.1x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-03-07 11:25:00 | 950.70 | 951.06 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 11:10:00 | 929.95 | 938.06 | 0.00 | ORB-short ORB[936.65,949.90] vol=5.1x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 11:20:00 | 925.70 | 937.17 | 0.00 | T1 1.5R @ 925.70 |
| Stop hit — per-position SL triggered | 2024-03-12 11:45:00 | 929.95 | 936.11 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:40:00 | 918.00 | 926.14 | 0.00 | ORB-short ORB[925.30,934.60] vol=2.3x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:50:00 | 912.21 | 924.48 | 0.00 | T1 1.5R @ 912.21 |
| Target hit | 2024-03-13 14:45:00 | 911.80 | 910.96 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — SELL (started 2024-03-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:55:00 | 908.00 | 909.57 | 0.00 | ORB-short ORB[908.90,914.95] vol=1.9x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 10:30:00 | 904.99 | 906.70 | 0.00 | T1 1.5R @ 904.99 |
| Target hit | 2024-03-19 12:35:00 | 903.45 | 902.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — BUY (started 2024-03-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 10:55:00 | 899.00 | 896.81 | 0.00 | ORB-long ORB[894.60,898.95] vol=4.1x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-03-20 11:00:00 | 897.44 | 896.82 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-03-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-22 10:45:00 | 888.05 | 889.82 | 0.00 | ORB-short ORB[889.00,895.10] vol=5.9x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-03-22 10:55:00 | 889.48 | 889.86 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 11:00:00 | 869.55 | 872.80 | 0.00 | ORB-short ORB[870.10,881.55] vol=12.6x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 11:35:00 | 866.34 | 869.61 | 0.00 | T1 1.5R @ 866.34 |
| Stop hit — per-position SL triggered | 2024-04-10 11:45:00 | 869.55 | 869.19 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-04-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 09:35:00 | 880.00 | 877.93 | 0.00 | ORB-long ORB[869.20,879.75] vol=5.0x ATR=3.73 |
| Stop hit — per-position SL triggered | 2024-04-12 09:40:00 | 876.27 | 877.91 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:45:00 | 863.80 | 860.37 | 0.00 | ORB-long ORB[858.90,863.25] vol=3.7x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-04-16 09:55:00 | 861.19 | 860.41 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 09:45:00 | 849.20 | 851.39 | 0.00 | ORB-short ORB[849.90,856.65] vol=2.1x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 11:45:00 | 843.58 | 849.62 | 0.00 | T1 1.5R @ 843.58 |
| Stop hit — per-position SL triggered | 2024-04-23 15:10:00 | 849.20 | 845.20 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-04-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 09:30:00 | 845.60 | 843.75 | 0.00 | ORB-long ORB[835.50,843.95] vol=4.9x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-04-26 12:10:00 | 842.36 | 844.26 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-05-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 10:30:00 | 844.45 | 841.43 | 0.00 | ORB-long ORB[831.55,842.60] vol=9.6x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-05-03 10:40:00 | 842.26 | 841.47 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:40:00 | 831.00 | 835.97 | 0.00 | ORB-short ORB[832.35,840.05] vol=2.4x ATR=3.14 |
| Stop hit — per-position SL triggered | 2024-05-06 09:45:00 | 834.14 | 835.82 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 10:50:00 | 813.30 | 809.60 | 0.00 | ORB-long ORB[803.20,809.00] vol=5.0x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-05-08 11:00:00 | 811.29 | 809.96 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-05-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 10:20:00 | 794.00 | 788.41 | 0.00 | ORB-long ORB[782.25,792.50] vol=3.0x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-05-10 10:25:00 | 791.26 | 788.82 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 11:00:00 | 1183.00 | 2023-05-15 11:15:00 | 1178.83 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-05-16 10:45:00 | 1177.75 | 2023-05-16 11:00:00 | 1174.77 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-05-19 09:50:00 | 1165.15 | 2023-05-19 10:05:00 | 1161.51 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-05-19 09:50:00 | 1165.15 | 2023-05-19 10:10:00 | 1165.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-26 09:55:00 | 1214.20 | 2023-05-26 10:10:00 | 1211.21 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-05-29 10:00:00 | 1197.00 | 2023-05-29 10:10:00 | 1189.93 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2023-05-29 10:00:00 | 1197.00 | 2023-05-29 15:20:00 | 1140.05 | TARGET_HIT | 0.50 | 4.76% |
| SELL | retest1 | 2023-06-06 11:00:00 | 1118.20 | 2023-06-06 11:30:00 | 1114.73 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-06-06 11:00:00 | 1118.20 | 2023-06-06 15:20:00 | 1091.85 | TARGET_HIT | 0.50 | 2.36% |
| SELL | retest1 | 2023-06-07 09:30:00 | 1091.65 | 2023-06-07 09:35:00 | 1097.06 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2023-06-09 10:45:00 | 1077.00 | 2023-06-09 11:00:00 | 1079.33 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-06-13 10:35:00 | 1075.00 | 2023-06-13 10:40:00 | 1076.69 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-06-20 10:25:00 | 1080.00 | 2023-06-20 10:30:00 | 1077.33 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-06-20 10:25:00 | 1080.00 | 2023-06-20 10:55:00 | 1078.65 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2023-07-11 10:35:00 | 984.35 | 2023-07-11 11:25:00 | 981.30 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-07-11 10:35:00 | 984.35 | 2023-07-11 15:15:00 | 984.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-25 10:15:00 | 975.65 | 2023-07-25 11:15:00 | 977.27 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-07-26 11:15:00 | 980.00 | 2023-07-26 12:40:00 | 981.47 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-07-27 09:50:00 | 972.50 | 2023-07-27 09:55:00 | 974.93 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-08-01 09:35:00 | 971.85 | 2023-08-01 09:40:00 | 975.71 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-08-02 10:10:00 | 944.90 | 2023-08-02 11:05:00 | 940.92 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-08-02 10:10:00 | 944.90 | 2023-08-02 15:20:00 | 926.35 | TARGET_HIT | 0.50 | 1.96% |
| SELL | retest1 | 2023-08-03 10:40:00 | 926.75 | 2023-08-03 12:55:00 | 923.09 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-08-03 10:40:00 | 926.75 | 2023-08-03 15:00:00 | 926.25 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2023-08-04 10:45:00 | 927.15 | 2023-08-04 10:50:00 | 928.80 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-08-10 10:55:00 | 938.60 | 2023-08-10 11:05:00 | 939.72 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2023-08-22 11:00:00 | 996.00 | 2023-08-22 11:25:00 | 1000.74 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-08-22 11:00:00 | 996.00 | 2023-08-22 15:20:00 | 1008.00 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2023-08-25 10:55:00 | 994.75 | 2023-08-25 14:15:00 | 998.45 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-08-30 10:05:00 | 1003.20 | 2023-08-30 10:10:00 | 1004.87 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-08-31 09:50:00 | 1012.50 | 2023-08-31 10:00:00 | 1009.79 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-09-04 10:55:00 | 997.00 | 2023-09-04 11:00:00 | 999.73 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-09-05 10:40:00 | 1005.40 | 2023-09-05 10:55:00 | 1008.46 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-09-05 10:40:00 | 1005.40 | 2023-09-05 11:10:00 | 1005.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-07 09:30:00 | 992.75 | 2023-09-07 09:35:00 | 996.03 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-09-08 11:00:00 | 992.45 | 2023-09-08 11:30:00 | 989.87 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-09-08 11:00:00 | 992.45 | 2023-09-08 15:20:00 | 983.05 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2023-09-12 09:35:00 | 970.50 | 2023-09-12 09:50:00 | 966.20 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-09-12 09:35:00 | 970.50 | 2023-09-12 10:00:00 | 970.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-15 09:30:00 | 956.95 | 2023-09-15 11:30:00 | 953.57 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-09-15 09:30:00 | 956.95 | 2023-09-15 15:20:00 | 943.05 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2023-09-20 09:55:00 | 933.60 | 2023-09-20 13:00:00 | 929.94 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-09-20 09:55:00 | 933.60 | 2023-09-20 15:20:00 | 924.60 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2023-09-26 09:40:00 | 888.40 | 2023-09-26 10:40:00 | 885.38 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-09-27 10:20:00 | 886.35 | 2023-09-27 10:30:00 | 889.68 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-09-27 10:20:00 | 886.35 | 2023-09-27 15:05:00 | 894.05 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2023-09-28 09:55:00 | 889.75 | 2023-09-28 10:15:00 | 886.07 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-09-28 09:55:00 | 889.75 | 2023-09-28 10:20:00 | 889.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-05 10:25:00 | 895.15 | 2023-10-05 10:40:00 | 899.42 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-10-05 10:25:00 | 895.15 | 2023-10-05 10:45:00 | 895.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-06 10:45:00 | 892.80 | 2023-10-06 10:50:00 | 894.89 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-10-10 10:25:00 | 880.00 | 2023-10-10 10:30:00 | 881.99 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-10-12 09:50:00 | 882.55 | 2023-10-12 09:55:00 | 884.50 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-10-16 10:30:00 | 879.95 | 2023-10-16 11:10:00 | 876.93 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-10-16 10:30:00 | 879.95 | 2023-10-16 14:45:00 | 879.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-18 09:55:00 | 852.90 | 2023-10-18 11:20:00 | 848.84 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-10-18 09:55:00 | 852.90 | 2023-10-18 12:35:00 | 852.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-19 10:40:00 | 855.50 | 2023-10-19 11:00:00 | 853.03 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-10-25 09:35:00 | 852.35 | 2023-10-25 09:40:00 | 856.16 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-10-25 09:35:00 | 852.35 | 2023-10-25 09:45:00 | 852.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-26 10:25:00 | 842.05 | 2023-10-26 10:30:00 | 843.70 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-10-27 10:50:00 | 851.70 | 2023-10-27 11:25:00 | 850.28 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-10-30 09:45:00 | 855.00 | 2023-10-30 12:00:00 | 858.96 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-10-30 09:45:00 | 855.00 | 2023-10-30 15:20:00 | 881.65 | TARGET_HIT | 0.50 | 3.12% |
| BUY | retest1 | 2023-11-01 10:20:00 | 882.80 | 2023-11-01 11:05:00 | 881.12 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-12-11 09:55:00 | 1037.80 | 2023-12-11 10:00:00 | 1039.80 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-12-13 10:50:00 | 1039.00 | 2023-12-13 10:55:00 | 1035.89 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-12-13 10:50:00 | 1039.00 | 2023-12-13 11:35:00 | 1039.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-14 11:05:00 | 1039.90 | 2023-12-14 13:15:00 | 1036.88 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-12-14 11:05:00 | 1039.90 | 2023-12-14 15:20:00 | 1032.55 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2023-12-19 10:10:00 | 1027.25 | 2023-12-19 10:20:00 | 1030.43 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-12-22 10:10:00 | 1017.55 | 2023-12-22 10:15:00 | 1020.13 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-01-01 11:15:00 | 1047.50 | 2024-01-01 12:15:00 | 1049.46 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-01-02 09:45:00 | 1082.30 | 2024-01-02 09:55:00 | 1077.57 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-01-08 11:15:00 | 1056.50 | 2024-01-08 14:35:00 | 1059.16 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-01-10 09:30:00 | 1004.25 | 2024-01-10 11:40:00 | 995.68 | PARTIAL | 0.50 | 0.85% |
| SELL | retest1 | 2024-01-10 09:30:00 | 1004.25 | 2024-01-10 15:20:00 | 956.15 | TARGET_HIT | 0.50 | 4.79% |
| BUY | retest1 | 2024-01-12 10:10:00 | 967.10 | 2024-01-12 10:15:00 | 963.66 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-01-18 09:45:00 | 920.45 | 2024-01-18 09:55:00 | 914.12 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-01-18 09:45:00 | 920.45 | 2024-01-18 10:00:00 | 920.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-20 09:45:00 | 939.00 | 2024-01-20 10:45:00 | 935.01 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-01-20 09:45:00 | 939.00 | 2024-01-20 15:05:00 | 936.50 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2024-01-29 11:15:00 | 943.35 | 2024-01-29 11:45:00 | 946.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-05 10:30:00 | 939.55 | 2024-02-05 10:55:00 | 934.67 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-02-05 10:30:00 | 939.55 | 2024-02-05 15:20:00 | 917.50 | TARGET_HIT | 0.50 | 2.35% |
| SELL | retest1 | 2024-02-07 11:05:00 | 918.00 | 2024-02-07 11:40:00 | 914.80 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-02-07 11:05:00 | 918.00 | 2024-02-07 12:50:00 | 918.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-08 11:10:00 | 908.00 | 2024-02-08 11:15:00 | 909.76 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-02-09 10:00:00 | 905.70 | 2024-02-09 10:10:00 | 909.52 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-02-12 10:25:00 | 886.00 | 2024-02-12 10:30:00 | 895.30 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest1 | 2024-02-19 10:45:00 | 907.95 | 2024-02-19 11:00:00 | 905.59 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-02-28 10:50:00 | 950.95 | 2024-02-28 11:20:00 | 948.62 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-02-28 10:50:00 | 950.95 | 2024-02-28 11:30:00 | 950.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-01 11:10:00 | 955.05 | 2024-03-01 11:15:00 | 956.61 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-03-06 10:00:00 | 949.55 | 2024-03-06 10:50:00 | 951.41 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-03-07 11:05:00 | 949.25 | 2024-03-07 11:25:00 | 950.70 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-03-12 11:10:00 | 929.95 | 2024-03-12 11:20:00 | 925.70 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-03-12 11:10:00 | 929.95 | 2024-03-12 11:45:00 | 929.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-13 09:40:00 | 918.00 | 2024-03-13 09:50:00 | 912.21 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-03-13 09:40:00 | 918.00 | 2024-03-13 14:45:00 | 911.80 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2024-03-19 09:55:00 | 908.00 | 2024-03-19 10:30:00 | 904.99 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-03-19 09:55:00 | 908.00 | 2024-03-19 12:35:00 | 903.45 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-03-20 10:55:00 | 899.00 | 2024-03-20 11:00:00 | 897.44 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-03-22 10:45:00 | 888.05 | 2024-03-22 10:55:00 | 889.48 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-04-10 11:00:00 | 869.55 | 2024-04-10 11:35:00 | 866.34 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-04-10 11:00:00 | 869.55 | 2024-04-10 11:45:00 | 869.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-12 09:35:00 | 880.00 | 2024-04-12 09:40:00 | 876.27 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-04-16 09:45:00 | 863.80 | 2024-04-16 09:55:00 | 861.19 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-04-23 09:45:00 | 849.20 | 2024-04-23 11:45:00 | 843.58 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-04-23 09:45:00 | 849.20 | 2024-04-23 15:10:00 | 849.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-26 09:30:00 | 845.60 | 2024-04-26 12:10:00 | 842.36 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-03 10:30:00 | 844.45 | 2024-05-03 10:40:00 | 842.26 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-05-06 09:40:00 | 831.00 | 2024-05-06 09:45:00 | 834.14 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-08 10:50:00 | 813.30 | 2024-05-08 11:00:00 | 811.29 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-10 10:20:00 | 794.00 | 2024-05-10 10:25:00 | 791.26 | STOP_HIT | 1.00 | -0.34% |
