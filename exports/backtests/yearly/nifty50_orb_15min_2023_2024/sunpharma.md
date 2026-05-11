# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55355 bars)
- **Last close:** 1845.00
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
| ENTRY1 | 103 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 15 |
| STOP_HIT | 88 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 88
- **Target hits / Stop hits / Partials:** 15 / 88 / 29
- **Avg / median % per leg:** 0.00% / -0.17%
- **Sum % (uncompounded):** 0.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 26 | 37.7% | 10 | 43 | 16 | 0.02% | 1.1% |
| BUY @ 2nd Alert (retest1) | 69 | 26 | 37.7% | 10 | 43 | 16 | 0.02% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 63 | 18 | 28.6% | 5 | 45 | 13 | -0.01% | -0.5% |
| SELL @ 2nd Alert (retest1) | 63 | 18 | 28.6% | 5 | 45 | 13 | -0.01% | -0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 132 | 44 | 33.3% | 15 | 88 | 29 | 0.00% | 0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 10:10:00 | 930.00 | 934.62 | 0.00 | ORB-short ORB[935.00,943.00] vol=1.8x ATR=1.94 |
| Stop hit — per-position SL triggered | 2023-05-18 10:40:00 | 931.94 | 933.01 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 10:45:00 | 926.35 | 927.59 | 0.00 | ORB-short ORB[929.55,934.40] vol=5.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-05-19 11:00:00 | 928.18 | 927.57 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 10:45:00 | 941.10 | 935.60 | 0.00 | ORB-long ORB[922.85,931.45] vol=2.0x ATR=1.62 |
| Stop hit — per-position SL triggered | 2023-05-22 11:15:00 | 939.48 | 938.11 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:40:00 | 950.65 | 947.18 | 0.00 | ORB-long ORB[943.15,950.10] vol=1.5x ATR=2.26 |
| Stop hit — per-position SL triggered | 2023-05-25 10:05:00 | 948.39 | 947.77 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 10:15:00 | 960.70 | 963.33 | 0.00 | ORB-short ORB[962.00,967.45] vol=2.2x ATR=1.71 |
| Stop hit — per-position SL triggered | 2023-05-30 10:35:00 | 962.41 | 962.84 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 09:30:00 | 970.35 | 967.00 | 0.00 | ORB-long ORB[960.75,969.90] vol=1.7x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 09:40:00 | 973.88 | 969.84 | 0.00 | T1 1.5R @ 973.88 |
| Target hit | 2023-05-31 10:10:00 | 971.80 | 972.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2023-06-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 09:35:00 | 1006.25 | 1004.67 | 0.00 | ORB-long ORB[998.50,1005.85] vol=2.3x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 09:55:00 | 1009.91 | 1006.30 | 0.00 | T1 1.5R @ 1009.91 |
| Target hit | 2023-06-05 13:40:00 | 1009.80 | 1011.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2023-06-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 10:40:00 | 1004.30 | 1005.91 | 0.00 | ORB-short ORB[1008.85,1016.00] vol=3.8x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 11:15:00 | 1002.10 | 1004.69 | 0.00 | T1 1.5R @ 1002.10 |
| Target hit | 2023-06-08 15:20:00 | 988.55 | 994.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2023-06-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 11:05:00 | 981.25 | 982.85 | 0.00 | ORB-short ORB[983.50,994.25] vol=1.6x ATR=2.02 |
| Stop hit — per-position SL triggered | 2023-06-23 11:30:00 | 983.27 | 982.89 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:45:00 | 998.40 | 994.78 | 0.00 | ORB-long ORB[991.50,996.00] vol=1.5x ATR=2.21 |
| Stop hit — per-position SL triggered | 2023-06-26 10:00:00 | 996.19 | 995.91 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 11:05:00 | 1008.90 | 1001.92 | 0.00 | ORB-long ORB[995.00,1005.70] vol=2.5x ATR=2.10 |
| Stop hit — per-position SL triggered | 2023-06-28 11:15:00 | 1006.80 | 1002.49 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 10:50:00 | 1046.70 | 1036.03 | 0.00 | ORB-long ORB[1020.05,1027.15] vol=3.6x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-30 11:15:00 | 1052.10 | 1038.79 | 0.00 | T1 1.5R @ 1052.10 |
| Stop hit — per-position SL triggered | 2023-06-30 13:50:00 | 1046.70 | 1043.49 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 09:40:00 | 1043.25 | 1044.86 | 0.00 | ORB-short ORB[1045.00,1053.65] vol=3.6x ATR=2.63 |
| Stop hit — per-position SL triggered | 2023-07-03 09:50:00 | 1045.88 | 1044.88 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 10:05:00 | 1052.95 | 1047.98 | 0.00 | ORB-long ORB[1040.00,1048.20] vol=1.5x ATR=2.51 |
| Stop hit — per-position SL triggered | 2023-07-06 10:10:00 | 1050.44 | 1048.31 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 11:15:00 | 1049.60 | 1044.52 | 0.00 | ORB-long ORB[1033.50,1047.10] vol=2.1x ATR=2.31 |
| Stop hit — per-position SL triggered | 2023-07-07 11:25:00 | 1047.29 | 1044.95 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:40:00 | 1062.30 | 1056.21 | 0.00 | ORB-long ORB[1049.05,1055.95] vol=2.9x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 10:20:00 | 1066.63 | 1059.98 | 0.00 | T1 1.5R @ 1066.63 |
| Stop hit — per-position SL triggered | 2023-07-11 10:50:00 | 1062.30 | 1062.12 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 10:30:00 | 1081.10 | 1074.89 | 0.00 | ORB-long ORB[1072.25,1078.00] vol=2.0x ATR=2.20 |
| Stop hit — per-position SL triggered | 2023-07-12 11:35:00 | 1078.90 | 1077.40 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 11:15:00 | 1069.05 | 1073.20 | 0.00 | ORB-short ORB[1070.75,1078.90] vol=1.5x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 11:40:00 | 1066.43 | 1072.35 | 0.00 | T1 1.5R @ 1066.43 |
| Stop hit — per-position SL triggered | 2023-07-18 11:45:00 | 1069.05 | 1072.24 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:50:00 | 1077.00 | 1072.21 | 0.00 | ORB-long ORB[1063.85,1073.60] vol=2.2x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 10:00:00 | 1080.43 | 1074.30 | 0.00 | T1 1.5R @ 1080.43 |
| Stop hit — per-position SL triggered | 2023-07-19 10:30:00 | 1077.00 | 1076.01 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:45:00 | 1096.60 | 1088.64 | 0.00 | ORB-long ORB[1082.30,1091.50] vol=1.5x ATR=2.68 |
| Stop hit — per-position SL triggered | 2023-07-20 11:00:00 | 1093.92 | 1089.91 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 10:50:00 | 1097.75 | 1102.95 | 0.00 | ORB-short ORB[1102.00,1110.00] vol=1.5x ATR=2.01 |
| Stop hit — per-position SL triggered | 2023-07-25 10:55:00 | 1099.76 | 1102.57 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 09:50:00 | 1136.00 | 1125.31 | 0.00 | ORB-long ORB[1116.30,1128.00] vol=2.9x ATR=4.18 |
| Stop hit — per-position SL triggered | 2023-07-27 09:55:00 | 1131.82 | 1126.01 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 11:15:00 | 1131.60 | 1136.07 | 0.00 | ORB-short ORB[1135.50,1144.30] vol=1.5x ATR=2.34 |
| Stop hit — per-position SL triggered | 2023-08-02 11:25:00 | 1133.94 | 1135.65 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 09:45:00 | 1154.85 | 1159.45 | 0.00 | ORB-short ORB[1158.20,1164.60] vol=3.5x ATR=2.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 10:05:00 | 1150.49 | 1155.90 | 0.00 | T1 1.5R @ 1150.49 |
| Target hit | 2023-08-08 13:25:00 | 1150.60 | 1149.54 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — SELL (started 2023-08-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:20:00 | 1144.20 | 1148.22 | 0.00 | ORB-short ORB[1145.95,1159.30] vol=3.2x ATR=2.75 |
| Stop hit — per-position SL triggered | 2023-08-09 10:30:00 | 1146.95 | 1148.15 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:20:00 | 1147.05 | 1151.92 | 0.00 | ORB-short ORB[1148.30,1157.90] vol=1.6x ATR=2.70 |
| Stop hit — per-position SL triggered | 2023-08-10 10:25:00 | 1149.75 | 1151.81 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 09:35:00 | 1138.50 | 1129.69 | 0.00 | ORB-long ORB[1120.50,1132.15] vol=1.7x ATR=3.46 |
| Stop hit — per-position SL triggered | 2023-08-14 10:15:00 | 1135.04 | 1134.22 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 10:55:00 | 1141.90 | 1137.12 | 0.00 | ORB-long ORB[1127.15,1139.95] vol=1.9x ATR=2.32 |
| Stop hit — per-position SL triggered | 2023-08-21 11:30:00 | 1139.58 | 1137.80 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 10:55:00 | 1133.30 | 1135.19 | 0.00 | ORB-short ORB[1135.00,1147.00] vol=1.7x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 11:15:00 | 1130.31 | 1134.65 | 0.00 | T1 1.5R @ 1130.31 |
| Stop hit — per-position SL triggered | 2023-08-22 11:55:00 | 1133.30 | 1134.23 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-08-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 09:45:00 | 1115.10 | 1112.67 | 0.00 | ORB-long ORB[1105.15,1114.25] vol=2.5x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 10:20:00 | 1120.64 | 1114.46 | 0.00 | T1 1.5R @ 1120.64 |
| Target hit | 2023-08-28 11:55:00 | 1117.35 | 1117.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2023-08-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 10:50:00 | 1114.70 | 1116.62 | 0.00 | ORB-short ORB[1118.40,1125.05] vol=2.2x ATR=1.74 |
| Stop hit — per-position SL triggered | 2023-08-29 11:30:00 | 1116.44 | 1116.32 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 10:30:00 | 1112.00 | 1115.47 | 0.00 | ORB-short ORB[1112.40,1117.50] vol=1.9x ATR=2.06 |
| Stop hit — per-position SL triggered | 2023-08-30 11:15:00 | 1114.06 | 1113.80 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 10:15:00 | 1139.95 | 1134.98 | 0.00 | ORB-long ORB[1130.00,1137.00] vol=2.0x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 10:25:00 | 1144.07 | 1138.13 | 0.00 | T1 1.5R @ 1144.07 |
| Target hit | 2023-09-06 11:50:00 | 1141.10 | 1141.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2023-09-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-11 10:50:00 | 1129.65 | 1133.53 | 0.00 | ORB-short ORB[1130.45,1140.00] vol=3.6x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 13:00:00 | 1126.69 | 1130.63 | 0.00 | T1 1.5R @ 1126.69 |
| Stop hit — per-position SL triggered | 2023-09-11 13:05:00 | 1129.65 | 1130.51 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 1153.65 | 1148.48 | 0.00 | ORB-long ORB[1137.10,1153.30] vol=1.6x ATR=2.58 |
| Stop hit — per-position SL triggered | 2023-09-12 09:35:00 | 1151.07 | 1148.61 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-13 09:55:00 | 1145.00 | 1149.87 | 0.00 | ORB-short ORB[1146.00,1156.70] vol=2.2x ATR=2.80 |
| Stop hit — per-position SL triggered | 2023-09-13 10:05:00 | 1147.80 | 1149.53 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 10:40:00 | 1143.90 | 1147.25 | 0.00 | ORB-short ORB[1146.05,1150.90] vol=1.6x ATR=2.29 |
| Stop hit — per-position SL triggered | 2023-09-14 10:45:00 | 1146.19 | 1147.07 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 11:10:00 | 1151.10 | 1149.55 | 0.00 | ORB-long ORB[1144.00,1148.95] vol=1.7x ATR=1.97 |
| Stop hit — per-position SL triggered | 2023-09-15 11:35:00 | 1149.13 | 1149.62 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-20 10:20:00 | 1155.85 | 1149.39 | 0.00 | ORB-long ORB[1139.00,1150.00] vol=2.1x ATR=2.70 |
| Stop hit — per-position SL triggered | 2023-09-20 10:25:00 | 1153.15 | 1150.15 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-09-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:50:00 | 1129.10 | 1132.81 | 0.00 | ORB-short ORB[1131.30,1148.00] vol=3.4x ATR=2.93 |
| Stop hit — per-position SL triggered | 2023-09-22 11:50:00 | 1132.03 | 1132.16 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 10:40:00 | 1118.70 | 1121.51 | 0.00 | ORB-short ORB[1120.05,1128.75] vol=7.5x ATR=1.69 |
| Stop hit — per-position SL triggered | 2023-09-26 10:55:00 | 1120.39 | 1121.09 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-09-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 10:25:00 | 1150.10 | 1146.99 | 0.00 | ORB-long ORB[1141.05,1149.35] vol=1.8x ATR=2.36 |
| Stop hit — per-position SL triggered | 2023-09-28 10:50:00 | 1147.74 | 1148.13 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-09-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:35:00 | 1150.30 | 1142.36 | 0.00 | ORB-long ORB[1134.30,1146.10] vol=1.8x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 10:45:00 | 1155.04 | 1147.45 | 0.00 | T1 1.5R @ 1155.04 |
| Target hit | 2023-09-29 15:00:00 | 1158.40 | 1159.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — SELL (started 2023-10-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-03 10:25:00 | 1148.20 | 1150.48 | 0.00 | ORB-short ORB[1151.05,1161.95] vol=3.4x ATR=3.20 |
| Stop hit — per-position SL triggered | 2023-10-03 10:50:00 | 1151.40 | 1150.09 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-10-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:50:00 | 1120.30 | 1121.43 | 0.00 | ORB-short ORB[1120.75,1128.65] vol=1.7x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 11:25:00 | 1117.19 | 1120.70 | 0.00 | T1 1.5R @ 1117.19 |
| Stop hit — per-position SL triggered | 2023-10-05 15:05:00 | 1120.30 | 1117.82 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-10-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 10:30:00 | 1123.75 | 1126.16 | 0.00 | ORB-short ORB[1126.70,1132.55] vol=1.6x ATR=2.20 |
| Stop hit — per-position SL triggered | 2023-10-12 10:40:00 | 1125.95 | 1126.11 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:45:00 | 1148.00 | 1140.78 | 0.00 | ORB-long ORB[1134.05,1143.00] vol=1.8x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 10:00:00 | 1151.64 | 1145.06 | 0.00 | T1 1.5R @ 1151.64 |
| Target hit | 2023-10-18 12:10:00 | 1152.75 | 1153.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — SELL (started 2023-10-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 10:15:00 | 1141.50 | 1143.86 | 0.00 | ORB-short ORB[1143.00,1150.00] vol=7.1x ATR=2.34 |
| Stop hit — per-position SL triggered | 2023-10-19 10:20:00 | 1143.84 | 1143.67 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-10-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-25 10:40:00 | 1118.45 | 1122.13 | 0.00 | ORB-short ORB[1123.15,1129.70] vol=2.5x ATR=2.31 |
| Stop hit — per-position SL triggered | 2023-10-25 10:45:00 | 1120.76 | 1121.98 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-10-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 10:55:00 | 1105.25 | 1105.84 | 0.00 | ORB-short ORB[1106.50,1117.95] vol=1.9x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 13:30:00 | 1101.04 | 1105.07 | 0.00 | T1 1.5R @ 1101.04 |
| Stop hit — per-position SL triggered | 2023-10-26 14:10:00 | 1105.25 | 1104.48 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 09:45:00 | 1102.90 | 1109.64 | 0.00 | ORB-short ORB[1109.60,1119.60] vol=3.2x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 10:00:00 | 1098.62 | 1105.45 | 0.00 | T1 1.5R @ 1098.62 |
| Stop hit — per-position SL triggered | 2023-10-31 10:05:00 | 1102.90 | 1103.99 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:40:00 | 1150.55 | 1147.56 | 0.00 | ORB-long ORB[1140.00,1149.80] vol=1.9x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-11-06 10:45:00 | 1148.72 | 1147.74 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 10:45:00 | 1180.80 | 1176.72 | 0.00 | ORB-long ORB[1172.05,1178.35] vol=2.2x ATR=2.26 |
| Stop hit — per-position SL triggered | 2023-11-08 11:10:00 | 1178.54 | 1177.43 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 10:45:00 | 1178.05 | 1175.20 | 0.00 | ORB-long ORB[1170.60,1175.30] vol=5.1x ATR=2.23 |
| Target hit | 2023-11-10 15:20:00 | 1179.10 | 1178.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2023-11-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 09:55:00 | 1185.00 | 1184.25 | 0.00 | ORB-long ORB[1176.85,1183.80] vol=7.5x ATR=2.55 |
| Stop hit — per-position SL triggered | 2023-11-13 10:35:00 | 1182.45 | 1184.22 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-11-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-15 10:40:00 | 1174.70 | 1175.99 | 0.00 | ORB-short ORB[1175.05,1183.95] vol=6.2x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 10:45:00 | 1172.32 | 1175.89 | 0.00 | T1 1.5R @ 1172.32 |
| Stop hit — per-position SL triggered | 2023-11-15 10:55:00 | 1174.70 | 1175.76 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-11-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:00:00 | 1189.20 | 1186.49 | 0.00 | ORB-long ORB[1180.25,1185.60] vol=1.9x ATR=1.92 |
| Stop hit — per-position SL triggered | 2023-11-16 11:35:00 | 1187.28 | 1186.99 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 11:15:00 | 1194.60 | 1191.62 | 0.00 | ORB-long ORB[1186.25,1192.10] vol=1.7x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 12:15:00 | 1197.46 | 1192.59 | 0.00 | T1 1.5R @ 1197.46 |
| Target hit | 2023-11-21 15:20:00 | 1203.50 | 1198.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2023-11-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:50:00 | 1197.10 | 1194.09 | 0.00 | ORB-long ORB[1184.50,1194.05] vol=1.6x ATR=2.20 |
| Stop hit — per-position SL triggered | 2023-11-29 11:25:00 | 1194.90 | 1195.31 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-11-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 09:30:00 | 1210.50 | 1205.48 | 0.00 | ORB-long ORB[1192.35,1208.55] vol=1.8x ATR=2.92 |
| Stop hit — per-position SL triggered | 2023-11-30 09:35:00 | 1207.58 | 1206.04 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 10:00:00 | 1255.35 | 1249.94 | 0.00 | ORB-long ORB[1243.15,1254.60] vol=2.7x ATR=3.24 |
| Stop hit — per-position SL triggered | 2023-12-06 10:05:00 | 1252.11 | 1251.19 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-12-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-07 10:25:00 | 1229.35 | 1235.08 | 0.00 | ORB-short ORB[1231.00,1238.65] vol=1.5x ATR=2.42 |
| Stop hit — per-position SL triggered | 2023-12-07 10:55:00 | 1231.77 | 1233.94 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2023-12-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:40:00 | 1230.30 | 1238.87 | 0.00 | ORB-short ORB[1240.40,1250.00] vol=2.9x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 11:40:00 | 1225.23 | 1233.56 | 0.00 | T1 1.5R @ 1225.23 |
| Target hit | 2023-12-12 15:20:00 | 1218.70 | 1226.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:15:00 | 1243.30 | 1249.71 | 0.00 | ORB-short ORB[1257.65,1267.95] vol=1.5x ATR=3.49 |
| Stop hit — per-position SL triggered | 2023-12-19 12:25:00 | 1246.79 | 1246.89 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-12-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 10:30:00 | 1238.30 | 1240.78 | 0.00 | ORB-short ORB[1242.10,1255.00] vol=3.8x ATR=2.26 |
| Stop hit — per-position SL triggered | 2023-12-20 10:45:00 | 1240.56 | 1240.44 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:45:00 | 1246.35 | 1241.97 | 0.00 | ORB-long ORB[1236.75,1244.75] vol=2.1x ATR=3.25 |
| Stop hit — per-position SL triggered | 2023-12-22 11:20:00 | 1243.10 | 1245.53 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2023-12-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:20:00 | 1246.20 | 1245.28 | 0.00 | ORB-long ORB[1239.70,1246.00] vol=2.3x ATR=2.25 |
| Stop hit — per-position SL triggered | 2023-12-26 10:50:00 | 1243.95 | 1245.90 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-01-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:55:00 | 1304.15 | 1302.09 | 0.00 | ORB-long ORB[1297.05,1304.00] vol=1.6x ATR=2.56 |
| Stop hit — per-position SL triggered | 2024-01-04 10:00:00 | 1301.59 | 1302.21 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-01-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:40:00 | 1317.05 | 1310.85 | 0.00 | ORB-long ORB[1306.85,1314.95] vol=2.0x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 10:50:00 | 1320.73 | 1314.95 | 0.00 | T1 1.5R @ 1320.73 |
| Target hit | 2024-01-09 15:10:00 | 1321.95 | 1323.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 70 — BUY (started 2024-01-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 10:40:00 | 1319.95 | 1314.49 | 0.00 | ORB-long ORB[1305.55,1319.00] vol=2.4x ATR=2.60 |
| Stop hit — per-position SL triggered | 2024-01-12 11:05:00 | 1317.35 | 1316.25 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 10:35:00 | 1298.30 | 1305.21 | 0.00 | ORB-short ORB[1308.00,1320.45] vol=5.0x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-01-17 11:00:00 | 1301.31 | 1303.07 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-18 10:25:00 | 1311.00 | 1302.75 | 0.00 | ORB-long ORB[1290.00,1303.80] vol=2.5x ATR=4.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 10:40:00 | 1317.61 | 1304.82 | 0.00 | T1 1.5R @ 1317.61 |
| Stop hit — per-position SL triggered | 2024-01-18 11:00:00 | 1311.00 | 1306.36 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 10:50:00 | 1331.15 | 1336.33 | 0.00 | ORB-short ORB[1335.70,1344.40] vol=3.3x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-01-19 12:20:00 | 1334.50 | 1335.24 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:50:00 | 1331.00 | 1339.10 | 0.00 | ORB-short ORB[1337.40,1346.35] vol=1.9x ATR=2.45 |
| Stop hit — per-position SL triggered | 2024-01-20 11:25:00 | 1333.45 | 1337.97 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-01-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 09:40:00 | 1372.80 | 1374.45 | 0.00 | ORB-short ORB[1375.00,1383.70] vol=2.8x ATR=3.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 10:10:00 | 1367.08 | 1373.24 | 0.00 | T1 1.5R @ 1367.08 |
| Target hit | 2024-01-25 13:35:00 | 1362.60 | 1362.51 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — SELL (started 2024-02-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 10:45:00 | 1408.00 | 1420.81 | 0.00 | ORB-short ORB[1410.20,1430.45] vol=1.6x ATR=5.57 |
| Stop hit — per-position SL triggered | 2024-02-01 10:50:00 | 1413.57 | 1420.34 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-02-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 11:10:00 | 1422.90 | 1418.31 | 0.00 | ORB-long ORB[1407.90,1422.05] vol=3.3x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:15:00 | 1427.21 | 1419.02 | 0.00 | T1 1.5R @ 1427.21 |
| Stop hit — per-position SL triggered | 2024-02-02 12:15:00 | 1422.90 | 1423.64 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-02-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 09:30:00 | 1445.60 | 1437.40 | 0.00 | ORB-long ORB[1423.60,1439.05] vol=3.7x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 09:50:00 | 1452.84 | 1441.94 | 0.00 | T1 1.5R @ 1452.84 |
| Stop hit — per-position SL triggered | 2024-02-05 10:20:00 | 1445.60 | 1444.42 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-02-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:05:00 | 1490.55 | 1500.09 | 0.00 | ORB-short ORB[1492.30,1506.00] vol=1.6x ATR=4.07 |
| Stop hit — per-position SL triggered | 2024-02-08 12:10:00 | 1494.62 | 1498.19 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-02-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-09 10:20:00 | 1517.55 | 1503.70 | 0.00 | ORB-long ORB[1492.25,1503.60] vol=3.0x ATR=5.11 |
| Stop hit — per-position SL triggered | 2024-02-09 10:25:00 | 1512.44 | 1505.12 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-12 09:30:00 | 1550.65 | 1545.49 | 0.00 | ORB-long ORB[1531.15,1547.50] vol=2.3x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-02-12 09:35:00 | 1546.06 | 1545.76 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 09:35:00 | 1539.80 | 1536.42 | 0.00 | ORB-long ORB[1525.45,1539.25] vol=2.7x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 10:30:00 | 1545.89 | 1538.86 | 0.00 | T1 1.5R @ 1545.89 |
| Target hit | 2024-02-13 11:35:00 | 1541.55 | 1541.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 83 — SELL (started 2024-02-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-14 09:35:00 | 1524.05 | 1528.56 | 0.00 | ORB-short ORB[1525.10,1545.80] vol=1.9x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-14 10:05:00 | 1518.19 | 1525.16 | 0.00 | T1 1.5R @ 1518.19 |
| Target hit | 2024-02-14 15:15:00 | 1520.45 | 1519.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 84 — SELL (started 2024-02-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 10:05:00 | 1508.40 | 1514.27 | 0.00 | ORB-short ORB[1514.10,1529.05] vol=2.1x ATR=3.66 |
| Stop hit — per-position SL triggered | 2024-02-15 10:35:00 | 1512.06 | 1512.89 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-02-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 09:45:00 | 1515.40 | 1511.20 | 0.00 | ORB-long ORB[1504.85,1515.00] vol=3.7x ATR=3.74 |
| Stop hit — per-position SL triggered | 2024-02-16 09:55:00 | 1511.66 | 1511.76 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-02-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 09:55:00 | 1519.95 | 1527.62 | 0.00 | ORB-short ORB[1527.75,1533.65] vol=1.6x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 10:10:00 | 1515.19 | 1525.87 | 0.00 | T1 1.5R @ 1515.19 |
| Stop hit — per-position SL triggered | 2024-02-20 10:15:00 | 1519.95 | 1525.15 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-02-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 10:40:00 | 1545.65 | 1541.84 | 0.00 | ORB-long ORB[1535.25,1545.00] vol=3.2x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-02-21 11:20:00 | 1542.40 | 1542.28 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 1569.60 | 1576.08 | 0.00 | ORB-short ORB[1576.30,1587.45] vol=1.9x ATR=2.42 |
| Stop hit — per-position SL triggered | 2024-02-28 11:00:00 | 1572.02 | 1575.57 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-02-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 10:00:00 | 1564.60 | 1568.83 | 0.00 | ORB-short ORB[1566.75,1576.35] vol=3.4x ATR=3.39 |
| Stop hit — per-position SL triggered | 2024-02-29 10:05:00 | 1567.99 | 1568.63 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-03-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 10:45:00 | 1590.55 | 1596.56 | 0.00 | ORB-short ORB[1591.55,1603.80] vol=2.0x ATR=3.60 |
| Stop hit — per-position SL triggered | 2024-03-12 11:05:00 | 1594.15 | 1595.52 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-03-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-18 11:10:00 | 1555.10 | 1554.70 | 0.00 | ORB-long ORB[1543.45,1554.90] vol=4.6x ATR=3.74 |
| Stop hit — per-position SL triggered | 2024-03-18 11:55:00 | 1551.36 | 1553.87 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:15:00 | 1562.25 | 1567.13 | 0.00 | ORB-short ORB[1565.80,1579.75] vol=4.9x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-03-19 12:30:00 | 1566.53 | 1565.17 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-03-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 11:05:00 | 1530.80 | 1537.97 | 0.00 | ORB-short ORB[1543.70,1559.95] vol=1.8x ATR=5.53 |
| Stop hit — per-position SL triggered | 2024-03-20 11:35:00 | 1536.33 | 1535.84 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 11:15:00 | 1607.00 | 1601.52 | 0.00 | ORB-long ORB[1592.10,1605.90] vol=3.8x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-03-27 11:20:00 | 1604.22 | 1602.41 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 11:15:00 | 1623.85 | 1623.66 | 0.00 | ORB-long ORB[1607.95,1619.90] vol=2.1x ATR=5.07 |
| Stop hit — per-position SL triggered | 2024-03-28 15:00:00 | 1618.78 | 1624.26 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-01 10:55:00 | 1620.00 | 1624.93 | 0.00 | ORB-short ORB[1621.10,1633.75] vol=1.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-04-01 11:30:00 | 1623.17 | 1623.93 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 1587.75 | 1604.24 | 0.00 | ORB-short ORB[1610.30,1626.15] vol=2.4x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-04-04 10:55:00 | 1592.43 | 1603.65 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2024-04-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 09:55:00 | 1635.45 | 1627.40 | 0.00 | ORB-long ORB[1606.20,1623.25] vol=1.6x ATR=5.79 |
| Stop hit — per-position SL triggered | 2024-04-05 10:00:00 | 1629.66 | 1627.70 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2024-04-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 10:45:00 | 1523.35 | 1543.21 | 0.00 | ORB-short ORB[1542.20,1562.45] vol=3.4x ATR=4.61 |
| Stop hit — per-position SL triggered | 2024-04-23 10:50:00 | 1527.96 | 1541.12 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2024-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:55:00 | 1494.35 | 1491.97 | 0.00 | ORB-long ORB[1479.35,1492.95] vol=1.8x ATR=3.37 |
| Stop hit — per-position SL triggered | 2024-04-24 11:05:00 | 1490.98 | 1491.96 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-26 11:15:00 | 1505.60 | 1514.33 | 0.00 | ORB-short ORB[1518.90,1532.00] vol=1.7x ATR=3.88 |
| Stop hit — per-position SL triggered | 2024-04-26 11:30:00 | 1509.48 | 1513.65 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2024-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 11:10:00 | 1523.15 | 1520.11 | 0.00 | ORB-long ORB[1508.75,1522.60] vol=1.6x ATR=3.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 11:45:00 | 1528.85 | 1521.14 | 0.00 | T1 1.5R @ 1528.85 |
| Stop hit — per-position SL triggered | 2024-05-06 12:50:00 | 1523.15 | 1524.16 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2024-05-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-07 09:55:00 | 1541.05 | 1535.94 | 0.00 | ORB-long ORB[1529.15,1536.75] vol=1.7x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-05-07 10:20:00 | 1537.60 | 1537.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-18 10:10:00 | 930.00 | 2023-05-18 10:40:00 | 931.94 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-05-19 10:45:00 | 926.35 | 2023-05-19 11:00:00 | 928.18 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-05-22 10:45:00 | 941.10 | 2023-05-22 11:15:00 | 939.48 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-05-25 09:40:00 | 950.65 | 2023-05-25 10:05:00 | 948.39 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-05-30 10:15:00 | 960.70 | 2023-05-30 10:35:00 | 962.41 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-05-31 09:30:00 | 970.35 | 2023-05-31 09:40:00 | 973.88 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-05-31 09:30:00 | 970.35 | 2023-05-31 10:10:00 | 971.80 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2023-06-05 09:35:00 | 1006.25 | 2023-06-05 09:55:00 | 1009.91 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-06-05 09:35:00 | 1006.25 | 2023-06-05 13:40:00 | 1009.80 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2023-06-08 10:40:00 | 1004.30 | 2023-06-08 11:15:00 | 1002.10 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-06-08 10:40:00 | 1004.30 | 2023-06-08 15:20:00 | 988.55 | TARGET_HIT | 0.50 | 1.57% |
| SELL | retest1 | 2023-06-23 11:05:00 | 981.25 | 2023-06-23 11:30:00 | 983.27 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-26 09:45:00 | 998.40 | 2023-06-26 10:00:00 | 996.19 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-28 11:05:00 | 1008.90 | 2023-06-28 11:15:00 | 1006.80 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-30 10:50:00 | 1046.70 | 2023-06-30 11:15:00 | 1052.10 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-06-30 10:50:00 | 1046.70 | 2023-06-30 13:50:00 | 1046.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-03 09:40:00 | 1043.25 | 2023-07-03 09:50:00 | 1045.88 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-07-06 10:05:00 | 1052.95 | 2023-07-06 10:10:00 | 1050.44 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-07 11:15:00 | 1049.60 | 2023-07-07 11:25:00 | 1047.29 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-11 09:40:00 | 1062.30 | 2023-07-11 10:20:00 | 1066.63 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-07-11 09:40:00 | 1062.30 | 2023-07-11 10:50:00 | 1062.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-12 10:30:00 | 1081.10 | 2023-07-12 11:35:00 | 1078.90 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-07-18 11:15:00 | 1069.05 | 2023-07-18 11:40:00 | 1066.43 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-07-18 11:15:00 | 1069.05 | 2023-07-18 11:45:00 | 1069.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-19 09:50:00 | 1077.00 | 2023-07-19 10:00:00 | 1080.43 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-07-19 09:50:00 | 1077.00 | 2023-07-19 10:30:00 | 1077.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-20 10:45:00 | 1096.60 | 2023-07-20 11:00:00 | 1093.92 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-25 10:50:00 | 1097.75 | 2023-07-25 10:55:00 | 1099.76 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-07-27 09:50:00 | 1136.00 | 2023-07-27 09:55:00 | 1131.82 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-08-02 11:15:00 | 1131.60 | 2023-08-02 11:25:00 | 1133.94 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-08-08 09:45:00 | 1154.85 | 2023-08-08 10:05:00 | 1150.49 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-08-08 09:45:00 | 1154.85 | 2023-08-08 13:25:00 | 1150.60 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2023-08-09 10:20:00 | 1144.20 | 2023-08-09 10:30:00 | 1146.95 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-10 10:20:00 | 1147.05 | 2023-08-10 10:25:00 | 1149.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-08-14 09:35:00 | 1138.50 | 2023-08-14 10:15:00 | 1135.04 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-08-21 10:55:00 | 1141.90 | 2023-08-21 11:30:00 | 1139.58 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-22 10:55:00 | 1133.30 | 2023-08-22 11:15:00 | 1130.31 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-08-22 10:55:00 | 1133.30 | 2023-08-22 11:55:00 | 1133.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-28 09:45:00 | 1115.10 | 2023-08-28 10:20:00 | 1120.64 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-08-28 09:45:00 | 1115.10 | 2023-08-28 11:55:00 | 1117.35 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2023-08-29 10:50:00 | 1114.70 | 2023-08-29 11:30:00 | 1116.44 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-08-30 10:30:00 | 1112.00 | 2023-08-30 11:15:00 | 1114.06 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-09-06 10:15:00 | 1139.95 | 2023-09-06 10:25:00 | 1144.07 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-09-06 10:15:00 | 1139.95 | 2023-09-06 11:50:00 | 1141.10 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2023-09-11 10:50:00 | 1129.65 | 2023-09-11 13:00:00 | 1126.69 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-09-11 10:50:00 | 1129.65 | 2023-09-11 13:05:00 | 1129.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-12 09:30:00 | 1153.65 | 2023-09-12 09:35:00 | 1151.07 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-09-13 09:55:00 | 1145.00 | 2023-09-13 10:05:00 | 1147.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-09-14 10:40:00 | 1143.90 | 2023-09-14 10:45:00 | 1146.19 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-09-15 11:10:00 | 1151.10 | 2023-09-15 11:35:00 | 1149.13 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-09-20 10:20:00 | 1155.85 | 2023-09-20 10:25:00 | 1153.15 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-09-22 10:50:00 | 1129.10 | 2023-09-22 11:50:00 | 1132.03 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-26 10:40:00 | 1118.70 | 2023-09-26 10:55:00 | 1120.39 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-09-28 10:25:00 | 1150.10 | 2023-09-28 10:50:00 | 1147.74 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-09-29 10:35:00 | 1150.30 | 2023-09-29 10:45:00 | 1155.04 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-09-29 10:35:00 | 1150.30 | 2023-09-29 15:00:00 | 1158.40 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2023-10-03 10:25:00 | 1148.20 | 2023-10-03 10:50:00 | 1151.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-10-05 10:50:00 | 1120.30 | 2023-10-05 11:25:00 | 1117.19 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-10-05 10:50:00 | 1120.30 | 2023-10-05 15:05:00 | 1120.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-12 10:30:00 | 1123.75 | 2023-10-12 10:40:00 | 1125.95 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-10-18 09:45:00 | 1148.00 | 2023-10-18 10:00:00 | 1151.64 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-10-18 09:45:00 | 1148.00 | 2023-10-18 12:10:00 | 1152.75 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2023-10-19 10:15:00 | 1141.50 | 2023-10-19 10:20:00 | 1143.84 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-10-25 10:40:00 | 1118.45 | 2023-10-25 10:45:00 | 1120.76 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-10-26 10:55:00 | 1105.25 | 2023-10-26 13:30:00 | 1101.04 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-10-26 10:55:00 | 1105.25 | 2023-10-26 14:10:00 | 1105.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-31 09:45:00 | 1102.90 | 2023-10-31 10:00:00 | 1098.62 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-10-31 09:45:00 | 1102.90 | 2023-10-31 10:05:00 | 1102.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-06 10:40:00 | 1150.55 | 2023-11-06 10:45:00 | 1148.72 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-08 10:45:00 | 1180.80 | 2023-11-08 11:10:00 | 1178.54 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-11-10 10:45:00 | 1178.05 | 2023-11-10 15:20:00 | 1179.10 | TARGET_HIT | 1.00 | 0.09% |
| BUY | retest1 | 2023-11-13 09:55:00 | 1185.00 | 2023-11-13 10:35:00 | 1182.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-11-15 10:40:00 | 1174.70 | 2023-11-15 10:45:00 | 1172.32 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2023-11-15 10:40:00 | 1174.70 | 2023-11-15 10:55:00 | 1174.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-16 11:00:00 | 1189.20 | 2023-11-16 11:35:00 | 1187.28 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-21 11:15:00 | 1194.60 | 2023-11-21 12:15:00 | 1197.46 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-11-21 11:15:00 | 1194.60 | 2023-11-21 15:20:00 | 1203.50 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2023-11-29 10:50:00 | 1197.10 | 2023-11-29 11:25:00 | 1194.90 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-11-30 09:30:00 | 1210.50 | 2023-11-30 09:35:00 | 1207.58 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-12-06 10:00:00 | 1255.35 | 2023-12-06 10:05:00 | 1252.11 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-12-07 10:25:00 | 1229.35 | 2023-12-07 10:55:00 | 1231.77 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-12-12 10:40:00 | 1230.30 | 2023-12-12 11:40:00 | 1225.23 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-12-12 10:40:00 | 1230.30 | 2023-12-12 15:20:00 | 1218.70 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2023-12-19 10:15:00 | 1243.30 | 2023-12-19 12:25:00 | 1246.79 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-12-20 10:30:00 | 1238.30 | 2023-12-20 10:45:00 | 1240.56 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-12-22 09:45:00 | 1246.35 | 2023-12-22 11:20:00 | 1243.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-26 10:20:00 | 1246.20 | 2023-12-26 10:50:00 | 1243.95 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-01-04 09:55:00 | 1304.15 | 2024-01-04 10:00:00 | 1301.59 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-01-09 10:40:00 | 1317.05 | 2024-01-09 10:50:00 | 1320.73 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-01-09 10:40:00 | 1317.05 | 2024-01-09 15:10:00 | 1321.95 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2024-01-12 10:40:00 | 1319.95 | 2024-01-12 11:05:00 | 1317.35 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-01-17 10:35:00 | 1298.30 | 2024-01-17 11:00:00 | 1301.31 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-01-18 10:25:00 | 1311.00 | 2024-01-18 10:40:00 | 1317.61 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-01-18 10:25:00 | 1311.00 | 2024-01-18 11:00:00 | 1311.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-19 10:50:00 | 1331.15 | 2024-01-19 12:20:00 | 1334.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-01-20 10:50:00 | 1331.00 | 2024-01-20 11:25:00 | 1333.45 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-01-25 09:40:00 | 1372.80 | 2024-01-25 10:10:00 | 1367.08 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-01-25 09:40:00 | 1372.80 | 2024-01-25 13:35:00 | 1362.60 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2024-02-01 10:45:00 | 1408.00 | 2024-02-01 10:50:00 | 1413.57 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-02-02 11:10:00 | 1422.90 | 2024-02-02 11:15:00 | 1427.21 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-02-02 11:10:00 | 1422.90 | 2024-02-02 12:15:00 | 1422.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-05 09:30:00 | 1445.60 | 2024-02-05 09:50:00 | 1452.84 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-02-05 09:30:00 | 1445.60 | 2024-02-05 10:20:00 | 1445.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-08 11:05:00 | 1490.55 | 2024-02-08 12:10:00 | 1494.62 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-02-09 10:20:00 | 1517.55 | 2024-02-09 10:25:00 | 1512.44 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-02-12 09:30:00 | 1550.65 | 2024-02-12 09:35:00 | 1546.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-02-13 09:35:00 | 1539.80 | 2024-02-13 10:30:00 | 1545.89 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-02-13 09:35:00 | 1539.80 | 2024-02-13 11:35:00 | 1541.55 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2024-02-14 09:35:00 | 1524.05 | 2024-02-14 10:05:00 | 1518.19 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-02-14 09:35:00 | 1524.05 | 2024-02-14 15:15:00 | 1520.45 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2024-02-15 10:05:00 | 1508.40 | 2024-02-15 10:35:00 | 1512.06 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-02-16 09:45:00 | 1515.40 | 2024-02-16 09:55:00 | 1511.66 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-20 09:55:00 | 1519.95 | 2024-02-20 10:10:00 | 1515.19 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-02-20 09:55:00 | 1519.95 | 2024-02-20 10:15:00 | 1519.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-21 10:40:00 | 1545.65 | 2024-02-21 11:20:00 | 1542.40 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-02-28 10:50:00 | 1569.60 | 2024-02-28 11:00:00 | 1572.02 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-02-29 10:00:00 | 1564.60 | 2024-02-29 10:05:00 | 1567.99 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-03-12 10:45:00 | 1590.55 | 2024-03-12 11:05:00 | 1594.15 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-03-18 11:10:00 | 1555.10 | 2024-03-18 11:55:00 | 1551.36 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-03-19 10:15:00 | 1562.25 | 2024-03-19 12:30:00 | 1566.53 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-03-20 11:05:00 | 1530.80 | 2024-03-20 11:35:00 | 1536.33 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-03-27 11:15:00 | 1607.00 | 2024-03-27 11:20:00 | 1604.22 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-03-28 11:15:00 | 1623.85 | 2024-03-28 15:00:00 | 1618.78 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-04-01 10:55:00 | 1620.00 | 2024-04-01 11:30:00 | 1623.17 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-04-04 10:50:00 | 1587.75 | 2024-04-04 10:55:00 | 1592.43 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-05 09:55:00 | 1635.45 | 2024-04-05 10:00:00 | 1629.66 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-04-23 10:45:00 | 1523.35 | 2024-04-23 10:50:00 | 1527.96 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-24 10:55:00 | 1494.35 | 2024-04-24 11:05:00 | 1490.98 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-04-26 11:15:00 | 1505.60 | 2024-04-26 11:30:00 | 1509.48 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-05-06 11:10:00 | 1523.15 | 2024-05-06 11:45:00 | 1528.85 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-05-06 11:10:00 | 1523.15 | 2024-05-06 12:50:00 | 1523.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-07 09:55:00 | 1541.05 | 2024-05-07 10:20:00 | 1537.60 | STOP_HIT | 1.00 | -0.22% |
