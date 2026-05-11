# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (53929 bars)
- **Last close:** 1671.00
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
| ENTRY1 | 77 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 16 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 107 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 61
- **Target hits / Stop hits / Partials:** 16 / 61 / 30
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 13.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 22 | 40.0% | 7 | 33 | 15 | 0.12% | 6.4% |
| BUY @ 2nd Alert (retest1) | 55 | 22 | 40.0% | 7 | 33 | 15 | 0.12% | 6.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 52 | 24 | 46.2% | 9 | 28 | 15 | 0.13% | 6.7% |
| SELL @ 2nd Alert (retest1) | 52 | 24 | 46.2% | 9 | 28 | 15 | 0.13% | 6.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 107 | 46 | 43.0% | 16 | 61 | 30 | 0.12% | 13.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 10:25:00 | 1015.10 | 1008.19 | 0.00 | ORB-long ORB[993.45,1000.70] vol=2.4x ATR=5.43 |
| Stop hit — per-position SL triggered | 2023-05-12 11:20:00 | 1009.67 | 1009.29 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 10:05:00 | 1020.45 | 1014.10 | 0.00 | ORB-long ORB[1003.15,1016.10] vol=4.4x ATR=3.47 |
| Stop hit — per-position SL triggered | 2023-05-16 12:30:00 | 1016.98 | 1017.98 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 10:55:00 | 1048.60 | 1042.29 | 0.00 | ORB-long ORB[1034.35,1047.45] vol=4.8x ATR=2.52 |
| Stop hit — per-position SL triggered | 2023-05-22 11:00:00 | 1046.08 | 1042.51 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 09:30:00 | 1061.05 | 1057.79 | 0.00 | ORB-long ORB[1053.00,1059.45] vol=1.8x ATR=2.49 |
| Stop hit — per-position SL triggered | 2023-05-23 09:40:00 | 1058.56 | 1059.43 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:35:00 | 1053.85 | 1051.30 | 0.00 | ORB-long ORB[1047.95,1053.05] vol=1.5x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 09:45:00 | 1057.46 | 1052.13 | 0.00 | T1 1.5R @ 1057.46 |
| Stop hit — per-position SL triggered | 2023-05-25 09:55:00 | 1053.85 | 1052.63 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-05-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 09:35:00 | 1051.70 | 1053.42 | 0.00 | ORB-short ORB[1052.65,1063.40] vol=4.2x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-26 10:25:00 | 1047.78 | 1052.77 | 0.00 | T1 1.5R @ 1047.78 |
| Target hit | 2023-05-26 15:00:00 | 1049.00 | 1048.48 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2023-05-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 09:45:00 | 1048.65 | 1053.24 | 0.00 | ORB-short ORB[1051.10,1058.00] vol=2.0x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-30 09:50:00 | 1045.34 | 1051.86 | 0.00 | T1 1.5R @ 1045.34 |
| Target hit | 2023-05-30 14:25:00 | 1043.75 | 1043.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2023-06-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 10:35:00 | 1039.75 | 1044.26 | 0.00 | ORB-short ORB[1043.00,1049.65] vol=1.5x ATR=2.13 |
| Stop hit — per-position SL triggered | 2023-06-02 10:45:00 | 1041.88 | 1044.01 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:20:00 | 1070.90 | 1063.52 | 0.00 | ORB-long ORB[1055.80,1064.60] vol=1.6x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 10:30:00 | 1075.06 | 1066.75 | 0.00 | T1 1.5R @ 1075.06 |
| Stop hit — per-position SL triggered | 2023-06-07 10:55:00 | 1070.90 | 1068.24 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 09:35:00 | 1088.75 | 1083.76 | 0.00 | ORB-long ORB[1075.60,1086.65] vol=2.1x ATR=3.45 |
| Stop hit — per-position SL triggered | 2023-06-13 09:55:00 | 1085.30 | 1085.25 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:40:00 | 1114.00 | 1104.75 | 0.00 | ORB-long ORB[1096.60,1106.60] vol=2.2x ATR=3.45 |
| Stop hit — per-position SL triggered | 2023-06-15 09:55:00 | 1110.55 | 1106.52 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 09:40:00 | 1143.80 | 1138.97 | 0.00 | ORB-long ORB[1127.55,1140.95] vol=2.0x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 09:50:00 | 1148.20 | 1142.83 | 0.00 | T1 1.5R @ 1148.20 |
| Stop hit — per-position SL triggered | 2023-06-19 10:15:00 | 1143.80 | 1143.82 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 09:45:00 | 1112.35 | 1119.36 | 0.00 | ORB-short ORB[1118.75,1135.00] vol=2.5x ATR=3.80 |
| Stop hit — per-position SL triggered | 2023-06-22 10:05:00 | 1116.15 | 1118.38 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 10:15:00 | 1179.90 | 1172.50 | 0.00 | ORB-long ORB[1164.00,1176.15] vol=3.2x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 10:25:00 | 1186.09 | 1173.52 | 0.00 | T1 1.5R @ 1186.09 |
| Target hit | 2023-07-04 13:45:00 | 1190.50 | 1191.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2023-07-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:40:00 | 1185.50 | 1175.24 | 0.00 | ORB-long ORB[1160.50,1171.95] vol=6.8x ATR=4.96 |
| Stop hit — per-position SL triggered | 2023-07-07 09:45:00 | 1180.54 | 1175.69 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 10:40:00 | 1157.20 | 1155.17 | 0.00 | ORB-long ORB[1143.55,1155.80] vol=3.9x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 11:10:00 | 1162.61 | 1156.66 | 0.00 | T1 1.5R @ 1162.61 |
| Target hit | 2023-07-19 15:20:00 | 1169.05 | 1163.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2023-07-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-21 10:50:00 | 1139.85 | 1151.12 | 0.00 | ORB-short ORB[1145.75,1159.00] vol=1.6x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 11:15:00 | 1133.94 | 1149.83 | 0.00 | T1 1.5R @ 1133.94 |
| Stop hit — per-position SL triggered | 2023-07-21 11:20:00 | 1139.85 | 1149.73 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 11:10:00 | 1129.10 | 1140.95 | 0.00 | ORB-short ORB[1142.45,1154.20] vol=2.1x ATR=3.53 |
| Stop hit — per-position SL triggered | 2023-07-25 11:15:00 | 1132.63 | 1140.27 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:15:00 | 1148.00 | 1142.25 | 0.00 | ORB-long ORB[1130.00,1139.90] vol=1.6x ATR=3.49 |
| Stop hit — per-position SL triggered | 2023-07-26 10:20:00 | 1144.51 | 1142.46 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 10:40:00 | 1148.65 | 1144.01 | 0.00 | ORB-long ORB[1134.05,1147.00] vol=1.6x ATR=3.24 |
| Stop hit — per-position SL triggered | 2023-07-28 11:05:00 | 1145.41 | 1145.07 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-08-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 10:25:00 | 1071.20 | 1073.74 | 0.00 | ORB-short ORB[1072.90,1083.80] vol=3.2x ATR=3.07 |
| Stop hit — per-position SL triggered | 2023-08-07 10:30:00 | 1074.27 | 1073.68 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 10:50:00 | 1064.25 | 1068.25 | 0.00 | ORB-short ORB[1064.40,1072.75] vol=1.9x ATR=3.03 |
| Stop hit — per-position SL triggered | 2023-08-08 10:55:00 | 1067.28 | 1068.09 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 10:40:00 | 1060.05 | 1058.73 | 0.00 | ORB-long ORB[1053.15,1059.05] vol=2.3x ATR=2.91 |
| Stop hit — per-position SL triggered | 2023-08-09 10:50:00 | 1057.14 | 1058.72 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:20:00 | 1044.65 | 1053.82 | 0.00 | ORB-short ORB[1051.05,1057.00] vol=2.1x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 10:45:00 | 1039.50 | 1050.40 | 0.00 | T1 1.5R @ 1039.50 |
| Stop hit — per-position SL triggered | 2023-08-10 10:55:00 | 1044.65 | 1050.26 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 10:40:00 | 1035.00 | 1028.50 | 0.00 | ORB-long ORB[1015.00,1028.45] vol=1.8x ATR=2.81 |
| Stop hit — per-position SL triggered | 2023-08-17 10:50:00 | 1032.19 | 1028.99 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:35:00 | 1006.35 | 1010.96 | 0.00 | ORB-short ORB[1007.30,1020.00] vol=1.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2023-08-18 09:45:00 | 1009.52 | 1009.55 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 10:15:00 | 1051.00 | 1046.85 | 0.00 | ORB-long ORB[1041.00,1049.85] vol=1.8x ATR=2.79 |
| Stop hit — per-position SL triggered | 2023-08-22 10:30:00 | 1048.21 | 1047.23 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:35:00 | 1064.95 | 1074.36 | 0.00 | ORB-short ORB[1073.45,1085.90] vol=1.7x ATR=3.39 |
| Stop hit — per-position SL triggered | 2023-08-25 10:45:00 | 1068.34 | 1073.56 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 10:45:00 | 1078.00 | 1084.52 | 0.00 | ORB-short ORB[1083.50,1092.95] vol=2.6x ATR=2.42 |
| Stop hit — per-position SL triggered | 2023-08-29 11:00:00 | 1080.42 | 1084.27 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-08-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 10:40:00 | 1115.00 | 1108.51 | 0.00 | ORB-long ORB[1097.50,1107.00] vol=3.0x ATR=2.94 |
| Stop hit — per-position SL triggered | 2023-08-30 11:00:00 | 1112.06 | 1108.93 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 09:45:00 | 1125.00 | 1120.51 | 0.00 | ORB-long ORB[1113.70,1124.45] vol=1.8x ATR=2.97 |
| Stop hit — per-position SL triggered | 2023-08-31 10:25:00 | 1122.03 | 1123.14 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-09-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 10:40:00 | 1109.75 | 1119.75 | 0.00 | ORB-short ORB[1122.15,1137.50] vol=3.4x ATR=3.33 |
| Stop hit — per-position SL triggered | 2023-09-04 11:05:00 | 1113.08 | 1116.68 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 11:15:00 | 1181.80 | 1170.46 | 0.00 | ORB-long ORB[1160.20,1174.95] vol=2.2x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 11:50:00 | 1186.68 | 1173.06 | 0.00 | T1 1.5R @ 1186.68 |
| Target hit | 2023-09-14 15:20:00 | 1209.00 | 1191.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2023-09-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 10:45:00 | 1178.00 | 1182.84 | 0.00 | ORB-short ORB[1178.80,1191.05] vol=2.1x ATR=3.09 |
| Stop hit — per-position SL triggered | 2023-09-26 11:05:00 | 1181.09 | 1182.24 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-10-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 09:40:00 | 1200.55 | 1203.42 | 0.00 | ORB-short ORB[1205.50,1221.80] vol=1.5x ATR=5.41 |
| Stop hit — per-position SL triggered | 2023-10-05 10:00:00 | 1205.96 | 1202.93 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-10-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 10:20:00 | 1235.95 | 1230.69 | 0.00 | ORB-long ORB[1219.60,1228.70] vol=5.7x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 12:00:00 | 1240.73 | 1232.85 | 0.00 | T1 1.5R @ 1240.73 |
| Target hit | 2023-10-10 15:20:00 | 1259.70 | 1247.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2023-10-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 11:10:00 | 1252.60 | 1255.20 | 0.00 | ORB-short ORB[1255.00,1260.60] vol=3.6x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 12:15:00 | 1248.70 | 1254.57 | 0.00 | T1 1.5R @ 1248.70 |
| Target hit | 2023-10-12 15:20:00 | 1242.90 | 1249.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2023-10-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 11:05:00 | 1259.00 | 1251.47 | 0.00 | ORB-long ORB[1241.15,1258.00] vol=2.2x ATR=2.81 |
| Stop hit — per-position SL triggered | 2023-10-13 12:05:00 | 1256.19 | 1252.97 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 09:55:00 | 1221.30 | 1229.85 | 0.00 | ORB-short ORB[1232.40,1248.70] vol=1.8x ATR=3.70 |
| Stop hit — per-position SL triggered | 2023-10-18 10:10:00 | 1225.00 | 1225.10 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-10-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 09:55:00 | 1199.00 | 1202.30 | 0.00 | ORB-short ORB[1199.80,1216.75] vol=4.3x ATR=3.29 |
| Stop hit — per-position SL triggered | 2023-10-19 10:20:00 | 1202.29 | 1201.83 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-10-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 09:35:00 | 1200.50 | 1197.05 | 0.00 | ORB-long ORB[1190.05,1198.00] vol=2.8x ATR=3.76 |
| Stop hit — per-position SL triggered | 2023-10-20 09:55:00 | 1196.74 | 1196.75 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-10-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-25 10:30:00 | 1152.90 | 1160.70 | 0.00 | ORB-short ORB[1155.60,1172.85] vol=3.1x ATR=4.53 |
| Stop hit — per-position SL triggered | 2023-10-25 10:40:00 | 1157.43 | 1160.25 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-10-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-30 10:10:00 | 1149.15 | 1153.97 | 0.00 | ORB-short ORB[1155.55,1168.15] vol=2.2x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 10:15:00 | 1142.80 | 1152.33 | 0.00 | T1 1.5R @ 1142.80 |
| Stop hit — per-position SL triggered | 2023-10-30 10:20:00 | 1149.15 | 1152.34 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 09:45:00 | 1149.80 | 1156.89 | 0.00 | ORB-short ORB[1156.85,1165.55] vol=1.9x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 09:55:00 | 1144.05 | 1152.62 | 0.00 | T1 1.5R @ 1144.05 |
| Target hit | 2023-10-31 12:40:00 | 1142.00 | 1139.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — SELL (started 2023-11-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 10:50:00 | 1140.60 | 1146.29 | 0.00 | ORB-short ORB[1142.65,1156.00] vol=3.6x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 11:10:00 | 1134.95 | 1145.64 | 0.00 | T1 1.5R @ 1134.95 |
| Target hit | 2023-11-07 13:40:00 | 1134.75 | 1134.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — SELL (started 2023-11-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 10:10:00 | 1132.75 | 1140.87 | 0.00 | ORB-short ORB[1139.45,1149.50] vol=2.2x ATR=3.78 |
| Stop hit — per-position SL triggered | 2023-11-08 10:15:00 | 1136.53 | 1140.43 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-11-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 09:55:00 | 1166.35 | 1161.65 | 0.00 | ORB-long ORB[1158.30,1166.00] vol=2.3x ATR=3.68 |
| Stop hit — per-position SL triggered | 2023-11-15 10:00:00 | 1162.67 | 1161.56 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-11-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 10:10:00 | 1115.85 | 1124.92 | 0.00 | ORB-short ORB[1123.25,1132.90] vol=1.7x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 10:35:00 | 1111.37 | 1122.27 | 0.00 | T1 1.5R @ 1111.37 |
| Target hit | 2023-11-21 15:20:00 | 1102.65 | 1112.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2023-11-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:40:00 | 1119.90 | 1113.48 | 0.00 | ORB-long ORB[1101.75,1111.20] vol=1.6x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 09:55:00 | 1125.37 | 1116.46 | 0.00 | T1 1.5R @ 1125.37 |
| Target hit | 2023-11-22 11:00:00 | 1122.50 | 1122.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2023-11-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 10:50:00 | 1120.10 | 1123.16 | 0.00 | ORB-short ORB[1123.75,1137.00] vol=2.1x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 14:35:00 | 1116.09 | 1120.45 | 0.00 | T1 1.5R @ 1116.09 |
| Target hit | 2023-11-23 15:20:00 | 1115.25 | 1118.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2023-12-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 10:40:00 | 1150.75 | 1147.99 | 0.00 | ORB-long ORB[1139.00,1149.40] vol=7.5x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 11:15:00 | 1155.07 | 1148.50 | 0.00 | T1 1.5R @ 1155.07 |
| Stop hit — per-position SL triggered | 2023-12-06 11:20:00 | 1150.75 | 1148.60 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-12-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 11:10:00 | 1180.40 | 1171.84 | 0.00 | ORB-long ORB[1166.00,1173.45] vol=2.6x ATR=3.72 |
| Stop hit — per-position SL triggered | 2023-12-11 11:30:00 | 1176.68 | 1174.73 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 10:30:00 | 1183.30 | 1179.70 | 0.00 | ORB-long ORB[1172.20,1180.00] vol=2.2x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 10:35:00 | 1187.31 | 1180.47 | 0.00 | T1 1.5R @ 1187.31 |
| Stop hit — per-position SL triggered | 2023-12-12 11:10:00 | 1183.30 | 1182.55 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 09:55:00 | 1243.80 | 1236.38 | 0.00 | ORB-long ORB[1230.00,1237.10] vol=2.0x ATR=4.21 |
| Stop hit — per-position SL triggered | 2023-12-15 10:20:00 | 1239.59 | 1237.53 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-12-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:05:00 | 1261.90 | 1246.55 | 0.00 | ORB-long ORB[1225.45,1236.30] vol=2.2x ATR=4.94 |
| Stop hit — per-position SL triggered | 2023-12-22 10:10:00 | 1256.96 | 1248.67 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 10:15:00 | 1260.00 | 1255.09 | 0.00 | ORB-long ORB[1246.30,1254.60] vol=1.9x ATR=3.59 |
| Stop hit — per-position SL triggered | 2023-12-27 10:20:00 | 1256.41 | 1255.32 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 10:55:00 | 1237.50 | 1245.26 | 0.00 | ORB-short ORB[1253.40,1260.00] vol=1.9x ATR=2.80 |
| Stop hit — per-position SL triggered | 2024-01-01 11:05:00 | 1240.30 | 1244.73 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 1217.10 | 1225.02 | 0.00 | ORB-short ORB[1222.35,1231.30] vol=1.6x ATR=3.71 |
| Stop hit — per-position SL triggered | 2024-01-02 10:00:00 | 1220.81 | 1224.79 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-01-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 10:25:00 | 1262.95 | 1256.27 | 0.00 | ORB-long ORB[1243.75,1259.50] vol=2.8x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-01-04 10:35:00 | 1259.02 | 1256.71 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-01-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 10:30:00 | 1229.00 | 1229.71 | 0.00 | ORB-short ORB[1233.05,1244.60] vol=9.4x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 10:55:00 | 1223.32 | 1229.24 | 0.00 | T1 1.5R @ 1223.32 |
| Stop hit — per-position SL triggered | 2024-01-09 11:10:00 | 1229.00 | 1229.15 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-01-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 10:55:00 | 1296.95 | 1289.11 | 0.00 | ORB-long ORB[1276.05,1292.75] vol=2.7x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 13:10:00 | 1303.33 | 1293.57 | 0.00 | T1 1.5R @ 1303.33 |
| Stop hit — per-position SL triggered | 2024-01-16 15:15:00 | 1296.95 | 1296.75 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 10:15:00 | 1156.65 | 1167.24 | 0.00 | ORB-short ORB[1176.05,1193.15] vol=4.2x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 10:55:00 | 1149.76 | 1161.69 | 0.00 | T1 1.5R @ 1149.76 |
| Stop hit — per-position SL triggered | 2024-02-01 12:10:00 | 1156.65 | 1154.36 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-02-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 11:10:00 | 1167.50 | 1164.30 | 0.00 | ORB-long ORB[1143.75,1160.00] vol=2.1x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:25:00 | 1174.19 | 1164.83 | 0.00 | T1 1.5R @ 1174.19 |
| Stop hit — per-position SL triggered | 2024-02-02 12:25:00 | 1167.50 | 1166.26 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 09:30:00 | 1117.00 | 1110.11 | 0.00 | ORB-long ORB[1103.00,1113.35] vol=2.8x ATR=4.26 |
| Stop hit — per-position SL triggered | 2024-02-13 09:45:00 | 1112.74 | 1114.52 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-19 09:30:00 | 1143.80 | 1147.87 | 0.00 | ORB-short ORB[1144.20,1155.55] vol=1.7x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-19 09:40:00 | 1138.16 | 1145.68 | 0.00 | T1 1.5R @ 1138.16 |
| Stop hit — per-position SL triggered | 2024-02-19 10:15:00 | 1143.80 | 1143.84 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-02-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 09:30:00 | 1071.75 | 1079.53 | 0.00 | ORB-short ORB[1075.95,1090.55] vol=1.9x ATR=4.01 |
| Stop hit — per-position SL triggered | 2024-02-22 09:35:00 | 1075.76 | 1078.85 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 09:45:00 | 1108.30 | 1103.69 | 0.00 | ORB-long ORB[1093.00,1105.15] vol=3.0x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-02-26 09:50:00 | 1104.89 | 1104.26 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-03-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 11:05:00 | 1082.55 | 1084.38 | 0.00 | ORB-short ORB[1087.85,1096.40] vol=3.8x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-03-01 11:20:00 | 1084.76 | 1084.38 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-03-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 11:00:00 | 1083.65 | 1089.45 | 0.00 | ORB-short ORB[1084.90,1094.95] vol=2.8x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 12:35:00 | 1079.19 | 1087.38 | 0.00 | T1 1.5R @ 1079.19 |
| Target hit | 2024-03-04 15:20:00 | 1072.65 | 1080.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — BUY (started 2024-04-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 10:45:00 | 1220.25 | 1210.32 | 0.00 | ORB-long ORB[1198.30,1215.70] vol=1.9x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-04-09 10:55:00 | 1217.08 | 1210.87 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-04-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 10:35:00 | 1205.25 | 1211.28 | 0.00 | ORB-short ORB[1211.70,1222.20] vol=1.6x ATR=2.69 |
| Stop hit — per-position SL triggered | 2024-04-10 10:40:00 | 1207.94 | 1210.98 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:00:00 | 1164.95 | 1156.43 | 0.00 | ORB-long ORB[1144.70,1152.00] vol=2.1x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 10:20:00 | 1170.57 | 1161.25 | 0.00 | T1 1.5R @ 1170.57 |
| Target hit | 2024-04-24 13:40:00 | 1167.70 | 1169.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — SELL (started 2024-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 10:50:00 | 1162.85 | 1167.28 | 0.00 | ORB-short ORB[1166.05,1174.20] vol=1.7x ATR=3.67 |
| Stop hit — per-position SL triggered | 2024-04-29 11:00:00 | 1166.52 | 1167.40 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-04-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 10:45:00 | 1188.55 | 1179.46 | 0.00 | ORB-long ORB[1173.15,1187.35] vol=1.7x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 10:50:00 | 1194.32 | 1181.42 | 0.00 | T1 1.5R @ 1194.32 |
| Stop hit — per-position SL triggered | 2024-04-30 10:55:00 | 1188.55 | 1182.27 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-05-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:45:00 | 1293.15 | 1309.24 | 0.00 | ORB-short ORB[1319.00,1331.05] vol=2.4x ATR=5.43 |
| Stop hit — per-position SL triggered | 2024-05-07 10:55:00 | 1298.58 | 1307.56 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-08 11:05:00 | 1278.40 | 1289.64 | 0.00 | ORB-short ORB[1287.00,1304.35] vol=5.6x ATR=5.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 11:30:00 | 1270.62 | 1288.36 | 0.00 | T1 1.5R @ 1270.62 |
| Target hit | 2024-05-08 15:20:00 | 1263.30 | 1271.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2024-05-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 09:50:00 | 1255.10 | 1241.45 | 0.00 | ORB-long ORB[1224.65,1241.45] vol=2.5x ATR=6.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:55:00 | 1264.59 | 1248.83 | 0.00 | T1 1.5R @ 1264.59 |
| Target hit | 2024-05-10 10:50:00 | 1262.95 | 1265.99 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 10:25:00 | 1015.10 | 2023-05-12 11:20:00 | 1009.67 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2023-05-16 10:05:00 | 1020.45 | 2023-05-16 12:30:00 | 1016.98 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-05-22 10:55:00 | 1048.60 | 2023-05-22 11:00:00 | 1046.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-23 09:30:00 | 1061.05 | 2023-05-23 09:40:00 | 1058.56 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-05-25 09:35:00 | 1053.85 | 2023-05-25 09:45:00 | 1057.46 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-05-25 09:35:00 | 1053.85 | 2023-05-25 09:55:00 | 1053.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-26 09:35:00 | 1051.70 | 2023-05-26 10:25:00 | 1047.78 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-05-26 09:35:00 | 1051.70 | 2023-05-26 15:00:00 | 1049.00 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2023-05-30 09:45:00 | 1048.65 | 2023-05-30 09:50:00 | 1045.34 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-05-30 09:45:00 | 1048.65 | 2023-05-30 14:25:00 | 1043.75 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2023-06-02 10:35:00 | 1039.75 | 2023-06-02 10:45:00 | 1041.88 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-07 10:20:00 | 1070.90 | 2023-06-07 10:30:00 | 1075.06 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-06-07 10:20:00 | 1070.90 | 2023-06-07 10:55:00 | 1070.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-13 09:35:00 | 1088.75 | 2023-06-13 09:55:00 | 1085.30 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-06-15 09:40:00 | 1114.00 | 2023-06-15 09:55:00 | 1110.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-06-19 09:40:00 | 1143.80 | 2023-06-19 09:50:00 | 1148.20 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-06-19 09:40:00 | 1143.80 | 2023-06-19 10:15:00 | 1143.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-22 09:45:00 | 1112.35 | 2023-06-22 10:05:00 | 1116.15 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-07-04 10:15:00 | 1179.90 | 2023-07-04 10:25:00 | 1186.09 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-07-04 10:15:00 | 1179.90 | 2023-07-04 13:45:00 | 1190.50 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2023-07-07 09:40:00 | 1185.50 | 2023-07-07 09:45:00 | 1180.54 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-07-19 10:40:00 | 1157.20 | 2023-07-19 11:10:00 | 1162.61 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-07-19 10:40:00 | 1157.20 | 2023-07-19 15:20:00 | 1169.05 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2023-07-21 10:50:00 | 1139.85 | 2023-07-21 11:15:00 | 1133.94 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-07-21 10:50:00 | 1139.85 | 2023-07-21 11:20:00 | 1139.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-25 11:10:00 | 1129.10 | 2023-07-25 11:15:00 | 1132.63 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-07-26 10:15:00 | 1148.00 | 2023-07-26 10:20:00 | 1144.51 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-07-28 10:40:00 | 1148.65 | 2023-07-28 11:05:00 | 1145.41 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-08-07 10:25:00 | 1071.20 | 2023-08-07 10:30:00 | 1074.27 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-08-08 10:50:00 | 1064.25 | 2023-08-08 10:55:00 | 1067.28 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-09 10:40:00 | 1060.05 | 2023-08-09 10:50:00 | 1057.14 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-10 10:20:00 | 1044.65 | 2023-08-10 10:45:00 | 1039.50 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-08-10 10:20:00 | 1044.65 | 2023-08-10 10:55:00 | 1044.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-17 10:40:00 | 1035.00 | 2023-08-17 10:50:00 | 1032.19 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-18 09:35:00 | 1006.35 | 2023-08-18 09:45:00 | 1009.52 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-08-22 10:15:00 | 1051.00 | 2023-08-22 10:30:00 | 1048.21 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-25 10:35:00 | 1064.95 | 2023-08-25 10:45:00 | 1068.34 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-08-29 10:45:00 | 1078.00 | 2023-08-29 11:00:00 | 1080.42 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-08-30 10:40:00 | 1115.00 | 2023-08-30 11:00:00 | 1112.06 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-08-31 09:45:00 | 1125.00 | 2023-08-31 10:25:00 | 1122.03 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-04 10:40:00 | 1109.75 | 2023-09-04 11:05:00 | 1113.08 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-09-14 11:15:00 | 1181.80 | 2023-09-14 11:50:00 | 1186.68 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-09-14 11:15:00 | 1181.80 | 2023-09-14 15:20:00 | 1209.00 | TARGET_HIT | 0.50 | 2.30% |
| SELL | retest1 | 2023-09-26 10:45:00 | 1178.00 | 2023-09-26 11:05:00 | 1181.09 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-10-05 09:40:00 | 1200.55 | 2023-10-05 10:00:00 | 1205.96 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-10-10 10:20:00 | 1235.95 | 2023-10-10 12:00:00 | 1240.73 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-10-10 10:20:00 | 1235.95 | 2023-10-10 15:20:00 | 1259.70 | TARGET_HIT | 0.50 | 1.92% |
| SELL | retest1 | 2023-10-12 11:10:00 | 1252.60 | 2023-10-12 12:15:00 | 1248.70 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-10-12 11:10:00 | 1252.60 | 2023-10-12 15:20:00 | 1242.90 | TARGET_HIT | 0.50 | 0.77% |
| BUY | retest1 | 2023-10-13 11:05:00 | 1259.00 | 2023-10-13 12:05:00 | 1256.19 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-10-18 09:55:00 | 1221.30 | 2023-10-18 10:10:00 | 1225.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-10-19 09:55:00 | 1199.00 | 2023-10-19 10:20:00 | 1202.29 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-10-20 09:35:00 | 1200.50 | 2023-10-20 09:55:00 | 1196.74 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-10-25 10:30:00 | 1152.90 | 2023-10-25 10:40:00 | 1157.43 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-10-30 10:10:00 | 1149.15 | 2023-10-30 10:15:00 | 1142.80 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-10-30 10:10:00 | 1149.15 | 2023-10-30 10:20:00 | 1149.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-31 09:45:00 | 1149.80 | 2023-10-31 09:55:00 | 1144.05 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-10-31 09:45:00 | 1149.80 | 2023-10-31 12:40:00 | 1142.00 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2023-11-07 10:50:00 | 1140.60 | 2023-11-07 11:10:00 | 1134.95 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-11-07 10:50:00 | 1140.60 | 2023-11-07 13:40:00 | 1134.75 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2023-11-08 10:10:00 | 1132.75 | 2023-11-08 10:15:00 | 1136.53 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-11-15 09:55:00 | 1166.35 | 2023-11-15 10:00:00 | 1162.67 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-11-21 10:10:00 | 1115.85 | 2023-11-21 10:35:00 | 1111.37 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-11-21 10:10:00 | 1115.85 | 2023-11-21 15:20:00 | 1102.65 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2023-11-22 09:40:00 | 1119.90 | 2023-11-22 09:55:00 | 1125.37 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-11-22 09:40:00 | 1119.90 | 2023-11-22 11:00:00 | 1122.50 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2023-11-23 10:50:00 | 1120.10 | 2023-11-23 14:35:00 | 1116.09 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-11-23 10:50:00 | 1120.10 | 2023-11-23 15:20:00 | 1115.25 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2023-12-06 10:40:00 | 1150.75 | 2023-12-06 11:15:00 | 1155.07 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-12-06 10:40:00 | 1150.75 | 2023-12-06 11:20:00 | 1150.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-11 11:10:00 | 1180.40 | 2023-12-11 11:30:00 | 1176.68 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-12-12 10:30:00 | 1183.30 | 2023-12-12 10:35:00 | 1187.31 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-12-12 10:30:00 | 1183.30 | 2023-12-12 11:10:00 | 1183.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-15 09:55:00 | 1243.80 | 2023-12-15 10:20:00 | 1239.59 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-12-22 10:05:00 | 1261.90 | 2023-12-22 10:10:00 | 1256.96 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-12-27 10:15:00 | 1260.00 | 2023-12-27 10:20:00 | 1256.41 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-01 10:55:00 | 1237.50 | 2024-01-01 11:05:00 | 1240.30 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-01-02 09:55:00 | 1217.10 | 2024-01-02 10:00:00 | 1220.81 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-01-04 10:25:00 | 1262.95 | 2024-01-04 10:35:00 | 1259.02 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-01-09 10:30:00 | 1229.00 | 2024-01-09 10:55:00 | 1223.32 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-01-09 10:30:00 | 1229.00 | 2024-01-09 11:10:00 | 1229.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-16 10:55:00 | 1296.95 | 2024-01-16 13:10:00 | 1303.33 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-01-16 10:55:00 | 1296.95 | 2024-01-16 15:15:00 | 1296.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-01 10:15:00 | 1156.65 | 2024-02-01 10:55:00 | 1149.76 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-02-01 10:15:00 | 1156.65 | 2024-02-01 12:10:00 | 1156.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 11:10:00 | 1167.50 | 2024-02-02 11:25:00 | 1174.19 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-02-02 11:10:00 | 1167.50 | 2024-02-02 12:25:00 | 1167.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-13 09:30:00 | 1117.00 | 2024-02-13 09:45:00 | 1112.74 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-02-19 09:30:00 | 1143.80 | 2024-02-19 09:40:00 | 1138.16 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-02-19 09:30:00 | 1143.80 | 2024-02-19 10:15:00 | 1143.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-22 09:30:00 | 1071.75 | 2024-02-22 09:35:00 | 1075.76 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-02-26 09:45:00 | 1108.30 | 2024-02-26 09:50:00 | 1104.89 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-03-01 11:05:00 | 1082.55 | 2024-03-01 11:20:00 | 1084.76 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-03-04 11:00:00 | 1083.65 | 2024-03-04 12:35:00 | 1079.19 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-03-04 11:00:00 | 1083.65 | 2024-03-04 15:20:00 | 1072.65 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2024-04-09 10:45:00 | 1220.25 | 2024-04-09 10:55:00 | 1217.08 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-10 10:35:00 | 1205.25 | 2024-04-10 10:40:00 | 1207.94 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-04-24 10:00:00 | 1164.95 | 2024-04-24 10:20:00 | 1170.57 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-04-24 10:00:00 | 1164.95 | 2024-04-24 13:40:00 | 1167.70 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2024-04-29 10:50:00 | 1162.85 | 2024-04-29 11:00:00 | 1166.52 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-04-30 10:45:00 | 1188.55 | 2024-04-30 10:50:00 | 1194.32 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-04-30 10:45:00 | 1188.55 | 2024-04-30 10:55:00 | 1188.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-07 10:45:00 | 1293.15 | 2024-05-07 10:55:00 | 1298.58 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-05-08 11:05:00 | 1278.40 | 2024-05-08 11:30:00 | 1270.62 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-05-08 11:05:00 | 1278.40 | 2024-05-08 15:20:00 | 1263.30 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2024-05-10 09:50:00 | 1255.10 | 2024-05-10 09:55:00 | 1264.59 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-05-10 09:50:00 | 1255.10 | 2024-05-10 10:50:00 | 1262.95 | TARGET_HIT | 0.50 | 0.63% |
