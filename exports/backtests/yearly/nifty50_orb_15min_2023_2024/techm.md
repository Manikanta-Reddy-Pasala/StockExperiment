# TECHM (TECHM)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-01-30 15:25:00 (32017 bars)
- **Last close:** 1681.00
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
| ENTRY1 | 104 |
| ENTRY2 | 0 |
| PARTIAL | 46 |
| TARGET_HIT | 14 |
| STOP_HIT | 90 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 150 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 90
- **Target hits / Stop hits / Partials:** 14 / 90 / 46
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 15.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 28 | 40.6% | 6 | 41 | 22 | 0.15% | 10.2% |
| BUY @ 2nd Alert (retest1) | 69 | 28 | 40.6% | 6 | 41 | 22 | 0.15% | 10.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 81 | 32 | 39.5% | 8 | 49 | 24 | 0.06% | 4.9% |
| SELL @ 2nd Alert (retest1) | 81 | 32 | 39.5% | 8 | 49 | 24 | 0.06% | 4.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 150 | 60 | 40.0% | 14 | 90 | 46 | 0.10% | 15.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 10:55:00 | 1046.50 | 1040.95 | 0.00 | ORB-long ORB[1034.05,1046.45] vol=3.7x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-12 12:20:00 | 1052.64 | 1043.77 | 0.00 | T1 1.5R @ 1052.64 |
| Stop hit — per-position SL triggered | 2023-05-12 15:00:00 | 1046.50 | 1045.77 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:40:00 | 1058.00 | 1054.18 | 0.00 | ORB-long ORB[1042.00,1056.85] vol=2.3x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 10:15:00 | 1061.66 | 1057.16 | 0.00 | T1 1.5R @ 1061.66 |
| Target hit | 2023-05-15 15:10:00 | 1063.25 | 1063.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2023-05-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-19 10:25:00 | 1071.00 | 1062.57 | 0.00 | ORB-long ORB[1049.00,1059.75] vol=3.0x ATR=2.94 |
| Stop hit — per-position SL triggered | 2023-05-19 11:00:00 | 1068.06 | 1065.92 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:45:00 | 1101.50 | 1095.25 | 0.00 | ORB-long ORB[1085.00,1094.10] vol=1.6x ATR=3.01 |
| Stop hit — per-position SL triggered | 2023-05-24 09:55:00 | 1098.49 | 1095.84 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 09:45:00 | 1115.15 | 1108.95 | 0.00 | ORB-long ORB[1099.00,1112.00] vol=1.8x ATR=3.24 |
| Stop hit — per-position SL triggered | 2023-05-26 09:50:00 | 1111.91 | 1109.25 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 11:15:00 | 1112.55 | 1116.89 | 0.00 | ORB-short ORB[1116.00,1128.05] vol=1.7x ATR=2.12 |
| Stop hit — per-position SL triggered | 2023-05-29 11:30:00 | 1114.67 | 1116.54 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 09:30:00 | 1110.60 | 1115.78 | 0.00 | ORB-short ORB[1111.60,1120.80] vol=2.5x ATR=2.85 |
| Stop hit — per-position SL triggered | 2023-05-30 09:55:00 | 1113.45 | 1114.47 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-05-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:00:00 | 1116.20 | 1110.72 | 0.00 | ORB-long ORB[1096.00,1111.45] vol=2.1x ATR=2.66 |
| Stop hit — per-position SL triggered | 2023-05-31 11:00:00 | 1113.54 | 1112.75 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 10:05:00 | 1084.35 | 1089.95 | 0.00 | ORB-short ORB[1086.60,1102.00] vol=1.5x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 10:20:00 | 1080.08 | 1088.31 | 0.00 | T1 1.5R @ 1080.08 |
| Target hit | 2023-06-06 13:10:00 | 1082.30 | 1082.30 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — SELL (started 2023-06-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 10:10:00 | 1061.45 | 1066.22 | 0.00 | ORB-short ORB[1064.00,1079.00] vol=1.8x ATR=3.01 |
| Stop hit — per-position SL triggered | 2023-06-09 10:55:00 | 1064.46 | 1065.66 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 11:15:00 | 1075.25 | 1071.51 | 0.00 | ORB-long ORB[1062.15,1073.45] vol=1.5x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 12:00:00 | 1078.05 | 1072.75 | 0.00 | T1 1.5R @ 1078.05 |
| Stop hit — per-position SL triggered | 2023-06-12 12:25:00 | 1075.25 | 1073.30 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 11:10:00 | 1084.55 | 1082.74 | 0.00 | ORB-long ORB[1075.40,1081.00] vol=1.8x ATR=1.54 |
| Stop hit — per-position SL triggered | 2023-06-13 11:20:00 | 1083.01 | 1082.82 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 10:50:00 | 1075.75 | 1081.05 | 0.00 | ORB-short ORB[1079.20,1086.95] vol=1.8x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-16 11:25:00 | 1072.72 | 1079.61 | 0.00 | T1 1.5R @ 1072.72 |
| Stop hit — per-position SL triggered | 2023-06-16 12:55:00 | 1075.75 | 1078.31 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:45:00 | 1100.85 | 1094.97 | 0.00 | ORB-long ORB[1088.20,1097.00] vol=3.9x ATR=2.94 |
| Stop hit — per-position SL triggered | 2023-06-20 10:25:00 | 1097.91 | 1097.85 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 10:45:00 | 1106.45 | 1113.30 | 0.00 | ORB-short ORB[1107.35,1117.90] vol=2.2x ATR=2.55 |
| Stop hit — per-position SL triggered | 2023-06-21 11:20:00 | 1109.00 | 1111.76 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 11:15:00 | 1119.40 | 1124.89 | 0.00 | ORB-short ORB[1125.00,1136.90] vol=1.8x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 11:20:00 | 1116.48 | 1124.61 | 0.00 | T1 1.5R @ 1116.48 |
| Stop hit — per-position SL triggered | 2023-07-03 11:35:00 | 1119.40 | 1124.31 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 09:30:00 | 1158.10 | 1161.61 | 0.00 | ORB-short ORB[1158.30,1168.00] vol=1.5x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 09:45:00 | 1153.80 | 1160.28 | 0.00 | T1 1.5R @ 1153.80 |
| Stop hit — per-position SL triggered | 2023-07-12 09:55:00 | 1158.10 | 1160.05 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 10:45:00 | 1197.75 | 1191.03 | 0.00 | ORB-long ORB[1183.00,1194.90] vol=1.5x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 10:55:00 | 1202.92 | 1193.28 | 0.00 | T1 1.5R @ 1202.92 |
| Target hit | 2023-07-14 15:20:00 | 1229.75 | 1214.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2023-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:30:00 | 1226.80 | 1236.05 | 0.00 | ORB-short ORB[1230.95,1248.95] vol=1.6x ATR=4.13 |
| Stop hit — per-position SL triggered | 2023-07-18 09:45:00 | 1230.93 | 1234.23 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 09:40:00 | 1098.00 | 1094.25 | 0.00 | ORB-long ORB[1087.15,1097.00] vol=3.6x ATR=3.10 |
| Stop hit — per-position SL triggered | 2023-07-31 09:45:00 | 1094.90 | 1094.51 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 09:40:00 | 1143.60 | 1139.30 | 0.00 | ORB-long ORB[1132.65,1143.00] vol=1.5x ATR=3.68 |
| Stop hit — per-position SL triggered | 2023-08-02 09:45:00 | 1139.92 | 1139.39 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 09:35:00 | 1160.65 | 1156.68 | 0.00 | ORB-long ORB[1142.80,1158.00] vol=1.8x ATR=4.13 |
| Stop hit — per-position SL triggered | 2023-08-04 09:40:00 | 1156.52 | 1156.98 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 09:45:00 | 1191.25 | 1184.44 | 0.00 | ORB-long ORB[1175.10,1189.00] vol=1.9x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 09:50:00 | 1196.57 | 1186.67 | 0.00 | T1 1.5R @ 1196.57 |
| Stop hit — per-position SL triggered | 2023-08-08 11:20:00 | 1191.25 | 1195.02 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-08-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 10:45:00 | 1220.00 | 1213.52 | 0.00 | ORB-long ORB[1200.20,1215.00] vol=1.9x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-09 11:00:00 | 1224.62 | 1214.94 | 0.00 | T1 1.5R @ 1224.62 |
| Stop hit — per-position SL triggered | 2023-08-09 11:10:00 | 1220.00 | 1215.45 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 11:10:00 | 1231.25 | 1225.18 | 0.00 | ORB-long ORB[1215.60,1227.40] vol=1.7x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 12:40:00 | 1236.10 | 1228.13 | 0.00 | T1 1.5R @ 1236.10 |
| Stop hit — per-position SL triggered | 2023-08-10 13:50:00 | 1231.25 | 1231.41 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 09:35:00 | 1236.50 | 1232.20 | 0.00 | ORB-long ORB[1220.00,1235.80] vol=1.5x ATR=3.26 |
| Stop hit — per-position SL triggered | 2023-08-16 09:40:00 | 1233.24 | 1232.70 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 09:55:00 | 1219.40 | 1224.18 | 0.00 | ORB-short ORB[1221.25,1232.00] vol=1.8x ATR=3.09 |
| Stop hit — per-position SL triggered | 2023-08-17 10:30:00 | 1222.49 | 1223.06 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 10:50:00 | 1210.00 | 1215.96 | 0.00 | ORB-short ORB[1219.25,1225.95] vol=1.5x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 11:25:00 | 1206.08 | 1214.54 | 0.00 | T1 1.5R @ 1206.08 |
| Stop hit — per-position SL triggered | 2023-08-18 12:05:00 | 1210.00 | 1213.86 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 09:50:00 | 1209.70 | 1214.15 | 0.00 | ORB-short ORB[1212.55,1223.00] vol=1.5x ATR=2.36 |
| Stop hit — per-position SL triggered | 2023-08-22 10:05:00 | 1212.06 | 1213.40 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 11:15:00 | 1201.00 | 1209.35 | 0.00 | ORB-short ORB[1206.15,1216.50] vol=2.1x ATR=2.52 |
| Stop hit — per-position SL triggered | 2023-08-24 11:20:00 | 1203.52 | 1208.83 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:55:00 | 1180.35 | 1187.79 | 0.00 | ORB-short ORB[1185.00,1197.20] vol=1.7x ATR=3.34 |
| Stop hit — per-position SL triggered | 2023-08-25 11:15:00 | 1183.69 | 1186.98 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 09:35:00 | 1198.95 | 1194.47 | 0.00 | ORB-long ORB[1188.25,1195.00] vol=2.1x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 09:50:00 | 1202.81 | 1197.01 | 0.00 | T1 1.5R @ 1202.81 |
| Stop hit — per-position SL triggered | 2023-08-29 09:55:00 | 1198.95 | 1197.21 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 11:05:00 | 1212.75 | 1211.91 | 0.00 | ORB-long ORB[1204.05,1211.95] vol=3.6x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 12:50:00 | 1216.85 | 1213.10 | 0.00 | T1 1.5R @ 1216.85 |
| Target hit | 2023-09-01 15:20:00 | 1227.95 | 1219.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2023-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 09:30:00 | 1231.15 | 1236.15 | 0.00 | ORB-short ORB[1233.05,1240.00] vol=1.8x ATR=3.21 |
| Stop hit — per-position SL triggered | 2023-09-04 09:40:00 | 1234.36 | 1235.36 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 10:45:00 | 1261.65 | 1249.59 | 0.00 | ORB-long ORB[1238.65,1252.95] vol=3.0x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 10:55:00 | 1266.47 | 1253.87 | 0.00 | T1 1.5R @ 1266.47 |
| Stop hit — per-position SL triggered | 2023-09-07 11:30:00 | 1261.65 | 1258.76 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 09:50:00 | 1265.25 | 1269.78 | 0.00 | ORB-short ORB[1265.65,1278.00] vol=1.9x ATR=3.69 |
| Stop hit — per-position SL triggered | 2023-09-08 09:55:00 | 1268.94 | 1269.82 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:30:00 | 1281.80 | 1276.26 | 0.00 | ORB-long ORB[1262.00,1279.95] vol=2.8x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 09:40:00 | 1286.56 | 1280.50 | 0.00 | T1 1.5R @ 1286.56 |
| Stop hit — per-position SL triggered | 2023-09-14 10:05:00 | 1281.80 | 1282.72 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-22 10:05:00 | 1303.55 | 1294.38 | 0.00 | ORB-long ORB[1284.00,1295.95] vol=1.8x ATR=4.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 10:25:00 | 1310.35 | 1297.72 | 0.00 | T1 1.5R @ 1310.35 |
| Stop hit — per-position SL triggered | 2023-09-22 13:50:00 | 1303.55 | 1308.03 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 09:45:00 | 1279.65 | 1274.69 | 0.00 | ORB-long ORB[1264.00,1278.00] vol=1.8x ATR=3.24 |
| Stop hit — per-position SL triggered | 2023-09-27 09:55:00 | 1276.41 | 1275.04 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-09-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 09:35:00 | 1256.00 | 1265.32 | 0.00 | ORB-short ORB[1262.60,1274.05] vol=3.3x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 09:50:00 | 1249.64 | 1260.56 | 0.00 | T1 1.5R @ 1249.64 |
| Target hit | 2023-09-28 15:20:00 | 1235.00 | 1240.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2023-10-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 09:40:00 | 1220.00 | 1214.36 | 0.00 | ORB-long ORB[1207.20,1217.15] vol=1.7x ATR=3.29 |
| Stop hit — per-position SL triggered | 2023-10-05 10:25:00 | 1216.71 | 1217.90 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-10-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 11:05:00 | 1216.85 | 1211.25 | 0.00 | ORB-long ORB[1205.00,1215.35] vol=1.9x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 11:40:00 | 1220.85 | 1213.26 | 0.00 | T1 1.5R @ 1220.85 |
| Stop hit — per-position SL triggered | 2023-10-06 13:30:00 | 1216.85 | 1215.69 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 09:30:00 | 1221.25 | 1215.16 | 0.00 | ORB-long ORB[1202.15,1217.55] vol=3.7x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:35:00 | 1226.63 | 1217.32 | 0.00 | T1 1.5R @ 1226.63 |
| Stop hit — per-position SL triggered | 2023-10-09 09:40:00 | 1221.25 | 1217.81 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-10-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 09:45:00 | 1219.85 | 1224.22 | 0.00 | ORB-short ORB[1221.50,1232.00] vol=1.6x ATR=3.01 |
| Stop hit — per-position SL triggered | 2023-10-12 10:10:00 | 1222.86 | 1223.02 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 09:30:00 | 1198.10 | 1190.05 | 0.00 | ORB-long ORB[1180.00,1196.55] vol=2.2x ATR=5.20 |
| Stop hit — per-position SL triggered | 2023-10-13 09:35:00 | 1192.90 | 1190.29 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 09:30:00 | 1188.40 | 1192.61 | 0.00 | ORB-short ORB[1189.00,1198.25] vol=1.8x ATR=2.78 |
| Stop hit — per-position SL triggered | 2023-10-16 09:45:00 | 1191.18 | 1191.78 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 10:40:00 | 1195.50 | 1194.64 | 0.00 | ORB-long ORB[1188.45,1194.00] vol=1.7x ATR=2.92 |
| Stop hit — per-position SL triggered | 2023-10-17 11:00:00 | 1192.58 | 1194.64 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-10-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 09:45:00 | 1181.00 | 1175.46 | 0.00 | ORB-long ORB[1170.55,1178.90] vol=2.5x ATR=2.92 |
| Stop hit — per-position SL triggered | 2023-10-19 09:50:00 | 1178.08 | 1175.88 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-10-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 10:00:00 | 1162.15 | 1165.06 | 0.00 | ORB-short ORB[1163.15,1171.60] vol=2.0x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 10:15:00 | 1158.17 | 1164.43 | 0.00 | T1 1.5R @ 1158.17 |
| Stop hit — per-position SL triggered | 2023-10-20 11:05:00 | 1162.15 | 1161.26 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-10-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 10:35:00 | 1119.85 | 1124.11 | 0.00 | ORB-short ORB[1123.10,1134.00] vol=2.0x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 11:00:00 | 1116.12 | 1123.28 | 0.00 | T1 1.5R @ 1116.12 |
| Stop hit — per-position SL triggered | 2023-10-31 11:20:00 | 1119.85 | 1122.40 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-02 09:55:00 | 1121.45 | 1125.65 | 0.00 | ORB-short ORB[1122.55,1130.00] vol=2.2x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 10:10:00 | 1116.96 | 1124.40 | 0.00 | T1 1.5R @ 1116.96 |
| Stop hit — per-position SL triggered | 2023-11-02 11:10:00 | 1121.45 | 1122.94 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 11:00:00 | 1144.95 | 1141.33 | 0.00 | ORB-long ORB[1138.00,1142.85] vol=1.7x ATR=1.86 |
| Stop hit — per-position SL triggered | 2023-11-07 11:20:00 | 1143.09 | 1141.83 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-11-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:05:00 | 1127.00 | 1127.51 | 0.00 | ORB-short ORB[1127.50,1137.95] vol=12.0x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 10:10:00 | 1123.54 | 1127.42 | 0.00 | T1 1.5R @ 1123.54 |
| Target hit | 2023-11-09 13:45:00 | 1124.15 | 1123.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 54 — BUY (started 2023-11-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 10:55:00 | 1130.65 | 1128.38 | 0.00 | ORB-long ORB[1122.00,1127.65] vol=1.7x ATR=2.18 |
| Stop hit — per-position SL triggered | 2023-11-10 11:05:00 | 1128.47 | 1128.42 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:35:00 | 1132.75 | 1136.02 | 0.00 | ORB-short ORB[1134.65,1142.15] vol=1.9x ATR=2.14 |
| Stop hit — per-position SL triggered | 2023-11-13 11:00:00 | 1134.89 | 1135.45 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-11-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:00:00 | 1180.55 | 1173.60 | 0.00 | ORB-long ORB[1167.40,1176.80] vol=3.6x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 11:20:00 | 1184.02 | 1175.62 | 0.00 | T1 1.5R @ 1184.02 |
| Target hit | 2023-11-16 15:20:00 | 1204.85 | 1198.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2023-11-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-17 11:10:00 | 1203.30 | 1207.97 | 0.00 | ORB-short ORB[1206.10,1213.95] vol=1.8x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 12:20:00 | 1199.23 | 1206.51 | 0.00 | T1 1.5R @ 1199.23 |
| Stop hit — per-position SL triggered | 2023-11-17 12:30:00 | 1203.30 | 1206.17 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 09:40:00 | 1218.00 | 1209.52 | 0.00 | ORB-long ORB[1202.75,1210.95] vol=1.5x ATR=3.71 |
| Stop hit — per-position SL triggered | 2023-11-20 10:00:00 | 1214.29 | 1212.72 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 09:35:00 | 1213.00 | 1215.75 | 0.00 | ORB-short ORB[1213.75,1221.15] vol=1.6x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 09:55:00 | 1209.41 | 1214.07 | 0.00 | T1 1.5R @ 1209.41 |
| Target hit | 2023-11-21 12:10:00 | 1209.10 | 1208.70 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — SELL (started 2023-11-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 11:00:00 | 1213.35 | 1213.86 | 0.00 | ORB-short ORB[1214.70,1220.90] vol=1.6x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 12:15:00 | 1209.65 | 1213.57 | 0.00 | T1 1.5R @ 1209.65 |
| Stop hit — per-position SL triggered | 2023-11-23 13:00:00 | 1213.35 | 1213.28 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-11-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:35:00 | 1205.05 | 1209.22 | 0.00 | ORB-short ORB[1208.00,1214.95] vol=3.0x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 09:55:00 | 1201.44 | 1206.89 | 0.00 | T1 1.5R @ 1201.44 |
| Stop hit — per-position SL triggered | 2023-11-24 10:00:00 | 1205.05 | 1206.68 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-28 09:45:00 | 1193.55 | 1197.62 | 0.00 | ORB-short ORB[1195.00,1202.20] vol=1.6x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 09:50:00 | 1189.73 | 1196.49 | 0.00 | T1 1.5R @ 1189.73 |
| Stop hit — per-position SL triggered | 2023-11-28 10:35:00 | 1193.55 | 1193.75 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 09:35:00 | 1232.65 | 1226.82 | 0.00 | ORB-long ORB[1222.00,1227.45] vol=1.9x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 09:40:00 | 1236.36 | 1230.85 | 0.00 | T1 1.5R @ 1236.36 |
| Stop hit — per-position SL triggered | 2023-12-06 09:55:00 | 1232.65 | 1235.57 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-12-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:50:00 | 1222.65 | 1225.12 | 0.00 | ORB-short ORB[1225.00,1230.00] vol=3.2x ATR=2.12 |
| Stop hit — per-position SL triggered | 2023-12-08 11:15:00 | 1224.77 | 1224.59 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 09:35:00 | 1219.85 | 1222.53 | 0.00 | ORB-short ORB[1220.05,1233.95] vol=2.1x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 09:50:00 | 1214.69 | 1220.51 | 0.00 | T1 1.5R @ 1214.69 |
| Target hit | 2023-12-13 14:35:00 | 1210.85 | 1209.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — BUY (started 2023-12-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 10:05:00 | 1282.00 | 1272.42 | 0.00 | ORB-long ORB[1268.00,1274.80] vol=2.1x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 10:15:00 | 1288.88 | 1276.70 | 0.00 | T1 1.5R @ 1288.88 |
| Target hit | 2023-12-15 14:15:00 | 1302.20 | 1302.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 67 — SELL (started 2023-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:40:00 | 1278.00 | 1285.44 | 0.00 | ORB-short ORB[1287.40,1300.95] vol=2.2x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 09:50:00 | 1271.35 | 1282.03 | 0.00 | T1 1.5R @ 1271.35 |
| Stop hit — per-position SL triggered | 2023-12-19 10:40:00 | 1278.00 | 1278.91 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-12-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:05:00 | 1275.60 | 1267.76 | 0.00 | ORB-long ORB[1262.50,1275.00] vol=1.6x ATR=4.78 |
| Stop hit — per-position SL triggered | 2023-12-26 10:20:00 | 1270.82 | 1268.65 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 11:15:00 | 1280.90 | 1285.76 | 0.00 | ORB-short ORB[1282.85,1291.00] vol=1.6x ATR=2.86 |
| Stop hit — per-position SL triggered | 2023-12-27 11:35:00 | 1283.76 | 1285.09 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-01-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 10:45:00 | 1275.80 | 1272.69 | 0.00 | ORB-long ORB[1265.25,1275.70] vol=2.2x ATR=2.66 |
| Stop hit — per-position SL triggered | 2024-01-01 11:00:00 | 1273.14 | 1272.84 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:35:00 | 1275.20 | 1284.87 | 0.00 | ORB-short ORB[1289.50,1299.10] vol=1.7x ATR=3.55 |
| Stop hit — per-position SL triggered | 2024-01-02 10:40:00 | 1278.75 | 1284.50 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-01-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:40:00 | 1250.20 | 1260.67 | 0.00 | ORB-short ORB[1259.10,1274.00] vol=2.0x ATR=4.07 |
| Stop hit — per-position SL triggered | 2024-01-03 09:50:00 | 1254.27 | 1259.07 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 09:45:00 | 1234.55 | 1243.16 | 0.00 | ORB-short ORB[1244.00,1254.70] vol=1.6x ATR=4.19 |
| Stop hit — per-position SL triggered | 2024-01-04 09:55:00 | 1238.74 | 1242.33 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 09:45:00 | 1234.80 | 1245.24 | 0.00 | ORB-short ORB[1244.30,1254.25] vol=1.5x ATR=4.03 |
| Stop hit — per-position SL triggered | 2024-01-08 09:50:00 | 1238.83 | 1244.33 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-01-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 10:25:00 | 1255.90 | 1250.47 | 0.00 | ORB-long ORB[1240.90,1253.40] vol=2.0x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 10:35:00 | 1261.33 | 1252.33 | 0.00 | T1 1.5R @ 1261.33 |
| Stop hit — per-position SL triggered | 2024-01-11 10:50:00 | 1255.90 | 1252.85 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 09:30:00 | 1382.35 | 1392.07 | 0.00 | ORB-short ORB[1390.05,1406.20] vol=1.6x ATR=4.68 |
| Stop hit — per-position SL triggered | 2024-01-20 09:35:00 | 1387.03 | 1391.92 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 09:35:00 | 1407.80 | 1399.50 | 0.00 | ORB-long ORB[1392.00,1404.90] vol=2.6x ATR=5.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 09:45:00 | 1415.48 | 1403.33 | 0.00 | T1 1.5R @ 1415.48 |
| Stop hit — per-position SL triggered | 2024-01-23 09:50:00 | 1407.80 | 1403.77 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 10:15:00 | 1331.55 | 1326.98 | 0.00 | ORB-long ORB[1316.80,1329.25] vol=2.2x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 11:35:00 | 1338.07 | 1329.36 | 0.00 | T1 1.5R @ 1338.07 |
| Stop hit — per-position SL triggered | 2024-01-30 12:00:00 | 1331.55 | 1329.95 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 11:05:00 | 1320.05 | 1327.26 | 0.00 | ORB-short ORB[1320.25,1331.05] vol=6.9x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-02-01 11:10:00 | 1324.33 | 1327.18 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-02-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 11:10:00 | 1342.70 | 1335.71 | 0.00 | ORB-long ORB[1323.05,1336.80] vol=1.6x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:30:00 | 1347.86 | 1337.53 | 0.00 | T1 1.5R @ 1347.86 |
| Stop hit — per-position SL triggered | 2024-02-02 12:25:00 | 1342.70 | 1341.20 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 09:30:00 | 1351.00 | 1341.04 | 0.00 | ORB-long ORB[1328.10,1344.15] vol=2.9x ATR=5.19 |
| Stop hit — per-position SL triggered | 2024-02-06 10:10:00 | 1345.81 | 1348.56 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-02-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:55:00 | 1310.00 | 1316.48 | 0.00 | ORB-short ORB[1321.05,1329.65] vol=2.7x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 11:00:00 | 1303.83 | 1315.18 | 0.00 | T1 1.5R @ 1303.83 |
| Stop hit — per-position SL triggered | 2024-02-08 11:40:00 | 1310.00 | 1313.08 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-13 09:30:00 | 1303.15 | 1310.60 | 0.00 | ORB-short ORB[1307.55,1323.95] vol=2.5x ATR=4.98 |
| Stop hit — per-position SL triggered | 2024-02-13 09:35:00 | 1308.13 | 1310.06 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-02-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 10:00:00 | 1305.00 | 1308.06 | 0.00 | ORB-short ORB[1306.10,1316.40] vol=1.7x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-02-20 10:15:00 | 1308.12 | 1307.63 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-03-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 11:10:00 | 1275.05 | 1277.62 | 0.00 | ORB-short ORB[1276.15,1287.00] vol=2.5x ATR=2.76 |
| Stop hit — per-position SL triggered | 2024-03-01 11:35:00 | 1277.81 | 1277.34 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 09:55:00 | 1264.35 | 1272.07 | 0.00 | ORB-short ORB[1270.70,1279.90] vol=2.2x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 10:00:00 | 1259.91 | 1269.57 | 0.00 | T1 1.5R @ 1259.91 |
| Stop hit — per-position SL triggered | 2024-03-05 10:05:00 | 1264.35 | 1269.52 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 10:55:00 | 1253.90 | 1256.87 | 0.00 | ORB-short ORB[1255.00,1269.25] vol=1.9x ATR=3.07 |
| Stop hit — per-position SL triggered | 2024-03-06 11:25:00 | 1256.97 | 1256.65 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-03-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 10:55:00 | 1281.60 | 1275.30 | 0.00 | ORB-long ORB[1265.55,1276.65] vol=1.7x ATR=3.48 |
| Stop hit — per-position SL triggered | 2024-03-07 12:05:00 | 1278.12 | 1277.24 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 10:40:00 | 1285.85 | 1288.30 | 0.00 | ORB-short ORB[1290.00,1299.40] vol=2.9x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:30:00 | 1280.63 | 1286.87 | 0.00 | T1 1.5R @ 1280.63 |
| Target hit | 2024-03-13 15:05:00 | 1281.40 | 1279.74 | 0.00 | Trail-exit close>VWAP |

### Cycle 90 — SELL (started 2024-03-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:55:00 | 1259.00 | 1264.55 | 0.00 | ORB-short ORB[1260.00,1272.65] vol=2.1x ATR=3.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 11:40:00 | 1253.30 | 1260.64 | 0.00 | T1 1.5R @ 1253.30 |
| Stop hit — per-position SL triggered | 2024-03-19 12:20:00 | 1259.00 | 1260.07 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-03-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 11:10:00 | 1283.00 | 1277.88 | 0.00 | ORB-long ORB[1270.65,1279.60] vol=1.5x ATR=3.02 |
| Stop hit — per-position SL triggered | 2024-03-21 11:40:00 | 1279.98 | 1280.97 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-03-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-28 09:45:00 | 1247.00 | 1249.78 | 0.00 | ORB-short ORB[1249.00,1260.00] vol=1.5x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-03-28 09:50:00 | 1250.28 | 1249.78 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 1248.10 | 1253.23 | 0.00 | ORB-short ORB[1252.10,1264.90] vol=2.0x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-04-04 10:55:00 | 1251.74 | 1253.11 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2024-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:30:00 | 1278.15 | 1272.76 | 0.00 | ORB-long ORB[1264.70,1276.00] vol=2.5x ATR=3.36 |
| Stop hit — per-position SL triggered | 2024-04-09 09:50:00 | 1274.79 | 1275.55 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-04-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:10:00 | 1262.15 | 1257.75 | 0.00 | ORB-long ORB[1252.00,1261.65] vol=1.5x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 10:25:00 | 1266.93 | 1262.30 | 0.00 | T1 1.5R @ 1266.93 |
| Target hit | 2024-04-10 11:45:00 | 1264.00 | 1264.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 96 — SELL (started 2024-04-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 10:20:00 | 1253.95 | 1259.48 | 0.00 | ORB-short ORB[1262.80,1269.75] vol=4.2x ATR=3.39 |
| Stop hit — per-position SL triggered | 2024-04-12 11:35:00 | 1257.34 | 1257.95 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-15 11:05:00 | 1217.00 | 1228.95 | 0.00 | ORB-short ORB[1225.05,1240.65] vol=3.0x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-04-15 11:35:00 | 1221.00 | 1227.49 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2024-04-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-19 10:35:00 | 1182.45 | 1176.00 | 0.00 | ORB-long ORB[1162.95,1178.90] vol=2.7x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-04-19 10:45:00 | 1177.86 | 1176.23 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2024-04-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 09:55:00 | 1205.70 | 1209.37 | 0.00 | ORB-short ORB[1209.40,1218.00] vol=2.2x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 12:45:00 | 1201.47 | 1206.74 | 0.00 | T1 1.5R @ 1201.47 |
| Stop hit — per-position SL triggered | 2024-04-23 13:20:00 | 1205.70 | 1206.23 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-24 10:00:00 | 1198.95 | 1201.08 | 0.00 | ORB-short ORB[1200.50,1207.95] vol=2.4x ATR=2.36 |
| Stop hit — per-position SL triggered | 2024-04-24 10:05:00 | 1201.31 | 1201.05 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2024-04-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:35:00 | 1177.85 | 1182.93 | 0.00 | ORB-short ORB[1184.00,1191.20] vol=10.1x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-04-25 10:40:00 | 1181.23 | 1182.84 | 0.00 | SL hit |

### Cycle 102 — SELL (started 2024-04-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 10:45:00 | 1275.55 | 1282.15 | 0.00 | ORB-short ORB[1278.75,1289.60] vol=1.7x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 11:05:00 | 1270.49 | 1280.26 | 0.00 | T1 1.5R @ 1270.49 |
| Target hit | 2024-04-30 15:20:00 | 1262.00 | 1270.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 103 — SELL (started 2024-05-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 10:10:00 | 1256.65 | 1259.81 | 0.00 | ORB-short ORB[1258.65,1274.00] vol=2.3x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-05-02 10:15:00 | 1260.26 | 1259.78 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-05-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 09:35:00 | 1271.35 | 1278.63 | 0.00 | ORB-short ORB[1275.10,1285.90] vol=1.5x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:20:00 | 1264.75 | 1269.15 | 0.00 | T1 1.5R @ 1264.75 |
| Target hit | 2024-05-09 10:55:00 | 1269.40 | 1268.65 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 10:55:00 | 1046.50 | 2023-05-12 12:20:00 | 1052.64 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2023-05-12 10:55:00 | 1046.50 | 2023-05-12 15:00:00 | 1046.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-15 09:40:00 | 1058.00 | 2023-05-15 10:15:00 | 1061.66 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-05-15 09:40:00 | 1058.00 | 2023-05-15 15:10:00 | 1063.25 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2023-05-19 10:25:00 | 1071.00 | 2023-05-19 11:00:00 | 1068.06 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-05-24 09:45:00 | 1101.50 | 2023-05-24 09:55:00 | 1098.49 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-05-26 09:45:00 | 1115.15 | 2023-05-26 09:50:00 | 1111.91 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-05-29 11:15:00 | 1112.55 | 2023-05-29 11:30:00 | 1114.67 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-05-30 09:30:00 | 1110.60 | 2023-05-30 09:55:00 | 1113.45 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-05-31 10:00:00 | 1116.20 | 2023-05-31 11:00:00 | 1113.54 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-06-06 10:05:00 | 1084.35 | 2023-06-06 10:20:00 | 1080.08 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-06-06 10:05:00 | 1084.35 | 2023-06-06 13:10:00 | 1082.30 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2023-06-09 10:10:00 | 1061.45 | 2023-06-09 10:55:00 | 1064.46 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-06-12 11:15:00 | 1075.25 | 2023-06-12 12:00:00 | 1078.05 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-06-12 11:15:00 | 1075.25 | 2023-06-12 12:25:00 | 1075.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-13 11:10:00 | 1084.55 | 2023-06-13 11:20:00 | 1083.01 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-06-16 10:50:00 | 1075.75 | 2023-06-16 11:25:00 | 1072.72 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-06-16 10:50:00 | 1075.75 | 2023-06-16 12:55:00 | 1075.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-20 09:45:00 | 1100.85 | 2023-06-20 10:25:00 | 1097.91 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-06-21 10:45:00 | 1106.45 | 2023-06-21 11:20:00 | 1109.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-07-03 11:15:00 | 1119.40 | 2023-07-03 11:20:00 | 1116.48 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-07-03 11:15:00 | 1119.40 | 2023-07-03 11:35:00 | 1119.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-12 09:30:00 | 1158.10 | 2023-07-12 09:45:00 | 1153.80 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-07-12 09:30:00 | 1158.10 | 2023-07-12 09:55:00 | 1158.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-14 10:45:00 | 1197.75 | 2023-07-14 10:55:00 | 1202.92 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-07-14 10:45:00 | 1197.75 | 2023-07-14 15:20:00 | 1229.75 | TARGET_HIT | 0.50 | 2.67% |
| SELL | retest1 | 2023-07-18 09:30:00 | 1226.80 | 2023-07-18 09:45:00 | 1230.93 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-07-31 09:40:00 | 1098.00 | 2023-07-31 09:45:00 | 1094.90 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-02 09:40:00 | 1143.60 | 2023-08-02 09:45:00 | 1139.92 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-08-04 09:35:00 | 1160.65 | 2023-08-04 09:40:00 | 1156.52 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-08-08 09:45:00 | 1191.25 | 2023-08-08 09:50:00 | 1196.57 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-08-08 09:45:00 | 1191.25 | 2023-08-08 11:20:00 | 1191.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-09 10:45:00 | 1220.00 | 2023-08-09 11:00:00 | 1224.62 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-08-09 10:45:00 | 1220.00 | 2023-08-09 11:10:00 | 1220.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-10 11:10:00 | 1231.25 | 2023-08-10 12:40:00 | 1236.10 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-08-10 11:10:00 | 1231.25 | 2023-08-10 13:50:00 | 1231.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-16 09:35:00 | 1236.50 | 2023-08-16 09:40:00 | 1233.24 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-08-17 09:55:00 | 1219.40 | 2023-08-17 10:30:00 | 1222.49 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-08-18 10:50:00 | 1210.00 | 2023-08-18 11:25:00 | 1206.08 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-08-18 10:50:00 | 1210.00 | 2023-08-18 12:05:00 | 1210.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-22 09:50:00 | 1209.70 | 2023-08-22 10:05:00 | 1212.06 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-24 11:15:00 | 1201.00 | 2023-08-24 11:20:00 | 1203.52 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-08-25 10:55:00 | 1180.35 | 2023-08-25 11:15:00 | 1183.69 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-29 09:35:00 | 1198.95 | 2023-08-29 09:50:00 | 1202.81 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-08-29 09:35:00 | 1198.95 | 2023-08-29 09:55:00 | 1198.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-01 11:05:00 | 1212.75 | 2023-09-01 12:50:00 | 1216.85 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-09-01 11:05:00 | 1212.75 | 2023-09-01 15:20:00 | 1227.95 | TARGET_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2023-09-04 09:30:00 | 1231.15 | 2023-09-04 09:40:00 | 1234.36 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-09-07 10:45:00 | 1261.65 | 2023-09-07 10:55:00 | 1266.47 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-09-07 10:45:00 | 1261.65 | 2023-09-07 11:30:00 | 1261.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-08 09:50:00 | 1265.25 | 2023-09-08 09:55:00 | 1268.94 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-09-14 09:30:00 | 1281.80 | 2023-09-14 09:40:00 | 1286.56 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-09-14 09:30:00 | 1281.80 | 2023-09-14 10:05:00 | 1281.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-22 10:05:00 | 1303.55 | 2023-09-22 10:25:00 | 1310.35 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-09-22 10:05:00 | 1303.55 | 2023-09-22 13:50:00 | 1303.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-27 09:45:00 | 1279.65 | 2023-09-27 09:55:00 | 1276.41 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-09-28 09:35:00 | 1256.00 | 2023-09-28 09:50:00 | 1249.64 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-09-28 09:35:00 | 1256.00 | 2023-09-28 15:20:00 | 1235.00 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2023-10-05 09:40:00 | 1220.00 | 2023-10-05 10:25:00 | 1216.71 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-10-06 11:05:00 | 1216.85 | 2023-10-06 11:40:00 | 1220.85 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-10-06 11:05:00 | 1216.85 | 2023-10-06 13:30:00 | 1216.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-09 09:30:00 | 1221.25 | 2023-10-09 09:35:00 | 1226.63 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-10-09 09:30:00 | 1221.25 | 2023-10-09 09:40:00 | 1221.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-12 09:45:00 | 1219.85 | 2023-10-12 10:10:00 | 1222.86 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-10-13 09:30:00 | 1198.10 | 2023-10-13 09:35:00 | 1192.90 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-10-16 09:30:00 | 1188.40 | 2023-10-16 09:45:00 | 1191.18 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-10-17 10:40:00 | 1195.50 | 2023-10-17 11:00:00 | 1192.58 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-10-19 09:45:00 | 1181.00 | 2023-10-19 09:50:00 | 1178.08 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-10-20 10:00:00 | 1162.15 | 2023-10-20 10:15:00 | 1158.17 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-10-20 10:00:00 | 1162.15 | 2023-10-20 11:05:00 | 1162.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-31 10:35:00 | 1119.85 | 2023-10-31 11:00:00 | 1116.12 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-10-31 10:35:00 | 1119.85 | 2023-10-31 11:20:00 | 1119.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-02 09:55:00 | 1121.45 | 2023-11-02 10:10:00 | 1116.96 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-11-02 09:55:00 | 1121.45 | 2023-11-02 11:10:00 | 1121.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-07 11:00:00 | 1144.95 | 2023-11-07 11:20:00 | 1143.09 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-11-09 10:05:00 | 1127.00 | 2023-11-09 10:10:00 | 1123.54 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-11-09 10:05:00 | 1127.00 | 2023-11-09 13:45:00 | 1124.15 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2023-11-10 10:55:00 | 1130.65 | 2023-11-10 11:05:00 | 1128.47 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-11-13 10:35:00 | 1132.75 | 2023-11-13 11:00:00 | 1134.89 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-11-16 11:00:00 | 1180.55 | 2023-11-16 11:20:00 | 1184.02 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-11-16 11:00:00 | 1180.55 | 2023-11-16 15:20:00 | 1204.85 | TARGET_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2023-11-17 11:10:00 | 1203.30 | 2023-11-17 12:20:00 | 1199.23 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-11-17 11:10:00 | 1203.30 | 2023-11-17 12:30:00 | 1203.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-20 09:40:00 | 1218.00 | 2023-11-20 10:00:00 | 1214.29 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-11-21 09:35:00 | 1213.00 | 2023-11-21 09:55:00 | 1209.41 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-11-21 09:35:00 | 1213.00 | 2023-11-21 12:10:00 | 1209.10 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2023-11-23 11:00:00 | 1213.35 | 2023-11-23 12:15:00 | 1209.65 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-11-23 11:00:00 | 1213.35 | 2023-11-23 13:00:00 | 1213.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-24 09:35:00 | 1205.05 | 2023-11-24 09:55:00 | 1201.44 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-11-24 09:35:00 | 1205.05 | 2023-11-24 10:00:00 | 1205.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-28 09:45:00 | 1193.55 | 2023-11-28 09:50:00 | 1189.73 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-11-28 09:45:00 | 1193.55 | 2023-11-28 10:35:00 | 1193.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-06 09:35:00 | 1232.65 | 2023-12-06 09:40:00 | 1236.36 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-12-06 09:35:00 | 1232.65 | 2023-12-06 09:55:00 | 1232.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-08 10:50:00 | 1222.65 | 2023-12-08 11:15:00 | 1224.77 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-12-13 09:35:00 | 1219.85 | 2023-12-13 09:50:00 | 1214.69 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-12-13 09:35:00 | 1219.85 | 2023-12-13 14:35:00 | 1210.85 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2023-12-15 10:05:00 | 1282.00 | 2023-12-15 10:15:00 | 1288.88 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-12-15 10:05:00 | 1282.00 | 2023-12-15 14:15:00 | 1302.20 | TARGET_HIT | 0.50 | 1.58% |
| SELL | retest1 | 2023-12-19 09:40:00 | 1278.00 | 2023-12-19 09:50:00 | 1271.35 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-12-19 09:40:00 | 1278.00 | 2023-12-19 10:40:00 | 1278.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 10:05:00 | 1275.60 | 2023-12-26 10:20:00 | 1270.82 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-12-27 11:15:00 | 1280.90 | 2023-12-27 11:35:00 | 1283.76 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-01-01 10:45:00 | 1275.80 | 2024-01-01 11:00:00 | 1273.14 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-01-02 10:35:00 | 1275.20 | 2024-01-02 10:40:00 | 1278.75 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-03 09:40:00 | 1250.20 | 2024-01-03 09:50:00 | 1254.27 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-01-04 09:45:00 | 1234.55 | 2024-01-04 09:55:00 | 1238.74 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-01-08 09:45:00 | 1234.80 | 2024-01-08 09:50:00 | 1238.83 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-01-11 10:25:00 | 1255.90 | 2024-01-11 10:35:00 | 1261.33 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-01-11 10:25:00 | 1255.90 | 2024-01-11 10:50:00 | 1255.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-20 09:30:00 | 1382.35 | 2024-01-20 09:35:00 | 1387.03 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-01-23 09:35:00 | 1407.80 | 2024-01-23 09:45:00 | 1415.48 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-01-23 09:35:00 | 1407.80 | 2024-01-23 09:50:00 | 1407.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-30 10:15:00 | 1331.55 | 2024-01-30 11:35:00 | 1338.07 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-01-30 10:15:00 | 1331.55 | 2024-01-30 12:00:00 | 1331.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-01 11:05:00 | 1320.05 | 2024-02-01 11:10:00 | 1324.33 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-02-02 11:10:00 | 1342.70 | 2024-02-02 11:30:00 | 1347.86 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-02-02 11:10:00 | 1342.70 | 2024-02-02 12:25:00 | 1342.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-06 09:30:00 | 1351.00 | 2024-02-06 10:10:00 | 1345.81 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-02-08 10:55:00 | 1310.00 | 2024-02-08 11:00:00 | 1303.83 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-02-08 10:55:00 | 1310.00 | 2024-02-08 11:40:00 | 1310.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-13 09:30:00 | 1303.15 | 2024-02-13 09:35:00 | 1308.13 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-02-20 10:00:00 | 1305.00 | 2024-02-20 10:15:00 | 1308.12 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-03-01 11:10:00 | 1275.05 | 2024-03-01 11:35:00 | 1277.81 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-03-05 09:55:00 | 1264.35 | 2024-03-05 10:00:00 | 1259.91 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-03-05 09:55:00 | 1264.35 | 2024-03-05 10:05:00 | 1264.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-06 10:55:00 | 1253.90 | 2024-03-06 11:25:00 | 1256.97 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-03-07 10:55:00 | 1281.60 | 2024-03-07 12:05:00 | 1278.12 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-03-13 10:40:00 | 1285.85 | 2024-03-13 11:30:00 | 1280.63 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-03-13 10:40:00 | 1285.85 | 2024-03-13 15:05:00 | 1281.40 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2024-03-19 09:55:00 | 1259.00 | 2024-03-19 11:40:00 | 1253.30 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-03-19 09:55:00 | 1259.00 | 2024-03-19 12:20:00 | 1259.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 11:10:00 | 1283.00 | 2024-03-21 11:40:00 | 1279.98 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-03-28 09:45:00 | 1247.00 | 2024-03-28 09:50:00 | 1250.28 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-04 10:50:00 | 1248.10 | 2024-04-04 10:55:00 | 1251.74 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-09 09:30:00 | 1278.15 | 2024-04-09 09:50:00 | 1274.79 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-04-10 10:10:00 | 1262.15 | 2024-04-10 10:25:00 | 1266.93 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-04-10 10:10:00 | 1262.15 | 2024-04-10 11:45:00 | 1264.00 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2024-04-12 10:20:00 | 1253.95 | 2024-04-12 11:35:00 | 1257.34 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-04-15 11:05:00 | 1217.00 | 2024-04-15 11:35:00 | 1221.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-04-19 10:35:00 | 1182.45 | 2024-04-19 10:45:00 | 1177.86 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-04-23 09:55:00 | 1205.70 | 2024-04-23 12:45:00 | 1201.47 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-04-23 09:55:00 | 1205.70 | 2024-04-23 13:20:00 | 1205.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-24 10:00:00 | 1198.95 | 2024-04-24 10:05:00 | 1201.31 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-04-25 10:35:00 | 1177.85 | 2024-04-25 10:40:00 | 1181.23 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-04-30 10:45:00 | 1275.55 | 2024-04-30 11:05:00 | 1270.49 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-04-30 10:45:00 | 1275.55 | 2024-04-30 15:20:00 | 1262.00 | TARGET_HIT | 0.50 | 1.06% |
| SELL | retest1 | 2024-05-02 10:10:00 | 1256.65 | 2024-05-02 10:15:00 | 1260.26 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-05-09 09:35:00 | 1271.35 | 2024-05-09 10:20:00 | 1264.75 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-05-09 09:35:00 | 1271.35 | 2024-05-09 10:55:00 | 1269.40 | TARGET_HIT | 0.50 | 0.15% |
