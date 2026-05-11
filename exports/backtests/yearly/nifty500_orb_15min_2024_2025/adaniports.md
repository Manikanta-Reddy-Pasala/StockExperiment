# Adani Ports and Special Economic Zone Ltd. (ADANIPORTS)

## Backtest Summary

- **Window:** 2024-11-07 09:15:00 → 2026-05-08 15:25:00 (27688 bars)
- **Last close:** 1760.00
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
| ENTRY1 | 42 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 6 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 36
- **Target hits / Stop hits / Partials:** 6 / 36 / 12
- **Avg / median % per leg:** 0.08% / -0.26%
- **Sum % (uncompounded):** 4.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 7 | 23.3% | 1 | 23 | 6 | -0.07% | -2.0% |
| BUY @ 2nd Alert (retest1) | 30 | 7 | 23.3% | 1 | 23 | 6 | -0.07% | -2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 24 | 11 | 45.8% | 5 | 13 | 6 | 0.27% | 6.5% |
| SELL @ 2nd Alert (retest1) | 24 | 11 | 45.8% | 5 | 13 | 6 | 0.27% | 6.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 54 | 18 | 33.3% | 6 | 36 | 12 | 0.08% | 4.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:45:00 | 1359.90 | 1351.23 | 0.00 | ORB-long ORB[1338.60,1356.10] vol=1.6x ATR=4.64 |
| Stop hit — per-position SL triggered | 2024-11-08 09:50:00 | 1355.26 | 1351.56 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-12-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:05:00 | 1241.50 | 1249.93 | 0.00 | ORB-short ORB[1250.40,1258.35] vol=2.2x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 11:15:00 | 1236.07 | 1245.74 | 0.00 | T1 1.5R @ 1236.07 |
| Target hit | 2024-12-11 15:20:00 | 1232.20 | 1239.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-12-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:30:00 | 1243.10 | 1231.38 | 0.00 | ORB-long ORB[1226.60,1237.95] vol=2.8x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-12-12 10:35:00 | 1239.14 | 1232.32 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-12-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:10:00 | 1244.20 | 1250.14 | 0.00 | ORB-short ORB[1251.55,1261.85] vol=2.4x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-12-16 12:35:00 | 1247.46 | 1248.77 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 1250.15 | 1246.27 | 0.00 | ORB-long ORB[1234.35,1249.80] vol=1.9x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-12-17 09:35:00 | 1247.19 | 1246.50 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:35:00 | 1209.70 | 1205.37 | 0.00 | ORB-long ORB[1199.30,1209.05] vol=1.7x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-12-20 09:45:00 | 1206.41 | 1206.43 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 11:00:00 | 1199.85 | 1193.53 | 0.00 | ORB-long ORB[1182.55,1194.50] vol=2.2x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:10:00 | 1204.97 | 1195.32 | 0.00 | T1 1.5R @ 1204.97 |
| Stop hit — per-position SL triggered | 2024-12-26 12:00:00 | 1199.85 | 1197.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:30:00 | 1249.85 | 1246.02 | 0.00 | ORB-long ORB[1233.00,1249.00] vol=2.7x ATR=4.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 09:35:00 | 1256.85 | 1248.30 | 0.00 | T1 1.5R @ 1256.85 |
| Stop hit — per-position SL triggered | 2024-12-30 09:55:00 | 1249.85 | 1250.03 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 11:15:00 | 1213.60 | 1216.55 | 0.00 | ORB-short ORB[1216.75,1225.80] vol=2.4x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-12-31 12:00:00 | 1217.60 | 1216.39 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-01-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 09:45:00 | 1216.05 | 1222.09 | 0.00 | ORB-short ORB[1220.00,1227.90] vol=2.2x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 10:00:00 | 1209.38 | 1218.07 | 0.00 | T1 1.5R @ 1209.38 |
| Stop hit — per-position SL triggered | 2025-01-01 10:35:00 | 1216.05 | 1216.02 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-01-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:55:00 | 1220.75 | 1225.89 | 0.00 | ORB-short ORB[1222.00,1237.60] vol=2.1x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 12:05:00 | 1215.13 | 1224.21 | 0.00 | T1 1.5R @ 1215.13 |
| Target hit | 2025-01-03 15:20:00 | 1198.50 | 1213.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:15:00 | 1172.10 | 1186.90 | 0.00 | ORB-short ORB[1189.65,1200.90] vol=1.6x ATR=3.44 |
| Stop hit — per-position SL triggered | 2025-01-06 11:30:00 | 1175.54 | 1185.96 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-01-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:05:00 | 1086.80 | 1094.76 | 0.00 | ORB-short ORB[1091.65,1106.45] vol=2.6x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:30:00 | 1080.39 | 1091.02 | 0.00 | T1 1.5R @ 1080.39 |
| Target hit | 2025-01-13 15:20:00 | 1066.50 | 1082.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2025-01-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 09:40:00 | 1139.65 | 1131.27 | 0.00 | ORB-long ORB[1120.05,1131.80] vol=2.3x ATR=4.68 |
| Stop hit — per-position SL triggered | 2025-01-15 09:50:00 | 1134.97 | 1132.27 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 1121.80 | 1134.82 | 0.00 | ORB-short ORB[1141.50,1154.25] vol=2.2x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-01-21 10:25:00 | 1125.19 | 1134.09 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-01-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:20:00 | 1095.30 | 1098.83 | 0.00 | ORB-short ORB[1102.05,1113.20] vol=2.1x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-01-24 10:35:00 | 1098.95 | 1098.76 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-01-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:00:00 | 1091.95 | 1084.49 | 0.00 | ORB-long ORB[1074.50,1089.45] vol=3.6x ATR=3.69 |
| Stop hit — per-position SL triggered | 2025-01-29 11:10:00 | 1088.26 | 1085.16 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:45:00 | 1108.45 | 1103.59 | 0.00 | ORB-long ORB[1093.50,1107.85] vol=2.1x ATR=4.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 11:15:00 | 1115.06 | 1109.48 | 0.00 | T1 1.5R @ 1115.06 |
| Stop hit — per-position SL triggered | 2025-01-30 12:15:00 | 1108.45 | 1109.98 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:45:00 | 1068.25 | 1070.59 | 0.00 | ORB-short ORB[1072.15,1085.00] vol=1.8x ATR=4.01 |
| Stop hit — per-position SL triggered | 2025-02-18 10:50:00 | 1072.26 | 1070.64 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:15:00 | 1102.35 | 1089.02 | 0.00 | ORB-long ORB[1074.00,1087.75] vol=2.6x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:20:00 | 1106.86 | 1090.77 | 0.00 | T1 1.5R @ 1106.86 |
| Target hit | 2025-02-20 15:20:00 | 1111.20 | 1110.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2025-02-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 11:00:00 | 1096.65 | 1104.80 | 0.00 | ORB-short ORB[1106.25,1119.10] vol=1.5x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 11:30:00 | 1090.99 | 1100.94 | 0.00 | T1 1.5R @ 1090.99 |
| Target hit | 2025-02-21 15:20:00 | 1081.90 | 1092.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 1083.75 | 1073.95 | 0.00 | ORB-long ORB[1067.75,1078.00] vol=1.6x ATR=3.55 |
| Stop hit — per-position SL triggered | 2025-02-25 09:50:00 | 1080.20 | 1078.01 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 10:55:00 | 1070.70 | 1077.04 | 0.00 | ORB-short ORB[1077.05,1087.95] vol=1.6x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-02-27 11:15:00 | 1073.26 | 1076.67 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-02-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 09:35:00 | 1061.00 | 1065.83 | 0.00 | ORB-short ORB[1061.95,1076.00] vol=1.5x ATR=4.14 |
| Stop hit — per-position SL triggered | 2025-02-28 09:50:00 | 1065.14 | 1065.01 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 09:35:00 | 1052.10 | 1044.01 | 0.00 | ORB-long ORB[1036.50,1047.95] vol=1.7x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 11:05:00 | 1059.14 | 1049.96 | 0.00 | T1 1.5R @ 1059.14 |
| Stop hit — per-position SL triggered | 2025-03-04 11:40:00 | 1052.10 | 1050.59 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:30:00 | 1105.00 | 1111.02 | 0.00 | ORB-short ORB[1107.70,1120.25] vol=1.6x ATR=4.16 |
| Stop hit — per-position SL triggered | 2025-03-06 09:40:00 | 1109.16 | 1110.29 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:35:00 | 1160.00 | 1150.44 | 0.00 | ORB-long ORB[1136.95,1153.20] vol=2.4x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 09:45:00 | 1166.40 | 1154.91 | 0.00 | T1 1.5R @ 1166.40 |
| Stop hit — per-position SL triggered | 2025-03-10 10:25:00 | 1160.00 | 1158.72 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-03-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:45:00 | 1152.60 | 1144.87 | 0.00 | ORB-long ORB[1135.80,1144.65] vol=2.8x ATR=3.92 |
| Stop hit — per-position SL triggered | 2025-03-12 10:00:00 | 1148.68 | 1148.17 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:00:00 | 1152.00 | 1145.28 | 0.00 | ORB-long ORB[1137.95,1150.00] vol=1.7x ATR=3.45 |
| Stop hit — per-position SL triggered | 2025-03-18 10:35:00 | 1148.55 | 1147.09 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-03-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:35:00 | 1180.25 | 1169.79 | 0.00 | ORB-long ORB[1158.80,1169.95] vol=1.8x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-03-19 12:05:00 | 1176.71 | 1175.26 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 1190.45 | 1181.67 | 0.00 | ORB-long ORB[1172.05,1183.95] vol=1.6x ATR=3.58 |
| Stop hit — per-position SL triggered | 2025-03-21 09:45:00 | 1186.87 | 1182.31 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 11:15:00 | 1205.45 | 1201.19 | 0.00 | ORB-long ORB[1194.05,1203.15] vol=1.8x ATR=3.24 |
| Stop hit — per-position SL triggered | 2025-03-24 11:40:00 | 1202.21 | 1201.66 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-03-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 10:55:00 | 1181.10 | 1193.71 | 0.00 | ORB-short ORB[1196.30,1209.10] vol=2.0x ATR=4.78 |
| Stop hit — per-position SL triggered | 2025-03-25 12:05:00 | 1185.88 | 1190.89 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 11:15:00 | 1194.40 | 1191.15 | 0.00 | ORB-long ORB[1173.05,1190.90] vol=1.9x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-03-26 11:45:00 | 1190.75 | 1191.66 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-04-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 11:05:00 | 1116.75 | 1119.57 | 0.00 | ORB-short ORB[1119.00,1133.85] vol=3.5x ATR=3.98 |
| Stop hit — per-position SL triggered | 2025-04-09 11:15:00 | 1120.73 | 1119.32 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-04-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:35:00 | 1165.85 | 1158.82 | 0.00 | ORB-long ORB[1148.00,1164.95] vol=2.3x ATR=4.88 |
| Stop hit — per-position SL triggered | 2025-04-11 10:05:00 | 1160.97 | 1161.97 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:15:00 | 1214.00 | 1209.26 | 0.00 | ORB-long ORB[1202.40,1211.90] vol=1.6x ATR=3.73 |
| Stop hit — per-position SL triggered | 2025-04-16 11:40:00 | 1210.27 | 1210.10 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:45:00 | 1252.50 | 1243.44 | 0.00 | ORB-long ORB[1233.90,1249.90] vol=2.2x ATR=3.62 |
| Stop hit — per-position SL triggered | 2025-04-22 10:50:00 | 1248.88 | 1243.88 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-04-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:05:00 | 1222.00 | 1232.60 | 0.00 | ORB-short ORB[1230.10,1241.90] vol=2.0x ATR=4.17 |
| Stop hit — per-position SL triggered | 2025-04-23 11:10:00 | 1226.17 | 1228.30 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 1220.70 | 1228.73 | 0.00 | ORB-short ORB[1226.20,1243.40] vol=2.7x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:35:00 | 1215.06 | 1225.76 | 0.00 | T1 1.5R @ 1215.06 |
| Target hit | 2025-04-25 12:55:00 | 1199.80 | 1198.97 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 1200.80 | 1192.31 | 0.00 | ORB-long ORB[1184.10,1199.00] vol=1.9x ATR=5.11 |
| Stop hit — per-position SL triggered | 2025-04-28 09:40:00 | 1195.69 | 1192.86 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:30:00 | 1312.40 | 1301.05 | 0.00 | ORB-long ORB[1285.70,1304.90] vol=2.5x ATR=6.77 |
| Stop hit — per-position SL triggered | 2025-05-05 09:35:00 | 1305.63 | 1302.20 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-11-08 09:45:00 | 1359.90 | 2024-11-08 09:50:00 | 1355.26 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-11 10:05:00 | 1241.50 | 2024-12-11 11:15:00 | 1236.07 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-11 10:05:00 | 1241.50 | 2024-12-11 15:20:00 | 1232.20 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2024-12-12 10:30:00 | 1243.10 | 2024-12-12 10:35:00 | 1239.14 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-16 11:10:00 | 1244.20 | 2024-12-16 12:35:00 | 1247.46 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-17 09:30:00 | 1250.15 | 2024-12-17 09:35:00 | 1247.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-20 09:35:00 | 1209.70 | 2024-12-20 09:45:00 | 1206.41 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-26 11:00:00 | 1199.85 | 2024-12-26 11:10:00 | 1204.97 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-26 11:00:00 | 1199.85 | 2024-12-26 12:00:00 | 1199.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 09:30:00 | 1249.85 | 2024-12-30 09:35:00 | 1256.85 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-12-30 09:30:00 | 1249.85 | 2024-12-30 09:55:00 | 1249.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-31 11:15:00 | 1213.60 | 2024-12-31 12:00:00 | 1217.60 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-01 09:45:00 | 1216.05 | 2025-01-01 10:00:00 | 1209.38 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-01 09:45:00 | 1216.05 | 2025-01-01 10:35:00 | 1216.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 10:55:00 | 1220.75 | 2025-01-03 12:05:00 | 1215.13 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-01-03 10:55:00 | 1220.75 | 2025-01-03 15:20:00 | 1198.50 | TARGET_HIT | 0.50 | 1.82% |
| SELL | retest1 | 2025-01-06 11:15:00 | 1172.10 | 2025-01-06 11:30:00 | 1175.54 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-13 11:05:00 | 1086.80 | 2025-01-13 13:30:00 | 1080.39 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-01-13 11:05:00 | 1086.80 | 2025-01-13 15:20:00 | 1066.50 | TARGET_HIT | 0.50 | 1.87% |
| BUY | retest1 | 2025-01-15 09:40:00 | 1139.65 | 2025-01-15 09:50:00 | 1134.97 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-21 10:20:00 | 1121.80 | 2025-01-21 10:25:00 | 1125.19 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-24 10:20:00 | 1095.30 | 2025-01-24 10:35:00 | 1098.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-29 11:00:00 | 1091.95 | 2025-01-29 11:10:00 | 1088.26 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-30 09:45:00 | 1108.45 | 2025-01-30 11:15:00 | 1115.06 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-01-30 09:45:00 | 1108.45 | 2025-01-30 12:15:00 | 1108.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-18 10:45:00 | 1068.25 | 2025-02-18 10:50:00 | 1072.26 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-02-20 11:15:00 | 1102.35 | 2025-02-20 11:20:00 | 1106.86 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-02-20 11:15:00 | 1102.35 | 2025-02-20 15:20:00 | 1111.20 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2025-02-21 11:00:00 | 1096.65 | 2025-02-21 11:30:00 | 1090.99 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-02-21 11:00:00 | 1096.65 | 2025-02-21 15:20:00 | 1081.90 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2025-02-25 09:30:00 | 1083.75 | 2025-02-25 09:50:00 | 1080.20 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-27 10:55:00 | 1070.70 | 2025-02-27 11:15:00 | 1073.26 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-02-28 09:35:00 | 1061.00 | 2025-02-28 09:50:00 | 1065.14 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-04 09:35:00 | 1052.10 | 2025-03-04 11:05:00 | 1059.14 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-03-04 09:35:00 | 1052.10 | 2025-03-04 11:40:00 | 1052.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-06 09:30:00 | 1105.00 | 2025-03-06 09:40:00 | 1109.16 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-03-10 09:35:00 | 1160.00 | 2025-03-10 09:45:00 | 1166.40 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-03-10 09:35:00 | 1160.00 | 2025-03-10 10:25:00 | 1160.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-12 09:45:00 | 1152.60 | 2025-03-12 10:00:00 | 1148.68 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-18 10:00:00 | 1152.00 | 2025-03-18 10:35:00 | 1148.55 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-19 10:35:00 | 1180.25 | 2025-03-19 12:05:00 | 1176.71 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-21 09:40:00 | 1190.45 | 2025-03-21 09:45:00 | 1186.87 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-24 11:15:00 | 1205.45 | 2025-03-24 11:40:00 | 1202.21 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-03-25 10:55:00 | 1181.10 | 2025-03-25 12:05:00 | 1185.88 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-26 11:15:00 | 1194.40 | 2025-03-26 11:45:00 | 1190.75 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-04-09 11:05:00 | 1116.75 | 2025-04-09 11:15:00 | 1120.73 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-11 09:35:00 | 1165.85 | 2025-04-11 10:05:00 | 1160.97 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-04-16 11:15:00 | 1214.00 | 2025-04-16 11:40:00 | 1210.27 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-22 10:45:00 | 1252.50 | 2025-04-22 10:50:00 | 1248.88 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-23 10:05:00 | 1222.00 | 2025-04-23 11:10:00 | 1226.17 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-04-25 09:30:00 | 1220.70 | 2025-04-25 09:35:00 | 1215.06 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-04-25 09:30:00 | 1220.70 | 2025-04-25 12:55:00 | 1199.80 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2025-04-28 09:30:00 | 1200.80 | 2025-04-28 09:40:00 | 1195.69 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-05-05 09:30:00 | 1312.40 | 2025-05-05 09:35:00 | 1305.63 | STOP_HIT | 1.00 | -0.52% |
