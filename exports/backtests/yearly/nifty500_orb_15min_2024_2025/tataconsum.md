# Tata Consumer Products Ltd. (TATACONSUM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1176.60
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
| ENTRY1 | 89 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 11 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 123 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 78
- **Target hits / Stop hits / Partials:** 11 / 78 / 34
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 10.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 26 | 36.6% | 6 | 45 | 20 | 0.09% | 6.6% |
| BUY @ 2nd Alert (retest1) | 71 | 26 | 36.6% | 6 | 45 | 20 | 0.09% | 6.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 52 | 19 | 36.5% | 5 | 33 | 14 | 0.07% | 3.8% |
| SELL @ 2nd Alert (retest1) | 52 | 19 | 36.5% | 5 | 33 | 14 | 0.07% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 123 | 45 | 36.6% | 11 | 78 | 34 | 0.08% | 10.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 11:15:00 | 1082.59 | 1079.57 | 0.00 | ORB-long ORB[1075.58,1082.10] vol=2.9x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-05-21 11:20:00 | 1080.51 | 1079.59 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 11:10:00 | 1092.62 | 1086.64 | 0.00 | ORB-long ORB[1081.66,1090.15] vol=4.5x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 11:25:00 | 1096.73 | 1088.83 | 0.00 | T1 1.5R @ 1096.73 |
| Target hit | 2024-05-22 15:20:00 | 1106.94 | 1098.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:35:00 | 1088.17 | 1092.56 | 0.00 | ORB-short ORB[1090.84,1097.36] vol=1.6x ATR=3.31 |
| Stop hit — per-position SL triggered | 2024-05-24 09:45:00 | 1091.48 | 1091.64 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:25:00 | 1071.78 | 1072.24 | 0.00 | ORB-short ORB[1072.07,1078.59] vol=2.2x ATR=2.42 |
| Stop hit — per-position SL triggered | 2024-05-28 11:05:00 | 1074.20 | 1072.17 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:15:00 | 1043.28 | 1044.03 | 0.00 | ORB-short ORB[1046.15,1057.85] vol=9.1x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-05-30 11:45:00 | 1045.66 | 1044.11 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:25:00 | 1125.81 | 1121.80 | 0.00 | ORB-long ORB[1112.87,1122.05] vol=2.5x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 10:35:00 | 1129.65 | 1122.70 | 0.00 | T1 1.5R @ 1129.65 |
| Stop hit — per-position SL triggered | 2024-06-11 12:00:00 | 1125.81 | 1124.98 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 1069.41 | 1074.65 | 0.00 | ORB-short ORB[1071.68,1082.99] vol=1.7x ATR=2.34 |
| Stop hit — per-position SL triggered | 2024-06-27 09:35:00 | 1071.75 | 1074.16 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:15:00 | 1112.52 | 1106.17 | 0.00 | ORB-long ORB[1098.40,1105.81] vol=1.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2024-07-03 10:20:00 | 1109.38 | 1106.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:50:00 | 1139.68 | 1130.25 | 0.00 | ORB-long ORB[1122.05,1130.94] vol=1.9x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 10:15:00 | 1144.38 | 1137.50 | 0.00 | T1 1.5R @ 1144.38 |
| Stop hit — per-position SL triggered | 2024-07-08 11:25:00 | 1139.68 | 1141.53 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 11:05:00 | 1122.79 | 1127.70 | 0.00 | ORB-short ORB[1126.00,1132.92] vol=1.6x ATR=3.89 |
| Stop hit — per-position SL triggered | 2024-07-10 11:50:00 | 1126.68 | 1127.32 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:10:00 | 1126.05 | 1129.61 | 0.00 | ORB-short ORB[1128.33,1139.24] vol=1.8x ATR=2.42 |
| Stop hit — per-position SL triggered | 2024-07-11 11:40:00 | 1128.47 | 1128.93 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:55:00 | 1143.09 | 1135.99 | 0.00 | ORB-long ORB[1128.72,1137.51] vol=1.5x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:15:00 | 1147.59 | 1139.41 | 0.00 | T1 1.5R @ 1147.59 |
| Target hit | 2024-07-16 15:20:00 | 1160.97 | 1155.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:15:00 | 1186.40 | 1178.69 | 0.00 | ORB-long ORB[1167.74,1180.18] vol=2.4x ATR=3.34 |
| Stop hit — per-position SL triggered | 2024-07-22 10:35:00 | 1183.06 | 1182.04 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 10:25:00 | 1202.06 | 1195.43 | 0.00 | ORB-long ORB[1185.51,1200.08] vol=3.8x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-07-23 10:30:00 | 1198.68 | 1195.69 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 11:00:00 | 1206.40 | 1211.55 | 0.00 | ORB-short ORB[1206.65,1219.84] vol=1.8x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 11:40:00 | 1200.97 | 1209.34 | 0.00 | T1 1.5R @ 1200.97 |
| Stop hit — per-position SL triggered | 2024-07-25 13:10:00 | 1206.40 | 1206.72 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 10:20:00 | 1194.85 | 1199.38 | 0.00 | ORB-short ORB[1199.40,1211.40] vol=2.1x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-07-30 10:30:00 | 1198.55 | 1199.28 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:55:00 | 1202.75 | 1198.21 | 0.00 | ORB-long ORB[1188.00,1193.45] vol=1.5x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:40:00 | 1207.14 | 1200.31 | 0.00 | T1 1.5R @ 1207.14 |
| Stop hit — per-position SL triggered | 2024-08-01 13:10:00 | 1202.75 | 1204.28 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:05:00 | 1175.35 | 1184.53 | 0.00 | ORB-short ORB[1188.40,1195.00] vol=1.9x ATR=3.06 |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 1178.41 | 1183.24 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 09:40:00 | 1172.25 | 1175.26 | 0.00 | ORB-short ORB[1172.55,1182.95] vol=1.6x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:55:00 | 1168.45 | 1173.78 | 0.00 | T1 1.5R @ 1168.45 |
| Stop hit — per-position SL triggered | 2024-08-12 11:05:00 | 1172.25 | 1171.33 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 10:05:00 | 1163.15 | 1169.06 | 0.00 | ORB-short ORB[1170.65,1187.10] vol=5.3x ATR=3.60 |
| Stop hit — per-position SL triggered | 2024-08-14 10:40:00 | 1166.75 | 1167.35 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:25:00 | 1173.40 | 1180.34 | 0.00 | ORB-short ORB[1183.00,1200.00] vol=2.6x ATR=3.09 |
| Stop hit — per-position SL triggered | 2024-08-19 11:10:00 | 1176.49 | 1178.95 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 11:00:00 | 1171.50 | 1175.28 | 0.00 | ORB-short ORB[1173.50,1182.95] vol=1.8x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:35:00 | 1168.77 | 1174.25 | 0.00 | T1 1.5R @ 1168.77 |
| Stop hit — per-position SL triggered | 2024-08-20 13:35:00 | 1171.50 | 1172.34 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 1194.90 | 1190.36 | 0.00 | ORB-long ORB[1181.00,1191.90] vol=3.8x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 09:40:00 | 1198.54 | 1192.35 | 0.00 | T1 1.5R @ 1198.54 |
| Stop hit — per-position SL triggered | 2024-08-22 09:45:00 | 1194.90 | 1192.63 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 11:15:00 | 1205.00 | 1207.06 | 0.00 | ORB-short ORB[1207.15,1215.00] vol=3.9x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:35:00 | 1201.06 | 1206.82 | 0.00 | T1 1.5R @ 1201.06 |
| Stop hit — per-position SL triggered | 2024-08-23 11:50:00 | 1205.00 | 1206.78 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:50:00 | 1207.25 | 1201.57 | 0.00 | ORB-long ORB[1194.05,1207.00] vol=1.7x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 11:20:00 | 1210.72 | 1203.20 | 0.00 | T1 1.5R @ 1210.72 |
| Target hit | 2024-08-26 15:20:00 | 1219.60 | 1214.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2024-08-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:45:00 | 1203.40 | 1209.33 | 0.00 | ORB-short ORB[1208.30,1215.35] vol=1.6x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-08-28 10:50:00 | 1205.50 | 1209.20 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:50:00 | 1205.70 | 1202.81 | 0.00 | ORB-long ORB[1197.25,1203.25] vol=2.2x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 1203.49 | 1204.23 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:15:00 | 1212.00 | 1205.36 | 0.00 | ORB-long ORB[1200.05,1209.00] vol=2.1x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 10:20:00 | 1215.57 | 1207.89 | 0.00 | T1 1.5R @ 1215.57 |
| Stop hit — per-position SL triggered | 2024-09-03 11:15:00 | 1212.00 | 1211.98 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 10:15:00 | 1192.85 | 1185.76 | 0.00 | ORB-long ORB[1169.90,1179.00] vol=2.3x ATR=3.85 |
| Stop hit — per-position SL triggered | 2024-09-09 10:45:00 | 1189.00 | 1189.05 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 11:10:00 | 1206.60 | 1203.11 | 0.00 | ORB-long ORB[1196.25,1204.40] vol=2.9x ATR=2.53 |
| Stop hit — per-position SL triggered | 2024-09-10 11:45:00 | 1204.07 | 1203.76 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 11:00:00 | 1210.40 | 1214.41 | 0.00 | ORB-short ORB[1215.10,1227.00] vol=3.7x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-09-13 11:30:00 | 1212.83 | 1213.46 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 11:00:00 | 1222.50 | 1217.08 | 0.00 | ORB-long ORB[1207.00,1220.00] vol=1.6x ATR=2.50 |
| Stop hit — per-position SL triggered | 2024-09-16 11:05:00 | 1220.00 | 1217.34 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:50:00 | 1213.35 | 1215.65 | 0.00 | ORB-short ORB[1215.70,1223.40] vol=2.3x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:00:00 | 1210.11 | 1214.78 | 0.00 | T1 1.5R @ 1210.11 |
| Target hit | 2024-09-18 15:20:00 | 1200.90 | 1203.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2024-09-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 10:00:00 | 1222.30 | 1216.59 | 0.00 | ORB-long ORB[1204.05,1215.35] vol=2.0x ATR=3.21 |
| Stop hit — per-position SL triggered | 2024-09-19 10:05:00 | 1219.09 | 1217.05 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 1198.75 | 1203.23 | 0.00 | ORB-short ORB[1203.00,1216.00] vol=2.3x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 09:40:00 | 1194.29 | 1202.05 | 0.00 | T1 1.5R @ 1194.29 |
| Target hit | 2024-09-25 15:10:00 | 1190.25 | 1190.15 | 0.00 | Trail-exit close>VWAP |

### Cycle 36 — SELL (started 2024-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:10:00 | 1194.10 | 1200.84 | 0.00 | ORB-short ORB[1196.95,1203.25] vol=1.7x ATR=2.48 |
| Stop hit — per-position SL triggered | 2024-10-01 11:20:00 | 1196.58 | 1200.38 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:30:00 | 1124.80 | 1132.95 | 0.00 | ORB-short ORB[1130.40,1142.65] vol=2.0x ATR=3.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:50:00 | 1119.10 | 1131.47 | 0.00 | T1 1.5R @ 1119.10 |
| Stop hit — per-position SL triggered | 2024-10-07 11:25:00 | 1124.80 | 1129.89 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 10:50:00 | 1114.00 | 1116.27 | 0.00 | ORB-short ORB[1115.60,1132.00] vol=1.9x ATR=4.04 |
| Stop hit — per-position SL triggered | 2024-10-09 11:20:00 | 1118.04 | 1116.20 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:35:00 | 1110.50 | 1111.80 | 0.00 | ORB-short ORB[1111.60,1120.00] vol=2.7x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:50:00 | 1107.37 | 1111.44 | 0.00 | T1 1.5R @ 1107.37 |
| Target hit | 2024-10-16 12:40:00 | 1108.85 | 1108.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — BUY (started 2024-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 09:35:00 | 1115.60 | 1113.89 | 0.00 | ORB-long ORB[1109.70,1115.40] vol=1.5x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-10-17 09:40:00 | 1112.85 | 1113.77 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:45:00 | 979.45 | 992.37 | 0.00 | ORB-short ORB[992.00,1003.55] vol=3.1x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-10-25 10:55:00 | 982.65 | 991.41 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 10:20:00 | 975.50 | 970.27 | 0.00 | ORB-long ORB[964.00,975.25] vol=2.9x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 11:55:00 | 981.25 | 973.27 | 0.00 | T1 1.5R @ 981.25 |
| Stop hit — per-position SL triggered | 2024-10-28 12:45:00 | 975.50 | 974.12 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:35:00 | 1000.65 | 994.78 | 0.00 | ORB-long ORB[988.15,995.90] vol=1.6x ATR=3.04 |
| Stop hit — per-position SL triggered | 2024-10-30 09:40:00 | 997.61 | 995.14 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 10:50:00 | 990.45 | 985.49 | 0.00 | ORB-long ORB[975.40,989.00] vol=1.6x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 12:15:00 | 994.01 | 987.83 | 0.00 | T1 1.5R @ 994.01 |
| Stop hit — per-position SL triggered | 2024-11-08 12:55:00 | 990.45 | 989.23 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 10:45:00 | 917.10 | 920.64 | 0.00 | ORB-short ORB[920.00,932.00] vol=1.9x ATR=2.72 |
| Stop hit — per-position SL triggered | 2024-11-18 10:55:00 | 919.82 | 920.46 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:35:00 | 901.95 | 907.47 | 0.00 | ORB-short ORB[905.50,918.95] vol=1.6x ATR=2.79 |
| Stop hit — per-position SL triggered | 2024-11-21 09:40:00 | 904.74 | 906.37 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 971.80 | 965.18 | 0.00 | ORB-long ORB[957.60,969.95] vol=2.2x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-11-28 09:35:00 | 969.26 | 965.73 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:55:00 | 959.05 | 952.02 | 0.00 | ORB-long ORB[941.50,948.60] vol=1.7x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-11-29 11:25:00 | 956.22 | 952.79 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 952.35 | 957.89 | 0.00 | ORB-short ORB[957.05,965.90] vol=2.0x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 11:20:00 | 948.94 | 956.13 | 0.00 | T1 1.5R @ 948.94 |
| Stop hit — per-position SL triggered | 2024-12-05 12:00:00 | 952.35 | 954.43 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:15:00 | 972.00 | 970.07 | 0.00 | ORB-long ORB[964.10,970.75] vol=1.9x ATR=2.24 |
| Stop hit — per-position SL triggered | 2024-12-06 11:45:00 | 969.76 | 970.13 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:35:00 | 923.55 | 929.06 | 0.00 | ORB-short ORB[928.00,938.45] vol=2.3x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:45:00 | 920.11 | 925.05 | 0.00 | T1 1.5R @ 920.11 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 923.55 | 924.97 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 920.55 | 928.92 | 0.00 | ORB-short ORB[926.40,933.85] vol=1.8x ATR=2.30 |
| Stop hit — per-position SL triggered | 2024-12-16 11:05:00 | 922.85 | 928.76 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:20:00 | 909.35 | 913.09 | 0.00 | ORB-short ORB[913.50,922.00] vol=1.9x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-12-17 11:45:00 | 911.01 | 911.80 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:30:00 | 916.00 | 910.64 | 0.00 | ORB-long ORB[901.10,913.50] vol=4.2x ATR=2.49 |
| Stop hit — per-position SL triggered | 2024-12-19 09:35:00 | 913.51 | 911.13 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 10:55:00 | 901.40 | 896.65 | 0.00 | ORB-long ORB[894.45,900.00] vol=1.5x ATR=3.09 |
| Stop hit — per-position SL triggered | 2024-12-23 12:05:00 | 898.31 | 898.95 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:35:00 | 906.65 | 903.49 | 0.00 | ORB-long ORB[899.00,904.00] vol=1.6x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:30:00 | 908.99 | 904.88 | 0.00 | T1 1.5R @ 908.99 |
| Stop hit — per-position SL triggered | 2024-12-27 11:45:00 | 906.65 | 905.02 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:45:00 | 909.30 | 903.92 | 0.00 | ORB-long ORB[900.55,907.95] vol=1.6x ATR=2.04 |
| Stop hit — per-position SL triggered | 2024-12-30 13:30:00 | 907.26 | 907.01 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 11:15:00 | 904.95 | 900.57 | 0.00 | ORB-long ORB[893.10,904.65] vol=1.7x ATR=2.98 |
| Stop hit — per-position SL triggered | 2024-12-31 11:45:00 | 901.97 | 900.76 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-01-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:25:00 | 918.65 | 915.72 | 0.00 | ORB-long ORB[911.00,918.20] vol=3.6x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 14:40:00 | 922.17 | 918.86 | 0.00 | T1 1.5R @ 922.17 |
| Stop hit — per-position SL triggered | 2025-01-01 15:05:00 | 918.65 | 919.00 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:40:00 | 934.75 | 930.45 | 0.00 | ORB-long ORB[926.00,932.95] vol=1.7x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-01-03 11:00:00 | 932.53 | 931.81 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 10:50:00 | 938.90 | 935.63 | 0.00 | ORB-long ORB[930.55,937.95] vol=1.9x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-01-06 11:05:00 | 936.05 | 935.79 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:05:00 | 961.60 | 959.09 | 0.00 | ORB-long ORB[955.00,961.40] vol=1.6x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:15:00 | 965.49 | 959.96 | 0.00 | T1 1.5R @ 965.49 |
| Target hit | 2025-01-09 11:25:00 | 971.95 | 972.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — BUY (started 2025-01-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-10 11:00:00 | 975.60 | 966.01 | 0.00 | ORB-long ORB[960.05,968.45] vol=2.6x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-01-10 11:15:00 | 972.67 | 967.03 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 10:55:00 | 973.70 | 967.46 | 0.00 | ORB-long ORB[965.45,973.00] vol=1.9x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-01-14 12:10:00 | 970.82 | 970.39 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-01-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 11:00:00 | 948.25 | 946.54 | 0.00 | ORB-long ORB[935.35,943.95] vol=2.4x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 11:45:00 | 951.97 | 946.97 | 0.00 | T1 1.5R @ 951.97 |
| Stop hit — per-position SL triggered | 2025-01-17 11:55:00 | 948.25 | 947.06 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:40:00 | 971.35 | 968.15 | 0.00 | ORB-long ORB[957.90,970.95] vol=2.8x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:55:00 | 975.26 | 970.79 | 0.00 | T1 1.5R @ 975.26 |
| Stop hit — per-position SL triggered | 2025-01-21 10:00:00 | 971.35 | 971.11 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:30:00 | 972.10 | 968.21 | 0.00 | ORB-long ORB[961.15,970.00] vol=1.6x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 09:35:00 | 976.27 | 969.68 | 0.00 | T1 1.5R @ 976.27 |
| Target hit | 2025-01-23 12:50:00 | 984.00 | 984.29 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — BUY (started 2025-01-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 10:50:00 | 996.40 | 992.88 | 0.00 | ORB-long ORB[987.20,996.25] vol=2.6x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:35:00 | 1001.20 | 994.31 | 0.00 | T1 1.5R @ 1001.20 |
| Stop hit — per-position SL triggered | 2025-01-24 13:30:00 | 996.40 | 996.25 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:00:00 | 982.80 | 991.47 | 0.00 | ORB-short ORB[987.10,996.90] vol=2.5x ATR=3.55 |
| Stop hit — per-position SL triggered | 2025-01-27 10:05:00 | 986.35 | 991.21 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 973.25 | 966.67 | 0.00 | ORB-long ORB[959.45,969.00] vol=1.6x ATR=3.60 |
| Stop hit — per-position SL triggered | 2025-01-30 09:40:00 | 969.65 | 967.78 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-02-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:05:00 | 1027.15 | 1035.87 | 0.00 | ORB-short ORB[1030.85,1043.80] vol=2.3x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 10:25:00 | 1021.24 | 1032.16 | 0.00 | T1 1.5R @ 1021.24 |
| Stop hit — per-position SL triggered | 2025-02-04 10:40:00 | 1027.15 | 1031.44 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-02-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 11:05:00 | 1018.40 | 1019.00 | 0.00 | ORB-short ORB[1021.50,1035.25] vol=1.7x ATR=3.03 |
| Stop hit — per-position SL triggered | 2025-02-05 11:20:00 | 1021.43 | 1019.27 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-02-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-10 10:10:00 | 1031.05 | 1029.43 | 0.00 | ORB-long ORB[1015.80,1028.90] vol=1.7x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 11:10:00 | 1036.61 | 1031.22 | 0.00 | T1 1.5R @ 1036.61 |
| Stop hit — per-position SL triggered | 2025-02-10 11:20:00 | 1031.05 | 1031.46 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-02-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-12 10:35:00 | 1027.35 | 1016.87 | 0.00 | ORB-long ORB[1011.30,1019.00] vol=1.5x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-02-12 10:40:00 | 1023.85 | 1017.23 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-02-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-14 10:00:00 | 1036.00 | 1028.48 | 0.00 | ORB-long ORB[1018.30,1027.90] vol=1.9x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-02-14 10:25:00 | 1032.61 | 1031.71 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-17 11:05:00 | 1011.75 | 1014.44 | 0.00 | ORB-short ORB[1012.15,1025.95] vol=6.1x ATR=3.33 |
| Stop hit — per-position SL triggered | 2025-02-17 11:25:00 | 1015.08 | 1014.10 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-02-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 11:10:00 | 1009.60 | 1005.44 | 0.00 | ORB-long ORB[1000.50,1008.90] vol=2.0x ATR=2.76 |
| Stop hit — per-position SL triggered | 2025-02-21 11:40:00 | 1006.84 | 1006.95 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:50:00 | 1009.75 | 1004.77 | 0.00 | ORB-long ORB[993.15,1007.30] vol=2.5x ATR=2.64 |
| Stop hit — per-position SL triggered | 2025-02-25 09:55:00 | 1007.11 | 1005.00 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2025-02-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 11:00:00 | 991.95 | 995.08 | 0.00 | ORB-short ORB[992.80,1000.60] vol=3.5x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 11:40:00 | 987.10 | 993.67 | 0.00 | T1 1.5R @ 987.10 |
| Target hit | 2025-02-28 15:20:00 | 962.50 | 969.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2025-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:15:00 | 964.35 | 960.62 | 0.00 | ORB-long ORB[950.15,962.15] vol=2.0x ATR=2.48 |
| Stop hit — per-position SL triggered | 2025-03-07 10:20:00 | 961.87 | 960.67 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-03-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:55:00 | 954.55 | 950.10 | 0.00 | ORB-long ORB[945.70,950.75] vol=2.3x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-03-19 11:35:00 | 952.88 | 952.88 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-03-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 11:00:00 | 960.00 | 967.10 | 0.00 | ORB-short ORB[966.05,976.30] vol=2.1x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-03-26 12:10:00 | 962.30 | 965.06 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-03-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:05:00 | 970.50 | 966.12 | 0.00 | ORB-long ORB[952.50,964.85] vol=4.7x ATR=3.09 |
| Stop hit — per-position SL triggered | 2025-03-27 11:05:00 | 967.41 | 968.29 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 11:15:00 | 1048.35 | 1050.41 | 0.00 | ORB-short ORB[1049.05,1059.65] vol=3.8x ATR=3.43 |
| Stop hit — per-position SL triggered | 2025-04-08 11:35:00 | 1051.78 | 1050.35 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:15:00 | 1110.00 | 1101.62 | 0.00 | ORB-long ORB[1094.50,1106.90] vol=2.4x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:30:00 | 1114.29 | 1102.83 | 0.00 | T1 1.5R @ 1114.29 |
| Target hit | 2025-04-17 15:20:00 | 1119.80 | 1117.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2025-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:40:00 | 1133.80 | 1125.69 | 0.00 | ORB-long ORB[1118.20,1128.10] vol=1.9x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:25:00 | 1139.31 | 1131.37 | 0.00 | T1 1.5R @ 1139.31 |
| Stop hit — per-position SL triggered | 2025-04-22 10:30:00 | 1133.80 | 1131.46 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2025-04-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:10:00 | 1153.00 | 1158.61 | 0.00 | ORB-short ORB[1157.30,1165.00] vol=1.6x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:25:00 | 1147.45 | 1157.02 | 0.00 | T1 1.5R @ 1147.45 |
| Target hit | 2025-04-25 12:55:00 | 1151.60 | 1150.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 88 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-30 11:15:00 | 1162.80 | 1171.74 | 0.00 | ORB-short ORB[1168.90,1180.50] vol=1.6x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 13:55:00 | 1158.71 | 1168.87 | 0.00 | T1 1.5R @ 1158.71 |
| Stop hit — per-position SL triggered | 2025-04-30 14:40:00 | 1162.80 | 1167.76 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:00:00 | 1172.90 | 1168.19 | 0.00 | ORB-long ORB[1153.00,1170.30] vol=1.8x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-05-05 11:05:00 | 1170.22 | 1168.40 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-21 11:15:00 | 1082.59 | 2024-05-21 11:20:00 | 1080.51 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-05-22 11:10:00 | 1092.62 | 2024-05-22 11:25:00 | 1096.73 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-05-22 11:10:00 | 1092.62 | 2024-05-22 15:20:00 | 1106.94 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2024-05-24 09:35:00 | 1088.17 | 2024-05-24 09:45:00 | 1091.48 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-28 10:25:00 | 1071.78 | 2024-05-28 11:05:00 | 1074.20 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-05-30 11:15:00 | 1043.28 | 2024-05-30 11:45:00 | 1045.66 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-11 10:25:00 | 1125.81 | 2024-06-11 10:35:00 | 1129.65 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-06-11 10:25:00 | 1125.81 | 2024-06-11 12:00:00 | 1125.81 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 09:30:00 | 1069.41 | 2024-06-27 09:35:00 | 1071.75 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-07-03 10:15:00 | 1112.52 | 2024-07-03 10:20:00 | 1109.38 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-08 09:50:00 | 1139.68 | 2024-07-08 10:15:00 | 1144.38 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-08 09:50:00 | 1139.68 | 2024-07-08 11:25:00 | 1139.68 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 11:05:00 | 1122.79 | 2024-07-10 11:50:00 | 1126.68 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-11 11:10:00 | 1126.05 | 2024-07-11 11:40:00 | 1128.47 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-16 09:55:00 | 1143.09 | 2024-07-16 10:15:00 | 1147.59 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-16 09:55:00 | 1143.09 | 2024-07-16 15:20:00 | 1160.97 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2024-07-22 10:15:00 | 1186.40 | 2024-07-22 10:35:00 | 1183.06 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-23 10:25:00 | 1202.06 | 2024-07-23 10:30:00 | 1198.68 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-25 11:00:00 | 1206.40 | 2024-07-25 11:40:00 | 1200.97 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-25 11:00:00 | 1206.40 | 2024-07-25 13:10:00 | 1206.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-30 10:20:00 | 1194.85 | 2024-07-30 10:30:00 | 1198.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-01 10:55:00 | 1202.75 | 2024-08-01 11:40:00 | 1207.14 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-08-01 10:55:00 | 1202.75 | 2024-08-01 13:10:00 | 1202.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 10:05:00 | 1175.35 | 2024-08-08 10:15:00 | 1178.41 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-12 09:40:00 | 1172.25 | 2024-08-12 09:55:00 | 1168.45 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-08-12 09:40:00 | 1172.25 | 2024-08-12 11:05:00 | 1172.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-14 10:05:00 | 1163.15 | 2024-08-14 10:40:00 | 1166.75 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-19 10:25:00 | 1173.40 | 2024-08-19 11:10:00 | 1176.49 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-20 11:00:00 | 1171.50 | 2024-08-20 11:35:00 | 1168.77 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2024-08-20 11:00:00 | 1171.50 | 2024-08-20 13:35:00 | 1171.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 09:30:00 | 1194.90 | 2024-08-22 09:40:00 | 1198.54 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-08-22 09:30:00 | 1194.90 | 2024-08-22 09:45:00 | 1194.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-23 11:15:00 | 1205.00 | 2024-08-23 11:35:00 | 1201.06 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-08-23 11:15:00 | 1205.00 | 2024-08-23 11:50:00 | 1205.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 10:50:00 | 1207.25 | 2024-08-26 11:20:00 | 1210.72 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-08-26 10:50:00 | 1207.25 | 2024-08-26 15:20:00 | 1219.60 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2024-08-28 10:45:00 | 1203.40 | 2024-08-28 10:50:00 | 1205.50 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-08-29 09:50:00 | 1205.70 | 2024-08-29 10:15:00 | 1203.49 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-09-03 10:15:00 | 1212.00 | 2024-09-03 10:20:00 | 1215.57 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-09-03 10:15:00 | 1212.00 | 2024-09-03 11:15:00 | 1212.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-09 10:15:00 | 1192.85 | 2024-09-09 10:45:00 | 1189.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-10 11:10:00 | 1206.60 | 2024-09-10 11:45:00 | 1204.07 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-09-13 11:00:00 | 1210.40 | 2024-09-13 11:30:00 | 1212.83 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-09-16 11:00:00 | 1222.50 | 2024-09-16 11:05:00 | 1220.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-09-18 10:50:00 | 1213.35 | 2024-09-18 11:00:00 | 1210.11 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-09-18 10:50:00 | 1213.35 | 2024-09-18 15:20:00 | 1200.90 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2024-09-19 10:00:00 | 1222.30 | 2024-09-19 10:05:00 | 1219.09 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-25 09:30:00 | 1198.75 | 2024-09-25 09:40:00 | 1194.29 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-25 09:30:00 | 1198.75 | 2024-09-25 15:10:00 | 1190.25 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2024-10-01 11:10:00 | 1194.10 | 2024-10-01 11:20:00 | 1196.58 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-10-07 10:30:00 | 1124.80 | 2024-10-07 10:50:00 | 1119.10 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-07 10:30:00 | 1124.80 | 2024-10-07 11:25:00 | 1124.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-09 10:50:00 | 1114.00 | 2024-10-09 11:20:00 | 1118.04 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-16 10:35:00 | 1110.50 | 2024-10-16 10:50:00 | 1107.37 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-10-16 10:35:00 | 1110.50 | 2024-10-16 12:40:00 | 1108.85 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2024-10-17 09:35:00 | 1115.60 | 2024-10-17 09:40:00 | 1112.85 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-25 10:45:00 | 979.45 | 2024-10-25 10:55:00 | 982.65 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-10-28 10:20:00 | 975.50 | 2024-10-28 11:55:00 | 981.25 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-10-28 10:20:00 | 975.50 | 2024-10-28 12:45:00 | 975.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-30 09:35:00 | 1000.65 | 2024-10-30 09:40:00 | 997.61 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-11-08 10:50:00 | 990.45 | 2024-11-08 12:15:00 | 994.01 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-11-08 10:50:00 | 990.45 | 2024-11-08 12:55:00 | 990.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-18 10:45:00 | 917.10 | 2024-11-18 10:55:00 | 919.82 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-11-21 09:35:00 | 901.95 | 2024-11-21 09:40:00 | 904.74 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-28 09:30:00 | 971.80 | 2024-11-28 09:35:00 | 969.26 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-11-29 10:55:00 | 959.05 | 2024-11-29 11:25:00 | 956.22 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-05 10:55:00 | 952.35 | 2024-12-05 11:20:00 | 948.94 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-05 10:55:00 | 952.35 | 2024-12-05 12:00:00 | 952.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-06 11:15:00 | 972.00 | 2024-12-06 11:45:00 | 969.76 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-12 09:35:00 | 923.55 | 2024-12-12 09:45:00 | 920.11 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-12 09:35:00 | 923.55 | 2024-12-12 09:50:00 | 923.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 11:00:00 | 920.55 | 2024-12-16 11:05:00 | 922.85 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-17 10:20:00 | 909.35 | 2024-12-17 11:45:00 | 911.01 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-12-19 09:30:00 | 916.00 | 2024-12-19 09:35:00 | 913.51 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-23 10:55:00 | 901.40 | 2024-12-23 12:05:00 | 898.31 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-27 10:35:00 | 906.65 | 2024-12-27 11:30:00 | 908.99 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-12-27 10:35:00 | 906.65 | 2024-12-27 11:45:00 | 906.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 10:45:00 | 909.30 | 2024-12-30 13:30:00 | 907.26 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-12-31 11:15:00 | 904.95 | 2024-12-31 11:45:00 | 901.97 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-01 10:25:00 | 918.65 | 2025-01-01 14:40:00 | 922.17 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-01-01 10:25:00 | 918.65 | 2025-01-01 15:05:00 | 918.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-03 10:40:00 | 934.75 | 2025-01-03 11:00:00 | 932.53 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-06 10:50:00 | 938.90 | 2025-01-06 11:05:00 | 936.05 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-09 10:05:00 | 961.60 | 2025-01-09 10:15:00 | 965.49 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-09 10:05:00 | 961.60 | 2025-01-09 11:25:00 | 971.95 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2025-01-10 11:00:00 | 975.60 | 2025-01-10 11:15:00 | 972.67 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-14 10:55:00 | 973.70 | 2025-01-14 12:10:00 | 970.82 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-17 11:00:00 | 948.25 | 2025-01-17 11:45:00 | 951.97 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-01-17 11:00:00 | 948.25 | 2025-01-17 11:55:00 | 948.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-21 09:40:00 | 971.35 | 2025-01-21 09:55:00 | 975.26 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-21 09:40:00 | 971.35 | 2025-01-21 10:00:00 | 971.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 09:30:00 | 972.10 | 2025-01-23 09:35:00 | 976.27 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-01-23 09:30:00 | 972.10 | 2025-01-23 12:50:00 | 984.00 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2025-01-24 10:50:00 | 996.40 | 2025-01-24 11:35:00 | 1001.20 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-01-24 10:50:00 | 996.40 | 2025-01-24 13:30:00 | 996.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 10:00:00 | 982.80 | 2025-01-27 10:05:00 | 986.35 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-30 09:30:00 | 973.25 | 2025-01-30 09:40:00 | 969.65 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-02-04 10:05:00 | 1027.15 | 2025-02-04 10:25:00 | 1021.24 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-02-04 10:05:00 | 1027.15 | 2025-02-04 10:40:00 | 1027.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-05 11:05:00 | 1018.40 | 2025-02-05 11:20:00 | 1021.43 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-02-10 10:10:00 | 1031.05 | 2025-02-10 11:10:00 | 1036.61 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-02-10 10:10:00 | 1031.05 | 2025-02-10 11:20:00 | 1031.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-12 10:35:00 | 1027.35 | 2025-02-12 10:40:00 | 1023.85 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-02-14 10:00:00 | 1036.00 | 2025-02-14 10:25:00 | 1032.61 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-17 11:05:00 | 1011.75 | 2025-02-17 11:25:00 | 1015.08 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-21 11:10:00 | 1009.60 | 2025-02-21 11:40:00 | 1006.84 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-02-25 09:50:00 | 1009.75 | 2025-02-25 09:55:00 | 1007.11 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-02-28 11:00:00 | 991.95 | 2025-02-28 11:40:00 | 987.10 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-02-28 11:00:00 | 991.95 | 2025-02-28 15:20:00 | 962.50 | TARGET_HIT | 0.50 | 2.97% |
| BUY | retest1 | 2025-03-07 10:15:00 | 964.35 | 2025-03-07 10:20:00 | 961.87 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-19 10:55:00 | 954.55 | 2025-03-19 11:35:00 | 952.88 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-03-26 11:00:00 | 960.00 | 2025-03-26 12:10:00 | 962.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-03-27 10:05:00 | 970.50 | 2025-03-27 11:05:00 | 967.41 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-08 11:15:00 | 1048.35 | 2025-04-08 11:35:00 | 1051.78 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-17 11:15:00 | 1110.00 | 2025-04-17 11:30:00 | 1114.29 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-04-17 11:15:00 | 1110.00 | 2025-04-17 15:20:00 | 1119.80 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2025-04-22 09:40:00 | 1133.80 | 2025-04-22 10:25:00 | 1139.31 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-22 09:40:00 | 1133.80 | 2025-04-22 10:30:00 | 1133.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 10:10:00 | 1153.00 | 2025-04-25 10:25:00 | 1147.45 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-04-25 10:10:00 | 1153.00 | 2025-04-25 12:55:00 | 1151.60 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2025-04-30 11:15:00 | 1162.80 | 2025-04-30 13:55:00 | 1158.71 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-04-30 11:15:00 | 1162.80 | 2025-04-30 14:40:00 | 1162.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 11:00:00 | 1172.90 | 2025-05-05 11:05:00 | 1170.22 | STOP_HIT | 1.00 | -0.23% |
