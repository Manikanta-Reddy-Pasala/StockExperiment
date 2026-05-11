# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55356 bars)
- **Last close:** 1198.00
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
| ENTRY1 | 79 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 11 |
| STOP_HIT | 68 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 112 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 68
- **Target hits / Stop hits / Partials:** 11 / 68 / 33
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 11.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 27 | 41.5% | 7 | 38 | 20 | 0.14% | 9.1% |
| BUY @ 2nd Alert (retest1) | 65 | 27 | 41.5% | 7 | 38 | 20 | 0.14% | 9.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 17 | 36.2% | 4 | 30 | 13 | 0.04% | 1.9% |
| SELL @ 2nd Alert (retest1) | 47 | 17 | 36.2% | 4 | 30 | 13 | 0.04% | 1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 112 | 44 | 39.3% | 11 | 68 | 33 | 0.10% | 11.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 10:15:00 | 1102.50 | 1099.02 | 0.00 | ORB-long ORB[1092.80,1102.30] vol=2.7x ATR=2.40 |
| Stop hit — per-position SL triggered | 2023-05-15 10:20:00 | 1100.10 | 1099.19 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-16 11:15:00 | 1097.45 | 1101.34 | 0.00 | ORB-short ORB[1099.45,1104.90] vol=4.0x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-16 14:05:00 | 1094.31 | 1099.21 | 0.00 | T1 1.5R @ 1094.31 |
| Target hit | 2023-05-16 15:20:00 | 1090.00 | 1097.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2023-05-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-19 10:35:00 | 1101.00 | 1096.00 | 0.00 | ORB-long ORB[1082.45,1094.55] vol=1.7x ATR=2.79 |
| Stop hit — per-position SL triggered | 2023-05-19 10:45:00 | 1098.21 | 1096.46 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 10:30:00 | 1111.85 | 1107.95 | 0.00 | ORB-long ORB[1098.00,1105.15] vol=1.7x ATR=2.51 |
| Stop hit — per-position SL triggered | 2023-05-24 10:55:00 | 1109.34 | 1108.57 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 10:45:00 | 1130.95 | 1124.91 | 0.00 | ORB-long ORB[1114.00,1126.95] vol=1.6x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-26 11:35:00 | 1134.66 | 1127.93 | 0.00 | T1 1.5R @ 1134.66 |
| Target hit | 2023-05-26 15:20:00 | 1138.25 | 1133.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-01 11:15:00 | 1137.65 | 1147.42 | 0.00 | ORB-short ORB[1145.10,1155.15] vol=1.6x ATR=1.94 |
| Stop hit — per-position SL triggered | 2023-06-01 11:20:00 | 1139.59 | 1147.25 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:55:00 | 1123.00 | 1126.89 | 0.00 | ORB-short ORB[1126.00,1131.00] vol=1.6x ATR=2.12 |
| Stop hit — per-position SL triggered | 2023-06-09 12:05:00 | 1125.12 | 1123.78 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 10:45:00 | 1133.45 | 1125.71 | 0.00 | ORB-long ORB[1111.50,1124.45] vol=2.5x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 11:25:00 | 1136.66 | 1128.73 | 0.00 | T1 1.5R @ 1136.66 |
| Stop hit — per-position SL triggered | 2023-06-12 12:25:00 | 1133.45 | 1130.54 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 11:05:00 | 1133.05 | 1136.91 | 0.00 | ORB-short ORB[1136.50,1144.95] vol=1.5x ATR=1.61 |
| Stop hit — per-position SL triggered | 2023-06-13 11:25:00 | 1134.66 | 1136.54 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 10:50:00 | 1129.00 | 1130.56 | 0.00 | ORB-short ORB[1131.00,1138.40] vol=1.7x ATR=1.75 |
| Stop hit — per-position SL triggered | 2023-06-14 10:55:00 | 1130.75 | 1130.53 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:35:00 | 1145.25 | 1142.22 | 0.00 | ORB-long ORB[1136.55,1143.90] vol=3.3x ATR=2.29 |
| Stop hit — per-position SL triggered | 2023-06-15 09:40:00 | 1142.96 | 1142.46 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 11:15:00 | 1149.10 | 1139.10 | 0.00 | ORB-long ORB[1132.50,1138.55] vol=1.5x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 11:25:00 | 1152.17 | 1140.78 | 0.00 | T1 1.5R @ 1152.17 |
| Target hit | 2023-06-20 15:20:00 | 1169.30 | 1159.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2023-06-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 09:40:00 | 1151.30 | 1153.85 | 0.00 | ORB-short ORB[1151.50,1161.90] vol=2.1x ATR=2.28 |
| Stop hit — per-position SL triggered | 2023-06-23 09:45:00 | 1153.58 | 1153.81 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:30:00 | 1174.65 | 1169.85 | 0.00 | ORB-long ORB[1165.00,1172.10] vol=1.8x ATR=2.96 |
| Stop hit — per-position SL triggered | 2023-06-26 09:50:00 | 1171.69 | 1170.89 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-06 09:35:00 | 1185.00 | 1188.66 | 0.00 | ORB-short ORB[1188.05,1198.00] vol=2.6x ATR=2.73 |
| Stop hit — per-position SL triggered | 2023-07-06 10:25:00 | 1187.73 | 1185.96 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 09:35:00 | 1126.00 | 1121.42 | 0.00 | ORB-long ORB[1111.55,1124.00] vol=1.9x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 09:45:00 | 1132.54 | 1124.43 | 0.00 | T1 1.5R @ 1132.54 |
| Target hit | 2023-07-14 15:20:00 | 1152.30 | 1139.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2023-07-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 10:55:00 | 1170.25 | 1161.22 | 0.00 | ORB-long ORB[1143.65,1158.15] vol=1.6x ATR=3.32 |
| Stop hit — per-position SL triggered | 2023-07-17 11:15:00 | 1166.93 | 1162.26 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 09:40:00 | 1152.45 | 1155.62 | 0.00 | ORB-short ORB[1153.00,1162.80] vol=1.9x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 10:05:00 | 1148.15 | 1153.28 | 0.00 | T1 1.5R @ 1148.15 |
| Stop hit — per-position SL triggered | 2023-07-20 10:30:00 | 1152.45 | 1152.87 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 09:35:00 | 1121.45 | 1118.69 | 0.00 | ORB-long ORB[1115.00,1119.10] vol=2.5x ATR=1.86 |
| Stop hit — per-position SL triggered | 2023-07-27 09:55:00 | 1119.59 | 1119.16 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 10:55:00 | 1107.55 | 1113.02 | 0.00 | ORB-short ORB[1108.60,1118.50] vol=1.9x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 11:00:00 | 1104.85 | 1112.55 | 0.00 | T1 1.5R @ 1104.85 |
| Target hit | 2023-07-28 15:20:00 | 1102.90 | 1105.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2023-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 09:35:00 | 1123.45 | 1117.50 | 0.00 | ORB-long ORB[1111.90,1120.00] vol=1.6x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 10:05:00 | 1128.06 | 1120.74 | 0.00 | T1 1.5R @ 1128.06 |
| Stop hit — per-position SL triggered | 2023-08-01 10:20:00 | 1123.45 | 1121.31 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 09:30:00 | 1142.00 | 1135.93 | 0.00 | ORB-long ORB[1126.00,1139.45] vol=1.9x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 09:50:00 | 1146.14 | 1140.19 | 0.00 | T1 1.5R @ 1146.14 |
| Target hit | 2023-08-04 10:35:00 | 1144.00 | 1144.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2023-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 09:40:00 | 1179.75 | 1176.94 | 0.00 | ORB-long ORB[1170.10,1179.40] vol=1.6x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 10:15:00 | 1184.24 | 1179.23 | 0.00 | T1 1.5R @ 1184.24 |
| Target hit | 2023-08-16 12:20:00 | 1180.10 | 1180.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — SELL (started 2023-08-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 09:50:00 | 1177.20 | 1182.45 | 0.00 | ORB-short ORB[1180.45,1187.00] vol=1.6x ATR=2.57 |
| Stop hit — per-position SL triggered | 2023-08-22 10:05:00 | 1179.77 | 1181.49 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:15:00 | 1162.00 | 1165.42 | 0.00 | ORB-short ORB[1162.55,1169.85] vol=1.7x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 13:10:00 | 1157.32 | 1162.49 | 0.00 | T1 1.5R @ 1157.32 |
| Target hit | 2023-08-25 15:20:00 | 1151.25 | 1159.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2023-08-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-28 09:50:00 | 1143.75 | 1146.44 | 0.00 | ORB-short ORB[1143.85,1159.70] vol=2.2x ATR=3.28 |
| Stop hit — per-position SL triggered | 2023-08-28 10:50:00 | 1147.03 | 1144.81 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 09:35:00 | 1182.20 | 1177.69 | 0.00 | ORB-long ORB[1172.00,1179.50] vol=2.0x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 09:50:00 | 1185.61 | 1179.48 | 0.00 | T1 1.5R @ 1185.61 |
| Stop hit — per-position SL triggered | 2023-08-31 10:30:00 | 1182.20 | 1181.54 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-09-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 10:50:00 | 1244.00 | 1235.85 | 0.00 | ORB-long ORB[1230.30,1239.60] vol=2.5x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 11:10:00 | 1247.71 | 1237.58 | 0.00 | T1 1.5R @ 1247.71 |
| Target hit | 2023-09-07 15:20:00 | 1253.70 | 1248.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2023-09-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:40:00 | 1265.60 | 1259.62 | 0.00 | ORB-long ORB[1254.30,1261.65] vol=1.5x ATR=3.39 |
| Stop hit — per-position SL triggered | 2023-09-08 10:15:00 | 1262.21 | 1263.40 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 09:35:00 | 1282.10 | 1276.73 | 0.00 | ORB-long ORB[1270.40,1279.95] vol=2.0x ATR=3.42 |
| Stop hit — per-position SL triggered | 2023-09-11 09:45:00 | 1278.68 | 1277.66 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 11:15:00 | 1271.80 | 1273.44 | 0.00 | ORB-short ORB[1276.50,1287.00] vol=3.9x ATR=2.70 |
| Stop hit — per-position SL triggered | 2023-09-12 11:30:00 | 1274.50 | 1273.48 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:40:00 | 1290.05 | 1285.20 | 0.00 | ORB-long ORB[1275.60,1285.90] vol=1.9x ATR=2.75 |
| Stop hit — per-position SL triggered | 2023-09-14 09:45:00 | 1287.30 | 1285.76 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 10:45:00 | 1303.55 | 1298.06 | 0.00 | ORB-long ORB[1289.70,1302.00] vol=2.9x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-18 12:20:00 | 1307.78 | 1301.66 | 0.00 | T1 1.5R @ 1307.78 |
| Stop hit — per-position SL triggered | 2023-09-18 14:35:00 | 1303.55 | 1303.53 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 11:15:00 | 1262.40 | 1265.65 | 0.00 | ORB-short ORB[1265.45,1280.00] vol=1.5x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-09-25 11:25:00 | 1264.23 | 1265.45 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-10-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:30:00 | 1268.75 | 1265.61 | 0.00 | ORB-long ORB[1260.00,1267.95] vol=2.0x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 10:00:00 | 1272.83 | 1267.25 | 0.00 | T1 1.5R @ 1272.83 |
| Stop hit — per-position SL triggered | 2023-10-11 10:05:00 | 1268.75 | 1267.30 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 10:05:00 | 1276.00 | 1270.39 | 0.00 | ORB-long ORB[1260.30,1275.85] vol=1.7x ATR=3.70 |
| Stop hit — per-position SL triggered | 2023-10-16 10:50:00 | 1272.30 | 1272.61 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 09:30:00 | 1264.30 | 1268.71 | 0.00 | ORB-short ORB[1266.00,1273.00] vol=1.7x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 10:40:00 | 1259.85 | 1266.34 | 0.00 | T1 1.5R @ 1259.85 |
| Stop hit — per-position SL triggered | 2023-10-31 11:20:00 | 1264.30 | 1265.70 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-11-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 09:30:00 | 1284.70 | 1280.09 | 0.00 | ORB-long ORB[1275.00,1280.00] vol=2.2x ATR=2.93 |
| Stop hit — per-position SL triggered | 2023-11-01 09:35:00 | 1281.77 | 1280.55 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-11-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 11:05:00 | 1271.00 | 1274.86 | 0.00 | ORB-short ORB[1271.10,1277.25] vol=1.6x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 12:20:00 | 1268.26 | 1273.61 | 0.00 | T1 1.5R @ 1268.26 |
| Stop hit — per-position SL triggered | 2023-11-06 14:20:00 | 1271.00 | 1270.76 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-11-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 11:05:00 | 1273.40 | 1277.21 | 0.00 | ORB-short ORB[1276.35,1280.80] vol=1.8x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 11:10:00 | 1270.82 | 1276.45 | 0.00 | T1 1.5R @ 1270.82 |
| Stop hit — per-position SL triggered | 2023-11-08 11:30:00 | 1273.40 | 1275.16 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-11-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 11:05:00 | 1272.55 | 1272.44 | 0.00 | ORB-long ORB[1265.50,1271.85] vol=2.0x ATR=1.76 |
| Stop hit — per-position SL triggered | 2023-11-09 11:10:00 | 1270.79 | 1272.33 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-11-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 10:40:00 | 1262.70 | 1266.62 | 0.00 | ORB-short ORB[1263.30,1270.60] vol=1.5x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-10 10:55:00 | 1259.46 | 1265.86 | 0.00 | T1 1.5R @ 1259.46 |
| Target hit | 2023-11-10 15:20:00 | 1256.30 | 1256.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2023-11-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 09:30:00 | 1277.70 | 1273.35 | 0.00 | ORB-long ORB[1268.40,1276.00] vol=2.5x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 10:05:00 | 1281.55 | 1275.27 | 0.00 | T1 1.5R @ 1281.55 |
| Stop hit — per-position SL triggered | 2023-11-15 10:35:00 | 1277.70 | 1276.38 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:05:00 | 1291.50 | 1283.95 | 0.00 | ORB-long ORB[1278.00,1283.80] vol=1.9x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 10:10:00 | 1294.92 | 1285.85 | 0.00 | T1 1.5R @ 1294.92 |
| Stop hit — per-position SL triggered | 2023-11-16 10:30:00 | 1291.50 | 1288.13 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-11-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 09:40:00 | 1327.90 | 1323.12 | 0.00 | ORB-long ORB[1314.00,1324.45] vol=1.7x ATR=3.18 |
| Stop hit — per-position SL triggered | 2023-11-20 10:05:00 | 1324.72 | 1325.20 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-11-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 10:40:00 | 1328.00 | 1330.69 | 0.00 | ORB-short ORB[1330.10,1335.25] vol=3.8x ATR=2.39 |
| Stop hit — per-position SL triggered | 2023-11-23 12:20:00 | 1330.39 | 1329.09 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-11-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:55:00 | 1315.60 | 1322.18 | 0.00 | ORB-short ORB[1323.50,1332.65] vol=1.8x ATR=2.83 |
| Stop hit — per-position SL triggered | 2023-11-24 10:05:00 | 1318.43 | 1321.20 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-11-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-28 11:10:00 | 1300.75 | 1302.37 | 0.00 | ORB-short ORB[1305.00,1310.00] vol=2.1x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-11-28 13:45:00 | 1302.58 | 1301.48 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:55:00 | 1336.80 | 1332.16 | 0.00 | ORB-long ORB[1321.50,1332.55] vol=1.9x ATR=2.83 |
| Stop hit — per-position SL triggered | 2023-11-29 11:25:00 | 1333.97 | 1332.97 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-12-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 11:05:00 | 1321.55 | 1330.00 | 0.00 | ORB-short ORB[1322.45,1331.00] vol=1.7x ATR=2.92 |
| Stop hit — per-position SL triggered | 2023-12-06 11:40:00 | 1324.47 | 1328.08 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-12-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 10:55:00 | 1462.00 | 1443.84 | 0.00 | ORB-long ORB[1422.00,1431.95] vol=1.9x ATR=4.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 11:10:00 | 1469.12 | 1447.86 | 0.00 | T1 1.5R @ 1469.12 |
| Target hit | 2023-12-15 15:20:00 | 1494.90 | 1470.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2023-12-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 11:00:00 | 1505.00 | 1495.19 | 0.00 | ORB-long ORB[1481.00,1500.00] vol=1.7x ATR=4.69 |
| Stop hit — per-position SL triggered | 2023-12-18 12:25:00 | 1500.31 | 1498.22 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:10:00 | 1441.00 | 1434.21 | 0.00 | ORB-long ORB[1424.05,1438.50] vol=2.8x ATR=4.43 |
| Stop hit — per-position SL triggered | 2023-12-22 10:30:00 | 1436.57 | 1435.09 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 11:05:00 | 1481.25 | 1474.85 | 0.00 | ORB-long ORB[1469.25,1476.00] vol=1.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2023-12-28 11:15:00 | 1478.08 | 1475.09 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-01-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 11:10:00 | 1470.00 | 1462.76 | 0.00 | ORB-long ORB[1455.05,1469.00] vol=2.1x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 11:15:00 | 1473.89 | 1465.71 | 0.00 | T1 1.5R @ 1473.89 |
| Stop hit — per-position SL triggered | 2024-01-01 11:40:00 | 1470.00 | 1466.88 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:00:00 | 1471.00 | 1474.61 | 0.00 | ORB-short ORB[1471.65,1488.80] vol=1.9x ATR=4.44 |
| Stop hit — per-position SL triggered | 2024-01-02 10:20:00 | 1475.44 | 1473.90 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 09:45:00 | 1432.30 | 1437.83 | 0.00 | ORB-short ORB[1433.05,1447.85] vol=2.3x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-01-08 09:50:00 | 1436.58 | 1437.62 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-01-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 09:50:00 | 1472.85 | 1462.95 | 0.00 | ORB-long ORB[1452.35,1467.40] vol=2.2x ATR=5.93 |
| Stop hit — per-position SL triggered | 2024-01-09 11:10:00 | 1466.92 | 1465.85 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-01-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 09:45:00 | 1485.35 | 1478.70 | 0.00 | ORB-long ORB[1466.05,1485.20] vol=2.1x ATR=5.33 |
| Stop hit — per-position SL triggered | 2024-01-10 09:50:00 | 1480.02 | 1479.01 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-01-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 09:35:00 | 1524.60 | 1512.88 | 0.00 | ORB-long ORB[1498.50,1520.00] vol=3.1x ATR=6.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 09:55:00 | 1533.97 | 1518.64 | 0.00 | T1 1.5R @ 1533.97 |
| Stop hit — per-position SL triggered | 2024-01-12 10:55:00 | 1524.60 | 1522.06 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 11:15:00 | 1603.80 | 1594.92 | 0.00 | ORB-long ORB[1584.00,1602.55] vol=4.1x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:40:00 | 1609.66 | 1597.72 | 0.00 | T1 1.5R @ 1609.66 |
| Stop hit — per-position SL triggered | 2024-02-02 11:50:00 | 1603.80 | 1598.50 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-02-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 09:50:00 | 1623.50 | 1635.86 | 0.00 | ORB-short ORB[1636.05,1646.75] vol=1.6x ATR=5.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 10:05:00 | 1614.92 | 1633.13 | 0.00 | T1 1.5R @ 1614.92 |
| Stop hit — per-position SL triggered | 2024-02-09 10:10:00 | 1623.50 | 1632.64 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-12 09:35:00 | 1666.50 | 1658.39 | 0.00 | ORB-long ORB[1639.00,1663.85] vol=2.4x ATR=6.59 |
| Stop hit — per-position SL triggered | 2024-02-12 09:40:00 | 1659.91 | 1658.66 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 09:30:00 | 1657.60 | 1663.26 | 0.00 | ORB-short ORB[1660.70,1672.75] vol=2.4x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 09:45:00 | 1651.60 | 1660.58 | 0.00 | T1 1.5R @ 1651.60 |
| Stop hit — per-position SL triggered | 2024-02-20 10:30:00 | 1657.60 | 1657.21 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 10:55:00 | 1645.20 | 1653.68 | 0.00 | ORB-short ORB[1652.00,1668.85] vol=4.9x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 12:10:00 | 1639.66 | 1650.00 | 0.00 | T1 1.5R @ 1639.66 |
| Stop hit — per-position SL triggered | 2024-02-26 12:50:00 | 1645.20 | 1649.42 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 11:05:00 | 1666.85 | 1660.89 | 0.00 | ORB-long ORB[1648.00,1665.95] vol=1.6x ATR=3.36 |
| Stop hit — per-position SL triggered | 2024-02-27 12:10:00 | 1663.49 | 1662.77 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-02-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 11:15:00 | 1654.70 | 1646.90 | 0.00 | ORB-long ORB[1636.65,1650.90] vol=1.8x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 11:30:00 | 1661.30 | 1649.03 | 0.00 | T1 1.5R @ 1661.30 |
| Stop hit — per-position SL triggered | 2024-02-29 15:00:00 | 1654.70 | 1656.29 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-03-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 10:45:00 | 1653.90 | 1660.52 | 0.00 | ORB-short ORB[1659.05,1679.20] vol=2.6x ATR=4.74 |
| Stop hit — per-position SL triggered | 2024-03-01 11:35:00 | 1658.64 | 1659.47 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 10:50:00 | 1628.50 | 1637.32 | 0.00 | ORB-short ORB[1637.10,1651.25] vol=1.5x ATR=3.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:30:00 | 1622.77 | 1635.65 | 0.00 | T1 1.5R @ 1622.77 |
| Stop hit — per-position SL triggered | 2024-03-13 15:00:00 | 1628.50 | 1626.41 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 11:05:00 | 1575.65 | 1569.26 | 0.00 | ORB-long ORB[1558.95,1572.00] vol=1.5x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-03-27 11:25:00 | 1572.43 | 1569.73 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-04-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:55:00 | 1546.00 | 1533.15 | 0.00 | ORB-long ORB[1518.00,1537.25] vol=1.5x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 10:15:00 | 1552.94 | 1538.93 | 0.00 | T1 1.5R @ 1552.94 |
| Stop hit — per-position SL triggered | 2024-04-03 10:40:00 | 1546.00 | 1541.53 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:50:00 | 1519.95 | 1527.44 | 0.00 | ORB-short ORB[1526.10,1543.65] vol=1.5x ATR=4.21 |
| Stop hit — per-position SL triggered | 2024-04-04 10:15:00 | 1524.16 | 1526.04 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:30:00 | 1557.30 | 1547.62 | 0.00 | ORB-long ORB[1538.05,1552.20] vol=1.7x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-04-09 09:55:00 | 1554.04 | 1553.17 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-04-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:35:00 | 1531.95 | 1535.31 | 0.00 | ORB-short ORB[1532.75,1542.95] vol=2.6x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-04-12 09:40:00 | 1535.24 | 1534.29 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-04-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-19 09:45:00 | 1439.05 | 1444.85 | 0.00 | ORB-short ORB[1442.00,1455.50] vol=2.4x ATR=5.86 |
| Stop hit — per-position SL triggered | 2024-04-19 10:00:00 | 1444.91 | 1444.02 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 09:35:00 | 1487.00 | 1480.62 | 0.00 | ORB-long ORB[1471.10,1482.00] vol=1.7x ATR=3.52 |
| Stop hit — per-position SL triggered | 2024-04-23 09:45:00 | 1483.48 | 1481.45 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-04-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 09:45:00 | 1498.10 | 1493.77 | 0.00 | ORB-long ORB[1479.95,1497.00] vol=1.8x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 10:10:00 | 1504.21 | 1496.70 | 0.00 | T1 1.5R @ 1504.21 |
| Stop hit — per-position SL triggered | 2024-04-25 10:30:00 | 1498.10 | 1497.80 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 11:15:00 | 1348.75 | 1359.71 | 0.00 | ORB-short ORB[1362.00,1370.00] vol=1.6x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 11:35:00 | 1344.74 | 1358.01 | 0.00 | T1 1.5R @ 1344.74 |
| Stop hit — per-position SL triggered | 2024-05-03 14:35:00 | 1348.75 | 1348.28 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 09:35:00 | 1330.00 | 1335.45 | 0.00 | ORB-short ORB[1333.00,1348.90] vol=2.0x ATR=4.01 |
| Stop hit — per-position SL triggered | 2024-05-07 09:45:00 | 1334.01 | 1335.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 10:15:00 | 1102.50 | 2023-05-15 10:20:00 | 1100.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-05-16 11:15:00 | 1097.45 | 2023-05-16 14:05:00 | 1094.31 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-05-16 11:15:00 | 1097.45 | 2023-05-16 15:20:00 | 1090.00 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2023-05-19 10:35:00 | 1101.00 | 2023-05-19 10:45:00 | 1098.21 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-05-24 10:30:00 | 1111.85 | 2023-05-24 10:55:00 | 1109.34 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-05-26 10:45:00 | 1130.95 | 2023-05-26 11:35:00 | 1134.66 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-05-26 10:45:00 | 1130.95 | 2023-05-26 15:20:00 | 1138.25 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2023-06-01 11:15:00 | 1137.65 | 2023-06-01 11:20:00 | 1139.59 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-06-09 09:55:00 | 1123.00 | 2023-06-09 12:05:00 | 1125.12 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-06-12 10:45:00 | 1133.45 | 2023-06-12 11:25:00 | 1136.66 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-06-12 10:45:00 | 1133.45 | 2023-06-12 12:25:00 | 1133.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-13 11:05:00 | 1133.05 | 2023-06-13 11:25:00 | 1134.66 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-06-14 10:50:00 | 1129.00 | 2023-06-14 10:55:00 | 1130.75 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-06-15 09:35:00 | 1145.25 | 2023-06-15 09:40:00 | 1142.96 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-20 11:15:00 | 1149.10 | 2023-06-20 11:25:00 | 1152.17 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-06-20 11:15:00 | 1149.10 | 2023-06-20 15:20:00 | 1169.30 | TARGET_HIT | 0.50 | 1.76% |
| SELL | retest1 | 2023-06-23 09:40:00 | 1151.30 | 2023-06-23 09:45:00 | 1153.58 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-26 09:30:00 | 1174.65 | 2023-06-26 09:50:00 | 1171.69 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-06 09:35:00 | 1185.00 | 2023-07-06 10:25:00 | 1187.73 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-14 09:35:00 | 1126.00 | 2023-07-14 09:45:00 | 1132.54 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-07-14 09:35:00 | 1126.00 | 2023-07-14 15:20:00 | 1152.30 | TARGET_HIT | 0.50 | 2.34% |
| BUY | retest1 | 2023-07-17 10:55:00 | 1170.25 | 2023-07-17 11:15:00 | 1166.93 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-07-20 09:40:00 | 1152.45 | 2023-07-20 10:05:00 | 1148.15 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-07-20 09:40:00 | 1152.45 | 2023-07-20 10:30:00 | 1152.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-27 09:35:00 | 1121.45 | 2023-07-27 09:55:00 | 1119.59 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-07-28 10:55:00 | 1107.55 | 2023-07-28 11:00:00 | 1104.85 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-07-28 10:55:00 | 1107.55 | 2023-07-28 15:20:00 | 1102.90 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2023-08-01 09:35:00 | 1123.45 | 2023-08-01 10:05:00 | 1128.06 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-08-01 09:35:00 | 1123.45 | 2023-08-01 10:20:00 | 1123.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-04 09:30:00 | 1142.00 | 2023-08-04 09:50:00 | 1146.14 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-08-04 09:30:00 | 1142.00 | 2023-08-04 10:35:00 | 1144.00 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2023-08-16 09:40:00 | 1179.75 | 2023-08-16 10:15:00 | 1184.24 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-08-16 09:40:00 | 1179.75 | 2023-08-16 12:20:00 | 1180.10 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2023-08-22 09:50:00 | 1177.20 | 2023-08-22 10:05:00 | 1179.77 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-08-25 10:15:00 | 1162.00 | 2023-08-25 13:10:00 | 1157.32 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-08-25 10:15:00 | 1162.00 | 2023-08-25 15:20:00 | 1151.25 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2023-08-28 09:50:00 | 1143.75 | 2023-08-28 10:50:00 | 1147.03 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-08-31 09:35:00 | 1182.20 | 2023-08-31 09:50:00 | 1185.61 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-08-31 09:35:00 | 1182.20 | 2023-08-31 10:30:00 | 1182.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-07 10:50:00 | 1244.00 | 2023-09-07 11:10:00 | 1247.71 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-09-07 10:50:00 | 1244.00 | 2023-09-07 15:20:00 | 1253.70 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2023-09-08 09:40:00 | 1265.60 | 2023-09-08 10:15:00 | 1262.21 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-09-11 09:35:00 | 1282.10 | 2023-09-11 09:45:00 | 1278.68 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-09-12 11:15:00 | 1271.80 | 2023-09-12 11:30:00 | 1274.50 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-09-14 09:40:00 | 1290.05 | 2023-09-14 09:45:00 | 1287.30 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-09-18 10:45:00 | 1303.55 | 2023-09-18 12:20:00 | 1307.78 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-09-18 10:45:00 | 1303.55 | 2023-09-18 14:35:00 | 1303.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-25 11:15:00 | 1262.40 | 2023-09-25 11:25:00 | 1264.23 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-10-11 09:30:00 | 1268.75 | 2023-10-11 10:00:00 | 1272.83 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-10-11 09:30:00 | 1268.75 | 2023-10-11 10:05:00 | 1268.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-16 10:05:00 | 1276.00 | 2023-10-16 10:50:00 | 1272.30 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-10-31 09:30:00 | 1264.30 | 2023-10-31 10:40:00 | 1259.85 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-10-31 09:30:00 | 1264.30 | 2023-10-31 11:20:00 | 1264.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-01 09:30:00 | 1284.70 | 2023-11-01 09:35:00 | 1281.77 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-11-06 11:05:00 | 1271.00 | 2023-11-06 12:20:00 | 1268.26 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-11-06 11:05:00 | 1271.00 | 2023-11-06 14:20:00 | 1271.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-08 11:05:00 | 1273.40 | 2023-11-08 11:10:00 | 1270.82 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2023-11-08 11:05:00 | 1273.40 | 2023-11-08 11:30:00 | 1273.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-09 11:05:00 | 1272.55 | 2023-11-09 11:10:00 | 1270.79 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-11-10 10:40:00 | 1262.70 | 2023-11-10 10:55:00 | 1259.46 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-11-10 10:40:00 | 1262.70 | 2023-11-10 15:20:00 | 1256.30 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2023-11-15 09:30:00 | 1277.70 | 2023-11-15 10:05:00 | 1281.55 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-11-15 09:30:00 | 1277.70 | 2023-11-15 10:35:00 | 1277.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-16 10:05:00 | 1291.50 | 2023-11-16 10:10:00 | 1294.92 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-11-16 10:05:00 | 1291.50 | 2023-11-16 10:30:00 | 1291.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-20 09:40:00 | 1327.90 | 2023-11-20 10:05:00 | 1324.72 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-11-23 10:40:00 | 1328.00 | 2023-11-23 12:20:00 | 1330.39 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-11-24 09:55:00 | 1315.60 | 2023-11-24 10:05:00 | 1318.43 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-11-28 11:10:00 | 1300.75 | 2023-11-28 13:45:00 | 1302.58 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-11-29 10:55:00 | 1336.80 | 2023-11-29 11:25:00 | 1333.97 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-12-06 11:05:00 | 1321.55 | 2023-12-06 11:40:00 | 1324.47 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-12-15 10:55:00 | 1462.00 | 2023-12-15 11:10:00 | 1469.12 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-12-15 10:55:00 | 1462.00 | 2023-12-15 15:20:00 | 1494.90 | TARGET_HIT | 0.50 | 2.25% |
| BUY | retest1 | 2023-12-18 11:00:00 | 1505.00 | 2023-12-18 12:25:00 | 1500.31 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-12-22 10:10:00 | 1441.00 | 2023-12-22 10:30:00 | 1436.57 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-12-28 11:05:00 | 1481.25 | 2023-12-28 11:15:00 | 1478.08 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-01-01 11:10:00 | 1470.00 | 2024-01-01 11:15:00 | 1473.89 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-01-01 11:10:00 | 1470.00 | 2024-01-01 11:40:00 | 1470.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 10:00:00 | 1471.00 | 2024-01-02 10:20:00 | 1475.44 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-01-08 09:45:00 | 1432.30 | 2024-01-08 09:50:00 | 1436.58 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-01-09 09:50:00 | 1472.85 | 2024-01-09 11:10:00 | 1466.92 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-01-10 09:45:00 | 1485.35 | 2024-01-10 09:50:00 | 1480.02 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-01-12 09:35:00 | 1524.60 | 2024-01-12 09:55:00 | 1533.97 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-01-12 09:35:00 | 1524.60 | 2024-01-12 10:55:00 | 1524.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 11:15:00 | 1603.80 | 2024-02-02 11:40:00 | 1609.66 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-02-02 11:15:00 | 1603.80 | 2024-02-02 11:50:00 | 1603.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-09 09:50:00 | 1623.50 | 2024-02-09 10:05:00 | 1614.92 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-02-09 09:50:00 | 1623.50 | 2024-02-09 10:10:00 | 1623.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-12 09:35:00 | 1666.50 | 2024-02-12 09:40:00 | 1659.91 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-02-20 09:30:00 | 1657.60 | 2024-02-20 09:45:00 | 1651.60 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-02-20 09:30:00 | 1657.60 | 2024-02-20 10:30:00 | 1657.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-26 10:55:00 | 1645.20 | 2024-02-26 12:10:00 | 1639.66 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-02-26 10:55:00 | 1645.20 | 2024-02-26 12:50:00 | 1645.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-27 11:05:00 | 1666.85 | 2024-02-27 12:10:00 | 1663.49 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-02-29 11:15:00 | 1654.70 | 2024-02-29 11:30:00 | 1661.30 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-02-29 11:15:00 | 1654.70 | 2024-02-29 15:00:00 | 1654.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-01 10:45:00 | 1653.90 | 2024-03-01 11:35:00 | 1658.64 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-03-13 10:50:00 | 1628.50 | 2024-03-13 11:30:00 | 1622.77 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-03-13 10:50:00 | 1628.50 | 2024-03-13 15:00:00 | 1628.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-27 11:05:00 | 1575.65 | 2024-03-27 11:25:00 | 1572.43 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-04-03 09:55:00 | 1546.00 | 2024-04-03 10:15:00 | 1552.94 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-04-03 09:55:00 | 1546.00 | 2024-04-03 10:40:00 | 1546.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-04 09:50:00 | 1519.95 | 2024-04-04 10:15:00 | 1524.16 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-04-09 09:30:00 | 1557.30 | 2024-04-09 09:55:00 | 1554.04 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-04-12 09:35:00 | 1531.95 | 2024-04-12 09:40:00 | 1535.24 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-04-19 09:45:00 | 1439.05 | 2024-04-19 10:00:00 | 1444.91 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-04-23 09:35:00 | 1487.00 | 2024-04-23 09:45:00 | 1483.48 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-25 09:45:00 | 1498.10 | 2024-04-25 10:10:00 | 1504.21 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-04-25 09:45:00 | 1498.10 | 2024-04-25 10:30:00 | 1498.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-03 11:15:00 | 1348.75 | 2024-05-03 11:35:00 | 1344.74 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-05-03 11:15:00 | 1348.75 | 2024-05-03 14:35:00 | 1348.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-07 09:35:00 | 1330.00 | 2024-05-07 09:45:00 | 1334.01 | STOP_HIT | 1.00 | -0.30% |
