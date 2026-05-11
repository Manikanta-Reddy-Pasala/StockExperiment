# Home First Finance Company India Ltd. (HOMEFIRST)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-05 15:25:00 (18238 bars)
- **Last close:** 1187.00
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
| ENTRY1 | 45 |
| ENTRY2 | 0 |
| PARTIAL | 21 |
| TARGET_HIT | 11 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 34
- **Target hits / Stop hits / Partials:** 11 / 34 / 21
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 14.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 13 | 38.2% | 4 | 21 | 9 | 0.16% | 5.3% |
| BUY @ 2nd Alert (retest1) | 34 | 13 | 38.2% | 4 | 21 | 9 | 0.16% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 32 | 19 | 59.4% | 7 | 13 | 12 | 0.29% | 9.4% |
| SELL @ 2nd Alert (retest1) | 32 | 19 | 59.4% | 7 | 13 | 12 | 0.29% | 9.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 66 | 32 | 48.5% | 11 | 34 | 21 | 0.22% | 14.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:40:00 | 1160.80 | 1153.03 | 0.00 | ORB-long ORB[1142.00,1157.50] vol=1.7x ATR=4.54 |
| Stop hit — per-position SL triggered | 2025-05-23 10:30:00 | 1156.26 | 1156.80 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 10:30:00 | 1179.00 | 1174.43 | 0.00 | ORB-long ORB[1167.10,1177.20] vol=1.8x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 10:55:00 | 1183.66 | 1175.32 | 0.00 | T1 1.5R @ 1183.66 |
| Target hit | 2025-05-27 15:15:00 | 1189.50 | 1190.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:15:00 | 1271.20 | 1274.44 | 0.00 | ORB-short ORB[1281.60,1297.40] vol=2.3x ATR=4.13 |
| Stop hit — per-position SL triggered | 2025-06-04 11:30:00 | 1275.33 | 1274.43 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:55:00 | 1280.00 | 1266.98 | 0.00 | ORB-long ORB[1257.00,1268.80] vol=2.0x ATR=4.39 |
| Stop hit — per-position SL triggered | 2025-06-11 11:05:00 | 1275.61 | 1268.08 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:35:00 | 1294.50 | 1286.42 | 0.00 | ORB-long ORB[1268.00,1283.90] vol=3.1x ATR=6.32 |
| Stop hit — per-position SL triggered | 2025-06-18 09:50:00 | 1288.18 | 1290.08 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-07-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 10:00:00 | 1437.00 | 1421.87 | 0.00 | ORB-long ORB[1414.10,1426.20] vol=1.6x ATR=7.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 12:20:00 | 1447.86 | 1431.94 | 0.00 | T1 1.5R @ 1447.86 |
| Target hit | 2025-07-16 13:55:00 | 1439.20 | 1440.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:15:00 | 1408.80 | 1413.04 | 0.00 | ORB-short ORB[1410.00,1423.00] vol=2.3x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 11:20:00 | 1403.82 | 1412.85 | 0.00 | T1 1.5R @ 1403.82 |
| Stop hit — per-position SL triggered | 2025-07-17 11:30:00 | 1408.80 | 1412.66 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:30:00 | 1375.90 | 1383.66 | 0.00 | ORB-short ORB[1380.80,1392.20] vol=2.0x ATR=5.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1368.24 | 1378.28 | 0.00 | T1 1.5R @ 1368.24 |
| Target hit | 2025-07-18 14:25:00 | 1369.30 | 1362.15 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2025-07-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-25 09:35:00 | 1436.70 | 1424.99 | 0.00 | ORB-long ORB[1413.00,1424.40] vol=6.2x ATR=5.77 |
| Stop hit — per-position SL triggered | 2025-07-25 09:40:00 | 1430.93 | 1425.58 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:35:00 | 1265.50 | 1259.35 | 0.00 | ORB-long ORB[1250.00,1264.80] vol=2.5x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:00:00 | 1272.58 | 1261.29 | 0.00 | T1 1.5R @ 1272.58 |
| Stop hit — per-position SL triggered | 2025-09-05 10:10:00 | 1265.50 | 1261.62 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 11:15:00 | 1258.10 | 1264.71 | 0.00 | ORB-short ORB[1260.00,1275.50] vol=3.3x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 12:30:00 | 1252.93 | 1262.10 | 0.00 | T1 1.5R @ 1252.93 |
| Stop hit — per-position SL triggered | 2025-09-09 12:55:00 | 1258.10 | 1261.40 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-09-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 11:05:00 | 1274.60 | 1281.65 | 0.00 | ORB-short ORB[1275.80,1292.10] vol=1.7x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 11:30:00 | 1268.41 | 1280.28 | 0.00 | T1 1.5R @ 1268.41 |
| Target hit | 2025-09-17 13:30:00 | 1272.30 | 1271.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2025-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:35:00 | 1273.30 | 1266.73 | 0.00 | ORB-long ORB[1254.50,1272.40] vol=1.6x ATR=5.88 |
| Stop hit — per-position SL triggered | 2025-09-24 11:30:00 | 1267.42 | 1271.27 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-10-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:55:00 | 1243.80 | 1248.02 | 0.00 | ORB-short ORB[1244.60,1260.00] vol=2.7x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 11:05:00 | 1237.99 | 1246.46 | 0.00 | T1 1.5R @ 1237.99 |
| Target hit | 2025-10-06 15:20:00 | 1228.90 | 1236.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2025-10-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 09:55:00 | 1222.10 | 1226.05 | 0.00 | ORB-short ORB[1223.30,1236.00] vol=2.8x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 10:35:00 | 1215.84 | 1221.43 | 0.00 | T1 1.5R @ 1215.84 |
| Target hit | 2025-10-07 15:20:00 | 1207.90 | 1211.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-10-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:50:00 | 1213.80 | 1216.05 | 0.00 | ORB-short ORB[1215.80,1226.60] vol=2.1x ATR=3.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 12:40:00 | 1208.90 | 1214.11 | 0.00 | T1 1.5R @ 1208.90 |
| Target hit | 2025-10-09 15:20:00 | 1205.90 | 1208.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2025-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:35:00 | 1238.60 | 1233.47 | 0.00 | ORB-long ORB[1220.70,1234.30] vol=1.6x ATR=4.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 09:45:00 | 1245.16 | 1234.99 | 0.00 | T1 1.5R @ 1245.16 |
| Stop hit — per-position SL triggered | 2025-10-15 09:50:00 | 1238.60 | 1234.97 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-10-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:45:00 | 1224.90 | 1215.88 | 0.00 | ORB-long ORB[1207.40,1221.30] vol=2.5x ATR=5.34 |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 1219.56 | 1217.61 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-10-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:10:00 | 1227.30 | 1222.35 | 0.00 | ORB-long ORB[1211.70,1224.40] vol=1.7x ATR=3.96 |
| Stop hit — per-position SL triggered | 2025-10-24 12:10:00 | 1223.34 | 1224.74 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 09:30:00 | 1216.30 | 1210.22 | 0.00 | ORB-long ORB[1200.00,1214.70] vol=1.8x ATR=4.95 |
| Stop hit — per-position SL triggered | 2025-10-30 10:05:00 | 1211.35 | 1210.88 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 11:15:00 | 1213.60 | 1208.30 | 0.00 | ORB-long ORB[1195.60,1211.80] vol=2.3x ATR=2.95 |
| Stop hit — per-position SL triggered | 2025-10-31 14:10:00 | 1210.65 | 1211.20 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-11-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:10:00 | 1222.50 | 1217.25 | 0.00 | ORB-long ORB[1195.60,1214.00] vol=1.8x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 10:35:00 | 1230.59 | 1218.50 | 0.00 | T1 1.5R @ 1230.59 |
| Target hit | 2025-11-03 15:20:00 | 1254.30 | 1240.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2025-11-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:20:00 | 1159.30 | 1148.61 | 0.00 | ORB-long ORB[1125.00,1134.70] vol=5.8x ATR=5.47 |
| Stop hit — per-position SL triggered | 2025-11-10 10:50:00 | 1153.83 | 1150.39 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 09:40:00 | 1149.90 | 1154.42 | 0.00 | ORB-short ORB[1152.30,1169.00] vol=1.7x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 09:50:00 | 1142.83 | 1147.08 | 0.00 | T1 1.5R @ 1142.83 |
| Target hit | 2025-11-12 10:50:00 | 1145.90 | 1145.60 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — BUY (started 2025-11-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:35:00 | 1173.40 | 1169.46 | 0.00 | ORB-long ORB[1159.70,1170.30] vol=7.0x ATR=5.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 10:20:00 | 1181.88 | 1172.44 | 0.00 | T1 1.5R @ 1181.88 |
| Target hit | 2025-11-14 15:20:00 | 1202.20 | 1186.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-11-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:55:00 | 1171.30 | 1176.43 | 0.00 | ORB-short ORB[1173.60,1186.10] vol=1.9x ATR=4.57 |
| Stop hit — per-position SL triggered | 2025-11-21 15:20:00 | 1172.20 | 1171.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2025-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:30:00 | 1122.90 | 1129.02 | 0.00 | ORB-short ORB[1129.20,1139.90] vol=3.3x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:40:00 | 1117.35 | 1126.69 | 0.00 | T1 1.5R @ 1117.35 |
| Stop hit — per-position SL triggered | 2025-11-27 10:10:00 | 1122.90 | 1124.66 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-12-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:50:00 | 1191.60 | 1186.77 | 0.00 | ORB-long ORB[1172.90,1189.60] vol=1.9x ATR=2.94 |
| Stop hit — per-position SL triggered | 2025-12-15 11:25:00 | 1188.66 | 1187.51 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-12-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:35:00 | 1144.30 | 1152.26 | 0.00 | ORB-short ORB[1148.40,1162.60] vol=1.7x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:40:00 | 1138.94 | 1150.18 | 0.00 | T1 1.5R @ 1138.94 |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 1144.30 | 1143.28 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:15:00 | 1140.00 | 1147.33 | 0.00 | ORB-short ORB[1148.00,1158.00] vol=2.0x ATR=3.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:05:00 | 1135.06 | 1143.76 | 0.00 | T1 1.5R @ 1135.06 |
| Stop hit — per-position SL triggered | 2025-12-19 15:00:00 | 1140.00 | 1133.56 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-12-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:50:00 | 1105.00 | 1108.56 | 0.00 | ORB-short ORB[1107.20,1115.50] vol=1.8x ATR=3.30 |
| Stop hit — per-position SL triggered | 2025-12-30 10:25:00 | 1108.30 | 1107.75 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-12-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 10:55:00 | 1100.20 | 1107.34 | 0.00 | ORB-short ORB[1106.20,1116.10] vol=2.1x ATR=3.58 |
| Stop hit — per-position SL triggered | 2025-12-31 11:00:00 | 1103.78 | 1106.17 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2026-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:00:00 | 1114.70 | 1106.73 | 0.00 | ORB-long ORB[1098.80,1108.10] vol=2.3x ATR=3.35 |
| Stop hit — per-position SL triggered | 2026-01-02 10:05:00 | 1111.35 | 1107.58 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 11:15:00 | 1092.20 | 1098.28 | 0.00 | ORB-short ORB[1096.20,1110.30] vol=1.8x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-01-05 11:20:00 | 1094.98 | 1098.22 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2026-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:55:00 | 1083.50 | 1086.58 | 0.00 | ORB-short ORB[1088.10,1095.90] vol=2.7x ATR=2.07 |
| Stop hit — per-position SL triggered | 2026-01-06 11:00:00 | 1085.57 | 1086.58 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2026-01-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 11:05:00 | 1056.80 | 1057.80 | 0.00 | ORB-short ORB[1059.80,1074.30] vol=3.3x ATR=2.57 |
| Stop hit — per-position SL triggered | 2026-01-07 11:55:00 | 1059.37 | 1057.70 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2026-01-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 09:40:00 | 1038.30 | 1034.55 | 0.00 | ORB-long ORB[1025.70,1037.00] vol=2.4x ATR=4.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:45:00 | 1045.15 | 1036.03 | 0.00 | T1 1.5R @ 1045.15 |
| Stop hit — per-position SL triggered | 2026-01-12 10:10:00 | 1038.30 | 1038.72 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2026-01-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 10:20:00 | 1076.00 | 1072.88 | 0.00 | ORB-long ORB[1063.60,1075.10] vol=2.1x ATR=3.26 |
| Stop hit — per-position SL triggered | 2026-01-19 10:35:00 | 1072.74 | 1072.98 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-02-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 11:10:00 | 1183.10 | 1165.30 | 0.00 | ORB-long ORB[1143.00,1160.30] vol=2.2x ATR=5.10 |
| Stop hit — per-position SL triggered | 2026-02-04 11:35:00 | 1178.00 | 1166.92 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 1176.70 | 1185.83 | 0.00 | ORB-short ORB[1183.20,1194.10] vol=1.9x ATR=4.03 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 1180.73 | 1181.78 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 1151.60 | 1146.63 | 0.00 | ORB-long ORB[1135.30,1150.00] vol=2.5x ATR=4.79 |
| Stop hit — per-position SL triggered | 2026-02-20 11:50:00 | 1146.81 | 1147.27 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-03-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:10:00 | 1065.00 | 1075.08 | 0.00 | ORB-short ORB[1071.00,1087.00] vol=1.8x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:55:00 | 1057.79 | 1070.41 | 0.00 | T1 1.5R @ 1057.79 |
| Target hit | 2026-03-05 15:20:00 | 1048.50 | 1047.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 1118.75 | 1112.31 | 0.00 | ORB-long ORB[1103.30,1109.95] vol=2.6x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:05:00 | 1124.64 | 1117.53 | 0.00 | T1 1.5R @ 1124.64 |
| Stop hit — per-position SL triggered | 2026-04-17 11:30:00 | 1118.75 | 1122.53 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 1146.60 | 1139.18 | 0.00 | ORB-long ORB[1134.00,1141.30] vol=1.9x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:55:00 | 1152.48 | 1140.78 | 0.00 | T1 1.5R @ 1152.48 |
| Stop hit — per-position SL triggered | 2026-04-27 10:25:00 | 1146.60 | 1142.32 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 1150.90 | 1135.24 | 0.00 | ORB-long ORB[1125.00,1139.95] vol=3.7x ATR=4.76 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 1146.14 | 1138.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-23 09:40:00 | 1160.80 | 2025-05-23 10:30:00 | 1156.26 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-05-27 10:30:00 | 1179.00 | 2025-05-27 10:55:00 | 1183.66 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-05-27 10:30:00 | 1179.00 | 2025-05-27 15:15:00 | 1189.50 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2025-06-04 11:15:00 | 1271.20 | 2025-06-04 11:30:00 | 1275.33 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-06-11 10:55:00 | 1280.00 | 2025-06-11 11:05:00 | 1275.61 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-18 09:35:00 | 1294.50 | 2025-06-18 09:50:00 | 1288.18 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-07-16 10:00:00 | 1437.00 | 2025-07-16 12:20:00 | 1447.86 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2025-07-16 10:00:00 | 1437.00 | 2025-07-16 13:55:00 | 1439.20 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2025-07-17 11:15:00 | 1408.80 | 2025-07-17 11:20:00 | 1403.82 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-17 11:15:00 | 1408.80 | 2025-07-17 11:30:00 | 1408.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 09:30:00 | 1375.90 | 2025-07-18 10:15:00 | 1368.24 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-07-18 09:30:00 | 1375.90 | 2025-07-18 14:25:00 | 1369.30 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-25 09:35:00 | 1436.70 | 2025-07-25 09:40:00 | 1430.93 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-09-05 09:35:00 | 1265.50 | 2025-09-05 10:00:00 | 1272.58 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-09-05 09:35:00 | 1265.50 | 2025-09-05 10:10:00 | 1265.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-09 11:15:00 | 1258.10 | 2025-09-09 12:30:00 | 1252.93 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-09-09 11:15:00 | 1258.10 | 2025-09-09 12:55:00 | 1258.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-17 11:05:00 | 1274.60 | 2025-09-17 11:30:00 | 1268.41 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-09-17 11:05:00 | 1274.60 | 2025-09-17 13:30:00 | 1272.30 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2025-09-24 09:35:00 | 1273.30 | 2025-09-24 11:30:00 | 1267.42 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-10-06 10:55:00 | 1243.80 | 2025-10-06 11:05:00 | 1237.99 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-10-06 10:55:00 | 1243.80 | 2025-10-06 15:20:00 | 1228.90 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2025-10-07 09:55:00 | 1222.10 | 2025-10-07 10:35:00 | 1215.84 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-10-07 09:55:00 | 1222.10 | 2025-10-07 15:20:00 | 1207.90 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2025-10-09 10:50:00 | 1213.80 | 2025-10-09 12:40:00 | 1208.90 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-09 10:50:00 | 1213.80 | 2025-10-09 15:20:00 | 1205.90 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2025-10-15 09:35:00 | 1238.60 | 2025-10-15 09:45:00 | 1245.16 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-15 09:35:00 | 1238.60 | 2025-10-15 09:50:00 | 1238.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 09:45:00 | 1224.90 | 2025-10-20 10:15:00 | 1219.56 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-10-24 10:10:00 | 1227.30 | 2025-10-24 12:10:00 | 1223.34 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-30 09:30:00 | 1216.30 | 2025-10-30 10:05:00 | 1211.35 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-10-31 11:15:00 | 1213.60 | 2025-10-31 14:10:00 | 1210.65 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-03 10:10:00 | 1222.50 | 2025-11-03 10:35:00 | 1230.59 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-11-03 10:10:00 | 1222.50 | 2025-11-03 15:20:00 | 1254.30 | TARGET_HIT | 0.50 | 2.60% |
| BUY | retest1 | 2025-11-10 10:20:00 | 1159.30 | 2025-11-10 10:50:00 | 1153.83 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-11-12 09:40:00 | 1149.90 | 2025-11-12 09:50:00 | 1142.83 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-11-12 09:40:00 | 1149.90 | 2025-11-12 10:50:00 | 1145.90 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2025-11-14 09:35:00 | 1173.40 | 2025-11-14 10:20:00 | 1181.88 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-11-14 09:35:00 | 1173.40 | 2025-11-14 15:20:00 | 1202.20 | TARGET_HIT | 0.50 | 2.45% |
| SELL | retest1 | 2025-11-21 10:55:00 | 1171.30 | 2025-11-21 15:20:00 | 1172.20 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest1 | 2025-11-27 09:30:00 | 1122.90 | 2025-11-27 09:40:00 | 1117.35 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-11-27 09:30:00 | 1122.90 | 2025-11-27 10:10:00 | 1122.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-15 10:50:00 | 1191.60 | 2025-12-15 11:25:00 | 1188.66 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-18 09:35:00 | 1144.30 | 2025-12-18 09:40:00 | 1138.94 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-12-18 09:35:00 | 1144.30 | 2025-12-18 10:15:00 | 1144.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-19 10:15:00 | 1140.00 | 2025-12-19 11:05:00 | 1135.06 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-12-19 10:15:00 | 1140.00 | 2025-12-19 15:00:00 | 1140.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 09:50:00 | 1105.00 | 2025-12-30 10:25:00 | 1108.30 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-31 10:55:00 | 1100.20 | 2025-12-31 11:00:00 | 1103.78 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-01-02 10:00:00 | 1114.70 | 2026-01-02 10:05:00 | 1111.35 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-05 11:15:00 | 1092.20 | 2026-01-05 11:20:00 | 1094.98 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-06 10:55:00 | 1083.50 | 2026-01-06 11:00:00 | 1085.57 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-01-07 11:05:00 | 1056.80 | 2026-01-07 11:55:00 | 1059.37 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-01-12 09:40:00 | 1038.30 | 2026-01-12 09:45:00 | 1045.15 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-01-12 09:40:00 | 1038.30 | 2026-01-12 10:10:00 | 1038.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-19 10:20:00 | 1076.00 | 2026-01-19 10:35:00 | 1072.74 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-04 11:10:00 | 1183.10 | 2026-02-04 11:35:00 | 1178.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-18 09:50:00 | 1176.70 | 2026-02-18 10:50:00 | 1180.73 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-20 10:50:00 | 1151.60 | 2026-02-20 11:50:00 | 1146.81 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-05 11:10:00 | 1065.00 | 2026-03-05 11:55:00 | 1057.79 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-03-05 11:10:00 | 1065.00 | 2026-03-05 15:20:00 | 1048.50 | TARGET_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2026-04-17 09:35:00 | 1118.75 | 2026-04-17 10:05:00 | 1124.64 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-17 09:35:00 | 1118.75 | 2026-04-17 11:30:00 | 1118.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:50:00 | 1146.60 | 2026-04-27 09:55:00 | 1152.48 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-27 09:50:00 | 1146.60 | 2026-04-27 10:25:00 | 1146.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:45:00 | 1150.90 | 2026-04-29 10:50:00 | 1146.14 | STOP_HIT | 1.00 | -0.41% |
