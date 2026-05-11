# ICICIBANK (ICICIBANK)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1267.80
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
| ENTRY1 | 72 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 14 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 104 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 58
- **Target hits / Stop hits / Partials:** 14 / 58 / 32
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 9.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 23 | 51.1% | 8 | 22 | 15 | 0.15% | 6.9% |
| BUY @ 2nd Alert (retest1) | 45 | 23 | 51.1% | 8 | 22 | 15 | 0.15% | 6.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 59 | 23 | 39.0% | 6 | 36 | 17 | 0.05% | 2.7% |
| SELL @ 2nd Alert (retest1) | 59 | 23 | 39.0% | 6 | 36 | 17 | 0.05% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 104 | 46 | 44.2% | 14 | 58 | 32 | 0.09% | 9.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:45:00 | 1114.35 | 1118.45 | 0.00 | ORB-short ORB[1116.00,1125.90] vol=1.6x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 10:25:00 | 1110.84 | 1116.79 | 0.00 | T1 1.5R @ 1110.84 |
| Target hit | 2024-05-22 15:10:00 | 1113.75 | 1111.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2024-05-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:50:00 | 1124.95 | 1120.74 | 0.00 | ORB-long ORB[1112.00,1117.70] vol=1.5x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 12:35:00 | 1128.80 | 1122.93 | 0.00 | T1 1.5R @ 1128.80 |
| Target hit | 2024-05-23 15:20:00 | 1135.45 | 1127.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-05-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:45:00 | 1139.00 | 1134.29 | 0.00 | ORB-long ORB[1130.45,1136.00] vol=1.8x ATR=2.31 |
| Stop hit — per-position SL triggered | 2024-05-27 11:00:00 | 1136.69 | 1134.93 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 11:00:00 | 1127.15 | 1129.49 | 0.00 | ORB-short ORB[1129.85,1135.20] vol=2.1x ATR=1.93 |
| Stop hit — per-position SL triggered | 2024-05-28 12:00:00 | 1129.08 | 1128.68 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 09:30:00 | 1142.40 | 1135.99 | 0.00 | ORB-long ORB[1126.15,1139.25] vol=1.6x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 09:55:00 | 1147.37 | 1140.98 | 0.00 | T1 1.5R @ 1147.37 |
| Stop hit — per-position SL triggered | 2024-06-19 10:20:00 | 1142.40 | 1141.74 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 11:15:00 | 1210.15 | 1218.38 | 0.00 | ORB-short ORB[1215.35,1227.00] vol=2.2x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 11:25:00 | 1206.34 | 1217.83 | 0.00 | T1 1.5R @ 1206.34 |
| Stop hit — per-position SL triggered | 2024-06-28 11:35:00 | 1210.15 | 1217.29 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:40:00 | 1197.40 | 1204.42 | 0.00 | ORB-short ORB[1201.10,1217.00] vol=1.9x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 11:35:00 | 1191.53 | 1199.18 | 0.00 | T1 1.5R @ 1191.53 |
| Stop hit — per-position SL triggered | 2024-07-02 13:00:00 | 1197.40 | 1196.59 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 11:05:00 | 1240.20 | 1235.72 | 0.00 | ORB-long ORB[1229.30,1237.80] vol=2.3x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-07-09 11:25:00 | 1237.91 | 1236.11 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:10:00 | 1227.55 | 1243.62 | 0.00 | ORB-short ORB[1245.25,1257.80] vol=1.6x ATR=2.92 |
| Stop hit — per-position SL triggered | 2024-07-11 11:40:00 | 1230.47 | 1241.83 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:50:00 | 1244.70 | 1237.22 | 0.00 | ORB-long ORB[1232.20,1241.85] vol=2.9x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:00:00 | 1249.70 | 1241.33 | 0.00 | T1 1.5R @ 1249.70 |
| Stop hit — per-position SL triggered | 2024-07-12 11:25:00 | 1244.70 | 1242.35 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 1240.20 | 1242.71 | 0.00 | ORB-short ORB[1240.25,1246.90] vol=4.7x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:50:00 | 1234.28 | 1242.04 | 0.00 | T1 1.5R @ 1234.28 |
| Stop hit — per-position SL triggered | 2024-07-23 11:55:00 | 1240.20 | 1241.96 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-24 10:25:00 | 1213.95 | 1223.81 | 0.00 | ORB-short ORB[1219.55,1230.35] vol=1.7x ATR=5.53 |
| Stop hit — per-position SL triggered | 2024-07-24 10:30:00 | 1219.48 | 1223.33 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 10:45:00 | 1168.20 | 1178.70 | 0.00 | ORB-short ORB[1177.75,1188.90] vol=1.6x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-08-05 11:05:00 | 1171.32 | 1176.86 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 11:00:00 | 1168.10 | 1174.30 | 0.00 | ORB-short ORB[1174.00,1187.70] vol=1.5x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-08-06 11:15:00 | 1171.30 | 1174.09 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 11:10:00 | 1164.70 | 1166.72 | 0.00 | ORB-short ORB[1165.10,1172.95] vol=1.6x ATR=3.37 |
| Stop hit — per-position SL triggered | 2024-08-08 11:30:00 | 1168.07 | 1166.68 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 11:00:00 | 1172.00 | 1175.99 | 0.00 | ORB-short ORB[1173.35,1180.50] vol=1.5x ATR=2.42 |
| Stop hit — per-position SL triggered | 2024-08-09 11:55:00 | 1174.42 | 1174.17 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:30:00 | 1174.75 | 1180.84 | 0.00 | ORB-short ORB[1178.15,1189.25] vol=1.5x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 12:30:00 | 1170.18 | 1178.20 | 0.00 | T1 1.5R @ 1170.18 |
| Target hit | 2024-08-13 15:20:00 | 1169.30 | 1172.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-08-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 11:05:00 | 1166.50 | 1171.12 | 0.00 | ORB-short ORB[1171.40,1178.65] vol=2.6x ATR=1.94 |
| Stop hit — per-position SL triggered | 2024-08-21 11:20:00 | 1168.44 | 1170.60 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:55:00 | 1184.15 | 1180.76 | 0.00 | ORB-long ORB[1176.60,1182.45] vol=2.2x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-08-22 11:15:00 | 1182.31 | 1181.03 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:55:00 | 1209.75 | 1207.30 | 0.00 | ORB-long ORB[1201.05,1207.70] vol=2.2x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-08-26 10:50:00 | 1207.73 | 1208.14 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 11:05:00 | 1220.90 | 1214.96 | 0.00 | ORB-long ORB[1210.00,1215.65] vol=2.0x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 11:25:00 | 1225.05 | 1216.82 | 0.00 | T1 1.5R @ 1225.05 |
| Target hit | 2024-08-27 15:20:00 | 1225.95 | 1222.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-09-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 11:10:00 | 1217.90 | 1213.72 | 0.00 | ORB-long ORB[1200.45,1213.00] vol=4.8x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 11:45:00 | 1221.54 | 1214.61 | 0.00 | T1 1.5R @ 1221.54 |
| Stop hit — per-position SL triggered | 2024-09-09 12:15:00 | 1217.90 | 1215.35 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:15:00 | 1257.35 | 1253.64 | 0.00 | ORB-long ORB[1244.70,1253.00] vol=2.0x ATR=2.33 |
| Stop hit — per-position SL triggered | 2024-09-16 10:25:00 | 1255.02 | 1253.77 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:40:00 | 1278.40 | 1271.17 | 0.00 | ORB-long ORB[1262.00,1270.80] vol=2.3x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:35:00 | 1282.03 | 1274.75 | 0.00 | T1 1.5R @ 1282.03 |
| Target hit | 2024-09-18 15:20:00 | 1287.55 | 1286.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-09-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:55:00 | 1307.55 | 1300.47 | 0.00 | ORB-long ORB[1291.55,1302.00] vol=1.7x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 10:25:00 | 1312.11 | 1305.10 | 0.00 | T1 1.5R @ 1312.11 |
| Stop hit — per-position SL triggered | 2024-09-20 13:30:00 | 1307.55 | 1311.28 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 11:10:00 | 1317.20 | 1323.82 | 0.00 | ORB-short ORB[1322.00,1331.80] vol=2.9x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:45:00 | 1312.05 | 1321.52 | 0.00 | T1 1.5R @ 1312.05 |
| Stop hit — per-position SL triggered | 2024-09-23 12:30:00 | 1317.20 | 1319.76 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 10:55:00 | 1320.90 | 1324.00 | 0.00 | ORB-short ORB[1321.70,1333.00] vol=2.1x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-09-27 11:35:00 | 1323.18 | 1323.37 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 11:15:00 | 1282.00 | 1287.03 | 0.00 | ORB-short ORB[1286.60,1296.80] vol=2.1x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 11:35:00 | 1278.24 | 1285.93 | 0.00 | T1 1.5R @ 1278.24 |
| Target hit | 2024-09-30 15:20:00 | 1273.65 | 1278.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2024-10-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:50:00 | 1258.55 | 1252.86 | 0.00 | ORB-long ORB[1245.35,1254.15] vol=2.7x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-10-04 11:00:00 | 1255.06 | 1253.10 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-10-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-07 10:45:00 | 1253.20 | 1253.10 | 0.00 | ORB-long ORB[1241.55,1253.15] vol=1.6x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-10-07 11:05:00 | 1249.98 | 1252.85 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:05:00 | 1226.40 | 1234.10 | 0.00 | ORB-short ORB[1232.65,1240.80] vol=1.7x ATR=2.31 |
| Stop hit — per-position SL triggered | 2024-10-11 11:10:00 | 1228.71 | 1233.95 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:45:00 | 1228.90 | 1225.57 | 0.00 | ORB-long ORB[1217.40,1227.35] vol=1.8x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 10:15:00 | 1232.64 | 1227.60 | 0.00 | T1 1.5R @ 1232.64 |
| Target hit | 2024-10-14 15:00:00 | 1231.60 | 1232.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — BUY (started 2024-10-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 10:55:00 | 1246.95 | 1241.99 | 0.00 | ORB-long ORB[1235.40,1242.70] vol=1.6x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-10-15 11:20:00 | 1244.33 | 1242.42 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 11:00:00 | 1245.45 | 1237.14 | 0.00 | ORB-long ORB[1225.25,1234.60] vol=2.3x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 11:35:00 | 1249.97 | 1238.82 | 0.00 | T1 1.5R @ 1249.97 |
| Target hit | 2024-10-18 15:20:00 | 1265.85 | 1255.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-10-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 11:00:00 | 1248.40 | 1256.39 | 0.00 | ORB-short ORB[1259.60,1267.95] vol=2.2x ATR=3.89 |
| Stop hit — per-position SL triggered | 2024-10-23 14:25:00 | 1252.29 | 1251.76 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-11-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 11:10:00 | 1273.35 | 1279.38 | 0.00 | ORB-short ORB[1278.20,1291.80] vol=2.3x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-11-04 11:20:00 | 1276.13 | 1279.13 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:35:00 | 1269.65 | 1275.47 | 0.00 | ORB-short ORB[1273.00,1283.55] vol=1.7x ATR=3.86 |
| Stop hit — per-position SL triggered | 2024-11-05 11:10:00 | 1273.51 | 1274.32 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-11-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:40:00 | 1298.20 | 1302.47 | 0.00 | ORB-short ORB[1298.80,1306.80] vol=3.6x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:55:00 | 1294.15 | 1301.13 | 0.00 | T1 1.5R @ 1294.15 |
| Target hit | 2024-11-28 15:20:00 | 1286.10 | 1292.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2024-12-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:30:00 | 1312.45 | 1308.18 | 0.00 | ORB-long ORB[1301.70,1307.80] vol=1.7x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:55:00 | 1315.72 | 1309.69 | 0.00 | T1 1.5R @ 1315.72 |
| Stop hit — per-position SL triggered | 2024-12-04 11:55:00 | 1312.45 | 1311.42 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 1311.00 | 1315.56 | 0.00 | ORB-short ORB[1313.95,1319.50] vol=1.6x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-12-05 11:00:00 | 1313.52 | 1315.42 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:10:00 | 1319.20 | 1323.07 | 0.00 | ORB-short ORB[1321.00,1335.15] vol=2.4x ATR=2.93 |
| Stop hit — per-position SL triggered | 2024-12-13 11:15:00 | 1322.13 | 1322.98 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 11:00:00 | 1335.95 | 1338.61 | 0.00 | ORB-short ORB[1341.10,1348.10] vol=1.6x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:25:00 | 1331.98 | 1337.87 | 0.00 | T1 1.5R @ 1331.98 |
| Target hit | 2024-12-17 15:05:00 | 1334.00 | 1333.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — SELL (started 2024-12-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 11:00:00 | 1294.05 | 1294.51 | 0.00 | ORB-short ORB[1294.20,1304.50] vol=2.3x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-12-24 11:40:00 | 1296.19 | 1294.53 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:40:00 | 1268.35 | 1276.58 | 0.00 | ORB-short ORB[1281.00,1288.00] vol=1.6x ATR=2.71 |
| Stop hit — per-position SL triggered | 2025-01-03 11:15:00 | 1271.06 | 1273.76 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:15:00 | 1260.40 | 1274.64 | 0.00 | ORB-short ORB[1275.05,1284.35] vol=2.3x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 12:45:00 | 1256.42 | 1267.78 | 0.00 | T1 1.5R @ 1256.42 |
| Stop hit — per-position SL triggered | 2025-01-08 13:40:00 | 1260.40 | 1265.39 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 1253.15 | 1256.97 | 0.00 | ORB-short ORB[1253.60,1263.70] vol=1.7x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 12:00:00 | 1249.07 | 1254.03 | 0.00 | T1 1.5R @ 1249.07 |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 1253.15 | 1253.81 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 10:45:00 | 1249.00 | 1254.07 | 0.00 | ORB-short ORB[1256.50,1264.95] vol=2.7x ATR=2.75 |
| Stop hit — per-position SL triggered | 2025-01-10 11:00:00 | 1251.75 | 1253.70 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:05:00 | 1237.05 | 1233.49 | 0.00 | ORB-long ORB[1230.00,1236.00] vol=2.2x ATR=2.49 |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 1234.56 | 1233.80 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:50:00 | 1240.55 | 1246.49 | 0.00 | ORB-short ORB[1240.65,1254.65] vol=1.8x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-01-16 10:55:00 | 1242.98 | 1246.21 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:55:00 | 1219.25 | 1232.51 | 0.00 | ORB-short ORB[1237.85,1249.00] vol=1.8x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-01-17 11:05:00 | 1222.18 | 1230.44 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 11:15:00 | 1227.10 | 1219.83 | 0.00 | ORB-long ORB[1215.25,1224.00] vol=2.3x ATR=2.70 |
| Stop hit — per-position SL triggered | 2025-01-20 11:20:00 | 1224.40 | 1220.25 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:15:00 | 1214.40 | 1223.67 | 0.00 | ORB-short ORB[1231.00,1238.95] vol=2.1x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 11:30:00 | 1210.46 | 1221.68 | 0.00 | T1 1.5R @ 1210.46 |
| Stop hit — per-position SL triggered | 2025-01-21 11:45:00 | 1214.40 | 1220.87 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 10:45:00 | 1205.10 | 1202.71 | 0.00 | ORB-long ORB[1196.50,1205.05] vol=1.6x ATR=3.13 |
| Stop hit — per-position SL triggered | 2025-01-22 10:50:00 | 1201.97 | 1202.72 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-01-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 11:10:00 | 1217.40 | 1208.50 | 0.00 | ORB-long ORB[1202.00,1212.00] vol=2.6x ATR=2.92 |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 1214.48 | 1208.92 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-28 09:30:00 | 1249.60 | 1244.61 | 0.00 | ORB-long ORB[1236.00,1249.00] vol=1.9x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:00:00 | 1255.83 | 1248.18 | 0.00 | T1 1.5R @ 1255.83 |
| Target hit | 2025-01-28 13:55:00 | 1253.00 | 1253.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — SELL (started 2025-01-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 10:55:00 | 1244.30 | 1245.84 | 0.00 | ORB-short ORB[1245.05,1256.00] vol=2.4x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:10:00 | 1240.91 | 1245.28 | 0.00 | T1 1.5R @ 1240.91 |
| Stop hit — per-position SL triggered | 2025-01-31 12:35:00 | 1244.30 | 1243.83 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-02-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:35:00 | 1273.80 | 1266.88 | 0.00 | ORB-long ORB[1260.15,1267.95] vol=2.2x ATR=3.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 09:40:00 | 1278.70 | 1269.66 | 0.00 | T1 1.5R @ 1278.70 |
| Target hit | 2025-02-04 10:30:00 | 1274.20 | 1274.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — SELL (started 2025-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:00:00 | 1266.70 | 1272.37 | 0.00 | ORB-short ORB[1270.25,1279.00] vol=1.7x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:15:00 | 1263.46 | 1271.80 | 0.00 | T1 1.5R @ 1263.46 |
| Stop hit — per-position SL triggered | 2025-02-06 11:45:00 | 1266.70 | 1271.12 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-02-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:40:00 | 1251.70 | 1257.94 | 0.00 | ORB-short ORB[1252.10,1263.35] vol=1.8x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:05:00 | 1247.15 | 1254.30 | 0.00 | T1 1.5R @ 1247.15 |
| Stop hit — per-position SL triggered | 2025-02-14 12:45:00 | 1251.70 | 1252.59 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:45:00 | 1240.60 | 1245.89 | 0.00 | ORB-short ORB[1245.00,1257.60] vol=1.7x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-02-18 10:55:00 | 1243.04 | 1245.21 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-02-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:55:00 | 1228.90 | 1234.47 | 0.00 | ORB-short ORB[1238.55,1244.70] vol=1.7x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-02-21 11:10:00 | 1231.09 | 1233.99 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-02-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 10:20:00 | 1212.10 | 1219.63 | 0.00 | ORB-short ORB[1223.85,1231.90] vol=2.1x ATR=2.40 |
| Stop hit — per-position SL triggered | 2025-02-24 10:25:00 | 1214.50 | 1218.96 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:00:00 | 1235.90 | 1230.97 | 0.00 | ORB-long ORB[1217.20,1230.90] vol=1.9x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 12:15:00 | 1239.94 | 1232.30 | 0.00 | T1 1.5R @ 1239.94 |
| Target hit | 2025-03-11 15:20:00 | 1244.90 | 1238.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 11:15:00 | 1269.40 | 1263.27 | 0.00 | ORB-long ORB[1254.75,1264.60] vol=2.6x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 12:05:00 | 1272.47 | 1264.52 | 0.00 | T1 1.5R @ 1272.47 |
| Stop hit — per-position SL triggered | 2025-03-17 14:40:00 | 1269.40 | 1267.26 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:50:00 | 1339.45 | 1326.86 | 0.00 | ORB-long ORB[1311.60,1322.45] vol=2.7x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:20:00 | 1343.79 | 1330.18 | 0.00 | T1 1.5R @ 1343.79 |
| Stop hit — per-position SL triggered | 2025-03-21 11:40:00 | 1339.45 | 1331.34 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:55:00 | 1371.60 | 1355.01 | 0.00 | ORB-long ORB[1345.05,1355.95] vol=1.5x ATR=3.18 |
| Stop hit — per-position SL triggered | 2025-03-24 11:05:00 | 1368.42 | 1356.06 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 11:15:00 | 1350.00 | 1358.15 | 0.00 | ORB-short ORB[1353.00,1361.75] vol=1.9x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 11:40:00 | 1345.35 | 1357.06 | 0.00 | T1 1.5R @ 1345.35 |
| Target hit | 2025-03-25 15:20:00 | 1344.20 | 1345.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2025-03-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 10:55:00 | 1340.00 | 1346.64 | 0.00 | ORB-short ORB[1341.10,1350.00] vol=4.0x ATR=3.60 |
| Stop hit — per-position SL triggered | 2025-03-26 11:05:00 | 1343.60 | 1346.36 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-04-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 10:55:00 | 1292.00 | 1298.58 | 0.00 | ORB-short ORB[1299.10,1310.05] vol=1.8x ATR=3.94 |
| Stop hit — per-position SL triggered | 2025-04-08 11:05:00 | 1295.94 | 1298.18 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-04-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 11:10:00 | 1397.60 | 1403.41 | 0.00 | ORB-short ORB[1402.60,1414.80] vol=2.5x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 11:45:00 | 1392.41 | 1401.70 | 0.00 | T1 1.5R @ 1392.41 |
| Stop hit — per-position SL triggered | 2025-04-25 12:20:00 | 1397.60 | 1400.17 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 1428.20 | 1418.73 | 0.00 | ORB-long ORB[1403.80,1423.80] vol=1.5x ATR=4.06 |
| Stop hit — per-position SL triggered | 2025-04-28 09:40:00 | 1424.14 | 1420.56 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 11:05:00 | 1442.90 | 1438.76 | 0.00 | ORB-long ORB[1432.30,1440.00] vol=1.5x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-05-08 11:20:00 | 1441.10 | 1438.92 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-22 09:45:00 | 1114.35 | 2024-05-22 10:25:00 | 1110.84 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-22 09:45:00 | 1114.35 | 2024-05-22 15:10:00 | 1113.75 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2024-05-23 10:50:00 | 1124.95 | 2024-05-23 12:35:00 | 1128.80 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-05-23 10:50:00 | 1124.95 | 2024-05-23 15:20:00 | 1135.45 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2024-05-27 10:45:00 | 1139.00 | 2024-05-27 11:00:00 | 1136.69 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-05-28 11:00:00 | 1127.15 | 2024-05-28 12:00:00 | 1129.08 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-06-19 09:30:00 | 1142.40 | 2024-06-19 09:55:00 | 1147.37 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-06-19 09:30:00 | 1142.40 | 2024-06-19 10:20:00 | 1142.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-28 11:15:00 | 1210.15 | 2024-06-28 11:25:00 | 1206.34 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-06-28 11:15:00 | 1210.15 | 2024-06-28 11:35:00 | 1210.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 09:40:00 | 1197.40 | 2024-07-02 11:35:00 | 1191.53 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-07-02 09:40:00 | 1197.40 | 2024-07-02 13:00:00 | 1197.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-09 11:05:00 | 1240.20 | 2024-07-09 11:25:00 | 1237.91 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-07-11 11:10:00 | 1227.55 | 2024-07-11 11:40:00 | 1230.47 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-12 10:50:00 | 1244.70 | 2024-07-12 11:00:00 | 1249.70 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-12 10:50:00 | 1244.70 | 2024-07-12 11:25:00 | 1244.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1240.20 | 2024-07-23 11:50:00 | 1234.28 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1240.20 | 2024-07-23 11:55:00 | 1240.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-24 10:25:00 | 1213.95 | 2024-07-24 10:30:00 | 1219.48 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-08-05 10:45:00 | 1168.20 | 2024-08-05 11:05:00 | 1171.32 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-06 11:00:00 | 1168.10 | 2024-08-06 11:15:00 | 1171.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-08 11:10:00 | 1164.70 | 2024-08-08 11:30:00 | 1168.07 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-09 11:00:00 | 1172.00 | 2024-08-09 11:55:00 | 1174.42 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-13 10:30:00 | 1174.75 | 2024-08-13 12:30:00 | 1170.18 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-08-13 10:30:00 | 1174.75 | 2024-08-13 15:20:00 | 1169.30 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-21 11:05:00 | 1166.50 | 2024-08-21 11:20:00 | 1168.44 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-08-22 10:55:00 | 1184.15 | 2024-08-22 11:15:00 | 1182.31 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-08-26 09:55:00 | 1209.75 | 2024-08-26 10:50:00 | 1207.73 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-08-27 11:05:00 | 1220.90 | 2024-08-27 11:25:00 | 1225.05 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-08-27 11:05:00 | 1220.90 | 2024-08-27 15:20:00 | 1225.95 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-09 11:10:00 | 1217.90 | 2024-09-09 11:45:00 | 1221.54 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-09-09 11:10:00 | 1217.90 | 2024-09-09 12:15:00 | 1217.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-16 10:15:00 | 1257.35 | 2024-09-16 10:25:00 | 1255.02 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-09-18 10:40:00 | 1278.40 | 2024-09-18 11:35:00 | 1282.03 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-09-18 10:40:00 | 1278.40 | 2024-09-18 15:20:00 | 1287.55 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2024-09-20 09:55:00 | 1307.55 | 2024-09-20 10:25:00 | 1312.11 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-09-20 09:55:00 | 1307.55 | 2024-09-20 13:30:00 | 1307.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-23 11:10:00 | 1317.20 | 2024-09-23 11:45:00 | 1312.05 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-09-23 11:10:00 | 1317.20 | 2024-09-23 12:30:00 | 1317.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-27 10:55:00 | 1320.90 | 2024-09-27 11:35:00 | 1323.18 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-09-30 11:15:00 | 1282.00 | 2024-09-30 11:35:00 | 1278.24 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-09-30 11:15:00 | 1282.00 | 2024-09-30 15:20:00 | 1273.65 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2024-10-04 10:50:00 | 1258.55 | 2024-10-04 11:00:00 | 1255.06 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-07 10:45:00 | 1253.20 | 2024-10-07 11:05:00 | 1249.98 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-11 11:05:00 | 1226.40 | 2024-10-11 11:10:00 | 1228.71 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-10-14 09:45:00 | 1228.90 | 2024-10-14 10:15:00 | 1232.64 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-10-14 09:45:00 | 1228.90 | 2024-10-14 15:00:00 | 1231.60 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-10-15 10:55:00 | 1246.95 | 2024-10-15 11:20:00 | 1244.33 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-10-18 11:00:00 | 1245.45 | 2024-10-18 11:35:00 | 1249.97 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-10-18 11:00:00 | 1245.45 | 2024-10-18 15:20:00 | 1265.85 | TARGET_HIT | 0.50 | 1.64% |
| SELL | retest1 | 2024-10-23 11:00:00 | 1248.40 | 2024-10-23 14:25:00 | 1252.29 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-11-04 11:10:00 | 1273.35 | 2024-11-04 11:20:00 | 1276.13 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-11-05 10:35:00 | 1269.65 | 2024-11-05 11:10:00 | 1273.51 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-11-28 10:40:00 | 1298.20 | 2024-11-28 10:55:00 | 1294.15 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-11-28 10:40:00 | 1298.20 | 2024-11-28 15:20:00 | 1286.10 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2024-12-04 10:30:00 | 1312.45 | 2024-12-04 10:55:00 | 1315.72 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2024-12-04 10:30:00 | 1312.45 | 2024-12-04 11:55:00 | 1312.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 10:55:00 | 1311.00 | 2024-12-05 11:00:00 | 1313.52 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-12-13 11:10:00 | 1319.20 | 2024-12-13 11:15:00 | 1322.13 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-17 11:00:00 | 1335.95 | 2024-12-17 11:25:00 | 1331.98 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-17 11:00:00 | 1335.95 | 2024-12-17 15:05:00 | 1334.00 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2024-12-24 11:00:00 | 1294.05 | 2024-12-24 11:40:00 | 1296.19 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-01-03 10:40:00 | 1268.35 | 2025-01-03 11:15:00 | 1271.06 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-08 11:15:00 | 1260.40 | 2025-01-08 12:45:00 | 1256.42 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-01-08 11:15:00 | 1260.40 | 2025-01-08 13:40:00 | 1260.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-09 10:45:00 | 1253.15 | 2025-01-09 12:00:00 | 1249.07 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-01-09 10:45:00 | 1253.15 | 2025-01-09 12:15:00 | 1253.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-10 10:45:00 | 1249.00 | 2025-01-10 11:00:00 | 1251.75 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-01-14 11:05:00 | 1237.05 | 2025-01-14 11:15:00 | 1234.56 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-01-16 10:50:00 | 1240.55 | 2025-01-16 10:55:00 | 1242.98 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-01-17 10:55:00 | 1219.25 | 2025-01-17 11:05:00 | 1222.18 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-20 11:15:00 | 1227.10 | 2025-01-20 11:20:00 | 1224.40 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-01-21 11:15:00 | 1214.40 | 2025-01-21 11:30:00 | 1210.46 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-01-21 11:15:00 | 1214.40 | 2025-01-21 11:45:00 | 1214.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-22 10:45:00 | 1205.10 | 2025-01-22 10:50:00 | 1201.97 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-24 11:10:00 | 1217.40 | 2025-01-24 11:15:00 | 1214.48 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-28 09:30:00 | 1249.60 | 2025-01-28 10:00:00 | 1255.83 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-28 09:30:00 | 1249.60 | 2025-01-28 13:55:00 | 1253.00 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-01-31 10:55:00 | 1244.30 | 2025-01-31 11:10:00 | 1240.91 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-01-31 10:55:00 | 1244.30 | 2025-01-31 12:35:00 | 1244.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-04 09:35:00 | 1273.80 | 2025-02-04 09:40:00 | 1278.70 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-02-04 09:35:00 | 1273.80 | 2025-02-04 10:30:00 | 1274.20 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2025-02-06 11:00:00 | 1266.70 | 2025-02-06 11:15:00 | 1263.46 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-02-06 11:00:00 | 1266.70 | 2025-02-06 11:45:00 | 1266.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-14 10:40:00 | 1251.70 | 2025-02-14 12:05:00 | 1247.15 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-02-14 10:40:00 | 1251.70 | 2025-02-14 12:45:00 | 1251.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-18 10:45:00 | 1240.60 | 2025-02-18 10:55:00 | 1243.04 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-02-21 10:55:00 | 1228.90 | 2025-02-21 11:10:00 | 1231.09 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-02-24 10:20:00 | 1212.10 | 2025-02-24 10:25:00 | 1214.50 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-03-11 11:00:00 | 1235.90 | 2025-03-11 12:15:00 | 1239.94 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-03-11 11:00:00 | 1235.90 | 2025-03-11 15:20:00 | 1244.90 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2025-03-17 11:15:00 | 1269.40 | 2025-03-17 12:05:00 | 1272.47 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-03-17 11:15:00 | 1269.40 | 2025-03-17 14:40:00 | 1269.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 10:50:00 | 1339.45 | 2025-03-21 11:20:00 | 1343.79 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-03-21 10:50:00 | 1339.45 | 2025-03-21 11:40:00 | 1339.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-24 10:55:00 | 1371.60 | 2025-03-24 11:05:00 | 1368.42 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-03-25 11:15:00 | 1350.00 | 2025-03-25 11:40:00 | 1345.35 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-03-25 11:15:00 | 1350.00 | 2025-03-25 15:20:00 | 1344.20 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-03-26 10:55:00 | 1340.00 | 2025-03-26 11:05:00 | 1343.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-04-08 10:55:00 | 1292.00 | 2025-04-08 11:05:00 | 1295.94 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-04-25 11:10:00 | 1397.60 | 2025-04-25 11:45:00 | 1392.41 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-04-25 11:10:00 | 1397.60 | 2025-04-25 12:20:00 | 1397.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-28 09:30:00 | 1428.20 | 2025-04-28 09:40:00 | 1424.14 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-05-08 11:05:00 | 1442.90 | 2025-05-08 11:20:00 | 1441.10 | STOP_HIT | 1.00 | -0.12% |
