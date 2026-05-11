# Godrej Consumer Products Ltd. (GODREJCP)

## Backtest Summary

- **Window:** 2025-08-11 09:15:00 → 2026-05-08 15:25:00 (12013 bars)
- **Last close:** 1041.90
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
| ENTRY1 | 59 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 7 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 52
- **Target hits / Stop hits / Partials:** 7 / 52 / 22
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 2.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 15 | 45.5% | 4 | 18 | 11 | 0.12% | 3.8% |
| BUY @ 2nd Alert (retest1) | 33 | 15 | 45.5% | 4 | 18 | 11 | 0.12% | 3.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 14 | 29.2% | 3 | 34 | 11 | -0.02% | -0.9% |
| SELL @ 2nd Alert (retest1) | 48 | 14 | 29.2% | 3 | 34 | 11 | -0.02% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 81 | 29 | 35.8% | 7 | 52 | 22 | 0.04% | 2.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 11:15:00 | 1192.30 | 1194.15 | 0.00 | ORB-short ORB[1192.70,1200.90] vol=2.6x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 11:20:00 | 1188.94 | 1192.26 | 0.00 | T1 1.5R @ 1188.94 |
| Target hit | 2025-08-14 14:40:00 | 1185.80 | 1185.52 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2025-08-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:35:00 | 1212.40 | 1206.30 | 0.00 | ORB-long ORB[1199.00,1209.60] vol=2.3x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 11:10:00 | 1218.88 | 1208.90 | 0.00 | T1 1.5R @ 1218.88 |
| Stop hit — per-position SL triggered | 2025-08-18 11:40:00 | 1212.40 | 1209.63 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-08-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:50:00 | 1253.30 | 1254.84 | 0.00 | ORB-short ORB[1254.90,1261.00] vol=5.9x ATR=3.36 |
| Stop hit — per-position SL triggered | 2025-08-26 11:05:00 | 1256.66 | 1254.84 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-09-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:40:00 | 1276.30 | 1268.92 | 0.00 | ORB-long ORB[1247.10,1263.90] vol=1.6x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 11:20:00 | 1281.54 | 1273.06 | 0.00 | T1 1.5R @ 1281.54 |
| Stop hit — per-position SL triggered | 2025-09-02 12:00:00 | 1276.30 | 1274.93 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-09-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 11:10:00 | 1272.00 | 1280.74 | 0.00 | ORB-short ORB[1280.10,1290.90] vol=4.1x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 11:15:00 | 1268.27 | 1278.66 | 0.00 | T1 1.5R @ 1268.27 |
| Stop hit — per-position SL triggered | 2025-09-03 11:20:00 | 1272.00 | 1277.82 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-09-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 11:05:00 | 1224.30 | 1239.98 | 0.00 | ORB-short ORB[1240.70,1252.80] vol=1.7x ATR=3.55 |
| Stop hit — per-position SL triggered | 2025-09-05 12:20:00 | 1227.85 | 1234.01 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-09-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 10:50:00 | 1239.50 | 1244.82 | 0.00 | ORB-short ORB[1247.00,1256.40] vol=2.6x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-09-10 11:20:00 | 1242.46 | 1244.36 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-09-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 10:45:00 | 1250.90 | 1254.68 | 0.00 | ORB-short ORB[1251.70,1264.50] vol=1.6x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-09-16 10:55:00 | 1252.87 | 1254.31 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-09-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 11:05:00 | 1225.50 | 1227.99 | 0.00 | ORB-short ORB[1228.00,1237.70] vol=2.2x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 1227.32 | 1228.01 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-09-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:05:00 | 1229.00 | 1233.12 | 0.00 | ORB-short ORB[1235.20,1241.80] vol=1.8x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-09-19 11:25:00 | 1230.85 | 1232.43 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-09-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:20:00 | 1214.30 | 1220.70 | 0.00 | ORB-short ORB[1224.00,1234.40] vol=1.5x ATR=3.74 |
| Stop hit — per-position SL triggered | 2025-09-23 10:35:00 | 1218.04 | 1220.29 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:15:00 | 1181.40 | 1190.68 | 0.00 | ORB-short ORB[1189.50,1199.00] vol=1.6x ATR=2.64 |
| Stop hit — per-position SL triggered | 2025-09-24 11:45:00 | 1184.04 | 1185.10 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-09-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 09:45:00 | 1180.60 | 1184.79 | 0.00 | ORB-short ORB[1182.00,1197.50] vol=1.6x ATR=4.48 |
| Stop hit — per-position SL triggered | 2025-09-25 10:40:00 | 1185.08 | 1183.22 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-09-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:20:00 | 1170.50 | 1172.37 | 0.00 | ORB-short ORB[1174.20,1188.80] vol=4.5x ATR=3.26 |
| Stop hit — per-position SL triggered | 2025-09-26 10:55:00 | 1173.76 | 1172.46 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-09-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:55:00 | 1166.70 | 1161.01 | 0.00 | ORB-long ORB[1152.00,1164.90] vol=3.1x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 10:20:00 | 1172.36 | 1163.25 | 0.00 | T1 1.5R @ 1172.36 |
| Stop hit — per-position SL triggered | 2025-09-29 11:35:00 | 1166.70 | 1164.82 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-09-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:35:00 | 1173.40 | 1175.64 | 0.00 | ORB-short ORB[1174.30,1184.60] vol=1.9x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-09-30 11:00:00 | 1176.90 | 1175.53 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:00:00 | 1154.40 | 1159.76 | 0.00 | ORB-short ORB[1159.80,1170.80] vol=2.4x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-10-01 11:05:00 | 1157.13 | 1159.35 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-10-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:45:00 | 1137.50 | 1140.31 | 0.00 | ORB-short ORB[1138.00,1149.90] vol=2.0x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-10-06 11:05:00 | 1139.71 | 1139.80 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 11:15:00 | 1117.20 | 1125.53 | 0.00 | ORB-short ORB[1125.80,1134.40] vol=2.0x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 11:45:00 | 1114.24 | 1122.88 | 0.00 | T1 1.5R @ 1114.24 |
| Target hit | 2025-10-13 15:20:00 | 1111.40 | 1114.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-10-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:20:00 | 1105.30 | 1108.40 | 0.00 | ORB-short ORB[1106.40,1116.10] vol=1.6x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-10-14 10:25:00 | 1107.55 | 1108.35 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-10-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:05:00 | 1129.20 | 1126.03 | 0.00 | ORB-long ORB[1120.70,1127.70] vol=2.2x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:15:00 | 1133.14 | 1126.81 | 0.00 | T1 1.5R @ 1133.14 |
| Target hit | 2025-10-17 12:55:00 | 1133.60 | 1134.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — SELL (started 2025-10-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 11:10:00 | 1125.50 | 1132.08 | 0.00 | ORB-short ORB[1127.00,1138.80] vol=1.5x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 11:25:00 | 1120.30 | 1130.87 | 0.00 | T1 1.5R @ 1120.30 |
| Stop hit — per-position SL triggered | 2025-10-23 11:30:00 | 1125.50 | 1130.22 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-10-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 11:05:00 | 1124.80 | 1125.61 | 0.00 | ORB-short ORB[1126.60,1135.00] vol=3.7x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 1126.79 | 1125.56 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-10-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:35:00 | 1113.90 | 1117.58 | 0.00 | ORB-short ORB[1114.40,1122.70] vol=3.8x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 11:20:00 | 1110.19 | 1114.71 | 0.00 | T1 1.5R @ 1110.19 |
| Stop hit — per-position SL triggered | 2025-10-29 11:30:00 | 1113.90 | 1114.46 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-10-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 11:10:00 | 1119.20 | 1116.21 | 0.00 | ORB-long ORB[1108.40,1118.70] vol=2.9x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 11:45:00 | 1123.12 | 1117.13 | 0.00 | T1 1.5R @ 1123.12 |
| Stop hit — per-position SL triggered | 2025-10-31 13:25:00 | 1119.20 | 1119.04 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-11-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:40:00 | 1161.60 | 1163.66 | 0.00 | ORB-short ORB[1168.90,1179.40] vol=1.8x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:20:00 | 1156.88 | 1161.57 | 0.00 | T1 1.5R @ 1156.88 |
| Stop hit — per-position SL triggered | 2025-11-04 12:30:00 | 1161.60 | 1160.00 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-12-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:55:00 | 1126.90 | 1130.32 | 0.00 | ORB-short ORB[1131.70,1140.10] vol=2.6x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:20:00 | 1123.36 | 1128.96 | 0.00 | T1 1.5R @ 1123.36 |
| Target hit | 2025-12-08 15:00:00 | 1121.90 | 1120.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 28 — BUY (started 2025-12-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:40:00 | 1159.40 | 1153.31 | 0.00 | ORB-long ORB[1144.10,1153.00] vol=2.4x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-12-15 11:30:00 | 1157.27 | 1155.11 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-12-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 11:10:00 | 1171.20 | 1176.27 | 0.00 | ORB-short ORB[1176.40,1188.30] vol=1.6x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-12-18 11:20:00 | 1173.62 | 1175.94 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-12-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:40:00 | 1194.20 | 1190.00 | 0.00 | ORB-long ORB[1182.10,1187.70] vol=2.3x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:45:00 | 1197.58 | 1192.90 | 0.00 | T1 1.5R @ 1197.58 |
| Target hit | 2025-12-23 12:15:00 | 1198.40 | 1199.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2025-12-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:35:00 | 1200.60 | 1197.90 | 0.00 | ORB-long ORB[1192.10,1200.10] vol=1.6x ATR=2.33 |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 1198.27 | 1199.16 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 11:05:00 | 1200.00 | 1198.29 | 0.00 | ORB-long ORB[1189.50,1198.00] vol=1.8x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-12-26 11:10:00 | 1198.42 | 1198.23 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-12-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:00:00 | 1198.60 | 1203.20 | 0.00 | ORB-short ORB[1199.70,1212.00] vol=2.8x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:05:00 | 1195.83 | 1202.51 | 0.00 | T1 1.5R @ 1195.83 |
| Stop hit — per-position SL triggered | 2025-12-29 11:10:00 | 1198.60 | 1201.67 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:50:00 | 1206.60 | 1205.09 | 0.00 | ORB-long ORB[1200.10,1206.30] vol=2.7x ATR=2.07 |
| Stop hit — per-position SL triggered | 2025-12-30 11:10:00 | 1204.53 | 1205.73 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-12-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:25:00 | 1218.30 | 1216.17 | 0.00 | ORB-long ORB[1208.20,1216.70] vol=1.6x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 10:55:00 | 1222.51 | 1217.66 | 0.00 | T1 1.5R @ 1222.51 |
| Target hit | 2025-12-31 13:55:00 | 1224.30 | 1225.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — BUY (started 2026-01-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:10:00 | 1231.10 | 1223.91 | 0.00 | ORB-long ORB[1214.50,1226.50] vol=2.7x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:15:00 | 1235.26 | 1230.94 | 0.00 | T1 1.5R @ 1235.26 |
| Stop hit — per-position SL triggered | 2026-01-01 11:40:00 | 1231.10 | 1231.23 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2026-01-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:50:00 | 1241.00 | 1234.90 | 0.00 | ORB-long ORB[1226.00,1238.30] vol=2.0x ATR=3.12 |
| Stop hit — per-position SL triggered | 2026-01-05 11:40:00 | 1237.88 | 1237.56 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:05:00 | 1234.30 | 1240.25 | 0.00 | ORB-short ORB[1240.90,1249.70] vol=2.0x ATR=3.07 |
| Stop hit — per-position SL triggered | 2026-01-08 11:30:00 | 1237.37 | 1239.40 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2026-01-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:00:00 | 1234.90 | 1236.21 | 0.00 | ORB-short ORB[1236.20,1244.80] vol=1.6x ATR=2.92 |
| Stop hit — per-position SL triggered | 2026-01-13 11:05:00 | 1237.82 | 1236.24 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-01-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 09:55:00 | 1246.80 | 1241.31 | 0.00 | ORB-long ORB[1228.10,1244.80] vol=1.7x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:35:00 | 1252.32 | 1244.95 | 0.00 | T1 1.5R @ 1252.32 |
| Stop hit — per-position SL triggered | 2026-01-16 10:45:00 | 1246.80 | 1245.48 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-01-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 10:45:00 | 1231.00 | 1238.71 | 0.00 | ORB-short ORB[1236.50,1247.40] vol=3.3x ATR=3.04 |
| Stop hit — per-position SL triggered | 2026-01-19 11:15:00 | 1234.04 | 1238.06 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2026-01-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:30:00 | 1237.50 | 1230.39 | 0.00 | ORB-long ORB[1222.70,1235.50] vol=1.8x ATR=3.91 |
| Stop hit — per-position SL triggered | 2026-01-22 09:55:00 | 1233.59 | 1233.51 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-02-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-02 11:05:00 | 1141.40 | 1145.98 | 0.00 | ORB-short ORB[1143.80,1159.70] vol=9.4x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:10:00 | 1135.32 | 1143.14 | 0.00 | T1 1.5R @ 1135.32 |
| Stop hit — per-position SL triggered | 2026-02-02 11:40:00 | 1141.40 | 1140.16 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2026-02-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 11:00:00 | 1200.50 | 1197.15 | 0.00 | ORB-long ORB[1192.30,1200.20] vol=9.6x ATR=2.58 |
| Stop hit — per-position SL triggered | 2026-02-13 11:50:00 | 1197.92 | 1198.35 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 1215.60 | 1211.64 | 0.00 | ORB-long ORB[1202.00,1211.50] vol=3.1x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-02-17 11:20:00 | 1213.39 | 1212.38 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 1206.30 | 1206.57 | 0.00 | ORB-short ORB[1206.60,1219.40] vol=3.5x ATR=2.38 |
| Stop hit — per-position SL triggered | 2026-02-18 11:45:00 | 1208.68 | 1206.74 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:05:00 | 1198.20 | 1194.84 | 0.00 | ORB-long ORB[1184.00,1195.60] vol=2.6x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:35:00 | 1202.61 | 1196.60 | 0.00 | T1 1.5R @ 1202.61 |
| Target hit | 2026-02-20 15:20:00 | 1205.20 | 1203.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 1214.20 | 1212.37 | 0.00 | ORB-long ORB[1202.10,1211.80] vol=1.6x ATR=2.47 |
| Stop hit — per-position SL triggered | 2026-02-23 11:25:00 | 1211.73 | 1212.51 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 1228.80 | 1222.93 | 0.00 | ORB-long ORB[1215.90,1226.80] vol=8.1x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-02-24 11:30:00 | 1226.36 | 1223.35 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-02-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:20:00 | 1231.90 | 1232.95 | 0.00 | ORB-short ORB[1232.10,1240.20] vol=5.0x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 1234.34 | 1233.15 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:35:00 | 1220.00 | 1224.31 | 0.00 | ORB-short ORB[1221.30,1238.90] vol=1.5x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:45:00 | 1215.85 | 1223.10 | 0.00 | T1 1.5R @ 1215.85 |
| Stop hit — per-position SL triggered | 2026-02-27 10:30:00 | 1220.00 | 1220.54 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1116.40 | 1125.65 | 0.00 | ORB-short ORB[1126.10,1132.50] vol=2.0x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 1119.64 | 1125.33 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-03-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:40:00 | 1033.10 | 1028.69 | 0.00 | ORB-long ORB[1019.50,1032.30] vol=2.3x ATR=4.04 |
| Stop hit — per-position SL triggered | 2026-03-16 09:55:00 | 1029.06 | 1029.68 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-03-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:30:00 | 1054.00 | 1042.79 | 0.00 | ORB-long ORB[1038.10,1046.40] vol=2.1x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:35:00 | 1059.51 | 1044.18 | 0.00 | T1 1.5R @ 1059.51 |
| Stop hit — per-position SL triggered | 2026-03-18 11:10:00 | 1054.00 | 1046.30 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:10:00 | 1027.60 | 1031.37 | 0.00 | ORB-short ORB[1028.50,1035.90] vol=1.5x ATR=3.10 |
| Stop hit — per-position SL triggered | 2026-03-20 10:45:00 | 1030.70 | 1030.74 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-03-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:25:00 | 1006.30 | 1008.80 | 0.00 | ORB-short ORB[1011.20,1022.30] vol=3.9x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:45:00 | 1000.52 | 1008.46 | 0.00 | T1 1.5R @ 1000.52 |
| Stop hit — per-position SL triggered | 2026-03-24 11:30:00 | 1006.30 | 1006.64 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-04-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:00:00 | 1123.50 | 1130.02 | 0.00 | ORB-short ORB[1126.30,1139.55] vol=1.9x ATR=2.96 |
| Stop hit — per-position SL triggered | 2026-04-23 11:55:00 | 1126.46 | 1128.60 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-04-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:45:00 | 1083.85 | 1087.42 | 0.00 | ORB-short ORB[1087.05,1093.75] vol=1.9x ATR=2.81 |
| Stop hit — per-position SL triggered | 2026-04-28 11:10:00 | 1086.66 | 1086.81 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:45:00 | 1090.00 | 1105.17 | 0.00 | ORB-short ORB[1109.90,1122.00] vol=1.6x ATR=4.38 |
| Stop hit — per-position SL triggered | 2026-05-06 14:20:00 | 1094.38 | 1097.81 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-08-14 11:15:00 | 1192.30 | 2025-08-14 11:20:00 | 1188.94 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-08-14 11:15:00 | 1192.30 | 2025-08-14 14:40:00 | 1185.80 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2025-08-18 10:35:00 | 1212.40 | 2025-08-18 11:10:00 | 1218.88 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-08-18 10:35:00 | 1212.40 | 2025-08-18 11:40:00 | 1212.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 10:50:00 | 1253.30 | 2025-08-26 11:05:00 | 1256.66 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-02 10:40:00 | 1276.30 | 2025-09-02 11:20:00 | 1281.54 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-09-02 10:40:00 | 1276.30 | 2025-09-02 12:00:00 | 1276.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-03 11:10:00 | 1272.00 | 2025-09-03 11:15:00 | 1268.27 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-03 11:10:00 | 1272.00 | 2025-09-03 11:20:00 | 1272.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-05 11:05:00 | 1224.30 | 2025-09-05 12:20:00 | 1227.85 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-10 10:50:00 | 1239.50 | 2025-09-10 11:20:00 | 1242.46 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-16 10:45:00 | 1250.90 | 2025-09-16 10:55:00 | 1252.87 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-09-17 11:05:00 | 1225.50 | 2025-09-17 11:15:00 | 1227.32 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-09-19 11:05:00 | 1229.00 | 2025-09-19 11:25:00 | 1230.85 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-09-23 10:20:00 | 1214.30 | 2025-09-23 10:35:00 | 1218.04 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-09-24 10:15:00 | 1181.40 | 2025-09-24 11:45:00 | 1184.04 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-25 09:45:00 | 1180.60 | 2025-09-25 10:40:00 | 1185.08 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-09-26 10:20:00 | 1170.50 | 2025-09-26 10:55:00 | 1173.76 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-29 09:55:00 | 1166.70 | 2025-09-29 10:20:00 | 1172.36 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-09-29 09:55:00 | 1166.70 | 2025-09-29 11:35:00 | 1166.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-30 10:35:00 | 1173.40 | 2025-09-30 11:00:00 | 1176.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-01 11:00:00 | 1154.40 | 2025-10-01 11:05:00 | 1157.13 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-06 10:45:00 | 1137.50 | 2025-10-06 11:05:00 | 1139.71 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-13 11:15:00 | 1117.20 | 2025-10-13 11:45:00 | 1114.24 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-10-13 11:15:00 | 1117.20 | 2025-10-13 15:20:00 | 1111.40 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2025-10-14 10:20:00 | 1105.30 | 2025-10-14 10:25:00 | 1107.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-17 10:05:00 | 1129.20 | 2025-10-17 10:15:00 | 1133.14 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-17 10:05:00 | 1129.20 | 2025-10-17 12:55:00 | 1133.60 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-10-23 11:10:00 | 1125.50 | 2025-10-23 11:25:00 | 1120.30 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-10-23 11:10:00 | 1125.50 | 2025-10-23 11:30:00 | 1125.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-27 11:05:00 | 1124.80 | 2025-10-27 11:15:00 | 1126.79 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-29 10:35:00 | 1113.90 | 2025-10-29 11:20:00 | 1110.19 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-29 10:35:00 | 1113.90 | 2025-10-29 11:30:00 | 1113.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-31 11:10:00 | 1119.20 | 2025-10-31 11:45:00 | 1123.12 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-31 11:10:00 | 1119.20 | 2025-10-31 13:25:00 | 1119.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-04 10:40:00 | 1161.60 | 2025-11-04 11:20:00 | 1156.88 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-11-04 10:40:00 | 1161.60 | 2025-11-04 12:30:00 | 1161.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 10:55:00 | 1126.90 | 2025-12-08 11:20:00 | 1123.36 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-08 10:55:00 | 1126.90 | 2025-12-08 15:00:00 | 1121.90 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2025-12-15 10:40:00 | 1159.40 | 2025-12-15 11:30:00 | 1157.27 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-18 11:10:00 | 1171.20 | 2025-12-18 11:20:00 | 1173.62 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-23 09:40:00 | 1194.20 | 2025-12-23 09:45:00 | 1197.58 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-12-23 09:40:00 | 1194.20 | 2025-12-23 12:15:00 | 1198.40 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-24 09:35:00 | 1200.60 | 2025-12-24 10:15:00 | 1198.27 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-26 11:05:00 | 1200.00 | 2025-12-26 11:10:00 | 1198.42 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-12-29 11:00:00 | 1198.60 | 2025-12-29 11:05:00 | 1195.83 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-12-29 11:00:00 | 1198.60 | 2025-12-29 11:10:00 | 1198.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-30 10:50:00 | 1206.60 | 2025-12-30 11:10:00 | 1204.53 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-31 10:25:00 | 1218.30 | 2025-12-31 10:55:00 | 1222.51 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-31 10:25:00 | 1218.30 | 2025-12-31 13:55:00 | 1224.30 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2026-01-01 11:10:00 | 1231.10 | 2026-01-01 11:15:00 | 1235.26 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-01-01 11:10:00 | 1231.10 | 2026-01-01 11:40:00 | 1231.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-05 10:50:00 | 1241.00 | 2026-01-05 11:40:00 | 1237.88 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-08 11:05:00 | 1234.30 | 2026-01-08 11:30:00 | 1237.37 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-13 11:00:00 | 1234.90 | 2026-01-13 11:05:00 | 1237.82 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-01-16 09:55:00 | 1246.80 | 2026-01-16 10:35:00 | 1252.32 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-01-16 09:55:00 | 1246.80 | 2026-01-16 10:45:00 | 1246.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-19 10:45:00 | 1231.00 | 2026-01-19 11:15:00 | 1234.04 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-22 09:30:00 | 1237.50 | 2026-01-22 09:55:00 | 1233.59 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-02 11:05:00 | 1141.40 | 2026-02-02 11:10:00 | 1135.32 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-02 11:05:00 | 1141.40 | 2026-02-02 11:40:00 | 1141.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-13 11:00:00 | 1200.50 | 2026-02-13 11:50:00 | 1197.92 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-17 11:15:00 | 1215.60 | 2026-02-17 11:20:00 | 1213.39 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-18 10:55:00 | 1206.30 | 2026-02-18 11:45:00 | 1208.68 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-20 11:05:00 | 1198.20 | 2026-02-20 11:35:00 | 1202.61 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-20 11:05:00 | 1198.20 | 2026-02-20 15:20:00 | 1205.20 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-23 11:00:00 | 1214.20 | 2026-02-23 11:25:00 | 1211.73 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-24 11:10:00 | 1228.80 | 2026-02-24 11:30:00 | 1226.36 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-25 10:20:00 | 1231.90 | 2026-02-25 11:20:00 | 1234.34 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-27 09:35:00 | 1220.00 | 2026-02-27 09:45:00 | 1215.85 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-27 09:35:00 | 1220.00 | 2026-02-27 10:30:00 | 1220.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1116.40 | 2026-03-06 10:50:00 | 1119.64 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-16 09:40:00 | 1033.10 | 2026-03-16 09:55:00 | 1029.06 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-18 10:30:00 | 1054.00 | 2026-03-18 10:35:00 | 1059.51 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-18 10:30:00 | 1054.00 | 2026-03-18 11:10:00 | 1054.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 10:10:00 | 1027.60 | 2026-03-20 10:45:00 | 1030.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-24 10:25:00 | 1006.30 | 2026-03-24 10:45:00 | 1000.52 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-24 10:25:00 | 1006.30 | 2026-03-24 11:30:00 | 1006.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:00:00 | 1123.50 | 2026-04-23 11:55:00 | 1126.46 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-28 10:45:00 | 1083.85 | 2026-04-28 11:10:00 | 1086.66 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-06 10:45:00 | 1090.00 | 2026-05-06 14:20:00 | 1094.38 | STOP_HIT | 1.00 | -0.40% |
