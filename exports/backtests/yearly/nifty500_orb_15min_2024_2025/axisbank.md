# Axis Bank Ltd. (AXISBANK)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-01-30 15:25:00 (31996 bars)
- **Last close:** 1366.00
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
| ENTRY1 | 75 |
| ENTRY2 | 0 |
| PARTIAL | 40 |
| TARGET_HIT | 12 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 115 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 63
- **Target hits / Stop hits / Partials:** 12 / 63 / 40
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 15.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 35 | 53.0% | 10 | 31 | 25 | 0.21% | 13.9% |
| BUY @ 2nd Alert (retest1) | 66 | 35 | 53.0% | 10 | 31 | 25 | 0.21% | 13.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 17 | 34.7% | 2 | 32 | 15 | 0.04% | 1.9% |
| SELL @ 2nd Alert (retest1) | 49 | 17 | 34.7% | 2 | 32 | 15 | 0.04% | 1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 115 | 52 | 45.2% | 12 | 63 | 40 | 0.14% | 15.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:20:00 | 1121.35 | 1128.31 | 0.00 | ORB-short ORB[1128.10,1134.50] vol=1.5x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:45:00 | 1117.27 | 1126.61 | 0.00 | T1 1.5R @ 1117.27 |
| Stop hit — per-position SL triggered | 2024-05-16 11:50:00 | 1121.35 | 1123.46 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:00:00 | 1140.00 | 1133.62 | 0.00 | ORB-long ORB[1126.10,1137.30] vol=1.8x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 15:15:00 | 1145.09 | 1137.51 | 0.00 | T1 1.5R @ 1145.09 |
| Target hit | 2024-05-17 15:20:00 | 1144.10 | 1137.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 11:00:00 | 1127.00 | 1131.57 | 0.00 | ORB-short ORB[1132.00,1142.00] vol=2.5x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 11:15:00 | 1123.40 | 1131.00 | 0.00 | T1 1.5R @ 1123.40 |
| Stop hit — per-position SL triggered | 2024-05-22 11:50:00 | 1127.00 | 1130.37 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:40:00 | 1141.55 | 1136.28 | 0.00 | ORB-long ORB[1125.75,1135.80] vol=1.6x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:10:00 | 1146.40 | 1140.37 | 0.00 | T1 1.5R @ 1146.40 |
| Target hit | 2024-05-23 15:20:00 | 1164.90 | 1155.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:40:00 | 1182.80 | 1186.59 | 0.00 | ORB-short ORB[1183.70,1192.40] vol=1.7x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-05-28 11:40:00 | 1186.06 | 1185.67 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 10:05:00 | 1186.95 | 1192.50 | 0.00 | ORB-short ORB[1192.05,1202.00] vol=1.5x ATR=3.31 |
| Stop hit — per-position SL triggered | 2024-06-13 11:05:00 | 1190.26 | 1191.24 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 09:30:00 | 1200.70 | 1197.06 | 0.00 | ORB-long ORB[1185.50,1200.65] vol=1.5x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 09:40:00 | 1204.79 | 1200.18 | 0.00 | T1 1.5R @ 1204.79 |
| Stop hit — per-position SL triggered | 2024-06-19 10:20:00 | 1200.70 | 1202.94 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:55:00 | 1259.80 | 1266.42 | 0.00 | ORB-short ORB[1264.20,1281.45] vol=2.0x ATR=3.33 |
| Stop hit — per-position SL triggered | 2024-06-26 11:00:00 | 1263.13 | 1266.25 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 11:15:00 | 1277.75 | 1283.48 | 0.00 | ORB-short ORB[1280.50,1289.90] vol=1.6x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 11:30:00 | 1273.55 | 1282.78 | 0.00 | T1 1.5R @ 1273.55 |
| Stop hit — per-position SL triggered | 2024-06-28 11:35:00 | 1277.75 | 1282.65 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:40:00 | 1246.55 | 1254.58 | 0.00 | ORB-short ORB[1251.00,1267.85] vol=1.8x ATR=3.47 |
| Stop hit — per-position SL triggered | 2024-07-02 10:05:00 | 1250.02 | 1251.20 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:15:00 | 1321.65 | 1311.91 | 0.00 | ORB-long ORB[1293.90,1309.70] vol=1.7x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:55:00 | 1328.01 | 1317.37 | 0.00 | T1 1.5R @ 1328.01 |
| Stop hit — per-position SL triggered | 2024-07-12 12:15:00 | 1321.65 | 1322.65 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 1276.30 | 1282.55 | 0.00 | ORB-short ORB[1280.45,1291.10] vol=4.7x ATR=4.52 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 1280.82 | 1282.47 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-24 10:25:00 | 1241.20 | 1255.07 | 0.00 | ORB-short ORB[1255.75,1266.70] vol=2.2x ATR=6.56 |
| Stop hit — per-position SL triggered | 2024-07-24 10:30:00 | 1247.76 | 1254.25 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 11:10:00 | 1185.00 | 1178.62 | 0.00 | ORB-long ORB[1174.85,1184.50] vol=1.9x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 11:40:00 | 1189.75 | 1179.47 | 0.00 | T1 1.5R @ 1189.75 |
| Stop hit — per-position SL triggered | 2024-07-29 12:10:00 | 1185.00 | 1181.27 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:45:00 | 1175.05 | 1170.60 | 0.00 | ORB-long ORB[1160.20,1170.55] vol=2.4x ATR=3.90 |
| Stop hit — per-position SL triggered | 2024-07-30 13:15:00 | 1171.15 | 1172.88 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:30:00 | 1128.55 | 1131.78 | 0.00 | ORB-short ORB[1130.55,1137.70] vol=2.0x ATR=3.13 |
| Stop hit — per-position SL triggered | 2024-08-08 11:05:00 | 1131.68 | 1130.72 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:15:00 | 1151.85 | 1144.57 | 0.00 | ORB-long ORB[1135.85,1146.00] vol=1.5x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 10:55:00 | 1156.10 | 1148.84 | 0.00 | T1 1.5R @ 1156.10 |
| Target hit | 2024-08-12 15:20:00 | 1165.95 | 1159.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-08-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 11:05:00 | 1173.40 | 1166.10 | 0.00 | ORB-long ORB[1154.65,1161.00] vol=1.7x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:20:00 | 1176.39 | 1167.43 | 0.00 | T1 1.5R @ 1176.39 |
| Stop hit — per-position SL triggered | 2024-08-20 11:25:00 | 1173.40 | 1167.77 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-09-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:50:00 | 1183.65 | 1180.12 | 0.00 | ORB-long ORB[1174.10,1179.60] vol=1.7x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-09-02 11:40:00 | 1181.62 | 1180.75 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 11:10:00 | 1178.15 | 1185.00 | 0.00 | ORB-short ORB[1185.05,1189.65] vol=3.0x ATR=1.89 |
| Stop hit — per-position SL triggered | 2024-09-03 11:15:00 | 1180.04 | 1184.64 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 10:30:00 | 1177.35 | 1179.71 | 0.00 | ORB-short ORB[1177.50,1189.65] vol=1.6x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 11:15:00 | 1173.88 | 1179.03 | 0.00 | T1 1.5R @ 1173.88 |
| Stop hit — per-position SL triggered | 2024-09-04 13:30:00 | 1177.35 | 1177.87 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 11:00:00 | 1175.00 | 1177.17 | 0.00 | ORB-short ORB[1177.70,1182.85] vol=1.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-09-05 11:05:00 | 1176.49 | 1177.15 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:00:00 | 1168.15 | 1170.79 | 0.00 | ORB-short ORB[1173.95,1181.35] vol=2.2x ATR=2.33 |
| Stop hit — per-position SL triggered | 2024-09-06 11:15:00 | 1170.48 | 1170.67 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:30:00 | 1188.00 | 1180.78 | 0.00 | ORB-long ORB[1173.35,1185.90] vol=1.6x ATR=3.88 |
| Stop hit — per-position SL triggered | 2024-09-10 09:55:00 | 1184.12 | 1183.47 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:40:00 | 1194.00 | 1188.72 | 0.00 | ORB-long ORB[1181.00,1188.70] vol=2.0x ATR=2.34 |
| Stop hit — per-position SL triggered | 2024-09-11 10:50:00 | 1191.66 | 1188.83 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:45:00 | 1236.00 | 1232.56 | 0.00 | ORB-long ORB[1227.60,1235.35] vol=1.8x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 12:05:00 | 1239.57 | 1234.97 | 0.00 | T1 1.5R @ 1239.57 |
| Stop hit — per-position SL triggered | 2024-09-17 14:00:00 | 1236.00 | 1236.52 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:40:00 | 1239.50 | 1232.49 | 0.00 | ORB-long ORB[1225.00,1232.80] vol=2.1x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-09-18 10:50:00 | 1236.89 | 1233.73 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:35:00 | 1246.95 | 1239.59 | 0.00 | ORB-long ORB[1231.10,1246.80] vol=1.5x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-09-20 10:50:00 | 1243.78 | 1240.40 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:35:00 | 1188.55 | 1183.49 | 0.00 | ORB-long ORB[1174.00,1186.65] vol=5.6x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 10:45:00 | 1193.57 | 1183.98 | 0.00 | T1 1.5R @ 1193.57 |
| Stop hit — per-position SL triggered | 2024-10-04 11:00:00 | 1188.55 | 1184.34 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:20:00 | 1176.45 | 1179.39 | 0.00 | ORB-short ORB[1178.10,1184.85] vol=2.2x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:45:00 | 1171.52 | 1178.56 | 0.00 | T1 1.5R @ 1171.52 |
| Stop hit — per-position SL triggered | 2024-10-07 11:20:00 | 1176.45 | 1176.63 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:55:00 | 1167.50 | 1162.12 | 0.00 | ORB-long ORB[1154.00,1166.25] vol=1.8x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 10:05:00 | 1173.98 | 1163.47 | 0.00 | T1 1.5R @ 1173.98 |
| Target hit | 2024-10-09 11:45:00 | 1172.10 | 1172.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2024-10-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:10:00 | 1187.00 | 1183.02 | 0.00 | ORB-long ORB[1177.15,1184.90] vol=1.5x ATR=2.99 |
| Stop hit — per-position SL triggered | 2024-10-11 10:15:00 | 1184.01 | 1183.29 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:05:00 | 1135.25 | 1140.40 | 0.00 | ORB-short ORB[1139.00,1151.35] vol=1.7x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:25:00 | 1131.14 | 1139.32 | 0.00 | T1 1.5R @ 1131.14 |
| Stop hit — per-position SL triggered | 2024-10-17 12:05:00 | 1135.25 | 1138.31 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:10:00 | 1181.60 | 1188.87 | 0.00 | ORB-short ORB[1184.20,1201.75] vol=1.8x ATR=5.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 11:40:00 | 1173.46 | 1185.05 | 0.00 | T1 1.5R @ 1173.46 |
| Stop hit — per-position SL triggered | 2024-10-25 14:45:00 | 1181.60 | 1181.41 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-11-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:10:00 | 1170.75 | 1155.21 | 0.00 | ORB-long ORB[1142.95,1158.75] vol=1.7x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 11:35:00 | 1175.46 | 1157.15 | 0.00 | T1 1.5R @ 1175.46 |
| Stop hit — per-position SL triggered | 2024-11-11 12:20:00 | 1170.75 | 1159.19 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-11-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 09:35:00 | 1132.25 | 1136.19 | 0.00 | ORB-short ORB[1133.70,1146.00] vol=3.4x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-11-22 10:00:00 | 1135.53 | 1135.88 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 1145.25 | 1149.94 | 0.00 | ORB-short ORB[1146.60,1154.35] vol=1.7x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:45:00 | 1141.13 | 1149.01 | 0.00 | T1 1.5R @ 1141.13 |
| Stop hit — per-position SL triggered | 2024-11-28 11:40:00 | 1145.25 | 1146.68 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:20:00 | 1150.90 | 1145.13 | 0.00 | ORB-long ORB[1138.00,1147.80] vol=2.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 11:20:00 | 1154.75 | 1148.11 | 0.00 | T1 1.5R @ 1154.75 |
| Target hit | 2024-12-03 15:20:00 | 1160.55 | 1156.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2024-12-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:20:00 | 1178.40 | 1169.62 | 0.00 | ORB-long ORB[1165.85,1171.75] vol=1.9x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:25:00 | 1185.97 | 1172.69 | 0.00 | T1 1.5R @ 1185.97 |
| Target hit | 2024-12-06 15:15:00 | 1183.05 | 1183.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2024-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:30:00 | 1116.45 | 1129.87 | 0.00 | ORB-short ORB[1135.50,1145.20] vol=1.7x ATR=2.86 |
| Stop hit — per-position SL triggered | 2024-12-13 10:55:00 | 1119.31 | 1127.02 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:40:00 | 1117.80 | 1109.37 | 0.00 | ORB-long ORB[1101.05,1112.25] vol=1.6x ATR=3.86 |
| Stop hit — per-position SL triggered | 2024-12-19 09:45:00 | 1113.94 | 1109.74 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:00:00 | 1078.60 | 1084.64 | 0.00 | ORB-short ORB[1082.00,1091.95] vol=1.7x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 12:05:00 | 1074.95 | 1082.43 | 0.00 | T1 1.5R @ 1074.95 |
| Target hit | 2024-12-26 15:20:00 | 1077.00 | 1078.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2024-12-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:00:00 | 1082.85 | 1081.34 | 0.00 | ORB-long ORB[1077.40,1081.50] vol=3.8x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:15:00 | 1086.00 | 1081.94 | 0.00 | T1 1.5R @ 1086.00 |
| Stop hit — per-position SL triggered | 2024-12-27 10:25:00 | 1082.85 | 1082.13 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:55:00 | 1076.75 | 1074.11 | 0.00 | ORB-long ORB[1068.30,1075.00] vol=2.5x ATR=2.24 |
| Stop hit — per-position SL triggered | 2025-01-02 11:05:00 | 1074.51 | 1074.24 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 09:40:00 | 1076.55 | 1071.35 | 0.00 | ORB-long ORB[1062.00,1074.65] vol=1.8x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 10:00:00 | 1080.89 | 1075.03 | 0.00 | T1 1.5R @ 1080.89 |
| Stop hit — per-position SL triggered | 2025-01-08 10:20:00 | 1076.55 | 1075.53 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 1064.00 | 1071.42 | 0.00 | ORB-short ORB[1071.00,1080.15] vol=2.7x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:55:00 | 1060.23 | 1070.26 | 0.00 | T1 1.5R @ 1060.23 |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 1064.00 | 1065.23 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 11:05:00 | 1047.60 | 1051.89 | 0.00 | ORB-short ORB[1055.20,1064.85] vol=1.7x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-01-10 11:10:00 | 1050.45 | 1051.77 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:00:00 | 1058.10 | 1056.89 | 0.00 | ORB-long ORB[1048.05,1056.90] vol=2.5x ATR=3.01 |
| Stop hit — per-position SL triggered | 2025-01-14 11:35:00 | 1055.09 | 1057.09 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:50:00 | 1037.40 | 1040.56 | 0.00 | ORB-short ORB[1038.20,1052.90] vol=1.5x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 10:30:00 | 1032.26 | 1039.19 | 0.00 | T1 1.5R @ 1032.26 |
| Stop hit — per-position SL triggered | 2025-01-15 10:55:00 | 1037.40 | 1038.57 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:25:00 | 1037.25 | 1033.88 | 0.00 | ORB-long ORB[1027.05,1035.50] vol=2.3x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 10:35:00 | 1042.12 | 1034.66 | 0.00 | T1 1.5R @ 1042.12 |
| Stop hit — per-position SL triggered | 2025-01-16 10:50:00 | 1037.25 | 1035.33 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 10:20:00 | 978.50 | 982.83 | 0.00 | ORB-short ORB[983.50,994.20] vol=2.0x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-01-20 10:25:00 | 980.77 | 982.66 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 09:40:00 | 949.85 | 950.91 | 0.00 | ORB-short ORB[951.35,957.15] vol=2.4x ATR=2.75 |
| Stop hit — per-position SL triggered | 2025-01-23 09:45:00 | 952.60 | 951.14 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-28 11:00:00 | 978.15 | 969.50 | 0.00 | ORB-long ORB[958.20,968.00] vol=2.2x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 11:40:00 | 982.30 | 971.64 | 0.00 | T1 1.5R @ 982.30 |
| Target hit | 2025-01-28 15:20:00 | 984.35 | 981.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-01-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:40:00 | 988.60 | 985.77 | 0.00 | ORB-long ORB[978.05,984.30] vol=5.7x ATR=2.57 |
| Stop hit — per-position SL triggered | 2025-01-31 11:00:00 | 986.03 | 986.11 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 1000.80 | 995.43 | 0.00 | ORB-long ORB[983.85,991.95] vol=6.4x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:15:00 | 1003.89 | 996.58 | 0.00 | T1 1.5R @ 1003.89 |
| Stop hit — per-position SL triggered | 2025-02-01 11:30:00 | 1000.80 | 996.95 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-02-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-03 09:30:00 | 997.50 | 994.46 | 0.00 | ORB-long ORB[987.10,997.30] vol=2.5x ATR=3.36 |
| Stop hit — per-position SL triggered | 2025-02-03 09:40:00 | 994.14 | 994.54 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-02-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:35:00 | 997.65 | 992.68 | 0.00 | ORB-long ORB[987.80,995.00] vol=2.4x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 09:40:00 | 1001.18 | 994.56 | 0.00 | T1 1.5R @ 1001.18 |
| Target hit | 2025-02-04 10:35:00 | 1000.20 | 1002.12 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — SELL (started 2025-02-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:35:00 | 1009.85 | 1015.12 | 0.00 | ORB-short ORB[1015.50,1024.45] vol=1.7x ATR=3.20 |
| Stop hit — per-position SL triggered | 2025-02-07 11:05:00 | 1013.05 | 1014.21 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:50:00 | 1012.05 | 1014.93 | 0.00 | ORB-short ORB[1018.50,1026.75] vol=1.6x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 12:00:00 | 1008.12 | 1013.34 | 0.00 | T1 1.5R @ 1008.12 |
| Stop hit — per-position SL triggered | 2025-02-10 12:20:00 | 1012.05 | 1012.98 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-02-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 11:05:00 | 995.00 | 1004.14 | 0.00 | ORB-short ORB[1009.25,1016.00] vol=1.8x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-02-14 11:10:00 | 997.35 | 1003.81 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-17 10:25:00 | 984.25 | 988.82 | 0.00 | ORB-short ORB[990.45,999.65] vol=5.0x ATR=2.50 |
| Stop hit — per-position SL triggered | 2025-02-17 10:35:00 | 986.75 | 988.52 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:00:00 | 1014.85 | 1012.56 | 0.00 | ORB-long ORB[1006.50,1014.70] vol=3.9x ATR=1.87 |
| Stop hit — per-position SL triggered | 2025-02-20 11:20:00 | 1012.98 | 1013.08 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-02-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 11:05:00 | 1008.20 | 1014.29 | 0.00 | ORB-short ORB[1012.00,1021.90] vol=3.4x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-02-21 11:10:00 | 1010.39 | 1014.18 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-27 11:10:00 | 1017.45 | 1015.29 | 0.00 | ORB-long ORB[1005.15,1015.70] vol=2.2x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 12:15:00 | 1020.11 | 1016.21 | 0.00 | T1 1.5R @ 1020.11 |
| Stop hit — per-position SL triggered | 2025-02-27 14:25:00 | 1017.45 | 1017.83 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-02-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-28 10:45:00 | 1030.00 | 1018.30 | 0.00 | ORB-long ORB[1009.95,1016.70] vol=2.9x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 11:10:00 | 1033.79 | 1020.43 | 0.00 | T1 1.5R @ 1033.79 |
| Stop hit — per-position SL triggered | 2025-02-28 11:35:00 | 1030.00 | 1021.78 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:35:00 | 1015.50 | 1011.17 | 0.00 | ORB-long ORB[1002.55,1010.45] vol=1.6x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-03-05 10:45:00 | 1013.13 | 1011.33 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 10:55:00 | 1048.55 | 1044.70 | 0.00 | ORB-long ORB[1035.00,1046.00] vol=1.7x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-03-10 11:05:00 | 1046.41 | 1044.95 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-03-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:05:00 | 1009.85 | 1015.16 | 0.00 | ORB-short ORB[1015.20,1029.80] vol=2.6x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 10:25:00 | 1006.07 | 1013.80 | 0.00 | T1 1.5R @ 1006.07 |
| Stop hit — per-position SL triggered | 2025-03-12 10:30:00 | 1009.85 | 1013.71 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 11:05:00 | 1023.85 | 1015.64 | 0.00 | ORB-long ORB[1011.10,1019.95] vol=2.1x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 11:20:00 | 1026.94 | 1016.75 | 0.00 | T1 1.5R @ 1026.94 |
| Target hit | 2025-03-17 15:20:00 | 1033.60 | 1026.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — BUY (started 2025-03-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:30:00 | 1061.80 | 1056.77 | 0.00 | ORB-long ORB[1050.55,1056.35] vol=1.7x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:00:00 | 1065.00 | 1058.24 | 0.00 | T1 1.5R @ 1065.00 |
| Stop hit — per-position SL triggered | 2025-03-21 11:45:00 | 1061.80 | 1059.56 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:05:00 | 1104.85 | 1101.45 | 0.00 | ORB-long ORB[1088.10,1103.40] vol=1.8x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-03-27 12:05:00 | 1102.17 | 1102.64 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 11:15:00 | 1091.25 | 1084.13 | 0.00 | ORB-long ORB[1075.45,1086.20] vol=2.1x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 12:05:00 | 1095.83 | 1086.33 | 0.00 | T1 1.5R @ 1095.83 |
| Stop hit — per-position SL triggered | 2025-04-04 14:20:00 | 1091.25 | 1089.31 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:00:00 | 1150.60 | 1132.50 | 0.00 | ORB-long ORB[1112.60,1125.00] vol=1.6x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 11:30:00 | 1157.09 | 1137.59 | 0.00 | T1 1.5R @ 1157.09 |
| Target hit | 2025-04-16 15:20:00 | 1162.40 | 1149.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — SELL (started 2025-04-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:45:00 | 1202.40 | 1218.15 | 0.00 | ORB-short ORB[1217.80,1226.70] vol=1.9x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:50:00 | 1196.61 | 1214.03 | 0.00 | T1 1.5R @ 1196.61 |
| Stop hit — per-position SL triggered | 2025-04-23 11:40:00 | 1202.40 | 1206.80 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 10:15:00 | 1171.00 | 1173.50 | 0.00 | ORB-short ORB[1173.20,1178.80] vol=2.0x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 10:35:00 | 1167.20 | 1172.76 | 0.00 | T1 1.5R @ 1167.20 |
| Target hit | 2025-05-06 15:20:00 | 1160.10 | 1166.09 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 10:20:00 | 1121.35 | 2024-05-16 10:45:00 | 1117.27 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-05-16 10:20:00 | 1121.35 | 2024-05-16 11:50:00 | 1121.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-17 11:00:00 | 1140.00 | 2024-05-17 15:15:00 | 1145.09 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-05-17 11:00:00 | 1140.00 | 2024-05-17 15:20:00 | 1144.10 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2024-05-22 11:00:00 | 1127.00 | 2024-05-22 11:15:00 | 1123.40 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-22 11:00:00 | 1127.00 | 2024-05-22 11:50:00 | 1127.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-23 09:40:00 | 1141.55 | 2024-05-23 10:10:00 | 1146.40 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-05-23 09:40:00 | 1141.55 | 2024-05-23 15:20:00 | 1164.90 | TARGET_HIT | 0.50 | 2.05% |
| SELL | retest1 | 2024-05-28 10:40:00 | 1182.80 | 2024-05-28 11:40:00 | 1186.06 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-13 10:05:00 | 1186.95 | 2024-06-13 11:05:00 | 1190.26 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-19 09:30:00 | 1200.70 | 2024-06-19 09:40:00 | 1204.79 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-06-19 09:30:00 | 1200.70 | 2024-06-19 10:20:00 | 1200.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-26 10:55:00 | 1259.80 | 2024-06-26 11:00:00 | 1263.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-06-28 11:15:00 | 1277.75 | 2024-06-28 11:30:00 | 1273.55 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-06-28 11:15:00 | 1277.75 | 2024-06-28 11:35:00 | 1277.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 09:40:00 | 1246.55 | 2024-07-02 10:05:00 | 1250.02 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-12 10:15:00 | 1321.65 | 2024-07-12 10:55:00 | 1328.01 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-12 10:15:00 | 1321.65 | 2024-07-12 12:15:00 | 1321.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1276.30 | 2024-07-23 11:20:00 | 1280.82 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-24 10:25:00 | 1241.20 | 2024-07-24 10:30:00 | 1247.76 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-07-29 11:10:00 | 1185.00 | 2024-07-29 11:40:00 | 1189.75 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-29 11:10:00 | 1185.00 | 2024-07-29 12:10:00 | 1185.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-30 10:45:00 | 1175.05 | 2024-07-30 13:15:00 | 1171.15 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-08 10:30:00 | 1128.55 | 2024-08-08 11:05:00 | 1131.68 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-12 10:15:00 | 1151.85 | 2024-08-12 10:55:00 | 1156.10 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-08-12 10:15:00 | 1151.85 | 2024-08-12 15:20:00 | 1165.95 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2024-08-20 11:05:00 | 1173.40 | 2024-08-20 11:20:00 | 1176.39 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2024-08-20 11:05:00 | 1173.40 | 2024-08-20 11:25:00 | 1173.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-02 10:50:00 | 1183.65 | 2024-09-02 11:40:00 | 1181.62 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-09-03 11:10:00 | 1178.15 | 2024-09-03 11:15:00 | 1180.04 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-09-04 10:30:00 | 1177.35 | 2024-09-04 11:15:00 | 1173.88 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-09-04 10:30:00 | 1177.35 | 2024-09-04 13:30:00 | 1177.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-05 11:00:00 | 1175.00 | 2024-09-05 11:05:00 | 1176.49 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2024-09-06 11:00:00 | 1168.15 | 2024-09-06 11:15:00 | 1170.48 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-09-10 09:30:00 | 1188.00 | 2024-09-10 09:55:00 | 1184.12 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-11 10:40:00 | 1194.00 | 2024-09-11 10:50:00 | 1191.66 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-09-17 10:45:00 | 1236.00 | 2024-09-17 12:05:00 | 1239.57 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-09-17 10:45:00 | 1236.00 | 2024-09-17 14:00:00 | 1236.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-18 10:40:00 | 1239.50 | 2024-09-18 10:50:00 | 1236.89 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-20 10:35:00 | 1246.95 | 2024-09-20 10:50:00 | 1243.78 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-10-04 10:35:00 | 1188.55 | 2024-10-04 10:45:00 | 1193.57 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-10-04 10:35:00 | 1188.55 | 2024-10-04 11:00:00 | 1188.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 10:20:00 | 1176.45 | 2024-10-07 10:45:00 | 1171.52 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-10-07 10:20:00 | 1176.45 | 2024-10-07 11:20:00 | 1176.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 09:55:00 | 1167.50 | 2024-10-09 10:05:00 | 1173.98 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-10-09 09:55:00 | 1167.50 | 2024-10-09 11:45:00 | 1172.10 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-10-11 10:10:00 | 1187.00 | 2024-10-11 10:15:00 | 1184.01 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-17 11:05:00 | 1135.25 | 2024-10-17 11:25:00 | 1131.14 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-10-17 11:05:00 | 1135.25 | 2024-10-17 12:05:00 | 1135.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 10:10:00 | 1181.60 | 2024-10-25 11:40:00 | 1173.46 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-10-25 10:10:00 | 1181.60 | 2024-10-25 14:45:00 | 1181.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 11:10:00 | 1170.75 | 2024-11-11 11:35:00 | 1175.46 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-11-11 11:10:00 | 1170.75 | 2024-11-11 12:20:00 | 1170.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-22 09:35:00 | 1132.25 | 2024-11-22 10:00:00 | 1135.53 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-28 10:35:00 | 1145.25 | 2024-11-28 10:45:00 | 1141.13 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-11-28 10:35:00 | 1145.25 | 2024-11-28 11:40:00 | 1145.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 10:20:00 | 1150.90 | 2024-12-03 11:20:00 | 1154.75 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-12-03 10:20:00 | 1150.90 | 2024-12-03 15:20:00 | 1160.55 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2024-12-06 10:20:00 | 1178.40 | 2024-12-06 10:25:00 | 1185.97 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-12-06 10:20:00 | 1178.40 | 2024-12-06 15:15:00 | 1183.05 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2024-12-13 10:30:00 | 1116.45 | 2024-12-13 10:55:00 | 1119.31 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-19 09:40:00 | 1117.80 | 2024-12-19 09:45:00 | 1113.94 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-26 11:00:00 | 1078.60 | 2024-12-26 12:05:00 | 1074.95 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-26 11:00:00 | 1078.60 | 2024-12-26 15:20:00 | 1077.00 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2024-12-27 10:00:00 | 1082.85 | 2024-12-27 10:15:00 | 1086.00 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-12-27 10:00:00 | 1082.85 | 2024-12-27 10:25:00 | 1082.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 10:55:00 | 1076.75 | 2025-01-02 11:05:00 | 1074.51 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-01-08 09:40:00 | 1076.55 | 2025-01-08 10:00:00 | 1080.89 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-08 09:40:00 | 1076.55 | 2025-01-08 10:20:00 | 1076.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-09 10:45:00 | 1064.00 | 2025-01-09 10:55:00 | 1060.23 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-01-09 10:45:00 | 1064.00 | 2025-01-09 12:15:00 | 1064.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-10 11:05:00 | 1047.60 | 2025-01-10 11:10:00 | 1050.45 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-14 11:00:00 | 1058.10 | 2025-01-14 11:35:00 | 1055.09 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-15 09:50:00 | 1037.40 | 2025-01-15 10:30:00 | 1032.26 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-01-15 09:50:00 | 1037.40 | 2025-01-15 10:55:00 | 1037.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-16 10:25:00 | 1037.25 | 2025-01-16 10:35:00 | 1042.12 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-01-16 10:25:00 | 1037.25 | 2025-01-16 10:50:00 | 1037.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-20 10:20:00 | 978.50 | 2025-01-20 10:25:00 | 980.77 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-01-23 09:40:00 | 949.85 | 2025-01-23 09:45:00 | 952.60 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-28 11:00:00 | 978.15 | 2025-01-28 11:40:00 | 982.30 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-01-28 11:00:00 | 978.15 | 2025-01-28 15:20:00 | 984.35 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-01-31 10:40:00 | 988.60 | 2025-01-31 11:00:00 | 986.03 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-02-01 11:00:00 | 1000.80 | 2025-02-01 11:15:00 | 1003.89 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-02-01 11:00:00 | 1000.80 | 2025-02-01 11:30:00 | 1000.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-03 09:30:00 | 997.50 | 2025-02-03 09:40:00 | 994.14 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-02-04 09:35:00 | 997.65 | 2025-02-04 09:40:00 | 1001.18 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-02-04 09:35:00 | 997.65 | 2025-02-04 10:35:00 | 1000.20 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-02-07 10:35:00 | 1009.85 | 2025-02-07 11:05:00 | 1013.05 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-10 10:50:00 | 1012.05 | 2025-02-10 12:00:00 | 1008.12 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-02-10 10:50:00 | 1012.05 | 2025-02-10 12:20:00 | 1012.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-14 11:05:00 | 995.00 | 2025-02-14 11:10:00 | 997.35 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-02-17 10:25:00 | 984.25 | 2025-02-17 10:35:00 | 986.75 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-02-20 11:00:00 | 1014.85 | 2025-02-20 11:20:00 | 1012.98 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-02-21 11:05:00 | 1008.20 | 2025-02-21 11:10:00 | 1010.39 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-02-27 11:10:00 | 1017.45 | 2025-02-27 12:15:00 | 1020.11 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-02-27 11:10:00 | 1017.45 | 2025-02-27 14:25:00 | 1017.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-28 10:45:00 | 1030.00 | 2025-02-28 11:10:00 | 1033.79 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-02-28 10:45:00 | 1030.00 | 2025-02-28 11:35:00 | 1030.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-05 10:35:00 | 1015.50 | 2025-03-05 10:45:00 | 1013.13 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-03-10 10:55:00 | 1048.55 | 2025-03-10 11:05:00 | 1046.41 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-03-12 10:05:00 | 1009.85 | 2025-03-12 10:25:00 | 1006.07 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-03-12 10:05:00 | 1009.85 | 2025-03-12 10:30:00 | 1009.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-17 11:05:00 | 1023.85 | 2025-03-17 11:20:00 | 1026.94 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-03-17 11:05:00 | 1023.85 | 2025-03-17 15:20:00 | 1033.60 | TARGET_HIT | 0.50 | 0.95% |
| BUY | retest1 | 2025-03-21 10:30:00 | 1061.80 | 2025-03-21 11:00:00 | 1065.00 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-03-21 10:30:00 | 1061.80 | 2025-03-21 11:45:00 | 1061.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-27 11:05:00 | 1104.85 | 2025-03-27 12:05:00 | 1102.17 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-04-04 11:15:00 | 1091.25 | 2025-04-04 12:05:00 | 1095.83 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-04-04 11:15:00 | 1091.25 | 2025-04-04 14:20:00 | 1091.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-16 11:00:00 | 1150.60 | 2025-04-16 11:30:00 | 1157.09 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-04-16 11:00:00 | 1150.60 | 2025-04-16 15:20:00 | 1162.40 | TARGET_HIT | 0.50 | 1.03% |
| SELL | retest1 | 2025-04-23 10:45:00 | 1202.40 | 2025-04-23 10:50:00 | 1196.61 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-04-23 10:45:00 | 1202.40 | 2025-04-23 11:40:00 | 1202.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-06 10:15:00 | 1171.00 | 2025-05-06 10:35:00 | 1167.20 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-05-06 10:15:00 | 1171.00 | 2025-05-06 15:20:00 | 1160.10 | TARGET_HIT | 0.50 | 0.93% |
