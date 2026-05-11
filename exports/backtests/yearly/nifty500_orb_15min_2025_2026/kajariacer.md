# Kajaria Ceramics Ltd. (KAJARIACER)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15388 bars)
- **Last close:** 1105.00
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
| ENTRY1 | 55 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 14 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 41
- **Target hits / Stop hits / Partials:** 14 / 41 / 19
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 12.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 27 | 56.2% | 13 | 21 | 14 | 0.26% | 12.3% |
| BUY @ 2nd Alert (retest1) | 48 | 27 | 56.2% | 13 | 21 | 14 | 0.26% | 12.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 26 | 6 | 23.1% | 1 | 20 | 5 | 0.01% | 0.2% |
| SELL @ 2nd Alert (retest1) | 26 | 6 | 23.1% | 1 | 20 | 5 | 0.01% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 74 | 33 | 44.6% | 14 | 41 | 19 | 0.17% | 12.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:45:00 | 980.85 | 968.61 | 0.00 | ORB-long ORB[959.90,970.00] vol=2.3x ATR=4.81 |
| Stop hit — per-position SL triggered | 2025-05-21 09:50:00 | 976.04 | 970.97 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:10:00 | 1014.20 | 1006.55 | 0.00 | ORB-long ORB[994.05,1008.90] vol=2.1x ATR=4.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 10:15:00 | 1020.81 | 1009.78 | 0.00 | T1 1.5R @ 1020.81 |
| Target hit | 2025-05-26 15:20:00 | 1044.95 | 1034.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-05-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:45:00 | 1038.15 | 1031.17 | 0.00 | ORB-long ORB[1019.50,1029.10] vol=2.4x ATR=2.82 |
| Stop hit — per-position SL triggered | 2025-05-28 12:50:00 | 1035.33 | 1034.63 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:55:00 | 1035.90 | 1038.79 | 0.00 | ORB-short ORB[1036.75,1049.80] vol=1.5x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-05-30 11:00:00 | 1038.43 | 1038.77 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 09:50:00 | 1045.80 | 1038.63 | 0.00 | ORB-long ORB[1030.00,1039.10] vol=2.1x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:55:00 | 1050.91 | 1041.79 | 0.00 | T1 1.5R @ 1050.91 |
| Target hit | 2025-06-02 15:20:00 | 1055.00 | 1052.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2025-06-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:50:00 | 1067.10 | 1062.58 | 0.00 | ORB-long ORB[1053.10,1064.00] vol=1.5x ATR=2.47 |
| Stop hit — per-position SL triggered | 2025-06-03 10:05:00 | 1064.63 | 1063.03 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 10:45:00 | 1023.80 | 1017.54 | 0.00 | ORB-long ORB[1007.00,1021.40] vol=5.8x ATR=3.93 |
| Stop hit — per-position SL triggered | 2025-06-23 11:00:00 | 1019.87 | 1019.05 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:45:00 | 1063.20 | 1070.88 | 0.00 | ORB-short ORB[1073.10,1081.80] vol=3.6x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-07-01 10:55:00 | 1065.33 | 1070.53 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:20:00 | 1150.00 | 1160.39 | 0.00 | ORB-short ORB[1162.40,1174.80] vol=2.0x ATR=3.67 |
| Stop hit — per-position SL triggered | 2025-07-08 11:30:00 | 1153.67 | 1153.28 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:30:00 | 1166.00 | 1159.32 | 0.00 | ORB-long ORB[1154.90,1165.00] vol=2.9x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 12:30:00 | 1172.25 | 1163.97 | 0.00 | T1 1.5R @ 1172.25 |
| Target hit | 2025-07-09 15:20:00 | 1180.40 | 1173.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-07-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 10:20:00 | 1190.20 | 1182.53 | 0.00 | ORB-long ORB[1177.20,1190.00] vol=1.6x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 14:35:00 | 1197.47 | 1189.43 | 0.00 | T1 1.5R @ 1197.47 |
| Target hit | 2025-07-10 15:20:00 | 1192.90 | 1191.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:05:00 | 1178.80 | 1190.64 | 0.00 | ORB-short ORB[1183.00,1200.60] vol=2.3x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:20:00 | 1172.52 | 1188.27 | 0.00 | T1 1.5R @ 1172.52 |
| Stop hit — per-position SL triggered | 2025-07-11 11:35:00 | 1178.80 | 1185.98 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:15:00 | 1181.60 | 1185.31 | 0.00 | ORB-short ORB[1184.80,1190.00] vol=7.6x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-08-06 12:00:00 | 1184.39 | 1184.86 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-08-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:00:00 | 1290.00 | 1301.11 | 0.00 | ORB-short ORB[1294.00,1308.00] vol=1.7x ATR=5.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 13:00:00 | 1281.68 | 1289.38 | 0.00 | T1 1.5R @ 1281.68 |
| Target hit | 2025-08-12 15:20:00 | 1257.80 | 1279.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2025-08-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:50:00 | 1287.00 | 1280.58 | 0.00 | ORB-long ORB[1274.60,1283.70] vol=4.8x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:00:00 | 1293.59 | 1287.53 | 0.00 | T1 1.5R @ 1293.59 |
| Target hit | 2025-08-18 11:55:00 | 1290.50 | 1292.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2025-08-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:40:00 | 1243.00 | 1249.83 | 0.00 | ORB-short ORB[1244.40,1259.70] vol=1.5x ATR=4.44 |
| Stop hit — per-position SL triggered | 2025-08-26 09:45:00 | 1247.44 | 1249.76 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-09-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 10:45:00 | 1214.00 | 1210.58 | 0.00 | ORB-long ORB[1204.00,1212.80] vol=6.9x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 10:50:00 | 1218.70 | 1211.52 | 0.00 | T1 1.5R @ 1218.70 |
| Target hit | 2025-09-03 15:20:00 | 1236.00 | 1225.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-09-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:50:00 | 1230.80 | 1226.54 | 0.00 | ORB-long ORB[1212.20,1229.90] vol=2.0x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 13:40:00 | 1236.63 | 1228.81 | 0.00 | T1 1.5R @ 1236.63 |
| Target hit | 2025-09-08 14:55:00 | 1232.50 | 1235.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2025-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:30:00 | 1247.00 | 1243.14 | 0.00 | ORB-long ORB[1231.80,1244.00] vol=5.5x ATR=4.39 |
| Stop hit — per-position SL triggered | 2025-09-09 10:20:00 | 1242.61 | 1244.62 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:30:00 | 1256.60 | 1251.78 | 0.00 | ORB-long ORB[1245.00,1252.00] vol=5.1x ATR=4.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 09:55:00 | 1262.62 | 1257.83 | 0.00 | T1 1.5R @ 1262.62 |
| Target hit | 2025-09-10 10:45:00 | 1259.50 | 1260.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — SELL (started 2025-09-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 11:00:00 | 1243.50 | 1251.26 | 0.00 | ORB-short ORB[1246.00,1264.00] vol=4.6x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:50:00 | 1237.08 | 1250.54 | 0.00 | T1 1.5R @ 1237.08 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1243.50 | 1244.88 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:50:00 | 1228.70 | 1224.56 | 0.00 | ORB-long ORB[1216.70,1226.70] vol=2.7x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 12:10:00 | 1233.78 | 1226.18 | 0.00 | T1 1.5R @ 1233.78 |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 1228.70 | 1227.26 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:15:00 | 1223.00 | 1228.62 | 0.00 | ORB-short ORB[1226.00,1236.00] vol=3.1x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:25:00 | 1218.57 | 1225.69 | 0.00 | T1 1.5R @ 1218.57 |
| Stop hit — per-position SL triggered | 2025-09-18 11:55:00 | 1223.00 | 1224.65 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:30:00 | 1203.10 | 1193.63 | 0.00 | ORB-long ORB[1183.80,1195.60] vol=2.9x ATR=5.14 |
| Stop hit — per-position SL triggered | 2025-09-24 09:35:00 | 1197.96 | 1194.45 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:15:00 | 1216.50 | 1209.90 | 0.00 | ORB-long ORB[1201.50,1213.90] vol=1.9x ATR=4.73 |
| Stop hit — per-position SL triggered | 2025-09-25 11:25:00 | 1211.77 | 1212.43 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:30:00 | 1197.20 | 1184.74 | 0.00 | ORB-long ORB[1174.10,1189.40] vol=3.5x ATR=5.85 |
| Stop hit — per-position SL triggered | 2025-09-29 09:35:00 | 1191.35 | 1186.84 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-10-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:45:00 | 1167.00 | 1166.05 | 0.00 | ORB-long ORB[1155.00,1165.00] vol=2.0x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 11:10:00 | 1172.55 | 1168.79 | 0.00 | T1 1.5R @ 1172.55 |
| Target hit | 2025-10-01 15:20:00 | 1191.00 | 1185.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-10-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:30:00 | 1186.60 | 1194.31 | 0.00 | ORB-short ORB[1189.60,1202.80] vol=2.1x ATR=3.72 |
| Stop hit — per-position SL triggered | 2025-10-08 10:40:00 | 1190.32 | 1193.37 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:55:00 | 1234.20 | 1223.94 | 0.00 | ORB-long ORB[1203.60,1213.90] vol=5.3x ATR=4.57 |
| Stop hit — per-position SL triggered | 2025-10-10 10:05:00 | 1229.63 | 1225.67 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 10:55:00 | 1244.70 | 1240.71 | 0.00 | ORB-long ORB[1226.10,1244.00] vol=6.1x ATR=3.88 |
| Stop hit — per-position SL triggered | 2025-10-13 11:00:00 | 1240.82 | 1241.98 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:10:00 | 1207.70 | 1214.13 | 0.00 | ORB-short ORB[1215.80,1225.00] vol=2.3x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:30:00 | 1202.66 | 1212.42 | 0.00 | T1 1.5R @ 1202.66 |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 1207.70 | 1207.27 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 10:50:00 | 1200.00 | 1206.82 | 0.00 | ORB-short ORB[1207.00,1214.00] vol=2.1x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-10-27 10:55:00 | 1202.96 | 1206.47 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:55:00 | 1234.10 | 1230.89 | 0.00 | ORB-long ORB[1219.00,1234.00] vol=2.1x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-10-29 10:10:00 | 1230.93 | 1231.21 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:50:00 | 1209.70 | 1211.51 | 0.00 | ORB-short ORB[1210.80,1219.00] vol=1.6x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-10-30 10:55:00 | 1212.11 | 1211.55 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:25:00 | 1198.80 | 1190.94 | 0.00 | ORB-long ORB[1180.10,1194.40] vol=1.8x ATR=4.66 |
| Stop hit — per-position SL triggered | 2025-11-04 10:30:00 | 1194.14 | 1191.31 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 1160.90 | 1163.52 | 0.00 | ORB-short ORB[1162.30,1170.30] vol=4.4x ATR=3.46 |
| Stop hit — per-position SL triggered | 2025-11-06 09:35:00 | 1164.36 | 1163.46 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2026-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 11:10:00 | 989.30 | 984.60 | 0.00 | ORB-long ORB[981.20,988.75] vol=2.3x ATR=4.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:20:00 | 996.30 | 990.18 | 0.00 | T1 1.5R @ 996.30 |
| Target hit | 2026-01-06 15:20:00 | 1003.05 | 999.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2026-01-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:55:00 | 995.15 | 990.51 | 0.00 | ORB-long ORB[985.25,993.55] vol=1.7x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 12:10:00 | 998.98 | 995.27 | 0.00 | T1 1.5R @ 998.98 |
| Target hit | 2026-01-16 13:30:00 | 997.05 | 997.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 11:15:00 | 985.10 | 988.35 | 0.00 | ORB-short ORB[985.50,993.90] vol=1.6x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-01-19 11:30:00 | 987.34 | 987.98 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2026-01-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:00:00 | 889.00 | 895.26 | 0.00 | ORB-short ORB[892.00,901.00] vol=1.6x ATR=3.96 |
| Stop hit — per-position SL triggered | 2026-01-28 11:20:00 | 892.96 | 892.27 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-01-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:50:00 | 893.00 | 885.56 | 0.00 | ORB-long ORB[871.00,882.80] vol=2.5x ATR=5.08 |
| Stop hit — per-position SL triggered | 2026-01-30 13:05:00 | 887.92 | 890.37 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-02-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 11:05:00 | 903.30 | 904.30 | 0.00 | ORB-short ORB[904.70,913.90] vol=1.7x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-02-05 11:20:00 | 905.39 | 904.27 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-02-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 11:05:00 | 921.00 | 914.97 | 0.00 | ORB-long ORB[910.35,920.00] vol=3.7x ATR=2.36 |
| Stop hit — per-position SL triggered | 2026-02-06 11:30:00 | 918.64 | 915.89 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-02-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:35:00 | 921.40 | 929.13 | 0.00 | ORB-short ORB[926.00,938.35] vol=3.9x ATR=3.64 |
| Stop hit — per-position SL triggered | 2026-02-16 09:40:00 | 925.04 | 926.10 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 950.10 | 940.32 | 0.00 | ORB-long ORB[924.00,935.95] vol=1.8x ATR=3.41 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 946.69 | 940.94 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-02-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:00:00 | 976.15 | 985.97 | 0.00 | ORB-short ORB[981.10,992.20] vol=1.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-02-24 11:10:00 | 979.29 | 985.53 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-02-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 11:05:00 | 966.15 | 959.77 | 0.00 | ORB-long ORB[952.65,957.25] vol=6.7x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 962.82 | 961.00 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:35:00 | 924.20 | 931.19 | 0.00 | ORB-short ORB[932.95,942.30] vol=2.5x ATR=3.88 |
| Stop hit — per-position SL triggered | 2026-03-10 10:20:00 | 928.08 | 926.28 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-03-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:05:00 | 927.00 | 933.60 | 0.00 | ORB-short ORB[932.85,943.00] vol=2.1x ATR=2.64 |
| Stop hit — per-position SL triggered | 2026-03-19 11:10:00 | 929.64 | 933.47 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-04-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:45:00 | 998.10 | 986.28 | 0.00 | ORB-long ORB[973.80,988.50] vol=3.9x ATR=3.45 |
| Stop hit — per-position SL triggered | 2026-04-07 10:50:00 | 994.65 | 986.56 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 1112.00 | 1104.12 | 0.00 | ORB-long ORB[1093.30,1107.70] vol=3.0x ATR=4.09 |
| Stop hit — per-position SL triggered | 2026-04-10 13:40:00 | 1107.91 | 1106.43 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 1150.40 | 1142.55 | 0.00 | ORB-long ORB[1135.45,1146.00] vol=1.7x ATR=5.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:45:00 | 1158.88 | 1145.54 | 0.00 | T1 1.5R @ 1158.88 |
| Target hit | 2026-04-15 13:35:00 | 1158.45 | 1159.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 1200.65 | 1190.46 | 0.00 | ORB-long ORB[1180.10,1194.00] vol=1.8x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:50:00 | 1206.48 | 1193.40 | 0.00 | T1 1.5R @ 1206.48 |
| Target hit | 2026-04-22 10:25:00 | 1202.40 | 1203.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 1232.30 | 1224.70 | 0.00 | ORB-long ORB[1213.00,1230.10] vol=1.9x ATR=4.22 |
| Stop hit — per-position SL triggered | 2026-04-27 10:10:00 | 1228.08 | 1227.24 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-05-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:40:00 | 1121.00 | 1113.10 | 0.00 | ORB-long ORB[1105.00,1120.00] vol=3.1x ATR=4.03 |
| Stop hit — per-position SL triggered | 2026-05-07 10:55:00 | 1116.97 | 1114.28 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-21 09:45:00 | 980.85 | 2025-05-21 09:50:00 | 976.04 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-05-26 10:10:00 | 1014.20 | 2025-05-26 10:15:00 | 1020.81 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-05-26 10:10:00 | 1014.20 | 2025-05-26 15:20:00 | 1044.95 | TARGET_HIT | 0.50 | 3.03% |
| BUY | retest1 | 2025-05-28 10:45:00 | 1038.15 | 2025-05-28 12:50:00 | 1035.33 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-30 10:55:00 | 1035.90 | 2025-05-30 11:00:00 | 1038.43 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-02 09:50:00 | 1045.80 | 2025-06-02 09:55:00 | 1050.91 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-06-02 09:50:00 | 1045.80 | 2025-06-02 15:20:00 | 1055.00 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2025-06-03 09:50:00 | 1067.10 | 2025-06-03 10:05:00 | 1064.63 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-23 10:45:00 | 1023.80 | 2025-06-23 11:00:00 | 1019.87 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-07-01 10:45:00 | 1063.20 | 2025-07-01 10:55:00 | 1065.33 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-08 10:20:00 | 1150.00 | 2025-07-08 11:30:00 | 1153.67 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-09 10:30:00 | 1166.00 | 2025-07-09 12:30:00 | 1172.25 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-07-09 10:30:00 | 1166.00 | 2025-07-09 15:20:00 | 1180.40 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2025-07-10 10:20:00 | 1190.20 | 2025-07-10 14:35:00 | 1197.47 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-07-10 10:20:00 | 1190.20 | 2025-07-10 15:20:00 | 1192.90 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2025-07-11 11:05:00 | 1178.80 | 2025-07-11 11:20:00 | 1172.52 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-07-11 11:05:00 | 1178.80 | 2025-07-11 11:35:00 | 1178.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 11:15:00 | 1181.60 | 2025-08-06 12:00:00 | 1184.39 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-12 11:00:00 | 1290.00 | 2025-08-12 13:00:00 | 1281.68 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-08-12 11:00:00 | 1290.00 | 2025-08-12 15:20:00 | 1257.80 | TARGET_HIT | 0.50 | 2.50% |
| BUY | retest1 | 2025-08-18 09:50:00 | 1287.00 | 2025-08-18 10:00:00 | 1293.59 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-08-18 09:50:00 | 1287.00 | 2025-08-18 11:55:00 | 1290.50 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-08-26 09:40:00 | 1243.00 | 2025-08-26 09:45:00 | 1247.44 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-03 10:45:00 | 1214.00 | 2025-09-03 10:50:00 | 1218.70 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-03 10:45:00 | 1214.00 | 2025-09-03 15:20:00 | 1236.00 | TARGET_HIT | 0.50 | 1.81% |
| BUY | retest1 | 2025-09-08 10:50:00 | 1230.80 | 2025-09-08 13:40:00 | 1236.63 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-09-08 10:50:00 | 1230.80 | 2025-09-08 14:55:00 | 1232.50 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2025-09-09 09:30:00 | 1247.00 | 2025-09-09 10:20:00 | 1242.61 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-10 09:30:00 | 1256.60 | 2025-09-10 09:55:00 | 1262.62 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-09-10 09:30:00 | 1256.60 | 2025-09-10 10:45:00 | 1259.50 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2025-09-11 11:00:00 | 1243.50 | 2025-09-11 11:50:00 | 1237.08 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-09-11 11:00:00 | 1243.50 | 2025-09-11 12:15:00 | 1243.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-16 10:50:00 | 1228.70 | 2025-09-16 12:10:00 | 1233.78 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-09-16 10:50:00 | 1228.70 | 2025-09-16 13:15:00 | 1228.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-18 11:15:00 | 1223.00 | 2025-09-18 11:25:00 | 1218.57 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-09-18 11:15:00 | 1223.00 | 2025-09-18 11:55:00 | 1223.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-24 09:30:00 | 1203.10 | 2025-09-24 09:35:00 | 1197.96 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-09-25 10:15:00 | 1216.50 | 2025-09-25 11:25:00 | 1211.77 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-29 09:30:00 | 1197.20 | 2025-09-29 09:35:00 | 1191.35 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-10-01 10:45:00 | 1167.00 | 2025-10-01 11:10:00 | 1172.55 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-01 10:45:00 | 1167.00 | 2025-10-01 15:20:00 | 1191.00 | TARGET_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2025-10-08 10:30:00 | 1186.60 | 2025-10-08 10:40:00 | 1190.32 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-10 09:55:00 | 1234.20 | 2025-10-10 10:05:00 | 1229.63 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-13 10:55:00 | 1244.70 | 2025-10-13 11:00:00 | 1240.82 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-10-24 11:10:00 | 1207.70 | 2025-10-24 11:30:00 | 1202.66 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-10-24 11:10:00 | 1207.70 | 2025-10-24 14:15:00 | 1207.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-27 10:50:00 | 1200.00 | 2025-10-27 10:55:00 | 1202.96 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-29 09:55:00 | 1234.10 | 2025-10-29 10:10:00 | 1230.93 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-30 10:50:00 | 1209.70 | 2025-10-30 10:55:00 | 1212.11 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-04 10:25:00 | 1198.80 | 2025-11-04 10:30:00 | 1194.14 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-11-06 09:30:00 | 1160.90 | 2025-11-06 09:35:00 | 1164.36 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-06 11:10:00 | 989.30 | 2026-01-06 11:20:00 | 996.30 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-01-06 11:10:00 | 989.30 | 2026-01-06 15:20:00 | 1003.05 | TARGET_HIT | 0.50 | 1.39% |
| BUY | retest1 | 2026-01-16 10:55:00 | 995.15 | 2026-01-16 12:10:00 | 998.98 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-01-16 10:55:00 | 995.15 | 2026-01-16 13:30:00 | 997.05 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2026-01-19 11:15:00 | 985.10 | 2026-01-19 11:30:00 | 987.34 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-01-28 10:00:00 | 889.00 | 2026-01-28 11:20:00 | 892.96 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-01-30 09:50:00 | 893.00 | 2026-01-30 13:05:00 | 887.92 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2026-02-05 11:05:00 | 903.30 | 2026-02-05 11:20:00 | 905.39 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-06 11:05:00 | 921.00 | 2026-02-06 11:30:00 | 918.64 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-16 09:35:00 | 921.40 | 2026-02-16 09:40:00 | 925.04 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-17 10:25:00 | 950.10 | 2026-02-17 10:30:00 | 946.69 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-24 11:00:00 | 976.15 | 2026-02-24 11:10:00 | 979.29 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 11:05:00 | 966.15 | 2026-02-26 11:30:00 | 962.82 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-10 09:35:00 | 924.20 | 2026-03-10 10:20:00 | 928.08 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-19 11:05:00 | 927.00 | 2026-03-19 11:10:00 | 929.64 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-07 10:45:00 | 998.10 | 2026-04-07 10:50:00 | 994.65 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-10 11:00:00 | 1112.00 | 2026-04-10 13:40:00 | 1107.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1150.40 | 2026-04-15 09:45:00 | 1158.88 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1150.40 | 2026-04-15 13:35:00 | 1158.45 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2026-04-22 09:30:00 | 1200.65 | 2026-04-22 09:50:00 | 1206.48 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-22 09:30:00 | 1200.65 | 2026-04-22 10:25:00 | 1202.40 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2026-04-27 09:30:00 | 1232.30 | 2026-04-27 10:10:00 | 1228.08 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-07 10:40:00 | 1121.00 | 2026-05-07 10:55:00 | 1116.97 | STOP_HIT | 1.00 | -0.36% |
