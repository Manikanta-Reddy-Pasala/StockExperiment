# Godrej Industries Ltd. (GODREJIND)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15463 bars)
- **Last close:** 1202.00
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
| ENTRY1 | 71 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 15 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 104 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 56
- **Target hits / Stop hits / Partials:** 15 / 56 / 33
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 22.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 24 | 50.0% | 7 | 24 | 17 | 0.26% | 12.4% |
| BUY @ 2nd Alert (retest1) | 48 | 24 | 50.0% | 7 | 24 | 17 | 0.26% | 12.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 24 | 42.9% | 8 | 32 | 16 | 0.18% | 10.0% |
| SELL @ 2nd Alert (retest1) | 56 | 24 | 42.9% | 8 | 32 | 16 | 0.18% | 10.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 104 | 48 | 46.2% | 15 | 56 | 33 | 0.22% | 22.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:55:00 | 1164.60 | 1155.62 | 0.00 | ORB-long ORB[1142.50,1157.50] vol=1.6x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 10:05:00 | 1172.70 | 1160.78 | 0.00 | T1 1.5R @ 1172.70 |
| Target hit | 2025-05-14 13:05:00 | 1165.30 | 1168.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2025-05-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 10:35:00 | 1170.80 | 1163.07 | 0.00 | ORB-long ORB[1157.50,1165.50] vol=1.6x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 10:45:00 | 1174.62 | 1163.97 | 0.00 | T1 1.5R @ 1174.62 |
| Stop hit — per-position SL triggered | 2025-05-22 10:50:00 | 1170.80 | 1167.60 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:35:00 | 1181.50 | 1176.05 | 0.00 | ORB-long ORB[1163.70,1178.10] vol=2.2x ATR=4.76 |
| Stop hit — per-position SL triggered | 2025-05-23 09:40:00 | 1176.74 | 1176.19 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:40:00 | 1190.50 | 1186.06 | 0.00 | ORB-long ORB[1173.20,1184.70] vol=3.6x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 10:05:00 | 1196.72 | 1188.93 | 0.00 | T1 1.5R @ 1196.72 |
| Target hit | 2025-05-29 10:30:00 | 1197.40 | 1197.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2025-06-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:30:00 | 1203.00 | 1193.01 | 0.00 | ORB-long ORB[1182.20,1199.00] vol=1.6x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:10:00 | 1209.63 | 1198.05 | 0.00 | T1 1.5R @ 1209.63 |
| Stop hit — per-position SL triggered | 2025-06-03 13:05:00 | 1203.00 | 1202.77 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 09:30:00 | 1205.10 | 1198.37 | 0.00 | ORB-long ORB[1190.00,1204.60] vol=2.7x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 09:35:00 | 1210.42 | 1201.90 | 0.00 | T1 1.5R @ 1210.42 |
| Stop hit — per-position SL triggered | 2025-06-06 09:40:00 | 1205.10 | 1202.02 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:40:00 | 1357.00 | 1341.78 | 0.00 | ORB-long ORB[1335.60,1353.50] vol=1.8x ATR=6.27 |
| Stop hit — per-position SL triggered | 2025-06-10 10:45:00 | 1350.73 | 1342.57 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 09:55:00 | 1297.60 | 1290.56 | 0.00 | ORB-long ORB[1280.00,1295.60] vol=1.9x ATR=6.00 |
| Stop hit — per-position SL triggered | 2025-06-12 10:00:00 | 1291.60 | 1290.84 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 1254.60 | 1263.72 | 0.00 | ORB-short ORB[1265.00,1276.50] vol=4.1x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:45:00 | 1246.77 | 1259.05 | 0.00 | T1 1.5R @ 1246.77 |
| Stop hit — per-position SL triggered | 2025-06-16 10:25:00 | 1254.60 | 1255.38 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 10:00:00 | 1339.30 | 1326.42 | 0.00 | ORB-long ORB[1303.70,1321.40] vol=3.8x ATR=6.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 10:25:00 | 1349.50 | 1333.23 | 0.00 | T1 1.5R @ 1349.50 |
| Stop hit — per-position SL triggered | 2025-06-18 10:30:00 | 1339.30 | 1334.10 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:10:00 | 1246.00 | 1251.62 | 0.00 | ORB-short ORB[1249.00,1260.00] vol=1.9x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 10:50:00 | 1239.57 | 1249.91 | 0.00 | T1 1.5R @ 1239.57 |
| Stop hit — per-position SL triggered | 2025-06-26 10:55:00 | 1246.00 | 1248.96 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:30:00 | 1256.00 | 1250.11 | 0.00 | ORB-long ORB[1242.00,1251.60] vol=3.9x ATR=4.54 |
| Stop hit — per-position SL triggered | 2025-06-27 09:40:00 | 1251.46 | 1250.30 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:40:00 | 1257.40 | 1248.63 | 0.00 | ORB-long ORB[1238.10,1251.70] vol=1.7x ATR=4.52 |
| Stop hit — per-position SL triggered | 2025-07-01 09:55:00 | 1252.88 | 1249.70 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:10:00 | 1151.30 | 1159.21 | 0.00 | ORB-short ORB[1156.80,1169.80] vol=1.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-07-08 11:50:00 | 1154.18 | 1158.34 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:50:00 | 1158.80 | 1150.78 | 0.00 | ORB-long ORB[1145.80,1154.50] vol=3.3x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:05:00 | 1164.33 | 1154.26 | 0.00 | T1 1.5R @ 1164.33 |
| Stop hit — per-position SL triggered | 2025-07-09 11:40:00 | 1158.80 | 1156.70 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 09:30:00 | 1094.90 | 1100.60 | 0.00 | ORB-short ORB[1097.10,1107.00] vol=1.8x ATR=6.02 |
| Stop hit — per-position SL triggered | 2025-07-15 10:55:00 | 1100.92 | 1097.44 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 10:55:00 | 1144.00 | 1149.62 | 0.00 | ORB-short ORB[1145.70,1155.00] vol=1.9x ATR=3.10 |
| Stop hit — per-position SL triggered | 2025-07-17 11:10:00 | 1147.10 | 1149.15 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:30:00 | 1131.90 | 1136.69 | 0.00 | ORB-short ORB[1136.80,1146.90] vol=2.1x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-07-22 10:35:00 | 1134.70 | 1136.56 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 11:05:00 | 1131.80 | 1135.62 | 0.00 | ORB-short ORB[1132.00,1147.40] vol=1.7x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-08-01 13:00:00 | 1134.66 | 1134.34 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 11:00:00 | 1118.50 | 1123.67 | 0.00 | ORB-short ORB[1120.50,1131.00] vol=1.5x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-08-05 11:05:00 | 1120.60 | 1123.65 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:00:00 | 1130.80 | 1137.86 | 0.00 | ORB-short ORB[1133.10,1144.90] vol=2.2x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-08-06 11:05:00 | 1133.80 | 1137.20 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:05:00 | 1125.00 | 1132.65 | 0.00 | ORB-short ORB[1127.30,1137.80] vol=3.8x ATR=3.51 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 1128.51 | 1132.40 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:50:00 | 1121.40 | 1112.30 | 0.00 | ORB-long ORB[1098.80,1115.50] vol=2.9x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 10:00:00 | 1127.80 | 1115.06 | 0.00 | T1 1.5R @ 1127.80 |
| Stop hit — per-position SL triggered | 2025-08-13 10:05:00 | 1121.40 | 1116.09 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:45:00 | 1302.80 | 1290.55 | 0.00 | ORB-long ORB[1282.10,1296.30] vol=2.4x ATR=4.92 |
| Stop hit — per-position SL triggered | 2025-08-22 10:00:00 | 1297.88 | 1293.47 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 10:05:00 | 1206.30 | 1215.69 | 0.00 | ORB-short ORB[1212.40,1224.10] vol=2.2x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:10:00 | 1200.16 | 1213.81 | 0.00 | T1 1.5R @ 1200.16 |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 1206.30 | 1211.77 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 10:55:00 | 1205.00 | 1197.19 | 0.00 | ORB-long ORB[1190.00,1198.10] vol=3.7x ATR=4.01 |
| Stop hit — per-position SL triggered | 2025-09-03 11:00:00 | 1200.99 | 1197.25 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:45:00 | 1214.90 | 1206.98 | 0.00 | ORB-long ORB[1194.80,1210.10] vol=2.2x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 09:55:00 | 1219.96 | 1213.70 | 0.00 | T1 1.5R @ 1219.96 |
| Stop hit — per-position SL triggered | 2025-09-09 10:00:00 | 1214.90 | 1214.73 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:35:00 | 1218.00 | 1212.38 | 0.00 | ORB-long ORB[1202.40,1213.80] vol=2.4x ATR=4.38 |
| Stop hit — per-position SL triggered | 2025-09-11 10:10:00 | 1213.62 | 1214.15 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:30:00 | 1200.20 | 1203.88 | 0.00 | ORB-short ORB[1201.00,1215.50] vol=3.9x ATR=3.01 |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1203.21 | 1203.58 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:35:00 | 1219.70 | 1215.70 | 0.00 | ORB-long ORB[1206.10,1219.10] vol=5.3x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 09:40:00 | 1225.58 | 1224.25 | 0.00 | T1 1.5R @ 1225.58 |
| Target hit | 2025-09-15 09:55:00 | 1224.30 | 1226.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2025-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:40:00 | 1213.00 | 1219.41 | 0.00 | ORB-short ORB[1216.90,1230.00] vol=1.9x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 10:05:00 | 1206.62 | 1215.86 | 0.00 | T1 1.5R @ 1206.62 |
| Target hit | 2025-09-17 13:40:00 | 1205.70 | 1204.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — SELL (started 2025-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:55:00 | 1204.40 | 1213.83 | 0.00 | ORB-short ORB[1209.70,1222.80] vol=2.0x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 11:40:00 | 1199.14 | 1208.50 | 0.00 | T1 1.5R @ 1199.14 |
| Target hit | 2025-09-19 14:40:00 | 1176.90 | 1175.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — BUY (started 2025-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:55:00 | 1192.40 | 1181.73 | 0.00 | ORB-long ORB[1172.00,1184.50] vol=2.0x ATR=4.01 |
| Stop hit — per-position SL triggered | 2025-09-24 10:30:00 | 1188.39 | 1185.89 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:20:00 | 1198.30 | 1192.53 | 0.00 | ORB-long ORB[1184.00,1195.00] vol=1.6x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-09-25 10:25:00 | 1195.51 | 1192.91 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:05:00 | 1147.80 | 1153.70 | 0.00 | ORB-short ORB[1150.80,1163.10] vol=2.0x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:55:00 | 1142.24 | 1150.25 | 0.00 | T1 1.5R @ 1142.24 |
| Target hit | 2025-10-06 15:20:00 | 1132.00 | 1134.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:10:00 | 1127.50 | 1130.59 | 0.00 | ORB-short ORB[1132.00,1141.00] vol=2.1x ATR=2.20 |
| Stop hit — per-position SL triggered | 2025-10-07 12:00:00 | 1129.70 | 1130.44 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 1060.50 | 1071.40 | 0.00 | ORB-short ORB[1069.60,1080.00] vol=5.4x ATR=3.38 |
| Stop hit — per-position SL triggered | 2025-10-10 09:35:00 | 1063.88 | 1070.55 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 11:15:00 | 1053.80 | 1062.26 | 0.00 | ORB-short ORB[1060.20,1073.40] vol=2.3x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-10-13 11:40:00 | 1056.45 | 1061.66 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:25:00 | 1090.70 | 1084.92 | 0.00 | ORB-long ORB[1077.10,1089.40] vol=1.6x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 11:00:00 | 1096.54 | 1087.26 | 0.00 | T1 1.5R @ 1096.54 |
| Stop hit — per-position SL triggered | 2025-10-20 11:10:00 | 1090.70 | 1087.85 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 09:40:00 | 1092.10 | 1099.24 | 0.00 | ORB-short ORB[1096.60,1109.50] vol=2.7x ATR=5.30 |
| Stop hit — per-position SL triggered | 2025-10-23 10:35:00 | 1097.40 | 1094.52 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:10:00 | 1087.70 | 1093.12 | 0.00 | ORB-short ORB[1091.20,1100.00] vol=1.6x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 10:20:00 | 1084.16 | 1088.46 | 0.00 | T1 1.5R @ 1084.16 |
| Target hit | 2025-10-24 11:35:00 | 1086.00 | 1083.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — SELL (started 2025-10-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 09:40:00 | 1085.80 | 1088.12 | 0.00 | ORB-short ORB[1086.60,1099.40] vol=3.4x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 10:00:00 | 1080.84 | 1086.61 | 0.00 | T1 1.5R @ 1080.84 |
| Target hit | 2025-10-28 12:15:00 | 1082.40 | 1078.83 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — BUY (started 2025-11-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 11:00:00 | 1115.10 | 1107.00 | 0.00 | ORB-long ORB[1099.60,1108.00] vol=2.6x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 12:45:00 | 1119.39 | 1111.03 | 0.00 | T1 1.5R @ 1119.39 |
| Target hit | 2025-11-03 15:20:00 | 1127.80 | 1120.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-11-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:10:00 | 1112.70 | 1121.06 | 0.00 | ORB-short ORB[1122.00,1128.40] vol=2.3x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:35:00 | 1108.57 | 1119.71 | 0.00 | T1 1.5R @ 1108.57 |
| Target hit | 2025-11-04 15:20:00 | 1110.20 | 1111.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2025-11-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:55:00 | 1065.90 | 1069.94 | 0.00 | ORB-short ORB[1070.20,1080.20] vol=2.3x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:10:00 | 1062.14 | 1069.21 | 0.00 | T1 1.5R @ 1062.14 |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1065.90 | 1068.99 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-11-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:45:00 | 1056.00 | 1057.40 | 0.00 | ORB-short ORB[1056.20,1069.50] vol=1.5x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-11-11 11:00:00 | 1058.79 | 1057.55 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 11:15:00 | 1066.20 | 1062.73 | 0.00 | ORB-long ORB[1059.60,1066.00] vol=2.7x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 11:35:00 | 1069.68 | 1063.65 | 0.00 | T1 1.5R @ 1069.68 |
| Stop hit — per-position SL triggered | 2025-11-14 12:45:00 | 1066.20 | 1065.24 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-11-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:25:00 | 1060.70 | 1062.55 | 0.00 | ORB-short ORB[1066.10,1076.80] vol=1.9x ATR=2.91 |
| Stop hit — per-position SL triggered | 2025-11-18 10:40:00 | 1063.61 | 1062.49 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:35:00 | 1050.20 | 1054.10 | 0.00 | ORB-short ORB[1052.10,1061.00] vol=1.5x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-11-19 09:45:00 | 1052.30 | 1053.48 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 11:05:00 | 1054.10 | 1056.16 | 0.00 | ORB-short ORB[1055.30,1068.20] vol=3.2x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 11:20:00 | 1050.65 | 1055.51 | 0.00 | T1 1.5R @ 1050.65 |
| Stop hit — per-position SL triggered | 2025-11-20 11:30:00 | 1054.10 | 1055.26 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:35:00 | 1039.40 | 1045.56 | 0.00 | ORB-short ORB[1047.50,1061.70] vol=1.8x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-11-21 11:35:00 | 1041.34 | 1044.09 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:00:00 | 1054.00 | 1048.34 | 0.00 | ORB-long ORB[1043.00,1051.00] vol=2.4x ATR=2.58 |
| Stop hit — per-position SL triggered | 2025-11-28 10:05:00 | 1051.42 | 1048.98 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 11:10:00 | 1036.40 | 1040.63 | 0.00 | ORB-short ORB[1038.50,1053.90] vol=2.0x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 11:45:00 | 1033.69 | 1039.51 | 0.00 | T1 1.5R @ 1033.69 |
| Stop hit — per-position SL triggered | 2025-12-02 14:50:00 | 1036.40 | 1037.14 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:15:00 | 1029.90 | 1034.09 | 0.00 | ORB-short ORB[1035.10,1044.50] vol=1.7x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-12-03 11:40:00 | 1031.90 | 1033.90 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:45:00 | 1042.80 | 1034.86 | 0.00 | ORB-long ORB[1026.10,1038.80] vol=1.8x ATR=2.84 |
| Stop hit — per-position SL triggered | 2025-12-04 11:30:00 | 1039.96 | 1036.35 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-01-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:35:00 | 1010.50 | 1003.34 | 0.00 | ORB-long ORB[1000.00,1005.80] vol=2.9x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:40:00 | 1016.76 | 1006.71 | 0.00 | T1 1.5R @ 1016.76 |
| Target hit | 2026-01-06 15:20:00 | 1021.30 | 1016.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2026-01-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:55:00 | 1030.90 | 1021.98 | 0.00 | ORB-long ORB[1015.70,1022.90] vol=5.4x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 14:20:00 | 1035.33 | 1028.38 | 0.00 | T1 1.5R @ 1035.33 |
| Target hit | 2026-01-07 15:20:00 | 1050.40 | 1039.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2026-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-20 09:30:00 | 1007.20 | 995.13 | 0.00 | ORB-long ORB[984.90,998.30] vol=6.9x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:40:00 | 1012.80 | 1000.70 | 0.00 | T1 1.5R @ 1012.80 |
| Stop hit — per-position SL triggered | 2026-01-20 09:45:00 | 1007.20 | 1002.10 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:10:00 | 996.30 | 1000.73 | 0.00 | ORB-short ORB[998.50,1006.30] vol=1.8x ATR=2.50 |
| Stop hit — per-position SL triggered | 2026-01-22 11:25:00 | 998.80 | 1000.61 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:35:00 | 985.10 | 982.73 | 0.00 | ORB-long ORB[977.80,984.90] vol=1.9x ATR=4.13 |
| Stop hit — per-position SL triggered | 2026-01-30 12:05:00 | 980.97 | 984.12 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-02-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:45:00 | 985.60 | 988.27 | 0.00 | ORB-short ORB[986.00,998.90] vol=4.2x ATR=2.65 |
| Stop hit — per-position SL triggered | 2026-02-06 11:00:00 | 988.25 | 988.10 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 979.45 | 970.88 | 0.00 | ORB-long ORB[965.90,974.00] vol=2.0x ATR=2.64 |
| Stop hit — per-position SL triggered | 2026-02-17 11:00:00 | 976.81 | 973.77 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 976.70 | 979.30 | 0.00 | ORB-short ORB[979.65,986.95] vol=2.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-02-18 11:40:00 | 978.17 | 979.10 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 1033.65 | 1038.87 | 0.00 | ORB-short ORB[1038.70,1052.00] vol=5.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:40:00 | 1029.80 | 1037.09 | 0.00 | T1 1.5R @ 1029.80 |
| Stop hit — per-position SL triggered | 2026-02-25 13:00:00 | 1033.65 | 1034.75 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 933.30 | 937.94 | 0.00 | ORB-short ORB[934.45,943.10] vol=2.5x ATR=2.71 |
| Stop hit — per-position SL triggered | 2026-03-05 11:05:00 | 936.01 | 937.61 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 932.65 | 938.02 | 0.00 | ORB-short ORB[934.80,942.80] vol=1.8x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:50:00 | 929.12 | 937.70 | 0.00 | T1 1.5R @ 929.12 |
| Stop hit — per-position SL triggered | 2026-03-06 11:20:00 | 932.65 | 937.27 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-03-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:30:00 | 915.00 | 923.54 | 0.00 | ORB-short ORB[918.95,930.85] vol=3.5x ATR=3.63 |
| Target hit | 2026-03-10 15:20:00 | 912.45 | 915.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 909.55 | 916.17 | 0.00 | ORB-short ORB[912.10,921.45] vol=2.3x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:20:00 | 904.75 | 912.38 | 0.00 | T1 1.5R @ 904.75 |
| Stop hit — per-position SL triggered | 2026-03-11 12:20:00 | 909.55 | 911.20 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 878.00 | 880.98 | 0.00 | ORB-short ORB[882.55,895.00] vol=3.3x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 873.36 | 879.05 | 0.00 | T1 1.5R @ 873.36 |
| Target hit | 2026-03-13 15:20:00 | 840.65 | 852.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2026-03-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:45:00 | 779.45 | 788.60 | 0.00 | ORB-short ORB[790.80,802.20] vol=5.0x ATR=4.77 |
| Stop hit — per-position SL triggered | 2026-03-24 11:45:00 | 784.22 | 783.16 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-05-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:00:00 | 999.05 | 993.22 | 0.00 | ORB-long ORB[985.00,995.00] vol=2.2x ATR=11.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 14:30:00 | 1015.71 | 1004.90 | 0.00 | T1 1.5R @ 1015.71 |
| Target hit | 2026-05-06 15:20:00 | 1023.50 | 1009.15 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:55:00 | 1164.60 | 2025-05-14 10:05:00 | 1172.70 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-05-14 09:55:00 | 1164.60 | 2025-05-14 13:05:00 | 1165.30 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2025-05-22 10:35:00 | 1170.80 | 2025-05-22 10:45:00 | 1174.62 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-05-22 10:35:00 | 1170.80 | 2025-05-22 10:50:00 | 1170.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-23 09:35:00 | 1181.50 | 2025-05-23 09:40:00 | 1176.74 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-05-29 09:40:00 | 1190.50 | 2025-05-29 10:05:00 | 1196.72 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-05-29 09:40:00 | 1190.50 | 2025-05-29 10:30:00 | 1197.40 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-06-03 10:30:00 | 1203.00 | 2025-06-03 11:10:00 | 1209.63 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-06-03 10:30:00 | 1203.00 | 2025-06-03 13:05:00 | 1203.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-06 09:30:00 | 1205.10 | 2025-06-06 09:35:00 | 1210.42 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-06-06 09:30:00 | 1205.10 | 2025-06-06 09:40:00 | 1205.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-10 10:40:00 | 1357.00 | 2025-06-10 10:45:00 | 1350.73 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-06-12 09:55:00 | 1297.60 | 2025-06-12 10:00:00 | 1291.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-06-16 09:30:00 | 1254.60 | 2025-06-16 09:45:00 | 1246.77 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-06-16 09:30:00 | 1254.60 | 2025-06-16 10:25:00 | 1254.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-18 10:00:00 | 1339.30 | 2025-06-18 10:25:00 | 1349.50 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2025-06-18 10:00:00 | 1339.30 | 2025-06-18 10:30:00 | 1339.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-26 10:10:00 | 1246.00 | 2025-06-26 10:50:00 | 1239.57 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-06-26 10:10:00 | 1246.00 | 2025-06-26 10:55:00 | 1246.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 09:30:00 | 1256.00 | 2025-06-27 09:40:00 | 1251.46 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-07-01 09:40:00 | 1257.40 | 2025-07-01 09:55:00 | 1252.88 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-07-08 11:10:00 | 1151.30 | 2025-07-08 11:50:00 | 1154.18 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-09 09:50:00 | 1158.80 | 2025-07-09 10:05:00 | 1164.33 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-09 09:50:00 | 1158.80 | 2025-07-09 11:40:00 | 1158.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-15 09:30:00 | 1094.90 | 2025-07-15 10:55:00 | 1100.92 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2025-07-17 10:55:00 | 1144.00 | 2025-07-17 11:10:00 | 1147.10 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-22 10:30:00 | 1131.90 | 2025-07-22 10:35:00 | 1134.70 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-01 11:05:00 | 1131.80 | 2025-08-01 13:00:00 | 1134.66 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-05 11:00:00 | 1118.50 | 2025-08-05 11:05:00 | 1120.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-06 11:00:00 | 1130.80 | 2025-08-06 11:05:00 | 1133.80 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-07 10:05:00 | 1125.00 | 2025-08-07 10:15:00 | 1128.51 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-13 09:50:00 | 1121.40 | 2025-08-13 10:00:00 | 1127.80 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-08-13 09:50:00 | 1121.40 | 2025-08-13 10:05:00 | 1121.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-22 09:45:00 | 1302.80 | 2025-08-22 10:00:00 | 1297.88 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-09-02 10:05:00 | 1206.30 | 2025-09-02 10:10:00 | 1200.16 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-09-02 10:05:00 | 1206.30 | 2025-09-02 10:15:00 | 1206.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 10:55:00 | 1205.00 | 2025-09-03 11:00:00 | 1200.99 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-09 09:45:00 | 1214.90 | 2025-09-09 09:55:00 | 1219.96 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-09-09 09:45:00 | 1214.90 | 2025-09-09 10:00:00 | 1214.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-11 09:35:00 | 1218.00 | 2025-09-11 10:10:00 | 1213.62 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-09-12 10:30:00 | 1200.20 | 2025-09-12 11:15:00 | 1203.21 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-15 09:35:00 | 1219.70 | 2025-09-15 09:40:00 | 1225.58 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-09-15 09:35:00 | 1219.70 | 2025-09-15 09:55:00 | 1224.30 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-09-17 09:40:00 | 1213.00 | 2025-09-17 10:05:00 | 1206.62 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-09-17 09:40:00 | 1213.00 | 2025-09-17 13:40:00 | 1205.70 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2025-09-19 09:55:00 | 1204.40 | 2025-09-19 11:40:00 | 1199.14 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-09-19 09:55:00 | 1204.40 | 2025-09-19 14:40:00 | 1176.90 | TARGET_HIT | 0.50 | 2.28% |
| BUY | retest1 | 2025-09-24 09:55:00 | 1192.40 | 2025-09-24 10:30:00 | 1188.39 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-25 10:20:00 | 1198.30 | 2025-09-25 10:25:00 | 1195.51 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-06 10:05:00 | 1147.80 | 2025-10-06 10:55:00 | 1142.24 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-10-06 10:05:00 | 1147.80 | 2025-10-06 15:20:00 | 1132.00 | TARGET_HIT | 0.50 | 1.38% |
| SELL | retest1 | 2025-10-07 11:10:00 | 1127.50 | 2025-10-07 12:00:00 | 1129.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-10 09:30:00 | 1060.50 | 2025-10-10 09:35:00 | 1063.88 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-13 11:15:00 | 1053.80 | 2025-10-13 11:40:00 | 1056.45 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-20 10:25:00 | 1090.70 | 2025-10-20 11:00:00 | 1096.54 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-10-20 10:25:00 | 1090.70 | 2025-10-20 11:10:00 | 1090.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-23 09:40:00 | 1092.10 | 2025-10-23 10:35:00 | 1097.40 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-10-24 10:10:00 | 1087.70 | 2025-10-24 10:20:00 | 1084.16 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-24 10:10:00 | 1087.70 | 2025-10-24 11:35:00 | 1086.00 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-10-28 09:40:00 | 1085.80 | 2025-10-28 10:00:00 | 1080.84 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-10-28 09:40:00 | 1085.80 | 2025-10-28 12:15:00 | 1082.40 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2025-11-03 11:00:00 | 1115.10 | 2025-11-03 12:45:00 | 1119.39 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-11-03 11:00:00 | 1115.10 | 2025-11-03 15:20:00 | 1127.80 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-11-04 11:10:00 | 1112.70 | 2025-11-04 11:35:00 | 1108.57 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-11-04 11:10:00 | 1112.70 | 2025-11-04 15:20:00 | 1110.20 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2025-11-10 09:55:00 | 1065.90 | 2025-11-10 10:10:00 | 1062.14 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-10 09:55:00 | 1065.90 | 2025-11-10 10:15:00 | 1065.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 10:45:00 | 1056.00 | 2025-11-11 11:00:00 | 1058.79 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-14 11:15:00 | 1066.20 | 2025-11-14 11:35:00 | 1069.68 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-11-14 11:15:00 | 1066.20 | 2025-11-14 12:45:00 | 1066.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-18 10:25:00 | 1060.70 | 2025-11-18 10:40:00 | 1063.61 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-19 09:35:00 | 1050.20 | 2025-11-19 09:45:00 | 1052.30 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-11-20 11:05:00 | 1054.10 | 2025-11-20 11:20:00 | 1050.65 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-20 11:05:00 | 1054.10 | 2025-11-20 11:30:00 | 1054.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 10:35:00 | 1039.40 | 2025-11-21 11:35:00 | 1041.34 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-28 10:00:00 | 1054.00 | 2025-11-28 10:05:00 | 1051.42 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-02 11:10:00 | 1036.40 | 2025-12-02 11:45:00 | 1033.69 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-02 11:10:00 | 1036.40 | 2025-12-02 14:50:00 | 1036.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 11:15:00 | 1029.90 | 2025-12-03 11:40:00 | 1031.90 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-04 10:45:00 | 1042.80 | 2025-12-04 11:30:00 | 1039.96 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-06 09:35:00 | 1010.50 | 2026-01-06 09:40:00 | 1016.76 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-01-06 09:35:00 | 1010.50 | 2026-01-06 15:20:00 | 1021.30 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2026-01-07 10:55:00 | 1030.90 | 2026-01-07 14:20:00 | 1035.33 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-01-07 10:55:00 | 1030.90 | 2026-01-07 15:20:00 | 1050.40 | TARGET_HIT | 0.50 | 1.89% |
| BUY | retest1 | 2026-01-20 09:30:00 | 1007.20 | 2026-01-20 09:40:00 | 1012.80 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-01-20 09:30:00 | 1007.20 | 2026-01-20 09:45:00 | 1007.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-22 11:10:00 | 996.30 | 2026-01-22 11:25:00 | 998.80 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-30 09:35:00 | 985.10 | 2026-01-30 12:05:00 | 980.97 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-06 10:45:00 | 985.60 | 2026-02-06 11:00:00 | 988.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-17 09:55:00 | 979.45 | 2026-02-17 11:00:00 | 976.81 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-18 11:15:00 | 976.70 | 2026-02-18 11:40:00 | 978.17 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2026-02-25 11:05:00 | 1033.65 | 2026-02-25 12:40:00 | 1029.80 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-25 11:05:00 | 1033.65 | 2026-02-25 13:00:00 | 1033.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:50:00 | 933.30 | 2026-03-05 11:05:00 | 936.01 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-06 10:45:00 | 932.65 | 2026-03-06 10:50:00 | 929.12 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-03-06 10:45:00 | 932.65 | 2026-03-06 11:20:00 | 932.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 10:30:00 | 915.00 | 2026-03-10 15:20:00 | 912.45 | TARGET_HIT | 1.00 | 0.28% |
| SELL | retest1 | 2026-03-11 10:25:00 | 909.55 | 2026-03-11 11:20:00 | 904.75 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-11 10:25:00 | 909.55 | 2026-03-11 12:20:00 | 909.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:50:00 | 878.00 | 2026-03-13 10:15:00 | 873.36 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-13 09:50:00 | 878.00 | 2026-03-13 15:20:00 | 840.65 | TARGET_HIT | 0.50 | 4.25% |
| SELL | retest1 | 2026-03-24 09:45:00 | 779.45 | 2026-03-24 11:45:00 | 784.22 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2026-05-06 10:00:00 | 999.05 | 2026-05-06 14:30:00 | 1015.71 | PARTIAL | 0.50 | 1.67% |
| BUY | retest1 | 2026-05-06 10:00:00 | 999.05 | 2026-05-06 15:20:00 | 1023.50 | TARGET_HIT | 0.50 | 2.45% |
