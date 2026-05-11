# Lodha Developers Ltd. (LODHA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 960.00
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
| ENTRY1 | 24 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 1 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 23
- **Target hits / Stop hits / Partials:** 1 / 23 / 9
- **Avg / median % per leg:** -0.03% / 0.00%
- **Sum % (uncompounded):** -0.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 7 | 33.3% | 1 | 14 | 6 | 0.01% | 0.1% |
| BUY @ 2nd Alert (retest1) | 21 | 7 | 33.3% | 1 | 14 | 6 | 0.01% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 3 | 25.0% | 0 | 9 | 3 | -0.08% | -0.9% |
| SELL @ 2nd Alert (retest1) | 12 | 3 | 25.0% | 0 | 9 | 3 | -0.08% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 10 | 30.3% | 1 | 23 | 9 | -0.03% | -0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:50:00 | 1572.80 | 1558.58 | 0.00 | ORB-long ORB[1545.10,1560.00] vol=2.9x ATR=8.00 |
| Stop hit — per-position SL triggered | 2024-07-09 09:55:00 | 1564.80 | 1558.93 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-07-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:50:00 | 1472.70 | 1464.88 | 0.00 | ORB-long ORB[1449.75,1471.75] vol=2.6x ATR=8.94 |
| Stop hit — per-position SL triggered | 2024-07-16 11:55:00 | 1463.76 | 1465.95 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-07-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:55:00 | 1406.20 | 1394.95 | 0.00 | ORB-long ORB[1371.00,1391.25] vol=2.9x ATR=7.08 |
| Stop hit — per-position SL triggered | 2024-07-24 11:10:00 | 1399.12 | 1395.41 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1220.30 | 1236.24 | 0.00 | ORB-short ORB[1231.30,1249.40] vol=1.5x ATR=6.47 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 1226.77 | 1233.87 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-09-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:45:00 | 1240.65 | 1251.05 | 0.00 | ORB-short ORB[1251.10,1259.45] vol=1.5x ATR=3.40 |
| Stop hit — per-position SL triggered | 2024-09-03 10:50:00 | 1244.05 | 1248.37 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 1317.10 | 1307.80 | 0.00 | ORB-long ORB[1293.15,1312.00] vol=3.7x ATR=5.95 |
| Stop hit — per-position SL triggered | 2024-09-19 09:35:00 | 1311.15 | 1309.20 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:00:00 | 1232.75 | 1237.22 | 0.00 | ORB-short ORB[1234.40,1252.00] vol=2.2x ATR=6.11 |
| Stop hit — per-position SL triggered | 2024-10-01 11:20:00 | 1238.86 | 1237.24 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-10-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:05:00 | 1204.10 | 1211.79 | 0.00 | ORB-short ORB[1211.95,1224.85] vol=1.5x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:30:00 | 1198.61 | 1210.48 | 0.00 | T1 1.5R @ 1198.61 |
| Stop hit — per-position SL triggered | 2024-10-10 11:45:00 | 1204.10 | 1209.33 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-10-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:10:00 | 1170.00 | 1177.63 | 0.00 | ORB-short ORB[1176.10,1190.90] vol=2.1x ATR=3.02 |
| Stop hit — per-position SL triggered | 2024-10-11 11:20:00 | 1173.02 | 1177.23 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-10-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 10:40:00 | 1198.95 | 1185.55 | 0.00 | ORB-long ORB[1171.10,1187.25] vol=4.1x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 11:25:00 | 1204.91 | 1194.78 | 0.00 | T1 1.5R @ 1204.91 |
| Target hit | 2024-10-14 15:20:00 | 1211.00 | 1200.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-12-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:40:00 | 1284.10 | 1267.76 | 0.00 | ORB-long ORB[1246.00,1261.00] vol=1.9x ATR=6.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 09:55:00 | 1293.91 | 1277.84 | 0.00 | T1 1.5R @ 1293.91 |
| Stop hit — per-position SL triggered | 2024-12-02 11:05:00 | 1284.10 | 1284.53 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-12-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:30:00 | 1307.15 | 1299.19 | 0.00 | ORB-long ORB[1290.90,1306.30] vol=1.9x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:40:00 | 1313.87 | 1301.10 | 0.00 | T1 1.5R @ 1313.87 |
| Stop hit — per-position SL triggered | 2024-12-04 10:45:00 | 1307.15 | 1301.59 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-12-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:40:00 | 1391.55 | 1370.69 | 0.00 | ORB-long ORB[1356.75,1371.85] vol=2.3x ATR=7.58 |
| Stop hit — per-position SL triggered | 2024-12-06 10:45:00 | 1383.97 | 1371.59 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-12-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:05:00 | 1390.60 | 1400.93 | 0.00 | ORB-short ORB[1394.00,1408.80] vol=1.7x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 10:15:00 | 1384.71 | 1399.64 | 0.00 | T1 1.5R @ 1384.71 |
| Stop hit — per-position SL triggered | 2024-12-11 10:20:00 | 1390.60 | 1399.40 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:30:00 | 1377.30 | 1406.04 | 0.00 | ORB-short ORB[1417.00,1434.75] vol=2.2x ATR=5.64 |
| Stop hit — per-position SL triggered | 2024-12-13 10:50:00 | 1382.94 | 1400.42 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:15:00 | 1410.55 | 1403.95 | 0.00 | ORB-long ORB[1387.55,1407.60] vol=1.7x ATR=5.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 10:55:00 | 1419.25 | 1405.49 | 0.00 | T1 1.5R @ 1419.25 |
| Stop hit — per-position SL triggered | 2024-12-30 12:10:00 | 1410.55 | 1410.21 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:50:00 | 1346.65 | 1360.07 | 0.00 | ORB-short ORB[1357.10,1375.00] vol=1.9x ATR=5.27 |
| Stop hit — per-position SL triggered | 2025-01-02 10:00:00 | 1351.92 | 1357.92 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-01-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 10:45:00 | 1095.60 | 1087.72 | 0.00 | ORB-long ORB[1076.00,1091.65] vol=2.5x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:10:00 | 1103.11 | 1091.85 | 0.00 | T1 1.5R @ 1103.11 |
| Stop hit — per-position SL triggered | 2025-01-24 12:30:00 | 1095.60 | 1095.15 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 1197.20 | 1210.86 | 0.00 | ORB-short ORB[1207.30,1222.20] vol=2.5x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:35:00 | 1190.69 | 1208.49 | 0.00 | T1 1.5R @ 1190.69 |
| Stop hit — per-position SL triggered | 2025-02-01 12:05:00 | 1197.20 | 1200.40 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-02-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 09:50:00 | 1233.25 | 1218.48 | 0.00 | ORB-long ORB[1202.00,1217.00] vol=1.7x ATR=6.60 |
| Stop hit — per-position SL triggered | 2025-02-07 09:55:00 | 1226.65 | 1219.19 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 10:40:00 | 1177.00 | 1164.91 | 0.00 | ORB-long ORB[1156.10,1172.00] vol=2.1x ATR=5.36 |
| Stop hit — per-position SL triggered | 2025-02-18 11:05:00 | 1171.64 | 1168.09 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-02-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:30:00 | 1182.00 | 1179.79 | 0.00 | ORB-long ORB[1169.45,1179.25] vol=4.4x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:10:00 | 1188.59 | 1180.98 | 0.00 | T1 1.5R @ 1188.59 |
| Stop hit — per-position SL triggered | 2025-02-20 12:15:00 | 1182.00 | 1182.01 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:40:00 | 1083.30 | 1071.20 | 0.00 | ORB-long ORB[1059.65,1074.45] vol=3.7x ATR=5.04 |
| Stop hit — per-position SL triggered | 2025-03-18 09:50:00 | 1078.26 | 1072.61 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:50:00 | 1269.00 | 1252.49 | 0.00 | ORB-long ORB[1239.00,1255.00] vol=2.4x ATR=5.86 |
| Stop hit — per-position SL triggered | 2025-04-21 09:55:00 | 1263.14 | 1253.19 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-09 09:50:00 | 1572.80 | 2024-07-09 09:55:00 | 1564.80 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-07-16 10:50:00 | 1472.70 | 2024-07-16 11:55:00 | 1463.76 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2024-07-24 10:55:00 | 1406.20 | 2024-07-24 11:10:00 | 1399.12 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1220.30 | 2024-08-28 09:40:00 | 1226.77 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-09-03 10:45:00 | 1240.65 | 2024-09-03 10:50:00 | 1244.05 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-19 09:30:00 | 1317.10 | 2024-09-19 09:35:00 | 1311.15 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-10-01 11:00:00 | 1232.75 | 2024-10-01 11:20:00 | 1238.86 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-10-10 11:05:00 | 1204.10 | 2024-10-10 11:30:00 | 1198.61 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-10 11:05:00 | 1204.10 | 2024-10-10 11:45:00 | 1204.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-11 11:10:00 | 1170.00 | 2024-10-11 11:20:00 | 1173.02 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-14 10:40:00 | 1198.95 | 2024-10-14 11:25:00 | 1204.91 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-10-14 10:40:00 | 1198.95 | 2024-10-14 15:20:00 | 1211.00 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2024-12-02 09:40:00 | 1284.10 | 2024-12-02 09:55:00 | 1293.91 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-12-02 09:40:00 | 1284.10 | 2024-12-02 11:05:00 | 1284.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 10:30:00 | 1307.15 | 2024-12-04 10:40:00 | 1313.87 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-12-04 10:30:00 | 1307.15 | 2024-12-04 10:45:00 | 1307.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-06 10:40:00 | 1391.55 | 2024-12-06 10:45:00 | 1383.97 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-12-11 10:05:00 | 1390.60 | 2024-12-11 10:15:00 | 1384.71 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-12-11 10:05:00 | 1390.60 | 2024-12-11 10:20:00 | 1390.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 10:30:00 | 1377.30 | 2024-12-13 10:50:00 | 1382.94 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-12-30 10:15:00 | 1410.55 | 2024-12-30 10:55:00 | 1419.25 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-12-30 10:15:00 | 1410.55 | 2024-12-30 12:10:00 | 1410.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-02 09:50:00 | 1346.65 | 2025-01-02 10:00:00 | 1351.92 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-01-24 10:45:00 | 1095.60 | 2025-01-24 11:10:00 | 1103.11 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-01-24 10:45:00 | 1095.60 | 2025-01-24 12:30:00 | 1095.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-01 11:00:00 | 1197.20 | 2025-02-01 11:35:00 | 1190.69 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-02-01 11:00:00 | 1197.20 | 2025-02-01 12:05:00 | 1197.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 09:50:00 | 1233.25 | 2025-02-07 09:55:00 | 1226.65 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-02-18 10:40:00 | 1177.00 | 2025-02-18 11:05:00 | 1171.64 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-02-20 10:30:00 | 1182.00 | 2025-02-20 11:10:00 | 1188.59 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-02-20 10:30:00 | 1182.00 | 2025-02-20 12:15:00 | 1182.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 09:40:00 | 1083.30 | 2025-03-18 09:50:00 | 1078.26 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-04-21 09:50:00 | 1269.00 | 2025-04-21 09:55:00 | 1263.14 | STOP_HIT | 1.00 | -0.46% |
